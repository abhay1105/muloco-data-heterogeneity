"""Download, filter, split and tokenize the four domains into flat uint16 token files.

Output layout (EXPERIMENT_DESIGN.md §3):
    <out>/{domain}_{train,val,probe}.bin   uint16 GPT-2 tokens, documents separated by <|endoftext|>
    <out>/meta.json                        sizes, source revisions, sha256 of every .bin

Sequences are windows of seq_len+1 tokens with stride seq_len, so a split holding n sequences has
exactly n*seq_len + 1 tokens.

Usage:
    python -m src.prepare_data --out data_bin                 # real data (~9 GB streamed, deleted as it goes)
    python -m src.prepare_data --out data_tiny --synthetic    # tiny synthetic data for local tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np

DOMAINS = ("arxiv", "github", "stackexchange", "wikipedia")
EOT = 50256
SEQ_LEN = 1024

# Pinned dataset revisions (resolved 2026-11-12) so every rebuild produces identical files.
SLIM_REPO, SLIM_REV = "DKYoon/SlimPajama-6B", "b5f90f419b7489cdba26fdbc8c022fcb5562f968"
WIKI_REPO, WIKI_REV = "wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SLIM_SET_NAMES = {
    "RedPajamaArXiv": "arxiv",
    "RedPajamaGithub": "github",
    "RedPajamaStackExchange": "stackexchange",
}

# Defaults: 7,500 steps x 64 seqs / 4 domains = 120,000 train sequences per domain.
N_TRAIN_SEQ = 120_000
N_VAL_SEQ = 1_000
N_PROBE_SEQ = 2_048
COLLECT_FACTOR = 1.25  # collect this multiple of the needed tokens before splitting


def ntok(n_seq: int, seq_len: int = SEQ_LEN) -> int:
    return n_seq * seq_len + 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def split_docs(docs: list[np.ndarray], need: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Shuffle whole documents, then deal them to val -> probe -> train (disjoint at document level).
    Each split is truncated to exactly need[split] tokens."""
    order = np.random.default_rng(seed).permutation(len(docs))
    out, i = {}, 0
    for split in ("val", "probe", "train"):
        chunks, got = [], 0
        while got < need[split]:
            if i >= len(order):
                raise RuntimeError(f"ran out of documents filling '{split}' ({got:,}/{need[split]:,} tokens)")
            d = docs[order[i]]
            i += 1
            chunks.append(d)
            got += len(d)
        out[split] = np.concatenate(chunks)[: need[split]]
    return out


def _tokenize(enc, texts: list[str]) -> list[np.ndarray]:
    toks = enc.encode_ordinary_batch(texts, num_threads=os.cpu_count() or 4)
    return [np.array(t + [EOT], dtype=np.uint16) for t in toks]


def collect_real(workdir: str, target: int) -> tuple[dict[str, list[np.ndarray]], dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import HfApi, hf_hub_download

    enc = tiktoken.get_encoding("gpt2")
    api = HfApi()
    docs = {d: [] for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    files_used = {}

    def process(repo, rev, prefix, domains, extract):
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev)
                       if f.startswith(prefix) and f.endswith(".parquet"))
        used = []
        for f in files:
            wanted = [d for d in domains if counts[d] < target]
            if not wanted:
                break
            t0 = time.time()
            path = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=workdir)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4096):
                for d, texts in extract(batch):
                    if counts[d] >= target or not texts:
                        continue
                    toks = _tokenize(enc, texts)
                    docs[d].extend(toks)
                    counts[d] += sum(len(x) for x in toks)
            os.remove(path)
            used.append(f)
            print(f"  {f}: {time.time() - t0:.0f}s | " + " ".join(f"{d}={counts[d] / 1e6:.0f}M" for d in DOMAINS),
                  flush=True)
        return used

    def slim_extract(batch):
        names = batch.column("meta").field("redpajama_set_name")
        text = batch.column("text")
        out = []
        for set_name, d in SLIM_SET_NAMES.items():
            mask = pc.equal(names, set_name)
            out.append((d, pc.filter(text, mask).to_pylist()))
        return out

    def wiki_extract(batch):
        return [("wikipedia", batch.column("text").to_pylist())]

    print(f"SlimPajama-6B ({SLIM_REV[:8]}): collecting {target / 1e6:.0f}M tokens per domain", flush=True)
    files_used["slimpajama"] = process(SLIM_REPO, SLIM_REV, "data/train-", list(SLIM_SET_NAMES.values()), slim_extract)
    print(f"wikimedia/wikipedia 20231101.en ({WIKI_REV[:8]})", flush=True)
    files_used["wikipedia"] = process(WIKI_REPO, WIKI_REV, "20231101.en/train-", ["wikipedia"], wiki_extract)
    sources = {
        "slimpajama": {"repo": SLIM_REPO, "revision": SLIM_REV, "files": files_used["slimpajama"]},
        "wikipedia": {"repo": WIKI_REPO, "revision": WIKI_REV, "config": "20231101.en",
                      "files": files_used["wikipedia"]},
    }
    return docs, sources


def collect_synthetic(target: int, seed: int = 0) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Domain-specific random Markov chains over disjoint-ish 256-token sub-vocabularies, so that
    domains are genuinely different and learnable. For tests only."""
    rng = np.random.default_rng(seed)
    docs = {}
    for di, d in enumerate(DOMAINS):
        vocab = np.arange(256) + 1000 * (di + 1)
        trans = rng.dirichlet(np.full(256, 0.05), size=256)
        cdf = trans.cumsum(1)
        out, total = [], 0
        while total < target:
            n = int(rng.integers(200, 2000))
            s = np.empty(n, dtype=np.int64)
            s[0] = rng.integers(256)
            u = rng.random(n)
            for i in range(1, n):
                s[i] = min(np.searchsorted(cdf[s[i - 1]], u[i]), 255)
            doc = np.append(vocab[s], EOT).astype(np.uint16)
            out.append(doc)
            total += len(doc)
        docs[d] = out
    return docs, {"synthetic": {"seed": seed}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_bin")
    ap.add_argument("--workdir", default="/tmp/hf_parquet", help="scratch dir for parquet downloads")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--n_train_seq", type=int, default=None)
    ap.add_argument("--n_val_seq", type=int, default=None)
    ap.add_argument("--n_probe_seq", type=int, default=None)
    args = ap.parse_args(argv)

    if args.synthetic:
        n_train = args.n_train_seq or 1_536
        n_val = args.n_val_seq or 64
        n_probe = args.n_probe_seq or 128
    else:
        n_train = args.n_train_seq or N_TRAIN_SEQ
        n_val = args.n_val_seq or N_VAL_SEQ
        n_probe = args.n_probe_seq or N_PROBE_SEQ
    L = args.seq_len
    need = {"train": ntok(n_train, L), "val": ntok(n_val, L), "probe": ntok(n_probe, L)}
    target = int(COLLECT_FACTOR * sum(need.values()))

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    t0 = time.time()
    if args.synthetic:
        docs, sources = collect_synthetic(target)
    else:
        docs, sources = collect_real(args.workdir, target)
        shutil.rmtree(args.workdir, ignore_errors=True)

    meta = {"tokenizer": "gpt2", "eot": EOT, "seq_len": L, "n_train_seq": n_train, "n_val_seq": n_val,
            "n_probe_seq": n_probe, "domains": list(DOMAINS), "sources": sources, "files": {}, "stats": {}}
    for di, d in enumerate(DOMAINS):
        splits = split_docs(docs[d], need, seed=1234 + di)
        meta["stats"][d] = {"n_docs_collected": len(docs[d]),
                            "tokens_collected": int(sum(len(x) for x in docs[d]))}
        for split, arr in splits.items():
            path = os.path.join(args.out, f"{d}_{split}.bin")
            arr.astype(np.uint16).tofile(path)
            meta["files"][f"{d}_{split}.bin"] = {"tokens": int(len(arr)), "sha256": sha256_file(path)}
        print(f"{d}: wrote train/val/probe ({meta['stats'][d]['tokens_collected'] / 1e6:.0f}M collected)", flush=True)
    meta["data_hash"] = hashlib.sha256(
        "".join(meta["files"][k]["sha256"] for k in sorted(meta["files"])).encode()).hexdigest()[:16]
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done in {time.time() - t0:.0f}s; data_hash={meta['data_hash']}")


if __name__ == "__main__":
    main()
