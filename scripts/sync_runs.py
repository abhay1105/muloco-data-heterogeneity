"""Back up finished runs: Modal volume -> local runs/ -> Hugging Face, plus checksums for git.

    .venv/bin/python scripts/sync_runs.py               # download, verify, checksum, upload to HF
    .venv/bin/python scripts/sync_runs.py --no-upload   # skip the Hugging Face step

Only runs that have a summary.json on the volume (= finished) are downloaded; runs still training are
listed and skipped. Files already present locally with the same size are not re-downloaded. summary.json is
written last, so a local summary.json also means the local copy is complete.

Afterwards commit the small files (checkpoints are git-ignored):
    git add runs CHECKPOINTS.sha256 && git commit -m "..." && git push
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HF_REPO = "paidvenk/muloco-data-heterogeneity"
VOLUME = "muloco-runs"
LOCAL = "runs"
CHECKSUMS = "CHECKPOINTS.sha256"
ORDER = ("config.json", "metrics.jsonl", "final.pt", "summary.json")  # summary.json last

CARD = f"""---
tags: [muon, diloco, muloco, federated-learning, non-iid, gpt]
---
# MuLoCo under non-IID data: run logs and checkpoints

Run logs and final checkpoints for the EECS 182 (UC Berkeley) project studying whether MuLoCo (Muon as
DiLoCo's inner optimizer) keeps its advantage over DiLoCo when workers train on different text domains.
Code, experiment design and results: https://github.com/abhay1105/muloco-data-heterogeneity

## Layout
`runs/<run_id>/` holds, per run:
- `config.json`: full run configuration
- `metrics.jsonl`: train loss, per-domain validation loss and diagnostics over training (one JSON object per line)
- `summary.json`: final per-domain evaluation (written last, marks the run as complete)
- `final.pt`: final (outer) model weights, bf16 state dict, ~51M parameters (absent for short tuning runs)

`{CHECKSUMS}` lists the SHA-256 of every checkpoint (the same file is committed to the GitHub repo).

## Loading a checkpoint
```python
import json, torch
from src.model import GPT          # from the GitHub repo
cfg = json.load(open("runs/<run_id>/config.json"))
model = GPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_embd"], cfg["n_head"], cfg["seq_len"])
model.load_state_dict({{k: v.float() for k, v in torch.load("runs/<run_id>/final.pt").items()}})
```
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(vol) -> list[str]:
    files: dict[str, dict[str, int]] = {}
    for e in vol.listdir("/", recursive=True):
        parts = e.path.strip("/").split("/")
        if len(parts) == 2 and e.type.name == "FILE":
            files.setdefault(parts[0], {})[parts[1]] = e.size
    finished = sorted(r for r, fs in files.items() if "summary.json" in fs)
    unfinished = sorted(set(files) - set(finished))
    if unfinished:
        print(f"skipping {len(unfinished)} unfinished run(s): {', '.join(unfinished)}")
    fetched = 0
    for rid in finished:
        os.makedirs(os.path.join(LOCAL, rid), exist_ok=True)
        for name in [n for n in ORDER if n in files[rid]] + sorted(set(files[rid]) - set(ORDER)):
            dst = os.path.join(LOCAL, rid, name)
            if os.path.exists(dst) and os.path.getsize(dst) == files[rid][name]:
                continue
            tmp = dst + ".part"
            with open(tmp, "wb") as fh:
                for chunk in vol.read_file(f"{rid}/{name}"):
                    fh.write(chunk)
            assert os.path.getsize(tmp) == files[rid][name], f"size mismatch for {rid}/{name}"
            os.replace(tmp, dst)
            fetched += 1
    print(f"{len(finished)} finished runs on the volume; downloaded {fetched} new/changed file(s)")
    return finished


def verify(finished: list[str]) -> bool:
    ok = True
    for rid in finished:
        d = os.path.join(LOCAL, rid)
        s = json.load(open(os.path.join(d, "summary.json")))
        cfg = json.load(open(os.path.join(d, "config.json")))
        kinds = {json.loads(ln)["kind"] for ln in open(os.path.join(d, "metrics.jsonl")) if ln.strip()}
        problems = []
        if s.get("diverged"):
            print(f"  note: {rid} diverged (a valid result, kept)")
        else:
            if "final_eval" not in kinds:
                problems.append("no final_eval in metrics.jsonl")
            if cfg.get("save_checkpoint", True) and not os.path.exists(os.path.join(d, "final.pt")):
                problems.append("missing final.pt")
        if problems:
            ok = False
            print(f"  PROBLEM {rid}: {'; '.join(problems)}")
    print("verify:", "all finished runs complete" if ok else "problems found (see above)")
    return ok


def write_checksums() -> list[str]:
    old = {}
    if os.path.exists(CHECKSUMS):  # reuse hashes of unchanged files (keyed by path + size)
        for ln in open(CHECKSUMS):
            digest, path, size = ln.split()
            old[(path, int(size))] = digest
    lines, ckpts = [], []
    for rid in sorted(os.listdir(LOCAL)):
        path = f"{LOCAL}/{rid}/final.pt"
        if os.path.exists(path):
            size = os.path.getsize(path)
            lines.append(f"{old.get((path, size)) or sha256(path)}  {path}  {size}")
            ckpts.append(path)
    with open(CHECKSUMS, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {CHECKSUMS} ({len(ckpts)} checkpoints)")
    return ckpts


def upload(ckpts: list[str]):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=HF_REPO,
                    commit_message="Update model card")
    api.upload_file(path_or_fileobj=CHECKSUMS, path_in_repo=CHECKSUMS, repo_id=HF_REPO,
                    commit_message="Update checkpoint checksums")
    api.upload_folder(folder_path=LOCAL, path_in_repo=LOCAL, repo_id=HF_REPO, ignore_patterns=["*.part"],
                      commit_message="Sync runs")
    remote = set(api.list_repo_files(HF_REPO))
    missing = [p for p in ckpts if p not in remote]
    logs = [f"{LOCAL}/{r}/summary.json" for r in os.listdir(LOCAL)
            if os.path.exists(os.path.join(LOCAL, r, "summary.json"))]
    missing += [p for p in logs if p not in remote]
    print(f"hugging face: {HF_REPO} has {len(ckpts) - sum(p in ckpts for p in missing)}/{len(ckpts)} checkpoints, "
          f"{len(logs) - sum(p in logs for p in missing)}/{len(logs)} run logs")
    if missing:
        sys.exit(f"MISSING on Hugging Face: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    import modal
    finished = download(modal.Volume.from_name(VOLUME))
    ok = verify(finished)
    ckpts = write_checksums()
    if not a.no_upload:
        upload(ckpts)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
