"""Token storage and batch construction.

All four domains' token files are loaded into RAM (~1 GB at full size). A *pool* is an (N, 2) array of
(domain, sequence-index) rows. Sequence s of a split covers tokens [s*L, s*L + L + 1).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .partition import dp_pool, shuffled, worker_pools


class TokenData:
    def __init__(self, data_dir: str, seq_len: int, n_train_needed: int):
        with open(os.path.join(data_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        if self.meta["seq_len"] != seq_len:
            raise ValueError(f"data seq_len {self.meta['seq_len']} != config seq_len {seq_len}")
        self.L = seq_len
        self.domains = self.meta["domains"]
        self.data_hash = self.meta.get("data_hash", "unknown")
        self.splits = {}
        for split in ("train", "val", "probe"):
            self.splits[split] = [np.fromfile(os.path.join(data_dir, f"{d}_{split}.bin"), dtype=np.uint16)
                                  for d in self.domains]
        self.n_seq = {split: min((len(a) - 1) // seq_len for a in arrs) for split, arrs in self.splits.items()}
        if self.n_seq["train"] < n_train_needed:
            raise ValueError(f"need {n_train_needed} train sequences per domain, data has {self.n_seq['train']}")

    def gather(self, split: str, rows: np.ndarray) -> np.ndarray:
        arrs, L = self.splits[split], self.L
        out = np.empty((len(rows), L + 1), dtype=np.int64)
        for i, (d, s) in enumerate(rows):
            out[i] = arrs[d][s * L: s * L + L + 1]
        return out


class Pool:
    """Consumes `batch_seqs` rows of `index` per step, in order."""

    def __init__(self, data: TokenData, index: np.ndarray, batch_seqs: int, device, split: str = "train"):
        self.data, self.index, self.bs, self.device, self.split = data, index, batch_seqs, device, split

    def batch(self, step: int, wrap: bool = False):
        lo, hi = step * self.bs, (step + 1) * self.bs
        if hi > len(self.index):
            if not wrap:
                raise IndexError(f"pool exhausted: step {step} needs rows [{lo},{hi}) of {len(self.index)}")
            rows = self.index[np.arange(lo, hi) % len(self.index)]
        else:
            rows = self.index[lo:hi]
        return to_device(self.data.gather(self.split, rows), self.device)

    def domain_counts(self, n_steps: int) -> np.ndarray:
        used = self.index[: n_steps * self.bs]
        return np.bincount(used[:, 0], minlength=len(self.data.domains))


def to_device(arr: np.ndarray, device):
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def build_train_pools(cfg, data: TokenData, device) -> list[Pool]:
    n_per_domain = cfg.steps * cfg.global_batch_seqs // len(data.domains)
    if cfg.K == 1:  # DP, or local mode with one worker: identical stream to DP
        return [Pool(data, dp_pool(n_per_domain, cfg.seed, len(data.domains)), cfg.global_batch_seqs, device)]
    idx = worker_pools(cfg.p, n_per_domain, cfg.seed, len(data.domains))
    return [Pool(data, ix, cfg.worker_batch_seqs, device) for ix in idx]


def n_train_needed(cfg, n_domains: int) -> int:
    return cfg.steps * cfg.global_batch_seqs // n_domains


def eval_rows(domain: int, n_seqs: int) -> np.ndarray:
    return np.stack([np.full(n_seqs, domain), np.arange(n_seqs)], axis=1)


def probe_pool(cfg, data: TokenData, device) -> Pool:
    """Uniform-mixture pool from the held-out probe split, for the DP reference learner."""
    nd, n = len(data.domains), data.n_seq["probe"]
    rows = np.concatenate([eval_rows(d, n) for d in range(nd)])
    return Pool(data, shuffled(rows, 0, 200), cfg.global_batch_seqs, device, split="probe")
