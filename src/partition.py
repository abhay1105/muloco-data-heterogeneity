"""Deterministic domain-skew partition (EXPERIMENT_DESIGN.md §3.3-3.4).

Worker k has home domain k. It receives fraction p of its sequences from domain k and (1-p)/3 from
each other domain. Every domain's training sequences are split exactly once across the workers, so
the global mixture is uniform for every p.
"""
from __future__ import annotations

import numpy as np


def seq_counts(p: float, n_per_domain: int, n_domains: int = 4) -> np.ndarray:
    """Integer matrix n[k][d] of sequences worker k takes from domain d.
    Rows and columns both sum to n_per_domain exactly (symmetric construction)."""
    if not (1.0 / n_domains - 1e-9 <= p <= 1.0 + 1e-9):
        raise ValueError(f"p must be in [1/{n_domains}, 1], got {p}")
    m = n_domains - 1
    off = int(round((1.0 - p) * n_per_domain / m))
    diag = n_per_domain - m * off
    n = np.full((n_domains, n_domains), off, dtype=np.int64)
    np.fill_diagonal(n, diag)
    return n


def tv_distance(p: float, n_domains: int = 4) -> float:
    """Total-variation distance between a worker's mixture and the uniform global mixture."""
    return p - 1.0 / n_domains


def worker_assignments(p: float, n_per_domain: int, n_domains: int = 4) -> list[list[tuple[int, int, int]]]:
    """For each worker k: list of (domain, start_seq, end_seq) contiguous, disjoint index ranges.
    Domain d's sequence range [0, n_per_domain) is cut in worker order 0..K-1."""
    n = seq_counts(p, n_per_domain, n_domains)
    out = [[] for _ in range(n_domains)]
    for d in range(n_domains):
        start = 0
        for k in range(n_domains):
            if n[k, d] > 0:
                out[k].append((d, start, start + int(n[k, d])))
            start += int(n[k, d])
        assert start == n_per_domain
    return out


def ranges_to_index(ranges: list[tuple[int, int, int]]) -> np.ndarray:
    """(domain, start, end) ranges -> int64 array of shape (N, 2) with columns (domain, seq)."""
    parts = [np.stack([np.full(e - s, d), np.arange(s, e)], axis=1) for d, s, e in ranges]
    return np.concatenate(parts).astype(np.int64)


def shuffled(index: np.ndarray, seed: int, stream: int) -> np.ndarray:
    """Seeded shuffle. `stream` separates workers (0..K-1) and the DP pool (100)."""
    rng = np.random.default_rng([seed, stream])
    return index[rng.permutation(len(index))]


def worker_pools(p: float, n_per_domain: int, seed: int, n_domains: int = 4) -> list[np.ndarray]:
    return [shuffled(ranges_to_index(r), seed, k)
            for k, r in enumerate(worker_assignments(p, n_per_domain, n_domains))]


def dp_pool(n_per_domain: int, seed: int, n_domains: int = 4) -> np.ndarray:
    return shuffled(ranges_to_index([(d, 0, n_per_domain) for d in range(n_domains)]), seed, 100)
