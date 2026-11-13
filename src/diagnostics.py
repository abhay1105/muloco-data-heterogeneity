"""Pseudo-gradient diagnostics (EXPERIMENT_DESIGN.md §7).

All functions take `deltas`: {param_name: Tensor[K, m, n]} holding each worker's pseudo-gradient
Delta_k = theta_outer - theta_k for the hidden 2-D matrices. Metrics are computed per matrix, then
averaged per matrix type (attn_q, ..., mlp_proj) and over all matrices ("all").
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from .model import hidden_type

EPS = 1e-12


def _aggregate(per_matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    buckets = defaultdict(list)
    for name, metrics in per_matrix.items():
        t = hidden_type(name)
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            buckets[(k, t)].append(v)
            buckets[(k, "all")].append(v)
    return {f"{k}/{t}": sum(vs) / len(vs) for (k, t), vs in sorted(buckets.items())}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + EPS))


@torch.no_grad()
def round_metrics(deltas: dict[str, torch.Tensor], momentum: dict[str, torch.Tensor] | None = None) -> dict:
    """Cheap per-round metrics: norm retention, pairwise cosine, cosine to mean, momentum alignment."""
    per = {}
    sq_norms = None
    for name, D3 in deltas.items():
        K = D3.shape[0]
        D = D3.reshape(K, -1).float()
        G = D @ D.T
        norms = G.diagonal().clamp_min(0).sqrt()
        bar_norm = G.sum().clamp_min(0).sqrt() / K
        m = {"rho_F": float(bar_norm / (norms.mean() + EPS))}
        if K > 1:
            C = G / (norms[:, None] * norms[None, :] + EPS)
            iu = torch.triu_indices(K, K, 1)
            m["pair_cos"] = float(C[iu[0], iu[1]].mean())
            m["cos_to_mean"] = float(((G.sum(1) / K) / (norms * bar_norm + EPS)).mean())
        if momentum is not None and name in momentum:
            m["mom_align"] = _cos(D.mean(0), momentum[name])
        per[name] = m
        sq_norms = G.diagonal() if sq_norms is None else sq_norms + G.diagonal()
    out = _aggregate(per)
    out["worker_norm"] = [float(x) for x in sq_norms.sqrt()]
    return out


@torch.no_grad()
def spectral_metrics(deltas: dict[str, torch.Tensor], top_frac: float = 0.05) -> dict:
    """Nuclear-norm retention and normalized top-S interference gap (MuLoCo Def. 4.1)."""
    per = {}
    for name, D3 in deltas.items():
        D3 = D3.float()
        if D3.device.type == "mps":  # svdvals is not implemented on MPS
            D3 = D3.cpu()
        s_k = torch.linalg.svdvals(D3)  # (K, r), descending
        s_bar = torch.linalg.svdvals(D3.mean(0))  # (r,)
        S = max(1, math.ceil(top_frac * s_bar.numel()))
        top_k = s_k[:, :S].sum(1).mean()
        per[name] = {
            "rho_nuc": float(s_bar.sum() / (s_k.sum(1).mean() + EPS)),
            "interference_gap": float((top_k - s_bar[:S].sum()) / (top_k + EPS)),
        }
    return _aggregate(per)


@torch.no_grad()
def reference_metrics(deltas: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict:
    """Alignment of the averaged pseudo-gradient (and of each worker's) with a DP reference
    pseudo-gradient computed from the same starting point."""
    per = {}
    for name, D3 in deltas.items():
        per[name] = {
            "ref_cos_mean": _cos(D3.mean(0), ref[name]),
            "ref_cos_worker": sum(_cos(D3[k], ref[name]) for k in range(D3.shape[0])) / D3.shape[0],
        }
    return _aggregate(per)
