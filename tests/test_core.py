"""Correctness tests (EXPERIMENT_DESIGN.md §10.3). Run locally: `pytest -q`."""
import json
import os

import numpy as np
import pytest
import torch

from src import prepare_data
from src.config import parse_args
from src.data import build_train_pools
from src.diagnostics import round_metrics, spectral_metrics
from src.model import GPT
from src.optim import InnerOptimizer, lr_mult
from src.partition import dp_pool, seq_counts, tv_distance, worker_assignments, worker_pools
from src.train import Runner

PS = (0.25, 0.5, 0.75, 1.0)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("data"))
    prepare_data.main(["--out", out, "--synthetic", "--seq_len", "64", "--n_train_seq", "600",
                       "--n_val_seq", "16", "--n_probe_seq", "32", "--workdir", out + "/_w"])
    return out


def tiny_args(data_dir, out_dir, **kw):
    base = dict(n_layer=1, n_embd=64, n_head=2, seq_len=64, global_batch_seqs=8, micro_batch_seqs=2, steps=4,
                H=2, warmup_steps=1, eval_every=2, spectral_every=2, eval_seqs=4, final_eval_seqs=8, compile=0,
                device="cpu", dtype="fp32", save_checkpoint=0, data_dir=data_dir, out_dir=out_dir)
    base.update(kw)
    return [f"--{k}={v}" for k, v in base.items()]


# ---------------------------------------------------------------- partition / data accounting
@pytest.mark.parametrize("p", PS)
def test_seq_counts_exact(p):
    n = seq_counts(p, 120_000)
    assert (n.sum(0) == 120_000).all() and (n.sum(1) == 120_000).all()
    assert n[0, 0] == round(p * 120_000)


def test_tv_distance():
    assert [tv_distance(p) for p in PS] == [0.0, 0.25, 0.5, 0.75]


@pytest.mark.parametrize("p", PS)
def test_worker_pools_disjoint_and_cover(p):
    n = 1200
    pools = worker_pools(p, n, seed=3)
    rows = [set(map(tuple, ix.tolist())) for ix in pools]
    assert sum(len(r) for r in rows) == len(set().union(*rows)) == 4 * n  # disjoint, no duplicates
    assert set().union(*rows) == set(map(tuple, dp_pool(n, seed=3).tolist()))  # same token multiset as DP
    counts = seq_counts(p, n)
    for k, ix in enumerate(pools):
        assert (np.bincount(ix[:, 0], minlength=4) == counts[k]).all()
    # contiguous, ordered slices per domain
    for d in range(4):
        spans = sorted((s, e) for k in range(4) for (dd, s, e) in worker_assignments(p, n)[k] if dd == d)
        assert spans[0][0] == 0 and spans[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_pools_from_config(tiny_data, tmp_path):
    from src.data import TokenData
    cfg = parse_args(tiny_args(tiny_data, str(tmp_path), method="muloco", p=0.75, steps=60, H=2))
    data = TokenData(cfg.data_dir, cfg.seq_len, 60 * 8 // 4)
    pools = build_train_pools(cfg, data, torch.device("cpu"))
    assert len(pools) == 4 and all(len(pl.index) == 60 * 2 for pl in pools)
    expected = seq_counts(0.75, 120)
    for k, pl in enumerate(pools):
        assert (pl.domain_counts(60) == expected[k]).all()


# ---------------------------------------------------------------- model / optimizer
def test_param_grouping_full_size():
    m = GPT(50304, 8, 512, 8, 1024)
    c = m.param_counts()
    assert c["hidden_matrices"] == 8 * (4 * 512 * 512 + 2 * 512 * 2048) == 25_165_824
    assert c["embedding"] == 50304 * 512
    cfg = parse_args(["--method=muloco", "--device=cpu"])
    opt = InnerOptimizer(m, cfg)
    muon, adam = opt.opts
    assert isinstance(muon, torch.optim.Muon) and isinstance(adam, torch.optim.AdamW)
    assert sum(p.numel() for g in muon.param_groups for p in g["params"]) == c["hidden_matrices"]
    assert sum(p.numel() for g in adam.param_groups for p in g["params"]) == c["total"] - c["hidden_matrices"]
    assert muon.param_groups[0]["weight_decay"] == 0.0


def test_lr_schedule():
    cfg = parse_args(["--method=diloco"])
    assert lr_mult(1, cfg) == pytest.approx(1 / 250)
    assert lr_mult(250, cfg) == pytest.approx(1.0)
    assert lr_mult(7500, cfg) == pytest.approx(0.1)


def test_fedavg_forces_plain_averaging():
    cfg = parse_args(["--method=fedavg_muon", "--outer_lr=0.3", "--outer_momentum=0.9"])
    assert (cfg.outer_lr, cfg.outer_momentum) == (1.0, 0.0)


# ---------------------------------------------------------------- diagnostics
def test_diagnostics_identical_and_orthogonal():
    name = "blocks.0.attn.q.weight"
    d = torch.randn(8, 8)
    m = round_metrics({name: d.expand(4, 8, 8).clone()})
    assert m["rho_F/all"] == pytest.approx(1.0, abs=1e-5) and m["pair_cos/all"] == pytest.approx(1.0, abs=1e-5)
    s = spectral_metrics({name: d.expand(4, 8, 8).clone()})
    assert s["rho_nuc/all"] == pytest.approx(1.0, abs=1e-4) and s["interference_gap/all"] == pytest.approx(0.0, abs=1e-4)
    e = torch.zeros(4, 8, 8)
    for k in range(4):
        e[k, k, k] = 1.0  # orthogonal, equal norms
    m = round_metrics({name: e})
    assert m["rho_F/all"] == pytest.approx(0.5, abs=1e-6) and m["pair_cos/all"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- equivalences
def test_local_k1_h1_fedavg_equals_dp(tiny_data, tmp_path):
    common = dict(steps=6, H=1, eval_every=6, spectral_every=6, seed=1)
    dp = Runner(parse_args(tiny_args(tiny_data, str(tmp_path), method="dp_adamw", **common))).run()
    loc = Runner(parse_args(tiny_args(tiny_data, str(tmp_path), method="fedavg_adamw", K=1, **common))).run()
    for key in ("balanced", "arxiv", "github"):
        assert loc["final_eval"][key] == pytest.approx(dp["final_eval"][key], abs=1e-4)


def test_fedavg_sgd_round_is_mean_of_worker_updates(tiny_data, tmp_path):
    args = tiny_args(tiny_data, str(tmp_path), method="fedavg_sgd", p=0.25, steps=1, H=1, eval_every=1,
                     spectral_every=1, grad_clip=0.0, lr=0.5, probe=0, global_batch_seqs=8, micro_batch_seqs=2)
    r = Runner(parse_args(args))
    theta0 = [p.detach().clone() for p in r.params]
    grads = [torch.zeros_like(p) for p in r.params]
    for k in range(4):
        r.model.zero_grad()
        x, y = r.pools[k].batch(0)
        r.model(x, y).backward()
        for g, p in zip(grads, r.params):
            g += p.grad / 4
    r.model.zero_grad()
    expected = [t - 0.5 * lr_mult(1, r.cfg) * g for t, g in zip(theta0, grads)]
    r.run_local()
    for p, e in zip(r.params, expected):
        torch.testing.assert_close(p.detach(), e, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- smoke: every method end to end
@pytest.mark.parametrize("method", ["dp_adamw", "dp_muon", "diloco", "muloco", "fedavg_adamw", "fedavg_muon"])
def test_smoke_methods(tiny_data, tmp_path, method):
    s = Runner(parse_args(tiny_args(tiny_data, str(tmp_path), method=method, p=1.0, steps=8, H=2, eval_every=4,
                                    spectral_every=4, probe_fracs="0.5"))).run()
    assert not s["diverged"] and np.isfinite(s["final_eval"]["balanced"])
    run_dir = os.path.join(str(tmp_path), s["run_id"])
    kinds = [json.loads(line)["kind"] for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    assert "final_eval" in kinds and os.path.exists(os.path.join(run_dir, "summary.json"))
    if method not in ("dp_adamw", "dp_muon"):
        diags = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))
                 if json.loads(line)["kind"] == "diag"]
        assert len(diags) == 4 and any("ref_cos_mean/all" in d for d in diags)
        assert any("interference_gap/all" in d for d in diags)
