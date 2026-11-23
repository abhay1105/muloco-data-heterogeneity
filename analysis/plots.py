"""Report figures and hypothesis tables from Phase 2 runs (EXPERIMENT_DESIGN.md §8).

    .venv/bin/python analysis/plots.py                       # reads runs/, writes figures/
    .venv/bin/python analysis/plots.py --runs X --out Y

Outputs in <out>/: fig*.png + fig*.pdf, and summary.md (tables for H1/H2 and final losses).
Error bars and bands are mean ± std over seeds (n shown in the tables); ΔL_het and the H2 rescue are
computed per seed (paired) and then summarized. Only production runs are used: untagged full-length runs
(Phase 2), plus the two Stage A runs that serve as the DP seed-0 baselines (same hyperparameters as frozen).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOMAINS = ("arxiv", "github", "stackexchange", "wikipedia")
DOMAIN_LABEL = {"arxiv": "ArXiv", "github": "GitHub", "stackexchange": "StackEx.", "wikipedia": "Wiki"}
PS = (0.25, 0.5, 0.75, 1.0)
FULL_STEPS = 7500
DP_SEED0_ALIASES = {"dp_adamw_s0_lr0.0015": ("dp_adamw", None, 0), "dp_muon_s0_lrm0.012": ("dp_muon", None, 0)}

# Colour = inner optimizer (validated categorical slots 1-2); shade = skew p (validated ordinal ramps);
# line style = outer optimizer. Identity never relies on colour alone (labels / legends / styles).
HUE = {"adamw": "#2a78d6", "muon": "#eb6834"}
RAMP = {"adamw": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], "muon": ["#f09a72", "#eb6834", "#b8461a", "#7a2c0c"]}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD = {  # method -> (inner, outer, label)
    "dp_adamw": ("adamw", "dp", "DP-AdamW"), "dp_muon": ("muon", "dp", "DP-Muon"),
    "diloco": ("adamw", "nesterov", "DiLoCo"), "muloco": ("muon", "nesterov", "MuLoCo"),
    "fedavg_adamw": ("adamw", "fedavg", "FedAvg-AdamW"), "fedavg_muon": ("muon", "fedavg", "FedAvg-Muon"),
}
STYLE = {"nesterov": "-", "fedavg": "--", "dp": ":"}


def h_of(p: float) -> float:
    return p - 0.25  # TV distance to the uniform mixture (src/partition.py::tv_distance)


# ---------------------------------------------------------------------------- loading
class Run:
    def __init__(self, d: str):
        self.summary = json.load(open(os.path.join(d, "summary.json")))
        self.cfg = self.summary["config"]
        self.rows = [json.loads(ln) for ln in open(os.path.join(d, "metrics.jsonl")) if ln.strip()]
        self.final = self.summary["final_eval"]  # None if diverged

    def series(self, kind: str, key: str):
        pts = [(r["step"], r[key]) for r in self.rows if r["kind"] == kind and key in r]
        return np.array([s for s, _ in pts], float), np.array([v for _, v in pts], float)

    def val_curve(self, key="balanced"):
        # Periodic evals only (fixed 250-seq/domain subsets). The final eval uses 1000 seqs/domain and reads ~0.02
        # higher on the same model, so it is not appended here; it is used for final-loss figures and tables.
        return self.series("eval", key)

    def late_mean(self, key: str, frac: float = 0.25):
        s, v = self.series("diag", key)
        if len(v) == 0:
            return np.nan
        return float(v[s >= s.max() * (1 - frac)].mean())


def load(runs_dir: str) -> dict:
    """Return {(method, p or None, seed): Run} for production runs only."""
    out = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        d = os.path.dirname(f)
        rid = os.path.basename(d)
        run = Run(d)
        c = run.cfg
        if rid in DP_SEED0_ALIASES:
            key = DP_SEED0_ALIASES[rid]
        elif not c.get("tag") and c["steps"] == FULL_STEPS and c["method"] in METHOD:
            key = (c["method"], None if c["method"].startswith("dp_") else float(c["p"]), int(c["seed"]))
        else:
            continue
        out[key] = run
    return out


def final(runs, method, p, key="balanced") -> dict:
    """{seed: final loss} (diverged runs -> nan)."""
    return {s: (r.final[key] if r.final else np.nan) for (m, pp, s), r in runs.items() if m == method and pp == p}


def ms(vals) -> tuple[float, float, int]:
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def paired(a: dict, b: dict) -> list[float]:
    return [a[s] - b[s] for s in sorted(set(a) & set(b))]


def stack(curves):
    """Align (steps, values) curves on common steps; return steps, mean, std, n."""
    if not curves:
        return None
    common = sorted(set.intersection(*[set(s.tolist()) for s, _ in curves]))
    vals = np.array([[dict(zip(s.tolist(), v.tolist()))[t] for t in common] for s, v in curves])
    sd = vals.std(0, ddof=1) if len(vals) > 1 else np.zeros(len(common))
    return np.array(common), vals.mean(0), sd, len(vals)


def smooth(v, w):
    if w <= 1 or len(v) < w:
        return v
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, v[0]), v])
    return np.convolve(pad, k, mode="valid")


# ---------------------------------------------------------------------------- styling
def style():
    plt.rcParams.update({
        # Figures are drawn at their printed size (NeurIPS text width 5.5in), so these sizes are what readers see.
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.4, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.facecolor": "white", "axes.titlecolor": INK,
        "legend.handlelength": 1.6, "legend.labelspacing": 0.3, "axes.titlepad": 3,
    })


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png"), metadata={"Software": None})
    fig.savefig(os.path.join(out, f"{name}.pdf"), metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".{png,pdf}"))


def band(ax, x, m, s, color, ls="-", label=None, alpha=0.18, lw=1.6):
    ax.plot(x, m, ls, color=color, label=label, lw=lw)
    if np.any(s > 0):
        ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0)


# ---------------------------------------------------------------------------- figures
def fig_loss_curves(runs, out):
    """Train and validation loss over training: DiLoCo | MuLoCo, one line per p, DP reference dotted."""
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.3), sharex=True, sharey="row")
    for col, method in enumerate(("diloco", "muloco")):
        inner, _, label = METHOD[method]
        dp = "dp_" + inner
        for row, which in enumerate(("train", "val")):
            ax = axes[row, col]
            for i, p in enumerate(PS):
                rs = [r for (m, pp, _), r in runs.items() if m == method and pp == p]
                if not rs:
                    continue
                curves = [r.series("train", "loss") for r in rs] if which == "train" else [r.val_curve() for r in rs]
                st = stack(curves)
                x, m, s = st[0], st[1], st[2]
                if which == "train":
                    m, s = smooth(m, 10), smooth(s, 10)
                band(ax, x, m, s, RAMP[inner][i], lw=1.2, label=f"p={p:g}" + (" (IID)" if p == 0.25 else ""))
            rs = [r for (m, _, _), r in runs.items() if m == dp]
            if rs:
                st = stack([r.series("train", "loss") for r in rs] if which == "train" else [r.val_curve() for r in rs])
                m, s = (smooth(st[1], 20), smooth(st[2], 20)) if which == "train" else (st[1], st[2])  # DP logs single steps
                band(ax, st[0], m, s, INK2, ls=":", lw=1.2, label=METHOD[dp][2], alpha=0.12)
            ax.set_title(f"{label}: {'training loss' if which == 'train' else 'validation loss'}", loc="left")
            ax.set_ylim(2.1, 3.6)
            if col == 0:
                ax.set_ylabel("loss (nats/token)")
            if row == 1:
                ax.set_xlabel("step")
            if row == 0:
                ax.legend(loc="upper right")
    save(fig, out, "fig1_loss_curves")


def fig_loss_curves_e2(runs, out):
    """E2: FedAvg (dashed) vs Nesterov (solid) at p=0.25 (light) and p=1.0 (dark), per inner optimizer."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    for ax, inner in zip(axes, ("adamw", "muon")):
        for method in [m for m, v in METHOD.items() if v[0] == inner and v[1] != "dp"]:
            outer = METHOD[method][1]
            for p, shade in ((0.25, 0), (1.0, 3)):
                rs = [r for (m, pp, _), r in runs.items() if m == method and pp == p]
                if rs:
                    st = stack([r.val_curve() for r in rs])
                    band(ax, st[0], st[1], st[2], RAMP[inner][shade], ls=STYLE[outer],
                         lw=1.2, label=f"{METHOD[method][2]}, p={p:g}")
        ax.set_title(f"inner {'AdamW' if inner == 'adamw' else 'Muon'}", loc="left")
        ax.set_xlabel("step")
        ax.set_ylim(2.1, 3.6)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("loss (nats/token)")
    save(fig, out, "fig1b_loss_curves_e2")


def fig_final_vs_h(runs, out):
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    hs = np.array([h_of(p) for p in PS])
    for method in ("diloco", "muloco"):
        inner = METHOD[method][0]
        st = [ms(final(runs, method, p).values()) for p in PS]
        m, s = np.array([x[0] for x in st]), np.array([x[1] for x in st])
        ok = np.isfinite(m)
        ax.errorbar(hs[ok], m[ok], yerr=s[ok], color=HUE[inner], marker="o", ms=4, capsize=2, lw=1.3,
                    label=METHOD[method][2])
        dm, ds, dn = ms(final(runs, "dp_" + inner, None).values())
        if dn:
            ax.axhspan(dm - ds, dm + ds, color=HUE[inner], alpha=0.12, lw=0)
            ax.axhline(dm, color=HUE[inner], ls=":", lw=1.1, label=METHOD['dp_' + inner][2])
    ax.set_xticks(hs, [f"{h:.2f}\n(p={p:g})" for h, p in zip(hs, PS)])
    ax.set_xlabel("heterogeneity h")
    ax.set_ylabel("final validation loss (nats)")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)
    save(fig, out, "fig2_final_vs_h")


def het_penalty(runs, method):
    base = final(runs, method, 0.25)
    return {p: paired(final(runs, method, p), base) for p in PS}


def slope(runs, method):
    xs, ys = [], []
    for p, vals in het_penalty(runs, method).items():
        xs += [h_of(p)] * len(vals)
        ys += vals
    xs, ys = np.array(xs), np.array(ys)
    ok = np.isfinite(ys)
    if ok.sum() < 2 or len(set(xs[ok])) < 2:
        return np.nan
    return float(np.polyfit(xs[ok], ys[ok], 1)[0])


def fig_delta_het(runs, out):
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    hs = np.array([h_of(p) for p in PS])
    for j, method in enumerate(("diloco", "muloco")):
        inner = METHOD[method][0]
        st = [ms(v) for v in het_penalty(runs, method).values()]
        m, s = np.array([x[0] for x in st]), np.array([x[1] for x in st])
        ok = np.isfinite(m)
        off = (j - 0.5) * 0.012  # small horizontal dodge so error bars don't overlap
        ax.errorbar(hs[ok] + off, m[ok], yerr=s[ok], color=HUE[inner], marker="o", ms=4, capsize=2, lw=1.3,
                    label=f"{METHOD[method][2]} (slope {slope(runs, method):.3f})")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(hs, [f"{h:.2f}\n(p={p:g})" for h, p in zip(hs, PS)])
    ax.set_xlabel("heterogeneity h")
    ax.set_ylabel("penalty ΔL_het (nats)")
    ax.legend(loc="upper left")
    save(fig, out, "fig3_delta_het")


def fig_domain_heatmap(runs, out):
    rows, labels = [], []
    for method in ("dp_adamw", "diloco", "dp_muon", "muloco"):
        for p in ([None] if method.startswith("dp_") else PS):
            vals = [ms(final(runs, method, p, d).values())[0] for d in DOMAINS]
            if np.all(np.isnan(vals)):
                continue
            rows.append(vals)
            labels.append(METHOD[method][2] + ("" if p is None else f", p={p:g}"))
    if not rows:
        return
    a = np.array(rows)
    rel = a - np.nanmin(a, axis=0, keepdims=True)  # colour = gap to the best row in that domain
    fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(rows) + 0.6))
    im = ax.imshow(rel, cmap="Blues", aspect="auto", vmin=0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i, j]):
                ax.text(j, i, f"{a[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if rel[i, j] > 0.6 * np.nanmax(rel) else INK)
    ax.set_xticks(range(4), [DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="gap to best row in column", shrink=0.9)
    save(fig, out, "fig4_domain_heatmap")


def rescue(runs, inner):
    fed, nes = ("fedavg_" + inner), ("diloco" if inner == "adamw" else "muloco")
    pen_f = dict(zip(sorted(set(final(runs, fed, 1.0)) & set(final(runs, fed, 0.25))),
                     paired(final(runs, fed, 1.0), final(runs, fed, 0.25))))
    pen_n = dict(zip(sorted(set(final(runs, nes, 1.0)) & set(final(runs, nes, 0.25))),
                     paired(final(runs, nes, 1.0), final(runs, nes, 0.25))))
    return pen_f, pen_n, paired(pen_f, pen_n)


def fig_e2_bars(runs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.1))
    groups = [("adamw", "fedavg"), ("adamw", "nesterov"), ("muon", "fedavg"), ("muon", "nesterov")]
    name = {("adamw", "fedavg"): "fedavg_adamw", ("adamw", "nesterov"): "diloco",
            ("muon", "fedavg"): "fedavg_muon", ("muon", "nesterov"): "muloco"}
    x = np.arange(len(groups))
    w = 0.36
    for k, (p, shade) in enumerate(((0.25, 1), (1.0, 3))):
        st = [ms(final(runs, name[g], p).values()) for g in groups]
        for i, g in enumerate(groups):
            if st[i][2]:
                a1.bar(x[i] + (k - 0.5) * (w + 0.02), st[i][0], w, yerr=st[i][1], capsize=2,
                       color=RAMP[g[0]][shade], hatch="///" if g[1] == "fedavg" else None, edgecolor="white", lw=0,
                       label=f"p={p:g}" if i == 0 else None)
    a1.set_xticks(x, [f"{'AdamW' if i == 'adamw' else 'Muon'}\n{'FedAvg' if o == 'fedavg' else 'Nesterov'}" for i, o in groups])
    vals = [b.get_height() for b in a1.patches]
    if vals:
        a1.set_ylim(min(vals) - 0.1, max(vals) + 0.08)
    a1.set_ylabel("final validation loss (nats)")
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=a) for a in (0.45, 1.0)],
              labels=["p=0.25 (IID)", "p=1.0"], loc="upper right")
    for i, g in enumerate(groups):
        pen = paired(final(runs, name[g], 1.0), final(runs, name[g], 0.25))
        m, s, n = ms(pen)
        if n:
            a2.bar(x[i], m, 0.6, yerr=s, capsize=3, color=HUE[g[0]], hatch="///" if g[1] == "fedavg" else None,
                   edgecolor="white", lw=0)
    a2.axhline(0, color=AXIS, lw=1)
    a2.set_xticks(x, [b.get_text() for b in a1.get_xticklabels()])
    a2.set_ylabel("penalty, p=1 minus IID (nats)")
    fig.subplots_adjust(wspace=0.38)
    save(fig, out, "fig5_e2_outer_momentum")


def fig_diag_over_training(runs, out):
    keys = [("rho_F/all", "norm retention ρ_F"), ("pair_cos/all", "pairwise cosine")]
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.9), sharex=True, sharey="row")
    for col, p in enumerate(PS):
        for row, (k, lab) in enumerate(keys):
            ax = axes[row, col]
            for method in ("diloco", "muloco"):
                inner = METHOD[method][0]
                rs = [r for (m, pp, _), r in runs.items() if m == method and pp == p]
                if rs:
                    st = stack([r.series("diag", k) for r in rs])
                    band(ax, st[0], smooth(st[1], 10), smooth(st[2], 10), HUE[inner], lw=1.1, label=METHOD[method][2])
            if row == 0:
                ax.set_title(f"p={p:g} (h={h_of(p):.2f})", loc="left")
            if col == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel("step")
    axes[0, 0].legend()
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30)
    save(fig, out, "fig6_alignment_over_training")


def fig_spectral_vs_h(runs, out):
    keys = [("rho_F/all", "norm retention ρ_F"), ("rho_nuc/all", "nuclear retention ρ_*"),
            ("interference_gap/all", "interference gap")]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    hs = np.array([h_of(p) for p in PS])
    for ax, (k, lab) in zip(axes, keys):
        for j, method in enumerate(("diloco", "muloco")):
            inner = METHOD[method][0]
            st = [ms([r.late_mean(k) for (m, pp, _), r in runs.items() if m == method and pp == p]) for p in PS]
            m, s = np.array([x[0] for x in st]), np.array([x[1] for x in st])
            ok = np.isfinite(m)
            ax.errorbar(hs[ok] + (j - 0.5) * 0.012, m[ok], yerr=s[ok], color=HUE[inner], marker="o", ms=4,
                        capsize=2, lw=1.3, label=METHOD[method][2])
        ax.set_xticks(hs, [f"{h:.2f}" for h in hs])
        ax.set_xlabel("heterogeneity h")
        ax.set_title(lab, loc="left")
    axes[0].legend()
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "fig7_diagnostics_vs_h")


def fig_specialization(runs, out, p=1.0, seed=0):
    panels = []
    for method in ("diloco", "muloco"):
        r = runs.get((method, p, seed))
        if r is None:
            continue
        spec = [x for x in r.rows if x["kind"] == "diag" and "specialization" in x]
        if spec:
            last = spec[-1]
            panels.append((method, last["step"], np.array([[last["specialization"][f"worker{k}"][d] for d in DOMAINS]
                                                           for k in range(4)])))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5, 2.4), squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    vmin = min(a.min() for *_, a in panels)
    vmax = max(a.max() for *_, a in panels)
    for col, (ax, (method, step, a)) in enumerate(zip(axes[0], panels)):
        im = ax.imshow(a, cmap="Blues", vmin=vmin, vmax=vmax)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if a[i, j] > vmin + 0.6 * (vmax - vmin) else INK)
        ax.set_xticks(range(4), [DOMAIN_LABEL[d] for d in DOMAINS])
        ax.set_yticks(range(4), [f"worker {k} ({DOMAIN_LABEL[d]})" for k, d in enumerate(DOMAINS)] if col == 0 else [])
        ax.set_xlabel("evaluated on")
        ax.grid(False)
        ax.set_title(METHOD[method][2], loc="left")
    fig.colorbar(im, ax=axes[0].tolist(), label="local-model loss (nats)", shrink=0.9)
    save(fig, out, "fig8_specialization")


def home_advantage(run, last_only=True):
    """Worker specialization (post-hoc): for each domain d = home of worker k, mean loss of the other workers' local
    models on d minus worker k's loss on d, averaged over the 4 domains. Measured before averaging at the probe
    rounds (25/50/75% of training). Returns (mean over domains, per-domain array) for the last (or every) probe round."""
    out = []
    for x in run.rows:
        if x["kind"] == "diag" and "specialization" in x:
            S = np.array([[x["specialization"][f"worker{k}"][d] for d in DOMAINS] for k in range(4)])
            a = np.array([np.mean([S[j, k] for j in range(4) if j != k]) - S[k, k] for k in range(4)])
            out.append(a)
    if not out:
        return np.nan, None
    a = out[-1] if last_only else np.mean(out, axis=0)
    return float(a.mean()), a


def fig_home_advantage(runs, out):
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    hs = np.array([h_of(p) for p in PS])
    for j, method in enumerate(("diloco", "muloco", "fedavg_adamw", "fedavg_muon")):
        inner, outer, label = METHOD[method]
        st = [ms([home_advantage(r)[0] for (m, pp, _), r in runs.items() if m == method and pp == p]) for p in PS]
        m, s = np.array([x[0] for x in st]), np.array([x[1] for x in st])
        ok = np.isfinite(m)
        ax.errorbar(hs[ok] + (j - 1.5) * 0.008, m[ok], yerr=s[ok], color=HUE[inner],
                    ls="none" if outer == "fedavg" else STYLE[outer],  # FedAvg ran at p=0.25 and 1.0 only
                    marker="o" if outer == "nesterov" else "s", mfc="white" if outer == "fedavg" else HUE[inner],
                    ms=4, capsize=2, lw=1.3, label=label)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(hs, [f"{h:.2f}\n(p={p:g})" for h, p in zip(hs, PS)])
    ax.set_xlabel("heterogeneity h")
    ax.set_ylabel("home-domain advantage (nats)")
    ax.legend(loc="upper left")
    save(fig, out, "fig9_home_advantage")


# ---------------------------------------------------------------------------- tables
def f(m, s, n, nd=4):
    return "—" if not n else f"{m:.{nd}f} ± {s:.{nd}f} (n={n})"


def tables(runs, out):
    L = ["# Phase 2 summary (generated by analysis/plots.py)", "",
         "All values: mean ± std over seeds. Loss = final domain-balanced validation loss (nats/token).", ""]
    L += ["## Final loss", "", "| method | " + " | ".join(f"p={p:g}" for p in PS) + " |", "|---|" + "---|" * 4]
    for method in ("diloco", "muloco", "fedavg_adamw", "fedavg_muon"):
        L.append(f"| {METHOD[method][2]} | " + " | ".join(f(*ms(final(runs, method, p).values())) for p in PS) + " |")
    for dp in ("dp_adamw", "dp_muon"):
        L.append(f"| {METHOD[dp][2]} (1 worker) | {f(*ms(final(runs, dp, None).values()))} | | | |")
    L += ["", "## H1: heterogeneity penalty ΔL_het = L(p) − L(0.25), paired by seed", "",
          "| | " + " | ".join(f"p={p:g} (h={h_of(p):.2f})" for p in PS[1:]) + " | slope vs h |", "|---|" + "---|" * 4]
    pens = {m: het_penalty(runs, m) for m in ("diloco", "muloco")}
    for m in ("diloco", "muloco"):
        L.append(f"| {METHOD[m][2]} | " + " | ".join(f(*ms(pens[m][p])) for p in PS[1:]) + f" | {slope(runs, m):.4f} |")
    diff = {p: paired(dict(zip(sorted(set(final(runs, 'muloco', p)) & set(final(runs, 'muloco', 0.25))), pens["muloco"][p])),
                      dict(zip(sorted(set(final(runs, 'diloco', p)) & set(final(runs, 'diloco', 0.25))), pens["diloco"][p])))
            for p in PS[1:]}
    L.append("| MuLoCo − DiLoCo (paired) | " + " | ".join(f(*ms(diff[p])) for p in PS[1:]) + " | |")
    L += ["", "Gap to DP of the same inner optimizer, G_DP = L / L_DP − 1 (%), using mean losses:", "",
          "| | " + " | ".join(f"p={p:g}" for p in PS) + " |", "|---|" + "---|" * 4]
    for m in ("diloco", "muloco"):
        dpm = ms(final(runs, "dp_" + METHOD[m][0], None).values())[0]
        L.append(f"| {METHOD[m][2]} | " + " | ".join(
            f"{100 * (ms(final(runs, m, p).values())[0] / dpm - 1):.2f}%" if np.isfinite(dpm) else "—" for p in PS) + " |")
    L += ["", "## H2: outer-momentum rescue", "",
          "Penalty = L(p=1.0) − L(p=0.25) per seed; R_i = penalty(FedAvg_i) − penalty(Nesterov_i). "
          "H2 predicts R_Muon > R_AdamW > 0 and penalty(Nesterov-Muon) > 0.", "",
          "| inner | penalty FedAvg | penalty Nesterov | rescue R |", "|---|---|---|---|"]
    for inner in ("adamw", "muon"):
        pf, pn, r = rescue(runs, inner)
        L.append(f"| {'AdamW' if inner == 'adamw' else 'Muon'} | {f(*ms(pf.values()))} | {f(*ms(pn.values()))} | {f(*ms(r))} |")
    L += ["", "## Late-training diagnostics (mean over the last 25% of rounds, then over seeds)", "",
          "| method | p | ρ_F | pair cos | mom. align | ρ_* | interference gap |", "|---|---|---|---|---|---|---|"]
    for m in ("diloco", "muloco", "fedavg_adamw", "fedavg_muon"):
        for p in PS:
            rs = [r for (mm, pp, _), r in runs.items() if mm == m and pp == p]
            if rs:
                cells = [f(*ms([r.late_mean(k) for r in rs]), nd=3) for k in
                         ("rho_F/all", "pair_cos/all", "mom_align/all", "rho_nuc/all", "interference_gap/all")]
                L.append(f"| {METHOD[m][2]} | {p:g} | " + " | ".join(cells) + " |")
    L += ["", "## Worker specialization (post-hoc): home-domain advantage of local models before averaging", "",
          "Mean over domains of [other workers' loss on d − home worker's loss on d], last probe round (75% of training).", "",
          "| method | " + " | ".join(f"p={p:g}" for p in PS) + " |", "|---|" + "---|" * 4]
    for m in ("diloco", "muloco", "fedavg_adamw", "fedavg_muon"):
        L.append(f"| {METHOD[m][2]} | " + " | ".join(
            f(*ms([home_advantage(r)[0] for (mm, pp, _), r in runs.items() if mm == m and pp == p])) for p in PS) + " |")
    diverged = [f"{k}" for k, r in runs.items() if r.summary.get("diverged")]
    L += ["", f"Runs loaded: {len(runs)}. Diverged: {diverged or 'none'}."]
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", os.path.join(out, "summary.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    runs = load(a.runs)
    print(f"{len(runs)} production runs:", ", ".join(f"{m}/{p}/{s}" for m, p, s in sorted(runs, key=str)))
    for fig in (fig_loss_curves, fig_loss_curves_e2, fig_final_vs_h, fig_delta_het, fig_domain_heatmap,
                fig_e2_bars, fig_diag_over_training, fig_spectral_vs_h, fig_specialization, fig_home_advantage):
        fig(runs, a.out)
    tables(runs, a.out)


if __name__ == "__main__":
    main()
