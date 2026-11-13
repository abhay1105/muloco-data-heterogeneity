"""Summarize finished runs into a markdown table (for pasting into RESULTS.md).

    python analysis/summarize.py                    # all runs in runs/
    python analysis/summarize.py "dp_adamw_s0_lr*"  # glob over run ids
    python analysis/summarize.py "*pilot" --diag    # add mean late-training diagnostics (local runs)
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(run_dir: str):
    with open(os.path.join(run_dir, "summary.json")) as fh:
        s = json.load(fh)
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    return s, rows


def late_diag(rows, keys, frac=0.25):
    """Mean of each diagnostic over the last `frac` of rounds (all probe rounds for ref_* keys)."""
    alld = [r for r in rows if r["kind"] == "diag"]
    late = alld[int(len(alld) * (1 - frac)):]
    out = {}
    for k in keys:
        src = alld if k.startswith("ref_") else late  # probe rounds are sparse (25/50/75%): use all
        vals = [r[k] for r in src if k in r]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


def fmt(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="*")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.runs, a.pattern)) if os.path.exists(os.path.join(d, "summary.json")))
    dkeys = ["rho_F/all", "pair_cos/all", "mom_align/all", "interference_gap/all", "ref_cos_mean/all"]
    head = ["run_id", "lr", "lr_muon", "outer", "balanced", "arxiv", "github", "stackex", "wiki", "worst", "div", "min"]
    if a.diag:
        head += ["ρ_F", "pair_cos", "mom_align", "interf_gap", "ref_cos"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for d in dirs:
        s, rows = load(d)
        c, f = s["config"], s["final_eval"] or {}
        outer = "—" if c["mode"] == "dp" else f"{c['outer_lr']:g}/{c['outer_momentum']:g}"
        cells = [s["run_id"], f"{c['lr']:g}", f"{c['lr_muon']:g}" if c["inner"] == "muon" else "—", outer,
                 fmt(f.get("balanced")), fmt(f.get("arxiv"), 3), fmt(f.get("github"), 3),
                 fmt(f.get("stackexchange"), 3), fmt(f.get("wikipedia"), 3), fmt(f.get("worst"), 3),
                 "**yes**" if s["diverged"] else "no", f"{s['wall_time_s'] / 60:.1f}"]
        if a.diag:
            ld = late_diag(rows, dkeys) if c["mode"] == "local" else {}
            cells += [fmt(ld.get(k), 3) for k in dkeys]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
