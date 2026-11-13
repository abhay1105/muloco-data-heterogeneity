# Results Log: MuLoCo Under Non-IID Data

A running record of every run, result and observation, kept for the written report. The design is in `EXPERIMENT_DESIGN.md`.

**Conventions**
- Losses are validation cross-entropy (nats/token) of the **outer** model on held-out data for each domain.
  - "Balanced" is the mean over the 4 domains, and "worst" is the maximum.
  - Final values use 1,000 sequences (1.02M tokens) per domain.
- Diagnostics are measured on the hidden block matrices (§7 of the design doc):
  - ρ_F: norm retention. 1 means the workers are aligned; 0.5 means they are orthogonal (for K=4).
  - pair_cos: mean pairwise cosine between workers' pseudo-gradients.
  - mom_align: cosine between the averaged pseudo-gradient and the outer momentum.
  - interf_gap: normalized top-5% interference gap.
  - ref_cos: cosine with the DP reference pseudo-gradient.
  - The tables report means over the last 25% of rounds, except ref_cos, which averages all probe rounds.
- Tables are generated with `python analysis/summarize.py "<glob>" [--diag]` from `runs/` (downloaded with `modal volume get muloco-runs / runs/`).
- Observations are dated. Anything labeled *preliminary* is not yet backed by enough runs or seeds to cite as a result.

---

## 0. Setup facts

| Item | Value |
|---|---|
| Data hash | `33c0c658fbf49a8a` (built 2026-11-13 on Modal, `modal_prepare.py`) |
| Tokens per domain | train 122,880,001 · val 1,024,001 · probe 2,097,153 (GPT-2 tokenizer) |
| Domains / sources | ArXiv, GitHub, StackExchange: `DKYoon/SlimPajama-6B` @ `b5f90f41` (StackExchange needed ~32 of 48 shards). Wikipedia: `wikimedia/wikipedia` 20231101.en @ `b04c8d1c` |
| Model | 50,930,176 params total: 25,174,528 non-embedding, 25,165,824 in hidden matrices (Muon-managed), 25,755,648 tied embedding |
| Hardware | Modal, NVIDIA H100 80GB HBM3, PyTorch 2.15.0, `torch.compile`, bf16 autocast |
| Local tests | 22/22 pass (`pytest -q`), including two exactness checks: (1) one worker, H=1, plain averaging reproduces DP training; (2) a FedAvg round with plain SGD equals the mean of the workers' SGD updates |

---

## 1. E0: Pilot (2026-11-13)

300-step runs with default (untuned) LRs (AdamW 3e-3, Muon 3e-3 AdamW-matched), warmup 30 steps. These check the pipeline and measure throughput. **They are not experimental results.**

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min | ρ_F | pair_cos | mom_align | interf_gap | ref_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diloco_p1.00_H30_s0_pilot | 0.003 | — | 0.7/0.9 | 5.6558 | 5.411 | 4.519 | 5.370 | 7.324 | 7.324 | no | 0.6 | 0.479 | -0.016 | 0.279 | 0.483 | 0.654 |
| dp_adamw_s0_pilot | 0.003 | — | — | 4.7192 | 4.309 | 3.739 | 4.521 | 6.308 | 6.308 | no | 1.1 | — | — | — | — | — |
| dp_muon_s0_pilot | 0.003 | 0.003 | — | 3.7935 | 3.342 | 2.873 | 3.592 | 5.367 | 5.367 | no | 0.4 | — | — | — | — | — |
| muloco_p1.00_H30_s0_pilot | 0.003 | 0.003 | 0.7/0.9 | 4.3681 | 3.926 | 3.352 | 4.109 | 6.086 | 6.086 | no | 0.6 | 0.518 | 0.024 | 0.096 | 0.377 | 0.498 |

**Throughput (steady state, H100):**

| Run type | Time per step or round | Training speed | Full run |
|---|---|---|---|
| DP | ~3.4 s per 50 steps | ~950k tok/s | ~9–10 min |
| DiLoCo | ~2.1 s per round | ~940k tok/s | ~11–12 min |
| MuLoCo | ~2.7 s per round | ~740k tok/s | ~13–14 min |

- Spectral diagnostics (every 300 steps) cost ~5 s each, and a probe round costs ~3.5 s.
- Estimated total budget is about $50.

**Observations:**
- (2026-11-13) Pipeline works end to end on GPU for all 4 run types: compile, worker simulation, diagnostics, probe rounds and checkpoint saving. No divergence.
- (2026-11-13) *Sanity:* at p = 1.0 (fully disjoint domains), the workers' pseudo-gradients are nearly orthogonal for both methods (pair_cos ≈ 0, ρ_F ≈ 0.5). During the probe round, each MuLoCo worker's local model already scores best on its own domain. This is the drift setting the hypotheses are about.
- (2026-11-13) *Preliminary, not a result:* at 300 steps, Muon-based runs are ahead of AdamW-based ones (DP 3.79 vs 4.72; local p = 1.0 4.37 vs 5.66), consistent with Muon's known fast start. LRs are untuned, so this says nothing about H1 or H2.

---

## 2. Phase 1 tuning

### Stage A: DP inner-LR sweeps (seed 0, full length)

#### A1: DP-AdamW (2026-11-13)

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dp_adamw_s0_lr0.00075 | 0.00075 | — | — | 2.3302 | 2.043 | 1.401 | 2.188 | 3.689 | 3.689 | no | 9.9 |
| **dp_adamw_s0_lr0.0015** | **0.0015** | — | — | **2.3079** | 2.028 | 1.383 | 2.165 | 3.655 | 3.655 | no | 8.9 |
| dp_adamw_s0_lr0.003 | 0.003 | — | — | 2.3241 | 2.041 | 1.398 | 2.182 | 3.675 | 3.675 | no | 8.9 |
| dp_adamw_s0_lr0.006 | 0.006 | — | — | 2.3447 | 2.059 | 1.416 | 2.202 | 3.702 | 3.702 | no | 9.9 |

Balanced loss on the periodic eval subset (256K tokens/domain) during training:

| lr | step 1500 | 3000 | 4500 | 6000 | 7200 |
|---|---|---|---|---|---|
| 7.5e-4 | 2.731 | 2.511 | 2.402 | 2.334 | 2.307 |
| **1.5e-3** | **2.715** | **2.500** | **2.385** | **2.314** | **2.285** |
| 3e-3 | 2.735 | 2.514 | 2.401 | 2.328 | 2.299 |
| 6e-3 | 2.775 | 2.541 | 2.424 | 2.351 | 2.320 |

**Selected: AdamW lr = 1.5e-3.**
- It's an interior point of the grid, so no extra edge point is needed.
- It is best at every eval checkpoint, not just the final one.
- `dp_adamw_s0_lr0.0015` becomes the **DP-AdamW seed-0 baseline**, and 1.5e-3 is also the AdamW lr for the embedding, head and norm parameters in all Muon arms.

**Observations:**
- (2026-11-13) The LR response is flat near the optimum: the full ×8 range spans only 0.037 nats, and the neighbors of 1.5e-3 are 0.016–0.022 worse. Seed-to-seed noise isn't measured yet (that comes with Phase 2 seeds), so the exact optimum between 1.5e-3 and 3e-3 is uncertain at the ~0.01 level. That's fine for tuning purposes.
- (2026-11-13) Domain difficulty is very uneven and consistent across LRs. GitHub is easiest (~1.38) and Wikipedia hardest (~3.66), with ArXiv ~2.03 and StackExchange ~2.17. For the report: the balanced loss is dominated by Wikipedia's level, while *changes* due to heterogeneity may show up in any domain, so per-domain deltas will matter.
- (2026-11-13) Final losses use the full 1M-token val set per domain and are ~0.02 higher than the 256K-subset evals at step 7200. This is a subset-composition effect, not overfitting (the train loss keeps falling). Always compare final with final.

#### A2: DP-Muon (launched 2026-11-13)
Muon lr ∈ {1.5e-3, 3e-3, 6e-3, 1.2e-2} (AdamW-matched scaling), with AdamW lr for non-hidden params = 1.5e-3.

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dp_muon_s0_lrm0.0015 | 0.0015 | 0.0015 | — | 2.2523 | 1.985 | 1.342 | 2.114 | 3.568 | 3.568 | no | 10.5 |
| dp_muon_s0_lrm0.003 | 0.0015 | 0.003 | — | 2.2557 | 1.986 | 1.344 | 2.116 | 3.576 | 3.576 | no | 10.0 |
| dp_muon_s0_lrm0.006 | 0.0015 | 0.006 | — | 2.2536 | 1.983 | 1.340 | 2.114 | 3.577 | 3.577 | no | 9.5 |
| dp_muon_s0_lrm0.012 | 0.0015 | 0.012 | — | 2.2490 | 1.978 | 1.335 | 2.109 | 3.573 | 3.573 | no | 10.9 |

Balanced loss on the periodic eval subset (256K tokens/domain) during training:

| lr_muon | step 600 | 1500 | 3000 | 4500 | 6000 | 7200 |
|---|---|---|---|---|---|---|
| 1.5e-3 | 2.926 | 2.595 | 2.418 | 2.320 | 2.256 | 2.229 |
| 3e-3 | 2.911 | 2.597 | 2.421 | 2.323 | 2.258 | 2.232 |
| 6e-3 | 2.887 | 2.585 | 2.414 | 2.319 | 2.255 | 2.230 |
| 1.2e-2 | 2.878 | 2.578 | 2.408 | 2.313 | 2.251 | 2.225 |

**Observations:**
- (2026-11-13) **DP-Muon beats DP-AdamW by about 0.055–0.06 nats at the end** (2.249 vs 2.308 for the best of each), and the gap holds in every domain (e.g. Wikipedia 3.57 vs 3.66, GitHub 1.335 vs 1.383). This matches the known Muon-over-AdamW gap at small scale, and it sets the baselines that each local method will be normalized against.
- (2026-11-13) Muon is **remarkably insensitive to its learning rate**: an ×8 range spans only 0.007 nats at the end. The curve isn't monotonic at the end (1.5e-3 < 6e-3 < 3e-3), which suggests seed-level noise. During training, though, the highest lr (1.2e-2) is consistently best by a small margin, and it trains fastest early (step 600: 2.878 vs 2.926).
- (2026-11-13) Because the best value (1.2e-2) is at the grid edge, I added one edge point per the protocol: **lr_muon = 2.4e-2** (`manifests/stageA_muon_edge.txt`).

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dp_muon_s0_lrm0.024 | 0.0015 | 0.024 | — | 2.2516 | 1.980 | 1.337 | 2.115 | 3.575 | 3.575 | no | 10.2 |

Eval-subset curve at steps 600 / 1500 / 3000 / 4500 / 6000 / 7200: 2.875, 2.578, 2.409, 2.314, 2.253, 2.228. That matches 1.2e-2 early and is slightly worse late.

**Selected: Muon lr = 1.2e-2** (AdamW lr 1.5e-3 for non-hidden params).
- It's now an interior optimum: 2.4e-2 gives 2.2516, worse than 1.2e-2's 2.2490.
- `dp_muon_s0_lrm0.012` becomes the **DP-Muon seed-0 baseline**.

**Stage A summary:** inner LRs are AdamW 1.5e-3 and Muon 1.2e-2.
- Seed-0 DP baselines: DP-AdamW 2.3079, DP-Muon 2.2490.
- The Muon LR curve is flat to within 0.007 over 1.5e-3 to 2.4e-2. Write-up note: the Muon arms are robust to the inner-LR choice, so any H1 effect is unlikely to come from Muon LR mistuning.

### Stage B: outer (lr, momentum) sweep. *Not started; paused 2026-11-13.*

**Resume here:**
- The manifest is ready: `manifests/stageB.txt`, 12 half-length IID runs with inner LRs AdamW 1.5e-3 and Muon 1.2e-2.
- Launch with `.venv/bin/modal run --detach modal_app.py::launch --manifest manifests/stageB.txt` (~$4–5, ~10 min wall-clock).
- Then Stage C (`scripts/make_manifest.py stageC ...` with the Stage-B winners), then write `configs/frozen.yaml`.

### Stage C: inner-LR check. *Not started.*

### Frozen hyperparameters. *Not yet frozen.*

---

## 3. Phase 2 (production). *Not started.*

### E1: heterogeneity sweep (H1)

### E2: outer-momentum ablation (H2)

---

## 4. Running observations and notes for the write-up
- (2026-11-13) Design decisions to describe in the report:
  - Deterministic skew p replaces Dirichlet, so the global mixture is identical across conditions and every arm trains on the same token multiset.
  - Each worker keeps its inner optimizer state across rounds.
  - Hyperparameters are tuned only on IID data.
- (2026-11-13) Infrastructure note: the data prep was split into a CPU-only Modal app, because Modal requires a payment method for any app containing an H100 function.
