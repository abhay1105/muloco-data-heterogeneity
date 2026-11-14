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

### Stage B: outer (lr, momentum) sweep (2026-11-14)

Setup: IID (p = 0.25), H = 30, K = 4, seed 0, **half length** (3,750 steps; schedule compressed: warmup 250, cosine to step 3,750). Inner LRs are fixed from Stage A (AdamW 1.5e-3, Muon 1.2e-2).

**Infrastructure change:** the run is on **A100-80GB**, not H100. The first launch sat queued for about 20 min with "H100 unavailable on starter plan," and nothing was charged. On A100 the training speed is about 340–390k tok/s for DiLoCo and about 290–340k tok/s for MuLoCo, about 2.6× slower than H100. GPU type affects speed only, not results, so the Stage A choices made on H100 still apply.

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min | ρ_F | pair_cos | mom_align | interf_gap | ref_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diloco_p0.25_H30_s0_eo0.5_mu0.6 | 0.0015 | — | 0.5/0.6 | 2.7524 | 2.339 | 1.722 | 2.561 | 4.387 | 4.387 | no | 14.2 | 0.527 | 0.037 | 0.101 | 0.471 | 0.318 |
| diloco_p0.25_H30_s0_eo0.5_mu0.9 | 0.0015 | — | 0.5/0.9 | 2.8582 | 2.438 | 1.821 | 2.676 | 4.497 | 4.497 | no | 14.2 | 0.562 | 0.089 | 0.133 | 0.431 | 0.421 |
| **diloco_p0.25_H30_s0_eo0.8_mu0.6** | 0.0015 | — | 0.8/0.6 | **2.7335** | 2.330 | 1.709 | 2.541 | 4.354 | 4.354 | no | 14.1 | 0.541 | 0.056 | 0.054 | 0.453 | 0.358 |
| diloco_p0.25_H30_s0_eo0.8_mu0.9 | 0.0015 | — | 0.8/0.9 | 2.8790 | 2.459 | 1.842 | 2.696 | 4.520 | 4.520 | no | 13.7 | 0.583 | 0.121 | 0.098 | 0.411 | 0.469 |
| diloco_p0.25_H30_s0_eo1_mu0.6 | 0.0015 | — | 1/0.6 | 2.7384 | 2.333 | 1.711 | 2.545 | 4.364 | 4.364 | no | 13.9 | 0.555 | 0.079 | 0.016 | 0.434 | 0.414 |
| diloco_p0.25_H30_s0_eo1_mu0.9 | 0.0015 | — | 1/0.9 | 2.9201 | 2.495 | 1.874 | 2.739 | 4.572 | 4.572 | no | 13.2 | 0.608 | 0.161 | 0.069 | 0.387 | 0.542 |
| muloco_p0.25_H30_s0_eo0.5_mu0.6 | 0.0015 | 0.012 | 0.5/0.6 | 2.5322 | 2.166 | 1.533 | 2.346 | 4.083 | 4.083 | no | 16.6 | 0.513 | 0.018 | 0.259 | 0.390 | 0.386 |
| muloco_p0.25_H30_s0_eo0.5_mu0.9 | 0.0015 | 0.012 | 0.5/0.9 | 2.5614 | 2.197 | 1.553 | 2.386 | 4.109 | 4.109 | no | 16.6 | 0.532 | 0.045 | 0.167 | 0.360 | 0.425 |
| **muloco_p0.25_H30_s0_eo0.8_mu0.6** | 0.0015 | 0.012 | 0.8/0.6 | **2.5157** | 2.157 | 1.521 | 2.334 | 4.052 | 4.052 | no | 16.4 | 0.517 | 0.024 | 0.247 | 0.383 | 0.397 |
| muloco_p0.25_H30_s0_eo0.8_mu0.9 | 0.0015 | 0.012 | 0.8/0.9 | 2.5632 | 2.205 | 1.556 | 2.387 | 4.104 | 4.104 | no | 16.7 | 0.538 | 0.053 | 0.162 | 0.351 | 0.443 |
| muloco_p0.25_H30_s0_eo1_mu0.6 | 0.0015 | 0.012 | 1/0.6 | 2.5205 | 2.163 | 1.522 | 2.335 | 4.062 | 4.062 | no | 14.2 | 0.521 | 0.029 | 0.238 | 0.377 | 0.411 |
| muloco_p0.25_H30_s0_eo1_mu0.9 | 0.0015 | 0.012 | 1/0.9 | 2.5684 | 2.206 | 1.560 | 2.392 | 4.116 | 4.116 | no | 15.0 | 0.543 | 0.060 | 0.159 | 0.339 | 0.460 |

Balanced loss on the eval subset at steps 300 / 900 / 1800 / 2700 / 3600 for the best setting of each method:
- DiLoCo 0.8/0.6: 4.685, 3.321, 2.934, 2.776, 2.707.
- MuLoCo 0.8/0.6: 4.104, 2.875, 2.658, 2.546, 2.491.

**Observations:**
- (2026-11-14) **Momentum 0.6 beats 0.9 at every outer lr, for both methods.**
  - The effect is larger for DiLoCo (0.11–0.18 nats) than for MuLoCo (0.03–0.05).
  - The outer lr matters much less: within μ = 0.6 the spread is 0.019 for DiLoCo and 0.017 for MuLoCo, with 0.8 best for both.
  - Because μ = 0.6 is the lowest value in the grid, I added an edge extension at μ ∈ {0, 0.3} with η_out = 0.8 (`manifests/stageB_edge.txt`, below).
  - This differs from the DiLoCo/MuLoCo defaults (μ ≈ 0.9). The likely reason is our regime: training from scratch, a short run, and small per-worker batches (16 seqs).
- (2026-11-14) **On IID data, MuLoCo is far ahead of DiLoCo:** 2.516 vs 2.734 at the best settings, a gap of 0.22 nats. That is much larger than the DP gap at full length (0.06). It agrees with MuLoCo's claim that Muon inner steps degrade less as K grows. (These are half-length runs, so absolute numbers aren't comparable to Stage A.) Every domain shows the same ordering.
- (2026-11-14) **Diagnostics at IID:**
  - Workers' pseudo-gradients are close to orthogonal even with identical data distributions (ρ_F ≈ 0.51–0.61, pair_cos ≈ 0.02–0.16). With 16-sequence worker batches, minibatch noise dominates each worker's 30-step update.
  - Write-up implication: in this regime ρ_F / pair_cos have little headroom to separate heterogeneity levels. In Phase 2, look at *differences relative to the same method's IID run*, plus ref_cos and the per-worker specialization matrices.
  - MuLoCo has lower pair_cos than DiLoCo at IID yet trains better, so "more aligned workers" is not the same as "better" here.
- (2026-11-14) **Counter-intuitive pattern:** higher outer momentum gives *higher* worker alignment and *higher* ref_cos, but *worse* loss. A plausible explanation: with large μ the outer model overshoots, so all workers share a common corrective direction, which raises their alignment without helping progress. Worth a sentence in the report on why alignment metrics must be read with care.

#### Stage B edge extension: outer momentum 0 and 0.3 (η_out = 0.8), 2026-11-14

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min | ρ_F | pair_cos | mom_align | interf_gap | ref_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diloco_p0.25_H30_s0_eo0.8_mu0 | 0.0015 | — | 0.8/0 | 2.7955 | 2.370 | 1.757 | 2.603 | 4.452 | 4.452 | no | 14.6 | 0.523 | 0.031 | — | 0.476 | 0.312 |
| diloco_p0.25_H30_s0_eo0.8_mu0.3 | 0.0015 | — | 0.8/0.3 | 2.7383 | 2.329 | 1.708 | 2.546 | 4.370 | 4.370 | no | 14.0 | 0.528 | 0.039 | 0.093 | 0.469 | 0.322 |
| muloco_p0.25_H30_s0_eo0.8_mu0 | 0.0015 | 0.012 | 0.8/0 | 2.5638 | 2.185 | 1.558 | 2.375 | 4.137 | 4.137 | no | 15.7 | 0.509 | 0.013 | — | 0.395 | 0.385 |
| muloco_p0.25_H30_s0_eo0.8_mu0.3 | 0.0015 | 0.012 | 0.8/0.3 | 2.5312 | 2.164 | 1.533 | 2.346 | 4.082 | 4.082 | no | 15.0 | 0.512 | 0.016 | 0.314 | 0.392 | 0.387 |


Momentum profile at η_out = 0.8 (IID, half length):

| outer momentum | 0 | 0.3 | **0.6** | 0.9 |
|---|---|---|---|---|
| DiLoCo | 2.795 | 2.738 | **2.734** | 2.879 |
| MuLoCo | 2.564 | 2.531 | **2.516** | 2.563 |

**Selected outer optimizer, both methods: η_out = 0.8, μ = 0.6 (Nesterov).** It's an interior optimum for both.

**Observations:**
- (2026-11-14) **Outer momentum helps both methods relative to none, at IID:** DiLoCo by 0.06 and MuLoCo by 0.05 (μ = 0.6 vs 0). Too much momentum (0.9) hurts as much as having none. This is the IID baseline for H2: E2 will test whether that benefit grows or shrinks under heterogeneity. (Note: μ = 0 at η_out = 0.8 is *not* FedAvg, which uses η_out = 1, μ = 0.)
- (2026-11-14) Takeaway for the report: in this from-scratch, small-batch regime the best outer momentum (0.6) is much lower than the DiLoCo/MuLoCo default (~0.9).

### Stage C: inner-LR check at the selected outer config (2026-11-14)
IID, H = 30, outer 0.8/0.6, seed 0, half length, A100. Compared against the Stage B 0.8/0.6 runs (×1).

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min | ρ_F | pair_cos | mom_align | interf_gap | ref_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diloco_p0.25_H30_s0_innerx0.5 | 0.00075 | — | 0.8/0.6 | 2.6867 | 2.288 | 1.662 | 2.495 | 4.301 | 4.301 | no | 13.1 | 0.538 | 0.053 | 0.047 | 0.455 | 0.352 |
| diloco_p0.25_H30_s0_innerx2 | 0.003 | — | 0.8/0.6 | 2.7900 | 2.377 | 1.758 | 2.598 | 4.426 | 4.426 | no | 14.1 | 0.545 | 0.063 | 0.059 | 0.449 | 0.375 |
| muloco_p0.25_H30_s0_innerx0.5 | 0.0015 | 0.006 | 0.8/0.6 | 2.5150 | 2.156 | 1.517 | 2.330 | 4.057 | 4.057 | no | 15.6 | 0.517 | 0.024 | 0.247 | 0.383 | 0.398 |
| muloco_p0.25_H30_s0_innerx2 | 0.0015 | 0.024 | 0.8/0.6 | 2.5135 | 2.153 | 1.519 | 2.331 | 4.051 | 4.051 | no | 15.1 | 0.517 | 0.024 | 0.246 | 0.383 | 0.398 |

| inner lr multiplier | ×0.5 | ×1 (Stage A) | ×2 |
|---|---|---|---|
| DiLoCo (AdamW lr) | **2.6867** (7.5e-4) | 2.7335 (1.5e-3) | 2.7900 (3e-3) |
| MuLoCo (Muon lr) | 2.5150 (6e-3) | 2.5157 (1.2e-2) | 2.5135 (2.4e-2) |

**Observations:**
- (2026-11-14) **DiLoCo wants a smaller inner AdamW lr than DP:** ×0.5 is better than ×1 by 0.047, and the trend is monotone. This is consistent with each worker stepping on ¼ of the DP batch (16 vs 64 seqs). The best value is at the low edge, so I launched an edge point at **×0.25 (3.75e-4)** (`manifests/stageC_edge.txt`).
- (2026-11-14) **MuLoCo is flat in its Muon lr:** ×0.5 / ×1 / ×2 are within 0.002 nats, far below any plausible seed noise. Per protocol (switch only if better by more than noise), **MuLoCo keeps Muon lr = 1.2e-2**.
  - Checked that this isn't a bug: worker pseudo-gradient norms scale exactly with the lr (‖Δ‖ at step 300 = 114 / 226 / 452 for ×0.5 / ×1 / ×2), and early evals differ (step 300: 4.177 / 4.104 / 4.061).
  - Likely mechanism, worth a sentence in the report: with RMSNorm + QK-norm and **no weight decay**, the hidden matrices are close to scale-invariant. A larger Muon lr grows the weights, which shrinks the effective lr (≈ lr/‖W‖), so the optimizer self-tunes. Muon's fixed-size orthogonalized steps make this especially clean.
- (2026-11-14) Fairness note for the write-up: both methods are tuned with the same protocol and the same number of inner-lr points. MuLoCo's AdamW part (embeddings / head / norms) stays at 1.5e-3 and was not re-tuned for the local setting.

#### Stage C edge extension: DiLoCo inner lr ×0.25 (2026-11-14)

| run_id | lr | lr_muon | outer | balanced | arxiv | github | stackex | wiki | worst | div | min | ρ_F | pair_cos | mom_align | interf_gap | ref_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diloco_p0.25_H30_s0_innerx0.25 | 0.000375 | — | 0.8/0.6 | 2.7499 | 2.329 | 1.708 | 2.554 | 4.409 | 4.409 | no | 14.8 | 0.538 | 0.053 | 0.047 | 0.454 | 0.360 |

DiLoCo inner-lr profile: ×0.25 → 2.750, **×0.5 → 2.687**, ×1 → 2.734, ×2 → 2.790. The best is interior, at **7.5e-4**.

- (2026-11-14) The DiLoCo inner-lr optimum (7.5e-4) is half the DP-AdamW optimum (1.5e-3), matching the ¼-size worker batch. Muon shows no such shift.
- Caveat for the write-up: DiLoCo's outer config (0.8/0.6) was chosen in Stage B at inner lr 1.5e-3, before Stage C lowered it. Per protocol we did not re-run Stage B. The outer optimum was broad (0.019 spread over η_out at μ = 0.6), so this is unlikely to matter.

### Frozen hyperparameters (2026-11-14): `configs/frozen.yaml`

| Method | Inner lr (AdamW) | Muon lr (hidden) | Outer lr / momentum |
|---|---|---|---|
| DP-AdamW | 1.5e-3 | — | — |
| DP-Muon | 1.5e-3 | 1.2e-2 | — |
| DiLoCo | **7.5e-4** | — | 0.8 / 0.6 (Nesterov) |
| MuLoCo | 1.5e-3 | 1.2e-2 | 0.8 / 0.6 (Nesterov) |
| FedAvg-AdamW | 7.5e-4 | — | 1.0 / 0 (plain averaging) |
| FedAvg-Muon | 1.5e-3 | 1.2e-2 | 1.0 / 0 (plain averaging) |

**DP seed-0 baselines come from Stage A** (same hyperparameters as frozen): `dp_adamw_s0_lr0.0015` (2.3079) and `dp_muon_s0_lrm0.012` (2.2490). Phase 2 adds DP seeds 1–2 as `dp_adamw_s1/s2` and `dp_muon_s1/s2`. The analysis must map the seed-0 baselines to these Stage A run ids. (Stage A seed 0 ran on H100 and Phase 2 runs on A100; hardware affects speed only.)

**Phase 1 complete (2026-11-14).** Spend: about $9 on day 1 (pilot + Stage A, H100) and about $11 on day 2 (Stages B and C plus edge points, A100), about $20 in total.


### Stage C: inner-LR check. *Not started.*

### Frozen hyperparameters. *Not yet frozen.*

---

## 3. Phase 2 (production). *Ready to launch.*
- 40 runs in `manifests/phase2_part{1..4}.txt`, 10 per part, each part including 3 seed-0 runs.
- Contents: E1 = {DiLoCo, MuLoCo} × p ∈ {0.25, 0.5, 0.75, 1.0} × seeds 0–2 (24 runs); E2 = {FedAvg-AdamW, FedAvg-Muon} × p ∈ {0.25, 1.0} × seeds 0–2 (12 runs); DP seeds 1–2 (4 runs).
- Everything uses `--config configs/frozen.yaml`.

### E1: heterogeneity sweep (H1)

### E2: outer-momentum ablation (H2)

---

## 4. Running observations and notes for the write-up
- (2026-11-13) Design decisions to describe in the report:
  - Deterministic skew p replaces Dirichlet, so the global mixture is identical across conditions and every arm trains on the same token multiset.
  - Each worker keeps its inner optimizer state across rounds.
  - Hyperparameters are tuned only on IID data.
- (2026-11-13) Infrastructure note: the data prep was split into a CPU-only Modal app, because Modal requires a payment method for any app containing an H100 function.
