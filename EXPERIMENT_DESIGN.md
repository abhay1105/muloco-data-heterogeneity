# Experimental Design: MuLoCo Under Non-IID Data

*EECS 182 Final Project. Shared reference for the team and for implementation.*
*Status: v3 (2026-11-13). The codebase is built and the local correctness tests pass. Scope is locked to the **minimal experiment** that answers H1 and H2. Items marked **[MEASURE]** get filled in after the pilot.*

---

## 0. TL;DR

- **Question:** MuLoCo (Muon as DiLoCo's inner optimizer) beats DiLoCo on IID data. Does that still hold when each worker trains on a different text domain? And does DiLoCo's outer Nesterov momentum protect Muon from the client drift that federated learning (FL) theory predicts?
- **Fixed setup:**
  - Model: GPT-style, d=512, 8 layers, about 25M non-embedding / 51M total parameters, context 1024.
  - Training: **K = 4 workers** simulated sequentially on 1 GPU, **~0.49B tokens per run** (7,500 steps × 65,536 tokens).
  - Data: 4 domains (ArXiv, GitHub, StackExchange, Wikipedia-en).
- **What we vary:**
  1. Inner optimizer: AdamW or Muon.
  2. Outer optimizer: Nesterov or FedAvg.
  3. Heterogeneity skew **p ∈ {0.25 (IID), 0.5, 0.75, 1.0 (one domain per worker)}**.
  - The sync interval is fixed at **H = 30**.
- **Heterogeneity is controlled by a deterministic skew `p`, not by random Dirichlet draws.** The global data mixture, and therefore the training objective and the DP baselines, stays identical across all heterogeneity levels (see §3.3).
- **Budget:** **58 runs** (18 for setup and tuning, 40 production), plus ~20% slack. Measured cost on Modal H100s is about **$55 total** (§6.2), mostly covered by the free $30/month per teammate account.
- **Workflow:**
  - Phase 1 (us): build the codebase, run smoke tests, pilot, and tune LRs.
  - Phase 2 (teammates): run frozen commands on hosted GPUs.
  - Then we compile the outputs (§6.3).

---

## 1. Changes from the original proposal

| # | Proposal said | Problem | Revised design |
|---|---|---|---|
| 1 | Each worker's domain mix is drawn from Dirichlet(α), α ∈ {∞, 1, 0.1, ~0} | Drawing a mix independently per worker with K=4 means that as α→0, two workers can land on the same domain and another domain gets no data. The *global* mixture also changes with α, so the objective, the DP baseline, and the balanced validation set no longer match across conditions. With 4 domains, α=0.1 is already nearly one domain per worker, so α=0.1 and α≈0 are almost the same condition. One random draw per seed also adds noise in how heterogeneous the split actually is. | **Deterministic skew p**: worker k gets fraction p of "its" domain k and (1−p)/3 of each other domain. The global mixture is always uniform. Heterogeneity is measured as total-variation distance h = p − 0.25 ∈ {0, 0.25, 0.5, 0.75}, which is evenly spaced (§3.3). |
| 2 | RedPajama-V1 ArXiv, Books, Wikipedia, GitHub | The RP-V1 Books subset was taken down (Books3 copyright). The RP-V1 HF loader relies on a custom loading script, which newer `datasets` releases no longer support. RP-V1 Wikipedia covers 20 languages, which the GPT-2 tokenizer handles badly and which mixes "domain" with "language." | **ArXiv, GitHub, StackExchange** from `DKYoon/SlimPajama-6B` (a RedPajama-derived parquet with a domain label on every document), plus **English Wikipedia** from `wikimedia/wikipedia` (§3.1). |
| 3 | 50–60 runs including 3 seeds | The full grid with 3 seeds is about 90 runs before any LR tuning. | Every run is accounted for in §6, about 71 runs total. Seeds are spent where they matter: 3 for the headline grid, 2 elsewhere. |
| 4 | "∼30–50M parameters" | With the GPT-2 vocabulary, the embedding is about half the model, so "50M" is ambiguous. It also matters for which parameters Muon handles. | Fixed architecture (§2.1): 25.2M non-embedding + 25.8M tied embedding ≈ 51M. Report both numbers. |
| 5 | H ∈ {15, 30, 60} swept across everything | A 4× range is narrow, and sweeping it at every heterogeneity level costs 3× the grid. Neither hypothesis needs it. | **H = 30 fixed** (MuLoCo's default). An H sweep is deferred (§4, "Deferred"). |
| 6 | "Learning rates tuned on IID, held fixed" | Didn't say *which* rates. Each worker's batch is 1/4 of the DP batch, so the inner LR optimum can differ from the DP optimum. | Explicit 3-stage tuning protocol (§5). |
| 7 | Not specified | Whether inner optimizer state persists across rounds, what reference the cosine-similarity diagnostic uses, and what "% over DP" means. | Each worker keeps its inner optimizer state across rounds (DiLoCo convention). Diagnostics are defined precisely (§7). Metrics are defined precisely (§8). |
| 8 | H1: "local matrices will drift irrecoverably." H2: FedAvg "will fail entirely" with Muon. | Overstated. Liu et al. found local Muon actually **beats** local AdamW on non-IID CIFAR. Adam is also a nonlinear, sign-like step, so the "averaging nonlinear updates is biased" argument applies to DiLoCo too. Takezawa et al. note that outer momentum cannot remove a biased fixed point. | Reworded hypotheses with competing predictions and measurable criteria (§2.3). |
| 9 | Use the `modded-nanogpt` Muon | It's built for multi-GPU / distributed training. | Use `torch.optim.Muon` (PyTorch ≥ 2.9), falling back to Keller Jordan's `SingleDeviceMuon`. |

---

## 2. Research questions and hypotheses

### 2.1 Fixed experimental context
- K = 4 workers, synchronous DiLoCo-style training, models trained **from scratch**.
  - DiLoCo's own non-IID experiment started from a pretrained checkpoint and split C4 by k-means. Training from scratch with domain-level skew is one of our main differences from that work.
- All arms see **exactly the same multiset of training tokens** (§3.4). Only the *assignment* of tokens to workers changes.

### 2.2 Questions
- **Q1 (alignment under heterogeneity):** On IID data, MuLoCo's averaged updates point more consistently and it loses less relative to its DP baseline. Does that advantage survive domain-skewed workers?
- **Q2 (degradation rate):** How does each method's *heterogeneity penalty* grow with skew h (at fixed H = 30)? Does it grow faster for MuLoCo (Muon inner) than for DiLoCo (AdamW inner)?
- **Q3 (role of outer momentum):** How much of the heterogeneity penalty does outer Nesterov momentum remove compared with plain averaging (FedAvg), and is the effect larger for Muon?

### 2.3 Hypotheses (stated as competing predictions)

**H1: does Muon degrade faster under heterogeneity?** Two predictions compete:
- **H1-FL:** the heterogeneity penalty ΔL_het(h) grows faster with h for MuLoCo than for DiLoCo. The reasoning is that averaging orthogonalized updates is not the same as orthogonalizing the averaged update (Takezawa et al. 2025). Because Muon's step size doesn't depend on gradient magnitude, workers' conflicting domain-specific updates cancel when averaged instead of shrinking gracefully.
- **H1-MuLoCo:** MuLoCo keeps a smaller or equal penalty. Its steps are better conditioned and its averaged updates lose less to averaging (as in Thérien et al. 2025), which offsets the drift. Liu et al. 2025 also found local Muon beating local AdamW on non-IID data.
- **Decision rule:** compare the paired seed differences of ΔL_het between the methods at h = 0.75. The comparison only counts when all paired differences have the same sign *and* their mean exceeds 2× the across-seed standard deviation of the IID runs. Otherwise we report "no detectable difference."
- **Mechanism check:** norm retention ρ_F and the other alignment diagnostics (§7) should fall more steeply with h for whichever method degrades more.

**H2: outer momentum partially rescues Muon.**
- Under FedAvg (outer lr = 1, μ = 0), the heterogeneity penalty is larger than under Nesterov for both inner optimizers.
- The rescue is **larger for Muon** than for AdamW.
- Nesterov does **not** remove the penalty entirely. Momentum can't move a biased fixed point (Takezawa et al.), so some bias floor remains.
- The test is a difference-in-differences (§8.3).

*(An H-dependence hypothesis, drift ∝ inner lr × H, is deferred along with the H sweep.)*

---

## 3. Data

### 3.1 Domains and sources

| Domain | Source | Notes |
|---|---|---|
| ArXiv | `DKYoon/SlimPajama-6B`, `meta.redpajama_set_name == "RedPajamaArXiv"` | LaTeX and math-heavy |
| GitHub | same, `"RedPajamaGithub"` | code |
| StackExchange | same, `"RedPajamaStackExchange"` | Q&A, mixed prose and code |
| Wikipedia (en) | `wikimedia/wikipedia`, config `20231101.en` (load only the first few parquet shards) | Replaces RP's multilingual wiki and the removed Books subset |

- The exact `redpajama_set_name` strings still need verifying during data prep.
- SlimPajama-6B is about 14 GB of parquet. The domains are interleaved across files, so download everything, filter, and discard the rest.
- Rough per-domain size in SlimPajama-6B: ArXiv ~3.4% (~200M tokens), GitHub ~4.2% (~250M), StackExchange ~2.8% (~170M).

**Needed per domain:** 122.9M train + 1M validation + 6M probe ≈ **130M GPT-2 tokens**.
- **[MEASURE]** Check the counts after tokenizing. StackExchange is the tightest.
- If any domain falls short, top up by streaming one `chunk2` file from `cerebras/SlimPajama-627B` and filtering by the same label.

Why not RedPajama-V1 directly: the Books subset was taken down, the HF loader relies on a custom loading script that newer `datasets` releases no longer support, and the raw jsonl URLs are a 2,084-file download. SlimPajama is RedPajama cleaned and deduplicated, so the proposal's "RedPajama domains" framing still holds.

### 3.2 Preprocessing (`src/prepare_data.py`)
1. Load the data, filter to the 4 domains, and split by document (seeded) into `train`, `val`, and `probe` pools.
2. Tokenize with the **GPT-2 tokenizer** (`tiktoken`, vocab 50,257, padded to 50,304), appending `<|endoftext|>` after each document.
3. Write `data/{domain}_{split}.bin` as uint16 token streams, nanoGPT-style, plus a `meta.json` with token counts.
4. Each train stream is truncated to exactly **T_d = 122,880,000 tokens** (= 7,500 × 65,536 / 4).
   - Train pools: the tokens actually trained on.
   - Val pool: 1M tokens per domain. Its fixed first 256K is used for periodic evals, and all of it for the final eval.
   - Probe pool: ~6M tokens per domain, used only for the reference-alignment diagnostic (§7).

### 3.3 Heterogeneity parameterization (replaces Dirichlet)
Worker k ∈ {0,1,2,3} is assigned "home" domain k. The allocation matrix is:

```
a[k][d] = p            if d == k
        = (1 - p) / 3  otherwise
```

Every column sums to 1, so **each domain's train pool is split exactly once across the 4 workers.** The global mixture is uniform (25% per domain) for every p.

| p | h = TV(worker mix, global mix) | Worker 0 mix (Arx/Git/SE/Wiki) | Meaning |
|---|---|---|---|
| 0.25 | 0.00 | 25/25/25/25 | IID |
| 0.50 | 0.25 | 50/16.7/16.7/16.7 | mild skew |
| 0.75 | 0.50 | 75/8.3/8.3/8.3 | strong skew |
| 1.00 | 0.75 | 100/0/0/0 | one domain per worker (disjoint) |

Why not Dirichlet: the reasons are in §1, row 1. **Decided:** the skew p replaces Dirichlet entirely. The proposal was only a draft.

### 3.4 Worker data streams
- For each domain d, cut its train stream into 4 **disjoint contiguous slices** of sizes a[k][d]·T_d, one per worker k.
- Worker k's pool is the union of its 4 slices, chunked into 1,025-token windows (1,024 inputs + 1 shifted target) and **shuffled with the run seed**.
- Worker k consumes 16 sequences per step for 7,500 steps, which uses its entire pool exactly once. There is no repetition and no overlap between workers.
- **DP baselines** use the union of all slices, which is the *same token multiset*, shuffled, at 64 sequences per step.
- **Pairing:** a given seed fixes the model initialization and shuffle order. Every method uses the same seed set, so we can compare methods with paired differences.

---

## 4. Experiment grid

Method names:

| Name | Inner (hidden matrices) | Inner (embed / head / norms) | Outer |
|---|---|---|---|
| **DP-AdamW** | AdamW | AdamW | (none: 1 worker, batch 64) |
| **DP-Muon** | Muon | AdamW | (none) |
| **DiLoCo** | AdamW | AdamW | Nesterov SGD (tuned η_out, μ) |
| **MuLoCo** | Muon | AdamW | Nesterov SGD (tuned η_out, μ) |
| **FedAvg-AdamW** | AdamW | AdamW | SGD, η_out = 1, μ = 0 (plain averaging) |
| **FedAvg-Muon** | Muon | AdamW | SGD, η_out = 1, μ = 0 |

### E0: Pilot and sanity checks (~2 runs-equivalent)
- Throughput measurement on the target GPU. This fills in **[MEASURE]** values: tokens/s, wall time per run.
- Correctness tests (§10.3).
- Center the LR grids by running one short run per optimizer at a guessed LR.

### E1: Core heterogeneity sweep (answers Q1, Q2, H1)
`{DiLoCo, MuLoCo} × p ∈ {0.25, 0.5, 0.75, 1.0} × H = 30 × seeds {0,1,2}` → **24 runs**

Plus DP baselines: `{DP-AdamW, DP-Muon} × seeds {1,2}` → **4 runs**. Seed 0 comes from the tuning sweep (§5, stage A). DP results don't depend on p, since the global mixture is fixed.

### E2: Outer-momentum ablation (answers Q3, H2)
`{FedAvg-AdamW, FedAvg-Muon} × p ∈ {0.25, 1.0} × H = 30 × seeds {0,1,2}` → **12 runs**

The Nesterov counterparts come from E1 (seeds 0–2). E2 was raised from 2 to 3 seeds on 2026-11-14 so that H2's error bars are meaningful, as the report requires.

### Dimensions deliberately *not* varied (and why)
- **Number of workers K:** fixed at 4, matching the 4 domains.
  - Changing K would change what heterogeneity means: more workers than domains means duplicate home domains, and fewer means mixed homes. That confounds everything.
  - MuLoCo's advantage first appears at K > 2, so K = 4 is the smallest K where there is something to lose.
- **Model size and token budget:** one scale. Scaling is MuLoCo's topic, not ours.
- **Compression / quantization:** out of scope.
- **Weight decay, warmup, schedule shape:** fixed, not tuned (§5.4).

- **Sync interval H:** fixed at 30 (see Deferred).

### Deferred (not planned; consider only after the minimal experiment is collected)
- H sweep (H ∈ {10, 100} at p ∈ {0.25, 1.0}), which would test drift ∝ H.
- Plain outer SGD with the *tuned* η_out and μ = 0, which separates momentum from "outer lr ≠ 1."
- A Takezawa-style control-variate "rescue" arm.
- Starting from a partly trained checkpoint instead of from scratch.
- A single Dirichlet-split robustness check.

---

## 5. Hyperparameters and tuning protocol

**Rule:** all tuning happens on **IID data (p = 0.25 or DP)** at **H = 30**. Everything is frozen before E1 starts, and nothing is retuned for non-IID runs or other values of H.

### 5.1 Fixed settings (not tuned)

| Setting | Value |
|---|---|
| Sequence length | 1,024 |
| Global batch | 64 sequences = 65,536 tokens/step. DP: 64 per step. DiLoCo-family: 16 per worker per step × 4 workers. |
| Total steps | 7,500 (inner steps per worker = DP steps), giving ~491.5M tokens |
| LR schedule | Linear warmup 250 steps, then cosine decay to 0.1× peak at step 7,500. Indexed by **global inner step**, identical across workers. The *inner* LR follows the schedule; the outer LR is constant. |
| AdamW | β = (0.9, 0.95), ε = 1e-8, weight decay **0** |
| Muon | momentum 0.95 (Nesterov), 5 Newton–Schulz steps, `adjust_lr_fn="match_rms_adamw"` so Muon LRs are on the AdamW scale, weight decay **0** |
| Muon param set | All 2-D weight matrices inside transformer blocks (q, k, v, o, fc, proj). AdamW handles the tied embedding/LM head and RMSNorm gains. The **same split is used in DP-Muon, MuLoCo and FedAvg-Muon.** |
| Grad clipping | 1.0 global norm, applied per worker per step |
| Precision | bf16 autocast, fp32 master weights, fp32 optimizer states, fp32 outer state |
| Inner optimizer state | **Persistent per worker across rounds.** Never averaged or reset. This is the DiLoCo convention. |
| Outer optimizer | `torch.optim.SGD(nesterov=True)` on the pseudo-gradient Δ̄ = θ_outer − mean_k θ_k. FedAvg = SGD(lr=1, momentum=0), so θ_outer ← mean_k θ_k exactly. |

Why weight decay is 0: it removes a tuning dimension and barely matters at about 20 tokens per non-embedding parameter. We note this as a limitation.

### 5.2 Tuning stages

**Stage A: DP inner LR.** Full-length runs, seed 0. The best run of each optimizer becomes the seed-0 DP baseline.
- DP-AdamW: lr ∈ {7.5e-4, 1.5e-3, 3e-3, 6e-3} → 4 runs.
- DP-Muon: hidden-matrix lr ∈ a 4-point ×2 grid centered by the pilot → 4 runs.
  - Embedding/head AdamW lr is fixed to the best DP-AdamW lr, so Stage A runs AdamW first.
- If the best value is at an edge of the grid, add one more point (budgeted in the slack).

**Stage B: outer (η_out, μ) for DiLoCo and MuLoCo.** IID, H = 30, seed 0, **half-length runs** (3,750 steps with the schedule compressed to fit).
- Inner LRs come from Stage A.
- Grid: η_out ∈ {0.5, 0.8, 1.0} × μ ∈ {0.6, 0.9} → 6 per method → 12 half-runs = **6 run-equivalents**.
- Expected optimum, from DiLoCo and MuLoCo at K = 4: around η_out 0.7–1.0, μ 0.8–0.9.

**Stage C: inner-LR sanity check for local training.** Each worker's batch is 16 sequences instead of 64.
- At the best Stage-B outer config, try inner lr × {0.5, 2}.
- 2 per method → 4 half-runs = **2 run-equivalents**.
- If one beats Stage A's lr by more than seed noise, use it. FedAvg arms then use the same inner lr as their Nesterov counterparts.

**Frozen after Stage C:** inner LRs for each method, and (η_out, μ) for DiLoCo and MuLoCo. FedAvg is fixed by definition. Record all of these in `configs/frozen.yaml`.

### 5.3 Fairness notes
- Each DiLoCo-family method is compared against **its own** DP baseline (MuLoCo vs DP-Muon, DiLoCo vs DP-AdamW), as in MuLoCo.
- **Absolute** losses are always reported as well.
- FedAvg isn't tuned by construction. That's intentional: it's the "no outer optimizer" control.

---

## 6. Compute budget

A unit **R** is one full-length run of 7,500 steps and ~491.5M tokens.
- Estimated FLOPs: ≈ 6 × (25.2M non-embedding + 25.8M LM-head) params × 4.9e8 tokens, plus attention ≈ **1.8e17 FLOPs**.
- Estimated time: **~0.25–0.35 h on an H100**, ~0.5 h on an A100. **[MEASURE]** in E0.
- Simulating workers sequentially adds no FLOPs. Diagnostics add under 3%.

### 6.1 Run budget

| Phase | What | Runs | R-equivalents |
|---|---|---|---|
| **1 (us)** | E0 pilot + correctness tests on GPU | short | 2 |
| 1 | Stage A: DP LR sweeps (best runs = DP seed 0) | 8 | 8 |
| 1 | Stage B: outer (η_out, μ) sweep, half-length | 12 | 6 |
| 1 | Stage C: inner-LR check, half-length | 4 | 2 |
| | **Phase 1 subtotal** | | **18 R** |
| **2 (team)** | E1: {DiLoCo, MuLoCo} × 4 p × seeds 0–2 | 24 | 24 |
| 2 | DP baselines, seeds 1–2 | 4 | 4 |
| 2 | E2: {FedAvg-AdamW, FedAvg-Muon} × p ∈ {0.25, 1.0} × seeds 0–2 | 12 | 12 |
| | **Phase 2 subtotal** | | **40 R** |
| | **Total** | | **58 R** (+~10 R slack for reruns or an extra edge-of-grid LR point) |

**Phase 2 run order:**
1. Seed 0 of everything (E1, E2): 12 runs. This alone gives a first answer to H1 and H2.
2. Remaining seeds.

**If money or time gets tight, cut in this order:** E1 seed 2 at p ∈ {0.5, 0.75} → E2 seed 2. **Never cut** the IID and p = 1.0 points of E1.

### 6.2 Platform, GPU and cost
**Platform: [Modal](https://modal.com). GPU: H100 for the pilot and Stage A, A100-80GB from Stage B on (see update below).**
- Modal runs our training function in a cloud GPU container launched from a laptop command, and bills per second only while code runs. Nothing idles, so a forgotten pod can't run up a bill.
- The Starter plan includes **$30/month of free credit per workspace**, and each teammate's account has its own. That should cover most or all of the project. **Verify on modal.com/pricing.**
- Up to 10 GPUs can run at once, so each tuning stage finishes in roughly one run's wall-clock time.
- Modal has no 4090s. On Modal, the H100 gives the best cost per run.

**Measured in the pilot (2026-11-13, H100 80GB HBM3, `torch.compile` on, 300-step runs):**

| Run type | Training throughput | Overheads (per full run) | Est. wall time per full run | ~$ per run at ~$3.95/hr |
|---|---|---|---|---|
| DP (AdamW or Muon) | ~900–960k tok/s (3.4 s per 50 steps) | ~1 min (compile, evals) | **~9–10 min** | ~$0.65 |
| DiLoCo (local, K=4) | ~940k tok/s (2.1 s per round) | spectral diagnostics ~5 s × 25, probes ~3.5 s × 3, evals | **~11–12 min** | ~$0.75 |
| MuLoCo (local, K=4) | ~740k tok/s (2.7 s per round) | same | **~13–14 min** | ~$0.90 |

| Phase | Runs | Est. GPU-hours | Est. cost |
|---|---|---|---|
| Phase 1 (pilot done; Stage A: 8 full DP; Stages B+C: 16 half-length local) | 24 | ~3.3 h | ~$13 |
| Phase 2 (40 full runs) | 40 | ~8.3 h | ~$33 |
| **Total incl. ~20% slack** | | **~14 h** | **~$55** |

Runs are about 2× faster than the original estimate. Every run is well under 30 minutes, so the run budget (§6.1) is unchanged.

**Update 2026-11-14: H100 is no longer available on the Modal Starter plan** (launches queue indefinitely with "H100 unavailable on starter plan"). From Stage B on, runs use **A100-80GB** (~$2.50/hr), now the default in `modal_app.py`.

| A100-80GB (measured in Stage B) | Training speed | Est. full run | ~$ per full run |
|---|---|---|---|
| DP | ~350k tok/s (est.) | ~25 min | ~$1.05 |
| DiLoCo | ~340–390k tok/s | ~28 min | ~$1.15 |
| MuLoCo | ~290–340k tok/s | ~32 min | ~$1.35 |

Phase 2 (40 runs) comes to **~$45–50**. Split across 3–4 teammates' workspaces, that's about $12–16 each, inside the free $30/month credit. Results don't depend on GPU type; only speed and cost do.

- **Develop for free:** write code and run tiny-config smoke tests and correctness tests on a laptop (CPU or MPS, `--compile 0`). Only spend credits once they pass.
- **Platform independence:**
  - The core is a plain CLI, `python -m src.train ...`, that runs anywhere.
  - `modal_app.py` is a thin wrapper.
  - Switching to RunPod or another provider means dropping the wrapper, not rewriting code.

**Modal resources:**

| Resource | What it is |
|---|---|
| Volume `muloco-data` | Tokenized `.bin` files and `meta.json`, about 1.1 GB, written by `modal_prepare.py` (a separate CPU-only app, costing a few cents) |
| Volume `muloco-runs` | One directory per run |
| Function `train` | GPU, one run per call |
| Entrypoint `launch` | Runs a manifest file of runs in parallel and skips runs that already have a `summary.json` |

- **Data identity across teammates:**
  - Each Modal workspace has its own volumes, so each teammate runs `prepare` once.
  - Dataset revisions are pinned and preparation is deterministic.
  - `meta.json` records a SHA-256 hash of every `.bin` file, and every run logs it. Hashes must match across teammates.

### 6.3 Workflow
- **Phase 1 (us):**
  1. Build the codebase.
  2. Pass the correctness tests (§10.3) locally.
  3. GPU pilot: measure throughput, center the LR grids.
  4. Stages A–C.
  5. Write `configs/frozen.yaml`.
  6. Produce a **run manifest**: one exact command per Phase 2 run, assigned to a teammate.
- **Phase 2 (one runner, decided 2026-11-15):** the team agreed one person runs all 40 runs on the workspace used for Phase 1, with costs split at the end. Runs go in two batches: `manifests/phase2_batch1.txt` (seed 0, 12 runs, a sanity check against Phase 1) and `manifests/phase2_batch2.txt` (seeds 1–2, 28 runs; can be split with `make_manifest.py phase2 --batch2_parts N` to fit the Modal spend limit).
- **Collection and backup:** after each batch, `scripts/sync_runs.py` downloads finished runs from the `muloco-runs` volume to `runs/`, verifies them, writes `CHECKPOINTS.sha256`, and uploads everything (logs + checkpoints) to the Hugging Face model repo `paidvenk/muloco-data-heterogeneity`. Logs and checksums are then committed to git. Copies: logs on GitHub, Hugging Face, laptop and Modal; checkpoints on Hugging Face, laptop and Modal. The tokenized data is not archived; it is rebuilt deterministically (pinned revisions, `data_hash`).
- **Parallelism:** `launch` fans out up to 10 runs at once per workspace.

**Storage:**
- Tokenized data: ~1.1 GB.
- Final checkpoints: ~100 MB each in bf16. Keep only the outer model.
- No intermediate checkpoints are needed, because the diagnostics are computed online (§7).

---

## 7. Diagnostics (mechanism evidence)

All diagnostics are computed on the **hidden 2-D matrices** (the Muon-managed set), which are **the same tensors in both arms**, so the comparisons are apples to apples.

Notation: worker k's pseudo-gradient is Δ_k = θ_outer − θ_k after H local steps, and Δ̄ = (1/4) Σ_k Δ_k. Diagnostics are computed per matrix, then averaged per type (attn-qkv, attn-o, mlp-fc, mlp-proj) and over all matrices.

| Diagnostic | Definition | Frequency | What it shows |
|---|---|---|---|
| Norm retention ρ_F | ‖Δ̄‖_F / mean_k ‖Δ_k‖_F | every round | 1 = workers fully aligned. ≈0.5 = orthogonal with equal norms (1/√K). →0 = updates cancel. **Primary drift metric.** |
| Pairwise cosine | mean over j<k of cos(Δ_j, Δ_k) | every round | Direct measure of worker conflict |
| Worker step norms | ‖Δ_k‖_F per worker | every round | MuLoCo's "erratic AdamW norms" finding, and domain imbalance |
| Momentum alignment | cos(Δ̄, outer momentum buffer m), taken before the update; Nesterov runs only | every round | Whether outer momentum agrees with or fights the new update (HeLoCo-style) |
| Nuclear retention ρ_* | ‖Δ̄‖_* / mean_k ‖Δ_k‖_* | rounds at multiples of 300 steps | Spectral cancellation |
| Interference gap G_S | MuLoCo Def. 4.1 with top-5% singular values, normalized: [mean_k Σ_{j≤S} σ_j(Δ_k) − Σ_{j≤S} σ_j(Δ̄)] / mean_k Σ_{j≤S} σ_j(Δ_k) | multiples of 300 steps | Comparable to MuLoCo's "spectrum collapse" result |
| Reference alignment | cos(Δ̄, Δ_ref). Δ_ref = H steps of a single DP learner from the same θ_outer, batch 64 from the **probe** pool (uniform mixture), with optimizer state = average of worker states. | 3 rounds (25/50/75% of training), E1 seed 0 only | MuLoCo's headline metric ("pseudo-gradients closer to the DP direction") |
| Specialization matrix | Before averaging, eval each worker's local θ_k on each domain's 256K val subset → a 4×4 loss matrix | same 3 rounds | Local–global inconsistency: how far workers drift toward their home domain |

All of these are logged to a per-run JSONL file. W&B is optional.

---

## 8. Metrics and analysis

### 8.1 Primary metric
**Domain-balanced validation loss** of the outer model θ_outer: L = (1/4) Σ_d L_d, using the full 1M-token val pool per domain, evaluated at the end of training.
- During training, evaluate every 300 steps on the 256K-token subsets. These points are always sync points, since 300 is divisible by 10, 30 and 100.
- Also report **per-domain** L_d and the **worst-domain** max_d L_d.

### 8.2 Derived quantities
- **Heterogeneity penalty:** ΔL_het(m, p, H) = L(m, p, H) − L(m, 0.25, H), computed per seed and paired. This is the main H1 quantity, because it cancels each method's IID efficiency and any mistuning at that H.
- **Gap to DP (MuLoCo-style):** G_DP(m, p) = L(m, p) / L(DP of same inner optimizer) − 1, in percent.

### 8.3 Hypothesis tests (3 seeds, so these are descriptive with a pre-stated rule, not p-values)
- **H1:** compare ΔL_het(MuLoCo, p) and ΔL_het(DiLoCo, p) across p.
  - Report mean ± std and paired differences.
  - Also report the slope of ΔL_het vs h from a least-squares fit.
  - Decision rule as in §2.3.
- **H2:** for each inner optimizer i, compute the outer-momentum rescue
  R_i = [L(FedAvg_i, 1.0) − L(FedAvg_i, 0.25)] − [L(Nest_i, 1.0) − L(Nest_i, 0.25)]
  - H2 predicts R_Muon > R_AdamW > 0, with L(Nest_Muon, 1.0) − L(Nest_Muon, 0.25) still > 0.

### 8.4 Planned figures
1. Final balanced loss vs h for DiLoCo and MuLoCo, with DP-AdamW and DP-Muon as horizontal bands. *(The headline figure.)*
2. ΔL_het vs h for both methods, with seed error bars.
3. Per-domain loss heatmap: method × p × domain.
4. 2×2 inner × outer grouped bars at p = 0.25 and p = 1.0 (H2).
5. ρ_F and pairwise cosine over training, one panel per p, both methods overlaid.
6. Interference gap and nuclear retention vs h.
7. Worker specialization matrices for p = 1.0, DiLoCo vs MuLoCo.

---

## 9. Model

| Setting | Value |
|---|---|
| Architecture | Decoder-only, pre-norm RMSNorm, RoPE, **QK-norm**, no biases, no dropout |
| Layers / d_model / heads / head dim | 8 / 512 / 8 / 64 |
| MLP | 4× (2,048), GELU |
| Vocab | 50,304 (GPT-2, padded), **tied** input embedding / LM head |
| Params | 25.2M non-embedding + 25.8M embedding ≈ 51M |
| Attention | `F.scaled_dot_product_attention` (causal, flash) |
| Init | Standard (normal, std 0.02, residual projections scaled by 1/√(2·n_layer)), seeded |
| Compile | `torch.compile` on the model |

QK-norm is included because it's cheap stabilization that helps Muon. Tokens per run ≈ 19.5 × non-embedding params.

---

## 10. Implementation plan

### 10.1 Repository layout (as built)
```
muloco-data-heterogeneity/
├── EXPERIMENT_DESIGN.md        # this file
├── requirements.txt            # local dev env (torch>=2.9 for torch.optim.Muon, modal, pytest, ...)
├── src/
│   ├── config.py               # Config dataclass; defaults = §5.1; --config YAML (+ per-method overrides)
│   ├── prepare_data.py         # download → filter → split → tokenize → {domain}_{train,val,probe}.bin + meta.json
│   ├── model.py                # GPT (§9); is_hidden() defines the Muon / diagnostics matrix set
│   ├── optim.py                # param split, InnerOptimizer (per worker), averaged_state, LR schedule
│   ├── partition.py            # skew-p allocation (§3.3), disjoint slices, seeded shuffles
│   ├── data.py                 # TokenData (in-RAM tokens), Pools, eval / probe rows
│   ├── diagnostics.py          # §7 metrics
│   └── train.py                # Runner: DP + local (DiLoCo-family) loops, eval, logging, summary
├── configs/frozen.example.yaml # template; copy to frozen.yaml after Stage C
├── scripts/make_manifest.py    # writes manifests/{pilot,stageA,stageB,stageC,phase2*}.txt
├── modal_app.py                # Modal GPU app: train / launch / status
├── modal_prepare.py            # Modal CPU app: builds the data volume
├── tests/test_core.py          # §10.3 correctness tests (pytest, CPU, synthetic data, ~10 s)
└── analysis/                   # (to do) §8 figures from runs/*/
```
**Run ID format:** `{method}_p{p}_H{H}_s{seed}` for local runs, `{method}_s{seed}` for DP runs, plus `_{tag}` for tuning runs. Examples: `muloco_p1.00_H30_s0`, `dp_adamw_s0_lr0.003`, `diloco_p0.25_H30_s0_eo0.8_mu0.9`.

**Run outputs** (`<out_dir>/<run_id>/`):
- `config.json`.
- `metrics.jsonl`, one JSON object per line with `kind` ∈ {start, train, eval, diag, final_eval, diverged}.
- `summary.json`, written last, so it doubles as the completion marker. It holds the final per-domain, balanced and worst losses, the diverged flag, throughput, and the data hash.
- `final.pt`, the bf16 outer model.

### 10.1b Commands
```bash
# local dev (once)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                       # correctness tests (must pass before spending credits)
.venv/bin/python -m src.prepare_data --out data_tiny --synthetic   # tiny data for manual smoke runs

# Modal (once per workspace)
modal setup
modal run modal_prepare.py                          # ~20-40 min CPU; prints data_hash (must match across teammates)

# Phase 1
python scripts/make_manifest.py pilot && modal run modal_app.py::launch --manifest manifests/pilot.txt
python scripts/make_manifest.py stageA ...          # then stageB / stageC with the winners
modal run --detach modal_app.py::launch --manifest manifests/stageA.txt
modal volume get muloco-runs / runs/                # download results

# Phase 2 (after configs/frozen.yaml exists)
python scripts/make_manifest.py phase2              # phase2.txt (40) + phase2_batch1.txt (seed 0) + phase2_batch2.txt
modal run --detach modal_app.py::launch --manifest manifests/phase2_batch1.txt
python scripts/sync_runs.py                         # download + verify + checksums + Hugging Face backup
```

### 10.2 Training loop (sequential worker simulation)
```text
θ_outer ← init(seed);  outer_opt ← SGD([θ_outer], lr=η_out, momentum=μ, nesterov=(μ>0))
for k in 0..3:  θ_k-state ← {}; inner_opt_k ← build(inner, params=model)   # one optimizer per worker,
                                                                           # all bound to the SAME model params
for round r = 0 .. 7500/H − 1:
    for k in 0..3:
        model.params.copy_(θ_outer)                      # all workers start from θ_outer
        for t in 1..H:  step = r·H + t
            set inner LR for every group by schedule(step)
            loss = model(next_batch(worker=k)); backward; clip; inner_opt_k.step(); zero_grad
        Δ_k ← θ_outer − model.params                     # store (fp32), per hidden matrix + rest
    Δ̄ ← mean_k Δ_k;  log diagnostics(Δ_1..4, Δ̄, outer momentum)
    θ_outer.grad ← Δ̄;  outer_opt.step()
    if (r+1)·H % 300 == 0: eval θ_outer
```
- One compiled model is reused. Each worker's optimizer state lives in its own optimizer instance.
- Only parameter *values* are swapped, using `copy_`, so the compiled graph and the optimizer's param references stay valid.
- DP mode is the same code path with K = 1, batch 64, and outer step = identity.

### 10.3 Correctness tests (E0)
1. **K=1, H=1, η_out=1, μ=0 ≡ DP:** losses match the DP run to bf16 tolerance (same seed, same data order).
2. **Averaging exactness:** with inner = plain SGD, IID, H = 1, FedAvg over 4 workers with 16 sequences each equals DP with 64 sequences (mean of gradients).
3. **Param grouping:** print the counts in Muon and in AdamW groups. Every block matrix should be in Muon, and the embedding and norms in AdamW.
4. **Data accounting:** realized per-worker domain token counts match a[k][d]·T_d exactly, there's no overlap between workers, and the DP union equals the worker union.
5. **Diagnostics sanity:** feeding identical Δ_k gives ρ_F = 1 and cosine = 1. Feeding 4 random orthogonal Δ_k gives ρ_F ≈ 0.5.

---

## 11. Known limitations (state these in the report)
- One model scale (~51M), one token budget (~0.49B), K = 4, H = 30 only.
- One sync interval (H = 30). Outer hyperparameters are tuned on IID data only and reused for non-IID runs.
- 2–3 seeds, so conclusions are about effect sizes, not significance tests.
- Weight decay is 0 and the schedule isn't tuned.
- Domains come from two sources (SlimPajama for 3 domains, Wikimedia for Wikipedia). Domain "distance" is uneven: StackExchange overlaps with GitHub and ArXiv more than Wikipedia does.

---

## 12. Decisions log
- ✅ Dirichlet is replaced by the deterministic skew p. The proposal was a draft.
- ✅ Platform: Modal, each teammate on their own free-credit workspace. GPU: H100 for the pilot and Stage A; **A100-80GB from 2026-11-14** (H100 blocked on the Starter plan). Development and correctness tests happen locally.
- ✅ Scope: minimal experiment only (E1 + E2 + DP baselines). Everything in "Deferred" waits until that data is collected.
- ✅ Workflow: we build, test and tune; teammates run frozen commands (§6.3). The collection mechanism will be decided later.
- ✅ Data built on Modal (2026-11-13): **data_hash = `33c0c658fbf49a8a`**. Each domain has 122,880,001 train, 1,024,001 val and 2,097,153 probe tokens. Every domain hit the 158M collection target (StackExchange needed ~32 of 48 SlimPajama shards). Teammates' `prepare` must reproduce this hash.
- ✅ Phase 1 tuning complete (2026-11-14). Frozen in `configs/frozen.yaml`: AdamW 1.5e-3 (DP / MuLoCo AdamW part), DiLoCo inner 7.5e-4, Muon 1.2e-2, outer 0.8/0.6 Nesterov for DiLoCo and MuLoCo. Details in RESULTS.md §2.
- ✅ E2 raised to 3 seeds (2026-11-14), so Phase 2 has 40 runs.
- ✅ Phase 2 runner (2026-11-15): one person runs all 40 runs (costs split at the end) instead of per-teammate manifests; two batches; backups via `scripts/sync_runs.py` to Hugging Face (`paidvenk/muloco-data-heterogeneity`, private until submission). Modal spend limit is capped at $20 for this account until charges build history.
- ✅ Phase 2 complete (2026-11-15): 40/40 runs, none diverged, all backed up. H1-FL supported by the pre-registered rule (MuLoCo penalty larger, +0.034 at h=0.75); H2 not supported (R_AdamW +0.010 > 0 > R_Muon −0.009). Details in RESULTS.md §3; figures in `figures/`.
- ✅ Pilot (2026-11-13): all 4 run types work on H100 with `torch.compile`, and throughput is recorded in §6.2. The pilot is too short to center the LR grids, so Stage A uses the default grids and adds an edge point if needed.

---

## References
- Douillard et al. 2023, *DiLoCo*, arXiv:2311.08105. (Non-IID result: k-means C4 split from a pretrained init; plain-SGD/FedAvg outer did poorly even on IID data.)
- Thérien et al. 2025, *MuLoCo*, arXiv:2505.23725. (Alignment = cosine to the K=1 pseudo-gradient; interference gap Def. 4.1; tuned outer lr/μ ≈ 0.7–1.0 / 0.6–0.9; IID only.)
- Jordan 2024, *Muon*. `torch.optim.Muon` (PyTorch ≥ 2.9).
- Liu et al. 2025, *FedMuon (momentum aggregation + local–global alignment)*, arXiv:2510.27403. (Local Muon > local AdamW on non-IID CIFAR; server averages only.)
- Takezawa et al. 2025, *FedMuon (bias-corrected LMO)*, arXiv:2509.26337. (Naive local LMO doesn't converge under heterogeneity; server momentum doesn't fix it; SCAFFOLD-style correction.)
- Karimireddy et al. 2020, *SCAFFOLD*. McMahan et al. 2017, *FedAvg*.
- Al Asif et al. 2026, *HeLoCo*, arXiv:2606.00271. (Cosine to outer momentum as a drift reference; async; language-split.)
