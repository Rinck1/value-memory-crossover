# Phase-2 & ICML Submission Master Handoff: `125M × 2.5B` Baselines, Ablations & Downstream Eval

**Date:** 2026-11-26  
**Code Status:** All Phase-2 arms (`engram`, `xgram`, `u123_no3g`, `plain184`, `mp_ve`), `CountData` exporter (`experiments/save_counts_125m.py`), batched `lm-eval` + bucketed-accuracy wrapper (`experiments/eval_lm.py`), and launcher (`experiments/run_phase2_125m.sh`) are implemented and ready.  
**Cloud Budget:** Full matrix ~250–350 GPU·h on 4×H20/4090 (~3–4 days, `$400–900`); **Minimum ICML-ready core** (P0 + downstream) ~160–200 GPU·h. Downstream evaluation on existing Phase-1 checkpoints can also run at **\$0** on a local 8 GB GPU (`--batch_size 1`).

---

## 1. What Phase 1 Already Established (`125M × 2.5B`, 20 runs)

| Arm | Seeds | Total / Body / Table | Mean 2500M Val Loss | B16 Infer (tok/s) | Role in Paper |
|---|---:|---|---:|---:|---|
| `plain` | 3 | 162.2M / 162.2M / 0 | `3.3488 ± 0.0027` | 380K–383K | Base equal-depth/width |
| `word` | 3 | 239.3M / 162.2M / 77.2M | `3.2896 ± 0.0019` | 377K–379K | Full-`d` last-2 BoV baseline |
| `ctrl_word` | 1 | 194.7M / 181.8M / 12.9M | `3.2902` | 323K | Adapter-only control (`−0.0009` vs word) |
| `ctrl_hs` | 3 | 453.1M / 184.2M / 268.9M | `3.2753 ± 0.0010` | 315K–317K | Hashed pair + adapters |
| `ctrl_hs_pruned` | 3 | 453.1M / 184.2M / 268.9M | `3.2713 ± 0.0010` | 314K–319K | + maturity pruning (co-main baseline) |
| `u1` | 3 | 403.8M / 166.9M / 236.9M | `3.2628 ± 0.0006` | **351K–353K** | Low-latency adapter-free tiered |
| **`u123`** | **3** | **459.5M / 184.2M / 275.3M** | **`3.2415 ± 0.0028`** | **313K–315K** | **Main exact-tiered architecture** |
| `plain_eq_weights` | 1 | 459.7M / 459.7M / 0 | `3.1682` | 186K | Dense upper bound (wins at 2.5B) |

Clean additive decomposition at 2.5B (3-seed paired stats):
- `u123` beats `ctrl_hs_pruned` across **all 3 seeds, all 6 milestones (`50M–2500M`), and all 4 frequency buckets**: **`-0.02982 ± 0.00325` (`t = -15.885, p = 0.0039`)**.
- Why `125M` cleanly wins while `30M` tied (`+0.0118, t=+1.25`): `30M` hashed `mid`, whereas `train_125m.py` uses **collision-free exact key lookup** (`cd.hi_keys`, `cd.mid_keys`, `cd.g3_keys`) across all tiers even with `mid_rank = 64` (validating the Model-2 / `test2b` high-`U(T)/M` collision scaling prediction).
- Boundaries to keep strictly honest:
  1. `plain_eq_weights` (`3.1682`) beats `u123` by `~0.0759` (seed 0) at 2.5B; crossover sits at `250M–500M`. Claim **Equal Active FLOPs / Throughput Pareto Frontier** (~70–90% faster `B16` inference than `plain_eq_weights`), **not** Equal Total Weights.
  2. `u123` body is `184.2M` vs `plain` `162.2M`; hence **`plain184`** (`--d 832 --heads 13`) is P0 in Phase 2.

---

## 2. Newly Implemented Phase-2 Arms in `train_125m.py`

All three missing arms are now implemented directly in [train_125m.py](file:///C:/Users/ROG/Desktop/ICML/train_125m.py):

1. **`--arm engram` (`EngramModule`)**:
   - Ported faithfully from [diag_week1.py](file:///C:/Users/ROG/Desktop/ICML/experiments/diag_week1.py#L247-L296) / [run_local_suite.py](file:///C:/Users/ROG/Desktop/ICML/experiments/run_local_suite.py#L268-L338).
   - Multi-head hashed 2-gram (`ng2`) + 3-gram (`ng3`) tables with `K=4` hash heads per order (`ng_n = pair_rows // 2 = 500,000`, `ng_dim = d_mem // 4 = 64` -> **`256.0M` table params**, matched to `ctrl_hs` `256.0M` pair table and comparable to `u123` `262.4M` n-gram table), concatenated K/V linear projections (`512 -> d=768`), RMSNorm cosine context gate, and zero-initialized causal dilated depthwise convolution (`kernel_size=4, dilation=3, groups=d`, `SiLU(c) + v`), injected before block 0.
2. **`--arm xgram` (`MemoryBank(mode="xgram")`)**:
   - Frequency-split word value table computed from the exact same `count_tokens` prefix (`compute_xgram_hi_words` in `build_counts`, persisted via `xgram_hi_words` in `CountData` and checkpoint `xgram_is_hi` / `xgram_slot` buffers).
   - Top words covering `80%` of prefix tokens get full-width `d` (`768`) rows; the tail vocabulary gets `d // 4` (`192`) rows + a learned `Linear(d//4, d, bias=False)` projection (`ve_word_proj`, optimized by Muon). Tests adaptive per-word capacity allocation (Baevski & Auli-style) against `word` and `u123`.
3. **`--arm u123_no3g`**:
   - Identical to `u123` (`tiered` exact `hi` + `mid`, shared across all 12 layers, context gate + residual injection + pre-head readout) with `use_3g=False`. Directly isolates the causal marginal contribution of exact 3-grams (`u123` vs `u123_no3g` vs `u1`) at `125M × 2.5B`.
4. **`--arm uniform_mgram` (Value-Side Over-Tokenized Multi-Gram Control)**:
   - Orthogonal control for *Over-Tokenized Transformer* (Huang et al., ICML 2025). Instead of altering the input tokenizer/embedding side, it pools both **2-gram and 3-gram** hashes uniformly (`use_hash_3g=True`, no frequency tiering, no pruning, no exact keys) into the exact same `pair_rows` table (`268.9M` table params vs `u123` `275.3M`) and identical `184.2M` adapter body — directly answering whether frequency-partitioned exact allocation beats uniform multi-gram allocation at matched value-side memory budget. (Also supported in 30M `experiments/run_v50257_multiseed.py` as `uniform_mgram_d256_prefix10`.)

### Collision-Load Quantitative Law (`results/collision_law.json`, `figs_v5/fig9_collision_law.pdf`)

Replacing the dropped $c^\star(d)$ slope claim, controlled $M$-sweep data ($V=6{,}000$, $M\in\{6\text{k},24\text{k},96\text{k}\}$) establishes a clean two-parameter log-linear collision-load law:
$$\Delta(U(T), M) \approx a \ln(U(T)/M) + b$$
- **Vs `CTRL_hs`**: $a=0.00992, b=-0.01538, R^2=0.963$, boundary $M^\star(T)\approx 0.212\,U(T)$ (`30M` pred `+0.0236` vs obs `+0.0197`; `125M×2.5B` pred `+0.0321` vs obs `+0.0338`; 5-pt joint $R^2=0.957$).
- **Vs `CTRL_hs_pruned`**: $a=0.00707, b=-0.01261, R^2=0.982$, boundary $M^\star(T)\approx 0.168\,U(T)$ (`30M` pred `+0.0151` vs obs `+0.0118`; `125M×2.5B` pred `+0.0212` vs obs `+0.0298`; 5-pt joint $R^2=0.878$).
- **Preregistered `350M × 5B` prediction (`U(5B) ≈ 195.6M, M=2M`)**: predicts `+0.0301` nats over `CTRL_hs` and **`+0.0198`–`+0.0225` nats** (or `~+0.028` with exact `mid`) over `ctrl_hs_pruned`.

---

## 3. Recommended Downstream Suite (`125M × 2.5B` Optimal Selection)

At `125M–184M` body × `2.5B` FineWeb tokens, tasks whose random-guessing floor dominates (e.g., `mmlu` ~25%, `gsm8k` ~0%, `arc_challenge`) hide real architectural differences under floor noise. Use the following **6-task 0-shot suite** (all implemented in `experiments/eval_lm.py`, evaluated strictly inside `--max_length 512` to match trained RoPE range):

| Task | Metric | Random Floor | Expected 125M Range | Why Essential for This Paper |
|---|---|---:|---|---|
| **`lambada_openai`** | `acc`, `perplexity` | ~0% | `25%–45%` | Exact final-word completion; directly tests multi-token & phrase recall from N-gram tables. |
| **`wikitext`** | `word_perplexity` | — | — | Cross-domain generalization (FineWeb train -> Wikipedia eval); refutes "table only overfits FineWeb prefix". |
| **`piqa`** | `acc`, `acc_norm` | 50% | `60%–65%` | Stable physical commonsense; low seed variance at 125M. |
| **`arc_easy`** | `acc`, `acc_norm` | 25% | `40%–48%` | Clean scientific fact/concept recall; separates capacity-augmented models. |
| **`sciq`** | `acc`, `acc_norm` | 25% | `58%–68%` | High SNR (~1,000 questions, <2 min); tests domain terminology & lexical knowledge lookup. |
| **`hellaswag`** | `acc`, `acc_norm` | 25% | `28%–33%` | Standard pretraining benchmark expected by ICML reviewers (slower: ~40k requests). |

Plus **zero-extra-code FineWeb Diagnostic (`--verify-val-shard`)**:
- Computes **per-frequency-bucket Top-1 next-token accuracy** (`acc_pc_0`, `acc_pc_2`, `acc_pc_3`, `acc_pc_4`) alongside cross-entropy loss, directly verifying that tiered lookup converts lower cross-entropy into higher exact next-word accuracy across rare and frequent buckets without hurting `pc_0`.

---

## 4. ICML Submission Experiment Matrix & Priorities

| Priority | Experiment / Arm | Runs / Seeds | Execution | Est. GPU·h | What Reviewer Question It Closes |
|---|---|---|---|---:|---|
| **P0** | **`engram`** (`K=4`, 2/3g, gate, conv) | 3 × 2.5B | Cloud | 30–36 | Head-to-head vs real Engram architecture at matched table budget. |
| **P0** | **`plain184`** (`--d 832 --heads 13`, ~183.2M body) | 3 × 2.5B | Cloud | 36 | Exact equal-active-FLOPs dense control matching `u123`'s `184.2M` body. |
| **P0** | **`plain_eq_weights`** (seeds `1, 2`) | 2 × 2.5B | Cloud | 14 | 3-seed variance on the dense equal-weights upper bound (`3.1682`). |
| **P0** | **Downstream 6-task 0-shot** on core Phase-1 (`plain`, `word`, `u123`, `ctrl_hs_pruned`, `u1`) + Phase-2 (`engram`, `plain184`, `plain_eq_weights`) | All seeds | **Local 8GB (\$0)** or Cloud | 10–20 | Proves FineWeb loss gains transfer to real downstream tasks & out-of-domain PPL. |
| **P1** | **`xgram`** (80%-cover frequency-split word) | 3 × 2.5B | Cloud | 30 | Rules out "simple frequency-split unigram compression is enough". |
| **P1** | **`ctrl_word`** (seeds `1, 2`) | 2 × 2.5B | Cloud | 8 | 3-seed proof that adapter compute alone (`181.8M` body) adds `~0` at `2.5B`. |
| **P1** | **`mp_ve`** (last-2 shared tiered, no gate/3g) | 3 × 2.5B | Cloud | 30 | Connects `V=6,000` `MP-VE` prototype to `125M` `u1` / `u123` scaling path. |
| **P1** | **FineWeb Bucket Top-1 Accuracy** (`pc_0/2/3/4`) | 1 pass/ckpt | **Local (\$0)** or Cloud | <1 | Shows accuracy + loss decomposition across sighting tiers. |
| **P1** | **`uniform_mgram`** (`RUN_OT_CONTROL=1`, uniform unpruned 2g+3g hash) | 3 × 2.5B | Cloud | 15 | Value-side Over-Tokenized control: frequency partitioning > uniform pool. |
| **P2** | **`u123_no3g`** (3 seeds, `RUN_P2_ABLATIONS=1`) | 3 × 2.5B | Cloud | 15–18 | Isolates exact 3-gram gain (`u123` vs `u123_no3g` vs `u1`) at `125M`. |

*(Total if running everything including P2 on 4×H20/4090: ~200–250 GPU·h ≈ 2.5–3 days. Tight budget `P0`-only run: `RUN_P2_ABLATIONS=0` plus skipping `mp_ve` cuts ~60 GPU·h.)*

---

## 5. Phase-3 / Third-Scale Extension: `350M × 5B` (`experiments/run_350m_5b.sh`)

To turn the `30M -> 125M` contrast into a full **3-point scaling study** (`30M -> 125M × 2.5B -> 350M × 5B`) for ICML main:

- **Backbone (`350M` class):** `L=24, d=1024, H=16` (`head_dim=64`, ~`404.8M` untied `plain`).
- **Budget & Prefix (`5.0B` tokens = 51 shards, ~10.2 GB):** `--count-tokens 500000000` (`500M` 10% prefix, proportional scaling), `2×` table capacity (`--pair-rows 2000000 --n-hi 1000000 --n-mid 3000000 --n-3g 300000 --d-mem 256`, saving `results/countdata_350m.npz`).
- **Micro-batch & Hardware:** `--micro-batch 8 --grad-accum 32` (`131,072` tokens/step, matching 125M batch dynamics while halving activation memory). Requires **48 GB+ GPUs** (`4090-48G / A6000 / A100-80G / H20-96G`) and **≥32 GB system RAM** for `500M`-token count construction (`np.unique`).
- **Core Arms (`2 seeds` default, ~120–160 GPU·h; set `RUN_SEEDS2=1` for 3 seeds):**
  1. `u123` (`d=1024, L=24` -> `475.34M` body, `537.68M` table, `1013.02M` total);
  2. `ctrl_hs_pruned` (`d=1024, L=24` -> `475.34M` body, `524.87M` table, `1000.21M` total);
  3. `engram` (`d=1024, L=24` -> `405.83M` body, `512.00M` table, `917.83M` total);
  4. `plain_eqFLOP` (`--d 1152 --heads 18 --layers 24` -> `498.06M` body, strictly conservative equal/above-active-FLOPs dense comparator);
  5. `plain` (`d=1024, L=24`, `404.77M` body) and `word` (`d=1024, L=24`, `404.77M` body, `102.93M` table, `507.70M` total).
- **Run command:**
  ```bash
  bash experiments/run_350m_5b.sh
  ```

---

## 6. Paper Completeness Checklist for ICML Submission

To lock in a clean main-conference ICML submission, make sure the final manuscript (`ICML_Research_Proposal_v5.tex` -> camera-ready draft) contains these 6 evidence blocks:

1. **Empirical Scaling Law (`V=1,500–6,000` + `<UNK>` disentanglement)**: Word vs Pair crossover (`<100` vs `100–999` vs `1000+` non-`<UNK>`) + Round-30 held-out 5/5 prediction (`figs_v5/fig1_crossover`, `fig5_heldout`). *(Already done)*
2. **Three Phenomenological Theory Tests (`results/theory_cstar_analysis.json`)**: KKT 3-tier capacity allocation, collision-load `M` sweep (`test2` & `test2b_pruned_M_sweep`), and inter-layer gradient cosine $\rho_{\ell,\ell'}$ depth sweep (`test3`). *(Already done)*
3. **`V=50,257` Bridge & Collision Transition Diagnosis (`30M` -> `2.5B`)**: Reconciles why hashed-mid `U123_v2_fullmid` tied `CTRL_hs_pruned` at `30M` (`t=+1.25`), while exact-key tiered `u123` wins decisively at `2.5B` (`-0.02982, t=-15.885, p=0.0039`). *(Already done)*
4. **Full `125M × 2.5B` Main Comparison + Causal Ablations (Phase 1 + Phase 2)**: `plain` vs `plain184` vs `word` vs `xgram` vs `ctrl_word` vs `ctrl_hs` vs `ctrl_hs_pruned` vs `engram` vs `u1` vs `u123_no3g` vs `u123` vs `plain_eq_weights` (3 seeds on all main arms, paired t-tests, additive decomposition). *(Phase 1 done; Phase 2 automated in `run_phase2_125m.sh`)*
5. **Compute-Speed Pareto Frontier**: Val loss vs `B1` ms/seq and `B16` tok/s (`u1` +90% / `u123` +70% inference speed over `plain_eq_weights`, early-training crossover `<500M` tokens). *(Phase 1 done; `plain184` & `engram` added by Phase 2)*
6. **Zero-Shot Downstream & Bucketed Accuracy Table**: 6-task suite (`lambada_openai`, `wikitext`, `piqa`, `arc_easy`, `sciq`, `hellaswag`) + FineWeb per-bucket accuracy (`eval_lm.py`). *(Ready to run locally on backed-up Phase-1 checkpoints or on cloud)*

---

## 7. Execution Runbook (Copy-Paste Ready)

### Option A: Zero-Cost Local Downstream Eval (RTX 4070 8GB, using saved Phase-1 checkpoints)
1. Ensure `runs_125m/<arm>_s<seed>/ckpt_2500M.pt`, `data/` (first 3 FineWeb train shards + val shard), and `pip install lm-eval transformers` are present locally.
2. Export exact prefix counts:
   ```bash
   python experiments/save_counts_125m.py --data-dir ./data --count-tokens 250000000 --train-tokens 2500000000 --out results/countdata_125m.npz
   ```
3. Verify exact Phase-1 parity (`u123_s0` must match `val ≈ 3.24411`):
   ```bash
   python experiments/eval_lm.py --checkpoint runs_125m/u123_s0/ckpt_2500M.pt --countdata results/countdata_125m.npz --verify-val-shard data/fineweb_val_000000.bin --device cuda
   ```
4. Smoke-test `lm_eval` before full run, then evaluate all backed-up checkpoints across available GPUs (`--dtype auto` uses `bfloat16` on CUDA so even `350M` fits in `<3 GB` VRAM on 8 GB cards):
   ```bash
   python experiments/eval_lm.py --checkpoint runs_125m/u123_s0/ckpt_2500M.pt --countdata results/countdata_125m.npz --tasks lambada_openai --limit 10 --batch_size 1
   bash experiments/run_downstream_eval.sh          # auto-detects 1 GPU locally or 4 GPUs on cloud
   # Faster pass without HellaSwag: TASKS=lambada_openai,wikitext,piqa,arc_easy,sciq bash experiments/run_downstream_eval.sh
   ```

### Option B: Full Cloud Phase-2 Suite (`4 × GPU`, `--micro-batch 16 --grad-accum 16`)
```bash
# Runs download -> save_counts -> 12-arm smoke + 25-step probes -> Phase-1 CountData gate -> Waves 1-4 -> 6-task lm-eval + bucket accuracy
bash experiments/run_phase2_125m.sh
# Or skip already-downloaded shards / skip P2 u123_no3g:
SKIP_DOWNLOAD=1 RUN_P2_ABLATIONS=0 bash experiments/run_phase2_125m.sh
```

---

## 8. Acceptance Gates

1. **Smoke + 25-step probe**: all 12 arms pass `--smoke`, 25-step probes (`engram`, `xgram`, `plain184`, `u123_no3g`) stay finite with `peak_vram_gb < 0.8 × VRAM`.
2. **CountData alignment gate**: `eval_lm.py --verify-val-shard` on Phase-1 `u123_s0` reproduces `val ≈ 3.24411` within float rounding. If it drifts, stop immediately — wrong shard order or `--count-tokens`.
3. **Equal-FLOPs gate**: `plain184` (`--d 832 --heads 13`) `body_params` within `±2%` of `u123`'s `184.2M`.
4. **Engram budget check**: verify `table_params` of `engram` (`~256.0M`) is logged accurately and `engram` runs with `K=4`, K/V proj, context gate, causal dilated conv (`ctrl_hs` stays separate).
5. **Downstream honesty gate**: if `u123` beats `ctrl_hs_pruned`/`plain` on `lambada_openai`/`wikitext`/`sciq`/`piqa`/`arc_easy`, report exact per-task 3-seed paired stats; any task near random floor (e.g. `hellaswag` if flat) must be reported transparently without cherry-picking.
6. **Artifacts to collect back**: `results/countdata_125m.npz`, `runs_125m_phase2/*/metrics_final.json` & `ckpt_2500M.pt`, `results/lm_eval/*.json`, `results/lm_eval/bucket_acc/*.json`, plus a `PHASE2_RESULTS.md` summary table.
