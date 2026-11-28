# Cloud Agent Final Handoff: Closing Suite (`125M` Uniform Control + `--zero-memory` Diag + `350M × 5B` OOS)

**Date:** 2026-11-28  
**Repo:** `https://github.com/Rinck1/value-memory-crossover.git` (branch `main`)  
**One-Shot Launcher:** [experiments/run_cloud_final_all.sh](experiments/run_cloud_final_all.sh) (fully idempotent — skips any existing checkpoint, `CountData`, or eval JSON so it can be safely re-run after interruption).

---

## 1. Current Completed State (`125M × 2.5B` Done — 39/39 Runs)

Archived in [docs/PHASE2_RESULTS_125M.md](docs/PHASE2_RESULTS_125M.md): all **13 arms × 3 seeds = 39 runs** (`plain`, `plain184`, `word`, `xgram`, `ctrl_word`, `ctrl_hs`, `ctrl_hs_pruned`, `engram`, `mp_ve`, `u1`, `u123_no3g`, `u123`, `plain_eq_weights`) plus **6-task zero-shot `lm-eval`** and **FineWeb bucket top-1 accuracy** are complete.

Key already-established numbers (do **not** re-run any of these 39 runs):
- `u123_no3g − ctrl_hs_pruned` (pure 2-gram de-collision, **32M fewer** table params): **`-0.02601 ± 0.00062` (`t = -73.22, p = 0.00019`)**, WikiText **`-1.80` PPL (`p=0.0065`)**, LAMBADA **`-4.83` PPL (`p=0.037`)**, ARC-Easy **`+1.24%` (`p=0.0082`)**, HellaSwag `acc_norm` **`+0.82%` (`p=0.0086`)**.
- `u123 − engram`: **`-0.06487 ± 0.00178` (`p=0.00025`)**; beats `engram` significantly ($p < 0.05$) on **all 6 zero-shot tasks**.
- `u123 − plain184`: **`-0.08590 ± 0.00207` (`p=0.00019`)**; LAMBADA **`+5.71%` (`p=0.0105`)**, WikiText **`-6.13` PPL (`p=0.00040`)**, SciQ **`+5.17%` (`p=0.045`)**, HellaSwag `acc_norm` **`+1.98%` (`p=4.5e-5`)**.
- `ctrl_word − word`: `+0.00123 ± 0.00189` (`p=0.375`, zero gain from adapters alone).
- `plain_eq_weights` 3-seed crossover: **`[250M, 500M]`** (`451.8M` tokens); at `2.5B`, `u123` runs **`1.70×`** faster (`B16`) and matches `plain_eq_weights` on `LAMBADA acc` (`0.3326 vs 0.3262`) and `SciQ` (`0.7413 vs 0.7370`).

---

## 2. Strictly Frozen Artifacts — DO NOT MODIFY

1. **Commit `b89068c`** (`results/collision_law.json`, `docs/ICML_Research_Proposal_v5.tex`, `experiments/run_350m_5b.sh`, `train_125m.py`): frozen base preregistration & model code.
2. **Commit `43d54ad`** ([docs/PREREG_350M_SUBZIPF.md](docs/PREREG_350M_SUBZIPF.md), [results/subzipf_350m_predictions.json](results/subzipf_350m_predictions.json)): pre-unblinding `350M × 5B` Sub-Zipf predictions & `95%` prediction intervals (`+0.02689`, `95% PI [+0.01689, +0.03689]`).

Do **not** edit hyperparameters (`--layers 24 --d 1024 --heads 16 --d-mem 256 --pair-rows 2000000 --n-hi 1000000 --n-mid 3000000 --n-3g 300000 --train-tokens 5000000000 --count-tokens 500000000`) before unblinding.

---

## 3. What Remains to Run (All Automated in One Command)

Hardware requirement: **4× 48 GB+ GPUs** (`4090-48G / A6000 / A100-80G / H20-96G`), **≥32 GB RAM** (for `500M`-token `np.unique` CountData), **≥80 GB free disk** (51 shards ~10.2 GB + 16 × `350M` checkpoints).

```bash
git pull origin main
bash experiments/run_cloud_final_all.sh
```

What this single script runs in order:

| Step | Task | Est. Time (4× GPU) | Purpose |
|---|---|---:|---|
| **Step 1** | `125M` `--zero-memory {none,ngram,all}` diagnostic (`u123`, `ctrl_hs_pruned`, `plain184`, `plain`, `word` × 3 seeds) | **~2 min** | Zero-training causal diagnostic for `pc_0 = -0.01765` + per-bucket gate openings (`gate_v`, `gate_res`, `head_gate`). |
| **Step 2** | `125M` `uniform_mgram` × 3 seeds (`s0, s1, s2`) + 6-task `lm-eval` + bucket acc + `--zero-memory` | **~4 h** (1 wave, GPUs 0–2) | Closes the Over-Tokenized value-side control (uniform unpruned 2g+3g hash inside the exact `u123` wrapper). |
| **Step 3** | **`350M × 5B` Scaling Suite** (Waves 1–3: 6 arms × 2 seeds; Wave 4 `RUN_SEEDS2=1`: headline 4 arms seed 2) + 6-task `lm-eval` + `350M` `--zero-memory` | **~40–48 h** (`RUN_SEEDS2=1`) or **~28–36 h** (`RUN_SEEDS2=0`) | True OOS validation of `43d54ad` (`+0.02689 [0.01689, 0.03689]`) + 3-seed paired $t$-tests on `u123 vs ctrl_hs_pruned / engram / plain_eqFLOP` + 350M zero-shot benchmarks. |

### Useful Environment Overrides

- **Custom checkpoint directories** (if Phase-1 checkpoints are stored outside `./runs_125m`):
  ```bash
  PHASE1_DIR=/path/to/runs_125m OUT_DIR_125M=/path/to/runs_125m_phase2 bash experiments/run_cloud_final_all.sh
  ```
- **2-seed budget mode for 350M** (~12h faster, skips Wave 4 seed 2):
  ```bash
  RUN_SEEDS2=0 bash experiments/run_cloud_final_all.sh
  ```
- **Shards already downloaded**:
  ```bash
  SKIP_DOWNLOAD=1 bash experiments/run_cloud_final_all.sh
  ```
- **Jump straight to 350M** (if Step 1/2 already completed):
  ```bash
  SKIP_125M_ZERO_MEM=1 SKIP_125M_UNIFORM=1 bash experiments/run_cloud_final_all.sh
  ```

---

## 4. Acceptance & Verification Gates

1. **Step 1 (`125M --zero-memory`):** Check printed table for `u123` (`zm=none` must reproduce `val ≈ 3.2415`), `u123[zm=all]` vs `plain184[zm=none]` on `pc_0`, and `gate_v(pc0/all)`. Note: interpret strictly as a **diagnostic** (*consistent with* / *not consistent with* capacity offloading), never standalone proof.
2. **Step 2 (`uniform_mgram`):** Verify `body_params ≈ 184.19M` and `table_params ≈ 268.87M` (matching `ctrl_hs` and `u123`'s wrapper), and compare `u123` & `u123_no3g` vs `uniform_mgram` across all 3 seeds on FineWeb val + 6 zero-shot tasks.
3. **Step 3 (`350M × 5B`):**
   - 25-step probes (`u123`, `plain_eqFLOP`) must stay finite with `peak_vram_gb < 0.8 × GPU VRAM`.
   - Confirm `plain_eqFLOP` (`d=1152, H=18, L=24`) has `body_params ≈ 498M` ($\ge$ `u123`'s `475M` body).
   - Check the final auto-printed **Preregistered `43d54ad` Check**:
     - Is observed `ctrl_hs_pruned − u123` strictly positive?
     - Does it fall inside **`[+0.01689, +0.03689]`** (point estimate `+0.02689` vs frozen `b89068c` `+0.02250`)?
     - Does it confirm the non-trivial collision-load direction (gap stays flat or slightly below `125M`'s `+0.0298` because $M$ doubles while $U(T)$ grows $1.64\times$)?

---

## 5. Exact Artifacts to Return / Export After Completion

Package or summarize the following files into a single `FINAL_RESULTS.md` (mirroring `PHASE2_RESULTS.md` format) plus raw JSONs:

1. `results/zero_memory_diag_125m/*.json`
2. `runs_125m_phase2/uniform_mgram_s{0,1,2}/metrics_final.json`
3. `results/lm_eval/uniform_mgram_s{0,1,2}.json` and `results/lm_eval/bucket_acc/uniform_mgram_s{0,1,2}.json`
4. `runs_350m/*/metrics_final.json` (all 12 or 16 runs, including `val`, `pc_0/2/3/4`, `train_tok_s`, `b1_ms`, `b16_tok_s`, `peak_vram_gb`)
5. `results/lm_eval_350m/*.json` and `results/lm_eval_350m/bucket_acc/*.json`
6. `results/zero_memory_diag_350m/*.json`
7. Exact stdout of `results/countdata_350m.npz` creation (`unique`, `hi`, `mid`, `mature`, `3g` counts on the `500M` prefix — this also gives the **directly measured** $U(500\text{M})$ to compare against our Heaps projection `37,821,433`!).
