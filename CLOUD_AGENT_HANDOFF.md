# Cloud GPU Server Agent Handoff (`125M × 2.5B FineWeb` Experiments)

> **Purpose**: This document tells the autonomous agent on the cloud GPU server **exactly what files we already verified locally, what to run in what order, what safety gates to check before burning multi-seed GPU hours, and what numbers/artifacts to send back** for our paper (Value-Memory Phase Diagram + Shared Multi-Layer Maturity-Tiered Memory `U123`).

---

## 1. Verified Code & Files Already in Workspace

All files below live in the project root (synced from `C:\Users\ROG\Desktop\ICML\`):

1. **`download_data.py`** — Multi-threaded concurrent downloader for `kjj0/fineweb10B-gpt2` (GPT-2 pre-tokenized `uint16` shards, `1024`-byte header, `100M` tokens / `~200 MB` per shard). Supports `--workers`, `--shards`, `--verify`, `.part` atomic writes, and `HF_ENDPOINT` fallback.
2. **`train_125m.py`** — Fully audited, single-file `bf16` trainer (969 lines). Already **fixed & verified locally** against 6 bugs:
   - Strict causal indexing verified (`--smoke` passes zero-target-leakage invariance test across ALL 10 arms: `plain`, `plain_eq_weights`, `word`, `hard_stack`, `mp_ve`, `u1`, `u123`, `ctrl_word`, `ctrl_hs`, `ctrl_hs_pruned`).
   - Full-width `d=768` for `word`/`hard_stack` baselines + automatic projection fallback if `d_mem != d`.
   - Exact actual table sizes (`len(cd.hi_keys/mid_keys/g3_keys)`) instead of allocating dead zero rows.
   - Zero unnecessary `all_keys` CPU binary search during training (`hash` computed on GPU; `tiered` only searches `hi_keys`/`mid_keys`; `ctrl_hs_pruned` searches compact `mature_keys`).
   - Automatic proportional prefix-threshold scaling (`--auto-scale-thresh 1`: e.g. `10%` prefix `250M` tokens scales `100 / 10 / 30` $\to$ `10 / 1 / 3`).
   - `math.ceil` step count + forced final milestone evaluation so `2.5B` never gets skipped; chunked cross-entropy (`chunk=2048`) to keep logits VRAM low; post-training `bench_latency()` (`b1_seq_ms`, `b16_infer_tok_s`) written into `metrics_final.json`.
3. **`run_125m_experiments.sh`** — End-to-end 4-wave orchestration script for a 4-GPU node (or adaptable sequentially on 1 GPU).

---

## 2. Step-by-Step Execution Protocol on the Cloud Server

### Step 0 — Environment & Hardware Check
Run first and record the exact specs:
```bash
nvidia-smi
lscpu | head -n 20
df -h .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.device_count())"
```
Ensure:
- At least **30 GB free disk** (5.2 GB shards + ~15–20 GB checkpoints if keeping `ckpt_*.pt`).
- If VRAM per GPU is **40 GB (A100-40G)** instead of **80/96 GB (A100-80G / H20 / H100)**, pass `--micro-batch 16 --grad-accum 16` (keeps exact global batch `131,072 tok/step`) so Dense Adam states on `~200M` table params never OOM.

### Step 1 — Download `2.5B` FineWeb Shards Concurrently (`~1–2 min`)
```bash
python download_data.py --shards 25 --workers 12 --data-dir ./data --counts-shards 2 --verify
```
*(If `huggingface.co` is slow/blocked in your region, prepend `HF_ENDPOINT=https://hf-mirror.com`.)*
Confirm final line prints: `[verify] 26/26 shards valid | Train tokens ready: 2.500B`.

### Step 2 — Mandatory CUDA Preflight Probe BEFORE Launching 4-GPU Waves (`~2 min`)
Do NOT skip this. Run a 25-step real-dimension `125M` probe (`d=768, L=12, V=50,257`) on GPU 0 to check **loss stability (no NaN/divergence from `lr_word=0.03 / lr_pair=0.02`), peak VRAM, and actual `tok/s`**:
```bash
# 1. Quick 10-arm smoke test on GPU
for arm in plain plain_eq_weights word hard_stack mp_ve u1 u123 ctrl_word ctrl_hs ctrl_hs_pruned; do
    python train_125m.py --smoke --arm "${arm}" --device cuda --out "/tmp/smoke_${arm}"
done

# 2. Real 125M 25-step probe (adjust --micro-batch if OOM)
python train_125m.py --arm u123 --data-dir ./data --count-tokens 50000000 --steps 25 \
    --eval-tokens 3000000 --log-interval 5 --compile 1 --out /tmp/probe_125m_u123
python train_125m.py --arm plain_eq_weights --eq-width 1152 --eq-heads 16 --data-dir ./data \
    --count-tokens 10000000 --steps 25 --eval-tokens 3000000 --log-interval 5 --compile 1 --out /tmp/probe_eqW
```
**Check the probe output before proceeding:**
1. Inspect `loss` over steps `1..25`: verify smooth monotonic decrease (if `lr_word=0.03 / lr_pair=0.02` causes any instability/spike at `d=768`, test `0.01 / 0.01` vs `0.03 / 0.02` in a 100-step check first!).
2. Check `table_params` & `peak_vram_gb`: for `u123` (`count_tokens=250M`, `d_mem=256`), verify `peak_vram_gb < 0.80 * GPU_VRAM`.
3. Check `plain_eq_weights` (`eq-width=1152`, `L=12`): verify `total` params roughly matches `u123` total params (adjust `--eq-width` to `1024` or `1280` if needed so `body_params(plain_eq_weights) ≈ total_params(u123)` within 15%).
4. Estimate total wall-clock time from `tok/s`: `total_hours = (2.5e9 / tok_s) / 3600`.

### Step 3 — Run Wave 1 (Seed 0 Core 4 Arms) & Inspect Gate A
Launch Wave 1 (in `run_125m_experiments.sh` or directly):
- `plain` (`seed=0`, Equal Active FLOPs baseline)
- `word` (`seed=0`, Bank-of-Values full-`d` last-2 baseline)
- `u123` (`seed=0`, Main method: shared bank + ctx-gated V + Res + Prehead + 1+2+3g maturity tiers)
- `ctrl_hs` (`seed=0`, Same dense adapters/gates/prehead, unpruned hashed pairs)

**Gate A (inspect `log.jsonl` & `metrics_final.json` after Wave 1):**
1. Confirm `2500M` checkpoint & final eval actually ran (`tokens_seen >= 2.5e9`).
2. Compare `val` across `50M -> 100M -> 250M -> 500M -> 1B -> 2.5B`:
   - `u123` vs `plain` (Equal Active FLOPs gap)
   - `u123` vs `word`
   - `u123` vs `ctrl_hs` trajectory (does `ctrl_hs` lead early and `u123` cross/maintain advantage by `1B–2.5B`, especially on `pc_0` unvisited and `pc_3`/`pc_4` mature collision protection?)
3. Check `latency.b1_seq_ms_mean`, `b16_infer_tok_s`, and `train_tok_s`.

### Step 4 — Run Wave 2 (Seed 0 Disentanglement & Bounds)
Run:
- `plain_eq_weights` (Equal Total Weights baseline)
- `ctrl_hs_pruned` (Hashed + `<10` pruning -> isolates pruning vs exact de-collision in `u123 - ctrl_hs`)
- `ctrl_word` (U123 dense adapters with 1-gram word bank only -> isolates dense adapter FLOPs)
- `u1` (Shared tiered bank across all 12 layers without ctx-gates/prehead -> low-latency deployment arm)

**Note on `ctrl_word` & `u1` parameters**: Log exact `params`, `body_params`, `table_params`, `train_tok_s`, `b1_seq_ms_mean`, `b16_infer_tok_s`, `val`, and `pc_0..pc_4`.

### Step 5 — Multi-Seed Confirmation (Seeds 1 & 2)
Run **Seeds `1` and `2`** for the headline arms:
`plain`, `word`, `u123`, `ctrl_hs`, `ctrl_hs_pruned`, `u1` (and `ctrl_word` / `plain_eq_weights` if GPU budget permits — at minimum 3 seeds on `{plain, word, u123, ctrl_hs, ctrl_hs_pruned, u1}`).
Compute **3-seed paired differences (`mean ± std, paired t`)** at `2.5B` (and intermediate milestones `50M..2.5B`).

### Step 6 — Downstream Benchmark Evaluation (Important for Main-Conference Venues)
If time/GPU budget allows, wrap `Model` from `train_125m.py` into a lightweight `lm_eval.api.model.LM` adapter (using `transformers.AutoTokenizer.from_pretrained("gpt2")` and `make_train_ids` causal lookup with saved `CountData` — remember: NEVER leak test target tokens!) and evaluate `ckpt_final.pt` across all arms on zero-shot tasks:
- `lambada_openai` (PPL & acc — strong test of phrase/next-word lookup)
- `hellaswag`, `piqa`, `winogrande`, `arc_easy` (general reasoning/commonsense, `acc_norm`)
- `wikitext` (word/byte perplexity)

*(Tip: Cache/save the built `CountData` arrays `hi_keys`, `mid_keys`, `g3_keys`, `mature_keys` or rebuild deterministically from the first `args.count_tokens` of `./data/fineweb_train_*.bin` so eval uses the EXACT same lookup IDs as training!)*

---

## 3. Strict Scientific & Reporting Rules (Do Not Violate!)

1. **Zero Hype / Full Decomposition**:
   - NEVER attribute the entire `u123 vs plain` or `u123 vs word` gap to maturity routing alone. Always split into:
     - **Architecture contribution**: `ctrl_hs - word` (and `ctrl_word - word` for dense adapter FLOPs);
     - **Maturity pruning contribution**: `ctrl_hs_pruned - ctrl_hs` (especially `pc_0`);
     - **Exact 2g/3g de-collision contribution**: `u123 - ctrl_hs_pruned` (especially `pc_3`/`pc_4` = `100+` sightings).
2. **Equal Active FLOPs vs Equal Total Weights**:
   - Report `plain_eq_weights` honestly even if it beats `u123` on `val`. Explicitly separate **Equal Active FLOPs / Inference Latency** from **Equal Total Weights**.
3. **Training Throughput vs Inference Latency**:
   - Report `train_tok_s`, `b1_seq_ms_mean ± std`, and `b16_infer_tok_s` side-by-side (`u1` vs `u123` vs `plain` vs `plain_eq_weights`).
4. **Preserve Raw Logs**:
   - Never delete or overwrite `runs_125m/*/log.jsonl` or `metrics_final.json`. Bundle all JSON/logs + a clean `RESULTS_125M.md` summary when handing results back.

---

## 4. Reference Baselines Already Measured Locally (`V=50,257, 30M tokens, L=6, d=256`, Seed 0)

So you have an immediate sanity reference while watching Wave 1 curves:
- `plain_d256_eqFLOP`: `val = 4.9855` (`b1 = 15.68 ms`, `train = 40.6k tok/s`)
- `word_d256_last2`: `val = 4.8961` (`b1 = 12.74 ms`, `train = 38.1k tok/s`)
- `U1_shared_d512_all6`: `val = 4.8658` (`b1 = 15.29 ms`, `b16_infer = 276.9k tok/s`, `train = 31.2k tok/s`)
- `plain_d384_eqWeights`: `val = 4.8570` (`b1 = 13.51 ms`, `b16_infer = 250.2k tok/s`, `train = 34.9k tok/s`)
- `CTRL_hs_d256_prefix10`: `val = 4.8324` (`pc_0 = 4.9028`, `pc_100 = 4.6003`, `b1 = 18.32 ms`)
- `CTRL_hs_pruned_d256_prefix10`: `val = 4.8263` (`pc_0 = 4.8597`, `pc_100 = 4.6064`, `b1 = 18.30 ms`)
- `U123_d256_prefix10`: `val = 4.8234` (`pc_0 = 4.8537`, `pc_100 = 4.5833`, `b1 = 18.93 ms`)
