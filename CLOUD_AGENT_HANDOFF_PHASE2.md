# Phase-2 handoff: 125M × 2.5B downstream + missing baselines (medium budget)

**Date:** 2026-11-26  
**Scope:** close the three holes left by phase 1: (1) no downstream evaluation,
(2) no Engram / X-GRAM / equal-active-FLOPs baselines, (3) one-seed
`plain_eq_weights` and `ctrl_word`.
**Budget:** ~250–350 GPU-hours on 4×H20/4090 (3–4 days); ~$400–900.

---

## 1. What phase 1 already established

125M × 2.5B, 3 seeds, 2500M val (lower is better):

| Arm | mean 2500M val | note |
|---|---:|---|
| `plain` (162M body) | 3.3488 | base width/depth |
| `word` (full-d last-2) | 3.2896 | BoV-style |
| `ctrl_hs` (hashed pair + adapters) | 3.2753 | architecture control |
| `ctrl_hs_pruned` | 3.2713 | + pruning |
| **`u123`** | **3.2415** | main |
| `u1` | 3.2628 | low-latency |
| `plain_eq_weights` (459.7M body) | 3.1682 | 1 seed |
| `ctrl_word` (181.8M body, word only) | 3.2902 | 1 seed |

Clean additive decomposition at 2.5B:

```
word − plain              = 0.0592
ctrl_hs − word            = 0.0143   (shared multi-layer memory)
ctrl_hs_pruned − ctrl_hs  = 0.0040   (pruning)
u123 − ctrl_hs_pruned     = 0.0298   (exact 2g/3g tiering; t=15.9, p=0.0039)
u123 − plain              = 0.1073   (exact sum)
```

Boundaries already known:
- `plain_eq_weights` beats `u123` by 0.0733 at 2.5B; crossover ~250–500M.
  **Claim only equal active FLOPs / compute-limited inference.**
- `u123` body is 184.2M vs `plain` 162.2M (+13.6% adapters). `ctrl_word`
  (181.8M body, word-only) is 3.2902 vs `word` 3.2896: adapters alone do not
  help. A **plain at ~184M body is still missing** for a clean equal-FLOPs
  baseline.
- The 30M `fullmid` experiment did **not** significantly beat `ctrl_hs_pruned`
  (3 seeds, +0.0118, t=1.25). The 125M `u123` uses the **old low-rank mid** and
  wins. **Do not port fullmid blindly; it is optional, not required.**

---

## 2. Experiments to run

| Priority | Arm / run | Seeds | Tokens | Est. GPU·h | Status |
|---|---|---|---:|---:|---:|
| P0 | `engram` (full design: 2/3-gram, K hash heads, key/value maps, context gate, conv) | 3 | 2.5B | 30–36 | **to implement/port** |
| P0 | `plain_eqFLOPs` (`d=832, H=13`, body ≈183M) | 3 | 2.5B | 36 | no code change; verify body |
| P0 | `plain_eq_weights` seeds 1,2 | 2 | 2.5B | 40–48 | command ready |
| P0 | `ctrl_word` seeds 1,2 | 2 | 2.5B | 20 | command ready |
| P0 | `lm-eval` on 2500M checkpoints: `plain`, `word`, `u123`, `ctrl_hs_pruned` | 3 each | — | 15–25 | **wrapper to implement** |
| P1 | `xgram` (frequency-split word table: full rows for top-cover words, short rows for the rest) | 3 | 2.5B | 30 | **to implement** |
| P1 | `mp_ve` (original tiered, shared last-2, no gates/prehead/3g) | 3 | 2.5B | 30 | arm exists |
| P1 | `lm-eval` on `u1`, `ctrl_hs`, `plain_eq_weights`, `ctrl_word` | 3 each | — | 20–30 | wrapper |
| P2 | `u123_no3g` (isolate 3-gram contribution) | 2 | 2.5B | 20 | small code change |
| P2 | bucket-wise next-token accuracy (FineWeb val) | — | — | <2 | cheap; do with lm-eval |
| P2 | MWE / idiom completion (custom) | — | — | 5–10 | optional |

Total: **~250–350 GPU·h** including reruns. On 4×H20 ≈ 65–90 h wall clock
(3–4 days). On 4×4090 (48GB) ≈ 80–110 h (3.5–4.5 days).

If the budget is tighter, the **minimum** is:
`engram` + `plain_eqFLOPs` + `plain_eq_weights` seeds 1/2 + `ctrl_word` seeds
1/2 + lm-eval on 4 arms. That is ~180–220 GPU·h.

---

## 3. Implementation tasks before any GPU run

### 3.1 Save the exact `CountData` (blocking for lm-eval)

The memory tables are indexed by IDs derived from the **first 250M training
tokens**. A checkpoint cannot be evaluated with a different count prefix.

Implemented: `experiments/save_counts_125m.py`. It calls the same
`build_counts` and writes:

```python
np.savez_compressed(
    "results/countdata_125m.npz",
    hi_keys=cd.hi_keys, mid_keys=cd.mid_keys, g3_keys=cd.g3_keys,
    mature_keys=cd.mature_keys, all_keys=cd.all_keys, all_counts=cd.all_counts,
    eff_hi=cd.eff_hi, eff_mid=cd.eff_mid, eff_g3=cd.eff_g3,
    count_tokens=args.count_tokens,
)
```

Run it on the same 26 shards before lm-eval. **Every eval must load this file.**

### 3.2 Port Engram into `train_125m.py`

The existing `ctr_hs` is **not** Engram. Port the full design from
`experiments/run_v50257_bridge.py` / `run_local_suite.py`:
- 2-gram and 3-gram hashed tables, K hash heads per order;
- key/value projections, context gate (`sigmoid(rmsnorm(h)·rmsnorm(k))`);
- causal depthwise conv;
- same body width/depth as `u123`; table budget reported.
Run `--smoke` + a 25-step real-dimension probe before the full run.

### 3.3 Add X-GRAM-style `xgram` arm

Frequency-split word table: full `d_mem` rows for the top words covering 80% of
text, short (`d_mem/4`) rows plus a shared projection for the rest. This is the
closest published prior; it is not `word` (all full) and not `ctrl_word`
(adapters only).

### 3.4 Equal-active-FLOPs plain

Use `--arm plain --d 832 --heads 13`. Expected body ≈183M (vs `u123` 184.2M).
Verify with the `[model] body=` line; if off by >5%, adjust `d` to 824/840.

### 3.5 lm-eval wrapper

Draft implemented: `experiments/eval_lm.py` (`lm_eval.api.model.LM`). The
`--dry-run` path was tested locally with a synthetic checkpoint (logits finite);
the `loglikelihood` path still needs one cloud smoke test.
- load `ckpt_2500M.pt`, rebuild `Model` from the checkpoint config + actual
  table shapes;
- load `results/countdata_125m.npz`; compute lookup IDs **only from the input
  prefix** (never from target tokens);
- GPT-2 tokenizer (`transformers.AutoTokenizer.from_pretrained("gpt2")`);
- implement `loglikelihood`, `loglikelihood_rolling`, `generate_until`;
- `--batch_size 1` on 8GB cards, 4–8 on 96GB cards.

Launcher: `experiments/run_phase2_125m.sh`. Engram/X-GRAM lines are commented
out until those arms are implemented in `train_125m.py`.

Run:
```bash
lm_eval --model local-completions --model_args ... \
  --tasks lambada_openai,wikitext,piqa,arc_easy,hellaswag \
  --num_fewshot 0 --batch_size 4
```
(Use the harness's Python API or register a custom model; the wrapper is the
only non-standard part.)

---

## 4. Run order and commands

### Wave 0 — probes (1 GPU, ~2 h)

```bash
# 10-arm smoke + zero-leakage invariance
for arm in plain word mp_ve u123 ctrl_hs ctrl_hs_pruned ctrl_word u1; do
  python train_125m.py --smoke --arm "$arm" --device cuda --out "/tmp/smoke_$arm"
done

# 25-step real-dimension probe for any new arm
python train_125m.py --arm engram --data-dir ./data --count-tokens 50000000 \
  --steps 25 --eval-tokens 3000000 --log-interval 5 --compile 1 --out /tmp/probe_engram
python train_125m.py --arm xgram  --data-dir ./data --count-tokens 50000000 \
  --steps 25 --eval-tokens 3000000 --log-interval 5 --compile 1 --out /tmp/probe_xgram
python train_125m.py --arm plain --d 832 --heads 13 --data-dir ./data \
  --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 \
  --compile 1 --out /tmp/probe_plain184
```

Gate: no NaN, `peak_vram_gb < 0.8 × GPU_VRAM`, `tok/s` within 20% of the
phase-1 probe.

### Wave 1 — new baselines, seed 0 (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --seed 0 --data-dir ./data --train-tokens 2500000000 --compile 1 --out runs_125m/engram_s0 &
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm plain --d 832 --heads 13 --seed 0 --data-dir ./data --train-tokens 2500000000 --compile 1 --out runs_125m/plain184_s0 &
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --seed 0 --data-dir ./data --train-tokens 2500000000 --compile 1 --out runs_125m/xgram_s0 &
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm mp_ve --seed 0 --data-dir ./data --train-tokens 2500000000 --compile 1 --out runs_125m/mpve_s0 &
wait
```

### Wave 2 — seeds 1/2 for new baselines + missing seeds

```bash
for sd in 1 2; do
  CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --seed "$sd" ... --out runs_125m/engram_s$sd &
  CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm plain --d 832 --heads 13 --seed "$sd" ... --out runs_125m/plain184_s$sd &
  CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --seed "$sd" ... --out runs_125m/xgram_s$sd &
  CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm mp_ve --seed "$sd" ... --out runs_125m/mpve_s$sd &
  wait
done
```

### Wave 3 — plain_eq_weights / ctrl_word seeds

```bash
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain_eq_weights --eq-width 1472 --eq-heads 16 --micro-batch 16 --grad-accum 16 --seed 1 ... --out runs_125m/plain_eqW_s1 &
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_word --seed 1 ... --out runs_125m/ctrl_word_s1 &
# repeat for seed 2
```

### Wave 4 — lm-eval

Evaluate `ckpt_2500M.pt` for:
1. `plain`, `word`, `u123`, `ctrl_hs_pruned` (3 seeds each) — **must**;
2. `u1`, `ctrl_hs`, `plain_eq_weights`, `ctrl_word` (3 seeds each) — if time;
3. `engram`, `plain184`, `xgram`, `mp_ve` — after their runs.

Report `acc`, `acc_norm`, `word_perplexity`; paired by seed.

---

## 5. Acceptance gates

1. **New arm gate:** smoke + 25-step probe pass; `peak_vram_gb < 0.8 × VRAM`;
   loss finite and decreasing.
2. **Engram gate:** the run must use the real Engram design (2/3g, K heads,
   key/value, conv, context gate), not `ctrl_hs`. If it is not, rename it and
   do not claim “beats Engram.”
3. **Equal-FLOPs gate:** `plain184` body within 5% of `u123` body (184.2M).
   If not, adjust `d`.
4. **Downstream gate:** LAMBADA / WikiText / PIQA / ARC-Easy; if no task moves
   beyond seed noise, do not claim broad downstream gains; report PPL-only.
5. **Equal-weights gate:** `plain_eq_weights` likely wins at 2.5B. Report it as
   a boundary; do not headline equal-weights.
6. **No leakage gate:** eval wrapper's lookup IDs must be invariant to changing
   the final target token. Run the same test as the training smoke.

---

## 6. What to send back

- `results/countdata_125m.npz` (CountData keys/counts).
- Per-arm `metrics_final.json`, `log.jsonl`, `ckpt_2500M.pt` paths.
- lm-eval JSONs: `results/lm_eval/<arm>_s<seed>.json`.
- A one-page summary table: 2500M val, per-bucket, LAMBADA/WikiText/PIQA/ARC,
  params, train tok/s, b1 latency, peak VRAM.
- `nvidia-smi`, `df -h`, PyTorch/CUDA versions.

---

## 7. Pitfalls

- **CountData alignment:** never evaluate a checkpoint with a different count
  prefix; table rows will be misindexed.
- **Causal leakage:** lookup IDs only from input tokens; test by changing the
  target token.
- **Disk:** 20+ runs × 6 milestones × ~1.5–2.5GB = 200–300GB. Keep only
  `ckpt_500M/1B/2.5B` or `ckpt_final`; delete the rest.
- **`pc_1` unresolved** at 10% prefix; report `pc_0`, `pc_2`, `pc_3`, `pc_4`
  as approximate.
- **Equal weights:** plain_eq overtakes `u123` at ~500M; do not claim
  equal-weights.
- **30M vs 125M:** the 30M `fullmid` result is not predictive; do not use it to
  reject the 125M `u123`.
- **Compile:** run the 25-step probe with `--compile 1`; fall back to
  `--compile 0` if it fails.
- **DDP:** use four independent jobs, not DDP; no NVLink on 4090.

---

## 8. Timeline

| Day | Work |
|---|---|
| 1 | Implement/port Engram + X-GRAM + `save_counts_125m.py`; smoke + probe |
| 2 | Wave 1 + Wave 2 (new baselines, seeds 0–2) |
| 3 | Wave 3 (`plain_eq_weights`, `ctrl_word` seeds); lm-eval wrapper |
| 4–5 | lm-eval on all 2500M checkpoints; aggregate; sanity checks |
| 6 | Buffer for reruns; write `PHASE2_RESULTS.md` |

---

## 9. One-line

**Medium budget = ~300 GPU·h: add Engram + equal-FLOPs plain + X-GRAM + MP-VE,
finish `plain_eq_weights`/`ctrl_word` seeds, and run zero-shot
LAMBADA/WikiText/PIQA/ARC on the 2500M checkpoints. The paper's ceiling is set
by Engram and downstream, not by more small-scale ablations.**
