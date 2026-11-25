# U123 v2: the low-rank mid tier was the bottleneck

**Date:** 2026-11-25  
**Setting:** V=50,257, 30M tokens, seq=256, 10% prefix counts, 3-gram tier,
same shared-bank + ctx-gate + residual + pre-head architecture as U123.

## Why this experiment

The 3-seed 30M bridge in `results/v50257_bridge_results.json` showed that the
original U123 did **not** beat the pruned hashed control:

| arm (3 seeds) | mean 30M val |
|---|---|
| `U123_d256_prefix10` | 4.8355 |
| `CTRL_hs_d256_prefix10` | 4.8269 |
| `CTRL_hs_pruned_d256_prefix10` | **4.8189** |
| `plain_d384_eqWeights` | 4.8599 |

The original U123 used a low-rank mid tier (`ve_mid_a @ ve_mid_p`, rank
`d_mem/4 = 128`). Two hypotheses were tested on seed 1:

1. **initfix** — align the `ve_mid_p` initialisation variance (`d_mem**-0.5`
   instead of `(r*d_mem)**-0.5`).
2. **fullmid** — replace the low-rank mid bottleneck with a direct full-rank
   hashed mid table (50k rows × 512).
3. **fullmid_mid2** — same, with a stricter mid prefix cutoff (`mid >= 2`).

## Seed-1 results (30M val)

| variant | val | pc_100 | pc_1000 | table params |
|---|---|---|---|---|
| U123 baseline (seed 1) | 4.8503 | 4.6042 | 5.0914 | 76.46M |
| `initfix` | 4.8445 | 4.6044 | 5.1015 | 76.46M |
| **`fullmid`** | **4.7998** | **4.5682** | **5.0734** | 76.39M |
| `fullmid_mid2` | 4.8028 | 4.5687 | 5.0690 | 76.39M |
| `CTRL_hs` (seed 1) | 4.8336 | 4.6089 | 5.0852 | 76.93M |
| `CTRL_hs_pruned` (seed 1) | 4.8190 | 4.5957 | 5.0855 | 76.93M |
| `plain_d384_eqWeights` (seed 1) | 4.8630 | — | — | 49.2M dense |

`initfix` recovers only 0.006; `fullmid` recovers **0.050** and becomes the
best arm. The low-rank mid bottleneck, not the init, was the problem.

## Seed 0 confirmation (fullmid)

| seed | val | pc_100 | pc_1000 |
|---|---|---|---|
| 0 | 4.8033 | 4.5727 | 5.0735 |
| 1 | 4.7998 | 4.5682 | 5.0734 |

**Paired 2-seed comparison (seeds 0,1):**

| comparison | diff (baseline − fullmid) | per seed | t |
|---|---|---|---|
| `fullmid` vs `CTRL_hs_pruned` | **+0.0211 ± 0.0028** | +0.0231, +0.0192 | **+10.9** |
| `fullmid` vs original `U123` | +0.0353 ± 0.0214 | +0.0201, +0.0505 | +2.3 |
| `fullmid` vs `plain_d384_eqWeights` | **+0.0585 ± 0.0067** | +0.0538, +0.0632 | **+12.4** |

Seed 2 is running to complete the 3-seed comparison.

## Implication for the 125M run

`train_125m.py` still builds U123 with the **low-rank mid tier**
(`ve_mid_a @ ve_mid_p`). The 30M v2 result says that before renting 4 GPUs:

1. Add a `fullmid` option to the 125M `MemoryBank` (full-rank hashed mid rows,
   e.g. 50k rows at 30M; scale rows with the 125M token budget), or
2. Do **not** expect the original U123 to beat `CTRL_hs_pruned` at 125M.

The fullmid table is parameter-neutral (76.39M vs 76.46M at 30M), so this is an
architecture fix, not a table-budget change.

## Caveats

- 30M tokens; 125M is untested.
- 2 seeds at the time of writing (seed 2 running); treat the t-stat as
  preliminary until the third seed lands.
- The fullmid hash row count (50k) is tuned for the 30M/10% prefix; at 125M,
  scale it with the number of mid pairs (or keep the same load factor).
