# U123 v2: the low-rank mid tier was the bottleneck — but fullmid does not reliably beat the pruned control

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
`d_mem/4 = 128`). Three variants were tested:

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
best arm on seed 1. The low-rank mid bottleneck, not the init, was the problem.

## 3-seed confirmation (fullmid)

| seed | val | pc_100 | pc_1000 |
|---|---|---|---|
| 0 | 4.8033 | 4.5727 | 5.0735 |
| 1 | 4.7998 | 4.5682 | 5.0734 |
| 2 | 4.8184 | 4.5756 | 5.0797 |

**Mean: 4.8072 ± 0.0099.**

**Paired 3-seed comparison (seeds 0,1,2):**

| comparison | diff (baseline − fullmid) | per seed | t |
|---|---|---|---|
| `fullmid` vs `CTRL_hs_pruned` | **+0.0118 ± 0.0163** | +0.0231, +0.0192, **−0.0069** | **+1.25 (n.s.)** |
| `fullmid` vs original `U123` | +0.0284 ± 0.0193 | +0.0201, +0.0505, +0.0146 | +2.54 (marginal) |
| `fullmid` vs `plain_d384_eqWeights` | **+0.0527 ± 0.0110** | +0.0538, +0.0632, +0.0412 | **+8.27** |

### Reading the result honestly

- The 2-seed +0.0211 vs `CTRL_hs_pruned` (t=10.9) was **optimistic**; seed 2 is
  −0.0069 vs pruned and the 3-seed effect is +0.0118 with t=1.25. **Do not
  claim fullmid beats the pruned hashed control.**
- `fullmid` is reliably better than the **original low-rank U123**
  (+0.0284, t=2.54; one-sided p≈0.043, two-sided p≈0.085), and better than
  `plain_d384` (+0.0527, t=8.27). So the low-rank bottleneck is real, but fixing
  it only brings U123 to parity with the pruned control.
- `fullmid_mid2` was not replicated beyond seed 1 (4.8028).

## Implication for the 125M run

`train_125m.py` still builds U123 with the **low-rank mid tier**
(`ve_mid_a @ ve_mid_p`). Before renting 4 GPUs:

1. Add a `fullmid` option to the 125M `MemoryBank` (full-rank hashed mid rows,
   e.g. 50k rows at 30M; scale rows with the 125M token budget), **and**
2. Treat `CTRL_hs_pruned` as a co-main baseline, not a straw man. At 30M the
   pruned hashed control is statistically indistinguishable from fullmid.

The fullmid table is parameter-neutral (76.39M vs 76.46M at 30M), so this is an
architecture fix, not a table-budget change.

## Caveats

- 30M tokens; 125M is untested.
- 3 seeds; the fullmid-vs-pruned effect is small and not significant.
- The fullmid hash row count (50k) is tuned for the 30M/10% prefix; at 125M,
  scale it with the number of mid pairs (or keep the same load factor).
- The honest claim is: "low-rank mid is worse than full-rank mid; fullmid
  recovers the loss but does not reliably beat pruning + hashing."
