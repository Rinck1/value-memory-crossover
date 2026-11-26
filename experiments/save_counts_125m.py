#!/usr/bin/env python3
"""Save the exact CountData used by a 125M run, for downstream evaluation.

The memory tables in a 125M checkpoint are indexed by IDs derived from the
first `--count-tokens` tokens.  lm-eval must rebuild the *same* keys, otherwise
table rows are misindexed.  Run this on the same shard directory before eval.

Example (cloud):
    python experiments/save_counts_125m.py \
        --data-dir ./data --count-tokens 250000000 --train-tokens 2500000000 \
        --out results/countdata_125m.npz

Local smoke (one 100M shard):
    python experiments/save_counts_125m.py \
        --data-dir data --count-tokens 10000000 --train-tokens 100000000 \
        --out /tmp/countdata_local.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_125m as t  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--V", type=int, default=50257)
    p.add_argument("--count-tokens", type=int, default=250_000_000)
    p.add_argument("--train-tokens", type=float, default=2_500_000_000)
    p.add_argument("--auto-scale-thresh", type=int, default=1)
    p.add_argument("--hi-thresh", type=int, default=100)
    p.add_argument("--mid-thresh", type=int, default=10)
    p.add_argument("--g3-thresh", type=int, default=30)
    p.add_argument("--n-hi", type=int, default=500_000)
    p.add_argument("--n-mid", type=int, default=1_500_000)
    p.add_argument("--n-3g", type=int, default=150_000)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "countdata_125m.npz"))
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "fineweb_train_*.bin")))
    if not paths:
        raise FileNotFoundError(f"No fineweb_train_*.bin in {args.data_dir}")
    stream = t.ShardStream(paths)
    if args.count_tokens > stream.total:
        raise ValueError(f"--count-tokens {args.count_tokens} > stream {stream.total}")

    scale = args.count_tokens / max(1.0, float(args.train_tokens))
    if args.auto_scale_thresh and scale < 0.99:
        eff_hi = max(2, int(round(args.hi_thresh * scale)))
        eff_mid = max(1, int(round(args.mid_thresh * scale)))
        eff_g3 = max(2, int(round(args.g3_thresh * scale)))
    else:
        eff_hi, eff_mid, eff_g3 = args.hi_thresh, args.mid_thresh, args.g3_thresh

    cd = t.build_counts(
        stream, args.V, args.count_tokens, eff_hi, eff_mid, eff_g3,
        args.n_hi, args.n_mid, args.n_3g,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        hi_keys=cd.hi_keys,
        mid_keys=cd.mid_keys,
        g3_keys=cd.g3_keys,
        mature_keys=cd.mature_keys,
        all_keys=cd.all_keys,
        all_counts=cd.all_counts,
        V=np.array(args.V),
        count_tokens=np.array(args.count_tokens),
        train_tokens=np.array(args.train_tokens),
        eff_hi=np.array(cd.eff_hi),
        eff_mid=np.array(cd.eff_mid),
        eff_g3=np.array(cd.eff_g3),
    )
    print(f"[ok] saved {args.out}")
    print(f"     hi={len(cd.hi_keys):,} mid={len(cd.mid_keys):,} "
          f"mature={len(cd.mature_keys):,} 3g={len(cd.g3_keys):,} "
          f"all_pairs={len(cd.all_keys):,}")


if __name__ == "__main__":
    main()
