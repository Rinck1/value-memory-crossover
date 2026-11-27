#!/usr/bin/env bash
# Inference-time --zero-memory diagnostic for pc_0 / backbone offloading analysis.
# Evaluates 3 seeds x {u123, ctrl_hs_pruned, plain184 (reference), plain, word} on fineweb_val_000000.bin
# under zero_memory in {none, ngram, all}, logging bucketed NLL (pc_0..pc_4) and actual gate openings.
#
# Interpretation rule (strictly DIAGNOSTIC, never standalone proof):
#   - u123[zero-all] vs plain184 (key reference):
#       approx equal  -> pc_0 gain comes from memory branch, not backbone
#       u123[zero-all] < plain184 -> training with memory altered backbone (CONSISTENT WITH capacity offloading)
#   - u123[zero-all] vs ctrl_hs_pruned[zero-all]:
#       approx equal  -> difference between u123 and ctrl_hs_pruned comes from memory branch
#       u123 < ctrl_hs_pruned -> backbone difference exists (attribute cautiously; residual pathways differ)
#
# Usage:
#   bash experiments/run_zero_memory_diag.sh
#   SCALE=350m bash experiments/run_zero_memory_diag.sh
set -euo pipefail

SCALE="${SCALE:-125m}"
DATA_DIR="${DATA_DIR:-./data}"
VAL_SHARD="${DATA_DIR}/fineweb_val_000000.bin"
DTYPE="${DTYPE:-auto}"

if [ ! -f "${VAL_SHARD}" ]; then
    echo "[error] missing ${VAL_SHARD}" >&2
    exit 1
fi

if [ "${SCALE}" = "125m" ]; then
    COUNTDATA="results/countdata_125m.npz"
    CKPT_NAME="ckpt_2500M.pt"
    OUT_DIR="results/zero_memory_diag_125m"
    SPECS=(
        "runs_125m:u123"
        "runs_125m:ctrl_hs_pruned"
        "runs_125m_phase2:plain184"
        "runs_125m:plain"
        "runs_125m:word"
    )
elif [ "${SCALE}" = "350m" ]; then
    COUNTDATA="results/countdata_350m.npz"
    CKPT_NAME="ckpt_5000M.pt"
    OUT_DIR="results/zero_memory_diag_350m"
    SPECS=(
        "runs_350m:u123"
        "runs_350m:ctrl_hs_pruned"
        "runs_350m:plain_eqFLOP"
        "runs_350m:plain"
        "runs_350m:word"
    )
else
    echo "[error] SCALE must be 125m or 350m" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

for spec in "${SPECS[@]}"; do
    root="${spec%%:*}"
    arm="${spec##*:}"
    for sd in 0 1 2; do
        ckpt="${root}/${arm}_s${sd}/${CKPT_NAME}"
        [ -f "${ckpt}" ] || continue
        modes=("none")
        if [ "${arm}" != "plain" ] && [ "${arm}" != "plain184" ] && [ "${arm}" != "plain_eqFLOP" ]; then
            modes=("none" "ngram" "all")
        fi
        for zm in "${modes[@]}"; do
            out_json="${OUT_DIR}/${arm}_s${sd}_zm_${zm}.json"
            if [ -f "${out_json}" ]; then
                echo "[skip existing] ${out_json}"
                continue
            fi
            python experiments/eval_lm.py \
                --checkpoint "${ckpt}" --countdata "${COUNTDATA}" \
                --dtype "${DTYPE}" --zero-memory "${zm}" \
                --verify-val-shard "${VAL_SHARD}" --output "${out_json}"
        done
    done
done

echo "=== Zero-Memory Diagnostic Summary (${SCALE}) ==="
python - "${OUT_DIR}" <<'PY'
import glob, json, os, sys
import numpy as np

out_dir = sys.argv[1]
by_key = {}
for p in sorted(glob.glob(os.path.join(out_dir, "*_zm_*.json"))):
    stem = os.path.basename(p).replace(".json", "")
    arm_sd, zm = stem.rsplit("_zm_", 1)
    arm, sd = arm_sd.rsplit("_s", 1)
    by_key.setdefault((arm, zm), {})[int(sd)] = json.load(open(p))

print(f"{'arm':18s} {'zm':6s} {'n':>2s} | {'val':>14s} | {'pc_0':>14s} | {'pc_2':>14s} | {'pc_4':>14s} | gate_v(pc0/all)")
print("-" * 105)
for (arm, zm), sdict in sorted(by_key.items()):
    vals = [d["val"] for d in sdict.values()]
    pc0 = [d["pc_0"] for d in sdict.values() if "pc_0" in d]
    pc2 = [d["pc_2"] for d in sdict.values() if "pc_2" in d]
    pc4 = [d["pc_4"] for d in sdict.values() if "pc_4" in d]
    fmt = lambda arr: f"{np.mean(arr):.5f}±{np.std(arr, ddof=1):.4f}" if len(arr) > 1 else (f"{arr[0]:.5f}      " if arr else "n/a")
    gv_all = [d["gate_openings"]["gate_v"]["overall"] for d in sdict.values() if "gate_openings" in d and "gate_v" in d["gate_openings"]]
    gv_pc0 = [d["gate_openings"]["gate_v"]["pc_0"] for d in sdict.values() if "gate_openings" in d and "gate_v" in d["gate_openings"] and "pc_0" in d["gate_openings"]["gate_v"]]
    g_str = f"{np.mean(gv_pc0):.3f}/{np.mean(gv_all):.3f}" if gv_all else "-"
    print(f"{arm:18s} {zm:6s} {len(sdict):>2d} | {fmt(vals):>14s} | {fmt(pc0):>14s} | {fmt(pc2):>14s} | {fmt(pc4):>14s} | {g_str}")
PY
