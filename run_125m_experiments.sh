#!/usr/bin/env bash
# Cloud 4-GPU / 1-GPU launcher for 125M x 2.5B FineWeb Value-Memory Experiments.
# Step 1: Concurrent data download (val + 25 train shards = 5.0 GB)
# Step 2: Phase-1 Single-Seed Complete Comparison (8 key arms across two 4-GPU waves)
# Step 3: Phase-2 Multi-Seed Confirmation (seeds 1 & 2 for headline arms: plain, word, u1, u123, ctrl_hs, ctrl_hs_pruned)
# Step 4: Downstream lm-eval checkpoint evaluation & summary aggregation

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./runs_125m}"
SHARDS="${SHARDS:-26}"          # 26 shards = 2.6B tokens (2.5B budget + margin)
TRAIN_TOKENS="${TRAIN_TOKENS:-2500000000}"

wait_all() {
    local fail=0
    for pid in "$@"; do
        if ! wait "$pid"; then fail=1; fi
    done
    if [ "$fail" -ne 0 ]; then
        echo "[error] at least one job in this wave failed; stopping." >&2
        exit 1
    fi
}

echo "=== [1/4] Downloading ${SHARDS} FineWeb GPT-2 shards (workers=12) ==="
python download_data.py --shards "${SHARDS}" --workers 12 --data-dir "${DATA_DIR}" --verify

echo "=== [2/4] Running 125M Smoke Regression Across All Core Arms ==="
for arm in plain plain_eq_weights word hard_stack mp_ve u1 u123 ctrl_word ctrl_hs ctrl_hs_pruned; do
    python train_125m.py --smoke --arm "${arm}" --out "/tmp/smoke_${arm}" --device cpu >/dev/null
    echo "  [smoke ok] ${arm}"
done

mkdir -p "${OUT_DIR}"

echo "=== [3/4] Wave 1 (Seed 0): Plain EqFLOP | Word | U123 (Main) | CTRL_hs (No Maturity) ==="
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain   --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/plain_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm word    --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/word_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm u123    --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/u123_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm ctrl_hs --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/ctrl_hs_s0" & pids+=($!)
wait_all "${pids[@]}"

echo "=== [3/4] Wave 2 (Seed 0 Controls): Plain EqWeights | CTRL_hs_pruned | CTRL_word | U1 (Low-Latency) ==="
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain_eq_weights --eq-width 1152 --eq-heads 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/plain_eqW_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_hs_pruned   --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/ctrl_hs_pruned_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm ctrl_word        --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/ctrl_word_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm u1               --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/u1_s0" & pids+=($!)
wait_all "${pids[@]}"

echo "=== [4/4] Wave 3 & 4 (Multi-Seed Confirmation: Seeds 1 & 2 on Headline Arms) ==="
for sd in 1 2; do
    pids=()
    CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain          --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/plain_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm word           --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/word_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm u123           --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/u123_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm ctrl_hs        --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/ctrl_hs_s${sd}" & pids+=($!)
    wait_all "${pids[@]}"
done

pids=()
for sd in 1 2; do
    CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm ctrl_hs_pruned --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/ctrl_hs_pruned_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm u1             --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile 1 --out "${OUT_DIR}/u1_s${sd}" & pids+=($!)
done
wait_all "${pids[@]}"

echo "=== Printing Final 125M Summary Across All Runs ==="
python -c '
import glob, json, os
for p in sorted(glob.glob(os.path.join("'"${OUT_DIR}"'", "*", "metrics_final.json"))):
    d = json.load(open(p))
    ev = d.get("final_eval", {})
    lat = d.get("latency", {})
    print(f"{os.path.basename(os.path.dirname(p)):20s} | val={ev.get(\"val\")} | pc0={ev.get(\"pc_0\")} pc100={ev.get(\"pc_3\")} | "
          f"body={d[\"body_params\"]/1e6:.1f}M tab={d[\"table_params\"]/1e6:.1f}M | tok/s={d[\"train_tok_s\"]:,.0f} | b1={lat.get(\"b1_seq_ms_mean\")}ms")
'
