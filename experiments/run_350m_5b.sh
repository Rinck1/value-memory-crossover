#!/usr/bin/env bash
# Phase-3 / Third-Scale 350M x 5B FineWeb Scaling Launcher (ICML Main Scaling Point).
#
# Architecture: L=24, d=1024, H=16 (head_dim=64; GPT-2 Medium / 350M class, ~404.8M untied plain).
# Table budget: 2x 125M rows (pair_rows=2M, n_hi=1M, n_mid=3M, n_3g=300K, d_mem=256 -> ~538M table, ~1.01B total).
# Batch: --micro-batch 8 --grad-accum 32 -> 131,072 tokens/step (matches 125M; fits 48GB/80GB/96GB GPUs).
# Recommended hardware: 4x H20 (96GB) / A100 (80GB) / 4090-48GB, >=32GB system RAM, ~60GB disk for 51 shards.
#
# Usage:
#   bash experiments/run_350m_5b.sh
#   SKIP_DOWNLOAD=1 RUN_SEEDS2=0 bash experiments/run_350m_5b.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./runs_350m}"
SHARDS="${SHARDS:-51}"                # 51 shards = 5.1B tokens (5.0B budget + margin)
TRAIN_TOKENS="${TRAIN_TOKENS:-5000000000}"
COUNT_TOKENS="${COUNT_TOKENS:-500000000}"
COMPILE="${COMPILE:-1}"
RUN_SEEDS2="${RUN_SEEDS2:-0}"         # Set 1 to run seed 2 as well (~50% more GPU-h)
TASKS="${TASKS:-lambada_openai,wikitext,piqa,arc_easy,sciq,hellaswag}"
EVAL_TOKENS="100000000,250000000,500000000,1000000000,2500000000,5000000000"

COMMON_350=(
    --layers 24 --seq-len 512
    --micro-batch 8 --grad-accum 32
    --train-tokens "${TRAIN_TOKENS}" --count-tokens "${COUNT_TOKENS}"
    --d-mem 256 --pair-rows 2000000 --n-hi 1000000 --n-mid 3000000 --n-3g 300000
    --eval-tokens "${EVAL_TOKENS}"
    --data-dir "${DATA_DIR}" --compile "${COMPILE}"
)

wait_all() {
    local fail=0
    for pid in "$@"; do
        if ! wait "$pid"; then fail=1; fi
    done
    if [ "$fail" -ne 0 ]; then
        echo "[error] at least one 350M job failed; stopping." >&2
        exit 1
    fi
}

echo "=== [0/5] Hardware / environment ==="
nvidia-smi
df -h .
free -h || true
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    echo "=== [1/5] Download ${SHARDS} FineWeb shards (~10.2 GB) ==="
    python download_data.py --shards "${SHARDS}" --workers 12 --data-dir "${DATA_DIR}" --verify
fi

echo "=== [2/5] Save exact 500M-prefix CountData for 350M eval ==="
python experiments/save_counts_125m.py \
    --data-dir "${DATA_DIR}" --V 50257 \
    --count-tokens "${COUNT_TOKENS}" --train-tokens "${TRAIN_TOKENS}" \
    --n-hi 1000000 --n-mid 3000000 --n-3g 300000 \
    --out results/countdata_350m.npz

mkdir -p "${OUT_DIR}" results/lm_eval_350m/bucket_acc

echo "=== [3/5] Smoke + 25-step 350M VRAM/throughput probes ==="
for arm in plain word engram u123 ctrl_hs_pruned; do
    python train_125m.py --smoke --arm "${arm}" --device cpu --out "/tmp/smoke350_${arm}" >/dev/null
    echo "  [smoke ok] ${arm}"
done

# Single-GPU 25-step probes to verify peak VRAM (<0.8 x GPU VRAM) before launching 5B waves.
python train_125m.py --arm u123 --d 1024 --heads 16 "${COMMON_350[@]}" \
    --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 --out /tmp/probe350_u123
python train_125m.py --arm plain --d 1152 --heads 18 "${COMMON_350[@]}" \
    --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 --out /tmp/probe350_plain_eqFLOP

echo "=== [4/5] 350M x 5B Training waves ==="

# Wave 1 (Seed 0, core 4): U123 vs CTRL_hs_pruned vs Engram vs Equal-Active-FLOPs Plain (d=1152, H=18, ~498M body)
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm u123           --d 1024 --heads 16 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/u123_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_hs_pruned --d 1024 --heads 16 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/ctrl_hs_pruned_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm engram         --d 1024 --heads 16 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/engram_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm plain          --d 1152 --heads 18 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/plain_eqFLOP_s0" & pids+=($!)
wait_all "${pids[@]}"

# Wave 2 (Seed 0 base anchors + Seed 1 main pair): Plain-1024, Word-1024, U123 s1, CTRL_hs_pruned s1
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain          --d 1024 --heads 16 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/plain_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm word           --d 1024 --heads 16 "${COMMON_350[@]}" --seed 0 --out "${OUT_DIR}/word_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm u123           --d 1024 --heads 16 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/u123_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm ctrl_hs_pruned --d 1024 --heads 16 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/ctrl_hs_pruned_s1" & pids+=($!)
wait_all "${pids[@]}"

# Wave 3 (Seed 1 remaining arms): Engram s1, Equal-FLOPs Plain s1, Plain-1024 s1, Word-1024 s1
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram         --d 1024 --heads 16 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/engram_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm plain          --d 1152 --heads 18 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/plain_eqFLOP_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm plain          --d 1024 --heads 16 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/plain_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm word           --d 1024 --heads 16 "${COMMON_350[@]}" --seed 1 --out "${OUT_DIR}/word_s1" & pids+=($!)
wait_all "${pids[@]}"

# Optional Wave 4 (Seed 2 for headline 4 arms when RUN_SEEDS2=1)
if [ "${RUN_SEEDS2}" = "1" ]; then
    pids=()
    CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm u123           --d 1024 --heads 16 "${COMMON_350[@]}" --seed 2 --out "${OUT_DIR}/u123_s2" & pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_hs_pruned --d 1024 --heads 16 "${COMMON_350[@]}" --seed 2 --out "${OUT_DIR}/ctrl_hs_pruned_s2" & pids+=($!)
    CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm engram         --d 1024 --heads 16 "${COMMON_350[@]}" --seed 2 --out "${OUT_DIR}/engram_s2" & pids+=($!)
    CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm plain          --d 1152 --heads 18 "${COMMON_350[@]}" --seed 2 --out "${OUT_DIR}/plain_eqFLOP_s2" & pids+=($!)
    wait_all "${pids[@]}"
fi

echo "=== [5/5] 350M Zero-shot lm-eval + FineWeb bucketed top-1 accuracy ==="
SCALE=350m OUT_DIR_350M="${OUT_DIR}" DATA_DIR="${DATA_DIR}" TASKS="${TASKS}" \
    bash experiments/run_downstream_eval.sh

echo "=== Done. 350M Summary: ==="
python - <<'PY'
import glob, json, os
for p in sorted(glob.glob(os.path.join(os.environ.get("OUT_DIR", "runs_350m"), "*", "metrics_final.json"))):
    d = json.load(open(p))
    ev = d.get("final_eval", {})
    print(f"{os.path.basename(os.path.dirname(p)):22s} val={ev.get('val')} "
          f"pc0={ev.get('pc_0')} pc2={ev.get('pc_2')} pc3={ev.get('pc_3')} pc4={ev.get('pc_4')}")
PY
