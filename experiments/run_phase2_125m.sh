#!/usr/bin/env bash
# Phase-2 125M launcher (medium budget).
#
# Covers: missing seeds, equal-active-FLOPs plain, lm-eval on phase-1
# checkpoints.  Engram/X-GRAM commands are included but REQUIRE their arms to be
# implemented in train_125m.py first; uncomment only after smoke + 25-step probe.
#
# Usage:
#   bash experiments/run_phase2_125m.sh
#   SKIP_DOWNLOAD=1 bash experiments/run_phase2_125m.sh   # data already there
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./runs_125m_phase2}"
SHARDS="${SHARDS:-26}"
TRAIN_TOKENS="${TRAIN_TOKENS:-2500000000}"
COUNT_TOKENS="${COUNT_TOKENS:-250000000}"
COMPILE="${COMPILE:-1}"

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

echo "=== [0/5] Hardware / environment ==="
nvidia-smi
df -h .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    echo "=== [1/5] Download ${SHARDS} shards ==="
    python download_data.py --shards "${SHARDS}" --workers 12 --data-dir "${DATA_DIR}" --verify
fi

echo "=== [2/5] Save the exact CountData for downstream eval ==="
python experiments/save_counts_125m.py \
    --data-dir "${DATA_DIR}" --V 50257 \
    --count-tokens "${COUNT_TOKENS}" --train-tokens "${TRAIN_TOKENS}" \
    --out results/countdata_125m.npz

mkdir -p "${OUT_DIR}" results/lm_eval

echo "=== [3/5] Smoke + 25-step probes ==="
for arm in plain plain_eq_weights word mp_ve u1 u123 ctrl_word ctrl_hs ctrl_hs_pruned; do
    python train_125m.py --smoke --arm "${arm}" --device cpu --out "/tmp/smoke2_${arm}" >/dev/null
    echo "  [smoke ok] ${arm}"
done

# 25-step real-dimension probes.  Uncomment after implementing the arm.
# python train_125m.py --arm engram --data-dir "${DATA_DIR}" --count-tokens 50000000 \
#   --steps 25 --eval-tokens 3000000 --log-interval 5 --compile "${COMPILE}" --out /tmp/probe_engram
# python train_125m.py --arm xgram --data-dir "${DATA_DIR}" --count-tokens 50000000 \
#   --steps 25 --eval-tokens 3000000 --log-interval 5 --compile "${COMPILE}" --out /tmp/probe_xgram
python train_125m.py --arm plain --d 832 --heads 13 --data-dir "${DATA_DIR}" \
    --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 \
    --compile "${COMPILE}" --out /tmp/probe_plain184

echo "=== [4/5] Training waves ==="

# Wave 1 (seed 0): new baselines.  Engram/X-GRAM require implementation.
pids=()
# CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/engram_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=${GPU_PLAIN184:-1} python train_125m.py --arm plain --d 832 --heads 13 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain184_s0" & pids+=($!)
# CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/xgram_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=${GPU_MPVE:-3} python train_125m.py --arm mp_ve --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/mpve_s0" & pids+=($!)
wait_all "${pids[@]}"

# Wave 2: seeds 1/2 for the new baselines (uncomment engram/xgram after implementation).
for sd in 1 2; do
    pids=()
    # CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --seed "${sd}" ... & pids+=($!)
    CUDA_VISIBLE_DEVICES=${GPU_PLAIN184:-1} python train_125m.py --arm plain --d 832 --heads 13 --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain184_s${sd}" & pids+=($!)
    # CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --seed "${sd}" ... & pids+=($!)
    CUDA_VISIBLE_DEVICES=${GPU_MPVE:-3} python train_125m.py --arm mp_ve --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/mpve_s${sd}" & pids+=($!)
    wait_all "${pids[@]}"
done

# Wave 3: missing seeds for equal weights and the adapter control.
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain_eq_weights --eq-width 1472 --eq-heads 16 --micro-batch 16 --grad-accum 16 --seed 1 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain_eqW_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_word --seed 1 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/ctrl_word_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm plain_eq_weights --eq-width 1472 --eq-heads 16 --micro-batch 16 --grad-accum 16 --seed 2 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain_eqW_s2" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm ctrl_word --seed 2 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/ctrl_word_s2" & pids+=($!)
wait_all "${pids[@]}"

echo "=== [5/5] Zero-shot lm-eval on phase-1 + new checkpoints ==="
# Phase-1 runs live under runs_125m/ (adjust if your paths differ).
for arm in u123 ctrl_hs_pruned plain word; do
    for sd in 0 1 2; do
        ckpt="runs_125m/${arm}_s${sd}/ckpt_2500M.pt"
        [ -f "${ckpt}" ] || { echo "[skip] ${ckpt}"; continue; }
        python experiments/eval_lm.py --checkpoint "${ckpt}" \
            --countdata results/countdata_125m.npz \
            --tasks lambada_openai,wikitext,piqa,arc_easy \
            --num_fewshot 0 --batch_size 4 \
            --output "results/lm_eval/${arm}_s${sd}.json"
    done
done

echo "=== Done. Summary: ==="
python - <<'PY'
import glob, json, os
for p in sorted(glob.glob("runs_125m_phase2/*/metrics_final.json")):
    d = json.load(open(p))
    ev = d.get("final_eval", {})
    print(f"{os.path.basename(os.path.dirname(p)):22s} val={ev.get('val')} "
          f"pc0={ev.get('pc_0')} pc2={ev.get('pc_2')} pc3={ev.get('pc_3')} pc4={ev.get('pc_4')}")
PY
