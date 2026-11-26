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
RUN_P2_ABLATIONS="${RUN_P2_ABLATIONS:-1}"
TASKS="${TASKS:-lambada_openai,wikitext,piqa,arc_easy,sciq,hellaswag}"

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
for arm in plain plain_eq_weights word xgram engram mp_ve u1 u123 u123_no3g ctrl_word ctrl_hs ctrl_hs_pruned; do
    python train_125m.py --smoke --arm "${arm}" --device cpu --out "/tmp/smoke2_${arm}" >/dev/null
    echo "  [smoke ok] ${arm}"
done

# 25-step real-dimension probes across all newly added / modified arms.
python train_125m.py --arm engram --micro-batch 16 --grad-accum 16 --data-dir "${DATA_DIR}" --count-tokens 50000000 \
    --steps 25 --eval-tokens 3000000 --log-interval 5 --compile "${COMPILE}" --out /tmp/probe_engram
python train_125m.py --arm xgram --micro-batch 16 --grad-accum 16 --data-dir "${DATA_DIR}" --count-tokens 50000000 \
    --steps 25 --eval-tokens 3000000 --log-interval 5 --compile "${COMPILE}" --out /tmp/probe_xgram
python train_125m.py --arm plain --d 832 --heads 13 --micro-batch 16 --grad-accum 16 --data-dir "${DATA_DIR}" \
    --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 \
    --compile "${COMPILE}" --out /tmp/probe_plain184
if [ "${RUN_P2_ABLATIONS}" = "1" ]; then
    python train_125m.py --arm u123_no3g --micro-batch 16 --grad-accum 16 --data-dir "${DATA_DIR}" --count-tokens 50000000 \
        --steps 25 --eval-tokens 3000000 --log-interval 5 --compile "${COMPILE}" --out /tmp/probe_u123_no3g
fi

if [ -f "runs_125m/u123_s0/ckpt_2500M.pt" ] && [ -f "${DATA_DIR}/fineweb_val_000000.bin" ]; then
    echo "=== [3b/5] Verify CountData alignment against Phase-1 u123_s0 ==="
    python experiments/eval_lm.py --checkpoint runs_125m/u123_s0/ckpt_2500M.pt \
        --countdata results/countdata_125m.npz \
        --verify-val-shard "${DATA_DIR}/fineweb_val_000000.bin"
fi

echo "=== [4/5] Training waves ==="

# Wave 1 (seed 0): new baselines (Engram, Equal-Active-FLOP Plain-184M, X-GRAM, MP-VE).
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --micro-batch 16 --grad-accum 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/engram_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm plain --d 832 --heads 13 --micro-batch 16 --grad-accum 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain184_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --micro-batch 16 --grad-accum 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/xgram_s0" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm mp_ve --micro-batch 16 --grad-accum 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/mpve_s0" & pids+=($!)
wait_all "${pids[@]}"

# Wave 2: seeds 1/2 for the new baselines.
for sd in 1 2; do
    pids=()
    CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm engram --micro-batch 16 --grad-accum 16 --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/engram_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm plain --d 832 --heads 13 --micro-batch 16 --grad-accum 16 --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain184_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm xgram --micro-batch 16 --grad-accum 16 --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/xgram_s${sd}" & pids+=($!)
    CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm mp_ve --micro-batch 16 --grad-accum 16 --seed "${sd}" --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/mpve_s${sd}" & pids+=($!)
    wait_all "${pids[@]}"
done

# Wave 3: missing seeds for equal weights and the adapter control.
pids=()
CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm plain_eq_weights --eq-width 1472 --eq-heads 16 --micro-batch 16 --grad-accum 16 --seed 1 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain_eqW_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm ctrl_word --micro-batch 16 --grad-accum 16 --seed 1 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/ctrl_word_s1" & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm plain_eq_weights --eq-width 1472 --eq-heads 16 --micro-batch 16 --grad-accum 16 --seed 2 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/plain_eqW_s2" & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python train_125m.py --arm ctrl_word --micro-batch 16 --grad-accum 16 --seed 2 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/ctrl_word_s2" & pids+=($!)
wait_all "${pids[@]}"

# Wave 4 (P2 ablation): isolate 3-gram contribution in U123 at 125M x 2.5B.
if [ "${RUN_P2_ABLATIONS}" = "1" ]; then
    pids=()
    CUDA_VISIBLE_DEVICES=0 python train_125m.py --arm u123_no3g --micro-batch 16 --grad-accum 16 --seed 0 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/u123_no3g_s0" & pids+=($!)
    CUDA_VISIBLE_DEVICES=1 python train_125m.py --arm u123_no3g --micro-batch 16 --grad-accum 16 --seed 1 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/u123_no3g_s1" & pids+=($!)
    CUDA_VISIBLE_DEVICES=2 python train_125m.py --arm u123_no3g --micro-batch 16 --grad-accum 16 --seed 2 --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS}" --compile "${COMPILE}" --out "${OUT_DIR}/u123_no3g_s2" & pids+=($!)
    wait_all "${pids[@]}"
fi

echo "=== [5/5] Zero-shot lm-eval + FineWeb bucketed top-1 accuracy ==="
mkdir -p results/lm_eval/bucket_acc
for base_arm in \
    "runs_125m:u123" "runs_125m:ctrl_hs_pruned" "runs_125m:plain" "runs_125m:word" \
    "runs_125m:u1" "runs_125m:ctrl_hs" "runs_125m:plain_eqW" "runs_125m:ctrl_word" \
    "${OUT_DIR}:engram" "${OUT_DIR}:plain184" "${OUT_DIR}:xgram" "${OUT_DIR}:mpve" \
    "${OUT_DIR}:plain_eqW" "${OUT_DIR}:ctrl_word" "${OUT_DIR}:u123_no3g"; do
    root="${base_arm%%:*}"
    arm="${base_arm##*:}"
    for sd in 0 1 2; do
        ckpt="${root}/${arm}_s${sd}/ckpt_2500M.pt"
        [ -f "${ckpt}" ] || continue
        if [ -f "${DATA_DIR}/fineweb_val_000000.bin" ] && [ ! -f "results/lm_eval/bucket_acc/${arm}_s${sd}.json" ]; then
            python experiments/eval_lm.py --checkpoint "${ckpt}" \
                --countdata results/countdata_125m.npz \
                --verify-val-shard "${DATA_DIR}/fineweb_val_000000.bin" \
                --output "results/lm_eval/bucket_acc/${arm}_s${sd}.json"
        fi
        out_json="results/lm_eval/${arm}_s${sd}.json"
        if [ -f "${out_json}" ]; then
            echo "[skip existing] ${out_json}"
            continue
        fi
        python experiments/eval_lm.py --checkpoint "${ckpt}" \
            --countdata results/countdata_125m.npz \
            --tasks "${TASKS}" \
            --num_fewshot 0 --batch_size 8 \
            --output "${out_json}"
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
