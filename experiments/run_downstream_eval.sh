#!/usr/bin/env bash
# Standalone multi-GPU (or local 1-GPU) downstream zero-shot lm-eval & bucket accuracy runner.
# Automatically uses all visible GPUs (1 GPU locally on RTX 4070, 4 GPUs on cloud).
#
# Usage:
#   bash experiments/run_downstream_eval.sh                       # evaluate both 125m and 350m found checkpoints
#   SCALE=125m BATCH_SIZE=2 bash experiments/run_downstream_eval.sh  # local 8GB RTX 4070
#   SCALE=350m TASKS=lambada_openai,wikitext,piqa,arc_easy,sciq bash experiments/run_downstream_eval.sh
set -euo pipefail

SCALE="${SCALE:-all}"  # 125m | 350m | all
DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR_125M="${OUT_DIR_125M:-./runs_125m_phase2}"
OUT_DIR_350M="${OUT_DIR_350M:-./runs_350m}"
TASKS="${TASKS:-lambada_openai,wikitext,piqa,arc_easy,sciq,hellaswag}"
DTYPE="${DTYPE:-auto}"
NGPUS="${NGPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

if [ -z "${BATCH_SIZE:-}" ]; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 8192)
    if [ "${VRAM_MB:-8192}" -lt 12000 ]; then
        BATCH_SIZE=1
    else
        BATCH_SIZE=8
    fi
fi

echo "=== Downstream Eval: scale=${SCALE} | ngpus=${NGPUS} | batch_size=${BATCH_SIZE} | dtype=${DTYPE} ==="
echo "    tasks=${TASKS}"

JOBS=()

enqueue_scale() {
    local countdata="$1"
    local ckpt_name="$2"
    local out_base="$3"
    shift 3
    if [ ! -f "${countdata}" ]; then
        echo "[skip scale] missing ${countdata}"
        return 0
    fi
    mkdir -p "${out_base}/bucket_acc"
    for base_arm in "$@"; do
        local root="${base_arm%%:*}"
        local arm="${base_arm##*:}"
        for sd in 0 1 2; do
            local ckpt="${root}/${arm}_s${sd}/${ckpt_name}"
            [ -f "${ckpt}" ] || continue
            local out_json="${out_base}/${arm}_s${sd}.json"
            local acc_json="${out_base}/bucket_acc/${arm}_s${sd}.json"
            JOBS+=("${ckpt}|${countdata}|${out_json}|${acc_json}")
        done
    done
}

if [ "${SCALE}" = "125m" ] || [ "${SCALE}" = "all" ]; then
    enqueue_scale "results/countdata_125m.npz" "ckpt_2500M.pt" "results/lm_eval" \
        "runs_125m:u123" "runs_125m:ctrl_hs_pruned" "runs_125m:plain" "runs_125m:word" \
        "runs_125m:u1" "runs_125m:ctrl_hs" "runs_125m:plain_eqW" "runs_125m:ctrl_word" \
        "${OUT_DIR_125M}:engram" "${OUT_DIR_125M}:plain184" "${OUT_DIR_125M}:xgram" \
        "${OUT_DIR_125M}:mpve" "${OUT_DIR_125M}:plain_eqW" "${OUT_DIR_125M}:ctrl_word" \
        "${OUT_DIR_125M}:u123_no3g" "${OUT_DIR_125M}:uniform_mgram"
fi

if [ "${SCALE}" = "350m" ] || [ "${SCALE}" = "all" ]; then
    enqueue_scale "results/countdata_350m.npz" "ckpt_5000M.pt" "results/lm_eval_350m" \
        "${OUT_DIR_350M}:u123" "${OUT_DIR_350M}:ctrl_hs_pruned" "${OUT_DIR_350M}:engram" \
        "${OUT_DIR_350M}:plain_eqFLOP" "${OUT_DIR_350M}:plain" "${OUT_DIR_350M}:word"
fi

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "[warn] no checkpoints found matching SCALE=${SCALE}."
    exit 0
fi

run_one_job() {
    local gpu="$1"
    local spec="$2"
    IFS='|' read -r ckpt countdata out_json acc_json <<< "${spec}"
    if [ -f "${DATA_DIR}/fineweb_val_000000.bin" ] && [ ! -f "${acc_json}" ]; then
        CUDA_VISIBLE_DEVICES="${gpu}" python experiments/eval_lm.py \
            --checkpoint "${ckpt}" --countdata "${countdata}" --dtype "${DTYPE}" \
            --verify-val-shard "${DATA_DIR}/fineweb_val_000000.bin" \
            --output "${acc_json}"
    fi
    if [ -f "${out_json}" ]; then
        echo "[skip existing] ${out_json}"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python experiments/eval_lm.py \
        --checkpoint "${ckpt}" --countdata "${countdata}" --dtype "${DTYPE}" \
        --tasks "${TASKS}" --num_fewshot 0 --batch_size "${BATCH_SIZE}" \
        --output "${out_json}"
}

idx=0
total="${#JOBS[@]}"
while [ "${idx}" -lt "${total}" ]; do
    pids=()
    for ((g=0; g<NGPUS && idx<total; g++)); do
        run_one_job "${g}" "${JOBS[${idx}]}" &
        pids+=($!)
        idx=$((idx + 1))
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
done

echo "=== Downstream Summary ==="
python - <<'PY'
import glob, json, os
for folder in ("results/lm_eval", "results/lm_eval_350m"):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        continue
    print(f"-- {folder} --")
    for p in files:
        d = json.load(open(p)).get("results", {})
        parts = []
        for task, m in sorted(d.items()):
            val = m.get("acc,none", m.get("word_perplexity,none", m.get("acc_norm,none")))
            if val is not None:
                parts.append(f"{task}={val:.4f}" if isinstance(val, (int, float)) else f"{task}={val}")
        print(f"  {os.path.basename(p):26s} " + " | ".join(parts))
PY
