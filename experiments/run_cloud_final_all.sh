#!/usr/bin/env bash
# Final One-Shot Idempotent Cloud Runner (ICML Closing Suite)
#
# Executes all remaining cloud experiments in order, automatically skipping any
# checkpoint, CountData file, lm-eval JSON, or diagnostic JSON that already exists
# (safe to re-run after preemption or interruption):
#
#   Step 1 (~2 min):  125M --zero-memory diagnostic + gate openings on existing Phase-1/2 checkpoints
#   Step 2 (~4 h):    125M uniform_mgram x 3 seeds (Over-Tokenized value-side control) + 6-task lm-eval + diag
#   Step 3 (~28-48h): 350M x 5B full scaling matrix (exact frozen b89068c hyperparams; 2 seeds + optional seed 2)
#                     + 350M 6-task lm-eval + 350M --zero-memory diagnostic + 43d54ad preregistration check
#
# Usage:
#   bash experiments/run_cloud_final_all.sh                     # default: RUN_SEEDS2=1 (core 4 arms get 3 seeds!)
#   RUN_SEEDS2=0 bash experiments/run_cloud_final_all.sh        # faster 2-seed 350M (~28-36h)
#   SKIP_125M_UNIFORM=1 bash experiments/run_cloud_final_all.sh # jump straight to 350M after 2-min 125M diag
#   SKIP_DOWNLOAD=1 bash experiments/run_cloud_final_all.sh     # shards already present in DATA_DIR
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
PHASE1_DIR="${PHASE1_DIR:-./runs_125m}"
OUT_DIR_125M="${OUT_DIR_125M:-./runs_125m_phase2}"
OUT_DIR_350M="${OUT_DIR_350M:-./runs_350m}"
COMPILE="${COMPILE:-1}"
DTYPE="${DTYPE:-auto}"
TASKS="${TASKS:-lambada_openai,wikitext,piqa,arc_easy,sciq,hellaswag}"

# Toggles (all enabled by default for full ICML closure)
SKIP_125M_ZERO_MEM="${SKIP_125M_ZERO_MEM:-0}"
SKIP_125M_UNIFORM="${SKIP_125M_UNIFORM:-0}"
SKIP_350M="${SKIP_350M:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
RUN_SEEDS2="${RUN_SEEDS2:-1}"   # 1 = Wave 4 runs seed 2 on headline 4 arms (enables 3-seed paired t-test at 350M!)

# Exact frozen b89068c 350M x 5B parameters (do NOT modify)
SHARDS_350M="${SHARDS_350M:-51}"
TRAIN_TOKENS_125M=2500000000
COUNT_TOKENS_125M=250000000
TRAIN_TOKENS_350M=5000000000
COUNT_TOKENS_350M=500000000
EVAL_TOKENS_350M="100000000,250000000,500000000,1000000000,2500000000,5000000000"

COMMON_350=(
    --layers 24 --seq-len 512
    --micro-batch 8 --grad-accum 32
    --train-tokens "${TRAIN_TOKENS_350M}" --count-tokens "${COUNT_TOKENS_350M}"
    --d-mem 256 --pair-rows 2000000 --n-hi 1000000 --n-mid 3000000 --n-3g 300000
    --eval-tokens "${EVAL_TOKENS_350M}"
    --data-dir "${DATA_DIR}" --compile "${COMPILE}"
)

wait_all() {
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    local fail=0
    for pid in "$@"; do
        if ! wait "$pid"; then fail=1; fi
    done
    if [ "$fail" -ne 0 ]; then
        echo "[error] at least one background training job failed; stopping." >&2
        exit 1
    fi
}

echo "=================================================================="
echo " ICML Final Cloud Suite | RUN_SEEDS2=${RUN_SEEDS2} | COMPILE=${COMPILE}"
echo " DATA_DIR=${DATA_DIR} | PHASE1_DIR=${PHASE1_DIR}"
echo " OUT_DIR_125M=${OUT_DIR_125M} | OUT_DIR_350M=${OUT_DIR_350M}"
echo "=================================================================="
nvidia-smi
df -h .
free -h || true
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'ngpus:', torch.cuda.device_count())"

mkdir -p "${OUT_DIR_125M}" "${OUT_DIR_350M}" results/lm_eval/bucket_acc results/lm_eval_350m/bucket_acc

# ---------------------------------------------------------------------------
# STEP 1: 125M --zero-memory diagnostic on existing Phase-1/2 checkpoints (~2 min)
# ---------------------------------------------------------------------------
if [ "${SKIP_125M_ZERO_MEM}" != "1" ]; then
    echo ""
    echo "=== [STEP 1/3] 125M --zero-memory diagnostic & gate openings ==="
    if [ ! -f "results/countdata_125m.npz" ]; then
        echo "[1/3] Generating results/countdata_125m.npz ..."
        python experiments/save_counts_125m.py \
            --data-dir "${DATA_DIR}" --V 50257 \
            --count-tokens "${COUNT_TOKENS_125M}" --train-tokens "${TRAIN_TOKENS_125M}" \
            --out results/countdata_125m.npz
    fi
    SCALE=125m DATA_DIR="${DATA_DIR}" PHASE1_DIR="${PHASE1_DIR}" OUT_DIR_125M="${OUT_DIR_125M}" DTYPE="${DTYPE}" \
        bash experiments/run_zero_memory_diag.sh
fi

# ---------------------------------------------------------------------------
# STEP 2: 125M uniform_mgram x 3 seeds (Over-Tokenized control) + downstream (~4h)
# ---------------------------------------------------------------------------
if [ "${SKIP_125M_UNIFORM}" != "1" ]; then
    echo ""
    echo "=== [STEP 2/3] 125M uniform_mgram x 3 seeds + 6-task lm-eval ==="
    if [ "${SKIP_DOWNLOAD}" != "1" ] && [ ! -f "${DATA_DIR}/fineweb_train_000026.bin" ]; then
        python download_data.py --shards 26 --workers 12 --data-dir "${DATA_DIR}" --verify
    fi
    if [ ! -f "results/countdata_125m.npz" ]; then
        python experiments/save_counts_125m.py \
            --data-dir "${DATA_DIR}" --V 50257 \
            --count-tokens "${COUNT_TOKENS_125M}" --train-tokens "${TRAIN_TOKENS_125M}" \
            --out results/countdata_125m.npz
    fi

    # Smoke check only if any uniform_mgram seed is still missing
    need_um=0
    for sd in 0 1 2; do
        [ -f "${OUT_DIR_125M}/uniform_mgram_s${sd}/ckpt_2500M.pt" ] || need_um=1
    done
    if [ "${need_um}" = "1" ]; then
        python train_125m.py --smoke --arm uniform_mgram --device cpu --out /tmp/smoke_uniform_mgram >/dev/null
        echo "  [smoke ok] uniform_mgram"
        pids=()
        for sd in 0 1 2; do
            out_sub="${OUT_DIR_125M}/uniform_mgram_s${sd}"
            if [ -f "${out_sub}/ckpt_2500M.pt" ]; then
                echo "  [skip existing] ${out_sub}/ckpt_2500M.pt"
                continue
            fi
            CUDA_VISIBLE_DEVICES="${sd}" python train_125m.py \
                --arm uniform_mgram --micro-batch 16 --grad-accum 16 --seed "${sd}" \
                --data-dir "${DATA_DIR}" --train-tokens "${TRAIN_TOKENS_125M}" \
                --compile "${COMPILE}" --out "${out_sub}" &
            pids+=($!)
        done
        wait_all "${pids[@]}"
    else
        echo "  [skip] all 3 uniform_mgram seeds already finished."
    fi

    # Automatically skips existing JSONs and evaluates only newly completed 125M checkpoints
    SCALE=125m DATA_DIR="${DATA_DIR}" OUT_DIR_125M="${OUT_DIR_125M}" TASKS="${TASKS}" DTYPE="${DTYPE}" \
        bash experiments/run_downstream_eval.sh
    # Also add uniform_mgram into the 125M zero-memory diagnostic table (skips existing JSONs)
    SCALE=125m DATA_DIR="${DATA_DIR}" PHASE1_DIR="${PHASE1_DIR}" OUT_DIR_125M="${OUT_DIR_125M}" DTYPE="${DTYPE}" \
        bash experiments/run_zero_memory_diag.sh
fi

# ---------------------------------------------------------------------------
# STEP 3: 350M x 5B full scaling matrix (idempotent) + downstream + zero-memory
# ---------------------------------------------------------------------------
if [ "${SKIP_350M}" != "1" ]; then
    echo ""
    echo "=== [STEP 3/3] 350M x 5B Scaling Suite (frozen b89068c hyperparams) ==="
    if [ "${SKIP_DOWNLOAD}" != "1" ] && [ ! -f "${DATA_DIR}/fineweb_train_000051.bin" ]; then
        echo "[3/3] Downloading ${SHARDS_350M} FineWeb shards (~10.2 GB) ..."
        python download_data.py --shards "${SHARDS_350M}" --workers 12 --data-dir "${DATA_DIR}" --verify
    fi

    if [ ! -f "results/countdata_350m.npz" ]; then
        echo "[3/3] Building exact 500M-prefix CountData (results/countdata_350m.npz) ..."
        python experiments/save_counts_125m.py \
            --data-dir "${DATA_DIR}" --V 50257 \
            --count-tokens "${COUNT_TOKENS_350M}" --train-tokens "${TRAIN_TOKENS_350M}" \
            --n-hi 1000000 --n-mid 3000000 --n-3g 300000 \
            --out results/countdata_350m.npz
    else
        echo "  [skip existing] results/countdata_350m.npz"
    fi

    # Run smoke + 25-step VRAM probe only if no 350M checkpoint exists yet
    if ! ls "${OUT_DIR_350M}"/*/ckpt_5000M.pt >/dev/null 2>&1; then
        echo "[3/3] Smoke + 25-step 350M VRAM/throughput probes ..."
        for arm in plain word engram u123 ctrl_hs_pruned; do
            python train_125m.py --smoke --arm "${arm}" --device cpu --out "/tmp/smoke350_${arm}" >/dev/null
            echo "  [smoke ok] ${arm}"
        done
        python train_125m.py --arm u123 --d 1024 --heads 16 "${COMMON_350[@]}" \
            --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 --out /tmp/probe350_u123
        python train_125m.py --arm plain --d 1152 --heads 18 "${COMMON_350[@]}" \
            --count-tokens 50000000 --steps 25 --eval-tokens 3000000 --log-interval 5 --out /tmp/probe350_plain_eqFLOP
    fi

    run_350m_wave() {
        local wave_name="$1"
        shift
        echo "--- 350M ${wave_name} ---"
        local pids=()
        for entry in "$@"; do
            # Format: gpu|arm|d|heads|seed|out_name
            IFS='|' read -r gpu arm d_w h_w sd out_nm <<< "${entry}"
            local out_sub="${OUT_DIR_350M}/${out_nm}"
            if [ -f "${out_sub}/ckpt_5000M.pt" ]; then
                echo "  [skip existing] ${out_sub}/ckpt_5000M.pt"
                continue
            fi
            echo "  [launch GPU ${gpu}] ${out_nm} (arm=${arm}, d=${d_w}, H=${h_w}, seed=${sd})"
            CUDA_VISIBLE_DEVICES="${gpu}" python train_125m.py \
                --arm "${arm}" --d "${d_w}" --heads "${h_w}" \
                "${COMMON_350[@]}" --seed "${sd}" --out "${out_sub}" &
            pids+=($!)
        done
        wait_all "${pids[@]}"
    }

    # Wave 1 (Seed 0, core 4): U123 vs CTRL_hs_pruned vs Engram vs Equal-Active-FLOPs Plain
    run_350m_wave "Wave 1 (Seed 0 core 4)" \
        "0|u123|1024|16|0|u123_s0" \
        "1|ctrl_hs_pruned|1024|16|0|ctrl_hs_pruned_s0" \
        "2|engram|1024|16|0|engram_s0" \
        "3|plain|1152|18|0|plain_eqFLOP_s0"

    # Wave 2 (Seed 0 base anchors + Seed 1 main pair)
    run_350m_wave "Wave 2 (Seed 0 plain/word + Seed 1 main pair)" \
        "0|plain|1024|16|0|plain_s0" \
        "1|word|1024|16|0|word_s0" \
        "2|u123|1024|16|1|u123_s1" \
        "3|ctrl_hs_pruned|1024|16|1|ctrl_hs_pruned_s1"

    # Wave 3 (Seed 1 remaining 4 arms)
    run_350m_wave "Wave 3 (Seed 1 remaining 4)" \
        "0|engram|1024|16|1|engram_s1" \
        "1|plain|1152|18|1|plain_eqFLOP_s1" \
        "2|plain|1024|16|1|plain_s1" \
        "3|word|1024|16|1|word_s1"

    # Wave 4 (Seed 2 for headline 4 arms -> enables 3-seed paired t-test at 350M!)
    if [ "${RUN_SEEDS2}" = "1" ]; then
        run_350m_wave "Wave 4 (Seed 2 headline 4 for 3-seed paired t-tests)" \
            "0|u123|1024|16|2|u123_s2" \
            "1|ctrl_hs_pruned|1024|16|2|ctrl_hs_pruned_s2" \
            "2|engram|1024|16|2|engram_s2" \
            "3|plain|1152|18|2|plain_eqFLOP_s2"
    fi

    echo "[3/3] 350M Zero-shot 6-task lm-eval + FineWeb bucket top-1 accuracy ..."
    SCALE=350m OUT_DIR_350M="${OUT_DIR_350M}" DATA_DIR="${DATA_DIR}" TASKS="${TASKS}" DTYPE="${DTYPE}" \
        bash experiments/run_downstream_eval.sh

    echo "[3/3] 350M --zero-memory diagnostic & gate openings ..."
    SCALE=350m OUT_DIR_350M="${OUT_DIR_350M}" DATA_DIR="${DATA_DIR}" DTYPE="${DTYPE}" \
        bash experiments/run_zero_memory_diag.sh

    echo ""
    echo "=== 350M x 5B Final Summary & Pre-Unblinding 43d54ad Verification ==="
    python - "${OUT_DIR_350M}" <<'PY'
import glob, json, math, os, sys
import numpy as np

out_dir = sys.argv[1]
by_arm = {}
for p in sorted(glob.glob(os.path.join(out_dir, "*", "metrics_final.json"))):
    nm = os.path.basename(os.path.dirname(p))
    arm, sd = nm.rsplit("_s", 1)
    d = json.load(open(p))
    ev = d.get("final_eval", {})
    by_arm.setdefault(arm, {})[int(sd)] = ev
    print(f"  {nm:22s} val={ev.get('val')} pc0={ev.get('pc_0')} pc2={ev.get('pc_2')} pc3={ev.get('pc_3')} pc4={ev.get('pc_4')}")

if "u123" in by_arm and "ctrl_hs_pruned" in by_arm:
    common = sorted(set(by_arm["u123"]) & set(by_arm["ctrl_hs_pruned"]))
    diffs = np.array([by_arm["ctrl_hs_pruned"][s]["val"] - by_arm["u123"][s]["val"] for s in common], float)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    print("\n--- Preregistered 350M Check (ctrl_hs_pruned - u123, positive = u123 wins) ---")
    print(f"  Observed ({len(common)} seeds {common}): {mean_d:+.5f} +- {sd_d:.5f} | per-seed: {[round(float(x), 5) for x in diffs]}")
    print("  Preregistered 43d54ad (Sub-Zipf U_keep): +0.02689  95% PI = [+0.01689, +0.03689]")
    print("  Preregistered b89068c (ln U_all/M):      +0.02250  (3-pt +0.01980)")
    print(f"  Inside 43d54ad 95% PI? {0.01689 <= mean_d <= 0.03689}")
PY
fi

echo ""
echo "=== ALL FINAL CLOUD SUITE STEPS COMPLETE ==="
