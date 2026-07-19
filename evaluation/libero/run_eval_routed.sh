#!/bin/bash
# =============================================================================
# Routed LIBERO Evaluation — dual local model, criticality-gated
# =============================================================================
#
# Usage:  bash run_eval_routed.sh <base_ckpt> <ft_ckpt> <criticality_ckpt> [num_trials] [output_prefix] ["<gpu_ids>"]
#
# Env overrides (optional):
#   NORM_STATS       path to norm_stats JSON
#   SMOLVLM_MODEL    SmolVLM backbone path or HF id
#   CRIT_THRESHOLD   routing threshold (default: 0.85)
# =============================================================================
set -e

SCRIPT_DIR="/mnt/hlx"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

BASE_CKPT=${1:-"YuankaiLuo/SimVLA-LIBERO"}
FT_CKPT=${2:-"/mnt/hlx/SimpleVLA_libero_data/runs/bc_continual/ckpt-20000"}
CRIT_CKPT=${3:-"/mnt/hlx/SimpleVLA_libero_data/runs/criticality/best.pt"}
NUM_TRIALS=${4:-50}
OUTPUT_PREFIX=${5:-"eval_routed"}
GPUS=${6:-"0 5 6 7"}
OUTPUT_DIR="./eval_routed_bc"
THRESHOLD=${CRIT_THRESHOLD:-0.7}

NORM_STATS=${NORM_STATS:-"${SCRIPT_DIR}/SimpleVLA_libero/norm_stats/libero_norm.json"}
SMOLVLM_MODEL=${SMOLVLM_MODEL:-"/mnt/hlx/SimpleVLA_libero_data/models/SmolVLM-500M-Instruct"}

read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: Need 4 GPUs, got ${#GPU_ARRAY[@]}"
    exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Routed LIBERO evaluation (local)"
echo "   Base ckpt:     $BASE_CKPT"
echo "   FT ckpt:       $FT_CKPT"
echo "   Criticality:   $CRIT_CKPT"
echo "   Threshold:     $THRESHOLD"
echo "   Norm stats:    $NORM_STATS"
echo "   SmolVLM:       $SMOLVLM_MODEL"
echo "   Num trials:    $NUM_TRIALS"
echo "   GPUs:          $GPUS"
echo ""

run_suite () {
    local gpu=$1
    local suite=$2
    local log=$3
    CUDA_VISIBLE_DEVICES=$gpu python -u libero_client.py \
        --client_type local \
        --checkpoint "$BASE_CKPT" \
        --ft_checkpoint "$FT_CKPT" \
        --norm_stats "$NORM_STATS" \
        --smolvlm_model "$SMOLVLM_MODEL" \
        --criticality_ckpt "$CRIT_CKPT" \
        --criticality_threshold "$THRESHOLD" \
        --task_suite "$suite" \
        --num_trials $NUM_TRIALS \
        --video_out "$OUTPUT_DIR" > "$log" 2>&1 &
}

echo "Launching 4 routed eval tasks..."
run_suite ${GPU_ARRAY[0]} libero_spatial "${OUTPUT_DIR}/${OUTPUT_PREFIX}_spatial.txt"
echo "   [PID $!] libero_spatial (GPU ${GPU_ARRAY[0]})"
run_suite ${GPU_ARRAY[1]} libero_object  "${OUTPUT_DIR}/${OUTPUT_PREFIX}_object.txt"
echo "   [PID $!] libero_object (GPU ${GPU_ARRAY[1]})"
run_suite ${GPU_ARRAY[2]} libero_goal    "${OUTPUT_DIR}/${OUTPUT_PREFIX}_goal.txt"
echo "   [PID $!] libero_goal (GPU ${GPU_ARRAY[2]})"
run_suite ${GPU_ARRAY[3]} libero_10      "${OUTPUT_DIR}/${OUTPUT_PREFIX}_10.txt"
echo "   [PID $!] libero_10 (GPU ${GPU_ARRAY[3]})"

echo ""
echo "Waiting for all evaluations..."
wait

echo ""
echo "Done. Results:"
for suite in spatial object goal 10; do
    file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Total success rate|Route stats" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done