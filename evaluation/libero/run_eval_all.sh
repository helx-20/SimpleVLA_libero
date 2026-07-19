#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script (parallel 4 task suites, local in-process model)
# =============================================================================

set -e

# =============================================================================
# LIBERO Environment Setup
# =============================================================================
SCRIPT_DIR="/mnt/hlx"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

# -----------------------------------------------------------------------------
# Arguments
#   $1  CHECKPOINT      path or HF id of the SimVLA checkpoint
#   $2  NUM_TRIALS      episodes per task (default: 50)
#   $3  OUTPUT_PREFIX   log/video output prefix (default: eval_simvla)
#   $4  GPUS            quoted, space-separated GPU ids (default: "0 1 2 3")
#
# Env overrides (optional):
#   NORM_STATS     path to norm_stats JSON (default: <repo>/code/norm_stats/libero_norm.json)
#   SMOLVLM_MODEL  SmolVLM backbone path or HF id
#                  (default: HuggingFaceTB/SmolVLM-500M-Instruct)
# -----------------------------------------------------------------------------
# Defaults mirror adversarial_training/configs/default.yaml (collect.policy).
CHECKPOINT=${1:-"/mnt/hlx/SimpleVLA_libero_data/runs/continual_ppo_round1_new/last"}
NUM_TRIALS=${2:-50}
OUTPUT_PREFIX=${3:-"eval_simvla"}
GPUS=${4:-"4 5 6 7"}
# Output directory
OUTPUT_DIR="./eval_simvla_last"

# Default norm stats: code/norm_stats/libero_norm.json (script lives in code/evaluation/libero/)
NORM_STATS=${NORM_STATS:-"${SCRIPT_DIR}/SimpleVLA_libero/norm_stats/libero_norm.json"}
SMOLVLM_MODEL=${SMOLVLM_MODEL:-"/mnt/hlx/SimpleVLA_libero_data/models/SmolVLM-500M-Instruct"}

# Parse GPU list
read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: Need at least 4 GPUs, got ${#GPU_ARRAY[@]}"
    echo "   Usage: $0 <checkpoint> [num_trials] [output_prefix] \"<gpu1> <gpu2> <gpu3> <gpu4>\""
    exit 1
fi

GPU_SPATIAL=${GPU_ARRAY[0]}
GPU_OBJECT=${GPU_ARRAY[1]}
GPU_GOAL=${GPU_ARRAY[2]}
GPU_10=${GPU_ARRAY[3]}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation (local model)..."
echo "   Checkpoint:    $CHECKPOINT"
echo "   Norm stats:    $NORM_STATS"
echo "   SmolVLM model: $SMOLVLM_MODEL"
echo "   Num Trials:    $NUM_TRIALS"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir:    $OUTPUT_DIR"
echo "   GPUs: spatial=$GPU_SPATIAL, object=$GPU_OBJECT, goal=$GPU_GOAL, 10=$GPU_10"
echo ""


run_suite () {
    local gpu=$1
    local suite=$2
    local log=$3
    CUDA_VISIBLE_DEVICES=$gpu python -u libero_client.py \
        --client_type local \
        --checkpoint "$CHECKPOINT" \
        --norm_stats "$NORM_STATS" \
        --smolvlm_model "$SMOLVLM_MODEL" \
        --task_suite "$suite" \
        --num_trials $NUM_TRIALS \
        --video_out "$OUTPUT_DIR" > "$log" 2>&1 &
}

# Run 4 task suites in parallel
echo "Launching 4 evaluation tasks..."

run_suite $GPU_SPATIAL libero_spatial "${OUTPUT_DIR}/${OUTPUT_PREFIX}_spatial.txt"
PID_SPATIAL=$!
echo "   [PID $PID_SPATIAL] libero_spatial (GPU $GPU_SPATIAL) -> ${OUTPUT_PREFIX}_spatial.txt"

run_suite $GPU_OBJECT libero_object "${OUTPUT_DIR}/${OUTPUT_PREFIX}_object.txt"
PID_OBJECT=$!
echo "   [PID $PID_OBJECT] libero_object (GPU $GPU_OBJECT) -> ${OUTPUT_PREFIX}_object.txt"

run_suite $GPU_GOAL libero_goal "${OUTPUT_DIR}/${OUTPUT_PREFIX}_goal.txt"
PID_GOAL=$!
echo "   [PID $PID_GOAL] libero_goal (GPU $GPU_GOAL) -> ${OUTPUT_PREFIX}_goal.txt"

run_suite $GPU_10 libero_10 "${OUTPUT_DIR}/${OUTPUT_PREFIX}_10.txt"
PID_10=$!
echo "   [PID $PID_10] libero_10 (GPU $GPU_10) -> ${OUTPUT_PREFIX}_10.txt"

echo ""
echo "Waiting for all evaluations to complete..."
echo "   Monitor progress with: tail -f ${OUTPUT_PREFIX}_*.txt"
echo ""

# Wait for all tasks
wait $PID_SPATIAL $PID_OBJECT $PID_GOAL $PID_10

echo ""
echo "All evaluations completed!"
echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Success Rate|Average|Total success" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="
