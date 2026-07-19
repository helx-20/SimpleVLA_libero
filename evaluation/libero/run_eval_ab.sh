#!/bin/bash
# =============================================================================
# SimVLA LIBERO A/B Comparison Evaluation
#   FT: /mnt/hlx/SimpleVLA_libero_data/runs/bc_continual_v4/ckpt-15000
#   Base: /mnt/hlx/SimpleVLA_libero_data/models/SimVLA-LIBERO
#   Threshold=0: all episodes treated as "hard" → both models run on every ep
#
# GPU allocation (8 GPUs total):
#   libero_10:   4 GPUs (0,1,2,3) — tasks split 0:3, 3:6, 6:8, 8:10
#   libero_goal: 2 GPUs (4,5)     — tasks split 0:5, 5:10
#   libero_spatial: 1 GPU (6)
#   libero_object:  1 GPU (7)
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LIBERO_ROOT=/mnt/hlx/LIBERO
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

BASE_CKPT="/mnt/hlx/SimpleVLA_libero_data/models/SimVLA-LIBERO"
FT_CKPT="/mnt/hlx/SimpleVLA_libero_data/runs/bc_continual_v4/ckpt-15000"
CRIT_CKPT="/mnt/hlx/SimpleVLA_libero_data/runs/criticality/best.pt"
NORM_STATS="/mnt/hlx/SimpleVLA_libero/norm_stats/libero_norm.json"
SMOLVLM="/mnt/hlx/SimpleVLA_libero_data/models/SmolVLM-500M-Instruct"
OUTPUT_DIR="${SCRIPT_DIR}/eval_ab"
NUM_TRIALS=50
THRESHOLD=-1

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "LIBERO A/B Comparison Evaluation"
echo "=============================================================================="
echo "  Base model:    $BASE_CKPT"
echo "  FT model:      $FT_CKPT"
echo "  Criticality:   $CRIT_CKPT"
echo "  Threshold:     $THRESHOLD (all episodes → both models)"
echo "  Norm stats:    $NORM_STATS"
echo "  SmolVLM:       $SMOLVLM"
echo "  Trials:        $NUM_TRIALS"
echo "  Output:        $OUTPUT_DIR"
echo "=============================================================================="
echo ""

# ---------------------------------------------------------------------------
# Helper: launch one evaluation process
# ---------------------------------------------------------------------------
run_ab() {
    local gpu=$1
    local suite=$2
    local t_start=$3
    local t_end=$4
    local log="$OUTPUT_DIR/${suite}_gpu${gpu}_t${t_start}-${t_end}.log"

    echo "  [GPU $gpu] $suite tasks [$t_start, $t_end) -> $(basename $log)"

    CUDA_VISIBLE_DEVICES=$gpu python -u libero_client.py \
        --client_type local \
        --checkpoint "$BASE_CKPT" \
        --ft_checkpoint "$FT_CKPT" \
        --norm_stats "$NORM_STATS" \
        --smolvlm_model "$SMOLVLM" \
        --criticality_ckpt "$CRIT_CKPT" \
        --criticality_threshold $THRESHOLD \
        --ab_compare \
        --task_suite "$suite" \
        --num_trials $NUM_TRIALS \
        --task_start $t_start \
        --task_end $t_end \
        --video_out "$OUTPUT_DIR" \
        --no_video \
        > "$log" 2>&1 &
}

# ---------------------------------------------------------------------------
# Launch all 8 processes
# ---------------------------------------------------------------------------
echo "Launching A/B evaluations..."

# --- libero_10: 4 GPUs (0,1,2,3), tasks split 0:3, 3:6, 6:8, 8:10 ---
run_ab 0 "libero_10" 0 3
PID_10_0=$!
run_ab 1 "libero_10" 3 6
PID_10_1=$!
run_ab 2 "libero_10" 6 8
PID_10_2=$!
run_ab 3 "libero_10" 8 10
PID_10_3=$!

# --- libero_goal: 2 GPUs (4,5), tasks split 0:5, 5:10 ---
run_ab 4 "libero_goal" 0 5
PID_GOAL_0=$!
run_ab 5 "libero_goal" 5 10
PID_GOAL_1=$!

# --- libero_spatial: 1 GPU (6) ---
run_ab 6 "libero_spatial" 0 10
PID_SPATIAL=$!

# --- libero_object: 1 GPU (7) ---
run_ab 7 "libero_object" 0 10
PID_OBJECT=$!

echo ""
echo "All 8 processes launched. PIDs:"
echo "  libero_10:    $PID_10_0 $PID_10_1 $PID_10_2 $PID_10_3"
echo "  libero_goal:  $PID_GOAL_0 $PID_GOAL_1"
echo "  libero_spatial: $PID_SPATIAL"
echo "  libero_object:  $PID_OBJECT"
echo ""
echo "Monitor with: tail -f $OUTPUT_DIR/*.log"
echo ""

# ---------------------------------------------------------------------------
# Wait for all
# ---------------------------------------------------------------------------
wait $PID_10_0 $PID_10_1 $PID_10_2 $PID_10_3 \
     $PID_GOAL_0 $PID_GOAL_1 \
     $PID_SPATIAL $PID_OBJECT

echo ""
echo "=============================================================================="
echo "All evaluations completed!"
echo "=============================================================================="

# ---------------------------------------------------------------------------
# Print per-suite summaries
# ---------------------------------------------------------------------------
for suite in libero_10 libero_goal libero_spatial libero_object; do
    echo ""
    echo "--- $suite ---"
    # Extract the A/B comparison table from each log
    for log in "$OUTPUT_DIR"/${suite}_gpu*.log; do
        if [ -f "$log" ]; then
            echo "  [$(basename $log)]"
            grep -A 20 "A/B Comparison Results" "$log" 2>/dev/null | head -15 || true
        fi
    done
done

echo ""
echo "Full logs: $OUTPUT_DIR/"
echo "=============================================================================="
