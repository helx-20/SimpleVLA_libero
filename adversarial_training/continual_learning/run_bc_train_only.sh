#!/usr/bin/env bash
# BC training only (assumes collect_buffer_bc.py has already run).
#
# Usage:
#   bash run_bc_train_only.sh [config] [resume_ckpt] [output_dir]

set -euo pipefail

CONFIG="${1:-adversarial_training/configs/default.yaml}"
RESUME_CKPT="${2:-}"
OUTPUT_DIR="${3:-}"

read_yaml() {
  python - "$CONFIG" "$1" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
node = cfg
for p in sys.argv[2].split("."):
    if not isinstance(node, dict):
        node = ""
        break
    node = node.get(p, "")
print(node if node is not None else "")
PY
}

[[ -z "${RESUME_CKPT}" ]] && RESUME_CKPT="$(read_yaml "bc_training.resume_ckpt")"
[[ -z "${OUTPUT_DIR}"  ]] && OUTPUT_DIR="$(read_yaml "bc_training.output_dir")"

BC_META="$(read_yaml "bc_training.bc_meta")"
if [[ -z "${BC_META}" ]]; then
  BC_META="$(read_yaml "bc_collect.output_dir")/bc_train_meta.json"
fi

BATCH_SIZE="$(read_yaml "bc_training.batch_size")"
LR="$(read_yaml "bc_training.learning_rate")"
ITERS="$(read_yaml "bc_training.iters")"

SMOLVLM_MODEL="$(read_yaml "bc_training.smolvlm_model")"
[[ -z "${SMOLVLM_MODEL}" ]] && SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

NORM_STATS="$(read_yaml "bc_training.norm_stats")"
[[ -z "${NORM_STATS}" ]] && NORM_STATS="$(read_yaml "bc_collect.norm_stats")"
[[ -z "${NORM_STATS}" ]] && NORM_STATS="./norm_stats/libero_norm.json"

echo "============================================================"
echo "[run_bc_train_only] BC Training"
echo "============================================================"
echo "  config:       ${CONFIG}"
echo "  resume_ckpt:  ${RESUME_CKPT}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  bc_meta:      ${BC_META}"
echo "  norm_stats:   ${NORM_STATS}"
echo "  iters:        ${ITERS:-200000}"
echo "  batch_size:   ${BATCH_SIZE:-16}"
echo "  lr:           ${LR:-1e-4}"
echo "============================================================"

if [[ ! -f "${BC_META}" ]]; then
  echo "[run_bc_train_only] ERROR: bc_train_meta.json not found at ${BC_META}"
  echo "  Run collect_buffer_bc.py first."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

BC_ARGS=(
  --config "${CONFIG}"
  --bc_meta "${BC_META}"
  --output_dir "${OUTPUT_DIR}"
  --norm_stats_path "${NORM_STATS}"
  --smolvlm_model_path "${SMOLVLM_MODEL}"
)

[[ -n "${RESUME_CKPT}" ]] && BC_ARGS+=(--models "${RESUME_CKPT}")
[[ -n "${ITERS}"       ]] && BC_ARGS+=(--iters "${ITERS}")
[[ -n "${BATCH_SIZE}"  ]] && BC_ARGS+=(--batch_size "${BATCH_SIZE}")
[[ -n "${LR}"          ]] && BC_ARGS+=(--learning_rate "${LR}")

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-$(echo "${GPUS}" | awk -F, '{print NF}')}"

echo "[run_bc_train_only] GPUs=${GPUS}  processes=${NUM_PROCESSES}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 25107 \
    adversarial_training/continual_learning/bc_offline.py \
    "${BC_ARGS[@]}"

echo "[run_bc_train_only] Done. Checkpoints in ${OUTPUT_DIR}"
