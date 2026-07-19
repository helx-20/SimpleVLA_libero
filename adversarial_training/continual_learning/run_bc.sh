#!/usr/bin/env bash
# BC continual-learning pipeline.
#
#   1. collect_buffer_bc.py — filter LIBERO official demos by criticality.
#   2. bc_offline.py — fine-tune SimpleVLA (same training as train_smolvlm.py).
#
# Usage:
#   bash run_bc.sh [config] [resume_ckpt] [output_dir]
#   SKIP_COLLECT=1 bash run_bc.sh ...      # reuse existing filtered demos

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

[[ -z "${RESUME_CKPT}" ]] && RESUME_CKPT="$(read_yaml "bc.resume_ckpt")"
[[ -z "${OUTPUT_DIR}"  ]] && OUTPUT_DIR="$(read_yaml "bc.output_dir")"

BC_META="$(read_yaml "bc.bc_meta")"
if [[ -z "${BC_META}" ]]; then
  BC_META="$(read_yaml "bc.output_dir")/bc_train_meta.json"
fi

BATCH_SIZE="$(read_yaml "bc.batch_size")"
LR="$(read_yaml "bc.learning_rate")"
ITERS="$(read_yaml "bc.iters")"
SMOLVLM_MODEL="$(read_yaml "bc.smolvlm_model")"
[[ -z "${SMOLVLM_MODEL}" ]] && SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

NORM_STATS="$(read_yaml "bc.norm_stats")"
[[ -z "${NORM_STATS}" ]] && NORM_STATS="$(read_yaml "collect.policy.norm_stats")"
[[ -z "${NORM_STATS}" ]] && NORM_STATS="./norm_stats/libero_norm.json"

echo "============================================================"
echo "[run_bc] BC Continual Learning"
echo "============================================================"
echo "  config:       ${CONFIG}"
echo "  resume_ckpt:  ${RESUME_CKPT}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  bc_meta:      ${BC_META}"
echo "  norm_stats:   ${NORM_STATS}"
echo "  iters:        ${ITERS:-200000}"
echo "  batch_size:   ${BATCH_SIZE:-32}"
echo "  lr:           ${LR:-1e-4}"
echo "============================================================"

# ── 1) Filter demos ─────────────────────────────────────────────
if [[ "${SKIP_COLLECT:-0}" == "1" ]]; then
  echo "[run_bc] SKIP_COLLECT=1, reusing existing demos"
else
  echo "[run_bc] Step 1/2: collect_buffer_bc.py"
  python adversarial_training/continual_learning/collect_buffer_bc.py \
      --config "${CONFIG}"
fi

# ── 2) BC training ──────────────────────────────────────────────
echo "[run_bc] Step 2/2: bc_offline.py"

# Resolve bc_meta path — it might have been generated in step 1.
if [[ ! -f "${BC_META}" ]]; then
  BC_OUT_DIR="$(read_yaml "bc.output_dir")"
  if [[ -f "${BC_OUT_DIR}/bc_train_meta.json" ]]; then
    BC_META="${BC_OUT_DIR}/bc_train_meta.json"
  fi
fi

if [[ ! -f "${BC_META}" ]]; then
  echo "[run_bc] ERROR: bc_train_meta.json not found at ${BC_META}"
  echo "  Make sure collect_buffer_bc.py ran successfully (or set SKIP_COLLECT=1)."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Build CLI args for bc_offline.py
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

# Multi-GPU via accelerate.
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-$(echo "${GPUS}" | awk -F, '{print NF}')}"

echo "[run_bc] Launching bc_offline.py on GPUs ${GPUS} (${NUM_PROCESSES} processes)"
CUDA_VISIBLE_DEVICES="${GPUS}" \
accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    adversarial_training/continual_learning/bc_offline.py \
    "${BC_ARGS[@]}"

echo "[run_bc] Done. Checkpoints in ${OUTPUT_DIR}"
