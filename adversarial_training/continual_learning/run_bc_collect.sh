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

# ── Filter demos ─────────────────────────────────────────────
if [[ "${SKIP_COLLECT:-0}" == "1" ]]; then
  echo "[run_bc] SKIP_COLLECT=1, reusing existing demos"
else
  echo "[run_bc] Step 1/2: collect_buffer_bc.py"
  python adversarial_training/continual_learning/collect_buffer_bc.py \
      --config "${CONFIG}"
fi
