#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QWEN_IR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${QWEN_IR_GPU:-0}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${QWEN_IR_SEQUENCE_LOG_DIR:-$PROJECT_ROOT/log/train_runs/R1_R2_R3_$RUN_STAMP}"

CONFIGS=(
  "config/train/R/R1.yml"
  "config/train/R/R2.yml"
  "config/train/R/R3.yml"
)

if (( $# > 0 )); then
  CONFIGS=("$@")
fi

mkdir -p "$RUN_ROOT"
QUEUE_LOG="$RUN_ROOT/sequence.log"

exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "sequence_start=$(date --iso-8601=seconds)"
echo "gpu_id=$GPU_ID"
echo "run_root=$RUN_ROOT"

for config in "${CONFIGS[@]}"; do
  if [[ "$config" != /* ]]; then
    config="$PROJECT_ROOT/$config"
  fi
  if [[ ! -f "$config" ]]; then
    echo "sequence_error=missing_config config=$config"
    exit 2
  fi

  run_name="$(basename "${config%.yml}")"
  run_log="$RUN_ROOT/${run_name}.log"
  echo "run_start=$(date --iso-8601=seconds) name=$run_name config=$config"
  echo "run_log=$run_log"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PROJECT_ROOT/script/train_assess_tpgd.py" --config "$config" 2>&1 | tee "$run_log"
  status=${PIPESTATUS[0]}
  set -e

  if (( status != 0 )); then
    echo "run_failed=$(date --iso-8601=seconds) name=$run_name exit_code=$status"
    exit "$status"
  fi
  echo "run_complete=$(date --iso-8601=seconds) name=$run_name"
done

echo "sequence_complete=$(date --iso-8601=seconds)"
