#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QWEN_IR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${QWEN_IR_TEST_CONFIG:-${QWEN_IR_CONFIG:-$PROJECT_ROOT/config/infer_qwen_prompt.yml}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

_cfg_get() {
  "$PYTHON_BIN" -c 'import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
value = data
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        value = default
        break
    value = value[part]
if value is None:
    value = default
elif isinstance(value, bool):
    value = str(value).lower()
elif isinstance(value, (list, tuple)):
    value = ",".join(str(item) for item in value)
print(value)' "$CONFIG_FILE" "$1" "$2"
}

_is_set() {
  case "${1:-}" in
    ""|"~"|"null"|"None") return 1 ;;
    *) return 0 ;;
  esac
}

_append_opt() {
  local key="$1"
  local flag="$2"
  local value
  value="$(_cfg_get "$key" "")"
  if _is_set "$value"; then
    RUN_ARGS+=("$flag" "$value")
  fi
}

PYTHON_BIN="$(_cfg_get commands.python "$PYTHON_BIN")"
export CUDA_VISIBLE_DEVICES="$(_cfg_get gpu_ids "$(_cfg_get runtime.cuda_visible_devices 1)")"
export TOKENIZERS_PARALLELISM="$(_cfg_get runtime.tokenizers_parallelism false)"

cd "$PROJECT_ROOT"
MODE="$(_cfg_get run.mode assess_batch)"
case "$MODE" in
  batch|batch_test|assess|assess_batch|test)
    RUN_ARGS=(--config "$CONFIG_FILE")
    _append_opt run.checkpoint --checkpoint
    _append_opt run.split --split
    _append_opt run.batch_size --batch-size
    _append_opt run.max_batches --max-batches
    _append_opt run.device --device
    _append_opt run.output_json --output-json
    "$PYTHON_BIN" script/test_assess_tpgd.py "${RUN_ARGS[@]}" "$@"
    ;;
  qwen|qwen_prompt|prompt)
    RUN_ARGS=(--config "$CONFIG_FILE")
    _append_opt run.checkpoint --checkpoint
    _append_opt run.device --device
    "$PYTHON_BIN" script/run_qwen_prompt_infer.py "${RUN_ARGS[@]}" "$@"
    ;;
  *)
    echo "Unknown test config run.mode: $MODE" >&2
    echo "Expected one of: assess_batch, qwen_prompt" >&2
    exit 2
    ;;
esac
