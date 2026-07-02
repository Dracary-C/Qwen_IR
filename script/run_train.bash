#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QWEN_IR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${QWEN_IR_TRAIN_CONFIG:-${QWEN_IR_CONFIG:-$PROJECT_ROOT/config/train/legacy/sample.yml}}"
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

PYTHON_BIN="$(_cfg_get commands.python "$PYTHON_BIN")"
export CUDA_VISIBLE_DEVICES="$(_cfg_get gpu_ids "$(_cfg_get runtime.cuda_visible_devices 0)")"
export TOKENIZERS_PARALLELISM="$(_cfg_get runtime.tokenizers_parallelism false)"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" script/train_assess_tpgd.py --config "$CONFIG_FILE" "$@"
