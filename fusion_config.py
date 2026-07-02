from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config" / "infer_qwen_prompt.yml"

DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "tokenizers_parallelism": False,
    },
    "run": {
        "mode": "qwen_prompt",
        "checkpoint": None,
        "device": None,
    },
    "fusion": {
        "resize": 256,
        "device": "cuda",
        "sampling_mode": "posterior",
        "image_range": "minus_one_one",
    },
    "paths": {
        "input": str(APP_DIR / "demo_sample" / "lowlight1.png"),
        "qwen_prompt_output_dir": str(APP_DIR / "log" / "qwen_prompt_run"),
        "tpgd_options": str(APP_DIR / "module" / "vendor" / "tpgdiff" / "universal-restoration" / "config" / "tpgd-sde" / "options" / "test_fast.yml"),
        "tpgd_checkpoint": "/data/chenzt/model_weights/tpgd/universal/ablation-d1-c1-s1/latest_G.pth",
        "tpgd_prior": "/data/chenzt/model_weights/tpgd/prior/tpgd_ViT-B-32.pt",
        "assess_tpgd_checkpoint": "/data/chenzt/checkpoints/20260612_050008_structured-prior-sample-qwen-v1-continue10/latest.pt",
    },
    "commands": {
        "python": sys.executable,
    },
    "prior_switch": {
        "degradation_prior_source": "qwen_prompt",
        "use_content_prior": False,
        "use_struct_prior": True,
    },
    "path": {
        "strict_load": False,
    },
    "test": {
        "adapter_hidden_dim": 1024,
        "adapter_pool": "mean",
        "adapter_dropout": 0.0,
    },
}



def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
        _deep_update(config, loaded)
    config["_config_path"] = str(config_path)
    return config


def config_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def config_path(config: dict[str, Any], dotted_key: str, default: str | Path | None = None) -> Path | None:
    value = config_get(config, dotted_key, default)
    if value in (None, ""):
        return None
    return Path(value).expanduser()
