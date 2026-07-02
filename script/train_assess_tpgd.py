#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.pipeline.assess_tpgd import AssessTPGDTrainConfig, train_assess_tpgd

DEFAULT_TRAIN_CONFIG_PATH = ROOT / "config" / "train" / "legacy" / "sample.yml"


def _get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _select_device(config: dict[str, Any]) -> str:
    device = str(config.get("device", "cuda"))
    if device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return "cuda"
        gpu_ids = config.get("gpu_ids")
        if isinstance(gpu_ids, list) and gpu_ids:
            return f"cuda:{gpu_ids[0]}"
        if isinstance(gpu_ids, int):
            return f"cuda:{gpu_ids}"
        if isinstance(gpu_ids, str) and gpu_ids.strip():
            return f"cuda:{gpu_ids.split(',')[0].strip()}"
    return device




def _slugify_name(value: Any) -> str:
    name = str(value or "run").strip() or "run"
    name = re.sub(r"[^0-9A-Za-z_.-]+", "-", name)
    return name.strip("-_.") or "run"


def _checkpoint_output_dir(path_opt: dict[str, Any], config: dict[str, Any]) -> Path:
    root = _resolve(path_opt.get("checkpoint_save"))
    if root is None:
        root = (ROOT / "outputs" / "assess_tpgd").resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (root / f"{timestamp}_{_slugify_name(config.get('name'))}").resolve()




def _t_max_value(value: Any) -> int | str:
    if value in (None, ""):
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return int(float(value))

def _list_ints(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [int(float(item)) for item in value]
    return [int(float(part.strip())) for part in str(value).split(",") if part.strip()]

def _resolve(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _as_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dataset_types(dataset_opt: dict[str, Any]) -> list[str]:
    for key in ("degradation", "dataset_type", "type"):
        values = _as_str_list(dataset_opt.get(key))
        if values:
            return values
    mode = dataset_opt.get("mode")
    if mode and str(mode) not in {"MD", "LQGT"}:
        return [str(mode)]
    return []


def _as_path_list(value: Any) -> list[Path]:
    if isinstance(value, list):
        return [resolved for item in value if item not in (None, "") and (resolved := _resolve(item)) is not None]
    resolved = _resolve(value)
    return [] if resolved is None else [resolved]


def _hidden_leaf(config: dict[str, Any], dataset_opt: dict[str, Any]) -> str:
    assessment_opt = config.get("assessment", {}) if isinstance(config.get("assessment"), dict) else {}
    if dataset_opt.get("hidden_leaf"):
        return str(dataset_opt["hidden_leaf"])
    max_new_tokens = int(assessment_opt.get("max_new_tokens", 256))
    output_dtype = str(assessment_opt.get("output_dtype", "bf16")).replace("float", "f")
    return f"features_condition_mnt{max_new_tokens}_{output_dtype}"


def _dataset_dirs(config: dict[str, Any], dataset_opt: dict[str, Any], kind: str, dataset_types: list[str]) -> list[Path]:
    direct_keys = {
        "lq": ("lq_dir", "dataroot_LQ"),
        "gt": ("gt_dir", "dataroot_GT"),
        "hidden": ("hidden_dir", "dataroot_hidden"),
    }[kind]
    for key in direct_keys:
        paths = _as_path_list(dataset_opt.get(key))
        if paths:
            return paths

    dataroot = _resolve(dataset_opt.get("dataroot"))
    if dataroot is None or not dataset_types:
        return []
    leaf = {"lq": "LQ", "gt": "GT", "hidden": _hidden_leaf(config, dataset_opt)}[kind]
    return [(dataroot / dtype / leaf).resolve() for dtype in dataset_types]


def _single_or_list(paths: list[Path]) -> Path | list[Path] | None:
    if not paths:
        return None
    return paths[0] if len(paths) == 1 else paths


def _bool_switch(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _degradation_prior_source(prior_switch: dict[str, Any]) -> str:
    value = prior_switch.get(
        "degradation_prior_source",
        prior_switch.get("deg_prior_source", prior_switch.get("degra_prior_source", "assessment_hidden")),
    )
    source = str(value).strip().lower().replace("-", "_")
    aliases = {
        "assessment": "assessment_hidden",
        "depictqa": "assessment_hidden",
        "depictqa_hidden": "assessment_hidden",
        "qa": "assessment_hidden",
        "qa_hidden": "assessment_hidden",
        "hidden": "assessment_hidden",
        "qwen": "qwen_prompt",
        "qwen_prompt": "qwen_prompt",
        "qwen_structured": "qwen_prompt",
        "structured_qwen": "qwen_prompt",
        "structured": "qwen_prompt",
        "a2": "qwen_probs",
        "qwen_probs": "qwen_probs",
        "calibrated_probs": "qwen_probs",
        "a3": "qwen_probs_severity",
        "qwen_probs_severity": "qwen_probs_severity",
        "calibrated_probs_severity": "qwen_probs_severity",
        "a4": "confidence_gate",
        "confidence": "confidence_gate",
        "confidence_gate": "confidence_gate",
        "zero": "zero_prior",
        "zero_prior": "zero_prior",
        "oracle": "oracle_type",
        "oracle_type": "oracle_type",
        "gt": "oracle_type",
        "tpgd_degradation": "tpgd",
        "degradation": "tpgd",
        "original": "tpgd",
        "off": "none",
        "false": "none",
        "null": "none",
    }
    source = aliases.get(source, source)
    if source not in {
        "assessment_hidden", "qwen_prompt", "qwen_probs",
        "qwen_probs_severity", "confidence_gate", "zero_prior", "oracle_type", "tpgd", "none",
    }:
        raise SystemExit(
            "prior_switch.degradation_prior_source must be one of "
            "assessment_hidden, qwen_prompt, qwen_probs, qwen_probs_severity, confidence_gate, "
            "zero_prior, oracle_type, tpgd, none"
        )
    return source


def _structured_prior_temperature(prior_switch: dict[str, Any]) -> float:
    """Resolve a frozen train-set temperature from scalar or JSON artifact."""
    calibration_value = prior_switch.get("calibration_path")
    if calibration_value not in (None, ""):
        calibration_path = _resolve(calibration_value)
        if calibration_path is None or not calibration_path.exists():
            raise SystemExit(f"prior_switch.calibration_path does not exist: {calibration_path}")
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("temperature") is None:
            raise SystemExit(f"Calibration JSON has no temperature: {calibration_path}")
        temperature = float(payload["temperature"])
    else:
        temperature = float(prior_switch.get("temperature", 1.0))
    if temperature <= 0:
        raise SystemExit(f"Structured-prior temperature must be positive, got {temperature}")
    return temperature


def _confidence_gate_options(config: dict[str, Any]) -> tuple[float, float, float | None]:
    options = config.get("confidence_gate", {}) or {}
    if not isinstance(options, dict):
        raise SystemExit("confidence_gate must be a YAML mapping")
    dropout = float(options.get("condition_dropout_probability", 0.0))
    corruption = float(options.get("prior_corruption_probability", 0.0))
    override_value = options.get("confidence_override")
    override = None if override_value in (None, "") else float(override_value)
    if not 0.0 <= dropout <= 1.0 or not 0.0 <= corruption <= 1.0:
        raise SystemExit("confidence gate probabilities must be in [0, 1]")
    if dropout + corruption > 1.0:
        raise SystemExit("condition dropout + prior corruption probabilities must not exceed 1")
    if override is not None and not 0.0 <= override <= 1.0:
        raise SystemExit("confidence_gate.confidence_override must be in [0, 1]")
    return dropout, corruption, override


def _inline_tpgd_options(config: dict[str, Any]) -> dict[str, Any]:
    inline: dict[str, Any] = {}
    for key in ("sde", "network_G", "structure_prior", "prior_switch", "degradation"):
        if key in config:
            inline[key] = config[key]

    prior_switch = config.get("prior_switch", {}) or {}
    if not isinstance(prior_switch, dict):
        prior_switch = {}
    source = _degradation_prior_source(prior_switch)
    use_content = _bool_switch(prior_switch.get("use_content_prior"), False)
    use_struct = _bool_switch(
        prior_switch.get("use_struct_prior", prior_switch.get("use_structure_prior")),
        False,
    )

    network = inline.setdefault("network_G", {})
    setting = network.setdefault("setting", {})
    setting["use_degra_context"] = source != "none"
    setting["use_image_context"] = use_content
    setting["use_struct_context"] = use_struct
    if use_struct:
        structure_setting = inline.get("structure_prior", {}).get("setting", {})
        if isinstance(structure_setting, dict) and "token_dim" in structure_setting:
            setting["struct_context_dim"] = int(structure_setting["token_dim"])
    return inline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    args = parser.parse_args()

    config_path = args.config.expanduser()
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {config_path}")

    dataset_opt = _get(cfg, "datasets.train", {}) or {}
    val_dataset_opt = _get(cfg, "datasets.val", _get(cfg, "datasets.validation", {})) or {}
    train_opt = cfg.get("train", {}) or {}
    logger_opt = cfg.get("logger", {}) or {}
    path_opt = cfg.get("path", {}) or {}
    scheduler_opt = train_opt.get("scheduler", {}) or {}
    if isinstance(scheduler_opt, str):
        scheduler_opt = {"type": scheduler_opt}
    eval_opt = cfg.get("evaluation", cfg.get("eval", {})) or {}
    train_sampling_opt = eval_opt.get("train_sampling", {}) if isinstance(eval_opt, dict) else {}
    if isinstance(train_sampling_opt, str):
        train_sampling_opt = {"mode": train_sampling_opt}
    if not isinstance(train_sampling_opt, dict):
        raise SystemExit("evaluation.train_sampling must be a mapping or mode string")
    warmup_opt = scheduler_opt.get("warmup", {}) if isinstance(scheduler_opt, dict) else {}
    if isinstance(warmup_opt, bool):
        warmup_opt = {"enabled": warmup_opt}
    if not isinstance(warmup_opt, dict):
        warmup_opt = {}

    dataset_types = _dataset_types(dataset_opt)
    lq_dirs = _dataset_dirs(cfg, dataset_opt, "lq", dataset_types)
    gt_dirs = _dataset_dirs(cfg, dataset_opt, "gt", dataset_types)
    hidden_dirs = _dataset_dirs(cfg, dataset_opt, "hidden", dataset_types)
    hidden_path = _resolve(dataset_opt.get("hidden_path"))
    val_dataset_types = _dataset_types(val_dataset_opt) if isinstance(val_dataset_opt, dict) else []
    val_lq_dirs = _dataset_dirs(cfg, val_dataset_opt, "lq", val_dataset_types) if isinstance(val_dataset_opt, dict) else []
    val_gt_dirs = _dataset_dirs(cfg, val_dataset_opt, "gt", val_dataset_types) if isinstance(val_dataset_opt, dict) else []
    val_hidden_dirs = _dataset_dirs(cfg, val_dataset_opt, "hidden", val_dataset_types) if isinstance(val_dataset_opt, dict) else []
    val_hidden_path = _resolve(val_dataset_opt.get("hidden_path")) if isinstance(val_dataset_opt, dict) else None
    output_dir = _checkpoint_output_dir(path_opt, cfg)
    tpgd_options = _resolve(path_opt.get("tpgd_options"))
    checkpoint = _resolve(path_opt.get("checkpoint_load"))
    prior_checkpoint = _resolve(path_opt.get("prior") or path_opt.get("prior_checkpoint"))
    prior_switch_opt = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch_opt, dict):
        prior_switch_opt = {}
    qwen_prompt_root = _resolve(
        dataset_opt.get("qwen_prompt_root")
        or dataset_opt.get("structured_prior_root")
        or prior_switch_opt.get("qwen_prompt_root")
        or prior_switch_opt.get("structured_prior_root")
    )
    val_qwen_prompt_root = (
        _resolve(val_dataset_opt.get("qwen_prompt_root") or val_dataset_opt.get("structured_prior_root"))
        if isinstance(val_dataset_opt, dict)
        else None
    )
    degradation_prior_source = _degradation_prior_source(prior_switch_opt)
    structured_prior_temperature = _structured_prior_temperature(prior_switch_opt)
    condition_dropout_probability, prior_corruption_probability, confidence_override = _confidence_gate_options(cfg)

    required = {"datasets.train.lq_dir": lq_dirs, "datasets.train.gt_dir": gt_dirs, "path.checkpoint_save": output_dir}
    missing = [key for key, value in required.items() if value is None or value == []]
    if missing:
        raise SystemExit(f"Missing required train.yml settings: {missing}")
    if len(lq_dirs) != len(gt_dirs):
        raise SystemExit(f"datasets.train lq/gt directory counts differ: {len(lq_dirs)} vs {len(gt_dirs)}")
    if degradation_prior_source == "assessment_hidden" and hidden_path is None and len(hidden_dirs) != len(lq_dirs):
        raise SystemExit(f"datasets.train hidden directory count differs from lq count: {len(hidden_dirs)} vs {len(lq_dirs)}")
    has_val = bool(val_lq_dirs or val_gt_dirs)
    if has_val and len(val_lq_dirs) != len(val_gt_dirs):
        raise SystemExit(f"datasets.val lq/gt directory counts differ: {len(val_lq_dirs)} vs {len(val_gt_dirs)}")
    if has_val and degradation_prior_source == "assessment_hidden" and val_hidden_path is None and len(val_hidden_dirs) != len(val_lq_dirs):
        raise SystemExit(f"datasets.val hidden directory count differs from lq count: {len(val_hidden_dirs)} vs {len(val_lq_dirs)}")

    train_cfg = AssessTPGDTrainConfig(
        lq_dir=_single_or_list(lq_dirs),
        gt_dir=_single_or_list(gt_dirs),
        hidden_dir=_single_or_list(hidden_dirs),
        hidden_path=hidden_path,
        output_dir=output_dir,
        tpgd_options=tpgd_options,
        val_lq_dir=_single_or_list(val_lq_dirs),
        val_gt_dir=_single_or_list(val_gt_dirs),
        val_hidden_dir=_single_or_list(val_hidden_dirs),
        val_hidden_path=val_hidden_path,
        structured_prior_root=qwen_prompt_root,
        val_structured_prior_root=val_qwen_prompt_root,
        structured_prior_temperature=structured_prior_temperature,
        condition_dropout_probability=condition_dropout_probability,
        prior_corruption_probability=prior_corruption_probability,
        structured_confidence_override=confidence_override,
        tpgd_checkpoint=checkpoint,
        tpgd_inline_options=_inline_tpgd_options(cfg),
        hidden_key=str(dataset_opt.get("hidden_key", "condition_hidden")),
        image_size=_as_int(dataset_opt.get("image_size", dataset_opt.get("patch_size")), 128),
        batch_size=_as_int(dataset_opt.get("batch_size"), 1),
        epochs=_as_int(train_opt.get("epochs"), 1),
        max_steps=_as_int(train_opt.get("max_steps", train_opt.get("niter")), 100),
        lr=_as_float(train_opt.get("lr_G", train_opt.get("lr")), 1e-4),
        optimizer=str(train_opt.get("optimizer", "AdamW")),
        beta1=_as_float(train_opt.get("beta1"), 0.9),
        beta2=_as_float(train_opt.get("beta2"), 0.999),
        weight_decay=_as_float(train_opt.get("weight_decay_G", train_opt.get("weight_decay")), 0.0),
        num_workers=_as_int(dataset_opt.get("n_workers", dataset_opt.get("num_workers")), 0),
        device=_select_device(cfg),
        train_backbone=bool(train_opt.get("train_backbone", False)),
        load_checkpoint=bool(path_opt.get("load_checkpoint", True)),
        strict_load=bool(path_opt.get("strict_load", False)),
        adapter_hidden_dim=_as_int(train_opt.get("adapter_hidden_dim"), 1024),
        adapter_pool=str(train_opt.get("adapter_pool", "mean")),
        adapter_dropout=_as_float(train_opt.get("adapter_dropout"), 0.0),
        random_content_context=bool(train_opt.get("random_content_context", False)),
        seed=_as_int(
            train_opt.get("manual_seed", train_opt.get("seed", cfg.get("manual_seed", cfg.get("seed")))),
            1234,
        ),
        degradation_prior_source=degradation_prior_source,
        use_content_prior=_bool_switch(prior_switch_opt.get("use_content_prior"), False),
        use_structure_prior=_bool_switch(prior_switch_opt.get("use_struct_prior", prior_switch_opt.get("use_structure_prior")), False),
        train_structure_prior=bool(train_opt.get("train_structure_prior", False)),
        prior_checkpoint=prior_checkpoint,
        objective=str(train_opt.get("objective", "sde")),
        prediction_target=str(train_opt.get("prediction_target", "image")),
        loss_type=str(train_opt.get("loss_type", "l1")),
        loss_weight=_as_float(train_opt.get("weight", train_opt.get("loss_weight")), 1.0),
        boundary_pad=_as_int(
            train_opt.get("boundary_pad", cfg.get("boundary_pad", _get(cfg, "fusion.boundary_pad", _get(cfg, "runtime.boundary_pad")))),
            32,
        ),
        direct_gt_time=_as_float(train_opt.get("direct_gt_time", train_opt.get("direct_time")), 0.0),
        sde_t_start=_as_int(train_opt.get("sde_t_start"), 1),
        sde_t_end=_as_int(train_opt.get("sde_t_end"), -1),
        save_every=_as_int(logger_opt.get("save_checkpoint_freq"), 100),
        log_every=_as_int(logger_opt.get("print_freq"), 10),
        save_full_model=bool(train_opt.get("save_full_model", False)),
        lr_scheduler=str(scheduler_opt.get("type", train_opt.get("lr_scheduler", "none"))),
        lr_min=_as_float(scheduler_opt.get("min_lr", scheduler_opt.get("eta_min", train_opt.get("lr_min"))), 0.0),
        lr_step_size=_as_int(scheduler_opt.get("step_size", train_opt.get("lr_step_size")), 10000),
        lr_gamma=_as_float(scheduler_opt.get("gamma", train_opt.get("lr_gamma")), 0.5),
        lr_milestones=_list_ints(scheduler_opt.get("milestones", train_opt.get("lr_milestones"))),
        lr_t_max=_t_max_value(scheduler_opt.get("t_max", scheduler_opt.get("T_max", train_opt.get("lr_t_max", "auto")))),
        lr_warmup_enabled=_bool_switch(warmup_opt.get("enabled", scheduler_opt.get("warmup_enabled", train_opt.get("lr_warmup_enabled"))), False),
        lr_warmup_steps=_as_int(warmup_opt.get("steps", scheduler_opt.get("warmup_steps", train_opt.get("lr_warmup_steps"))), 0),
        lr_warmup_start_factor=_as_float(warmup_opt.get("start_factor", scheduler_opt.get("warmup_start_factor", train_opt.get("lr_warmup_start_factor"))), 0.01),
        eval_enabled=_bool_switch(eval_opt.get("enabled"), False),
        eval_every_steps=_as_int(eval_opt.get("every_steps", eval_opt.get("step_freq")), 0),
        eval_every_epochs=_as_int(eval_opt.get("every_epochs", eval_opt.get("epoch_freq")), 0),
        eval_batch_size=_as_int(eval_opt.get("batch_size"), 0),
        eval_train_max_batches=_as_int(eval_opt.get("train_max_batches"), 10),
        eval_val_max_batches=_as_int(eval_opt.get("val_max_batches"), 0),
        eval_train_sampling_mode=str(train_sampling_opt.get("mode", "none")),
        eval_train_fraction_per_task=_as_float(train_sampling_opt.get("fraction_per_task"), 0.0),
        eval_train_sampling_seed=_as_int(
            train_sampling_opt.get("seed"),
            _as_int(train_opt.get("manual_seed", train_opt.get("seed")), 1234),
        ),
        use_wandb=bool(cfg.get("use_wandb", False)),
        wandb_project=str(cfg.get("wandb_project", "MyFusion_AssessTPGD")),
        wandb_run_name=str(cfg.get("name")) if cfg.get("name") else None,
    )
    train_assess_tpgd(train_cfg)


if __name__ == "__main__":
    main()
