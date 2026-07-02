#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "script"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_assess_tpgd as train_entry
from module.pipeline.assess_tpgd import AssessTPGDTrainConfig, test_assess_tpgd


def _path_opt(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("path", {}) or {}
    if not isinstance(value, dict):
        raise SystemExit("batch test config path must be a YAML mapping")
    return value


def _split_dataset_opt(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    datasets = cfg.get("datasets", {}) or {}
    if split == "train":
        opt = datasets.get("train", {}) or {}
    elif split in {"val", "validation"}:
        opt = datasets.get("val", datasets.get("validation", {})) or {}
    else:
        opt = datasets.get(split, {}) or {}
    if not isinstance(opt, dict) or not opt:
        raise SystemExit(f"datasets.{split} is not configured in the batch test config")
    return opt


def _test_opt(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("test", {}) or {}
    if not isinstance(value, dict):
        raise SystemExit("batch test config test must be a YAML mapping")
    return value


def _checkpoint_path(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    value = _path_opt(cfg).get("checkpoint_load")
    if value in (None, ""):
        raise SystemExit("Set path.checkpoint_load in the batch test config or pass --checkpoint")
    checkpoint = train_entry._resolve(value)
    if checkpoint is None or not checkpoint.exists():
        raise SystemExit(f"checkpoint_load does not exist: {checkpoint}")
    return checkpoint


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_QWEN_PROMPT_CONFIG = ROOT / "config" / "tpgdiff_fewshot_qwen3vl.yml"
DEFAULT_QWEN_PROMPT_EXPORT = ROOT / "module" / "qwen" / "export_tpgdiff_qwen_structured_dataset.py"


def _bool_opt(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_name(value: Any) -> str:
    text = str(value or "run").strip().replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text).strip("-._") or "run"


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _dataset_root_for_lq_dir(lq_dir: Path) -> Path:
    return lq_dir.parents[2] if lq_dir.name.lower() == "lq" and len(lq_dir.parents) >= 3 else lq_dir.parent


def _effective_batch_size(cfg: dict[str, Any], dataset_opt: dict[str, Any], batch_size: int | None) -> int:
    train_dataset_opt = cfg.get("datasets", {}).get("train", {}) or {}
    test_opt = _test_opt(cfg)
    return int(batch_size or test_opt.get("batch_size") or dataset_opt.get("batch_size") or train_dataset_opt.get("batch_size") or 1)


def _collect_qwen_prompt_targets(
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int | None,
    max_batches: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    dataset_opt = _split_dataset_opt(cfg, split)
    dataset_types = train_entry._dataset_types(dataset_opt)
    lq_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "lq", dataset_types)
    if not lq_dirs:
        raise SystemExit(f"datasets.{split} lq dirs are missing; cannot generate qwen_prompt priors online")

    targets: list[dict[str, Any]] = []
    roots: list[Path] = []
    for lq_dir in lq_dirs:
        dataset_root = _dataset_root_for_lq_dir(lq_dir).resolve()
        if dataset_root not in roots:
            roots.append(dataset_root)
        for image in _image_files(lq_dir):
            rel = image.resolve().relative_to(dataset_root)
            rel_parts = rel.parts
            split_name = rel_parts[0] if len(rel_parts) > 0 else split
            dataset_name = rel_parts[1] if len(rel_parts) > 1 else lq_dir.parent.name
            targets.append(
                {
                    "split": split_name,
                    "dataset": dataset_name,
                    "image": str(image.resolve()),
                    "relative_image": str(rel),
                }
            )

    if max_batches and max_batches > 0:
        limit = max_batches * _effective_batch_size(cfg, dataset_opt, batch_size)
        targets = targets[:limit]
    return targets, roots


def prepare_qwen_prompt_online(
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int | None,
    max_batches: int,
) -> dict[str, Any]:
    prior_switch = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch, dict):
        return cfg
    if train_entry._degradation_prior_source(prior_switch) not in {
        "qwen_prompt", "qwen_probs", "qwen_probs_severity", "confidence_gate",
    }:
        return cfg

    qwen_opt = cfg.get("qwen_prompt", {}) or {}
    if not isinstance(qwen_opt, dict) or not _bool_opt(qwen_opt.get("online"), False):
        return cfg

    targets, roots = _collect_qwen_prompt_targets(cfg, split=split, batch_size=batch_size, max_batches=max_batches)
    if not targets:
        raise SystemExit("No qwen_prompt targets collected for online generation")

    output_root = Path(
        qwen_opt.get("output_root")
        or Path("/data/chenzt/Dataset/Qwen3VL/tmp_test") / _safe_name(cfg.get("name", "myfusion-test")) / split
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    missing = [item for item in targets if not (output_root / Path(item["relative_image"]).with_suffix(".json")).exists()]
    print(
        f"qwen_prompt_online=enabled targets={len(targets)} missing={len(missing)} output_root={output_root}",
        flush=True,
    )
    if missing:
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": split,
            "max_batches": max_batches,
            "batch_size": batch_size,
            "targets": targets,
        }
        manifest_path = output_root / f"online_targets_{split}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        python_bin = str((cfg.get("commands", {}) or {}).get("python") or sys.executable)
        export_script = train_entry._resolve(qwen_opt.get("export_script")) or DEFAULT_QWEN_PROMPT_EXPORT
        qwen_config = train_entry._resolve(qwen_opt.get("config")) or DEFAULT_QWEN_PROMPT_CONFIG
        data_root = roots[0] if roots else Path("/")
        cmd = [
            python_bin,
            str(export_script),
            "--config",
            str(qwen_config),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--target-manifest",
            str(manifest_path),
        ]
        if qwen_opt.get("gpu") not in (None, ""):
            cmd += ["--gpu", str(qwen_opt.get("gpu"))]
        if _bool_opt(qwen_opt.get("overwrite"), False):
            cmd.append("--overwrite")

        env = os.environ.copy()
        if qwen_opt.get("gpu") not in (None, ""):
            env["CUDA_VISIBLE_DEVICES"] = str(qwen_opt.get("gpu"))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_path = output_root / f"qwen_prompt_online_{split}.log"
        print(f"qwen_prompt_online_cmd={' '.join(cmd)}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("\n# qwen_prompt online run\n")
            log_handle.write(" ".join(cmd) + "\n")
            log_handle.flush()
            subprocess.run(cmd, check=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        print(f"qwen_prompt_online_log={log_path}", flush=True)

    missing_after = [item for item in targets if not (output_root / Path(item["relative_image"]).with_suffix(".json")).exists()]
    if missing_after:
        raise SystemExit(f"qwen_prompt online generation left {len(missing_after)} missing JSON files under {output_root}")

    updated = dict(cfg)
    datasets = dict(updated.get("datasets", {}) or {})
    split_key = "val" if split in {"val", "validation"} else split
    split_opt = dict(datasets.get(split_key, {}) or {})
    split_opt["qwen_prompt_root"] = str(output_root)
    datasets[split_key] = split_opt
    updated["datasets"] = datasets
    prior_switch = dict(prior_switch)
    prior_switch["qwen_prompt_root"] = str(output_root)
    updated["prior_switch"] = prior_switch
    return updated


def _build_config(cfg: dict[str, Any], *, split: str, batch_size: int | None, device: str | None) -> AssessTPGDTrainConfig:
    dataset_opt = _split_dataset_opt(cfg, split)
    train_dataset_opt = cfg.get("datasets", {}).get("train", {}) or {}
    test_opt = _test_opt(cfg)
    path_opt = _path_opt(cfg)
    prior_switch_opt = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch_opt, dict):
        prior_switch_opt = {}

    dataset_types = train_entry._dataset_types(dataset_opt)
    lq_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "lq", dataset_types)
    gt_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "gt", dataset_types)
    hidden_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "hidden", dataset_types)
    hidden_path = train_entry._resolve(dataset_opt.get("hidden_path"))
    degradation_prior_source = train_entry._degradation_prior_source(prior_switch_opt)
    structured_prior_temperature = train_entry._structured_prior_temperature(prior_switch_opt)
    condition_dropout_probability, prior_corruption_probability, confidence_override = (
        train_entry._confidence_gate_options(cfg)
    )
    qwen_prompt_root = train_entry._resolve(
        dataset_opt.get("qwen_prompt_root")
        or dataset_opt.get("structured_prior_root")
        or prior_switch_opt.get("qwen_prompt_root")
        or prior_switch_opt.get("structured_prior_root")
    )

    if not lq_dirs or not gt_dirs:
        raise SystemExit(f"datasets.{split} lq/gt dirs are missing")
    if len(lq_dirs) != len(gt_dirs):
        raise SystemExit(f"datasets.{split} lq/gt directory counts differ: {len(lq_dirs)} vs {len(gt_dirs)}")
    if degradation_prior_source == "assessment_hidden" and hidden_path is None and len(hidden_dirs) != len(lq_dirs):
        raise SystemExit(f"datasets.{split} hidden directory count differs from lq count: {len(hidden_dirs)} vs {len(lq_dirs)}")

    effective_batch = batch_size or int(test_opt.get("batch_size") or dataset_opt.get("batch_size") or train_dataset_opt.get("batch_size") or 1)
    image_size = int(dataset_opt.get("image_size") or test_opt.get("image_size") or train_dataset_opt.get("image_size") or train_dataset_opt.get("patch_size") or 128)
    boundary_pad_value = test_opt.get(
        "boundary_pad",
        dataset_opt.get(
            "boundary_pad",
            train_dataset_opt.get(
                "boundary_pad",
                train_entry._get(cfg, "fusion.boundary_pad", train_entry._get(cfg, "runtime.boundary_pad", cfg.get("boundary_pad", 32))),
            ),
        ),
    )
    boundary_pad = int(boundary_pad_value)
    hidden_key = str(dataset_opt.get("hidden_key") or test_opt.get("hidden_key") or train_dataset_opt.get("hidden_key") or "condition_hidden")

    return AssessTPGDTrainConfig(
        lq_dir=train_entry._single_or_list(lq_dirs),
        gt_dir=train_entry._single_or_list(gt_dirs),
        hidden_dir=train_entry._single_or_list(hidden_dirs),
        hidden_path=hidden_path,
        output_dir=Path("."),
        tpgd_options=train_entry._resolve(path_opt.get("tpgd_options")),
        tpgd_checkpoint=train_entry._resolve(path_opt.get("base_checkpoint_load")),
        tpgd_inline_options=train_entry._inline_tpgd_options(cfg),
        hidden_key=hidden_key,
        image_size=image_size,
        batch_size=effective_batch,
        epochs=1,
        max_steps=0,
        lr=0.0,
        optimizer="AdamW",
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        num_workers=int(dataset_opt.get("n_workers", dataset_opt.get("num_workers", test_opt.get("n_workers", 0)))),
        device=device or train_entry._select_device(cfg),
        train_backbone=False,
        load_checkpoint=False,
        strict_load=bool(path_opt.get("strict_load", False)),
        adapter_hidden_dim=int(test_opt.get("adapter_hidden_dim", 1024)),
        adapter_pool=str(test_opt.get("adapter_pool", "mean")),
        adapter_dropout=float(test_opt.get("adapter_dropout", 0.0)),
        random_content_context=False,
        degradation_prior_source=degradation_prior_source,
        use_content_prior=train_entry._bool_switch(prior_switch_opt.get("use_content_prior"), False),
        use_structure_prior=train_entry._bool_switch(prior_switch_opt.get("use_struct_prior", prior_switch_opt.get("use_structure_prior")), False),
        train_structure_prior=False,
        prior_checkpoint=train_entry._resolve(path_opt.get("prior")),
        objective=str(test_opt.get("objective", "sde")),
        prediction_target=str(test_opt.get("prediction_target", "image")),
        loss_type=str(test_opt.get("loss_type", "l1")),
        loss_weight=float(test_opt.get("weight", test_opt.get("loss_weight", 1.0))),
        boundary_pad=boundary_pad,
        direct_gt_time=float(test_opt.get("direct_gt_time", test_opt.get("direct_time", 0.0))),
        sde_t_start=int(test_opt.get("sde_t_start", 1)),
        sde_t_end=int(test_opt.get("sde_t_end", -1)),
        save_full_model=True,
        structured_prior_root=qwen_prompt_root,
        structured_prior_temperature=structured_prior_temperature,
        condition_dropout_probability=condition_dropout_probability,
        prior_corruption_probability=prior_corruption_probability,
        structured_confidence_override=confidence_override,
        eval_batch_size=effective_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-test Assess-TPGD checkpoints and compute PSNR/SSIM metrics.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "test" / "batch.yml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override batch config path.checkpoint_load")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the whole split")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-images", action="store_true", help="Save restored images in addition to metrics")
    parser.add_argument("--save-dir", type=Path, default=None, help="Restored-image output directory; implies --save-images")
    parser.add_argument(
        "--prior-variant",
        choices=["correct", "zero", "uniform", "shuffled", "forced_wrong"],
        default=None,
        help="Optional structured-prior perturbation for one evaluation",
    )
    parser.add_argument(
        "--prior-suite",
        action="store_true",
        help="Evaluate correct/zero/uniform/shuffled/forced_wrong priors in one command",
    )
    parser.add_argument(
        "--confidence-override",
        type=float,
        default=None,
        help="Force A4 gate confidence to a scalar in [0,1], e.g. 0 or 1",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {args.config}")
    cfg = prepare_qwen_prompt_online(cfg, split=args.split, batch_size=args.batch_size, max_batches=args.max_batches)
    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    output_json = args.output_json
    if output_json is None:
        suffix = "_prior_suite" if args.prior_suite else ""
        output_json = checkpoint.parent / f"metrics_{args.split}{suffix}.json"

    test_opt = _test_opt(cfg)
    save_images = args.save_images or args.save_dir is not None or _bool_opt(test_opt.get("save_images"), False)
    configured_save_dir = args.save_dir or test_opt.get("save_dir")
    variants = (
        ["correct", "zero", "uniform", "shuffled", "forced_wrong"]
        if args.prior_suite
        else [args.prior_variant or str(test_opt.get("prior_variant", "correct"))]
    )
    metrics_by_variant: dict[str, dict[str, float]] = {}
    per_image_by_variant: dict[str, list[dict[str, Any]]] = {}
    saved_images_dir = None
    for variant in variants:
        test_cfg = _build_config(cfg, split=args.split, batch_size=args.batch_size, device=args.device)
        test_cfg.structured_prior_variant = variant
        if args.confidence_override is not None:
            if not 0.0 <= args.confidence_override <= 1.0:
                raise SystemExit("--confidence-override must be in [0, 1]")
            test_cfg.structured_confidence_override = args.confidence_override
        variant_save_dir = None
        if save_images and (not args.prior_suite or variant == "correct"):
            if configured_save_dir not in (None, ""):
                variant_save_dir = Path(configured_save_dir).expanduser()
                if not variant_save_dir.is_absolute():
                    variant_save_dir = (ROOT / variant_save_dir).resolve()
            else:
                variant_save_dir = output_json.parent / f"{output_json.stem}_images"
            saved_images_dir = variant_save_dir
        metric_name = f"test_{args.split}" if len(variants) == 1 else f"test_{args.split}_{variant}"
        per_image_by_variant[variant] = []
        metrics_by_variant[variant] = test_assess_tpgd(
            test_cfg,
            checkpoint_path=checkpoint,
            max_batches=args.max_batches,
            metrics_name=metric_name,
            save_images_dir=variant_save_dir,
            per_image_records=per_image_by_variant[variant],
        )

    result = {
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "max_batches": args.max_batches,
        "prediction_target": test_cfg.prediction_target,
        "confidence_override": args.confidence_override,
    }
    if len(variants) == 1:
        result["prior_variant"] = variants[0]
        result["metrics"] = metrics_by_variant[variants[0]]
    else:
        result["prior_variants"] = variants
        result["metrics_by_prior_variant"] = metrics_by_variant
    if saved_images_dir is not None:
        result["saved_images_dir"] = str(saved_images_dir.expanduser().resolve())
    per_image_json = output_json.with_name(f"{output_json.stem}_per_image.json")
    per_image_json.parent.mkdir(parents=True, exist_ok=True)
    per_image_json.write_text(
        json.dumps(per_image_by_variant, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result["per_image_metrics"] = str(per_image_json.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"metrics_saved={output_json}")
    if saved_images_dir is not None:
        print(f"images_saved={saved_images_dir}")


if __name__ == "__main__":
    main()
