#!/usr/bin/env python3
"""Export Qwen3-VL structured degradation priors for the TPGDiff dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tpgdiff_fewshot_qwen3vl.yml"
DEFAULT_OUTPUT_ROOT = Path("/data/chenzt/Dataset/Qwen3VL/0611")
DEFAULT_DATA_ROOT = Path("/data/chenzt/Dataset/TPGDiff")
DEFAULT_CLASSES = ["GoPro", "Denoising", "LOL-v2", "DehazeFormer", "Rain200L"]
DEFAULT_SPLITS = ["Train", "Val"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEGRADATION_KEYS = ["noise", "blur", "haze", "rain", "low_light"]
LAYOUT_KEYS = [
    "global",
    "local_region",
    "object_specific",
    "continuous",
    "discrete",
    "directional",
    "depth_dependent",
    "shadow_dependent",
    "texture_dependent",
    "uncertain",
]
SEVERITY_TO_FLOAT = {
    "none": 0.0,
    "mild": 1.0 / 3.0,
    "moderate": 2.0 / 3.0,
    "serious": 1.0,
    "severe": 1.0,
}
CLASS_TO_MAIN = {
    "GoPro": "blur",
    "Denoising": "noise",
    "LOL-v2": "low_light",
    "DehazeFormer": "haze",
    "Rain200L": "rain",
}


def _load_qwen_runner():
    path = Path(__file__).resolve().with_name("run_tpgdiff_fewshot_qwen3vl.py")
    spec = importlib.util.spec_from_file_location("_qwen_tpgdiff_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Qwen runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _build_targets(data_root: Path, splits: list[str], classes: list[str], lq_subdir: str) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for split in splits:
        for dataset in classes:
            lq_dir = data_root / split / dataset / lq_subdir
            for image in _image_files(lq_dir):
                rel = image.relative_to(data_root)
                targets.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "image": image,
                        "relative_image": rel,
                        "expected_main_degradation": CLASS_TO_MAIN.get(dataset, ""),
                    }
                )
    return targets




def _load_manifest_targets(path: Path, data_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    raw_targets = payload.get("targets", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_targets, list):
        raise ValueError(f"target manifest must be a list or contain targets list: {path}")
    targets: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_targets, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"target manifest item {idx} is not an object")
        image = Path(item.get("image", "")).expanduser()
        if not image.is_absolute():
            image = data_root / image
        relative = item.get("relative_image")
        if relative in (None, ""):
            relative_path = image.resolve().relative_to(data_root.resolve())
        else:
            relative_path = Path(str(relative))
        dataset = str(item.get("dataset") or relative_path.parts[1] if len(relative_path.parts) > 1 else "")
        split = str(item.get("split") or relative_path.parts[0] if relative_path.parts else "")
        targets.append(
            {
                "split": split,
                "dataset": dataset,
                "image": image.resolve(),
                "relative_image": relative_path,
                "expected_main_degradation": item.get("expected_main_degradation", CLASS_TO_MAIN.get(dataset, "")),
            }
        )
    return targets

def _result_path(output_root: Path, relative_image: Path) -> Path:
    return output_root / relative_image.with_suffix(".json")


def _probabilities_from_condition(condition_scoring: dict[str, Any] | None) -> dict[str, float]:
    probs = {key: 0.0 for key in DEGRADATION_KEYS}
    if not condition_scoring:
        return probs
    for item in condition_scoring.get("scores", []):
        candidate = item.get("candidate")
        if candidate in probs:
            probs[candidate] = float(item.get("score_probability", 0.0))
    return probs


def _margin_from_probs(probs: dict[str, float]) -> float:
    values = sorted((float(value) for value in probs.values()), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return values[0] - values[1]


def _vector_from_outputs(parsed: dict[str, Any] | None, probs: dict[str, float], margin: float) -> list[float]:
    parsed = parsed or {}
    layout = parsed.get("degradation_layout")
    if not isinstance(layout, dict):
        layout = {}
    severity = [SEVERITY_TO_FLOAT.get(str(parsed.get(key, "none")), 0.0) for key in DEGRADATION_KEYS]
    prob_vec = [float(probs.get(key, 0.0)) for key in DEGRADATION_KEYS]
    layout_vec = [1.0 if bool(layout.get(key, False)) else 0.0 for key in LAYOUT_KEYS]
    return severity + prob_vec + layout_vec + [float(margin)]


def _main_from_probs(probs: dict[str, float]) -> str:
    if not probs:
        return "none"
    return max(probs, key=probs.get)


def _write_readme(output_root: Path, config_path: Path, data_root: Path, total: int) -> None:
    text = f"""# Qwen3-VL Structured Degradation Priors

Generated on 2026-06-11 for the TPGDiff image restoration dataset.

## Source And Output
- source dataset: `{data_root}`
- output root: `{output_root}`
- config: `{config_path}`
- images scheduled: `{total}`

The output mirrors the original TPGDiff directory structure. For example:

```text
TPGDiff/Train/GoPro/LQ/xxx.png
-> Qwen3VL/0611/Train/GoPro/LQ/xxx.json
```

## Structured Output Idea
Each low-quality image is analyzed by Qwen3-VL with the iter17 prompt. The model
generates five degradation severities and ten layout attributes:

```text
severity_5 = [noise, blur, haze, rain, low_light]
layout_10 = [
  global, local_region, object_specific, continuous, discrete,
  directional, depth_dependent, shadow_dependent, texture_dependent, uncertain
]
```

In addition, the script scores five candidate degradation labels with Qwen
log-probabilities. The softmax-normalized candidate scores are saved as:

```text
degradation_probs_5 = [p_noise, p_blur, p_haze, p_rain, p_low_light]
probability_margin = top_probability - second_probability
```

The final `prior_vector_21` is:

```text
severity_5 + degradation_probs_5 + layout_10 + [probability_margin]
```

This vector is the intended input prior for MyFusion/TPGDiff:
- `severity_5 + degradation_probs_5` are encoded into `deg_context` and added to
  the UNet time-embedding path.
- `layout_10` is encoded into structure tokens and injected through the UNet
  FiLM structure adapters.

## Files
- one JSON per image under the mirrored directory tree
- `manifest.json`: scheduled images and run settings
- `results.jsonl`: one compact line per successfully processed image
- `README.md`: this description
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--lq-subdir", default="LQ")
    parser.add_argument("--target-manifest", type=Path, default=None, help="JSON list/manifest of exact images to process")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    runtime_cfg = config.get("runtime", {}) or {}
    gpu = args.gpu if args.gpu is not None else runtime_cfg.get("gpu")
    if gpu is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    qwen = _load_qwen_runner()
    qwen_args = qwen.load_config(args.config)
    qwen_args.fixed_manifest = None
    qwen_args.data_root = args.data_root
    qwen_args.output_dir = args.output_root
    qwen_args.mode = "both"
    qwen_args.dry_run = args.dry_run
    qwen_args.classes = args.classes
    qwen_args.target_split = ",".join(args.splits)

    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    qwen_args.cache_dir.mkdir(parents=True, exist_ok=True)

    fewshot_examples = qwen_args.fewshot_examples
    targets = (
        _load_manifest_targets(args.target_manifest, data_root)
        if args.target_manifest is not None
        else _build_targets(data_root, args.splits, args.classes, args.lq_subdir)
    )
    if args.limit > 0:
        targets = targets[: args.limit]

    manifest = {
        "config": str(args.config),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "classes": args.classes,
        "lq_subdir": args.lq_subdir,
        "mode": "both",
        "max_pixels": qwen_args.max_pixels,
        "max_new_tokens": qwen_args.max_new_tokens,
        "model": qwen_args.model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "total_images": len(targets),
        "targets": [
            {
                "split": item["split"],
                "dataset": item["dataset"],
                "image": str(item["image"]),
                "relative_image": str(item["relative_image"]),
                "output": str(_result_path(output_root, item["relative_image"])),
                "expected_main_degradation": item["expected_main_degradation"],
            }
            for item in targets
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(output_root, args.config, data_root, len(targets))
    qwen.write_prompt_snapshot(qwen_args, fewshot_examples)

    print(f"output_root={output_root}", flush=True)
    print(f"total_images={len(targets)}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if args.dry_run:
        return

    model, processor = qwen.load_model(qwen_args)
    results_jsonl = output_root / "results.jsonl"
    processed = 0
    skipped = 0
    failed = 0

    with results_jsonl.open("a", encoding="utf-8") as results_handle:
        for idx, item in enumerate(targets, start=1):
            image: Path = item["image"]
            out_path = _result_path(output_root, item["relative_image"])
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            started = time.time()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{idx}/{len(targets)}] {item['split']}/{item['dataset']} {image}", flush=True)
            try:
                raw, parsed, parse_error = qwen.run_one(
                    model=model,
                    processor=processor,
                    target=image,
                    fewshot_examples=fewshot_examples,
                    max_pixels=qwen_args.max_pixels,
                    max_new_tokens=qwen_args.max_new_tokens,
                    system_intro=qwen_args.system_intro,
                    target_instruction=qwen_args.target_instruction,
                    schema_prompt=qwen_args.schema_prompt,
                )
                condition_scoring = qwen.run_condition_scoring(
                    model=model,
                    processor=processor,
                    target=image,
                    fewshot_examples=fewshot_examples,
                    max_pixels=qwen_args.max_pixels,
                    system_intro=qwen_args.system_intro,
                    target_instruction=qwen_args.target_instruction,
                    schema_prompt=qwen_args.schema_prompt,
                )
                strict_parsed, strict_error = qwen.strict_load_json(raw) if raw else (None, "empty raw output")
                validation_target = strict_parsed if strict_error is None else parsed
                validation_errors = qwen.validate_structured_json(validation_target)
                probs = _probabilities_from_condition(condition_scoring)
                margin = _margin_from_probs(probs)
                final_record = {
                    "source_image": str(image),
                    "relative_image": str(item["relative_image"]),
                    "split": item["split"],
                    "dataset": item["dataset"],
                    "expected_main_degradation": item["expected_main_degradation"],
                    "qwen_raw_output": raw,
                    "qwen_parsed": parsed,
                    "parse_error": parse_error,
                    "strict_json_valid": strict_error is None,
                    "strict_json_error": strict_error,
                    "schema_valid": not validation_errors,
                    "validation_errors": validation_errors,
                    "condition_scoring": condition_scoring,
                    "main_degradation_from_probs": _main_from_probs(probs),
                    "degradation_probs": probs,
                    "probability_margin": margin,
                    "prior_vector_order": DEGRADATION_KEYS + DEGRADATION_KEYS + LAYOUT_KEYS + ["probability_margin"],
                    "prior_vector_21": _vector_from_outputs(validation_target, probs, margin),
                    "elapsed_sec": round(time.time() - started, 3),
                }
                out_path.write_text(json.dumps(final_record, ensure_ascii=False, indent=2), encoding="utf-8")
                results_handle.write(json.dumps({k: final_record[k] for k in [
                    "source_image",
                    "relative_image",
                    "split",
                    "dataset",
                    "schema_valid",
                    "main_degradation_from_probs",
                    "probability_margin",
                    "elapsed_sec",
                ]}, ensure_ascii=False) + "\n")
                results_handle.flush()
                processed += 1
            except Exception as exc:
                failed += 1
                error_record = {
                    "source_image": str(image),
                    "relative_image": str(item["relative_image"]),
                    "split": item["split"],
                    "dataset": item["dataset"],
                    "error": repr(exc),
                    "elapsed_sec": round(time.time() - started, 3),
                }
                out_path.write_text(json.dumps(error_record, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"ERROR {image}: {exc!r}", flush=True)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total": len(targets),
        "output_root": str(output_root),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
