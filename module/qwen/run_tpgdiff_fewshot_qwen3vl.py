#!/usr/bin/env python3
"""Run Qwen3-VL few-shot degradation analysis on TPGDiff samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import yaml


def configure_gpu_from_default_config_before_torch() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    config_path = Path(__file__).resolve().parents[2] / "config" / "tpgdiff_fewshot_qwen3vl.yml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    gpu = (config.get("runtime") or {}).get("gpu")
    if gpu is not None and str(gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


configure_gpu_from_default_config_before_torch()

import torch
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages: list[dict[str, Any]]) -> tuple[list[Image.Image], None]:
        images: list[Image.Image] = []
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                image_ref = str(item["image"])
                if image_ref.startswith("file://"):
                    image_ref = image_ref[len("file://") :]
                image = Image.open(image_ref).convert("RGB")
                max_pixels = item.get("max_pixels")
                if isinstance(max_pixels, int) and max_pixels > 0 and image.width * image.height > max_pixels:
                    scale = (max_pixels / float(image.width * image.height)) ** 0.5
                    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                images.append(image)
        return images, None
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tpgdiff_fewshot_qwen3vl.yml"
CLASSES = ["GoPro", "Denoising", "LOL-v2", "DehazeFormer", "Rain200L"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEGRADATION_INFO = {
    "GoPro": {
        "primary": "blur",
        "task": "deblurring",
        "hint": "streaked edges, smeared details, ghosted contours, and camera/object motion blur",
    },
    "Denoising": {
        "primary": "noise",
        "task": "denoising",
        "hint": "grain, color speckles, random high-frequency noise, and rough flat regions",
    },
    "LOL-v2": {
        "primary": "low_light",
        "task": "low-light enhancement",
        "hint": "under-exposure, dim regions, low contrast, shadow detail loss, and possible amplified noise",
    },
    "DehazeFormer": {
        "primary": "haze",
        "task": "dehazing",
        "hint": "foggy veil, reduced contrast, desaturated distant regions, and airlight",
    },
    "Rain200L": {
        "primary": "rain",
        "task": "deraining",
        "hint": "visible rain streaks, thin bright lines, and rain-induced occlusion",
    },
}
SCORING_CANDIDATES = ["noise", "blur", "haze", "rain", "low_light"]
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


def layout_template(
    *,
    global_: bool = False,
    local_region: bool = False,
    object_specific: bool = False,
    continuous: bool = False,
    discrete: bool = False,
    directional: bool = False,
    depth_dependent: bool = False,
    shadow_dependent: bool = False,
    texture_dependent: bool = False,
    uncertain: bool = False,
) -> dict[str, bool]:
    return {
        "global": global_,
        "local_region": local_region,
        "object_specific": object_specific,
        "continuous": continuous,
        "discrete": discrete,
        "directional": directional,
        "depth_dependent": depth_dependent,
        "shadow_dependent": shadow_dependent,
        "texture_dependent": texture_dependent,
        "uncertain": uncertain,
    }


SCORING_OUTPUTS = {
    "noise": {
        "noise": "moderate",
        "blur": "none",
        "haze": "none",
        "rain": "none",
        "low_light": "none",
        "degradation_layout": layout_template(global_=True, continuous=True, texture_dependent=True),
    },
    "blur": {
        "noise": "none",
        "blur": "moderate",
        "haze": "none",
        "rain": "none",
        "low_light": "none",
        "degradation_layout": layout_template(global_=True, continuous=True, directional=True),
    },
    "haze": {
        "noise": "none",
        "blur": "none",
        "haze": "moderate",
        "rain": "none",
        "low_light": "none",
        "degradation_layout": layout_template(global_=True, continuous=True, depth_dependent=True),
    },
    "rain": {
        "noise": "none",
        "blur": "none",
        "haze": "none",
        "rain": "moderate",
        "low_light": "none",
        "degradation_layout": layout_template(global_=True, discrete=True, directional=True),
    },
    "low_light": {
        "noise": "none",
        "blur": "none",
        "haze": "none",
        "rain": "none",
        "low_light": "moderate",
        "degradation_layout": layout_template(global_=True, continuous=True, shadow_dependent=True),
    },
}


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


DEFAULT_SYSTEM_INTRO = (
    "You are an image restoration dataset inspector. "
    "Use the labeled examples as few-shot references for degradation recognition. "
    "For each labeled example, learn the visual pattern, not the filename."
)
DEFAULT_TARGET_INSTRUCTION = (
    "Now analyze the target image. Do not infer from file path or dataset name; "
    "judge from visual evidence only."
)


def sample_images(
    root: Path,
    split: str,
    dataset: str,
    count: int,
    rng: random.Random,
    lq_subdir: str,
) -> list[Path]:
    lq_dir = root / split / dataset / lq_subdir
    files = image_files(lq_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {lq_dir}")
    if count >= len(files):
        return files
    return rng.sample(files, count)


DEGRADATION_KEYS = ["noise", "blur", "haze", "rain", "low_light"]
EXPECTED_JSON_KEYS = DEGRADATION_KEYS + ["degradation_layout"]
SEVERITY_OPTIONS = {"none", "mild", "moderate", "serious", "severe"}
SEVERITY_RANK = {"none": 0, "mild": 1, "moderate": 2, "serious": 3, "severe": 3}
MAIN_DEGRADATION_OPTIONS = {"none", "noise", "blur", "haze", "rain", "low_light"}


def json_schema_prompt() -> str:
    return """Output one valid JSON object only. No markdown, no comments, no text before or after JSON.

Think through the target image internally, but do not output your reasoning. The final answer must contain exactly the fields below.

Required JSON keys, in this exact order:
noise, blur, haze, rain, low_light, degradation_layout

Allowed severity values:
- noise: none | mild | moderate | serious
- blur: none | mild | moderate | serious
- haze: none | mild | moderate | serious
- rain: none | mild | moderate | serious
- low_light: none | mild | moderate | serious

Required degradation_layout keys, in this exact order. Every value must be true or false:
global, local_region, object_specific, continuous, discrete, directional, depth_dependent, shadow_dependent, texture_dependent, uncertain

Layout definitions:
- global: degradation affects the whole image or most visible regions.
- local_region: degradation is concentrated in specific regions, not a single moving object.
- object_specific: degradation is tied mainly to a specific object or moving subject.
- continuous: degradation forms a spatially continuous field, such as noise, blur, haze, or low illumination.
- discrete: degradation appears as separated structures, such as rain streaks or sparse occluding artifacts.
- directional: degradation has an obvious direction, such as slanted rain streaks or motion blur direction.
- depth_dependent: degradation strength changes with scene depth, such as haze being stronger in distant regions.
- shadow_dependent: degradation is stronger in dark or shadowed regions, such as low-light darkness or amplified shadow noise.
- texture_dependent: degradation is most visible on flat regions, fine textures, edges, or low-texture surfaces.
- uncertain: use true only when the layout pattern cannot be judged reliably; if uncertain is true, set the other layout fields to false.

Decision rules:
- Use blur for motion blur, camera shake, defocus-like smearing, doubled contours, ghosted edges, or subtle edge softness.
- Use noise for grain, random speckles, color speckles, salt-and-pepper artifacts, or rough low-texture surfaces.
- Use haze for fog/veil/airlight, washed-out contrast, pale or gray veiling, desaturated distant regions, or reduced visibility while the scene is not primarily dark.
- Use rain for visible rain streaks, thin bright or dark slanted lines, raindrop streak occlusion, or repeated line artifacts.
- Use low_light for under-exposure, globally dim illumination, crushed shadows, very dark regions, and low contrast caused by insufficient light.
- Use serious only for heavy/dominant degradation; use moderate for clear visible degradation; use mild for subtle degradation; use none when absent.
- Do not output main_degradation, confidence, degradation_position, degradation_probs, probability_margin, or vector fields. Those are computed outside this generation step.
- Stop immediately after the closing brace.

Layout examples:
- Global noise: {"global": true, "local_region": false, "object_specific": false, "continuous": true, "discrete": false, "directional": false, "depth_dependent": false, "shadow_dependent": false, "texture_dependent": true, "uncertain": false}
- Object motion blur: {"global": false, "local_region": false, "object_specific": true, "continuous": true, "discrete": false, "directional": true, "depth_dependent": false, "shadow_dependent": false, "texture_dependent": true, "uncertain": false}
- Rain: {"global": true, "local_region": false, "object_specific": false, "continuous": false, "discrete": true, "directional": true, "depth_dependent": false, "shadow_dependent": false, "texture_dependent": false, "uncertain": false}
- Haze: {"global": true, "local_region": false, "object_specific": false, "continuous": true, "discrete": false, "directional": false, "depth_dependent": true, "shadow_dependent": false, "texture_dependent": false, "uncertain": false}
- Low light with shadow-dominant regions: {"global": true, "local_region": true, "object_specific": false, "continuous": true, "discrete": false, "directional": false, "depth_dependent": false, "shadow_dependent": true, "texture_dependent": false, "uncertain": false}

Example output format:
{
  "noise": "none",
  "blur": "moderate",
  "haze": "none",
  "rain": "none",
  "low_light": "none",
  "degradation_layout": {
    "global": true,
    "local_region": false,
    "object_specific": false,
    "continuous": true,
    "discrete": false,
    "directional": true,
    "depth_dependent": false,
    "shadow_dependent": false,
    "texture_dependent": true,
    "uncertain": false
  }
}"""

def strict_load_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "top-level JSON value is not an object"
    return parsed, None


def validate_structured_json(obj: dict[str, Any] | None) -> list[str]:
    if obj is None:
        return ["parsed object is None"]
    errors: list[str] = []
    if list(obj.keys()) != EXPECTED_JSON_KEYS:
        errors.append(f"keys must be exactly {EXPECTED_JSON_KEYS}, got {list(obj.keys())}")
        return errors

    for key in DEGRADATION_KEYS:
        value = obj.get(key)
        if value not in SEVERITY_OPTIONS:
            errors.append(f"{key} must be one of {sorted(SEVERITY_OPTIONS)}, got {value!r}")

    layout = obj.get("degradation_layout")
    if not isinstance(layout, dict):
        errors.append("degradation_layout must be a JSON object")
        return errors
    if list(layout.keys()) != LAYOUT_KEYS:
        errors.append(f"degradation_layout keys must be exactly {LAYOUT_KEYS}, got {list(layout.keys())}")
        return errors
    for key in LAYOUT_KEYS:
        if not isinstance(layout.get(key), bool):
            errors.append(f"degradation_layout.{key} must be true or false, got {layout.get(key)!r}")
    if layout.get("uncertain") and any(layout.get(key) for key in LAYOUT_KEYS if key != "uncertain"):
        errors.append("if degradation_layout.uncertain is true, all other layout fields must be false")
    if not any(layout.get(key) for key in LAYOUT_KEYS):
        errors.append("at least one degradation_layout field must be true")

    return errors


def build_messages(
    target: Path,
    fewshot_examples: list[dict[str, Any]],
    max_pixels: int,
    system_intro: str,
    target_instruction: str,
    schema_prompt: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": system_intro}]

    for idx, example in enumerate(fewshot_examples, start=1):
        expected_json = json.dumps(example["output"], ensure_ascii=False, indent=2)
        content.extend(
            [
                {"type": "text", "text": f"Reference example {idx} image:"},
                {"type": "image", "image": f"file://{example['image']}", "max_pixels": max_pixels},
                {"type": "text", "text": f"Reference example {idx} expected output:\n{expected_json}"},
            ]
        )

    content.extend(
        [
            {
                "type": "text",
                "text": (
                    target_instruction
                ),
            },
            {"type": "image", "image": f"file://{target}", "max_pixels": max_pixels},
            {"type": "text", "text": schema_prompt},
        ]
    )
    return [{"role": "user", "content": content}]

def build_scoring_messages(
    target: Path,
    fewshot_examples: list[dict[str, Any]],
    max_pixels: int,
    system_intro: str,
    target_instruction: str,
    schema_prompt: str,
) -> list[dict[str, Any]]:
    messages = build_messages(
        target,
        fewshot_examples,
        max_pixels,
        system_intro,
        target_instruction,
        schema_prompt,
    )
    messages[0]["content"][-1]["text"] = (
        "Choose the single most likely main_degradation for the target image. "
        "Answer with exactly one label from this set and no other text: "
        + ", ".join(SCORING_CANDIDATES)
    )
    return messages


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no JSON object found"
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def load_model(args: argparse.Namespace) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": args.device_map,
        "cache_dir": str(args.cache_dir),
    }
    if args.attn != "auto":
        model_kwargs["attn_implementation"] = args.attn

    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=str(args.cache_dir))
    model.eval()
    return model, processor


def run_one(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    target: Path,
    fewshot_examples: list[dict[str, Any]],
    max_pixels: int,
    max_new_tokens: int,
    system_intro: str,
    target_instruction: str,
    schema_prompt: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    messages = build_messages(
        target,
        fewshot_examples,
        max_pixels,
        system_intro,
        target_instruction,
        schema_prompt,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    generated_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed, parse_error = strict_load_json(output_text)
    if parsed is None:
        parsed, parse_error = extract_json(output_text)
        if parse_error is None:
            parse_error = "non-strict JSON: extra text or markdown around JSON object"
    return output_text, parsed, parse_error

def completion_logprob(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    completion: str,
) -> dict[str, float]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt")

    prompt_len = inputs.input_ids.shape[1]
    completion_ids = processor.tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids
    if completion_ids.numel() == 0:
        raise ValueError("empty completion for condition scoring")

    inputs["input_ids"] = torch.cat([inputs.input_ids, completion_ids], dim=1)
    inputs["attention_mask"] = torch.cat([inputs.attention_mask, torch.ones_like(completion_ids)], dim=1)
    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100
    inputs = inputs.to(model.device)
    labels = labels.to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    token_logprobs = torch.log_softmax(shift_logits[valid], dim=-1)
    gathered = token_logprobs.gather(1, shift_labels[valid].unsqueeze(1)).squeeze(1)
    total = gathered.sum().item()
    count = int(valid.sum().item())
    return {"total_logprob": total, "avg_logprob": total / max(count, 1), "num_tokens": count}


def run_condition_scoring(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    target: Path,
    fewshot_examples: list[dict[str, Any]],
    max_pixels: int,
    system_intro: str,
    target_instruction: str,
    schema_prompt: str,
) -> dict[str, Any]:
    messages = build_scoring_messages(
        target,
        fewshot_examples,
        max_pixels,
        system_intro,
        target_instruction,
        schema_prompt,
    )
    scored: list[dict[str, Any]] = []
    for candidate in SCORING_CANDIDATES:
        # Score only the discriminative label, then map the winner back to structured JSON.
        completion = candidate
        score = completion_logprob(model, processor, messages, completion)
        scored.append({"candidate": candidate, "completion": completion, **score})

    scored.sort(key=lambda item: item["avg_logprob"], reverse=True)
    avg_scores = torch.tensor([item["avg_logprob"] for item in scored])
    probabilities = torch.softmax(avg_scores, dim=0).tolist()
    for item, probability in zip(scored, probabilities, strict=True):
        item["score_probability"] = round(float(probability), 6)

    best = json.loads(json.dumps(SCORING_OUTPUTS[scored[0]["candidate"]]))
    return {
        "parsed": best,
        "scores": scored,
        "confidence": scored[0]["score_probability"],
        "raw_output": json.dumps(best, ensure_ascii=False),
    }


def resolve_config_path(value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def legacy_output_to_layout_output(output: dict[str, Any]) -> dict[str, Any]:
    if list(output.keys()) == EXPECTED_JSON_KEYS:
        return output
    if not all(key in output for key in DEGRADATION_KEYS):
        return output
    main = output.get("main_degradation")
    if main not in SCORING_CANDIDATES:
        ranks = {key: SEVERITY_RANK.get(str(output.get(key)), -1) for key in DEGRADATION_KEYS}
        main = max(ranks, key=ranks.get)
    layout_by_main = {
        "noise": layout_template(global_=True, continuous=True, texture_dependent=True),
        "blur": layout_template(global_=True, continuous=True, directional=True, texture_dependent=True),
        "haze": layout_template(global_=True, continuous=True, depth_dependent=True),
        "rain": layout_template(global_=True, discrete=True, directional=True),
        "low_light": layout_template(global_=True, continuous=True, shadow_dependent=True),
    }
    return {
        **{key: output.get(key) for key in DEGRADATION_KEYS},
        "degradation_layout": layout_by_main.get(str(main), layout_template(uncertain=True)),
    }


def normalize_fewshot_examples(raw_examples: Any) -> list[dict[str, Any]]:
    if raw_examples is None:
        return []
    if not isinstance(raw_examples, list):
        raise ValueError("fewshot_examples must be a list")

    examples: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_examples, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"fewshot_examples[{idx}] must be an object")
        if "image" not in raw or "output" not in raw:
            raise ValueError(f"fewshot_examples[{idx}] must contain image and output")
        image = resolve_config_path(raw["image"])
        if not image.exists():
            raise FileNotFoundError(f"fewshot_examples[{idx}].image not found: {image}")
        output = legacy_output_to_layout_output(raw["output"])
        validation_errors = validate_structured_json(output)
        if validation_errors:
            raise ValueError(f"fewshot_examples[{idx}].output is invalid: {validation_errors}")
        examples.append({"image": image, "output": output})
    return examples


def load_fixed_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], list[tuple[str, Path, str]]]:
    manifest_path = resolve_config_path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    fewshot_examples = normalize_fewshot_examples(data.get("fewshot_examples", []))
    targets: list[tuple[str, Path, str]] = []
    for idx, item in enumerate(data.get("targets", []), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"fixed_manifest targets[{idx}] must be an object")
        dataset = str(item.get("dataset", "unknown"))
        expected = str(item.get("expected_main_degradation", ""))
        if expected not in MAIN_DEGRADATION_OPTIONS:
            raise ValueError(f"fixed_manifest targets[{idx}] has invalid expected_main_degradation: {expected!r}")
        image = resolve_config_path(item["image"])
        if not image.exists():
            raise FileNotFoundError(f"fixed_manifest targets[{idx}].image not found: {image}")
        targets.append((dataset, image, expected))
    return fewshot_examples, targets


def load_config(config_path: Path) -> argparse.Namespace:
    config_path = resolve_config_path(config_path, Path.cwd())
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    sampling_cfg = config.get("sampling", {})
    runtime_cfg = config.get("runtime", {})
    inference_cfg = config.get("inference", {})
    output_cfg = config.get("output", {})
    prompt_cfg = config.get("prompt", {})
    fewshot_cfg = config.get("fewshot_examples", [])

    args = argparse.Namespace()
    args.config = config_path
    args.gpu = runtime_cfg.get("gpu")
    if args.gpu is not None:
        args.gpu = str(args.gpu)
    args.model = str(model_cfg.get("name_or_path", "/data/chenzt/model_weights/Qwen3-VL-8B-Instruct"))
    args.cache_dir = resolve_config_path(model_cfg.get("cache_dir", "/data/chenzt/model_weights/huggingface_cache"))
    args.attn = model_cfg.get("attn", "flash_attention_2")
    args.device_map = str(model_cfg.get("device_map", "auto"))

    args.fixed_manifest = data_cfg.get("fixed_manifest")
    if args.fixed_manifest is not None:
        args.fixed_manifest = resolve_config_path(args.fixed_manifest)
    args.data_root = resolve_config_path(data_cfg.get("root", "/data/chenzt/Dataset/TPGDiff"))
    args.target_split = str(data_cfg.get("target_split", "Val"))
    args.fewshot_split = str(data_cfg.get("fewshot_split", "Train"))
    args.lq_subdir = str(data_cfg.get("lq_subdir", "LQ"))
    args.classes = list(data_cfg.get("classes", CLASSES))
    args.fewshot_examples = normalize_fewshot_examples(fewshot_cfg)

    args.samples_per_class = int(sampling_cfg.get("samples_per_class", 2))
    args.seed = int(sampling_cfg.get("seed", 20260601))
    args.shuffle_targets = bool(sampling_cfg.get("shuffle_targets", True))

    args.max_pixels = int(inference_cfg.get("max_pixels", 448 * 448))
    args.max_new_tokens = int(inference_cfg.get("max_new_tokens", 256))
    args.mode = str(inference_cfg.get("mode", "generate"))
    args.dry_run = bool(inference_cfg.get("dry_run", False))

    args.output_dir = resolve_config_path(output_cfg.get("dir", "outputs/tpgdiff_fewshot_qwen3vl"))

    args.system_intro = str(prompt_cfg.get("system_intro", DEFAULT_SYSTEM_INTRO))
    args.target_instruction = str(prompt_cfg.get("target_instruction", DEFAULT_TARGET_INSTRUCTION))
    args.schema_prompt = str(prompt_cfg.get("schema_prompt", json_schema_prompt()))
    args.prompt_change_summary = str(prompt_cfg.get("change_summary", "Not specified."))

    if args.attn not in {"auto", "sdpa", "flash_attention_2"}:
        raise ValueError(f"model.attn must be one of auto, sdpa, flash_attention_2; got {args.attn!r}")
    if args.mode not in {"generate", "condition_score", "both"}:
        raise ValueError(f"inference.mode must be one of generate, condition_score, both; got {args.mode!r}")
    missing_info = [dataset for dataset in args.classes if dataset not in DEGRADATION_INFO]
    if missing_info:
        raise ValueError(f"Unknown classes: {missing_info}")
    return args


def configure_gpu(gpu: str | None) -> None:
    if gpu is None or not gpu.strip():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def write_prompt_snapshot(args: argparse.Namespace, fewshot_examples: list[dict[str, Any]]) -> Path:
    prompt_path = args.output_dir / "prompt_used.md"
    text = f"""# Prompt Snapshot

## Change Summary
{args.prompt_change_summary}

## Run Settings
- config: {args.config}
- mode: {args.mode}
- fixed_manifest: {args.fixed_manifest}
- samples_per_class: {args.samples_per_class}
- fewshot_examples_count: {len(fewshot_examples)}
- max_pixels: {args.max_pixels}
- max_new_tokens: {args.max_new_tokens}

## Prompt Construction
Each target request is built as one user message containing:
1. system_intro text
2. each reference image and its expected JSON output
3. target_instruction text
4. target image
5. schema_prompt text

For condition_score mode, the final schema prompt is replaced by a single-label choice instruction over: noise, blur, haze, rain, low_light.

## system_intro
```text
{args.system_intro}
```

## target_instruction
```text
{args.target_instruction}
```

## schema_prompt
```text
{args.schema_prompt}
```
"""
    prompt_path.write_text(text, encoding="utf-8")
    return prompt_path


def main() -> None:
    args = load_config(DEFAULT_CONFIG)
    configure_gpu(args.gpu)
    print(f"Using config: {args.config}", flush=True)
    if args.gpu:
        print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.fixed_manifest is not None:
        fewshot_examples, fixed_targets = load_fixed_manifest(args.fixed_manifest)
        targets = [(dataset, path) for dataset, path, _expected in fixed_targets]
        expected_by_target = {str(path): expected for _dataset, path, expected in fixed_targets}
    else:
        fewshot_examples = args.fewshot_examples
        targets: list[tuple[str, Path]] = []
        for dataset in args.classes:
            targets.extend(
                (dataset, path)
                for path in sample_images(
                    args.data_root,
                    args.target_split,
                    dataset,
                    args.samples_per_class,
                    rng,
                    args.lq_subdir,
                )
            )
        if args.shuffle_targets:
            rng.shuffle(targets)
        expected_by_target = {str(path): DEGRADATION_INFO[dataset]["primary"] for dataset, path in targets}

    sample_manifest = {
        "config": str(args.config),
        "gpu": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": args.model,
        "cache_dir": str(args.cache_dir),
        "seed": args.seed,
        "fixed_manifest": str(args.fixed_manifest) if args.fixed_manifest is not None else None,
        "data_root": str(args.data_root),
        "target_split": args.target_split,
        "fewshot_split": args.fewshot_split,
        "lq_subdir": args.lq_subdir,
        "classes": args.classes,
        "samples_per_class": args.samples_per_class,
        "fewshot_examples_count": len(fewshot_examples),
        "mode": args.mode,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "fewshot_examples": [
            {"image": str(example["image"]), "output": example["output"]}
            for example in fewshot_examples
        ],
        "targets": [
            {"dataset": dataset, "path": str(path), "expected_main_degradation": expected_by_target[str(path)]}
            for dataset, path in targets
        ],
    }
    manifest_path = args.output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}", flush=True)
    prompt_path = write_prompt_snapshot(args, fewshot_examples)
    print(f"Saved prompt snapshot: {prompt_path}", flush=True)
    if args.dry_run:
        return

    print(f"Loading model: {args.model}", flush=True)
    model, processor = load_model(args)

    jsonl_path = args.output_dir / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, (dataset, target) in enumerate(targets, start=1):
            started = time.time()
            print(f"[{idx}/{len(targets)}] {dataset}: {target}", flush=True)
            raw = ""
            parsed = None
            parse_error = None
            condition_scoring = None
            if args.mode in {"generate", "both"}:
                raw, parsed, parse_error = run_one(
                    model=model,
                    processor=processor,
                    target=target,
                    fewshot_examples=fewshot_examples,
                    max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                    system_intro=args.system_intro,
                    target_instruction=args.target_instruction,
                    schema_prompt=args.schema_prompt,
                )
            if args.mode in {"condition_score", "both"}:
                condition_scoring = run_condition_scoring(
                    model=model,
                    processor=processor,
                    target=target,
                    fewshot_examples=fewshot_examples,
                    max_pixels=args.max_pixels,
                    system_intro=args.system_intro,
                    target_instruction=args.target_instruction,
                    schema_prompt=args.schema_prompt,
                )
                if args.mode == "condition_score":
                    raw = condition_scoring["raw_output"]
                    parsed = condition_scoring["parsed"]
            strict_parsed, strict_error = strict_load_json(raw) if raw else (None, "empty raw output")
            validation_target = strict_parsed if strict_error is None else parsed
            validation_errors = validate_structured_json(validation_target)
            expected_degradation = expected_by_target[str(target)]
            if (
                validation_target is not None
                and "main_degradation" in validation_target
                and validation_target.get("main_degradation") != expected_degradation
            ):
                validation_errors.append(
                    f"main_degradation must match expected {expected_degradation!r}, "
                    f"got {validation_target.get('main_degradation')!r}"
                )
            record = {
                "dataset": dataset,
                "target": str(target),
                "expected_degradation": expected_degradation,
                "raw_output": raw,
                "parsed": parsed,
                "parse_error": parse_error,
                "strict_json_valid": strict_error is None,
                "strict_json_error": strict_error,
                "schema_valid": not validation_errors,
                "validation_errors": validation_errors,
                "condition_scoring": condition_scoring,
                "elapsed_sec": round(time.time() - started, 3),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(raw, flush=True)

    print(f"Saved predictions: {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
