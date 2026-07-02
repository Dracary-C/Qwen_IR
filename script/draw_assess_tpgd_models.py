#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "script"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_assess_tpgd as train_entry
from module.backbone import AssessConditionedTPGDUNet, PlainTPGDUNet, TPGDBackboneConfig
from module.pipeline.assess_tpgd import _deep_update, _load_tpgd_options


@dataclass
class ModelSpec:
    mode: str
    config_path: str
    model_variant: str
    degradation_prior_source: str
    use_content_prior: bool
    use_structure_prior: bool
    use_structured_degra_prior: bool
    use_external_structure_prior: bool
    train_backbone: bool
    train_structure_prior: bool
    objective: str
    image_size: int
    batch_size: int
    boundary_pad: int
    in_nc: int
    out_nc: int
    nf: int
    ch_mult: list[int]
    context_dim: int
    use_degra_context: bool
    use_image_context: bool
    use_struct_context: bool
    struct_context_dim: int
    adapter_hidden_dim: int
    adapter_pool: str
    adapter_dropout: float
    structured_num_tokens: int
    checkpoint_load: str
    base_checkpoint_load: str
    tpgd_options: str
    total_params: int = 0
    trainable_params: int = 0
    backbone_params: int = 0
    assess_prior_params: int = 0
    structured_prior_params: int = 0
    warnings: list[str] | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {path}")
    return data


def _as_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _resolve_from_config(value: Any) -> Path | None:
    return train_entry._resolve(value)


def _load_effective_tpgd_options(cfg: dict[str, Any], warnings: list[str]) -> tuple[dict[str, Any], str]:
    path_opt = cfg.get("path", {}) or {}
    if not isinstance(path_opt, dict):
        path_opt = {}
    tpgd_options_path = _resolve_from_config(path_opt.get("tpgd_options"))
    tpgd_options: dict[str, Any] = {}
    if tpgd_options_path is not None:
        if tpgd_options_path.exists():
            tpgd_options = _load_tpgd_options(tpgd_options_path)
        else:
            warnings.append(f"tpgd_options not found: {tpgd_options_path}")
    inline_options = train_entry._inline_tpgd_options(cfg)
    if inline_options:
        _deep_update(tpgd_options, inline_options)
    return tpgd_options, "" if tpgd_options_path is None else str(tpgd_options_path)


def _train_spec(config_path: Path) -> ModelSpec:
    cfg = _read_yaml(config_path)
    warnings: list[str] = []
    options, tpgd_options_path = _load_effective_tpgd_options(cfg, warnings)
    setting = options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()

    train_opt = cfg.get("train", {}) or {}
    dataset_opt = train_entry._get(cfg, "datasets.train", {}) or {}
    prior_switch = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch, dict):
        prior_switch = {}
    source = train_entry._degradation_prior_source(prior_switch)
    use_content_prior = train_entry._bool_switch(prior_switch.get("use_content_prior"), False)
    use_structure_prior = train_entry._bool_switch(prior_switch.get("use_struct_prior", prior_switch.get("use_structure_prior")), False)
    use_structured_degra_prior = source == "qwen_prompt"
    use_external_structure_prior = bool(use_structure_prior and not use_structured_degra_prior)
    use_plain_unet = source == "none" and not use_content_prior and not use_structure_prior
    path_opt = cfg.get("path", {}) or {}
    if not isinstance(path_opt, dict):
        path_opt = {}
    return ModelSpec(
        mode="train",
        config_path=str(config_path.resolve()),
        model_variant="plain_unet" if use_plain_unet else "prior_conditioned_unet",
        degradation_prior_source=source,
        use_content_prior=bool(use_content_prior and backbone_cfg.use_image_context),
        use_structure_prior=bool(use_structure_prior and backbone_cfg.use_struct_context),
        use_structured_degra_prior=use_structured_degra_prior,
        use_external_structure_prior=use_external_structure_prior,
        train_backbone=bool(train_opt.get("train_backbone", False)),
        train_structure_prior=bool(train_opt.get("train_structure_prior", False)),
        objective=str(train_opt.get("objective", "sde")),
        image_size=_as_int(dataset_opt.get("image_size", dataset_opt.get("patch_size")), 128),
        batch_size=_as_int(dataset_opt.get("batch_size"), 1),
        boundary_pad=_as_int(
            train_opt.get("boundary_pad", cfg.get("boundary_pad", train_entry._get(cfg, "fusion.boundary_pad", train_entry._get(cfg, "runtime.boundary_pad")))),
            32,
        ),
        in_nc=backbone_cfg.in_nc,
        out_nc=backbone_cfg.out_nc,
        nf=backbone_cfg.nf,
        ch_mult=list(backbone_cfg.ch_mult),
        context_dim=backbone_cfg.context_dim,
        use_degra_context=backbone_cfg.use_degra_context,
        use_image_context=backbone_cfg.use_image_context,
        use_struct_context=backbone_cfg.use_struct_context,
        struct_context_dim=backbone_cfg.struct_context_dim,
        adapter_hidden_dim=_as_int(train_opt.get("adapter_hidden_dim"), 1024),
        adapter_pool=str(train_opt.get("adapter_pool", "mean")),
        adapter_dropout=_as_float(train_opt.get("adapter_dropout"), 0.0),
        structured_num_tokens=int(options.get("structure_prior", {}).get("setting", {}).get("num_latent_tokens", 32)),
        checkpoint_load="" if _resolve_from_config(path_opt.get("checkpoint_load")) is None else str(_resolve_from_config(path_opt.get("checkpoint_load"))),
        base_checkpoint_load="",
        tpgd_options=tpgd_options_path,
        warnings=warnings,
    )


def _test_spec(config_path: Path, split: str) -> ModelSpec:
    cfg = _read_yaml(config_path)
    warnings: list[str] = []
    options, tpgd_options_path = _load_effective_tpgd_options(cfg, warnings)
    setting = options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()

    test_opt = cfg.get("test", {}) or {}
    datasets = cfg.get("datasets", {}) or {}
    dataset_opt = datasets.get(split, datasets.get("val", datasets.get("validation", {}))) or {}
    train_dataset_opt = datasets.get("train", {}) or {}
    prior_switch = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch, dict):
        prior_switch = {}
    source = train_entry._degradation_prior_source(prior_switch)
    use_content_prior = train_entry._bool_switch(prior_switch.get("use_content_prior"), False)
    use_structure_prior = train_entry._bool_switch(prior_switch.get("use_struct_prior", prior_switch.get("use_structure_prior")), False)
    use_structured_degra_prior = source == "qwen_prompt"
    use_external_structure_prior = bool(use_structure_prior and not use_structured_degra_prior)
    use_plain_unet = source == "none" and not use_content_prior and not use_structure_prior
    path_opt = cfg.get("path", {}) or {}
    if not isinstance(path_opt, dict):
        path_opt = {}
    checkpoint = _resolve_from_config(path_opt.get("checkpoint_load"))
    base_checkpoint = _resolve_from_config(path_opt.get("base_checkpoint_load"))
    return ModelSpec(
        mode=f"test:{split}",
        config_path=str(config_path.resolve()),
        model_variant="plain_unet" if use_plain_unet else "prior_conditioned_unet",
        degradation_prior_source=source,
        use_content_prior=bool(use_content_prior and backbone_cfg.use_image_context),
        use_structure_prior=bool(use_structure_prior and backbone_cfg.use_struct_context),
        use_structured_degra_prior=use_structured_degra_prior,
        use_external_structure_prior=use_external_structure_prior,
        train_backbone=False,
        train_structure_prior=False,
        objective=str(test_opt.get("objective", "sde")),
        image_size=_as_int(dataset_opt.get("image_size") or test_opt.get("image_size") or train_dataset_opt.get("image_size") or train_dataset_opt.get("patch_size"), 128),
        batch_size=_as_int(test_opt.get("batch_size") or dataset_opt.get("batch_size") or train_dataset_opt.get("batch_size"), 1),
        boundary_pad=_as_int(test_opt.get("boundary_pad", dataset_opt.get("boundary_pad", train_dataset_opt.get("boundary_pad", cfg.get("boundary_pad", 32)))), 32),
        in_nc=backbone_cfg.in_nc,
        out_nc=backbone_cfg.out_nc,
        nf=backbone_cfg.nf,
        ch_mult=list(backbone_cfg.ch_mult),
        context_dim=backbone_cfg.context_dim,
        use_degra_context=backbone_cfg.use_degra_context,
        use_image_context=backbone_cfg.use_image_context,
        use_struct_context=backbone_cfg.use_struct_context,
        struct_context_dim=backbone_cfg.struct_context_dim,
        adapter_hidden_dim=_as_int(test_opt.get("adapter_hidden_dim"), 1024),
        adapter_pool=str(test_opt.get("adapter_pool", "mean")),
        adapter_dropout=_as_float(test_opt.get("adapter_dropout"), 0.0),
        structured_num_tokens=int(options.get("structure_prior", {}).get("setting", {}).get("num_latent_tokens", 32)),
        checkpoint_load="" if checkpoint is None else str(checkpoint),
        base_checkpoint_load="" if base_checkpoint is None else str(base_checkpoint),
        tpgd_options=tpgd_options_path,
        warnings=warnings,
    )


def _build_model(spec: ModelSpec) -> torch.nn.Module:
    backbone_cfg = TPGDBackboneConfig(
        in_nc=spec.in_nc,
        out_nc=spec.out_nc,
        nf=spec.nf,
        ch_mult=spec.ch_mult,
        context_dim=spec.context_dim,
        use_degra_context=spec.use_degra_context,
        use_image_context=spec.use_image_context,
        upscale=1,
        use_struct_context=spec.use_struct_context,
        struct_context_dim=spec.struct_context_dim,
    )
    if spec.model_variant == "plain_unet":
        return PlainTPGDUNet(backbone_cfg, freeze_backbone=False)
    return AssessConditionedTPGDUNet(
        backbone_cfg,
        adapter_hidden_dim=spec.adapter_hidden_dim,
        adapter_pool=spec.adapter_pool,
        adapter_dropout=spec.adapter_dropout,
        use_structured_prior=spec.use_structured_degra_prior,
        structured_hidden_dim=spec.adapter_hidden_dim,
        structured_num_tokens=spec.structured_num_tokens,
        freeze_backbone=False,
    )


def _param_count(module: torch.nn.Module | None, recurse: bool = True) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters(recurse=recurse))


def _fill_param_counts(spec: ModelSpec, model: torch.nn.Module) -> None:
    spec.total_params = _param_count(model)
    if spec.mode.startswith("test"):
        spec.trainable_params = 0
    elif spec.train_backbone:
        spec.trainable_params = _param_count(model)
    elif spec.degradation_prior_source == "assessment_hidden":
        spec.trainable_params = _param_count(getattr(model, "assess_prior", None))
    elif spec.degradation_prior_source == "qwen_prompt":
        spec.trainable_params = _param_count(getattr(model, "structured_prior", None))
    else:
        spec.trainable_params = 0
    spec.backbone_params = _param_count(getattr(model, "backbone", None))
    spec.assess_prior_params = _param_count(getattr(model, "assess_prior", None))
    spec.structured_prior_params = _param_count(getattr(model, "structured_prior", None))


def _human_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _module_tree(module: torch.nn.Module, max_depth: int = 3) -> str:
    lines: list[str] = []

    def visit(current: torch.nn.Module, prefix: str, depth: int, name: str) -> None:
        params = _param_count(current)
        own_params = _param_count(current, recurse=False)
        lines.append(f"{prefix}{name}: {current.__class__.__name__} params={_human_params(params)} own={_human_params(own_params)}")
        if depth >= max_depth:
            child_count = len(list(current.children()))
            if child_count:
                lines.append(f"{prefix}  ... {child_count} children hidden at depth limit")
            return
        children = list(current.named_children())
        for child_name, child in children:
            visit(child, prefix + "  ", depth + 1, child_name)

    visit(module, "", 0, module.__class__.__name__)
    return "\n".join(lines) + "\n"


def _svg_text(x: int, y: int, text: str, *, size: int = 13, weight: str = "400", fill: str = "#172033") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'


def _svg_box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    lines: list[str],
    *,
    fill: str,
    stroke: str,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>',
        _svg_text(x + 14, y + 25, title, size=15, weight="700"),
    ]
    line_y = y + 49
    for line in lines:
        out.append(_svg_text(x + 14, line_y, line, size=12, fill="#334155"))
        line_y += 18
    return "\n".join(out)


def _svg_arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2 - 8
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow)"/>']
    if label:
        out.append(
            f'<rect x="{mid_x - 78}" y="{mid_y - 13}" width="156" height="20" rx="5" fill="#ffffff" stroke="#e2e8f0"/>'
        )
        out.append(_svg_text(mid_x - 70, mid_y + 2, label, size=11, fill="#475569"))
    return "\n".join(out)


def _svg_doc(width: int, height: int, title: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
  </marker>
</defs>
<rect width="{width}" height="{height}" fill="#f8fafc"/>
{_svg_text(32, 40, title, size=22, weight="800", fill="#0f172a")}
{body}
</svg>
'''


def _core_model_svg(spec: ModelSpec) -> str:
    title = f"{spec.mode} core model: {spec.model_variant}"
    h_model = spec.image_size + 2 * spec.boundary_pad
    w_model = h_model
    prior_lines = [
        f"source={spec.degradation_prior_source}",
        "input prior [B, 21]" if spec.use_structured_degra_prior else "input hidden [B, T, 4096]",
        f"context_dim={spec.context_dim}",
    ]
    if spec.use_struct_context:
        prior_lines.append(f"struct tokens [B,{spec.structured_num_tokens},{spec.struct_context_dim}]")
    body = "\n".join(
        [
            _svg_box(36, 78, 215, 106, "Inputs", [f"xt/state [B,{spec.in_nc},{h_model},{w_model}]", f"cond/LQ [B,{spec.in_nc},{h_model},{w_model}]", "time [B]"], fill="#eef2ff", stroke="#6366f1"),
            _svg_box(36, 250, 215, 112, "Qwen/Assessment Prior", prior_lines, fill="#ecfeff", stroke="#0891b2"),
            _svg_box(308, 94, 250, 86, "Concat + init_conv", [f"Conv2d({spec.in_nc * 2}->{spec.nf}, 7x7)", "time_mlp + prompt_mlp"], fill="#fefce8", stroke="#ca8a04"),
            _svg_box(308, 240, 250, 126, "Prior Adapters", [f"QwenPromptPriorAdapter params={_human_params(spec.structured_prior_params)}", "severity/prob -> deg_context", "layout -> LayoutTokenEncoder", "AssessPriorAdapter present" if spec.assess_prior_params else "AssessPriorAdapter absent"], fill="#fff7ed", stroke="#ea580c", dashed=not spec.use_structured_degra_prior),
            _svg_box(620, 68, 280, 136, "ConditionalUNet Down Path", [f"nf={spec.nf}, ch_mult={spec.ch_mult}", "ResBlock x2 + LinearAttention", "Downsample stages", "channels: 64 -> 64 -> 128 -> 256 -> 512"], fill="#f0fdf4", stroke="#16a34a"),
            _svg_box(620, 238, 280, 104, "Middle", ["ResBlock", "LinearAttention", "ResBlock", "conditioned by time/prior tokens"], fill="#fdf2f8", stroke="#db2777"),
            _svg_box(620, 376, 280, 136, "ConditionalUNet Up Path", ["skip connections from down path", "ResBlock x2 + LinearAttention", "Upsample stages", "channels: 512 -> 256 -> 128 -> 64"], fill="#f0f9ff", stroke="#0284c7"),
            _svg_box(960, 230, 220, 112, "Output Head", ["final_res_block", f"final Conv2d({spec.nf}->{spec.out_nc}, 3x3)", f"output [B,{spec.out_nc},{h_model},{w_model}]"], fill="#f1f5f9", stroke="#64748b"),
            _svg_box(960, 402, 220, 80, "Parameter Counts", [f"total={_human_params(spec.total_params)}", f"backbone={_human_params(spec.backbone_params)}", f"train/eval update={_human_params(spec.trainable_params)}"], fill="#ffffff", stroke="#94a3b8"),
            _svg_arrow(251, 131, 308, 131, "xt, cond, t"),
            _svg_arrow(251, 306, 308, 306, "prior"),
            _svg_arrow(558, 138, 620, 138, "features"),
            _svg_arrow(558, 302, 620, 302, "deg/struct context"),
            _svg_arrow(760, 204, 760, 238),
            _svg_arrow(760, 342, 760, 376),
            _svg_arrow(900, 444, 960, 286, "decoded features"),
        ]
    )
    return _svg_doc(1220, 560, title, body)


def _pipeline_svg(spec: ModelSpec) -> str:
    if spec.mode.startswith("train"):
        boxes = [
            _svg_box(38, 82, 230, 126, "PairedAssessmentDataset", ["LQ image", "GT image", "Qwen prior [21]", f"batch_size={spec.batch_size}"], fill="#eef2ff", stroke="#6366f1"),
            _svg_box(318, 82, 220, 126, "Preprocess", [f"resize={spec.image_size}", f"boundary_pad={spec.boundary_pad}", "LQ/GT -> model size", "optional eval loader"], fill="#ecfeff", stroke="#0891b2"),
            _svg_box(588, 82, 230, 126, "IR-SDE Objective", ["generate timesteps", "states from GT + LQ", "target reverse optimum", f"objective={spec.objective}"], fill="#fefce8", stroke="#ca8a04"),
            _svg_box(868, 82, 250, 126, "AssessConditionedTPGDUNet", ["states, LQ, time", "Qwen prior -> contexts", "predict noise/score", f"params={_human_params(spec.total_params)}"], fill="#f0fdf4", stroke="#16a34a"),
            _svg_box(318, 278, 220, 110, "Loss", ["score -> reverse SDE step", "crop boundary", "L1/L2 matching loss"], fill="#fff7ed", stroke="#ea580c"),
            _svg_box(588, 278, 230, 110, "Optimizer", ["Adam/AdamW/Lion", f"train_backbone={spec.train_backbone}", f"updated params={_human_params(spec.trainable_params)}"], fill="#fdf2f8", stroke="#db2777"),
            _svg_box(868, 278, 250, 110, "Checkpoint/Eval", ["save model or prior adapter", "optional train/val eval", "PSNR/SSIM one-step"], fill="#f1f5f9", stroke="#64748b"),
            _svg_arrow(268, 145, 318, 145),
            _svg_arrow(538, 145, 588, 145),
            _svg_arrow(818, 145, 868, 145),
            _svg_arrow(993, 208, 428, 278, "prediction"),
            _svg_arrow(538, 333, 588, 333),
            _svg_arrow(818, 333, 868, 333),
        ]
        return _svg_doc(1160, 430, "Training structure and data flow", "\n".join(boxes))

    boxes = [
        _svg_box(38, 82, 230, 126, "Dataset Split", ["LQ image", "GT image", "Qwen prior [21]", f"batch_size={spec.batch_size}"], fill="#eef2ff", stroke="#6366f1"),
        _svg_box(318, 82, 240, 126, "Checkpoint Load", ["checkpoint_load", "full model or prior state", "base checkpoint optional"], fill="#ecfeff", stroke="#0891b2"),
        _svg_box(608, 82, 250, 126, "Eval Model", ["AssessConditionedTPGDUNet", "model.eval()", "no optimizer update", f"params={_human_params(spec.total_params)}"], fill="#f0fdf4", stroke="#16a34a"),
        _svg_box(908, 82, 220, 126, "IR-SDE Eval", ["generate states", "predict score", "reverse one step"], fill="#fefce8", stroke="#ca8a04"),
        _svg_box(318, 278, 240, 110, "Restored Output", ["crop boundary", "compare with GT", "range [-1, 1] -> [0, 1]"], fill="#fff7ed", stroke="#ea580c"),
        _svg_box(608, 278, 250, 110, "Metrics", ["loss", "PSNR one-step", "SSIM one-step"], fill="#fdf2f8", stroke="#db2777"),
        _svg_box(908, 278, 220, 110, "Result JSON", ["metrics_<split>.json", "config/checkpoint recorded", "no training state change"], fill="#f1f5f9", stroke="#64748b"),
        _svg_arrow(268, 145, 318, 145),
        _svg_arrow(558, 145, 608, 145),
        _svg_arrow(858, 145, 908, 145),
        _svg_arrow(1018, 208, 438, 278, "restored"),
        _svg_arrow(558, 333, 608, 333),
        _svg_arrow(858, 333, 908, 333),
    ]
    return _svg_doc(1168, 430, "Testing structure and data flow", "\n".join(boxes))


def _mermaid_core(spec: ModelSpec) -> str:
    prior = "QwenPromptPriorAdapter" if spec.use_structured_degra_prior else "AssessPriorAdapter"
    return f"""```mermaid
flowchart LR
  xt["xt/state [B,{spec.in_nc},H,W]"] --> init["concat + init_conv"]
  cond["cond/LQ [B,{spec.in_nc},H,W]"] --> init
  time["time [B]"] --> tm["time_mlp"]
  prior_in["prior input ({spec.degradation_prior_source})"] --> prior["{prior}"]
  prior --> deg["deg_context [B,{spec.context_dim}]"]
  prior --> struct["struct_tokens [B,{spec.structured_num_tokens},{spec.struct_context_dim}]"]
  init --> down["ConditionalUNet down blocks"]
  tm --> down
  deg --> down
  struct --> down
  down --> mid["middle ResBlock + attention"]
  mid --> up["up blocks + skip connections"]
  up --> out["final Conv2d -> output [B,{spec.out_nc},H,W]"]
```
"""


def _html_index(out_dir: Path, train_spec: ModelSpec, test_spec: ModelSpec) -> str:
    def img(name: str, title: str) -> str:
        return f"<h2>{html.escape(title)}</h2><img src='{html.escape(name)}' alt='{html.escape(title)}'/>"

    style = """
body { font-family: Inter, Arial, sans-serif; margin: 28px; color: #172033; background: #f8fafc; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 30px; }
img { max-width: 100%; border: 1px solid #dbe3ef; background: white; }
pre { background: #0f172a; color: #e2e8f0; padding: 16px; overflow-x: auto; }
.meta { color: #475569; }
"""
    summary = json.dumps({"train": asdict(train_spec), "test": asdict(test_spec)}, indent=2)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Assess TPGD Model Graphs</title>
  <style>{style}</style>
</head>
<body>
  <h1>Assess TPGD Model Graphs</h1>
  <p class="meta">Generated in {html.escape(str(out_dir.resolve()))}</p>
  {img("train_core_model.svg", "Train Core Model")}
  {img("train_pipeline.svg", "Train Pipeline")}
  {img("test_core_model.svg", "Test Core Model")}
  {img("test_pipeline.svg", "Test Pipeline")}
  <h2>Summary JSON</h2>
  <pre>{html.escape(summary)}</pre>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Assess-TPGD train/test model structure diagrams.")
    parser.add_argument("--train-config", type=Path, default=ROOT / "config" / "train" / "legacy" / "sample.yml")
    parser.add_argument("--test-config", type=Path, default=ROOT / "config" / "test" / "batch.yml")
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "model_graphs")
    parser.add_argument("--module-depth", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_spec = _train_spec(args.train_config.expanduser().resolve())
    test_spec = _test_spec(args.test_config.expanduser().resolve(), args.test_split)
    train_model = _build_model(train_spec)
    test_model = _build_model(test_spec)
    _fill_param_counts(train_spec, train_model)
    _fill_param_counts(test_spec, test_model)

    outputs = {
        "train_core_model.svg": _core_model_svg(train_spec),
        "test_core_model.svg": _core_model_svg(test_spec),
        "train_pipeline.svg": _pipeline_svg(train_spec),
        "test_pipeline.svg": _pipeline_svg(test_spec),
        "train_core_model.mmd": _mermaid_core(train_spec),
        "test_core_model.mmd": _mermaid_core(test_spec),
        "train_module_tree.txt": _module_tree(train_model, max_depth=args.module_depth),
        "test_module_tree.txt": _module_tree(test_model, max_depth=args.module_depth),
        "summary.json": json.dumps({"train": asdict(train_spec), "test": asdict(test_spec)}, indent=2) + "\n",
    }
    for filename, content in outputs.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    (out_dir / "index.html").write_text(_html_index(out_dir, train_spec, test_spec), encoding="utf-8")

    print(f"output_dir={out_dir}")
    for filename in sorted(outputs):
        print(out_dir / filename)
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
