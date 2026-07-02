"""Assessment Reasoning hidden prior + TPGDiff SDE training.

Paired LQ/GT images plus precomputed Assessment Reasoning hidden states are fed
into a TPGDiff ConditionalUNet whose degradation context is produced by
`AssessPriorAdapter`. The default objective follows TPGDiff's IR-SDE training
loss; direct-GT training is also available for plain supervised ablations.
"""

from __future__ import annotations

import copy
import contextlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from module.legacy.source_paths import source_repo
from module.degradation_prompt import AssessPriorAdapter
from module.degradation_prompt import load_assessment_hidden, select_hidden
from module.layout_prompt import (
    DEGRADATION_ORDER,
    StructuredPriorV2,
    build_prior_from_degradation_name,
    normalize_degradation_name,
)
from module.backbone import (
    AssessConditionedTPGDUNet,
    PlainTPGDUNet,
    TPGDBackboneConfig,
    load_tpgd_unet_weights,
)

HiddenKey = Literal["prefix_hidden", "generated_hidden", "condition_hidden"]
TrainObjective = Literal["sde", "direct_gt", "direct_mse"]
LossType = Literal["l1", "l2"]
PredictionTarget = Literal["image", "residual"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QWEN_V2_PRIOR_SOURCES = {"qwen_probs", "qwen_probs_severity", "confidence_gate"}
STRUCTURED_PRIOR_SOURCES = {"qwen_prompt", "zero_prior", "oracle_type", *QWEN_V2_PRIOR_SOURCES}
SUPPORTED_PRIOR_SOURCES = {"assessment_hidden", "tpgd", "none", *STRUCTURED_PRIOR_SOURCES}
STRUCTURED_PRIOR_VARIANTS = {"correct", "zero", "uniform", "shuffled", "forced_wrong"}
BEST_VAL_PSNR_KEY = "eval_val/psnr_macro"
BEST_VAL_PSNR_WANDB_KEY = "eval_val/best_psnr_macro"


@dataclass
class AssessTPGDConfig:
    hidden_path: Path
    target_prior_dim: int
    hidden_key: HiddenKey = "condition_hidden"
    pool: str = "mean"


@dataclass
class AssessTPGDTrainConfig:
    lq_dir: Path | list[Path]
    gt_dir: Path | list[Path]
    output_dir: Path
    tpgd_options: Path | None = None
    tpgd_checkpoint: Path | None = None
    tpgd_inline_options: dict[str, Any] | None = None
    hidden_dir: Path | list[Path] | None = None
    hidden_path: Path | None = None
    hidden_key: HiddenKey = "condition_hidden"
    image_size: int = 128
    batch_size: int = 1
    epochs: int = 1
    max_steps: int = 100
    lr: float = 1e-4
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda"
    train_backbone: bool = False
    load_checkpoint: bool = True
    strict_load: bool = False
    adapter_hidden_dim: int = 1024
    adapter_pool: str = "mean"
    adapter_dropout: float = 0.0
    random_content_context: bool = False
    seed: int = 1234
    degradation_prior_source: str = "assessment_hidden"
    use_content_prior: bool = False
    use_structure_prior: bool = False
    train_structure_prior: bool = False
    prior_checkpoint: Path | None = None
    objective: TrainObjective = "sde"
    prediction_target: PredictionTarget = "image"
    loss_type: LossType = "l1"
    loss_weight: float = 1.0
    boundary_pad: int = 32
    direct_gt_time: float = 0.0
    sde_max_sigma: float | None = None
    sde_T: int | None = None
    sde_schedule: str | None = None
    sde_eps: float | None = None
    sde_t_start: int = 1
    sde_t_end: int = -1
    save_every: int = 100
    log_every: int = 10
    save_full_model: bool = False
    lr_scheduler: str = "none"
    lr_min: float = 0.0
    lr_step_size: int = 10000
    lr_gamma: float = 0.5
    lr_milestones: list[int] | None = None
    lr_t_max: int | str = "auto"
    lr_warmup_enabled: bool = False
    lr_warmup_steps: int = 0
    lr_warmup_start_factor: float = 0.01
    val_lq_dir: Path | list[Path] | None = None
    val_gt_dir: Path | list[Path] | None = None
    val_hidden_dir: Path | list[Path] | None = None
    val_hidden_path: Path | None = None
    structured_prior_root: Path | None = None
    val_structured_prior_root: Path | None = None
    structured_prior_temperature: float = 1.0
    structured_prior_variant: str = "correct"
    condition_dropout_probability: float = 0.0
    prior_corruption_probability: float = 0.0
    structured_confidence_override: float | None = None
    eval_enabled: bool = False
    eval_every_steps: int = 0
    eval_every_epochs: int = 0
    eval_batch_size: int = 0
    eval_train_max_batches: int = 10
    eval_val_max_batches: int = 0
    eval_train_sampling_mode: str = "none"
    eval_train_fraction_per_task: float = 0.0
    eval_train_sampling_seed: int = 1234
    use_wandb: bool = False
    wandb_project: str = "MyFusion_AssessTPGD"
    wandb_run_name: str | None = None


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_tpgd_setting(path: Path | None) -> dict[str, Any]:
    return _load_tpgd_options(path).get("network_G", {}).get("setting", {})


def _load_tpgd_options(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_symbol_from_file(path: Path, symbol: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_qwen_ir_vendor_{path.stem}_{symbol}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {symbol} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _load_irsde_cls() -> type:
    sde_path = source_repo("tpgdiff").path / "universal-restoration" / "utils" / "sde_utils.py"
    return _load_symbol_from_file(sde_path, "IRSDE")


def _matching_loss(predict: torch.Tensor, target: torch.Tensor, loss_type: LossType) -> torch.Tensor:
    if loss_type == "l1":
        loss = torch.nn.functional.l1_loss(predict, target, reduction="none")
    elif loss_type == "l2":
        loss = torch.nn.functional.mse_loss(predict, target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss.flatten(start_dim=1).mean(dim=1).mean()


def _direct_restored(
    prediction: torch.Tensor,
    lq: torch.Tensor,
    prediction_target: PredictionTarget,
) -> torch.Tensor:
    """Convert a direct-model prediction into the restored image tensor."""
    if prediction_target == "image":
        return prediction
    if prediction_target == "residual":
        return lq + prediction
    raise ValueError(f"Unsupported prediction_target: {prediction_target}")


def _direct_time(batch_size: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((batch_size,), float(value), device=device)


def _boundary_pad_hw(tensor: torch.Tensor, pad: int) -> tuple[int, int]:
    if pad <= 0:
        return 0, 0
    height, width = tensor.shape[-2:]
    pad_h = min(int(pad), max(height - 1, 0))
    pad_w = min(int(pad), max(width - 1, 0))
    return pad_h, pad_w


def _apply_boundary_pad(tensor: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h <= 0 and pad_w <= 0:
        return tensor
    return F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect")


def _pad_for_boundary(tensor: torch.Tensor, pad: int) -> tuple[torch.Tensor, tuple[int, int]]:
    pad_hw = _boundary_pad_hw(tensor, pad)
    return _apply_boundary_pad(tensor, pad_hw), pad_hw


def _crop_boundary(tensor: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        tensor = tensor[..., pad_h:-pad_h, :]
    if pad_w > 0:
        tensor = tensor[..., :, pad_w:-pad_w]
    return tensor


def _cfg_value(config_value: Any, options: dict[str, Any], key: str, default: Any) -> Any:
    if config_value is not None:
        return config_value
    return options.get(key, default)


def _build_irsde(config: AssessTPGDTrainConfig, tpgd_options: dict[str, Any], device: torch.device):
    sde_options = tpgd_options.get("sde", {})
    irsde_cls = _load_irsde_cls()
    return irsde_cls(
        max_sigma=float(_cfg_value(config.sde_max_sigma, sde_options, "max_sigma", 50)),
        T=int(_cfg_value(config.sde_T, sde_options, "T", 100)),
        schedule=str(_cfg_value(config.sde_schedule, sde_options, "schedule", "cosine")),
        eps=float(_cfg_value(config.sde_eps, sde_options, "eps", 0.005)),
        device=device,
    )


def _tpgd_universal_root() -> Path:
    return source_repo("tpgdiff").path / "universal-restoration"


def _tpgd_code_root() -> Path:
    return _tpgd_universal_root() / "config" / "tpgd-sde"


@contextlib.contextmanager
def _push_sys_path(path: Path):
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


def _infer_prior_num_degradations(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported prior checkpoint type: {type(checkpoint)!r}")
    for key in ("deg_head.cls.weight", "module.deg_head.cls.weight"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[0])
    raise KeyError(f"Cannot infer prior degradation count from {checkpoint_path}")


def _build_content_prior_model(checkpoint_path: Path, device: torch.device):
    with _push_sys_path(_tpgd_universal_root()):
        import open_clip
        from open_clip.prior_stage_model import PriorStageModel

    base_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device, precision="fp32")
    teacher_encoder = base_model.visual
    student_encoder = copy.deepcopy(base_model.visual)
    deg_backbone = copy.deepcopy(base_model.visual)

    if hasattr(base_model.visual, "output_dim"):
        embed_dim = base_model.visual.output_dim
    elif hasattr(base_model, "embed_dim"):
        embed_dim = base_model.embed_dim
    else:
        raise RuntimeError("Cannot infer embed_dim from open_clip visual model")

    prior_model = PriorStageModel(
        teacher_encoder=teacher_encoder,
        student_encoder=student_encoder,
        deg_backbone=deg_backbone,
        embed_dim=embed_dim,
        num_degradations=_infer_prior_num_degradations(checkpoint_path),
        content_loss_weight=1.0,
        deg_loss_weight=1.0,
        use_cosine_distill=True,
        normalize_embedding=True,
        freeze_teacher=True,
        freeze_deg_backbone=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if len(state_dict) and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    prior_model.load_state_dict(state_dict, strict=True)
    prior_model.eval()
    for param in prior_model.parameters():
        param.requires_grad = False
    return prior_model


def _build_structure_prior(tpgd_options: dict[str, Any], checkpoint_path: Path | None, device: torch.device, strict: bool) -> torch.nn.Module | None:
    sp_options = tpgd_options.get("structure_prior")
    if not sp_options:
        return None
    with _push_sys_path(_tpgd_code_root()):
        from models import modules as tpgd_modules

    which_model = sp_options.get("which_model", "StructurePriorModule")
    setting = sp_options.get("setting", {})
    struct_prior = getattr(tpgd_modules, which_model)(**setting).to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("SP"), dict):
            incompatible = struct_prior.load_state_dict(checkpoint["SP"], strict=strict)
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
            print(f"structure_prior_checkpoint={checkpoint_path} missing={missing} unexpected={unexpected}", flush=True)
        else:
            print(f"structure_prior_checkpoint=no_SP path={checkpoint_path}", flush=True)
    return struct_prior


def _build_optimizer(params: list[torch.nn.Parameter], config: AssessTPGDTrainConfig) -> torch.optim.Optimizer:
    optimizer = config.optimizer.lower()
    kwargs = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "betas": (config.beta1, config.beta2),
    }
    if optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if optimizer == "adam":
        return torch.optim.Adam(params, **kwargs)
    if optimizer == "lion":
        optimizer_path = source_repo("tpgdiff").path / "universal-restoration" / "config" / "tpgd-sde" / "models" / "optimizer.py"
        lion_cls = _load_symbol_from_file(optimizer_path, "Lion")
        return lion_cls(params, **kwargs)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}. Use Adam, AdamW, or Lion.")


def _planned_train_steps(config: AssessTPGDTrainConfig, steps_per_epoch: int) -> int:
    epoch_steps = max(0, int(config.epochs)) * max(1, int(steps_per_epoch))
    if config.max_steps and config.max_steps > 0:
        return min(epoch_steps, int(config.max_steps)) if epoch_steps > 0 else int(config.max_steps)
    return epoch_steps


def _resolve_lr_t_max(config: AssessTPGDTrainConfig, planned_steps: int) -> int:
    value = config.lr_t_max
    if value in (None, "", 0, "0"):
        return max(1, planned_steps)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return max(1, planned_steps)
    return max(1, int(float(value)))


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, config: AssessTPGDTrainConfig, planned_steps: int):
    scheduler = str(config.lr_scheduler or "none").strip().lower().replace("-", "_")
    warmup_steps = max(0, int(config.lr_warmup_steps)) if config.lr_warmup_enabled else 0
    warmup_start = float(config.lr_warmup_start_factor)
    warmup_start = min(max(warmup_start, 0.0), 1.0)
    t_max = _resolve_lr_t_max(config, planned_steps)

    if scheduler in {"", "none", "constant", "off"} and warmup_steps <= 0:
        return None
    if scheduler not in {"", "none", "constant", "off", "cosine", "cosine_annealing", "cosineannealing", "step", "step_lr", "steplr", "multistep", "multi_step", "multistep_lr", "multisteplr", "exponential", "exp", "exponential_lr"}:
        raise ValueError("Unsupported lr scheduler: " + str(config.lr_scheduler) + ". Use none, cosine, step, multistep, or exponential.")

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def factor_for_group(base_lr: float):
        min_factor = 0.0 if base_lr <= 0 else min(float(config.lr_min) / float(base_lr), 1.0)

        def lr_lambda(step: int) -> float:
            # LambdaLR calls this with step=0 during initialization.
            if warmup_steps > 0 and step < warmup_steps:
                alpha = step / max(1, warmup_steps)
                return warmup_start + (1.0 - warmup_start) * alpha

            post_step = max(0, step - warmup_steps)
            post_total = max(1, t_max - warmup_steps)
            if scheduler in {"", "none", "constant", "off"}:
                return 1.0
            if scheduler in {"cosine", "cosine_annealing", "cosineannealing"}:
                progress = min(post_step / post_total, 1.0)
                return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
            if scheduler in {"step", "step_lr", "steplr"}:
                return float(config.lr_gamma) ** (post_step // max(1, int(config.lr_step_size)))
            if scheduler in {"multistep", "multi_step", "multistep_lr", "multisteplr"}:
                milestones = config.lr_milestones or [config.lr_step_size]
                passed = sum(post_step >= int(item) for item in milestones)
                return float(config.lr_gamma) ** passed
            if scheduler in {"exponential", "exp", "exponential_lr"}:
                return float(config.lr_gamma) ** post_step
            return 1.0

        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[factor_for_group(lr) for lr in base_lrs])


def _to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5


def _safe_output_component(value: Any, default: str) -> str:
    text = str(value or default).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return cleaned.strip("-_.") or default


def _save_restored_batch(restored: torch.Tensor, batch: dict[str, Any], output_dir: Path) -> int:
    """Save a restored [-1, 1] tensor batch as RGB PNG files."""

    images = (
        _to_01(restored.detach())
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    names = batch.get("name") or [f"sample_{idx:06d}" for idx in range(len(images))]
    degradations = batch.get("degradation") or ["unknown"] * len(images)
    saved = 0
    for idx, array in enumerate(images):
        degradation = _safe_output_component(degradations[idx], "unknown")
        name = _safe_output_component(names[idx], f"sample_{idx:06d}")
        destination = output_dir / degradation / f"{name}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(destination)
        saved += 1
    return saved


def _psnr_per_image_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_01 = _to_01(pred)
    target_01 = _to_01(target)
    mse = torch.nn.functional.mse_loss(pred_01, target_01, reduction="none").flatten(start_dim=1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _psnr_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _psnr_per_image_minus1_1(pred, target).mean()


def _ssim_per_image_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_01 = _to_01(pred).float()
    target_01 = _to_01(target).float()
    channels = pred_01.shape[1]
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=pred_01.device, dtype=pred_01.dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] @ gauss[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(channels, 1, window_size, window_size)

    padding = window_size // 2
    mu_x = torch.nn.functional.conv2d(pred_01, window, padding=padding, groups=channels)
    mu_y = torch.nn.functional.conv2d(target_01, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(pred_01 * pred_01, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(target_01 * target_01, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(pred_01 * target_01, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)).clamp_min(1e-12)
    return ssim_map.flatten(start_dim=1).mean(dim=1)


def _ssim_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _ssim_per_image_minus1_1(pred, target).mean()


class _WandbLogger:
    def __init__(self, config: AssessTPGDTrainConfig) -> None:
        self.run = None
        if not config.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            print("wandb=unavailable", flush=True)
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
        )

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _read_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size > 0:
            image = image.resize((image_size, image_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _read_clip_image(path: Path, resolution: int = 224) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = tensor.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return (tensor - mean) / std


def _find_gt(lq_path: Path, lq_dir: Path, gt_dir: Path) -> Path:
    rel = lq_path.relative_to(lq_dir)
    candidates = [gt_dir / rel, gt_dir / lq_path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing GT for {lq_path}. Tried: {candidates}")


def _build_hidden_index(hidden_dir: Path | None) -> dict[str, list[Path]]:
    if hidden_dir is None:
        return {}
    if not hidden_dir.exists():
        raise FileNotFoundError(f"hidden_dir does not exist: {hidden_dir}")
    index: dict[str, list[Path]] = {}
    for path in sorted(hidden_dir.rglob("*.pt")):
        name = path.name
        stem = name.split("_round", 1)[0]
        stem = stem.removesuffix("_assessment_reasoning_hidden")
        index.setdefault(stem, []).append(path)
        index.setdefault(path.stem, []).append(path)
    return index


def _find_hidden(stem: str, hidden_index: dict[str, list[Path]]) -> Path:
    if stem in hidden_index:
        return hidden_index[stem][0]
    for key, paths in hidden_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return paths[0]
    raise FileNotFoundError(f"Missing Assessment hidden-state file for image stem: {stem}")


def _infer_degradation_from_lq_dir(lq_dir: Path) -> str:
    # TPGDiff convention: <degradation>/LQ/*.png. Direct custom LQ roots fall
    # back to their own folder name.
    if lq_dir.name.lower() == "lq" and lq_dir.parent.name:
        return lq_dir.parent.name
    return lq_dir.name


class PairedAssessmentDataset(Dataset):
    def __init__(
        self,
        lq_dir: Path,
        gt_dir: Path,
        *,
        hidden_dir: Path | None = None,
        hidden_path: Path | None = None,
        hidden_key: HiddenKey = "condition_hidden",
        image_size: int = 128,
        require_hidden: bool = True,
        structured_prior_root: Path | None = None,
        structured_prior_mode: str = "qwen_prompt",
        structured_prior_temperature: float = 1.0,
    ) -> None:
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.hidden_path = hidden_path
        self.hidden_key = hidden_key
        self.image_size = image_size
        self.require_hidden = require_hidden
        self.structured_prior_root = structured_prior_root
        self.structured_prior_mode = str(structured_prior_mode).strip().lower()
        self.structured_prior_temperature = float(structured_prior_temperature)
        if self.structured_prior_temperature <= 0:
            raise ValueError("structured_prior_temperature must be positive")
        self.degradation_name = _infer_degradation_from_lq_dir(lq_dir)
        self.dataset_root = lq_dir.parents[2] if lq_dir.name.lower() == "lq" and len(lq_dir.parents) >= 3 else lq_dir.parent

        if not lq_dir.exists():
            raise FileNotFoundError(f"lq_dir does not exist: {lq_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
        if self.require_hidden and hidden_path is None and hidden_dir is None:
            raise ValueError("Either hidden_dir or hidden_path must be set for Assess-TPGD training.")
        if self.require_hidden and hidden_path is not None and not hidden_path.exists():
            raise FileNotFoundError(f"hidden_path does not exist: {hidden_path}")

        self.hidden_index = {} if hidden_path is not None else (_build_hidden_index(hidden_dir) if self.require_hidden else {})
        self.samples: list[tuple[Path, Path, Path | None]] = []
        for lq_path in _image_files(lq_dir):
            gt_path = _find_gt(lq_path, lq_dir, gt_dir)
            sample_hidden = (hidden_path or _find_hidden(lq_path.stem, self.hidden_index)) if self.require_hidden else None
            self.samples.append((lq_path, gt_path, sample_hidden))
        if not self.samples:
            raise RuntimeError(f"No image pairs found under {lq_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_structured_prior(self, lq_path: Path) -> torch.Tensor:
        if self.structured_prior_mode == "zero_prior":
            return torch.zeros(21, dtype=torch.float32)
        if self.structured_prior_mode == "oracle_type":
            return build_prior_from_degradation_name(self.degradation_name)
        if self.structured_prior_mode not in {"qwen_prompt", *QWEN_V2_PRIOR_SOURCES}:
            # The tensor is always collated for a uniform batch schema, but it
            # is ignored by non-structured-prior experiments.
            return torch.zeros(21, dtype=torch.float32)
        if self.structured_prior_root is None:
            # Backward compatibility for older oracle configs that selected
            # qwen_prompt without providing a JSON root.
            if self.structured_prior_mode == "qwen_prompt":
                return build_prior_from_degradation_name(self.degradation_name)
            raise ValueError(
                f"structured_prior_root is required for {self.structured_prior_mode}"
            )
        rel = lq_path.relative_to(self.dataset_root).with_suffix(".json")
        prior_path = self.structured_prior_root / rel
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing Qwen structured prior for {lq_path}: {prior_path}")
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        if self.structured_prior_mode in QWEN_V2_PRIOR_SOURCES:
            prior_v2 = StructuredPriorV2.from_qwen_payload(
                payload,
                temperature=self.structured_prior_temperature,
            )
            return prior_v2.to_model_vector(self.structured_prior_mode)
        vector = payload.get("prior_vector_21")
        if not isinstance(vector, list) or len(vector) != 21:
            raise ValueError(f"Invalid prior_vector_21 in {prior_path}")
        return torch.tensor(vector, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lq_path, gt_path, hidden_path = self.samples[index]
        hidden = None
        if hidden_path is not None:
            pack = load_assessment_hidden(hidden_path, map_location="cpu")
            hidden = select_hidden(pack, self.hidden_key)
            if hidden.ndim == 3:
                if hidden.shape[0] != 1:
                    raise ValueError(f"Expected hidden batch 1 in {hidden_path}, got {tuple(hidden.shape)}")
                hidden = hidden[0]
            if hidden.ndim != 2:
                raise ValueError(f"Expected hidden [T, C] or [1, T, C], got {tuple(hidden.shape)} in {hidden_path}")
        structured_prior = self._load_structured_prior(lq_path)
        has_prior_metadata = self.structured_prior_mode in QWEN_V2_PRIOR_SOURCES
        prior_confidence = float(structured_prior[5:10].max()) if has_prior_metadata else 0.0
        prior_correct = False
        if has_prior_metadata:
            expected = normalize_degradation_name(self.degradation_name)
            prior_correct = int(structured_prior[5:10].argmax()) == DEGRADATION_ORDER.index(expected)
        return {
            "lq": _read_image(lq_path, self.image_size),
            "gt": _read_image(gt_path, self.image_size),
            "lq_clip": _read_clip_image(lq_path),
            "hidden": hidden,
            "structured_prior": structured_prior,
            "prior_confidence": prior_confidence,
            "prior_correct": prior_correct,
            "has_prior_metadata": has_prior_metadata,
            "degradation": self.degradation_name,
            "name": lq_path.stem,
            "hidden_path": str(hidden_path) if hidden_path is not None else "",
        }


def assessment_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lq = torch.stack([item["lq"] for item in batch], dim=0)
    gt = torch.stack([item["gt"] for item in batch], dim=0)
    lq_clip = torch.stack([item["lq_clip"] for item in batch], dim=0)
    structured_prior = torch.stack([item["structured_prior"] for item in batch], dim=0)
    prior_confidence = torch.tensor([item["prior_confidence"] for item in batch], dtype=torch.float32)
    prior_correct = torch.tensor([item["prior_correct"] for item in batch], dtype=torch.bool)
    has_prior_metadata = torch.tensor([item["has_prior_metadata"] for item in batch], dtype=torch.bool)
    hidden_items = [item["hidden"] for item in batch]
    if all(item_hidden is None for item_hidden in hidden_items):
        hidden = None
        mask = None
    elif any(item_hidden is None for item_hidden in hidden_items):
        raise ValueError("Batch mixes samples with and without Assessment hidden states")
    else:
        hidden_list = [item_hidden for item_hidden in hidden_items if item_hidden is not None]
        max_len = max(hidden.shape[0] for hidden in hidden_list)
        hidden_dim = hidden_list[0].shape[1]
        hidden = hidden_list[0].new_zeros(len(batch), max_len, hidden_dim)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for idx, item_hidden in enumerate(hidden_list):
            length = item_hidden.shape[0]
            hidden[idx, :length] = item_hidden
            mask[idx, :length] = True
    return {
        "lq": lq,
        "gt": gt,
        "lq_clip": lq_clip,
        "hidden": hidden,
        "mask": mask,
        "structured_prior": structured_prior,
        "prior_confidence": prior_confidence,
        "prior_correct": prior_correct,
        "has_prior_metadata": has_prior_metadata,
        "degradation": [item["degradation"] for item in batch],
        "name": [item["name"] for item in batch],
        "hidden_path": [item["hidden_path"] for item in batch],
    }


def _forced_wrong_prior(prior: torch.Tensor, degradation: str) -> torch.Tensor:
    """Preserve probability values/confidence but force top-1 to a wrong class."""
    result = prior.clone()
    probs = prior[5:10]
    expected = DEGRADATION_ORDER.index(normalize_degradation_name(degradation))
    wrong = (expected + 1) % len(DEGRADATION_ORDER)
    ranked = torch.sort(probs, descending=True).values
    remaining = [index for index in range(len(DEGRADATION_ORDER)) if index != wrong]
    result[5:10].zero_()
    result[5 + wrong] = ranked[0]
    for index, value in zip(remaining, ranked[1:]):
        result[5 + index] = value

    original_top = int(probs.argmax())
    shift = wrong - original_top
    result[0:5] = torch.roll(prior[0:5], shifts=shift, dims=0)
    result[10:20].zero_()
    return result


class StructuredPriorVariantDataset(Dataset):
    """Opt-in test-time prior perturbations without modifying cached Qwen JSON."""

    def __init__(self, dataset: Dataset, variant: str) -> None:
        self.dataset = dataset
        self.variant = str(variant).strip().lower()
        if self.variant not in STRUCTURED_PRIOR_VARIANTS:
            raise ValueError(
                f"structured_prior_variant must be one of {sorted(STRUCTURED_PRIOR_VARIANTS)}, "
                f"got {variant!r}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        prior = item["structured_prior"]
        if self.variant == "correct":
            return item
        if self.variant == "zero":
            item["structured_prior"] = torch.zeros_like(prior)
        elif self.variant == "uniform":
            uniform = torch.zeros_like(prior)
            uniform[5:10] = 1.0 / len(DEGRADATION_ORDER)
            item["structured_prior"] = uniform
        elif self.variant == "shuffled":
            # Five-task datasets are concatenated task-wise. A one-task-block
            # shift guarantees a donor from another task for the standard
            # sample and fixed-5x40 protocols.
            donor_offset = max(1, len(self.dataset) // len(DEGRADATION_ORDER))
            donor = self.dataset[(index + donor_offset) % len(self.dataset)]
            item["structured_prior"] = donor["structured_prior"].clone()
        elif self.variant == "forced_wrong":
            item["structured_prior"] = _forced_wrong_prior(prior, item["degradation"])
        return item


def load_prior_from_hidden(config: AssessTPGDConfig, device: str | torch.device = "cpu") -> torch.Tensor:
    pack = load_assessment_hidden(config.hidden_path, map_location="cpu")
    hidden = select_hidden(pack, config.hidden_key).to(device)
    adapter = AssessPriorAdapter(output_dim=config.target_prior_dim, pool=config.pool).to(device)
    return adapter(hidden)


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    epoch: int,
    config: AssessTPGDTrainConfig,
    structure_prior: torch.nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    best_metric_step: int | None = None,
    best_only: bool = False,
) -> Path:
    def serialize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    save_full_state = config.save_full_model or config.train_backbone
    if save_full_state:
        state_key = "model"
        state_value = model.state_dict()
    elif config.degradation_prior_source in STRUCTURED_PRIOR_SOURCES:
        state_key = "qwen_prompt_prior"
        if model.structured_prior is None:
            raise RuntimeError("Cannot save qwen_prompt prior adapter because it is disabled")
        state_value = model.structured_prior.state_dict()
    else:
        state_key = "assess_prior"
        state_value = model.assess_prior.state_dict()
    payload = {
        state_key: state_value,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "save_full_model": save_full_state,
        "config": {key: serialize(value) for key, value in config.__dict__.items()},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if structure_prior is not None:
        payload["structure_prior"] = structure_prior.state_dict()
    if best_metric_name is not None and best_metric_value is not None:
        payload["best_metric_name"] = str(best_metric_name)
        payload["best_metric_value"] = float(best_metric_value)
        payload["best_metric_step"] = int(best_metric_step if best_metric_step is not None else step)
    if best_only:
        best = output_dir / "best.pt"
        torch.save(payload, best)
        return best
    latest = output_dir / "latest.pt"
    torch.save(payload, latest)
    numbered = output_dir / f"step_{step:06d}.pt"
    torch.save(payload, numbered)
    return latest


def _as_path_list(value: Path | list[Path] | None, *, name: str) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value
    return [value]


def _build_paired_dataset(
    *,
    lq_dir: Path | list[Path] | None,
    gt_dir: Path | list[Path] | None,
    hidden_dir: Path | list[Path] | None,
    hidden_path: Path | None,
    hidden_key: HiddenKey,
    image_size: int,
    require_hidden: bool,
    split_name: str,
    structured_prior_root: Path | None = None,
    structured_prior_mode: str = "qwen_prompt",
    structured_prior_temperature: float = 1.0,
) -> Dataset | None:
    if lq_dir is None or gt_dir is None:
        return None
    lq_dirs = _as_path_list(lq_dir, name=f"{split_name}.lq_dir")
    gt_dirs = _as_path_list(gt_dir, name=f"{split_name}.gt_dir")
    if len(lq_dirs) != len(gt_dirs):
        raise ValueError(f"{split_name} lq/gt directory counts differ: {len(lq_dirs)} vs {len(gt_dirs)}")

    if hidden_path is not None:
        hidden_dirs: list[Path | None] = [None] * len(lq_dirs)
    elif require_hidden:
        hidden_dirs = _as_path_list(hidden_dir, name=f"{split_name}.hidden_dir")
        if len(hidden_dirs) != len(lq_dirs):
            raise ValueError(f"{split_name} hidden/lq directory counts differ: {len(hidden_dirs)} vs {len(lq_dirs)}")
    else:
        hidden_dirs = [None] * len(lq_dirs)

    datasets = [
        PairedAssessmentDataset(
            lq_path,
            gt_path,
            hidden_dir=hidden_path_item,
            hidden_path=hidden_path,
            hidden_key=hidden_key,
            image_size=image_size,
            require_hidden=require_hidden,
            structured_prior_root=structured_prior_root,
            structured_prior_mode=structured_prior_mode,
            structured_prior_temperature=structured_prior_temperature,
        )
        for lq_path, gt_path, hidden_path_item in zip(lq_dirs, gt_dirs, hidden_dirs)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _build_train_dataset(config: AssessTPGDTrainConfig) -> Dataset:
    dataset = _build_paired_dataset(
        lq_dir=config.lq_dir,
        gt_dir=config.gt_dir,
        hidden_dir=config.hidden_dir,
        hidden_path=config.hidden_path,
        hidden_key=config.hidden_key,
        image_size=config.image_size,
        require_hidden=config.degradation_prior_source == "assessment_hidden",
        split_name="train",
        structured_prior_root=config.structured_prior_root,
        structured_prior_mode=config.degradation_prior_source,
        structured_prior_temperature=config.structured_prior_temperature,
    )
    if dataset is None:
        raise ValueError("Training dataset is not configured")
    if config.structured_prior_variant != "correct":
        if config.degradation_prior_source not in QWEN_V2_PRIOR_SOURCES:
            raise ValueError("Prior perturbation variants require a Qwen V2 prior source")
        dataset = StructuredPriorVariantDataset(dataset, config.structured_prior_variant)
    return dataset


def _build_val_dataset(config: AssessTPGDTrainConfig) -> Dataset | None:
    dataset = _build_paired_dataset(
        lq_dir=config.val_lq_dir,
        gt_dir=config.val_gt_dir,
        hidden_dir=config.val_hidden_dir,
        hidden_path=config.val_hidden_path,
        hidden_key=config.hidden_key,
        image_size=config.image_size,
        require_hidden=config.degradation_prior_source == "assessment_hidden",
        split_name="val",
        structured_prior_root=config.val_structured_prior_root or config.structured_prior_root,
        structured_prior_mode=config.degradation_prior_source,
        structured_prior_temperature=config.structured_prior_temperature,
    )
    if dataset is not None and config.structured_prior_variant != "correct":
        if config.degradation_prior_source not in QWEN_V2_PRIOR_SOURCES:
            raise ValueError("Prior perturbation variants require a Qwen V2 prior source")
        dataset = StructuredPriorVariantDataset(dataset, config.structured_prior_variant)
    return dataset


def _build_fixed_stratified_subset(
    dataset: Dataset,
    *,
    fraction_per_task: float,
    seed: int,
) -> tuple[Subset, list[int]]:
    if not 0.0 < fraction_per_task <= 1.0:
        raise ValueError(
            "evaluation.train_sampling.fraction_per_task must be in (0, 1], "
            f"got {fraction_per_task}"
        )

    components = list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
    offset = 0
    selected_by_task: list[list[int]] = []
    counts: list[int] = []
    for task_index, component in enumerate(components):
        task_size = len(component)
        sample_count = min(task_size, max(1, math.ceil(task_size * fraction_per_task)))
        rng = random.Random(int(seed) + task_index)
        local_indices = sorted(rng.sample(range(task_size), sample_count))
        selected_by_task.append([offset + index for index in local_indices])
        counts.append(sample_count)
        offset += task_size

    # Interleave tasks so even a later max_batches limit remains approximately
    # class-balanced instead of selecting only the first ConcatDataset member.
    interleaved: list[int] = []
    for position in range(max(counts, default=0)):
        for task_indices in selected_by_task:
            if position < len(task_indices):
                interleaved.append(task_indices[position])
    return Subset(dataset, interleaved), counts


def _evaluate_loader(
    *,
    name: str,
    loader: DataLoader,
    max_batches: int,
    device: torch.device,
    model: AssessConditionedTPGDUNet,
    backbone_cfg: TPGDBackboneConfig,
    content_prior_model: torch.nn.Module | None,
    structure_prior: torch.nn.Module | None,
    config: AssessTPGDTrainConfig,
    sde: Any,
    use_assessment_degra_prior: bool,
    use_tpgd_degra_prior: bool,
    use_content_prior: bool,
    use_structured_degra_prior: bool = False,
    save_images_dir: Path | None = None,
    per_image_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    model_was_training = model.training
    structure_was_training = structure_prior.training if structure_prior is not None else False
    model.eval()
    if structure_prior is not None:
        structure_prior.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    total_batches = 0
    total_saved_images = 0
    task_psnr_sums: dict[str, float] = {}
    task_sample_counts: dict[str, int] = {}
    confidence_values: list[float] = []
    confidence_psnr: list[float] = []
    confidence_ssim: list[float] = []
    qwen_correct_values: list[bool] = []
    gate_confidence_values: list[float] = []
    if save_images_dir is not None:
        save_images_dir.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                lq = batch["lq"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                hidden = batch["hidden"]
                mask = batch["mask"]
                if hidden is not None:
                    hidden = hidden.to(device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)
                structured_prior = batch["structured_prior"].to(device, non_blocking=True) if use_structured_degra_prior else None

                content_context = None
                deg_context_input = None
                lq_clip = batch["lq_clip"].to(device, non_blocking=True) if (use_content_prior or use_tpgd_degra_prior) else None
                if content_prior_model is not None and lq_clip is not None:
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        if use_content_prior:
                            content_context = content_prior_model.get_content_prior(lq_clip).float()
                        if use_tpgd_degra_prior:
                            deg_context_input = content_prior_model.encode_for_degradation(lq_clip).float()
                elif backbone_cfg.use_image_context:
                    content_context = torch.zeros(lq.shape[0], backbone_cfg.context_dim, device=device)

                lq_model, boundary_crop = _pad_for_boundary(lq, config.boundary_pad)
                gt_model = _apply_boundary_pad(gt, boundary_crop)

                struct_tokens = None
                if structure_prior is not None:
                    struct_tokens = structure_prior((lq_model + 1.0) * 0.5)

                if config.objective == "sde":
                    if sde is None:
                        raise RuntimeError("SDE evaluation requested but sde is None")
                    timesteps, states = sde.generate_random_states(
                        x0=gt_model,
                        mu=lq_model,
                        T_start=config.sde_t_start,
                        T_end=config.sde_t_end,
                    )
                    output, _ = model(
                        states,
                        lq_model,
                        timesteps.reshape(-1),
                        assessment_hidden=hidden if use_assessment_degra_prior else None,
                        assessment_mask=mask if use_assessment_degra_prior else None,
                        structured_prior=structured_prior,
                        structured_confidence_override=config.structured_confidence_override,
                        deg_context=deg_context_input,
                        content_context=content_context,
                        struct_tokens=struct_tokens,
                        return_context=True,
                    )
                    score = sde.get_score_from_noise(output, timesteps)
                    restored_model = sde.reverse_sde_step_mean(states, score, timesteps)
                    target_model = sde.reverse_optimum_step(states, gt_model, timesteps)
                    restored = _crop_boundary(restored_model, boundary_crop)
                    target = _crop_boundary(target_model, boundary_crop)
                    loss = config.loss_weight * _matching_loss(restored, target, config.loss_type)
                    ssim_per_image = _ssim_per_image_minus1_1(restored, gt)
                else:
                    time = _direct_time(lq_model.shape[0], device, config.direct_gt_time)
                    output_model, _ = model(
                        lq_model,
                        lq_model,
                        time,
                        assessment_hidden=hidden if use_assessment_degra_prior else None,
                        assessment_mask=mask if use_assessment_degra_prior else None,
                        structured_prior=structured_prior,
                        structured_confidence_override=config.structured_confidence_override,
                        deg_context=deg_context_input,
                        content_context=content_context,
                        struct_tokens=struct_tokens,
                        return_context=True,
                    )
                    restored_model = _direct_restored(
                        output_model,
                        lq_model,
                        config.prediction_target,
                    )
                    restored = _crop_boundary(restored_model, boundary_crop)
                    loss = config.loss_weight * _matching_loss(restored, gt, config.loss_type)
                    ssim_per_image = _ssim_per_image_minus1_1(restored, gt)

                psnr_per_image = _psnr_per_image_minus1_1(restored, gt)
                ssim = ssim_per_image.mean()
                gate_info = (
                    model.structured_prior.last_gate_info
                    if config.degradation_prior_source == "confidence_gate"
                    and model.structured_prior is not None
                    else {}
                )
                gate_confidence_batch = gate_info.get("gate_confidence")
                if gate_confidence_batch is not None:
                    gate_confidence_values.extend(
                        gate_confidence_batch.flatten().detach().cpu().tolist()
                    )

                if per_image_records is not None:
                    names = batch.get("name") or [f"sample_{index:06d}" for index in range(lq.shape[0])]
                    degradations_for_records = batch.get("degradation") or ["unknown"] * lq.shape[0]
                    priors_for_records = batch.get("structured_prior")
                    for record_index in range(lq.shape[0]):
                        has_metadata = bool(batch["has_prior_metadata"][record_index])
                        record = {
                            "name": str(names[record_index]),
                            "degradation": str(degradations_for_records[record_index]),
                            "prior_variant": config.structured_prior_variant,
                            "psnr": float(psnr_per_image[record_index].detach().cpu()),
                            "ssim": float(ssim_per_image[record_index].detach().cpu()),
                        }
                        if priors_for_records is not None:
                            record["used_prior_top1"] = DEGRADATION_ORDER[
                                int(priors_for_records[record_index, 5:10].argmax())
                            ]
                        if has_metadata:
                            record["qwen_confidence"] = float(batch["prior_confidence"][record_index])
                            record["qwen_top1_correct"] = bool(batch["prior_correct"][record_index])
                        if gate_confidence_batch is not None:
                            record["gate_confidence"] = float(
                                gate_confidence_batch[record_index].detach().cpu()
                            )
                        per_image_records.append(record)

                metadata_mask = batch.get("has_prior_metadata")
                if metadata_mask is not None and bool(metadata_mask.any()):
                    selected = metadata_mask.bool()
                    confidence_values.extend(batch["prior_confidence"][selected].tolist())
                    confidence_psnr.extend(psnr_per_image.detach().cpu()[selected].tolist())
                    confidence_ssim.extend(ssim_per_image.detach().cpu()[selected].tolist())
                    qwen_correct_values.extend(batch["prior_correct"][selected].tolist())

                if save_images_dir is not None:
                    total_saved_images += _save_restored_batch(restored, batch, save_images_dir)

                batch_size = int(lq.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_psnr += float(psnr_per_image.detach().sum().cpu())
                total_ssim += float(ssim.detach().cpu()) * batch_size
                degradations = batch.get("degradation") or ["unknown"] * batch_size
                if len(degradations) != batch_size:
                    raise ValueError(f"Expected {batch_size} degradation labels, got {len(degradations)}")
                for degradation, value in zip(degradations, psnr_per_image.detach().cpu().tolist()):
                    task = _safe_output_component(degradation, "unknown")
                    task_psnr_sums[task] = task_psnr_sums.get(task, 0.0) + float(value)
                    task_sample_counts[task] = task_sample_counts.get(task, 0) + 1
                total_samples += batch_size
                total_batches += 1
                if max_batches > 0 and total_batches >= max_batches:
                    break
    finally:
        model.train(model_was_training)
        if structure_prior is not None:
            structure_prior.train(structure_was_training)

    if total_samples == 0:
        empty_metrics = {f"{name}/loss": math.nan, f"{name}/psnr_one_step": math.nan, f"{name}/psnr_macro": math.nan, f"{name}/ssim_one_step": math.nan}
        if save_images_dir is not None:
            empty_metrics[f"{name}/saved_images"] = 0.0
        return empty_metrics
    task_psnr_means = {
        task: task_psnr_sums[task] / task_sample_counts[task]
        for task in task_psnr_sums
    }
    metrics = {
        f"{name}/loss": total_loss / total_samples,
        f"{name}/psnr_one_step": total_psnr / total_samples,
        f"{name}/psnr_macro": sum(task_psnr_means.values()) / len(task_psnr_means),
        f"{name}/ssim_one_step": total_ssim / total_samples,
    }
    metrics.update({f"{name}/psnr_{task}": value for task, value in task_psnr_means.items()})
    if confidence_values:
        count = len(confidence_values)
        bucket_size = max(1, math.ceil(count * 0.2))
        ordered = sorted(range(count), key=lambda index: (confidence_values[index], index))
        buckets = {
            "conf_bottom20": ordered[:bucket_size],
            "conf_middle60": ordered[bucket_size:count - bucket_size],
            "conf_top20": ordered[count - bucket_size:],
            "qwen_correct": [index for index, value in enumerate(qwen_correct_values) if value],
            "qwen_wrong": [index for index, value in enumerate(qwen_correct_values) if not value],
        }
        for bucket, indices in buckets.items():
            metrics[f"{name}/count_{bucket}"] = float(len(indices))
            if indices:
                metrics[f"{name}/psnr_{bucket}"] = sum(confidence_psnr[index] for index in indices) / len(indices)
                metrics[f"{name}/ssim_{bucket}"] = sum(confidence_ssim[index] for index in indices) / len(indices)
        metrics[f"{name}/qwen_top1_accuracy"] = sum(qwen_correct_values) / count
        metrics[f"{name}/confidence_mean"] = sum(confidence_values) / count
        metrics[f"{name}/confidence_p20"] = confidence_values[ordered[min(bucket_size - 1, count - 1)]]
        metrics[f"{name}/confidence_p80"] = confidence_values[ordered[max(0, count - bucket_size)]]
    if gate_confidence_values:
        metrics[f"{name}/gate_confidence_mean"] = sum(gate_confidence_values) / len(gate_confidence_values)
        metrics[f"{name}/gate_confidence_min"] = min(gate_confidence_values)
        metrics[f"{name}/gate_confidence_max"] = max(gate_confidence_values)
    if save_images_dir is not None:
        metrics[f"{name}/saved_images"] = float(total_saved_images)
    return metrics


def test_assess_tpgd(
    config: AssessTPGDTrainConfig,
    *,
    checkpoint_path: Path,
    max_batches: int = 0,
    metrics_name: str = "test",
    save_images_dir: Path | None = None,
    per_image_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    tpgd_options = _load_tpgd_options(config.tpgd_options)
    if config.tpgd_inline_options:
        _deep_update(tpgd_options, config.tpgd_inline_options)
    setting = tpgd_options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()

    if config.degradation_prior_source not in SUPPORTED_PRIOR_SOURCES:
        raise ValueError(f"Unsupported degradation_prior_source: {config.degradation_prior_source}")
    use_assessment_degra_prior = config.degradation_prior_source == "assessment_hidden"
    use_structured_degra_prior = config.degradation_prior_source in STRUCTURED_PRIOR_SOURCES
    use_tpgd_degra_prior = config.degradation_prior_source == "tpgd"
    use_content_prior = bool(config.use_content_prior and backbone_cfg.use_image_context)
    use_structure_prior = bool(config.use_structure_prior and backbone_cfg.use_struct_context)
    use_external_structure_prior = bool(use_structure_prior and not use_structured_degra_prior)

    if config.objective not in ("sde", "direct_gt", "direct_mse"):
        raise ValueError(f"Unsupported objective: {config.objective}")
    if config.prediction_target not in ("image", "residual"):
        raise ValueError(f"Unsupported prediction_target: {config.prediction_target}")
    if config.objective == "sde" and config.prediction_target != "image":
        raise ValueError("prediction_target=residual is only supported for direct objectives")
    sde = _build_irsde(config, tpgd_options, device) if config.objective == "sde" else None
    use_plain_unet = config.degradation_prior_source == "none" and not use_content_prior and not use_structure_prior
    if use_plain_unet:
        model = PlainTPGDUNet(backbone_cfg, freeze_backbone=False).to(device)
    else:
        model = AssessConditionedTPGDUNet(
            backbone_cfg,
            adapter_hidden_dim=config.adapter_hidden_dim,
            adapter_pool=config.adapter_pool,
            adapter_dropout=config.adapter_dropout,
            use_structured_prior=use_structured_degra_prior,
            structured_hidden_dim=config.adapter_hidden_dim,
            structured_num_tokens=int(tpgd_options.get("structure_prior", {}).get("setting", {}).get("num_latent_tokens", 32)),
            use_confidence_gate=config.degradation_prior_source == "confidence_gate",
            condition_dropout_probability=config.condition_dropout_probability,
            prior_corruption_probability=config.prior_corruption_probability,
            freeze_backbone=False,
        ).to(device)

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        checkpoint_target = str(checkpoint_config.get("prediction_target", "image"))
        if checkpoint_target != config.prediction_target:
            raise ValueError(
                "Checkpoint prediction_target mismatch: "
                f"checkpoint={checkpoint_target}, test_config={config.prediction_target}"
            )

    if isinstance(checkpoint.get("model"), dict):
        incompatible = model.load_state_dict(checkpoint["model"], strict=config.strict_load)
        print(
            f"checkpoint={checkpoint_path} key=model missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    elif isinstance(checkpoint.get("qwen_prompt_prior"), dict) or isinstance(checkpoint.get("structured_prior"), dict):
        qwen_prompt_state = checkpoint.get("qwen_prompt_prior") if isinstance(checkpoint.get("qwen_prompt_prior"), dict) else checkpoint.get("structured_prior")
        qwen_prompt_key = "qwen_prompt_prior" if isinstance(checkpoint.get("qwen_prompt_prior"), dict) else "structured_prior"
        if model.structured_prior is None:
            raise RuntimeError("Checkpoint has qwen_prompt prior but model was not built with qwen_prompt prior")
        if config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
            missing, unexpected = load_tpgd_unet_weights(
                model.backbone,
                config.tpgd_checkpoint,
                strict=config.strict_load,
                map_location="cpu",
            )
            print(f"base_checkpoint={config.tpgd_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        incompatible = model.structured_prior.load_state_dict(qwen_prompt_state, strict=config.strict_load)
        print(
            f"checkpoint={checkpoint_path} key={qwen_prompt_key} missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    elif isinstance(checkpoint.get("assess_prior"), dict):
        if config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
            missing, unexpected = load_tpgd_unet_weights(
                model.backbone,
                config.tpgd_checkpoint,
                strict=config.strict_load,
                map_location="cpu",
            )
            print(f"base_checkpoint={config.tpgd_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        incompatible = model.assess_prior.load_state_dict(checkpoint["assess_prior"], strict=config.strict_load)
        print(
            f"checkpoint={checkpoint_path} key=assess_prior missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    else:
        raise KeyError("Checkpoint must contain either 'model' or 'assess_prior'.")

    content_prior_model = None
    if use_content_prior or use_tpgd_degra_prior:
        if config.prior_checkpoint is None:
            raise ValueError("path.prior must be set when content/degradation prior is enabled")
        content_prior_model = _build_content_prior_model(config.prior_checkpoint, device)

    structure_prior = _build_structure_prior(tpgd_options, config.tpgd_checkpoint, device, config.strict_load) if use_external_structure_prior else None
    if structure_prior is not None and isinstance(checkpoint.get("structure_prior"), dict):
        incompatible = structure_prior.load_state_dict(checkpoint["structure_prior"], strict=config.strict_load)
        print(
            f"structure_prior_payload=loaded missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if structure_prior is not None:
        structure_prior.eval()
        for param in structure_prior.parameters():
            param.requires_grad = False

    dataset = _build_train_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=assessment_collate,
        pin_memory=device.type == "cuda",
    )
    metrics = _evaluate_loader(
        name=metrics_name,
        loader=loader,
        max_batches=max_batches,
        device=device,
        model=model,
        backbone_cfg=backbone_cfg,
        content_prior_model=content_prior_model,
        structure_prior=structure_prior,
        config=config,
        sde=sde,
        use_assessment_degra_prior=use_assessment_degra_prior,
        use_tpgd_degra_prior=use_tpgd_degra_prior,
        use_content_prior=use_content_prior,
        use_structured_degra_prior=use_structured_degra_prior,
        save_images_dir=save_images_dir,
        per_image_records=per_image_records,
    )
    return metrics


def train_assess_tpgd(config: AssessTPGDTrainConfig) -> dict[str, Any]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "train.log"

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    tpgd_options = _load_tpgd_options(config.tpgd_options)
    if config.tpgd_inline_options:
        _deep_update(tpgd_options, config.tpgd_inline_options)
    setting = tpgd_options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()
    if config.degradation_prior_source not in SUPPORTED_PRIOR_SOURCES:
        raise ValueError(f"Unsupported degradation_prior_source: {config.degradation_prior_source}")
    use_assessment_degra_prior = config.degradation_prior_source == "assessment_hidden"
    use_structured_degra_prior = config.degradation_prior_source in STRUCTURED_PRIOR_SOURCES
    use_tpgd_degra_prior = config.degradation_prior_source == "tpgd"
    use_content_prior = bool(config.use_content_prior and backbone_cfg.use_image_context)
    use_structure_prior = bool(config.use_structure_prior and backbone_cfg.use_struct_context)
    use_external_structure_prior = bool(use_structure_prior and not use_structured_degra_prior)
    if use_structure_prior:
        sp_setting = tpgd_options.get("structure_prior", {}).get("setting", {})
        sp_token_dim = int(sp_setting.get("token_dim", backbone_cfg.struct_context_dim))
        if sp_token_dim != backbone_cfg.struct_context_dim:
            raise ValueError(
                "structure prior token_dim must match network_G.setting.struct_context_dim "
                f"when use_struct_prior is enabled, got {sp_token_dim} vs {backbone_cfg.struct_context_dim}"
            )
    if config.objective not in ("sde", "direct_gt", "direct_mse"):
        raise ValueError(f"Unsupported objective: {config.objective}")
    if config.prediction_target not in ("image", "residual"):
        raise ValueError(f"Unsupported prediction_target: {config.prediction_target}")
    if config.objective == "sde" and config.prediction_target != "image":
        raise ValueError("prediction_target=residual is only supported for direct objectives")
    sde = _build_irsde(config, tpgd_options, device) if config.objective == "sde" else None
    # Keep backbone parameters requiring grad because TPGDiff's custom
    # checkpoint function differentiates through its parameter list. When
    # train_backbone=False, we simply exclude backbone params from optimizer.
    use_plain_unet = config.degradation_prior_source == "none" and not use_content_prior and not use_structure_prior
    if use_plain_unet:
        model = PlainTPGDUNet(backbone_cfg, freeze_backbone=False).to(device)
    else:
        model = AssessConditionedTPGDUNet(
            backbone_cfg,
            adapter_hidden_dim=config.adapter_hidden_dim,
            adapter_pool=config.adapter_pool,
            adapter_dropout=config.adapter_dropout,
            use_structured_prior=use_structured_degra_prior,
            structured_hidden_dim=config.adapter_hidden_dim,
            structured_num_tokens=int(tpgd_options.get("structure_prior", {}).get("setting", {}).get("num_latent_tokens", 32)),
            use_confidence_gate=config.degradation_prior_source == "confidence_gate",
            condition_dropout_probability=config.condition_dropout_probability,
            prior_corruption_probability=config.prior_corruption_probability,
            freeze_backbone=False,
        ).to(device)

    loaded_checkpoint: dict[str, Any] | None = None
    if config.load_checkpoint and config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
        checkpoint_payload = torch.load(config.tpgd_checkpoint.expanduser(), map_location="cpu")
        if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model"), dict):
            incompatible = model.load_state_dict(checkpoint_payload["model"], strict=config.strict_load)
            loaded_checkpoint = checkpoint_payload
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(
                "checkpoint_key=model "
                f"load_missing_keys={len(incompatible.missing_keys)} "
                f"load_unexpected_keys={len(incompatible.unexpected_keys)} "
                f"checkpoint_step={checkpoint_payload.get('step')} checkpoint_epoch={checkpoint_payload.get('epoch')}"
            )
        elif isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("assess_prior"), dict):
            incompatible = model.assess_prior.load_state_dict(checkpoint_payload["assess_prior"], strict=config.strict_load)
            loaded_checkpoint = checkpoint_payload
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(
                "checkpoint_key=assess_prior "
                f"load_missing_keys={len(incompatible.missing_keys)} "
                f"load_unexpected_keys={len(incompatible.unexpected_keys)} "
                f"checkpoint_step={checkpoint_payload.get('step')} checkpoint_epoch={checkpoint_payload.get('epoch')}"
            )
        elif isinstance(checkpoint_payload, dict) and (
            isinstance(checkpoint_payload.get("qwen_prompt_prior"), dict)
            or isinstance(checkpoint_payload.get("structured_prior"), dict)
        ):
            qwen_prompt_state = checkpoint_payload.get("qwen_prompt_prior") if isinstance(checkpoint_payload.get("qwen_prompt_prior"), dict) else checkpoint_payload.get("structured_prior")
            qwen_prompt_key = "qwen_prompt_prior" if isinstance(checkpoint_payload.get("qwen_prompt_prior"), dict) else "structured_prior"
            if model.structured_prior is None:
                raise RuntimeError("Checkpoint has qwen_prompt prior but model was not built with qwen_prompt prior")
            incompatible = model.structured_prior.load_state_dict(qwen_prompt_state, strict=config.strict_load)
            loaded_checkpoint = checkpoint_payload
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(
                f"checkpoint_key={qwen_prompt_key} "
                f"load_missing_keys={len(incompatible.missing_keys)} "
                f"load_unexpected_keys={len(incompatible.unexpected_keys)} "
                f"checkpoint_step={checkpoint_payload.get('step')} checkpoint_epoch={checkpoint_payload.get('epoch')}"
            )
        else:
            missing, unexpected = load_tpgd_unet_weights(
                model.backbone,
                config.tpgd_checkpoint,
                strict=config.strict_load,
                map_location="cpu",
            )
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(f"checkpoint_key=tpgd_G load_missing_keys={len(missing)} load_unexpected_keys={len(unexpected)}")
    else:
        log("checkpoint=skipped")

    content_prior_model = None
    if use_content_prior or use_tpgd_degra_prior:
        if config.prior_checkpoint is None:
            raise ValueError("path.prior must be set when content/degradation prior is enabled")
        content_prior_model = _build_content_prior_model(config.prior_checkpoint, device)

    structure_prior_checkpoint = config.tpgd_checkpoint if config.load_checkpoint else None
    structure_prior = _build_structure_prior(tpgd_options, structure_prior_checkpoint, device, config.strict_load) if use_external_structure_prior else None
    if structure_prior is not None and isinstance(loaded_checkpoint, dict) and isinstance(loaded_checkpoint.get("structure_prior"), dict):
        incompatible = structure_prior.load_state_dict(loaded_checkpoint["structure_prior"], strict=config.strict_load)
        log(
            "checkpoint_key=structure_prior "
            f"load_missing_keys={len(incompatible.missing_keys)} "
            f"load_unexpected_keys={len(incompatible.unexpected_keys)}"
        )
    if structure_prior is not None:
        structure_prior.train(config.train_structure_prior)
        for param in structure_prior.parameters():
            param.requires_grad = bool(config.train_structure_prior)

    dataset = _build_train_dataset(config)
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=assessment_collate,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    if not config.train_backbone:
        if use_assessment_degra_prior:
            params = list(model.assess_prior.parameters())
        elif use_structured_degra_prior:
            params = list(model.structured_prior.parameters()) if model.structured_prior is not None else []
        else:
            params = []
    else:
        params = [param for param in model.parameters() if param.requires_grad]
    if structure_prior is not None and config.train_structure_prior:
        params.extend(param for param in structure_prior.parameters() if param.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters. Check train_backbone and adapter settings.")
    optimizer = _build_optimizer(params, config)
    planned_steps = _planned_train_steps(config, len(loader))
    scheduler = _build_lr_scheduler(optimizer, config, planned_steps)

    global_step = int(loaded_checkpoint.get("step") or 0) if isinstance(loaded_checkpoint, dict) else 0
    resume_epoch = int(loaded_checkpoint.get("epoch") or 0) if isinstance(loaded_checkpoint, dict) else 0
    start_epoch = resume_epoch + 1 if resume_epoch > 0 else 1
    target_max_step = global_step + int(config.max_steps) if global_step > 0 and int(config.max_steps) > 0 else int(config.max_steps)
    configured_epochs = max(0, int(config.epochs))
    if configured_epochs > 0:
        end_epoch = resume_epoch + configured_epochs if resume_epoch > 0 else configured_epochs
    elif target_max_step > global_step:
        remaining_steps = target_max_step - global_step
        required_epochs = (remaining_steps + len(loader) - 1) // len(loader)
        end_epoch = start_epoch + required_epochs - 1
    else:
        end_epoch = start_epoch - 1

    if isinstance(loaded_checkpoint, dict):
        if isinstance(loaded_checkpoint.get("optimizer"), dict):
            try:
                optimizer.load_state_dict(loaded_checkpoint["optimizer"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)
                log("checkpoint_key=optimizer restored=true")
            except ValueError as exc:
                log(f"checkpoint_key=optimizer restored=false error={exc}")
        else:
            log("checkpoint_key=optimizer restored=false reason=missing")
        if scheduler is not None and isinstance(loaded_checkpoint.get("scheduler"), dict):
            try:
                scheduler.load_state_dict(loaded_checkpoint["scheduler"])
                log("checkpoint_key=scheduler restored=true")
            except ValueError as exc:
                log(f"checkpoint_key=scheduler restored=false error={exc}")
        elif scheduler is not None:
            log("checkpoint_key=scheduler restored=false reason=missing")

    best_val_psnr = -math.inf
    best_val_step = 0
    if isinstance(loaded_checkpoint, dict):
        loaded_best_name = loaded_checkpoint.get("best_metric_name")
        loaded_best_value = loaded_checkpoint.get("best_metric_value")
        if loaded_best_name == BEST_VAL_PSNR_KEY and loaded_best_value is not None:
            best_val_psnr = float(loaded_best_value)
            best_val_step = int(loaded_checkpoint.get("best_metric_step") or global_step)

    wandb_logger = _WandbLogger(config)

    def update_best_checkpoint(eval_payload: dict[str, float], *, step: int, epoch: int) -> None:
        nonlocal best_val_psnr, best_val_step
        candidate = eval_payload.get(BEST_VAL_PSNR_KEY)
        if candidate is not None and math.isfinite(float(candidate)) and float(candidate) > best_val_psnr:
            previous = best_val_psnr
            best_val_psnr = float(candidate)
            best_val_step = int(step)
            best_path = save_checkpoint(
                config.output_dir,
                model,
                optimizer,
                step=step,
                epoch=epoch,
                config=config,
                structure_prior=structure_prior,
                scheduler=scheduler,
                best_metric_name=BEST_VAL_PSNR_KEY,
                best_metric_value=best_val_psnr,
                best_metric_step=best_val_step,
                best_only=True,
            )
            previous_text = f"{previous:.6f}" if math.isfinite(previous) else "none"
            log(
                f"best_checkpoint_updated metric={BEST_VAL_PSNR_KEY} value={best_val_psnr:.6f} "
                f"previous={previous_text} step={step} epoch={epoch} saved={best_path}"
            )
        if math.isfinite(best_val_psnr):
            eval_payload[BEST_VAL_PSNR_WANDB_KEY] = best_val_psnr

    eval_batch_size = config.eval_batch_size or config.batch_size
    eval_train_loader = None
    eval_val_loader = None
    if config.eval_enabled:
        eval_train_dataset: Dataset = dataset
        train_sampling_mode = str(config.eval_train_sampling_mode or "none").strip().lower().replace("-", "_")
        train_sampling_counts: list[int] | None = None
        if train_sampling_mode in {"stratified_fixed", "fixed_stratified"}:
            eval_train_dataset, train_sampling_counts = _build_fixed_stratified_subset(
                dataset,
                fraction_per_task=config.eval_train_fraction_per_task,
                seed=config.eval_train_sampling_seed,
            )
        elif train_sampling_mode not in {"", "none", "off", "full"}:
            raise ValueError(f"Unsupported evaluation.train_sampling.mode: {config.eval_train_sampling_mode}")
        eval_train_loader = DataLoader(
            eval_train_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=assessment_collate,
            pin_memory=device.type == "cuda",
        )
        val_dataset = _build_val_dataset(config)
        if val_dataset is not None:
            eval_val_loader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=assessment_collate,
                pin_memory=device.type == "cuda",
            )

    log(f"samples={len(dataset)} output_dir={config.output_dir}")
    log(f"device={device} train_backbone={config.train_backbone} batch_size={config.batch_size} seed={config.seed}")
    log(f"optimizer={config.optimizer} lr={config.lr} betas=({config.beta1},{config.beta2}) weight_decay={config.weight_decay}")
    log(
        f"lr_scheduler={config.lr_scheduler} lr_min={config.lr_min} step_size={config.lr_step_size} "
        f"gamma={config.lr_gamma} milestones={config.lr_milestones} t_max={config.lr_t_max} "
        f"resolved_t_max={_resolve_lr_t_max(config, planned_steps)} planned_steps={planned_steps} "
        f"warmup_enabled={config.lr_warmup_enabled} warmup_steps={config.lr_warmup_steps} "
        f"warmup_start_factor={config.lr_warmup_start_factor}"
    )
    if resume_epoch > 0 or global_step > 0:
        log(f"resume=enabled start_step={global_step} start_epoch={start_epoch} end_epoch={end_epoch} target_max_step={target_max_step}")
    else:
        log(f"resume=disabled start_step={global_step} start_epoch={start_epoch} end_epoch={end_epoch} target_max_step={target_max_step}")
    log(f"objective={config.objective} prediction_target={config.prediction_target} loss_type={config.loss_type} loss_weight={config.loss_weight}")
    if config.objective != "sde":
        log(f"direct_gt_time={config.direct_gt_time}")
    log(f"boundary_pad={config.boundary_pad}")
    log(f"degradation_prior_source={config.degradation_prior_source}")
    if config.degradation_prior_source in QWEN_V2_PRIOR_SOURCES:
        log(f"structured_prior_temperature={config.structured_prior_temperature:.8f}")
    if config.degradation_prior_source == "confidence_gate":
        log(
            f"confidence_gate=enabled condition_dropout_probability={config.condition_dropout_probability:.4f} "
            f"prior_corruption_probability={config.prior_corruption_probability:.4f} "
            f"confidence_override={config.structured_confidence_override} unknown_init=zeros"
        )
    log(f"model_variant={'plain_unet' if use_plain_unet else 'prior_conditioned_unet'}")
    log(f"content_prior={'enabled' if use_content_prior else 'disabled'} external_structure_prior={'enabled' if structure_prior is not None else 'disabled'} qwen_prompt_layout_prior={'enabled' if use_structured_degra_prior and backbone_cfg.use_struct_context else 'disabled'} train_structure_prior={config.train_structure_prior}")
    log(f"wandb={'enabled' if wandb_logger.run is not None else 'disabled'}")
    if config.eval_enabled:
        log(
            f"evaluation=enabled every_steps={config.eval_every_steps} every_epochs={config.eval_every_epochs} "
            f"batch_size={eval_batch_size} train_max_batches={config.eval_train_max_batches} "
            f"val={'enabled' if eval_val_loader is not None else 'disabled'} val_max_batches={config.eval_val_max_batches}"
        )
        if train_sampling_counts is not None:
            log(
                f"eval_train_sampling=mode:{train_sampling_mode} "
                f"fraction_per_task:{config.eval_train_fraction_per_task} "
                f"seed:{config.eval_train_sampling_seed} counts:{train_sampling_counts} "
                f"samples:{sum(train_sampling_counts)}"
            )
    else:
        log("evaluation=disabled")
    if sde is not None:
        log(
            f"sde=max_sigma:{sde.max_sigma:.6f} T:{sde.T} schedule:{_cfg_value(config.sde_schedule, tpgd_options.get('sde', {}), 'schedule', 'cosine')} "
            f"eps:{_cfg_value(config.sde_eps, tpgd_options.get('sde', {}), 'eps', 0.005)} t_range:{config.sde_t_start}-{config.sde_t_end}"
        )

    last_loss = None
    model.train()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, end_epoch + 1):
        for batch in loader:
            global_step += 1
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            hidden = batch["hidden"]
            mask = batch["mask"]
            if hidden is not None:
                hidden = hidden.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            structured_prior = batch["structured_prior"].to(device, non_blocking=True) if use_structured_degra_prior else None

            content_context = None
            deg_context_input = None
            lq_clip = batch["lq_clip"].to(device, non_blocking=True) if (use_content_prior or use_tpgd_degra_prior) else None
            if content_prior_model is not None and lq_clip is not None:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    if use_content_prior:
                        content_context = content_prior_model.get_content_prior(lq_clip).float()
                    if use_tpgd_degra_prior:
                        deg_context_input = content_prior_model.encode_for_degradation(lq_clip).float()
            elif config.random_content_context:
                content_context = torch.randn(
                    lq.shape[0],
                    backbone_cfg.context_dim,
                    device=device,
                    requires_grad=not config.train_backbone,
                )
            elif backbone_cfg.use_image_context:
                content_context = torch.zeros(
                    lq.shape[0],
                    backbone_cfg.context_dim,
                    device=device,
                    requires_grad=not config.train_backbone,
                )

            lq_model, boundary_crop = _pad_for_boundary(lq, config.boundary_pad)
            gt_model = _apply_boundary_pad(gt, boundary_crop)

            struct_tokens = None
            if structure_prior is not None:
                struct_input = (lq_model + 1.0) * 0.5
                if config.train_structure_prior:
                    struct_tokens = structure_prior(struct_input)
                else:
                    with torch.no_grad():
                        struct_tokens = structure_prior(struct_input)

            optimizer.zero_grad(set_to_none=True)
            if config.objective == "sde":
                assert sde is not None
                timesteps, states = sde.generate_random_states(
                    x0=gt_model,
                    mu=lq_model,
                    T_start=config.sde_t_start,
                    T_end=config.sde_t_end,
                )
                # TPGDiff uses a custom checkpoint function that expects selected
                # forward inputs to require grad. These tensors are not optimized;
                # they only keep the adapter-only backward path valid.
                states_for_model = states.detach().requires_grad_(True) if not config.train_backbone else states
                lq_for_model = lq_model.detach().requires_grad_(True) if not config.train_backbone else lq_model
                output_model, deg_context = model(
                    states_for_model,
                    lq_for_model,
                    timesteps.reshape(-1),
                    assessment_hidden=hidden if use_assessment_degra_prior else None,
                    assessment_mask=mask if use_assessment_degra_prior else None,
                    structured_prior=structured_prior,
                    structured_confidence_override=config.structured_confidence_override,
                    deg_context=deg_context_input,
                    content_context=content_context,
                    struct_tokens=struct_tokens,
                    return_context=True,
                )
                score = sde.get_score_from_noise(output_model, timesteps)
                xt_1_expectation_model = sde.reverse_sde_step_mean(states_for_model, score, timesteps)
                xt_1_optimum_model = sde.reverse_optimum_step(states, gt_model, timesteps)
                xt_1_expectation = _crop_boundary(xt_1_expectation_model, boundary_crop)
                xt_1_optimum = _crop_boundary(xt_1_optimum_model, boundary_crop)
                output = _crop_boundary(output_model, boundary_crop)
                loss = config.loss_weight * _matching_loss(xt_1_expectation, xt_1_optimum, config.loss_type)
            else:
                lq_for_model = lq_model.detach().requires_grad_(True) if not config.train_backbone else lq_model
                time = _direct_time(lq_for_model.shape[0], device, config.direct_gt_time)
                output_model, deg_context = model(
                    lq_for_model,
                    lq_for_model,
                    time,
                    assessment_hidden=hidden if use_assessment_degra_prior else None,
                    assessment_mask=mask if use_assessment_degra_prior else None,
                    structured_prior=structured_prior,
                    structured_confidence_override=config.structured_confidence_override,
                    deg_context=deg_context_input,
                    content_context=content_context,
                    struct_tokens=struct_tokens,
                    return_context=True,
                )
                restored_model = _direct_restored(
                    output_model,
                    lq_model,
                    config.prediction_target,
                )
                output = _crop_boundary(restored_model, boundary_crop)
                loss = config.loss_weight * _matching_loss(output, gt, config.loss_type)
            loss.backward()
            if not config.train_backbone:
                for param in model.backbone.parameters():
                    param.grad = None
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr_next = float(optimizer.param_groups[0]["lr"])
            last_loss = float(loss.detach().cpu())

            if global_step == 1 or global_step % config.log_every == 0:
                train_payload = {
                    "train/loss": last_loss,
                    "train/epoch": epoch,
                    "train/lr": lr_used,
                    "train/lr_next": lr_next,
                }
                gate_text = ""
                if config.degradation_prior_source == "confidence_gate" and model.structured_prior is not None:
                    gate_info = model.structured_prior.last_gate_info
                    gate_metrics = {
                        "train/gate_raw_confidence": float(gate_info["raw_confidence"].mean()),
                        "train/gate_effective_confidence": float(gate_info["gate_confidence"].mean()),
                        "train/condition_dropout_rate": float(gate_info["condition_dropout_mask"].float().mean()),
                        "train/prior_corruption_rate": float(gate_info["prior_corruption_mask"].float().mean()),
                        "train/qwen_context_norm": float(gate_info["qwen_context_norm"].mean()),
                        "train/deg_context_norm": float(gate_info["deg_context_norm"].mean()),
                    }
                    train_payload.update(gate_metrics)
                    gate_text = (
                        f" gate_raw={gate_metrics['train/gate_raw_confidence']:.4f}"
                        f" gate_effective={gate_metrics['train/gate_effective_confidence']:.4f}"
                        f" condition_dropout={gate_metrics['train/condition_dropout_rate']:.4f}"
                        f" prior_corruption={gate_metrics['train/prior_corruption_rate']:.4f}"
                    )
                log(
                    f"step={global_step} epoch={epoch} loss={last_loss:.6f} lr={lr_used:.8g} lr_next={lr_next:.8g} "
                    f"output_shape={tuple(output.shape)} deg_context_shape={tuple(deg_context.shape) if deg_context is not None else None}"
                    f"{gate_text}"
                )
                wandb_logger.log(train_payload, step=global_step)
            if config.eval_enabled and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                eval_payload: dict[str, float] = {}
                if eval_train_loader is not None:
                    eval_payload.update(_evaluate_loader(
                        name="eval_train",
                        loader=eval_train_loader,
                        max_batches=config.eval_train_max_batches,
                        device=device,
                        model=model,
                        backbone_cfg=backbone_cfg,
                        content_prior_model=content_prior_model,
                        structure_prior=structure_prior,
                        config=config,
                        sde=sde,
                        use_assessment_degra_prior=use_assessment_degra_prior,
                        use_tpgd_degra_prior=use_tpgd_degra_prior,
                        use_content_prior=use_content_prior,
                        use_structured_degra_prior=use_structured_degra_prior,
                    ))
                if eval_val_loader is not None:
                    eval_payload.update(_evaluate_loader(
                        name="eval_val",
                        loader=eval_val_loader,
                        max_batches=config.eval_val_max_batches,
                        device=device,
                        model=model,
                        backbone_cfg=backbone_cfg,
                        content_prior_model=content_prior_model,
                        structure_prior=structure_prior,
                        config=config,
                        sde=sde,
                        use_assessment_degra_prior=use_assessment_degra_prior,
                        use_tpgd_degra_prior=use_tpgd_degra_prior,
                        use_content_prior=use_content_prior,
                        use_structured_degra_prior=use_structured_degra_prior,
                    ))
                if eval_payload:
                    update_best_checkpoint(eval_payload, step=global_step, epoch=epoch)
                    metrics = " ".join(f"{key}={value:.6f}" for key, value in eval_payload.items())
                    log(f"eval step={global_step} epoch={epoch} {metrics}")
                    wandb_logger.log(eval_payload, step=global_step)
            if config.save_every > 0 and global_step % config.save_every == 0:
                save_path = save_checkpoint(
                    config.output_dir,
                    model,
                    optimizer,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                    structure_prior=structure_prior,
                    scheduler=scheduler,
                    best_metric_name=BEST_VAL_PSNR_KEY if math.isfinite(best_val_psnr) else None,
                    best_metric_value=best_val_psnr if math.isfinite(best_val_psnr) else None,
                    best_metric_step=best_val_step,
                )
                log(f"saved={save_path}")
            if target_max_step > 0 and global_step >= target_max_step:
                break
        if config.eval_enabled and config.eval_every_epochs > 0 and epoch % config.eval_every_epochs == 0:
            eval_payload: dict[str, float] = {}
            if eval_train_loader is not None:
                eval_payload.update(_evaluate_loader(
                    name="eval_train",
                    loader=eval_train_loader,
                    max_batches=config.eval_train_max_batches,
                    device=device,
                    model=model,
                    backbone_cfg=backbone_cfg,
                    content_prior_model=content_prior_model,
                    structure_prior=structure_prior,
                    config=config,
                    sde=sde,
                    use_assessment_degra_prior=use_assessment_degra_prior,
                    use_tpgd_degra_prior=use_tpgd_degra_prior,
                    use_content_prior=use_content_prior,
                    use_structured_degra_prior=use_structured_degra_prior,
                ))
            if eval_val_loader is not None:
                eval_payload.update(_evaluate_loader(
                    name="eval_val",
                    loader=eval_val_loader,
                    max_batches=config.eval_val_max_batches,
                    device=device,
                    model=model,
                    backbone_cfg=backbone_cfg,
                    content_prior_model=content_prior_model,
                    structure_prior=structure_prior,
                    config=config,
                    sde=sde,
                    use_assessment_degra_prior=use_assessment_degra_prior,
                    use_tpgd_degra_prior=use_tpgd_degra_prior,
                    use_content_prior=use_content_prior,
                    use_structured_degra_prior=use_structured_degra_prior,
                ))
            if eval_payload:
                update_best_checkpoint(eval_payload, step=global_step, epoch=epoch)
                metrics = " ".join(f"{key}={value:.6f}" for key, value in eval_payload.items())
                log(f"eval step={global_step} epoch={epoch} {metrics}")
                wandb_logger.log(eval_payload, step=global_step)
        if target_max_step > 0 and global_step >= target_max_step:
            break

    save_path = save_checkpoint(
        config.output_dir,
        model,
        optimizer,
        step=global_step,
        epoch=epoch,
        config=config,
        structure_prior=structure_prior,
        scheduler=scheduler,
        best_metric_name=BEST_VAL_PSNR_KEY if math.isfinite(best_val_psnr) else None,
        best_metric_value=best_val_psnr if math.isfinite(best_val_psnr) else None,
        best_metric_step=best_val_step,
    )
    best_path = config.output_dir / "best.pt"
    best_summary = f"{best_val_psnr:.6f}@{best_val_step}" if math.isfinite(best_val_psnr) else "none"
    log(f"train_ok steps={global_step} last_loss={last_loss:.6f} saved={save_path} best_val_psnr_macro={best_summary}")
    wandb_logger.finish()
    return {
        "steps": global_step,
        "last_loss": last_loss,
        "checkpoint": str(save_path),
        "best_checkpoint": str(best_path) if best_path.exists() else "",
        "best_val_psnr_macro": best_val_psnr if math.isfinite(best_val_psnr) else None,
        "best_step": best_val_step if math.isfinite(best_val_psnr) else None,
    }
