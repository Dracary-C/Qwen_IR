"""TPGDiff restoration backbone with Assessment Reasoning degradation prior.

This module keeps the TPGDiff ConditionalUNet architecture, but replaces the
source of `deg_context`:

    original TPGDiff:
        LQ image -> CLIP/PriorStageModel.encode_for_degradation -> deg_context [B, D]

    Qwen-IR version:
        Assessment Reasoning hidden states [B, T, 4096]
        -> AssessPriorAdapter
        -> deg_context [B, D]
The underlying ConditionalUNet is loaded from the TPGDiff runtime bundled in
this repository. The wrapper is the Qwen-IR-owned module boundary; third-party
source and attribution are kept under `module/vendor/tpgdiff/`.
and attribution are kept under module/vendor/tpgdiff.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from module.legacy.source_paths import source_repo
from module.degradation_prompt import AssessPriorAdapter
from module.layout_prompt import QwenPromptPriorAdapter


@dataclass
class TPGDBackboneConfig:
    """Config needed to build the TPGDiff ConditionalUNet backbone."""

    in_nc: int = 3
    out_nc: int = 3
    nf: int = 32
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4])
    context_dim: int = 512
    use_degra_context: bool = True
    use_image_context: bool = True
    upscale: int = 1
    use_struct_context: bool = False
    struct_context_dim: int = 256

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TPGDBackboneConfig":
        allowed = cls.__dataclass_fields__.keys()
        values = {key: value for key, value in mapping.items() if key in allowed}
        return cls(**values)

    def to_tpgd_kwargs(self) -> dict[str, Any]:
        return {
            "in_nc": self.in_nc,
            "out_nc": self.out_nc,
            "nf": self.nf,
            "ch_mult": self.ch_mult,
            "context_dim": self.context_dim,
            "use_degra_context": self.use_degra_context,
            "use_image_context": self.use_image_context,
            "upscale": self.upscale,
            "use_struct_context": self.use_struct_context,
            "struct_context_dim": self.struct_context_dim,
        }


@contextlib.contextmanager
def _temporary_sys_path(paths: Iterable[Path]):
    inserted: list[str] = []
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted.append(path_str)
    try:
        yield
    finally:
        for path_str in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


def _tpgd_code_root(tpgd_root: Path | None = None) -> Path:
    root = Path(tpgd_root).expanduser().resolve() if tpgd_root else source_repo("tpgdiff").path
    return root / "universal-restoration" / "config" / "tpgd-sde"


def load_tpgd_conditional_unet_cls(tpgd_root: Path | None = None):
    """Load TPGDiff's ConditionalUNet class from the vendored runtime tree."""

    code_root = _tpgd_code_root(tpgd_root)
    if not code_root.exists():
        raise FileNotFoundError(f"TPGDiff code root not found: {code_root}")
    with _temporary_sys_path([code_root]):
        from models.modules import ConditionalUNet

    return ConditionalUNet


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["G", "params", "state_dict", "model", "netG", "network_G"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ["module.", "model.", "denoise_fn."]
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_tpgd_unet_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load a TPGDiff restoration checkpoint into the wrapped ConditionalUNet.

    Returns:
        (missing_keys, unexpected_keys)
    """

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=map_location)
    state_dict = _strip_known_prefixes(_unwrap_state_dict(checkpoint))
    skipped: list[str] = []
    if not strict:
        model_state = model.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is not None and tuple(target.shape) != tuple(value.shape):
                skipped.append(key)
                continue
            compatible[key] = value
        state_dict = compatible
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys) + skipped, list(incompatible.unexpected_keys)


class PlainTPGDUNet(nn.Module):
    """TPGDiff ConditionalUNet without any prior or adapter modules."""

    def __init__(
        self,
        config: TPGDBackboneConfig | dict[str, Any] | None = None,
        *,
        tpgd_root: str | Path | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if config is None:
            self.config = TPGDBackboneConfig()
        elif isinstance(config, TPGDBackboneConfig):
            self.config = config
        else:
            self.config = TPGDBackboneConfig.from_mapping(config)

        conditional_unet_cls = load_tpgd_conditional_unet_cls(Path(tpgd_root) if tpgd_root else None)
        self.backbone = conditional_unet_cls(**self.config.to_tpgd_kwargs())
        self.assess_prior = None
        self.structured_prior = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor | int | float,
        *,
        assessment_hidden: torch.Tensor | None = None,
        assessment_mask: torch.Tensor | None = None,
        structured_prior: torch.Tensor | None = None,
        structured_confidence_override: torch.Tensor | float | None = None,
        deg_context: torch.Tensor | None = None,
        content_context: torch.Tensor | None = None,
        struct_tokens: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        output = self.backbone(
            xt,
            cond,
            time,
            deg_context=None,
            content_context=None,
            struct_tokens=None,
        )
        if return_context:
            return output, None
        return output


class AssessConditionedTPGDUNet(nn.Module):
    """TPGDiff ConditionalUNet with Assessment Reasoning as degradation prior.

    Forward inputs:
        xt: noisy restoration state, [B, 3, H, W]
        cond: low-quality condition image, [B, 3, H, W]
        time: diffusion timestep tensor/scalar
        assessment_hidden: [B, T, 4096], usually `condition_hidden`
        content_context: optional original TPGDiff content context, [B, context_dim]
        struct_tokens: optional structure prior tokens

    Output:
        predicted noise / score tensor from TPGDiff ConditionalUNet.
    """

    def __init__(
        self,
        config: TPGDBackboneConfig | dict[str, Any] | None = None,
        *,
        tpgd_root: str | Path | None = None,
        assessment_hidden_dim: int = 4096,
        adapter_hidden_dim: int = 1024,
        adapter_pool: str = "mean",
        adapter_dropout: float = 0.0,
        use_structured_prior: bool = False,
        structured_hidden_dim: int = 256,
        structured_num_tokens: int = 32,
        use_confidence_gate: bool = False,
        condition_dropout_probability: float = 0.0,
        prior_corruption_probability: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if config is None:
            self.config = TPGDBackboneConfig()
        elif isinstance(config, TPGDBackboneConfig):
            self.config = config
        else:
            self.config = TPGDBackboneConfig.from_mapping(config)

        conditional_unet_cls = load_tpgd_conditional_unet_cls(Path(tpgd_root) if tpgd_root else None)
        self.backbone = conditional_unet_cls(**self.config.to_tpgd_kwargs())
        self.assess_prior = AssessPriorAdapter(
            input_dim=assessment_hidden_dim,
            output_dim=self.config.context_dim,
            hidden_dim=adapter_hidden_dim,
            pool=adapter_pool,
            dropout=adapter_dropout,
        )
        self.structured_prior = (
            QwenPromptPriorAdapter(
                context_dim=self.config.context_dim,
                struct_context_dim=self.config.struct_context_dim,
                num_struct_tokens=structured_num_tokens,
                hidden_dim=structured_hidden_dim,
                use_layout_tokens=self.config.use_struct_context,
                use_confidence_gate=use_confidence_gate,
                condition_dropout_probability=condition_dropout_probability,
                prior_corruption_probability=prior_corruption_probability,
            )
            if use_structured_prior
            else None
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def build_deg_context(
        self,
        assessment_hidden: torch.Tensor,
        assessment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert Assessment Reasoning hidden states to TPGDiff deg_context."""

        return self.assess_prior(assessment_hidden, mask=assessment_mask)

    def build_structured_contexts(
        self,
        structured_prior: torch.Tensor,
        confidence_override: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert a Qwen structured prior to TPGDiff UNet conditions."""

        if self.structured_prior is None:
            raise RuntimeError("structured_prior was provided but use_structured_prior=False")
        contexts = self.structured_prior(
            structured_prior,
            confidence_override=confidence_override,
        )
        return contexts.deg_context, contexts.struct_tokens

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor | int | float,
        *,
        assessment_hidden: torch.Tensor | None = None,
        assessment_mask: torch.Tensor | None = None,
        structured_prior: torch.Tensor | None = None,
        structured_confidence_override: torch.Tensor | float | None = None,
        deg_context: torch.Tensor | None = None,
        content_context: torch.Tensor | None = None,
        struct_tokens: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if deg_context is None and assessment_hidden is not None:
            deg_context = self.build_deg_context(assessment_hidden, assessment_mask)
        if structured_prior is not None:
            structured_deg_context, structured_tokens = self.build_structured_contexts(
                structured_prior,
                confidence_override=structured_confidence_override,
            )
            if deg_context is None:
                deg_context = structured_deg_context
            if struct_tokens is None:
                struct_tokens = structured_tokens

        if self.config.use_image_context and content_context is None:
            # TPGDiff's SpatialTransformer is built with context_dim when
            # use_image_context=True, so it cannot receive context=None. For the
            # first degradation-prior ablation we allow content prior to be
            # omitted by passing a zero context token. Formal experiments can
            # still provide the original TPGDiff content_context here.
            batch = xt.shape[0]
            content_context = xt.new_zeros(batch, self.config.context_dim)

        output = self.backbone(
            xt,
            cond,
            time,
            deg_context=deg_context,
            content_context=content_context,
            struct_tokens=struct_tokens,
        )
        if return_context:
            return output, deg_context
        return output
