"""Adapters from Qwen-prompt priors to TPGDiff UNet conditions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from module.degradation_prompt.qwen_degradation import DegradationTimeContextEncoder
from module.layout_prompt.schema import split_structured_prior


@dataclass
class QwenPromptPriorContexts:
    deg_context: torch.Tensor
    struct_tokens: torch.Tensor | None
    qwen_context: torch.Tensor
    raw_confidence: torch.Tensor | None = None
    gate_confidence: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None
    prior_corruption_mask: torch.Tensor | None = None


class LayoutTokenEncoder(nn.Module):
    """Encode layout_10 into structure tokens consumed by StructFiLMAdapter."""

    def __init__(
        self,
        token_dim: int,
        *,
        num_tokens: int = 32,
        hidden_dim: int = 256,
        layout_dim: int = 10,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.base_tokens = nn.Parameter(torch.randn(1, self.num_tokens, self.token_dim) * 0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(layout_dim),
            nn.Linear(layout_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_tokens * self.token_dim),
        )

    def forward(self, layout: torch.Tensor) -> torch.Tensor:
        batch = layout.shape[0]
        offsets = self.net(layout).view(batch, self.num_tokens, self.token_dim)
        return self.base_tokens.to(dtype=offsets.dtype, device=offsets.device) + offsets


class QwenPromptPriorAdapter(nn.Module):
    """Convert a 21-d Qwen-prompt prior into UNet degradation and structure conditions.

    Current routing:
        severity_5 + degradation_probs_5 -> deg_context -> time embedding path
        layout_10 -> struct_tokens -> FiLM adapters in TPGDiff ConditionalUNet

    In confidence-gate mode, max(calibrated_probs) blends the encoded Qwen
    context with a learnable generic context. Other modes retain the original
    direct Qwen-context behavior.
    """

    def __init__(
        self,
        *,
        context_dim: int = 512,
        struct_context_dim: int = 256,
        num_struct_tokens: int = 32,
        hidden_dim: int = 256,
        use_layout_tokens: bool = True,
        use_confidence_gate: bool = False,
        condition_dropout_probability: float = 0.0,
        prior_corruption_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_layout_tokens = bool(use_layout_tokens)
        self.use_confidence_gate = bool(use_confidence_gate)
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.prior_corruption_probability = float(prior_corruption_probability)
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if not 0.0 <= self.prior_corruption_probability <= 1.0:
            raise ValueError("prior_corruption_probability must be in [0, 1]")
        if self.condition_dropout_probability + self.prior_corruption_probability > 1.0:
            raise ValueError("condition dropout + prior corruption probabilities must not exceed 1")
        self.degradation_encoder = DegradationTimeContextEncoder(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            input_dim=10,
        )
        self.layout_encoder = (
            LayoutTokenEncoder(
                token_dim=struct_context_dim,
                num_tokens=num_struct_tokens,
                hidden_dim=hidden_dim,
                layout_dim=10,
            )
            if self.use_layout_tokens
            else None
        )
        self.unknown_context = (
            nn.Parameter(torch.zeros(1, context_dim))
            if self.use_confidence_gate
            else None
        )
        self.last_gate_info: dict[str, torch.Tensor] = {}

    @staticmethod
    def _confidence_override(
        value: torch.Tensor | float,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        confidence = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if confidence.ndim == 0:
            confidence = confidence.expand(batch_size).reshape(batch_size, 1)
        elif confidence.ndim == 1 and confidence.shape[0] == batch_size:
            confidence = confidence.reshape(batch_size, 1)
        elif confidence.shape != (batch_size, 1):
            raise ValueError(
                f"confidence_override must be scalar, [B], or [B,1], got {tuple(confidence.shape)}"
            )
        if (confidence < 0).any() or (confidence > 1).any():
            raise ValueError("confidence_override values must be in [0, 1]")
        return confidence

    def forward(
        self,
        prior: torch.Tensor,
        *,
        confidence_override: torch.Tensor | float | None = None,
    ) -> QwenPromptPriorContexts:
        original_parts = split_structured_prior(prior)
        effective_prior = prior
        batch_size = prior.shape[0]
        dropout_mask = torch.zeros(batch_size, dtype=torch.bool, device=prior.device)
        corruption_mask = torch.zeros(batch_size, dtype=torch.bool, device=prior.device)

        if self.use_confidence_gate and self.training:
            draw = torch.rand(batch_size, device=prior.device)
            dropout_mask = draw < self.condition_dropout_probability
            corruption_mask = (
                (draw >= self.condition_dropout_probability)
                & (draw < self.condition_dropout_probability + self.prior_corruption_probability)
            )
            if bool(corruption_mask.any()) and batch_size > 1:
                # A non-zero cyclic shift guarantees that corrupted samples use
                # another batch member's complete structured prior.
                shift = int(torch.randint(1, batch_size, (), device=prior.device))
                shuffled = torch.roll(prior, shifts=shift, dims=0)
                effective_prior = torch.where(corruption_mask[:, None], shuffled, prior)

        parts = split_structured_prior(effective_prior)
        qwen_context = self.degradation_encoder(parts["severity"], parts["degradation_probs"])
        raw_confidence = original_parts["degradation_probs"].max(dim=-1, keepdim=True).values

        if self.use_confidence_gate:
            if self.unknown_context is None:
                raise RuntimeError("confidence gate enabled without unknown_context")
            gate_confidence = (
                raw_confidence
                if confidence_override is None
                else self._confidence_override(
                    confidence_override,
                    batch_size=batch_size,
                    reference=raw_confidence,
                )
            )
            if self.training:
                fallback_mask = dropout_mask | corruption_mask
                gate_confidence = torch.where(
                    fallback_mask[:, None],
                    torch.zeros_like(gate_confidence),
                    gate_confidence,
                )
            unknown = self.unknown_context.to(
                dtype=qwen_context.dtype,
                device=qwen_context.device,
            ).expand(batch_size, -1)
            deg_context = gate_confidence * qwen_context + (1.0 - gate_confidence) * unknown
        else:
            gate_confidence = None
            deg_context = qwen_context

        struct_tokens = self.layout_encoder(parts["layout"]) if self.layout_encoder is not None else None
        self.last_gate_info = {
            "raw_confidence": raw_confidence.detach(),
            "gate_confidence": (
                gate_confidence.detach()
                if gate_confidence is not None
                else torch.ones_like(raw_confidence)
            ),
            "condition_dropout_mask": dropout_mask.detach(),
            "prior_corruption_mask": corruption_mask.detach(),
            "qwen_context_norm": qwen_context.detach().norm(dim=-1, keepdim=True),
            "deg_context_norm": deg_context.detach().norm(dim=-1, keepdim=True),
        }
        return QwenPromptPriorContexts(
            deg_context=deg_context,
            struct_tokens=struct_tokens,
            qwen_context=qwen_context,
            raw_confidence=raw_confidence,
            gate_confidence=gate_confidence,
            condition_dropout_mask=dropout_mask,
            prior_corruption_mask=corruption_mask,
        )
