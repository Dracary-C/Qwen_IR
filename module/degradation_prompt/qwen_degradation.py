"""Qwen-prompt degradation severity/probability encoders."""

from __future__ import annotations

import torch
from torch import nn


class DegradationTimeContextEncoder(nn.Module):
    """Encode severity_5 + degradation_probs_5 into TPGDiff deg_context."""

    def __init__(self, context_dim: int, hidden_dim: int = 256, input_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, severity: torch.Tensor, degradation_probs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([severity, degradation_probs], dim=-1))
