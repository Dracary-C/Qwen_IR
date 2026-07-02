"""Projection module for using Assessment Reasoning hidden states as priors."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

PoolMode = Literal["mean", "last", "first"]


class AssessPriorAdapter(nn.Module):
    """Project DepictQA/RAR hidden states into a target prior dimension.

    Input contract:
        hidden: [B, T, input_dim], usually a cached visual-language assessment hidden state.

    Output contract:
        prior: [B, output_dim]

    The output_dim should be set to the dimension expected by the TPGDiff prior
    location that the experiment chooses to replace.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        pool: PoolMode = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pool not in {"mean", "last", "first"}:
            raise ValueError(f"Unsupported pool mode: {pool}")
        self.pool = pool
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [B, T, C], got shape {tuple(hidden.shape)}")

        if self.pool == "mean":
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                denom = weights.sum(dim=1).clamp_min(1.0)
                pooled = (hidden * weights).sum(dim=1) / denom
        elif self.pool == "last":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden[:, 0, :]

        param_dtype = next(self.proj.parameters()).dtype
        pooled = pooled.to(dtype=param_dtype)
        return self.proj(pooled)
