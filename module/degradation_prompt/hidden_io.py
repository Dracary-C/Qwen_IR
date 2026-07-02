"""I/O helpers for Assessment Reasoning hidden-state packs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import torch

HiddenKey = Literal["prefix_hidden", "generated_hidden", "condition_hidden"]


class AssessmentHiddenPack(TypedDict, total=False):
    round_index: int
    query: str
    answer: str
    task_type: str
    prefix_hidden: torch.Tensor
    generated_hidden: torch.Tensor
    condition_hidden: torch.Tensor


def load_assessment_hidden(path: str | Path, map_location: str = "cpu") -> AssessmentHiddenPack:
    """Load one `.pt` file produced by the legacy assessment-hidden extraction pipeline."""

    return torch.load(Path(path).expanduser(), map_location=map_location)


def select_hidden(pack: AssessmentHiddenPack, key: HiddenKey = "condition_hidden") -> torch.Tensor:
    """Select the hidden-state tensor intended for downstream prior experiments."""

    if key not in pack:
        raise KeyError(f"Hidden key not found in pack: {key}")
    return pack[key]
