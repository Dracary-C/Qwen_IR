"""Named schemas and compatibility helpers for Qwen degradation priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

DEGRADATION_ORDER = ("noise", "blur", "haze", "rain", "low_light")
LAYOUT_ORDER = (
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
)

DATASET_TO_DEGRADATION = {
    "denoising": "noise", "noise": "noise", "noisy": "noise",
    "gopro": "blur", "blur": "blur", "blurry": "blur",
    "dehazeformer": "haze", "haze": "haze", "hazy": "haze",
    "rain200l": "rain", "rain": "rain", "rainy": "rain",
    "lol-v2": "low_light", "lol": "low_light", "low-light": "low_light",
    "low_light": "low_light", "lowlight": "low_light",
}
SEVERITY_TO_FLOAT = {
    "none": 0.0, "mild": 1.0 / 3.0, "moderate": 2.0 / 3.0,
    "serious": 1.0, "severe": 1.0,
}
DEFAULT_LAYOUTS: Mapping[str, tuple[float, ...]] = {
    "noise": (1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    "blur": (1, 0, 0, 1, 0, 0, 0, 0, 1, 0),
    "haze": (1, 0, 0, 1, 0, 0, 1, 0, 0, 0),
    "rain": (1, 0, 0, 0, 1, 1, 0, 0, 0, 0),
    "low_light": (1, 0, 0, 1, 0, 0, 0, 1, 0, 0),
}


def normalize_degradation_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    if key in DATASET_TO_DEGRADATION:
        return DATASET_TO_DEGRADATION[key]
    raise ValueError(f"Unsupported degradation name: {name!r}")


def normalize_severity(value: Any) -> float:
    """Map prompt severity text to [0, 1], accepting serious and severe."""
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Severity numeric value must be in [0, 1], got {numeric}")
        return numeric
    key = str(value).strip().lower()
    if key not in SEVERITY_TO_FLOAT:
        raise ValueError(f"Unsupported severity value: {value!r}")
    return SEVERITY_TO_FLOAT[key]


def _vector(value: Any, size: int, *, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = value.detach().clone().to(dtype=dtype) if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype)
    if tensor.ndim != 1 or tensor.numel() != size:
        raise ValueError(f"{name} must have shape [{size}], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def extract_main_logits(payload: Mapping[str, Any]) -> torch.Tensor:
    """Extract five raw Qwen average log-probabilities in canonical order."""
    scoring = payload.get("condition_scoring")
    scores = scoring.get("scores") if isinstance(scoring, Mapping) else None
    if not isinstance(scores, list):
        raise ValueError("Qwen record is missing condition_scoring.scores")
    by_name: dict[str, float] = {}
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        candidate = str(item.get("candidate", "")).strip()
        if candidate in DEGRADATION_ORDER and item.get("avg_logprob") is not None:
            by_name[candidate] = float(item["avg_logprob"])
    missing = [name for name in DEGRADATION_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"Qwen record is missing avg_logprob for: {missing}")
    return _vector([by_name[name] for name in DEGRADATION_ORDER], 5, name="main_logits_5")


def _parsed_output(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed = payload.get("qwen_parsed", payload.get("parsed"))
    if isinstance(parsed, Mapping):
        return parsed
    scoring = payload.get("condition_scoring")
    if isinstance(scoring, Mapping) and isinstance(scoring.get("parsed"), Mapping):
        return scoring["parsed"]
    return {}


@dataclass(frozen=True)
class StructuredPriorV2:
    """Named per-sample prior with calibrated probabilities and confidence."""

    severity_5: torch.Tensor
    main_logits_5: torch.Tensor
    main_probs_5: torch.Tensor
    layout_10: torch.Tensor
    raw_margin: torch.Tensor
    calibrated_confidence: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity_5", _vector(self.severity_5, 5, name="severity_5"))
        object.__setattr__(self, "main_logits_5", _vector(self.main_logits_5, 5, name="main_logits_5"))
        probs = _vector(self.main_probs_5, 5, name="main_probs_5")
        if (probs < 0).any() or not torch.isclose(probs.sum(), probs.new_tensor(1.0), atol=1e-5):
            raise ValueError(f"main_probs_5 must sum to 1, got {float(probs.sum())}")
        object.__setattr__(self, "main_probs_5", probs)
        layout = _vector(self.layout_10, 10, name="layout_10")
        if (layout < 0).any() or (layout > 1).any():
            raise ValueError("layout_10 values must be in [0, 1]")
        object.__setattr__(self, "layout_10", layout)
        object.__setattr__(self, "raw_margin", _vector(self.raw_margin, 1, name="raw_margin"))
        confidence = _vector(self.calibrated_confidence, 1, name="calibrated_confidence")
        if not 0.0 <= float(confidence[0]) <= 1.0:
            raise ValueError("calibrated_confidence must be in [0, 1]")
        object.__setattr__(self, "calibrated_confidence", confidence)

    @classmethod
    def from_qwen_payload(cls, payload: Mapping[str, Any], *, temperature: float) -> "StructuredPriorV2":
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        logits = extract_main_logits(payload)
        raw_probs = torch.softmax(logits, dim=0)
        probs = torch.softmax(logits / float(temperature), dim=0)
        top2 = torch.topk(raw_probs, k=2).values
        parsed = _parsed_output(payload)
        legacy = payload.get("prior_vector_21")
        legacy_values = legacy if isinstance(legacy, list) and len(legacy) == 21 else None

        severity_values = []
        for index, name in enumerate(DEGRADATION_ORDER):
            value = parsed.get(name)
            if value is None and legacy_values is not None:
                value = legacy_values[index]
            severity_values.append(normalize_severity(value if value is not None else "none"))

        layout_obj = parsed.get("degradation_layout")
        layout_values = []
        for index, name in enumerate(LAYOUT_ORDER):
            if isinstance(layout_obj, Mapping) and name in layout_obj:
                value = layout_obj[name]
            elif isinstance(layout_obj, Mapping) and f"is_{name}" in layout_obj:
                value = layout_obj[f"is_{name}"]
            elif legacy_values is not None:
                value = legacy_values[10 + index]
            else:
                value = False
            layout_values.append(1.0 if bool(value) else 0.0)

        return cls(
            severity_5=torch.tensor(severity_values),
            main_logits_5=logits,
            main_probs_5=probs,
            layout_10=torch.tensor(layout_values),
            raw_margin=(top2[0] - top2[1]).reshape(1),
            calibrated_confidence=probs.max().reshape(1),
        )

    @classmethod
    def from_legacy_vector(cls, prior: torch.Tensor | list[float]) -> "StructuredPriorV2":
        vector = _vector(prior, 21, name="prior_vector_21")
        probs = vector[5:10].clamp_min(0)
        total = probs.sum()
        probs = probs / total if float(total) > 0 else torch.full_like(probs, 0.2)
        return cls(
            severity_5=vector[0:5],
            main_logits_5=probs.clamp_min(1e-12).log(),
            main_probs_5=probs,
            layout_10=vector[10:20],
            raw_margin=vector[20:21],
            calibrated_confidence=probs.max().reshape(1),
        )

    def to_legacy_vector(self) -> torch.Tensor:
        return torch.cat([self.severity_5, self.main_probs_5, self.layout_10, self.raw_margin])

    def to_model_vector(self, mode: str) -> torch.Tensor:
        """Create the unchanged 21D model input for a named ablation."""
        normalized = str(mode).strip().lower()
        if normalized == "qwen_probs":
            severity = torch.zeros_like(self.severity_5)
        elif normalized in {"qwen_probs_severity", "confidence_gate"}:
            severity = self.severity_5
        else:
            raise ValueError(f"Unsupported V2 model mode: {mode!r}")
        # A2/A3 deliberately do not activate layout.
        return torch.cat([severity, self.main_probs_5, torch.zeros_like(self.layout_10), self.raw_margin])

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "severity_5": self.severity_5,
            "main_logits_5": self.main_logits_5,
            "main_probs_5": self.main_probs_5,
            "layout_10": self.layout_10,
            "raw_margin": self.raw_margin,
            "calibrated_confidence": self.calibrated_confidence,
        }


def build_prior_from_degradation_name(
    name: str, *, severity_value: float = 1.0, prob_value: float = 1.0,
    margin_value: float = 1.0, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a legacy 21D training prior from a degradation folder name."""
    degradation = normalize_degradation_name(name)
    deg_idx = DEGRADATION_ORDER.index(degradation)
    severity = torch.zeros(5, dtype=dtype)
    severity[deg_idx] = float(severity_value)
    probs = torch.zeros(5, dtype=dtype)
    probs[deg_idx] = float(prob_value)
    return torch.cat([
        severity, probs, torch.tensor(DEFAULT_LAYOUTS[degradation], dtype=dtype),
        torch.tensor([float(margin_value)], dtype=dtype),
    ])


def split_structured_prior(prior: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split legacy [B, 20/21] or [20/21] prior into named parts."""
    if prior.ndim == 1:
        prior = prior.unsqueeze(0)
    if prior.ndim != 2:
        raise ValueError(f"Expected qwen_prompt prior [B, 20/21], got {tuple(prior.shape)}")
    if prior.shape[-1] not in {20, 21}:
        raise ValueError(f"Expected qwen_prompt prior dim 20 or 21, got {prior.shape[-1]}")
    return {
        "severity": prior[:, 0:5],
        "degradation_probs": prior[:, 5:10],
        "layout": prior[:, 10:20],
        "margin": prior[:, 20:21] if prior.shape[-1] == 21 else prior.new_ones(prior.shape[0], 1),
    }
