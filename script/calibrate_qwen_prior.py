#!/usr/bin/env python
"""Fit one temperature on train Qwen logits and report calibration metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.layout_prompt import DEGRADATION_ORDER, extract_main_logits
from module.layout_prompt.schema import normalize_degradation_name


def _record_files(root: Path, split: str) -> list[Path]:
    split_root = root / split
    search_root = split_root if split_root.exists() else root
    files = sorted(search_root.rglob("*.json"))
    return [
        path for path in files
        if path.name not in {"manifest.json", "summary.json", "evaluation_summary.json"}
    ]


def _label(payload: dict[str, Any], path: Path) -> int:
    value = payload.get("expected_main_degradation") or payload.get("dataset")
    if value in (None, ""):
        raise ValueError(f"Missing expected degradation label in {path}")
    name = normalize_degradation_name(str(value))
    return DEGRADATION_ORDER.index(name)


def load_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, list[Path]]:
    logits: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[Path] = []
    for path in _record_files(root, split):
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            logit = extract_main_logits(payload)
            label = _label(payload, path)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Qwen calibration record {path}: {exc}") from exc
        logits.append(logit.to(torch.float64))
        labels.append(label)
        paths.append(path)
    if not logits:
        raise RuntimeError(f"No Qwen prior JSON records found for split={split!r} under {root}")
    return torch.stack(logits), torch.tensor(labels, dtype=torch.long), paths


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=200,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not 1e-3 <= temperature <= 1e3:
        raise RuntimeError(f"Fitted temperature is numerically implausible: {temperature}")
    return temperature


def _binary_auroc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    values = [(float(score), int(label)) for score, label in zip(scores, positives)]
    positives_n = sum(label for _, label in values)
    negatives_n = len(values) - positives_n
    if positives_n == 0 or negatives_n == 0:
        return float("nan")
    values.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in values[index:end])
        index = end
    return (positive_rank_sum - positives_n * (positives_n + 1) / 2.0) / (positives_n * negatives_n)


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict[str, Any]:
    probs = torch.softmax(logits / float(temperature), dim=1)
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels)
    errors = ~correct
    one_hot = F.one_hot(labels, num_classes=len(DEGRADATION_ORDER)).to(probs.dtype)
    nll = F.nll_loss(probs.clamp_min(1e-15).log(), labels)
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()

    ece = probs.new_zeros(())
    bins = torch.linspace(0.0, 1.0, 16, dtype=probs.dtype)
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.float().mean() * (
                confidence[in_bin].mean() - correct[in_bin].to(probs.dtype).mean()
            ).abs()

    order = torch.argsort(confidence, descending=True, stable=True)
    cumulative_risk = errors[order].to(probs.dtype).cumsum(0) / torch.arange(
        1, len(errors) + 1, dtype=probs.dtype
    )
    quantiles = torch.quantile(
        confidence,
        torch.tensor([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0], dtype=probs.dtype),
    )
    return {
        "samples": int(labels.numel()),
        "accuracy": float(correct.to(probs.dtype).mean()),
        "nll": float(nll),
        "ece_15_equal_width": float(ece),
        "brier": float(brier),
        "error_detection_auroc": _binary_auroc(1.0 - confidence, errors),
        "aurc": float(cumulative_risk.mean()),
        "confidence_quantiles": {
            key: float(value)
            for key, value in zip(("min", "p10", "p20", "p50", "p80", "p90", "max"), quantiles)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--train-split", default="Train")
    parser.add_argument("--val-split", default="Val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.input_root.expanduser().resolve()
    train_logits, train_labels, train_paths = load_split(root, args.train_split)
    val_logits, val_labels, val_paths = load_split(root, args.val_split)
    temperature = fit_temperature(train_logits, train_labels)
    result = {
        "schema_version": "qwen_temperature_v1",
        "temperature": temperature,
        "class_order": list(DEGRADATION_ORDER),
        "fit_split": args.train_split,
        "evaluation_split": args.val_split,
        "input_root": str(root),
        "fit_uses_validation": False,
        "train": {
            "raw_temperature_1": calibration_metrics(train_logits, train_labels, 1.0),
            "calibrated": calibration_metrics(train_logits, train_labels, temperature),
            "first_record": str(train_paths[0]),
        },
        "val": {
            "raw_temperature_1": calibration_metrics(val_logits, val_labels, 1.0),
            "calibrated": calibration_metrics(val_logits, val_labels, temperature),
            "first_record": str(val_paths[0]),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
