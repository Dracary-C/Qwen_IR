#!/usr/bin/env python3
"""Validate final TPGDiff degradation JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEGRADATION_KEYS = ["noise", "blur", "haze", "rain", "low_light"]
EXPECTED_KEYS = DEGRADATION_KEYS + ["main_degradation", "degradation_distribution", "degradation_position", "confidence"]
SEVERITY_OPTIONS = {"none", "mild", "moderate", "serious", "severe"}
SEVERITY_RANK = {"none": 0, "mild": 1, "moderate": 2, "serious": 3, "severe": 3}
MAIN_OPTIONS = {"none", "noise", "blur", "haze", "rain", "low_light"}
CONFIDENCE_OPTIONS = {"low", "medium", "high"}
DISTRIBUTION_OPTIONS = {"uniform", "non_uniform", "uncertain"}


def expected_main_from_record(record: dict[str, Any]) -> str | None:
    for key in ("expected_degradation", "expected_main_degradation", "expected_main"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def validate_output(obj: Any, expected_main: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["output is not a JSON object"]
    if list(obj.keys()) != EXPECTED_KEYS:
        errors.append(f"keys must be exactly {EXPECTED_KEYS}, got {list(obj.keys())}")
        return errors
    for key in DEGRADATION_KEYS:
        if obj.get(key) not in SEVERITY_OPTIONS:
            errors.append(f"{key} must be one of {sorted(SEVERITY_OPTIONS)}, got {obj.get(key)!r}")
    main = obj.get("main_degradation")
    if main not in MAIN_OPTIONS:
        errors.append(f"main_degradation must be one of {sorted(MAIN_OPTIONS)}, got {main!r}")
    elif main != "none" and obj.get(main) == "none":
        errors.append(f"main_degradation={main!r} but {main} severity is none")
    if expected_main is not None and main != expected_main:
        errors.append(f"main_degradation must match expected {expected_main!r}, got {main!r}")
    if main == "none":
        if any(obj.get(key) != "none" for key in DEGRADATION_KEYS):
            errors.append("main_degradation='none' requires all degradation severities to be none")
    elif not any(obj.get(key) != "none" for key in DEGRADATION_KEYS):
        errors.append("at least one degradation severity must be non-none")

    distribution = obj.get("degradation_distribution")
    if distribution not in DISTRIBUTION_OPTIONS:
        errors.append(f"degradation_distribution must be one of {sorted(DISTRIBUTION_OPTIONS)}, got {distribution!r}")

    position = obj.get("degradation_position")
    if not isinstance(position, list) or not all(isinstance(item, str) for item in position):
        errors.append("degradation_position must be a list of strings")
    elif not position:
        errors.append("degradation_position must not be empty; use ['uncertain'] if location is unclear")

    confidence = obj.get("confidence")
    if confidence not in CONFIDENCE_OPTIONS:
        errors.append(f"confidence must be one of {sorted(CONFIDENCE_OPTIONS)}, got {confidence!r}")

    if main in MAIN_OPTIONS and main != "none" and all(obj.get(key) in SEVERITY_RANK for key in DEGRADATION_KEYS):
        main_rank = SEVERITY_RANK[obj[main]]
        max_rank = max(SEVERITY_RANK[obj[key]] for key in DEGRADATION_KEYS)
        if main_rank < max_rank:
            strongest = [key for key in DEGRADATION_KEYS if SEVERITY_RANK[obj[key]] == max_rank]
            errors.append(
                f"main_degradation={main!r} has severity {obj[main]!r}, "
                f"but strongest severity is {max_rank} for {strongest}"
            )
    return errors


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected .jsonl records, a JSON list, or a JSON object with records")


def output_from_record(record: dict[str, Any]) -> Any:
    if "output" in record:
        return record["output"]
    if "parsed" in record:
        return record["parsed"]
    if "raw_output" in record:
        return json.loads(record["raw_output"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    records = load_records(args.input)
    failures = []
    for idx, record in enumerate(records, start=1):
        output = output_from_record(record)
        expected_main = expected_main_from_record(record)
        errors = validate_output(output, expected_main=expected_main)
        if errors:
            failures.append({
                "index": idx,
                "dataset": record.get("dataset"),
                "target": record.get("target"),
                "expected_main_degradation": expected_main,
                "output": output,
                "errors": errors,
            })

    summary = {
        "input": str(args.input),
        "total": len(records),
        "valid": len(records) - len(failures),
        "invalid": len(failures),
        "all_valid": not failures,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
