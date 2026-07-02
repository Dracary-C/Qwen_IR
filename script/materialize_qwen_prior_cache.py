#!/usr/bin/env python
"""Materialize existing Qwen JSONL records into an image-mirrored prior cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.layout_prompt import StructuredPriorV2


def _payload(record: dict[str, Any], relative_image: Path) -> dict[str, Any]:
    parsed = record.get("qwen_parsed", record.get("parsed"))
    scoring = record.get("condition_scoring")
    if not isinstance(parsed, dict) or not isinstance(scoring, dict):
        raise ValueError("record must contain parsed and condition_scoring objects")
    target = record.get("source_image", record.get("target"))
    result = {
        "source_image": str(target),
        "relative_image": str(relative_image),
        "dataset": record.get("dataset"),
        "expected_main_degradation": record.get(
            "expected_main_degradation", record.get("expected_degradation")
        ),
        "qwen_parsed": parsed,
        "condition_scoring": scoring,
    }
    # Keep legacy consumers functional while V2 consumers recompute calibrated
    # probabilities from the preserved raw avg_logprob values.
    result["prior_vector_21"] = StructuredPriorV2.from_qwen_payload(
        result, temperature=1.0
    ).to_legacy_vector().tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    records = 0
    written = 0
    skipped = 0
    with args.input_jsonl.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records += 1
            record = json.loads(line)
            target_value = record.get("source_image", record.get("target"))
            if target_value in (None, ""):
                raise ValueError(f"line {line_number} has no target image")
            target = Path(str(target_value)).expanduser().resolve()
            try:
                relative_image = target.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number} target is outside image root: {target}"
                ) from exc
            output = output_root / relative_image.with_suffix(".json")
            if output.exists() and not args.overwrite:
                skipped += 1
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(_payload(record, relative_image), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1
    summary = {
        "input_jsonl": str(args.input_jsonl.expanduser().resolve()),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "records": records,
        "written": written,
        "skipped": skipped,
    }
    (output_root / "materialize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
