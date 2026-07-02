"""Vendored TPGDiff runtime source and attribution files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIVERSAL_RESTORATION_ROOT = ROOT / "universal-restoration"
TPGD_CODE_ROOT = UNIVERSAL_RESTORATION_ROOT / "config" / "tpgd-sde"

__all__ = ["ROOT", "UNIVERSAL_RESTORATION_ROOT", "TPGD_CODE_ROOT"]

