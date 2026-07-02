"""Runtime source repository metadata for Qwen_IR.

TPGDiff runtime code is vendored inside this project by default.
Set QWEN_IR_TPGDIFF_ROOT only when intentionally testing another source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_TPGDIFF_ROOT = PROJECT_ROOT / "module" / "vendor" / "tpgdiff"


@dataclass(frozen=True)
class SourceRepo:
    name: str
    path: Path
    license_note: str
    role: str


def _env_path(key: str, fallback: Path) -> Path:
    value = os.environ.get(key)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


SOURCE_REPOS: dict[str, SourceRepo] = {
    "tpgdiff": SourceRepo(
        name="TPGDiff",
        path=_env_path("QWEN_IR_TPGDIFF_ROOT", VENDORED_TPGDIFF_ROOT),
        license_note="Vendored TPGDiff runtime; see module/vendor/tpgdiff/NOTICE.md and LICENSE.",
        role="restoration backbone and SDE runtime",
    ),
}


def source_repo(name: str) -> SourceRepo:
    key = name.lower()
    if key not in SOURCE_REPOS:
        raise KeyError(f"Unknown source repo: {name}")
    return SOURCE_REPOS[key]
