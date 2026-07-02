from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "module" / "vendor"

_DEFAULT_REPOS = {
    "tpgdiff": VENDOR_ROOT / "tpgdiff",
}



def default_repo_root(key: str) -> Path:
    env_key = f"METHODHUB_{key.upper().replace('-', '_')}_ROOT"
    env_value = os.environ.get(env_key) or os.environ.get(f"QWEN_IR_{key.upper().replace('-', '_')}_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    if key not in _DEFAULT_REPOS:
        raise KeyError(f"Unknown repo key: {key}")
    internal = _DEFAULT_REPOS[key]
    if internal.exists():
        return internal.resolve()
    return internal.resolve()


def require_exists(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@contextmanager
def push_sys_path(*paths: Path) -> Iterator[None]:
    inserted = []
    for path in reversed([Path(p).resolve() for p in paths if p]):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted.append(value)
    try:
        yield
    finally:
        for value in inserted:
            if value in sys.path:
                sys.path.remove(value)


@contextmanager
def push_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
