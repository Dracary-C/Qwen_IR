from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class SourceRef:
    repo: str
    entrypoints: Tuple[str, ...]
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "entrypoints": list(self.entrypoints),
            "notes": self.notes,
        }


class MethodAdapter:
    name = "method"
    capabilities: Tuple[str, ...] = ()
    source_refs: Tuple[SourceRef, ...] = ()

    def __init__(self) -> None:
        self.model: Any = None
        self.loaded: bool = False

    def load(self) -> "MethodAdapter":
        raise NotImplementedError

    def ensure_loaded(self) -> "MethodAdapter":
        if not self.loaded or self.model is None:
            return self.load()
        return self

    def close(self) -> None:
        self.model = None
        self.loaded = False

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "source_refs": [ref.as_dict() for ref in self.source_refs],
        }


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

