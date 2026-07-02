"""Runtime adapters used by Qwen_IR scripts."""

from __future__ import annotations

from typing import Any

from module.runtime.adapters.tpgdiff import TPGDiffPriorAdapter, TPGDiffRuntimeAdapter

_ALIASES = {
    "tpgdiff": TPGDiffPriorAdapter,
    "tpgdiff-prior": TPGDiffPriorAdapter,
    "tpgdiff-runtime": TPGDiffRuntimeAdapter,
    "tpgdiff-restore": TPGDiffRuntimeAdapter,
    "tpgd-runtime": TPGDiffRuntimeAdapter,
    "tpgd": TPGDiffRuntimeAdapter,
}


def build(name: str, **kwargs: Any):
    key = name.lower()
    if key not in _ALIASES:
        raise KeyError(f"Unknown runtime: {name}. Available: {', '.join(sorted(_ALIASES))}")
    return _ALIASES[key](**kwargs)


def available_methods() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "class": cls.__name__,
            "capabilities": list(getattr(cls, "capabilities", ())),
            "source_refs": [ref.as_dict() for ref in getattr(cls, "source_refs", ())],
        }
        for name, cls in _ALIASES.items()
    }


__all__ = ["available_methods", "build", "TPGDiffPriorAdapter", "TPGDiffRuntimeAdapter"]
