"""Small, domain-neutral primitives for JSON-backed Studio registries."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_identifier(value: str, label: str) -> str:
    """Validate one portable filename-safe Spec or Instance identifier."""

    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} may contain only letters, numbers, '.', '_' and '-'")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace one human-readable JSON registry document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_normalized_documents(
    root: Path,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load valid JSON documents and attach their repository path read model."""

    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            value = normalizer(raw)
        except (OSError, TypeError, ValueError):
            continue
        rows.append({**value, "registry_path": str(path)})
    return rows


__all__ = ["atomic_write_json", "load_normalized_documents", "validate_identifier"]
