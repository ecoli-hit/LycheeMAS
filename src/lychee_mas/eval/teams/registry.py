"""Unified file-backed TeamSpec registry for Eval Studio."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.json_store import atomic_write_json
from .compiler import normalize_team_spec_document
from .contracts import TEAM_SPEC_SCHEMA_VERSION

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _file_updated_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


class TeamSpecRegistry:
    """Store canonical TeamSpec v14 documents independently of experiments."""

    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root.resolve() / "configs/eval_studio/teams/specs"

    def all(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        rows = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                value = normalize_team_spec_document(raw)
            except (OSError, ValueError):
                continue
            rows.append(
                {
                    **value,
                    "registry_path": str(path),
                    "updated_at_utc": _file_updated_at(path),
                }
            )
        return rows

    def save(self, spec: dict[str, Any]) -> dict[str, Any]:
        # API reads attach registry metadata beside the canonical document so the
        # Studio can round-trip an existing row. These two fields are transport
        # metadata and must never enter the strict TeamSpec contract on disk.
        document = {
            key: value
            for key, value in spec.items()
            if key not in {"registry_path", "updated_at_utc"}
        }
        value = normalize_team_spec_document(
            {"schema_version": TEAM_SPEC_SCHEMA_VERSION, **document}
        )
        team_id = str(value["id"])
        if not _SAFE_ID.fullmatch(team_id):
            raise ValueError("team id may contain only letters, numbers, '.', '_' and '-'")
        path = self.root / f"{team_id}.json"
        atomic_write_json(path, value)
        return {
            **value,
            "registry_path": str(path),
            "updated_at_utc": _file_updated_at(path),
        }

    def delete(self, team_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(team_id):
            raise ValueError("invalid team id")
        path = self.root / f"{team_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": team_id, "deleted": True, "path": str(path)}

    def get(self, team_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(team_id):
            raise ValueError("invalid team id")
        path = self.root / f"{team_id}.json"
        if not path.is_file():
            raise ValueError(f"unknown TeamSpec {team_id!r}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid TeamSpec {team_id!r}: {exc}") from exc
        return normalize_team_spec_document(raw)
