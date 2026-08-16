"""Unified file-backed TeamSpec registry for Eval Studio."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .team_spec import normalize_team_spec_document

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TeamSpecRegistry:
    """Store reusable TeamSpec v4 documents independently of experiments."""

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
                    "updated_at_utc": raw.get("updated_at_utc"),
                }
            )
        return rows

    def save(self, spec: dict[str, Any]) -> dict[str, Any]:
        value = normalize_team_spec_document({"schema_version": 4, **spec})
        team_id = str(value["id"])
        if not _SAFE_ID.fullmatch(team_id):
            raise ValueError("team id may contain only letters, numbers, '.', '_' and '-'")
        value["updated_at_utc"] = _utc_now()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{team_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return {**value, "registry_path": str(path)}

    def delete(self, team_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(team_id):
            raise ValueError("invalid team id")
        path = self.root / f"{team_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": team_id, "deleted": True, "path": str(path)}

    def get(self, team_id: str) -> dict[str, Any]:
        match = next((item for item in self.all() if item["id"] == team_id), None)
        if match is None:
            raise ValueError(f"unknown TeamSpec {team_id!r}")
        return match
