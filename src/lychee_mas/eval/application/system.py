"""Application service for Eval bootstrap, workspaces, and environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..environment.service import discover_environments, inspect_environment


class SystemApplicationService:
    """Expose Eval-wide read models without depending on an HTTP transport."""

    def __init__(self, *, repo_root: Path, workspace_catalog) -> None:
        self.repo_root = repo_root
        self.workspace_catalog = workspace_catalog

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "repo_root": str(self.repo_root)}

    def bootstrap(self) -> dict[str, Any]:
        return self.workspace_catalog.core()

    def workspace(self, workspace_name: str) -> dict[str, Any]:
        return self.workspace_catalog.workspace(workspace_name)

    def environments(self) -> dict[str, Any]:
        return discover_environments(self.repo_root)

    def check_environment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return inspect_environment(
            self.repo_root,
            payload.get("environment") or payload,
        )


__all__ = ["SystemApplicationService"]
