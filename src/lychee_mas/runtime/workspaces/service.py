"""Framework-neutral isolated workspace materialization."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lychee_mas.core.types import TaskQuery


@dataclass(frozen=True)
class MaterializedWorkspace:
    root: Path
    task_text: str
    visible_paths: tuple[str, ...]
    details: dict[str, Any]


class WorkspaceService:
    """Create one opaque, attempt-local workspace with benchmark attachments."""

    def __init__(self, work_root: str | Path) -> None:
        self.work_root = Path(work_root)

    def materialize(self, query: TaskQuery, benchmark: Any = None) -> MaterializedWorkspace:
        workspace = self._allocate()
        task_text = query.question
        copied: list[str] = []
        materialized = None
        if benchmark is not None:
            materialized = benchmark.materialize_case(query, workspace, task_text)
            task_text = materialized.task_text
            copied.extend(materialized.visible_paths)
        for source_text in self.referenced_paths(query):
            source = Path(source_text).expanduser()
            if not source.exists():
                continue
            destination = workspace / source.name
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            copied.append(str(destination))
            task_text = task_text.replace(str(source), destination.name)
        copied = list(dict.fromkeys(copied))
        if copied:
            names = ", ".join(Path(path).name for path in copied)
            task_text = (
                f"{task_text}\n\nThe referenced file(s) are available in the current "
                f"workspace: {names}"
            )
        details: dict[str, Any] = {
            "opaque_workspace_id": workspace.name,
            "workspace_policy": "uuid4_isolated_per_attempt_v1",
            "workspace_is_new": True,
            "preexisting_entry_count": 0,
            "attachment_filename_policy": "preserve_official_filename",
            "visible_attachment_names": [Path(path).name for path in copied],
        }
        if benchmark is not None and materialized is not None:
            details["benchmark_id"] = benchmark.id
            details["benchmark_materialization"] = materialized.details
        return MaterializedWorkspace(
            root=workspace,
            task_text=task_text,
            visible_paths=tuple(copied),
            details=details,
        )

    def _allocate(self) -> Path:
        self.work_root.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            workspace = self.work_root / f"ws_{uuid.uuid4().hex}"
            try:
                workspace.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return workspace
        raise RuntimeError("failed to allocate a unique tool workspace after 8 attempts")

    @staticmethod
    def referenced_paths(query: TaskQuery) -> list[str]:
        haystack = "\n".join(
            part for part in (query.question, query.context or "") if isinstance(part, str)
        )
        values: list[str] = []
        for match in re.finditer(r"Referenced file path:\s*(.+)", haystack):
            value = match.group(1).strip()
            if value not in values:
                values.append(value)
        return values
