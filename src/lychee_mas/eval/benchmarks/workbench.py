"""WorkBench Revisited loader, native tools, and official state scorer."""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .base import Benchmark, ToolBundle, resolve_provider_ids
from .common import (
    clone_git_source,
    prepared_benchmark_dir,
    raw_source_dir,
    remove_path,
)
from .registry import register_benchmark

SOURCES: dict[str, Any] = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {"env": None, "default_ids": []},
    "github": {
        "env": None,
        "default_ids": ["https://github.com/olly-styles/WorkBench.git"],
        "revision": "da6f8ee8d9f8efc87b2f5f3cc9edc1befdac726e",
        "source_type": "git",
        "purpose": "official v2 tasks, state databases, tools, and evaluator",
    },
    "other_defaults": [],
    "fallback_files": [],
}

OFFICIAL_REPO = resolve_provider_ids(SOURCES, "github")[0]
OFFICIAL_REVISION = str(SOURCES["github"]["revision"])
DATETIME_PREFIX = (
    "Today's date is Thursday, 2023-11-30 and the current time is 00:00:00. "
    "Remember the current date and time when completing tasks. "
    "Meetings must not start before 9am or end after 6pm."
)
DOMAINS = (
    "analytics",
    "calendar",
    "customer_relationship_manager",
    "email",
    "multi_domain",
    "project_management",
)


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("workbench")


def _raw_repo() -> Path:
    return raw_source_dir("workbench", "github", "olly-styles/WorkBench")


def _ready(path: Path) -> bool:
    tasks = path / "data" / "processed" / "tasks_and_outcomes"
    return all((tasks / f"{domain}_tasks_and_outcomes.csv").is_file() for domain in DOMAINS)


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "github"}:
        raise ValueError("WorkBench Revisited is published through its official GitHub repository")
    prepared = source_dir(root)
    if _ready(prepared) and not force_download:
        return prepared
    raw = clone_git_source(
        "workbench",
        OFFICIAL_REPO,
        _raw_repo(),
        revision=OFFICIAL_REVISION,
        force=force_download,
    )
    remove_path(prepared)
    shutil.copytree(raw, prepared, ignore=shutil.ignore_patterns(".git"))
    if not _ready(prepared):
        raise RuntimeError("official WorkBench source is incomplete")
    return prepared


def load_workbench(n: Optional[int] = None) -> list[dict[str, Any]]:
    import csv

    root = ensure_source()
    task_root = root / "data" / "processed" / "tasks_and_outcomes"
    records = []
    for domain in DOMAINS:
        path = task_root / f"{domain}_tasks_and_outcomes.csv"
        with path.open("r", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                task_id = f"{domain}/{index}"
                records.append(
                    {
                        "task": "workbench",
                        "kind": "workbench",
                        "question": str(row["task"]),
                        "gold": {
                            "task_id": task_id,
                            "actions": list(ast.literal_eval(row["outcome"])),
                        },
                        "context": None,
                        "metadata": {
                            "task_id": task_id,
                            "benchmark_adapter": "workbench",
                            "workbench_source": str(root),
                            "domain": domain,
                            "domains": list(ast.literal_eval(row.get("domains") or "[]")),
                            "system_prompt": DATETIME_PREFIX,
                            "official_revision": OFFICIAL_REVISION,
                        },
                    }
                )
                if n is not None and len(records) >= n:
                    return records
    return records


class WorkBenchSession:
    """Keep one case's official mutable state in one worker process."""

    def __init__(self, source: str | Path):
        worker = Path(__file__).with_name("workbench_worker.py")
        self._process = subprocess.Popen(
            [sys.executable, str(worker), str(Path(source).resolve())],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self._lock = threading.Lock()
        hello = self._read()
        if hello.get("status") != "ready":
            raise RuntimeError(f"WorkBench worker failed to start: {hello}")
        self.tool_specs = list(hello["tools"])

    def _read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"WorkBench worker exited unexpectedly: {stderr}")
        return json.loads(line)

    def request(self, payload: dict[str, Any]) -> Any:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("WorkBench worker is not running")
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            response = self._read()
        if response.get("status") != "ok":
            raise RuntimeError(
                f"{response.get('error_type', 'WorkBenchError')}: "
                f"{response.get('error_message', 'unknown error')}"
            )
        return response.get("result")

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.request({"operation": "close"})
            except Exception:
                self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()


def _annotation(schema_type: str) -> type:
    return {"integer": int, "number": float, "boolean": bool}.get(schema_type, str)


def create_tools(metadata: dict[str, Any]) -> tuple[list[Any], WorkBenchSession]:
    source = metadata.get("workbench_source") or str(ensure_source())
    session = WorkBenchSession(source)
    wrappers = []
    for tool_spec in session.tool_specs:
        official_name = str(tool_spec["name"])
        exposed_name = re.sub(r"[^a-zA-Z0-9_-]", "_", official_name)

        def invoke(_official_name=official_name, **kwargs) -> str:
            result = session.request(
                {"operation": "call", "name": _official_name, "arguments": kwargs}
            )
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)

        parameters = []
        annotations = {}
        for name, schema in tool_spec.get("args_schema", {}).items():
            annotation = _annotation(str(schema.get("type") or "string"))
            annotations[name] = annotation
            default = schema["default"] if "default" in schema else inspect.Parameter.empty
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        invoke.__name__ = exposed_name
        invoke.__qualname__ = exposed_name
        invoke.__doc__ = str(tool_spec.get("description") or official_name)
        invoke.__annotations__ = {**annotations, "return": str}
        setattr(
            invoke,
            "__signature__",
            inspect.Signature(parameters=parameters, return_annotation=str),
        )
        wrappers.append(invoke)
    return wrappers, session


def predicted_actions(
    record: dict[str, Any], exposed_to_official: dict[str, str] | None = None
) -> list[str]:
    actions = []
    for request in record.get("tool_requests") or []:
        exposed_name = str(request.get("tool_name") or "")
        name = (exposed_to_official or {}).get(exposed_name, exposed_name)
        if not name:
            continue
        try:
            arguments = json.loads(request.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        rendered = ", ".join(
            f"{key}={json.dumps(str(value), ensure_ascii=False)}"
            for key, value in arguments.items()
        )
        actions.append(f"{name}.func({rendered})")
    return actions


def score_record(record: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    source = ensure_source()
    session = WorkBenchSession(source)
    exposed_to_official = {
        re.sub(r"[^a-zA-Z0-9_-]", "_", str(tool["name"])): str(tool["name"])
        for tool in session.tool_specs
    }
    actions = predicted_actions(record, exposed_to_official)
    try:
        result = session.request(
            {
                "operation": "score",
                "predicted_actions": actions,
                "ground_truth_actions": gold.get("actions") or [],
                "error": record.get("error_message") or "",
            }
        )
    finally:
        session.close()
    return {**dict(result), "predicted_actions": actions}


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(_prediction: str, gold, record: Mapping[str, Any]) -> dict:
    expected = gold if isinstance(gold, dict) else {"actions": []}
    return score_record(dict(record), expected)


class WorkBenchBenchmark(Benchmark):
    def create_tools(self, slot: str, query_meta: Mapping[str, Any]) -> ToolBundle:
        if slot != "benchmark:tools":
            return super().create_tools(slot, query_meta)
        tools, session = create_tools(dict(query_meta))
        return ToolBundle(tools=tools, resources=[session])

    def aggregate(
        self,
        samples: Sequence[Mapping[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        details = [sample.get("score_details") or {} for sample in samples]
        count = sum(1 for item in details if bool(item.get("harmful_side_effect")))
        metrics["harmful_side_effect_count"] = count
        metrics["harmful_side_effect_rate"] = round(count / len(samples), 4) if samples else None
        return metrics


BENCHMARK = register_benchmark(
    WorkBenchBenchmark(
        benchmark_id="workbench",
        name="WorkBench Revisited",
        category="workplace_tool_use",
        sources=SOURCES,
        full_prepare_target="workbench",
        prepare_handlers={"workbench": _prepare},
        loaders={"workbench": load_workbench},
        scorer_kinds={"workbench": "workbench"},
        score_handlers={"workbench": _score},
        binary_kinds=("workbench",),
        result_kind="action",
        capabilities={"required": ["text_generation", "tool_calls"]},
        runtime_defaults={"max_new_tokens": 32768, "max_rounds": 1, "max_turns": 1},
        scoring_profiles={
            "workbench": {
                "default_profile": "official",
                "profiles": {
                    "official": {
                        "scorer_id": "workbench",
                        "parameters": {
                            "evaluator": "official_final_state",
                            "reports_harmful_side_effects": True,
                        },
                    }
                },
            }
        },
    )
)
