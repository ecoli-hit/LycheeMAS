"""HumanEval loader + official-style scoring.

The shape mirrors the existing gsm8k/aime loaders: load lazily, cache under
the benchmark raw root, and return records with
``{task, kind, question, gold, context}``.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

from .base import Benchmark, resolve_provider_ids, resolve_source_backend_order
from .common import (
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
    restore_prepared_from_raw,
)
from .registry import register_benchmark

SOURCES = {
    "modelscope": {
        "env": "LYCHEE_HUMANEVAL_MODELSCOPE_ID",
        "default_ids": ["modelscope/humaneval", "opencompass/humaneval"],
    },
    "huggingface": {
        "env": "LYCHEE_HUMANEVAL_HF_ID",
        "default_ids": ["openai/openai_humaneval"],
    },
    "github": {
        "env": None,
        "default_ids": ["https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"],
        "source_type": "raw_file",
        "purpose": "official raw archive fallback",
    },
    "other_defaults": [],
    "fallback_files": [],
}

URL = resolve_provider_ids(SOURCES, "github")[0]
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("LYCHEEMAS_EVAL_TIMEOUT", "20"))

_RUNNER = r"""
from pathlib import Path
import json
import sys
import traceback

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

try:
    namespace = {}
    exec(payload["test"], namespace)
    exec(payload["answer"], namespace)
    check = namespace["check"]
    candidate = namespace[payload["entry_point"]]
    check(candidate)
    print(json.dumps({"success": True, "error": None}))
except Exception:
    print(json.dumps({"success": False, "error": traceback.format_exc()}))
    sys.exit(1)
"""


def source_archive(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else prepared_benchmark_dir("human_eval")
    return base / "HumanEval.jsonl.gz"


def _raw_archive(provider: str, identifier: str) -> Path:
    return raw_source_dir("human_eval", provider, identifier) / "HumanEval.jsonl.gz"


def _publish_archive(src: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, archive)
    return archive


def safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "_")


def _write_records_archive(archive: Path, records: list[dict[str, Any]]) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    with gzip.open(archive, "wb") as fh:
        fh.write(payload.encode("utf-8"))
    return archive


def _archive_ready(archive: Path) -> bool:
    if not archive.is_file() or archive.stat().st_size <= 0:
        return False
    required = {"task_id", "prompt", "test", "entry_point"}
    try:
        with gzip.open(archive, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                return required.issubset(row)
    except Exception:
        return False
    return False


def _records_from_dataset(dataset) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    required = {"task_id", "prompt", "test", "entry_point"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"HumanEval dataset must contain {sorted(required)}; "
            f"got {list(rows[0]) if rows else []}"
        )
    return rows


def _download_from_huggingface(archive: Path) -> Path:
    from datasets import load_dataset

    dataset_id = resolve_provider_ids(SOURCES, "huggingface")[0]
    raw_archive = _raw_archive("huggingface", dataset_id)
    log_download_source("human_eval", "huggingface", dataset_id, raw_archive)
    _write_records_archive(
        raw_archive, _records_from_dataset(load_dataset(dataset_id, split="test"))
    )
    return _publish_archive(raw_archive, archive)


def _download_from_modelscope(archive: Path) -> Path:
    from datasets import Dataset
    from modelscope.msdatasets import MsDataset

    dataset_ids = resolve_provider_ids(SOURCES, "modelscope")
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            raw_archive = _raw_archive("modelscope", dataset_id)
            log_download_source("human_eval", "modelscope", dataset_id, raw_archive)
            ms_dataset = MsDataset.load(dataset_id, split="test")
            if hasattr(ms_dataset, "to_hf_dataset"):
                dataset = ms_dataset.to_hf_dataset()
            elif hasattr(ms_dataset, "to_list"):
                dataset = Dataset.from_list(ms_dataset.to_list())
            else:
                dataset = Dataset.from_list(list(ms_dataset))
            _write_records_archive(raw_archive, _records_from_dataset(dataset))
            return _publish_archive(raw_archive, archive)
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope HumanEval download failed; " + " | ".join(errors))


def _download_from_github(archive: Path) -> Path:
    raw_archive = _raw_archive("github", URL)
    log_download_source("human_eval", "github", URL, raw_archive)
    raw_archive.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "LycheeMAS/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw_archive.write_bytes(response.read())
    return _publish_archive(raw_archive, archive)


def _backend_order(source: str | None) -> list[str]:
    return resolve_source_backend_order("human_eval", SOURCES, source)


def _raw_archive_candidates(source: str | None) -> list[tuple[str, str, Path]]:
    candidates: list[tuple[str, str, Path]] = []
    for backend in _backend_order(source):
        if backend == "modelscope":
            for dataset_id in resolve_provider_ids(SOURCES, "modelscope"):
                candidates.append(
                    ("modelscope", dataset_id, _raw_archive("modelscope", dataset_id))
                )
        elif backend == "huggingface":
            for dataset_id in resolve_provider_ids(SOURCES, "huggingface"):
                candidates.append(
                    ("huggingface", dataset_id, _raw_archive("huggingface", dataset_id))
                )
        elif backend == "github":
            candidates.append(("github", URL, _raw_archive("github", URL)))
    return candidates


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    archive = source_archive(root)
    if _archive_ready(archive) and not force_download:
        return archive
    if not force_download:
        restored = restore_prepared_from_raw(
            "human_eval",
            archive,
            _raw_archive_candidates(source),
            ready=_archive_ready,
        )
        if restored is not None:
            return restored
    if force_download and archive.exists():
        archive.unlink()
    errors: list[str] = []
    for backend in _backend_order(source):
        try:
            if backend == "modelscope":
                return _download_from_modelscope(archive)
            if backend == "huggingface":
                return _download_from_huggingface(archive)
            return _download_from_github(archive)
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare HumanEval; " + " | ".join(errors))


def load_source_records(
    root: str | Path | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    archive = ensure_source(root, force_download=force_download)
    records: list[dict[str, Any]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(archive.read_bytes())) as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def load_human_eval(n: Optional[int] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in load_source_records():
        prompt = task["prompt"].strip()
        out.append(
            {
                "task": "human_eval",
                "kind": "human_eval",
                "question": (
                    "Complete the following Python function. Return only the complete "
                    "function implementation.\n\n"
                    f"```python\n{prompt}\n```"
                ),
                "gold": {
                    "task_id": task["task_id"],
                    "entry_point": task["entry_point"],
                    "test": task["test"],
                },
                "context": None,
                "metadata": {
                    "task_id": task["task_id"],
                    "safe_task_id": safe_task_id(task["task_id"]),
                },
            }
        )
        if n and len(out) >= n:
            break
    return out


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()


def evaluate_code(answer_code: str, test_code: str, entry_point: str) -> dict[str, Any]:
    try:
        namespace: dict[str, Any] = {}
        exec(test_code, namespace)
        exec(answer_code, namespace)
        check = namespace["check"]
        candidate = namespace[entry_point]
        check(candidate)
        return {"success": True, "error": None}
    except Exception:
        return {"success": False, "error": traceback.format_exc()}


def evaluate_answer(
    answer_text: str,
    gold: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = {
        "answer": extract_python_code(answer_text),
        "test": gold["test"],
        "entry_point": gold["entry_point"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _RUNNER, str(payload_path)],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "error": f"Evaluation timed out after {exc.timeout} seconds.",
                "returncode": None,
                "stderr": exc.stderr,
            }

    eval_result = parse_eval_output(result.stdout)
    eval_result["returncode"] = result.returncode
    eval_result["stderr"] = result.stderr
    return eval_result


def parse_eval_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "success" in value:
            return value
    return {"success": False, "error": "No JSON evaluation result was produced."}


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(prediction: str, gold, _record) -> dict:
    if not isinstance(gold, dict) or "test" not in gold or "entry_point" not in gold:
        return {"score": 0.0, "execution_success": False, "execution_error": "invalid gold"}
    result = evaluate_answer(prediction, gold)
    return {
        "score": 1.0 if result.get("success") else 0.0,
        "execution_success": bool(result.get("success")),
        "execution_error": result.get("error"),
    }


def _message_content(message: Any) -> str:
    value = (
        message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    )
    return value if isinstance(value, str) else str(value)


def _message_source(message: Any) -> str:
    value = (
        message.get("source", "") if isinstance(message, dict) else getattr(message, "source", "")
    )
    return str(value)


class HumanEvalBenchmark(Benchmark):
    def collect_prediction(
        self,
        *,
        query: Any,
        messages: Sequence[Any],
        workspace: Path | None,
        default_text: str,
    ) -> str:
        del query, workspace
        for message in reversed(messages):
            text = _message_content(message)
            if _message_source(message) == "Coder" and "```" in text:
                return text
        return default_text


BENCHMARK = register_benchmark(
    HumanEvalBenchmark(
        benchmark_id="human_eval",
        name="HumanEval",
        category="coding",
        sources=SOURCES,
        full_prepare_target="human_eval",
        prepare_handlers={"human_eval": _prepare},
        loaders={"human_eval": load_human_eval},
        scorer_kinds={"human_eval": "human_eval"},
        score_handlers={"human_eval": _score},
        binary_kinds=("human_eval",),
        capabilities={"required": ["code_generation", "code_execution"]},
        runtime_defaults={
            "code_executor": "docker",
            "code_timeout": 60,
            "docker_image": "lychee-human-eval:local",
            "max_turns": 12,
        },
    )
)
