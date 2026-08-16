"""BIG-Bench Extra Hard loader and official deterministic scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .base import Benchmark, resolve_provider_ids
from .common import (
    clone_git_source,
    copy_raw_to_prepared,
    prepared_benchmark_dir,
    raw_source_dir,
    remove_path,
)
from .registry import register_benchmark

SOURCES = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {"env": None, "default_ids": []},
    "github": {
        "env": None,
        "default_ids": ["https://github.com/google-deepmind/bbeh.git"],
        "revision": "80d12ca916b7158f22293fcf3144f4d3d854d4be",
        "source_type": "git",
        "purpose": "official BBEH tasks and deterministic evaluator",
    },
    "other_defaults": [],
    "fallback_files": [],
}

OFFICIAL_REPO = resolve_provider_ids(SOURCES, "github")[0]
OFFICIAL_REVISION = str(SOURCES["github"]["revision"])


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("bbeh")


def _raw_repo() -> Path:
    return raw_source_dir("bbeh", "github", "google-deepmind/bbeh")


def _ready(path: Path) -> bool:
    tasks = path / "benchmark_tasks"
    return tasks.is_dir() and len(list(tasks.glob("*/task.json"))) == 23


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "github"}:
        raise ValueError("BBEH is published through its official GitHub repository")
    prepared = source_dir(root)
    if _ready(prepared) and not force_download:
        return prepared
    raw = clone_git_source(
        "bbeh",
        OFFICIAL_REPO,
        _raw_repo(),
        revision=OFFICIAL_REVISION,
        force=force_download,
    )
    remove_path(prepared)
    copy_raw_to_prepared(raw / "bbeh", prepared)
    if not _ready(prepared):
        raise RuntimeError("official BBEH source is incomplete: expected 23 task.json files")
    return prepared


def load_source_records(root: str | Path | None = None) -> list[dict[str, Any]]:
    prepared = ensure_source(root)
    rows: list[dict[str, Any]] = []
    for path in sorted((prepared / "benchmark_tasks").glob("*/task.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_name = path.parent.name
        for index, example in enumerate(payload.get("examples") or []):
            rows.append(
                {
                    "task_name": task_name,
                    "example_index": index,
                    "input": str(example["input"]),
                    "target": str(example["target"]),
                }
            )
    return rows


def load_bbeh(n: Optional[int] = None) -> list[dict[str, Any]]:
    records = []
    for row in load_source_records():
        task_id = f"{row['task_name']}/{row['example_index']}"
        records.append(
            {
                "task": "bbeh",
                "kind": "bbeh",
                "question": row["input"],
                "gold": row["target"],
                "context": None,
                "metadata": {
                    "task_id": task_id,
                    "bbeh_task": row["task_name"],
                    "example_index": row["example_index"],
                    "official_revision": OFFICIAL_REVISION,
                },
            }
        )
        if n is not None and len(records) >= n:
            break
    return records


def _strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    for marker in ("boxed{", "text{", "texttt{"):
        if marker in response and response.endswith("}"):
            response = response[:-1].split(marker)[-1]
    return response


def extract_answer(sample: str) -> str:
    answer = sample
    for prefix in (
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return _strip_latex(answer)


def _fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return bool(prediction.endswith("?") and prediction[:-1] == reference)


def evaluate_correctness(sample: str, reference: str) -> bool:
    """Mirror ``bbeh/evaluate.py`` from the pinned official revision."""

    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[:-1] if prediction.endswith(".") else prediction
    normalized_reference = reference.strip().lower().replace(", ", ",")
    return _fuzzy_match(prediction, normalized_reference)


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(prediction: str, gold, _record) -> dict:
    return {"score": 1.0 if evaluate_correctness(prediction, str(gold)) else 0.0}


BENCHMARK = register_benchmark(
    Benchmark(
        benchmark_id="bbeh",
        name="BIG-Bench Extra Hard",
        category="reasoning",
        sources=SOURCES,
        full_prepare_target="bbeh",
        prepare_handlers={"bbeh": _prepare},
        loaders={"bbeh": load_bbeh},
        scorer_kinds={"bbeh": "bbeh"},
        score_handlers={"bbeh": _score},
        binary_kinds=("bbeh",),
        runtime_defaults={"max_new_tokens": 32768, "max_turns": 1},
        scoring_profiles={
            "bbeh": {
                "default_profile": "official",
                "profiles": {
                    "official": {
                        "scorer_id": "bbeh",
                        "parameters": {"evaluator": "pinned_official_deterministic"},
                    }
                },
            }
        },
    )
)
