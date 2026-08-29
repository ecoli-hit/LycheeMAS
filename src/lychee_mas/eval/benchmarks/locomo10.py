"""LoCoMo-style memory benchmark loaders."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Benchmark, resolve_provider_ids, resolve_source_backend_order
from .common import (
    download_url,
    hf_resolve_url,
    json_file_ready,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
)
from .registry import register_benchmark

SOURCES: dict[str, Any] = {
    "modelscope": {
        "env": "LYCHEE_LOCOMO10_MODELSCOPE_ID",
        "default_ids": ["evalscope/locomo"],
    },
    "huggingface": {"env": None, "default_ids": ["Percena/locomo-mc10"]},
    "github": {
        "env": None,
        "default_ids": [
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        ],
        "source_type": "raw_file",
        "purpose": "official LoCoMo raw JSON fallback",
    },
    "other_defaults": [],
    "fallback_files": [
        {
            "provider": "huggingface_direct",
            "repo_id": "Percena/locomo-mc10",
            "files": ["raw/locomo10.json"],
            "strict": True,
            "purpose": "single-file source used by the LoCoMo10 loader",
        }
    ],
}

_FALLBACK = SOURCES["fallback_files"][0]
REPO_ID = resolve_provider_ids(SOURCES, "huggingface")[0]
RAW_FILE = _FALLBACK["files"][0]
GITHUB_URL = resolve_provider_ids(SOURCES, "github")[0]


def _target_file() -> str:
    return str(prepared_benchmark_dir("locomo10") / "locomo10.json")


def _download_huggingface() -> str:
    src = raw_source_dir("locomo10", "huggingface", REPO_ID)
    raw_file = src / RAW_FILE
    log_download_source("locomo10", "huggingface", REPO_ID, src)
    download_url(hf_resolve_url(REPO_ID, RAW_FILE), str(raw_file))
    os.makedirs(os.path.dirname(_target_file()), exist_ok=True)
    shutil.copyfile(raw_file, _target_file())
    return _target_file()


def _publish_raw_json(provider: str, identifier: str, raw_file: Path) -> str | None:
    if not json_file_ready(raw_file):
        return None
    os.makedirs(os.path.dirname(_target_file()), exist_ok=True)
    shutil.copyfile(raw_file, _target_file())
    print(
        f"[prepare] locomo10: restored prepared from raw provider={provider} "
        f"id={identifier} -> {_target_file()}",
        flush=True,
    )
    return _target_file()


def _download_modelscope() -> str:
    dataset_ids = resolve_provider_ids(SOURCES, "modelscope")
    if not any(dataset_ids):
        raise RuntimeError(
            "no known ModelScope mirror for LoCoMo10; set LYCHEE_LOCOMO10_MODELSCOPE_ID "
            "or use source=huggingface"
        )
    errors: list[str] = []
    for dataset_id in [x for x in dataset_ids if x]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            src = raw_source_dir("locomo10", "modelscope", dataset_id)
            log_download_source("locomo10", "modelscope", dataset_id, src)
            snapshot_download(
                dataset_id,
                repo_type="dataset",
                local_dir=str(src),
                allow_patterns=[RAW_FILE, "locomo10.json", "raw/*.json"],
            )
            candidates = [
                src / RAW_FILE,
                src / "locomo10.json",
                *sorted(src.glob("**/locomo10.json")),
            ]
            for candidate in candidates:
                if candidate.is_file():
                    os.makedirs(os.path.dirname(_target_file()), exist_ok=True)
                    shutil.copyfile(candidate, _target_file())
                    return _target_file()
            raise FileNotFoundError(f"locomo10.json not found under {src}")
        except Exception as exc:  # pragma: no cover - provider/network dependent
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("ModelScope LoCoMo download failed; " + " | ".join(errors))


def _download_github_raw() -> str:
    src = raw_source_dir("locomo10", "github", GITHUB_URL)
    raw_file = src / "locomo10.json"
    log_download_source("locomo10", "github", GITHUB_URL, src)
    download_url(GITHUB_URL, str(raw_file))
    os.makedirs(os.path.dirname(_target_file()), exist_ok=True)
    shutil.copyfile(raw_file, _target_file())
    return _target_file()


def prepare_locomo10(force: bool = False, source: Optional[str] = None) -> str:
    dest = _target_file()
    if not force and json_file_ready(dest):
        return dest
    if not force:
        restored = _restore_from_raw(source)
        if restored:
            return restored
    if force and os.path.isfile(dest):
        os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    errors: list[str] = []
    for backend in _backend_order(source):
        try:
            if backend == "modelscope":
                return _download_modelscope()
            if backend == "huggingface":
                return _download_huggingface()
            if backend == "github":
                return _download_github_raw()
            raise ValueError(f"unknown LoCoMo10 backend {backend!r}")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError("Failed to prepare LoCoMo10; " + " | ".join(errors))


def _backend_order(source: Optional[str]) -> list[str]:
    return resolve_source_backend_order("locomo10", SOURCES, source)


def _restore_from_raw(source: Optional[str]) -> str | None:
    for backend in _backend_order(source):
        if backend == "modelscope":
            for dataset_id in resolve_provider_ids(SOURCES, "modelscope"):
                src = raw_source_dir("locomo10", "modelscope", dataset_id)
                candidates = [
                    src / RAW_FILE,
                    src / "locomo10.json",
                    *sorted(src.glob("**/locomo10.json")),
                ]
                for candidate in candidates:
                    restored = _publish_raw_json("modelscope", dataset_id, candidate)
                    if restored:
                        return restored
        elif backend == "huggingface":
            restored = _publish_raw_json(
                "huggingface",
                REPO_ID,
                raw_source_dir("locomo10", "huggingface", REPO_ID) / RAW_FILE,
            )
            if restored:
                return restored
        elif backend == "github":
            restored = _publish_raw_json(
                "github",
                GITHUB_URL,
                raw_source_dir("locomo10", "github", GITHUB_URL) / "locomo10.json",
            )
            if restored:
                return restored
    return None


def _turn_text(turn) -> str:
    if isinstance(turn, dict):
        role = turn.get("role") or turn.get("speaker") or turn.get("name") or "speaker"
        content = turn.get("content") or turn.get("text") or turn.get("message") or ""
        return f"{role}: {content}"
    return str(turn)


def _session_text(session) -> list[str]:
    if isinstance(session, list):
        return [_turn_text(turn) for turn in session]
    return [str(session)]


def _session_number(name: str) -> int:
    match = re.search(r"session_(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def _conversation_sessions(conv: dict) -> list[tuple[str, str, object]]:
    if {"sessions_ids", "sessions_dates", "sessions"}.issubset(conv):
        return list(zip(conv["sessions_ids"], conv["sessions_dates"], conv["sessions"]))

    conversation = conv.get("conversation")
    if isinstance(conversation, dict):
        session_keys = sorted(
            [key for key in conversation if re.fullmatch(r"session_\d+", str(key))],
            key=_session_number,
        )
        return [
            (
                key,
                str(conversation.get(f"{key}_date_time") or ""),
                conversation[key],
            )
            for key in session_keys
        ]
    return []


def _load_original_conversation_records(
    data: list[dict], n: Optional[int], max_qa_per_conv: int
) -> List[Dict]:
    out = []
    for conv in data:
        parts = []
        for sid, date, sess in _conversation_sessions(conv):
            title = f"=== {sid}"
            if date:
                title += f" ({date})"
            title += " ==="
            parts.append(title)
            parts.extend(_session_text(sess))
        history = "\n".join(parts)
        for qa in conv["qa"][:max_qa_per_conv]:
            ans = qa.get("answer")
            if ans is None:
                continue
            out.append(
                {
                    "task": "locomo10",
                    "kind": "f1",
                    "question": qa["question"].strip() + "\nAnswer concisely.",
                    "gold": [str(ans)],
                    "context": history,
                    "metadata": {
                        "sample_id": conv.get("sample_id"),
                        "category": qa.get("category"),
                        "evidence": qa.get("evidence"),
                    },
                }
            )
            if n and len(out) >= n:
                return out
    return out


def _load_mc10_records(data: list[dict], n: Optional[int]) -> List[Dict]:
    out = []
    for row in data:
        sessions = row.get("haystack_sessions") or row.get("sessions") or []
        session_ids = row.get("haystack_session_ids") or row.get("sessions_ids") or []
        session_dates = row.get("haystack_session_datetimes") or row.get("sessions_dates") or []
        parts = []
        for idx, sess in enumerate(sessions):
            sid = session_ids[idx] if idx < len(session_ids) else f"session_{idx + 1}"
            date = session_dates[idx] if idx < len(session_dates) else ""
            title = f"=== {sid}"
            if date:
                title += f" ({date})"
            title += " ==="
            parts.append(title)
            parts.extend(_session_text(sess))
        history = "\n".join(parts)
        answer = row.get("answer")
        question = row.get("question")
        if answer is None or question is None:
            continue
        out.append(
            {
                "task": "locomo10",
                "kind": "f1",
                "question": str(question).strip() + "\nAnswer concisely.",
                "gold": [str(answer)],
                "context": history,
                "metadata": {
                    "question_id": row.get("question_id"),
                    "question_type": row.get("question_type"),
                    "num_sessions": row.get("num_sessions"),
                },
            }
        )
        if n and len(out) >= n:
            return out
    return out


def load_locomo10(n: Optional[int] = None, max_qa_per_conv: int = 10) -> List[Dict]:
    """Flatten LoCoMo: each record is one question plus full dialogue history."""
    prepare_locomo10()
    with open(_target_file(), encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("locomo10.json must contain a list")
    if data and isinstance(data[0], dict) and "qa" in data[0]:
        return _load_original_conversation_records(data, n, max_qa_per_conv)
    return _load_mc10_records(data, n)


def _score(prediction: str, gold, _record) -> dict:
    from ..evaluation.metrics import score_f1

    references = [str(item) for item in gold] if isinstance(gold, list) else [str(gold)]
    return {"score": score_f1(prediction, references)}


BENCHMARK = register_benchmark(
    Benchmark(
        benchmark_id="locomo10",
        name="LoCoMo10",
        category="memory",
        sources=SOURCES,
        full_prepare_target="locomo10",
        prepare_handlers={"locomo10": prepare_locomo10},
        loaders={"locomo10": load_locomo10},
        scorer_kinds={"locomo10": "f1"},
        score_handlers={"f1": _score},
        capabilities={"required": ["text_generation", "long_context"]},
    )
)
