#!/usr/bin/env python3
"""Run the AgenTracer adapter on TracerTraj-style records.

Real mode injects a local HF model through LycheeMAS ``HFBackend``.  ``--smoke``
uses each record's gold Who/When label as a deterministic fake tracer response,
so the integration path can be checked without model weights or an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.core.types import Answer, Message, Trajectory  # noqa: E402
from lychee_mas.plugins.postrun import analyze_run  # noqa: E402


def _load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("records", rows.get("data", [rows]))
    else:
        raise ValueError(f"unsupported input format: {path.suffix}; use parquet/json/jsonl")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("input must contain a list of record objects")
    return rows[:limit] if limit else rows


def _history_items(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError("record.history must be a list")
    return [dict(item) for item in value]


def _sender(item: dict[str, Any]) -> str:
    explicit = item.get("name", item.get("sender"))
    if explicit:
        return str(explicit)
    role = str(item.get("role", "unknown"))
    target = re.search(r"->\s*([^,)]+)", role)
    if target:
        return target.group(1).strip()
    return role.split("(", 1)[0].strip() or "unknown"


def _to_trajectory(row: dict[str, Any]) -> Trajectory:
    task_id = str(row.get("question_ID", row.get("task_id", "")))
    prediction = str(row.get("model_prediction", ""))
    trajectory = Trajectory(task_id=task_id, final_answer=Answer(content=prediction))
    for index, item in enumerate(_history_items(row.get("history", []))):
        sender = _sender(item)
        trajectory.add(
            Message(
                sender=sender,
                role=str(item.get("role", "assistant")),
                round=int(item.get("round", 0)),
                content=str(item.get("content", "")),
                meta={"source_step": item.get("step", index)},
            )
        )
    return trajectory


def _smoke_llm(row: dict[str, Any]):
    agent = str(row.get("mistake_agent", ""))
    step = row.get("mistake_step", -1)
    reason = str(row.get("mistake_reason", "smoke response"))
    response = json.dumps(
        {
            "mistake_step": int(step),
            "mistake_agent": agent,
            "reason": reason,
            "suggested_fix": "smoke-only response",
            "confidence": 1.0,
        }
    )
    return lambda _prompt: response


def _hf_llm(model_path: str, device: str, max_new_tokens: int):
    from lychee_mas.backends.hf_backend import HFBackend

    backend = HFBackend(
        model_name=model_path, device=device, enable_thinking=False, strict_hidden=False
    )

    def call(prompt: str) -> str:
        result = backend.generate_chat(
            [{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens
        )
        return result.text

    return call


def _score_output(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    valid = [row for row in rows if row.get("attribution")]
    agent_hits = []
    step_hits = []
    joint_hits = []
    distances = []
    for row in rows:
        pred = (row.get("attribution") or [{}])[0]
        pred_agent = pred.get("agent")
        pred_step = pred.get("step")
        gold_agent = row.get("gold_agent")
        gold_step = row.get("gold_step")
        try:
            gold_step_int = int(gold_step)
            pred_step_int = int(pred_step)
        except (TypeError, ValueError):
            gold_step_int = pred_step_int = None
        agent_ok = pred_agent == gold_agent
        step_ok = pred_step_int is not None and pred_step_int == gold_step_int
        agent_hits.append(agent_ok)
        step_hits.append(step_ok)
        joint_hits.append(agent_ok and step_ok)
        if pred_step_int is not None and gold_step_int is not None:
            distances.append(abs(pred_step_int - gold_step_int))
    n = len(rows)
    return {
        "count": n,
        "valid_count": len(valid),
        "agent_exact": sum(agent_hits) / n if n else 0.0,
        "step_exact": sum(step_hits) / n if n else 0.0,
        "joint_exact": sum(joint_hits) / n if n else 0.0,
        "step_distance": sum(distances) / len(distances) if distances else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TracerTraj parquet/json/jsonl file")
    parser.add_argument("--output", required=True, help="prediction JSONL output")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="use gold labels as a fake tracer")
    parser.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--on-error",
        choices=("record", "raise"),
        default="record",
        help="record a per-row error and continue, or abort on the first invalid diagnosis",
    )
    parser.add_argument("--metrics-output", default=None, help="metrics JSON path")
    args = parser.parse_args()

    records = _load_records(Path(args.input), args.limit or None)
    if not records:
        raise SystemExit("input contains no records")
    if args.smoke:
        llm_factory = _smoke_llm
    else:
        if not args.model_path:
            raise SystemExit("real mode requires --model-path or LYCHEE_HF_MODEL")
        tracer_llm = _hf_llm(args.model_path, args.device, args.max_new_tokens)

        def llm_factory(_row: dict[str, Any]):
            return tracer_llm

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            trajectory = _to_trajectory(row)
            try:
                attribution = analyze_run(
                    trajectory,
                    method="agentracer",
                    tracer_llm=llm_factory(row),
                    credit_assigner=None,
                    context={
                        "question": row.get("question", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "environment_feedback": row.get("environment_feedback", ""),
                    },
                )
                error = None
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                attribution = []
                error = f"{type(exc).__name__}: {exc}"
            result = {
                "task_id": trajectory.task_id,
                "attribution": trajectory.meta.get("attribution", []),
                "gold_agent": row.get("mistake_agent"),
                "gold_step": row.get("mistake_step"),
                "credits": attribution,
            }
            if error is not None:
                result["error"] = error
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    metrics_path = (
        Path(args.metrics_output) if args.metrics_output else output.with_suffix(".metrics.json")
    )
    metrics_path.write_text(
        json.dumps(_score_output(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_score_output(results), ensure_ascii=False))


if __name__ == "__main__":
    main()
