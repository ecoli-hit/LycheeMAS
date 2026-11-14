#!/usr/bin/env python3
"""Model-simulated counterfactual replay for Who&When records.

This is not a real browser/terminal replay: one model call proposes a local
replacement and simulates all downstream messages plus a final answer.
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
sys.path.insert(0, str(ROOT / "scripts"))
from lychee_mas.core.types import Answer, Message, Trajectory  # noqa: E402
from lychee_mas.methods.postrun.agentracer import format_trajectory  # noqa: E402
from run_agentracer_attribution import _history_items, _load_records, _sender  # noqa: E402


def _trajectory(row: dict[str, Any]) -> Trajectory:
    task_id = str(row.get("question_ID", row.get("task_id", "")))
    traj = Trajectory(
        task_id=task_id, final_answer=Answer(content=str(row.get("model_prediction", "")))
    )
    for index, item in enumerate(_history_items(row.get("history", []))):
        traj.add(
            Message(
                sender=_sender(item),
                role=str(item.get("role", "assistant")),
                round=0,
                content=str(item.get("content", "")),
                meta={"source_step": index},
            )
        )
    return traj


def _norm(value: Any) -> str:
    text = str(value).strip().lower().replace("```", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w.%-]+", "", text)
    return text


def _matches(pred: Any, gold: Any) -> bool:
    p, g = _norm(pred), _norm(gold)
    return bool(p) and (p == g or g in p)


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return (
        text
        if len(text) <= limit
        else text[: limit // 2] + "\n...[clipped]...\n" + text[-limit // 2 :]
    )


def _prompt(row: dict[str, Any], traj: Trajectory, index: int, context_chars: int) -> str:
    history = format_trajectory(traj)
    if len(history) > context_chars:
        history = (
            history[: context_chars // 2]
            + "\n...[middle omitted]...\n"
            + history[-context_chars // 2 :]
        )
    step = traj.messages[index]
    step_content = _clip(step.content, 6000)
    agents = ", ".join(f"{i}:{m.sender}" for i, m in enumerate(traj.messages))
    contract = (
        '{{"replacement":"corrected output for step {index}", '
        '"replayed_steps":[{{"step":{index_plus},"agent":"...","output":"..."}}], '
        '"final_answer":"simulated final answer", "reason":"local repair", '
        '"confidence":0.0}}'
    ).format(index=index, index_plus=index + 1)
    return f"""You are running a MODEL-SIMULATED counterfactual replay, not real tools.
Task: {row.get("question", "")}
Target candidate step: {index}; target agent: {step.sender}
Candidate step input/output:
{step_content}

Agent index table:
{agents}

Observed trajectory (may be truncated):
{history}

Replace only candidate step {index}, then simulate every later agent.\n{contract}\nDo not claim that a browser, terminal, file, or API was actually executed."""  # noqa: E501


def _parse(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("simulated replay output has no JSON object")
    data = json.loads(raw[start : end + 1])
    if not str(data.get("replacement", "")).strip():
        raise ValueError("simulated replay replacement is empty")
    if "final_answer" not in data:
        raise ValueError("simulated replay output needs final_answer")
    return data


def _smoke(row: dict[str, Any], index: int) -> str:
    gold_step = int(row.get("mistake_step", -1))
    final = (
        row.get("ground_truth", row.get("groundtruth", "")) if index == gold_step else "__wrong__"
    )
    return json.dumps(
        {
            "replacement": f"SMOKE_REPAIR_STEP_{index}",
            "replayed_steps": [],
            "final_answer": final,
            "reason": "smoke-only gold-gated simulation",
            "confidence": 1.0,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=12000)
    ap.add_argument("--context-chars", type=int, default=12000)
    args = ap.parse_args()
    rows = _load_records(Path(args.input), args.limit or None)
    if not rows:
        raise SystemExit("input contains no records")
    backend = None
    if not args.smoke:
        if not args.model_path:
            raise SystemExit("real mode requires --model-path")
        from lychee_mas.backends.hf_backend import HFBackend

        backend = HFBackend(
            model_name=args.model_path,
            device=args.device,
            enable_thinking=False,
            max_input_tokens=args.max_input_tokens,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            traj = _trajectory(row)
            gold = row.get("ground_truth", row.get("groundtruth", ""))
            tried, calls, prompt_chars, response_chars = [], 0, 0, 0
            found = None
            attempts = []
            for index, step in enumerate(traj.messages):
                tried.append(index)
                prompt = _prompt(row, traj, index, args.context_chars)
                prompt_chars += len(prompt)
                raw = (
                    _smoke(row, index)
                    if args.smoke
                    else backend.generate_chat(
                        [{"role": "user", "content": prompt}], max_new_tokens=args.max_new_tokens
                    ).text
                )
                calls += 1
                response_chars += len(raw)
                try:
                    proposal = _parse(raw)
                except Exception as exc:
                    attempts.append({"step": index, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                matched = _matches(proposal["final_answer"], gold)
                attempts.append(
                    {"step": index, "final_answer": proposal["final_answer"], "matched": matched}
                )
                if matched:
                    found = {
                        "first_effective_step": index,
                        "responsible_agent": step.sender,
                        "repair_reason": str(proposal.get("reason", "")),
                        "replacement": proposal["replacement"],
                        "replayed_steps": proposal.get("replayed_steps", []),
                        "simulated_final_answer": proposal["final_answer"],
                        "confidence": proposal.get("confidence"),
                        "model_calls": calls,
                        "tried_steps": tried,
                    }
                    break
            result = {
                "replay_mode": "model_simulated",
                "task_id": traj.task_id,
                "gold_answer": gold,
                "baseline_answer": traj.final_answer.content if traj.final_answer else "",
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "model_calls": calls,
                "attribution": found,
                "attempts": attempts,
                "error": None,
            }
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    valid = [x for x in results if x["attribution"]]
    metrics = {
        "replay_mode": "model_simulated",
        "count": len(results),
        "localized_count": len(valid),
        "localization_rate": len(valid) / len(results) if results else 0.0,
        "mean_model_calls": sum(x["model_calls"] for x in results) / len(results)
        if results
        else 0.0,
        "mean_prompt_chars": sum(x["prompt_chars"] for x in results) / len(results)
        if results
        else 0.0,
        "mean_response_chars": sum(x["response_chars"] for x in results) / len(results)
        if results
        else 0.0,
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
