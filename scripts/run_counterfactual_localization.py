#!/usr/bin/env python3
"""Evaluate counterfactual first-error localization on hand-verified arithmetic traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lychee_mas.methods.postrun.counterfactual import (  # noqa: E402
    ArithmeticReplayTask,
    locate_first_effective_fix,
)


def load_rows(path: Path, limit: int | None):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("input must contain a list of records")
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", default=str(ROOT / "data/counterfactual/hand_verified_arithmetic.json")
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="use the deterministic oracle proposer")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()
    task = ArithmeticReplayTask()
    rows = load_rows(Path(args.input), args.limit or None)
    if not rows:
        raise SystemExit("input contains no samples")
    if args.smoke:

        def proposer_factory(t):
            return lambda _prompt, index, _trace: task.oracle_repair(t, index)
    else:
        if not args.model_path:
            raise SystemExit("real mode requires --model-path or LYCHEE_HF_MODEL")
        from lychee_mas.backends.hf_backend import HFBackend

        backend = HFBackend(
            model_name=args.model_path,
            device=args.device,
            enable_thinking=False,
            strict_hidden=False,
        )

        def proposer_factory(_t):
            def propose(prompt, _index, _trace):
                return backend.generate_chat(
                    [{"role": "user", "content": prompt}], max_new_tokens=args.max_new_tokens
                ).text

            return propose

    results = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            t = task.make_task(str(row["task_id"]), row["a"], row["b"], row["c"])
            failed = task.run(t, {int(row["failure_step"]): "CORRUPTED_OUTPUT: 999"})
            try:
                result = locate_first_effective_fix(task, failed, proposer_factory(t))
                item = result.to_dict()
                item["gold_step"] = int(row["failure_step"])
                item["gold_agent"] = task.agents[int(row["failure_step"])]
                item["step_exact"] = result.first_effective_step == item["gold_step"]
                item["agent_exact"] = result.responsible_agent == item["gold_agent"]
                item["joint_exact"] = item["step_exact"] and item["agent_exact"]
                item["error"] = None
            except Exception as exc:
                item = {
                    "task_id": t["task_id"],
                    "gold_step": int(row["failure_step"]),
                    "gold_agent": task.agents[int(row["failure_step"])],
                    "step_exact": False,
                    "agent_exact": False,
                    "joint_exact": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            results.append(item)
    valid = [x for x in results if not x.get("error")]
    metrics = {
        "count": len(results),
        "valid_count": len(valid),
        "step_exact": sum(x["step_exact"] for x in valid) / len(valid) if valid else 0.0,
        "agent_exact": sum(x["agent_exact"] for x in valid) / len(valid) if valid else 0.0,
        "joint_exact": sum(x["joint_exact"] for x in valid) / len(valid) if valid else 0.0,
        "mean_model_calls": sum(x.get("model_calls", 0) for x in valid) / len(valid)
        if valid
        else 0.0,
        "mean_prompt_chars": sum(x.get("prompt_chars", 0) for x in valid) / len(valid)
        if valid
        else 0.0,
        "mean_proposer_chars": sum(x.get("proposer_chars", 0) for x in valid) / len(valid)
        if valid
        else 0.0,
    }
    mp = out.with_suffix(".metrics.json")
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
