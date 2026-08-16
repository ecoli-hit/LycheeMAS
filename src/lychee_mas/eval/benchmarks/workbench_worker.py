"""Persistent JSON-line bridge to one pinned WorkBench checkout."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _reply(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))

    from src.evals.evaluation import has_side_effects, is_correct
    from src.tools.state import reset_state
    from src.tools.toolkits import all_tools

    tools = {tool.name: tool for tool in all_tools}
    reset_state()
    _reply(
        {
            "status": "ready",
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": tool.args_schema,
                }
                for tool in all_tools
            ],
        }
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation")
            if operation == "call":
                name = str(request["name"])
                kwargs = {
                    str(key): str(value)
                    for key, value in request.get("arguments", {}).items()
                }
                result = tools[name](**kwargs)
                _reply({"status": "ok", "result": _jsonable(result)})
            elif operation == "score":
                predicted = [str(item) for item in request.get("predicted_actions") or []]
                gold = [str(item) for item in request.get("ground_truth_actions") or []]
                error = str(request.get("error") or "")
                correct = bool(is_correct(predicted, gold, error))
                side_effects = bool(has_side_effects(predicted, correct))
                _reply(
                    {
                        "status": "ok",
                        "result": {
                            "score": 1.0 if correct else 0.0,
                            "correct": correct,
                            "harmful_side_effect": side_effects,
                        },
                    }
                )
            elif operation == "close":
                _reply({"status": "ok"})
                return
            else:
                raise ValueError(f"unknown operation {operation!r}")
        except Exception as exc:
            _reply(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )


if __name__ == "__main__":
    main()
