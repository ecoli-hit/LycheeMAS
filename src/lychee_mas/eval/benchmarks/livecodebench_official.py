"""Thin process boundary around the pinned LiveCodeBench evaluator.

The benchmark adapter owns data and process orchestration, while the official
LiveCodeBench checkout owns code extraction and execution semantics. Keeping
that boundary in a subprocess prevents the evaluator's global multiprocessing
and environment mutations from leaking into the Eval runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def evaluate(payload: dict[str, Any], evaluator_repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(evaluator_repo))
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    rows = list(payload.get("rows") or [])
    samples = [{"input_output": str(row["input_output"])} for row in rows]
    extracted = [extract_code(str(row.get("prediction") or ""), LMStyle.OpenAIChat) for row in rows]
    _, results, metadata = codegen_metrics(
        samples,
        [[code] for code in extracted],
        k_list=[1],
        num_process_evaluate=max(1, int(payload.get("workers") or 1)),
        timeout=max(1, int(payload.get("timeout") or 6)),
        debug=False,
    )
    evaluated = []
    for index, row in enumerate(rows):
        test_results = list(results[index][0])
        raw_metadata = metadata[index][0]
        try:
            evaluator_metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            evaluator_metadata = {"raw": raw_metadata}
        evaluated.append(
            {
                "case_id": str(row["case_id"]),
                "trial_index": int(row.get("trial_index", 0)),
                "question_id": str(row["question_id"]),
                "extracted_code": extracted[index],
                "test_results": test_results,
                "passed": bool(test_results)
                and all(value is True or value == 1 for value in test_results),
                "evaluator_metadata": evaluator_metadata,
            }
        )
    return {"schema_version": 1, "results": evaluated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-repo", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = evaluate(payload, Path(args.evaluator_repo))
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
