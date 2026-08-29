"""Humanity's Last Exam data adapter and official judge contract."""

# ruff: noqa: E501 - the judge prompt is copied verbatim from the official repository.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from .base import Benchmark, BenchmarkEvaluationError, EvaluationContext, resolve_provider_ids
from .common import (
    conversion_only,
    load_local_dataset_split,
    log_download_source,
    prepared_benchmark_dir,
    raw_source_dir,
    raw_source_matches,
    remove_path,
)
from .registry import register_benchmark

SOURCES: dict[str, Any] = {
    "modelscope": {"env": None, "default_ids": []},
    "huggingface": {
        "env": "LYCHEE_HLE_HF_ID",
        "default_ids": ["cais/hle"],
        "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
    },
    "github": {"env": None, "default_ids": []},
    "other_defaults": [
        {
            "provider": "github",
            "id": "https://github.com/centerforaisafety/hle",
            "revision": "73ae974b1844c3ffa64c3f4343d9f1f259575700",
            "selectable": False,
            "purpose": "official prompts, judge contract, and calibration metric reference",
        }
    ],
    "fallback_files": [],
}

DATASET_ID = str(SOURCES["huggingface"]["default_ids"][0])
DATASET_REVISION = str(SOURCES["huggingface"]["revision"])
EVALUATOR_REVISION = str(SOURCES["other_defaults"][0]["revision"])
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def source_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else prepared_benchmark_dir("hle")


def dataset_id() -> str:
    return resolve_provider_ids(SOURCES, "huggingface")[0]


def _raw_dataset() -> Path:
    return raw_source_dir("hle", "huggingface", dataset_id())


def _prepared_file(root: str | Path | None = None) -> Path:
    return source_dir(root) / "test-00000-of-00001.parquet"


def _contract_path(root: str | Path | None = None) -> Path:
    return source_dir(root) / "adapter_contract.json"


def _ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        contract = json.loads(_contract_path(path.parent).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if contract != {
        "dataset_id": dataset_id(),
        "dataset_revision": DATASET_REVISION,
        "evaluator_revision": EVALUATOR_REVISION,
    }:
        return False
    try:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files={"test": str(path)}, split="test")
        return bool(len(dataset)) and {"id", "question", "answer", "image"}.issubset(
            dataset.column_names
        )
    except Exception:
        return False


def ensure_source(
    root: str | Path | None = None,
    force_download: bool = False,
    source: str | None = None,
) -> Path:
    if source not in {None, "auto", "huggingface"}:
        raise ValueError("HLE is a gated HuggingFace dataset; use --source huggingface")
    destination = _prepared_file(root)
    if _ready(destination) and not force_download:
        return source_dir(root)
    raw = _raw_dataset()
    resolved_dataset_id = dataset_id()
    raw_ready = raw_source_matches(
        raw,
        provider="huggingface",
        source_id=resolved_dataset_id,
        revision=DATASET_REVISION,
    )
    if conversion_only() and not raw_ready:
        raise FileNotFoundError(
            f"HLE Raw source does not match pinned revision {DATASET_REVISION}: {raw}"
        )
    if not conversion_only() and (force_download or not raw_ready):
        from huggingface_hub import snapshot_download

        log_download_source(
            "hle", "huggingface", resolved_dataset_id, raw, revision=DATASET_REVISION
        )
        try:
            snapshot_download(
                repo_id=resolved_dataset_id,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=str(raw),
                token=os.environ.get("HF_TOKEN") or None,
            )
        except Exception as exc:
            (raw / ".lychee_source.json").unlink(missing_ok=True)
            raise RuntimeError(
                "HLE is gated. Accept the cais/hle terms and set HF_TOKEN before preparing it."
            ) from exc
    if not raw.exists():
        raise FileNotFoundError(f"HLE raw source is missing: {raw}")
    dataset = load_local_dataset_split(raw, "test")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    remove_path(temporary)
    dataset.to_parquet(str(temporary))
    os.replace(temporary, destination)
    metadata = raw / ".lychee_source.json"
    if metadata.is_file():
        shutil.copy2(metadata, destination.parent / ".lychee_source.json")
    _contract_path(root).write_text(
        json.dumps(
            {
                "dataset_id": dataset_id(),
                "dataset_revision": DATASET_REVISION,
                "evaluator_revision": EVALUATOR_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not _ready(destination):
        raise RuntimeError("prepared HLE parquet is incomplete")
    return source_dir(root)


def _image_content(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("path"):
            return str(value["path"])
        if value.get("bytes"):
            import base64

            return "data:image/png;base64," + base64.b64encode(value["bytes"]).decode("ascii")
    if hasattr(value, "save"):
        import base64

        buffer = BytesIO()
        image_format = str(getattr(value, "format", None) or "PNG").upper()
        value.save(buffer, format=image_format)
        mime_subtype = "jpeg" if image_format in {"JPG", "JPEG"} else image_format.lower()
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/{mime_subtype};base64,{encoded}"
    return str(value)


def load_hle(n: Optional[int] = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    source = ensure_source()
    dataset = load_dataset(
        "parquet", data_files={"test": str(source / "test-00000-of-00001.parquet")}, split="test"
    )
    records = []
    for row in dataset:
        image = _image_content(row.get("image"))
        question = str(row["question"])
        records.append(
            {
                "task": "hle",
                "kind": "hle",
                "question": question,
                "gold": str(row["answer"]),
                "context": None,
                "metadata": {
                    "task_id": str(row["id"]),
                    "category": row.get("category"),
                    "official_dataset_revision": DATASET_REVISION,
                    "official_evaluator_revision": EVALUATOR_REVISION,
                    "image": image,
                    "system_prompt": SYSTEM_PROMPT,
                    "multimodal_content": [
                        {"type": "text", "text": question},
                        *([{"type": "image_url", "image_url": {"url": image}}] if image else []),
                    ],
                },
            }
        )
        if n is not None and len(records) >= n:
            break
    return records


def parse_judge_response(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judge_score_details(judge_response: Any) -> dict[str, Any]:
    parsed = parse_judge_response(judge_response)
    if parsed is None:
        return {"score": 0.0, "judge_status": "missing"}
    correct = str(parsed.get("correct") or "").lower() == "yes"
    return {
        "score": 1.0 if correct else 0.0,
        "judge_status": "available",
        "extracted_final_answer": parsed.get("extracted_final_answer")
        or parsed.get("model_answer"),
        "judge_reasoning": parsed.get("reasoning"),
        "confidence": parsed.get("confidence"),
        "correct": "yes" if correct else "no",
    }


def _normalize_local_judge_candidate(
    candidate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize narrow scalar variants from the local JSON-object judge.

    This deliberately does not repair JSON syntax or alter the preserved raw
    model output. It only canonicalizes already parsed scalar values that carry
    the same unambiguous meaning as the local judge contract.
    """

    if not isinstance(candidate, dict):
        raise TypeError("HLE local judge response must be one JSON object")
    normalized = dict(candidate)
    coercions: list[dict[str, Any]] = []

    correct = normalized.get("correct")
    if isinstance(correct, bool):
        replacement = "yes" if correct else "no"
        normalized["correct"] = replacement
        coercions.append({"field": "correct", "from": correct, "to": replacement})
    elif isinstance(correct, str):
        replacement = correct.strip().lower()
        if replacement != correct:
            normalized["correct"] = replacement
            coercions.append({"field": "correct", "from": correct, "to": replacement})

    confidence = normalized.get("confidence")
    if isinstance(confidence, str):
        stripped = confidence.strip()
        replacement = stripped[:-1].strip() if stripped.endswith("%") else stripped
        if replacement.isdigit() and replacement != confidence:
            normalized["confidence"] = replacement
            coercions.append(
                {"field": "confidence", "from": confidence, "to": replacement}
            )

    for field in ("extracted_final_answer", "reasoning"):
        scalar = normalized.get(field)
        if scalar is not None and not isinstance(scalar, str):
            replacement = json.dumps(scalar, ensure_ascii=False, separators=(",", ":"))
            normalized[field] = replacement
            coercions.append({"field": field, "from": scalar, "to": replacement})
    return normalized, coercions


def calibration_error(confidences: list[float], correctness: list[bool], beta: int = 100) -> float:
    """Compute HLE's official L2 calibration error without a NumPy dependency."""

    if not confidences or len(confidences) != len(correctness):
        return 0.0
    ordered = sorted(zip(confidences, correctness), key=lambda item: item[0])
    total = len(ordered)
    squared_error = 0.0
    for start in range(0, total, max(1, beta)):
        bucket = ordered[start : start + max(1, beta)]
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        mean_correct = sum(1.0 if item[1] else 0.0 for item in bucket) / len(bucket)
        squared_error += (len(bucket) / total) * (mean_confidence - mean_correct) ** 2
    return squared_error**0.5


def _load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            request_fingerprint = str(row.get("judge_request_fingerprint") or "")
            if request_fingerprint:
                cached[request_fingerprint] = row
    return cached


def prepare_judge_responses(
    predictions: list[dict[str, Any]],
    gold_by_id: Mapping[str, dict[str, Any]],
    run_dir: Path,
    *,
    model: str,
    base_url: str,
    auth_mode: str,
    api_key_env: str,
    api_key: str | None,
    workers: int,
    timeout: float,
    max_tokens: int,
    thinking_mode: str,
    output_mode: str,
    max_attempts: int,
) -> None:
    """Attach official-format HLE judge responses with a resumable cache."""

    from pydantic import BaseModel

    from ...runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

    class OfficialExtractedAnswer(BaseModel):
        extracted_final_answer: str
        reasoning: str
        correct: Literal["yes", "no"]
        confidence: int
        strict: Literal[True]

    class LocalExtractedAnswer(BaseModel):
        extracted_final_answer: str
        reasoning: str
        correct: Literal["yes", "no"]
        confidence: int

    cache_path = run_dir / "official_evaluation" / "hle_judge_responses.jsonl"
    attempts_path = run_dir / "official_evaluation" / "hle_judge_attempts.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_judge_cache(cache_path)
    lock = threading.Lock()
    if thinking_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError("HLE judge thinking_mode must be inherit, enabled or disabled")
    if output_mode not in {"official_schema", "local_json_object"}:
        raise ValueError(
            "HLE judge output_mode must be official_schema or local_json_object"
        )
    max_attempts = max(1, int(max_attempts))
    extra_body = {}
    if thinking_mode != "inherit":
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": thinking_mode == "enabled"}
        }
    backend = OpenAICompatibleBackend(
        model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        auth_mode=auth_mode,
        timeout=timeout,
        do_sample=False,
        model_info={"json_output": True, "structured_output": True},
        request_limits={"max_concurrency": workers},
        max_retries=1,
        extra_body=extra_body,
    )

    def judge(
        prediction: dict[str, Any],
    ) -> tuple[tuple[str, int], dict[str, Any] | None, dict[str, Any] | None]:
        key = (str(prediction.get("case_id") or ""), int(prediction.get("trial_index", 0)))
        item = gold_by_id[key[0]]
        prompt = JUDGE_PROMPT.format(
            question=item["question"],
            correct_answer=item["gold"],
            response=prediction.get("prediction", ""),
        )
        if output_mode == "local_json_object":
            prompt += (
                "\n\nReturn exactly one compact JSON object with keys "
                '"extracted_final_answer", "reasoning", "correct", and "confidence". '
                'Keep reasoning to at most 100 words.'
            )
        request_contract = {
            "schema_version": 1,
            "case_id": key[0],
            "trial_index": key[1],
            "judge_model": model,
            "judge_base_url": base_url,
            "thinking_mode": thinking_mode,
            "output_mode": output_mode,
            "max_tokens": max_tokens,
            "prompt": prompt,
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(
                request_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if request_fingerprint in cache:
            return key, dict(cache[request_fingerprint]["judge_response"]), None
        failures: list[str] = []
        parsed: dict[str, Any] | None = None
        field_coercions: list[dict[str, Any]] = []
        json_output: type[BaseModel] | bool = OfficialExtractedAnswer
        request_overrides: dict[str, Any] | None = None
        if output_mode == "local_json_object":
            json_output = False
            request_overrides = {
                "extra_body": {
                    "structured_outputs": {
                        "json_object": True,
                        "disable_any_whitespace": True,
                    }
                }
            }
        for attempt in range(1, max_attempts + 1):
            parse_error: str | None = None
            result = None
            attempt_prompt = prompt
            if failures:
                attempt_prompt += (
                    "\n\nCorrection: the previous response was invalid. Produce a fresh, "
                    "complete JSON object now. Keep both string fields short, use correct "
                    'as exactly "yes" or "no", and use an integer confidence from 0 to 100.'
                )
            try:
                result = backend.generate_chat(
                    [{"role": "user", "content": attempt_prompt}],
                    max_new_tokens=max_tokens,
                    json_output=json_output,
                    request_overrides=request_overrides,
                )
                candidate = json.loads(result.text)
                if output_mode == "local_json_object":
                    candidate, field_coercions = _normalize_local_judge_candidate(candidate)
                    parsed = LocalExtractedAnswer.model_validate(candidate).model_dump(
                        mode="json"
                    )
                else:
                    parsed = OfficialExtractedAnswer.model_validate(candidate).model_dump(
                        mode="json"
                    )
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    f"attempt={attempt} "
                    f"finish_reason={getattr(result, 'finish_reason', None)} "
                    f"content_chars={len(getattr(result, 'text', '') or '')} "
                    f"reasoning_chars={len(getattr(result, 'reasoning_content', '') or '')} "
                    f"error={parse_error}"
                )
            attempt_row = {
                "case_id": key[0],
                "trial_index": key[1],
                "judge_model": model,
                "judge_request_fingerprint": request_fingerprint,
                "thinking_mode": thinking_mode,
                "output_mode": output_mode,
                "attempt": attempt,
                "prompt_variant": "initial" if attempt == 1 else "corrective_retry",
                "finish_reason": getattr(result, "finish_reason", None),
                "output_text": getattr(result, "text", None),
                "reasoning_content": getattr(result, "reasoning_content", None),
                "parse_error": parse_error,
                "valid": parsed is not None,
                "field_coercions": field_coercions,
            }
            with lock, attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt_row, ensure_ascii=False) + "\n")
                handle.flush()
            if parse_error is not None:
                field_coercions = []
                continue
            break
        if parsed is None:
            message = (
                f"HLE judge failed to return valid structured output for {key}: "
                + " | ".join(failures)
            )
            return key, None, {
                "error_type": "BenchmarkEvaluationError",
                "error_message": message,
                "judge_request_fingerprint": request_fingerprint,
                "attempt_failures": failures,
            }
        row = {
            "case_id": key[0],
            "trial_index": key[1],
            "judge_model": model,
            "judge_request_fingerprint": request_fingerprint,
            "thinking_mode": thinking_mode,
            "output_mode": output_mode,
            "judge_response": parsed,
            "field_coercions": field_coercions,
        }
        with lock, cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        return key, parsed, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        judged = {key: (response, error) for key, response, error in pool.map(judge, predictions)}
    for prediction in predictions:
        key = (str(prediction.get("case_id") or ""), int(prediction.get("trial_index", 0)))
        response, error = judged[key]
        if response is not None:
            prediction["hle_judge_response"] = response
            prediction.pop("hle_judge_error", None)
        else:
            prediction["hle_judge_error"] = error


def _prepare(force: bool = False, source: str | None = None) -> str:
    return str(ensure_source(force_download=force, source=source))


def _score(_prediction: str, _gold, record: Mapping[str, Any]) -> dict:
    if record.get("hle_judge_error"):
        error = record["hle_judge_error"]
        raise BenchmarkEvaluationError(
            str(error.get("error_message") if isinstance(error, Mapping) else error)
        )
    return judge_score_details(record.get("hle_judge_response"))


class HLEBenchmark(Benchmark):
    def add_analysis_arguments(self, parser: Any) -> None:
        if any("--hle-judge-model" in action.option_strings for action in parser._actions):
            return
        parser.add_argument(
            "--hle-judge-model",
            default=os.environ.get("LYCHEE_HLE_JUDGE_MODEL", "o3-mini-2025-01-31"),
        )
        parser.add_argument(
            "--hle-judge-base-url",
            default=os.environ.get("LYCHEE_HLE_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )
        parser.add_argument(
            "--hle-judge-api-key-env",
            default=os.environ.get("LYCHEE_HLE_JUDGE_API_KEY_ENV", "OPENAI_API_KEY"),
        )
        parser.add_argument(
            "--hle-judge-auth-mode",
            choices=("env", "none"),
            default=os.environ.get("LYCHEE_HLE_JUDGE_AUTH_MODE", "env"),
            help="Judge authentication contract; use none for a trusted local endpoint.",
        )
        parser.add_argument("--hle-judge-api-key", default=None)
        parser.add_argument(
            "--hle-judge-workers",
            type=int,
            default=int(os.environ.get("LYCHEE_HLE_JUDGE_WORKERS", "8")),
        )
        parser.add_argument(
            "--hle-judge-timeout",
            type=float,
            default=float(os.environ.get("LYCHEE_HLE_JUDGE_TIMEOUT", "300")),
        )
        parser.add_argument(
            "--hle-judge-max-tokens",
            type=int,
            default=int(os.environ.get("LYCHEE_HLE_JUDGE_MAX_TOKENS", "4096")),
        )
        parser.add_argument(
            "--hle-judge-thinking-mode",
            choices=("inherit", "enabled", "disabled"),
            default=os.environ.get("LYCHEE_HLE_JUDGE_THINKING_MODE", "inherit"),
        )
        parser.add_argument(
            "--hle-judge-max-attempts",
            type=int,
            default=int(os.environ.get("LYCHEE_HLE_JUDGE_MAX_ATTEMPTS", "3")),
        )
        parser.add_argument(
            "--hle-judge-output-mode",
            choices=("official_schema", "local_json_object"),
            default=os.environ.get("LYCHEE_HLE_JUDGE_OUTPUT_MODE", "official_schema"),
            help=(
                "official_schema uses HLE's Pydantic response_format; "
                "local_json_object is a concise vLLM development-judge contract."
            ),
        )

    def evaluation_options(self, args: Any) -> dict[str, Any]:
        return {
            "model": args.hle_judge_model,
            "base_url": args.hle_judge_base_url,
            "auth_mode": args.hle_judge_auth_mode,
            "api_key_env": args.hle_judge_api_key_env,
            "api_key": args.hle_judge_api_key,
            "workers": args.hle_judge_workers,
            "timeout": args.hle_judge_timeout,
            "max_tokens": args.hle_judge_max_tokens,
            "thinking_mode": args.hle_judge_thinking_mode,
            "output_mode": args.hle_judge_output_mode,
            "max_attempts": args.hle_judge_max_attempts,
        }

    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        missing = [item for item in predictions if not item.get("hle_judge_response")]
        if not missing:
            return
        if context.external_evaluator == "skip":
            raise BenchmarkEvaluationError(
                "HLE requires official judge responses and cannot be locally exact-scored."
            )
        prepare_judge_responses(missing, gold_by_id, context.run_dir, **context.options)

    def aggregate(
        self,
        samples: Sequence[Mapping[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        judged = [
            sample.get("score_details") or {}
            for sample in samples
            if (sample.get("score_details") or {}).get("judge_status") == "available"
        ]
        confidences: list[float] = []
        correctness: list[bool] = []
        for details in judged:
            try:
                raw_confidence = details.get("confidence")
                if raw_confidence is None:
                    continue
                confidence = min(100.0, max(0.0, float(raw_confidence))) / 100.0
            except (TypeError, ValueError):
                continue
            confidences.append(confidence)
            correctness.append(float(details.get("score", 0.0) or 0.0) == 1.0)
        metrics["hle_judged_predictions"] = len(judged)
        metrics["hle_calibration_predictions"] = len(confidences)
        metrics["hle_calibration_error_percent"] = (
            round(100.0 * calibration_error(confidences, correctness), 2) if confidences else None
        )
        return metrics


BENCHMARK = register_benchmark(
    HLEBenchmark(
        benchmark_id="hle",
        name="Humanity's Last Exam",
        category="expert_knowledge_multimodal",
        sources=SOURCES,
        full_prepare_target="hle",
        prepare_handlers={"hle": _prepare},
        loaders={"hle": load_hle},
        scorer_kinds={"hle": "hle"},
        score_handlers={"hle": _score},
        binary_kinds=("hle",),
        evaluation_mode="batch_final",
        evaluation_requirements={
            "config_key": "hle_judge",
            "required_when_external_evaluator": ["auto", "run"],
            "required_fields": ["model", "base_url"],
        },
        capabilities={
            "required": ["text_generation", "vision"],
            "attachments_required": True,
        },
        runtime_defaults={"max_new_tokens": 32768, "max_turns": 1},
        scoring_profiles={
            "hle": {
                "default_profile": "official",
                "profiles": {
                    "official": {
                        "scorer_id": "hle",
                        "parameters": {
                            "evaluator": "official_structured_llm_judge",
                            "reports_calibration_error": True,
                        },
                    }
                },
            }
        },
    )
)
