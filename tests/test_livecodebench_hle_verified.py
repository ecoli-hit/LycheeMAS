"""Contracts for the LiveCodeBench and HLE-Verified adapters."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from io import BytesIO
from types import SimpleNamespace

from lychee_mas.eval.benchmarks import BENCHMARKS
from lychee_mas.eval.benchmarks.hle import prepare_judge_responses
from lychee_mas.eval.benchmarks.hle_verified import _score as score_hle_verified
from lychee_mas.eval.benchmarks.livecodebench import (
    RELEASE_VERSION,
    _decode_tests,
    _question,
)
from lychee_mas.eval.benchmarks.livecodebench import _score as score_livecodebench
from lychee_mas.runtime.model.multimodal import pil_image_from_data_uri


def test_livecodebench_private_test_encoding_matches_official_format() -> None:
    tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert _decode_tests(json.dumps(tests)) == tests
    assert _decode_tests(encoded) == tests


def test_multimodal_data_uri_accepts_gif_for_framework_adapters() -> None:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="GIF")
    uri = "data:image/gif;base64," + base64.b64encode(buffer.getvalue()).decode()

    decoded = pil_image_from_data_uri(uri)

    assert decoded.size == (2, 2)
    assert decoded.format == "GIF"


def test_livecodebench_prompt_never_contains_private_tests() -> None:
    prompt = _question(
        {
            "question_content": "Double an integer.",
            "starter_code": "",
            "private_test_cases": "SECRET_PRIVATE_TEST",
        }
    )
    assert "Double an integer." in prompt
    assert "SECRET_PRIVATE_TEST" not in prompt


def test_livecodebench_score_uses_official_checker_result() -> None:
    details = score_livecodebench(
        "ignored",
        None,
        {
            "livecodebench_official_result": {
                "passed": True,
                "question_id": "q1",
                "test_results": [True],
                "evaluator_metadata": {},
            }
        },
    )
    assert details["score"] == 1.0
    assert details["passed"] is True


def test_hle_verified_score_preserves_verified_subset() -> None:
    details = score_hle_verified(
        "ignored",
        None,
        {
            "hle_judge_response": {
                "correct": "yes",
                "confidence": 80,
                "extracted_final_answer": "D",
                "reasoning": "matches",
            }
        },
    )
    assert details["score"] == 1.0
    assert details["verified_subset"] == "Gold subset"


def test_hle_local_judge_disables_thinking_retries_and_reuses_fingerprinted_cache(
    tmp_path, monkeypatch
) -> None:
    from lychee_mas.runtime.adapters.inference import openai_compatible

    calls = []

    class Backend:
        def __init__(self, *_args, **kwargs):
            calls.append({"init": kwargs})

        def generate_chat(self, messages, **kwargs):
            calls.append({"messages": messages, "request": kwargs})
            if len(calls) == 2:
                return SimpleNamespace(
                    text="",
                    reasoning_content="unfinished judge reasoning",
                    finish_reason="length",
                )
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "extracted_final_answer": "4",
                        "reasoning": "The answers match.",
                        "correct": "yes",
                        "confidence": 100,
                        "strict": True,
                    }
                ),
                reasoning_content="",
                finish_reason="stop",
            )

    monkeypatch.setattr(openai_compatible, "OpenAICompatibleBackend", Backend)
    predictions = [{"case_id": "case-1", "trial_index": 0, "prediction": "Answer: 4"}]
    prepare_judge_responses(
        predictions,
        {"case-1": {"question": "What is 2+2?", "gold": "4"}},
        tmp_path,
        model="local-judge",
        base_url="http://127.0.0.1:6200/v1",
        auth_mode="none",
        api_key_env="UNSET_LOCAL_KEY",
        api_key=None,
        workers=1,
        timeout=30.0,
        max_tokens=512,
        thinking_mode="disabled",
        output_mode="local_json_object",
        max_attempts=2,
    )

    assert calls[0]["init"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert len([call for call in calls if "request" in call]) == 2
    assert "Correction: the previous response was invalid" in calls[2]["messages"][0][
        "content"
    ]
    structured_outputs = calls[1]["request"]["request_overrides"]["extra_body"][
        "structured_outputs"
    ]
    assert structured_outputs["disable_any_whitespace"] is True
    assert structured_outputs["json_object"] is True
    assert calls[1]["request"]["json_output"] is False
    assert '"correct", and "confidence"' in calls[1]["messages"][0]["content"]
    assert predictions[0]["hle_judge_response"]["correct"] == "yes"
    attempts = [
        json.loads(line)
        for line in (tmp_path / "official_evaluation/hle_judge_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["valid"] for row in attempts] == [False, True]
    assert attempts[0]["output_text"] == ""
    assert attempts[0]["output_mode"] == "local_json_object"
    assert len(attempts[0]["judge_request_fingerprint"]) == 64

    prepare_judge_responses(
        predictions,
        {"case-1": {"question": "What is 2+2?", "gold": "4"}},
        tmp_path,
        model="local-judge",
        base_url="http://127.0.0.1:6200/v1",
        auth_mode="none",
        api_key_env="UNSET_LOCAL_KEY",
        api_key=None,
        workers=1,
        timeout=30.0,
        max_tokens=512,
        thinking_mode="disabled",
        output_mode="local_json_object",
        max_attempts=2,
    )
    assert len([call for call in calls if "request" in call]) == 2


def test_hle_local_judge_isolates_invalid_items_and_normalizes_scalar_variants(
    tmp_path, monkeypatch
) -> None:
    from lychee_mas.runtime.adapters.inference import openai_compatible

    responses = iter(
        [
            SimpleNamespace(
                text='{"correct": "maybe"}',
                reasoning_content="",
                finish_reason="stop",
            ),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "extracted_final_answer": 4,
                        "reasoning": "matches",
                        "correct": True,
                        "confidence": "95%",
                    }
                ),
                reasoning_content="",
                finish_reason="stop",
            ),
        ]
    )

    class Backend:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_chat(self, _messages, **_kwargs):
            return next(responses)

    monkeypatch.setattr(openai_compatible, "OpenAICompatibleBackend", Backend)
    predictions = [
        {"case_id": "bad", "trial_index": 0, "prediction": "unknown"},
        {"case_id": "good", "trial_index": 0, "prediction": "Answer: 4"},
    ]
    prepare_judge_responses(
        predictions,
        {
            "bad": {"question": "Bad?", "gold": "No"},
            "good": {"question": "What is 2+2?", "gold": "4"},
        },
        tmp_path,
        model="local-judge",
        base_url="http://127.0.0.1:6200/v1",
        auth_mode="none",
        api_key_env="UNSET_LOCAL_KEY",
        api_key=None,
        workers=1,
        timeout=30.0,
        max_tokens=512,
        thinking_mode="disabled",
        output_mode="local_json_object",
        max_attempts=1,
    )

    assert "hle_judge_error" in predictions[0]
    assert predictions[1]["hle_judge_response"]["correct"] == "yes"
    assert predictions[1]["hle_judge_response"]["confidence"] == 95
    attempts = [
        json.loads(line)
        for line in (tmp_path / "official_evaluation/hle_judge_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert attempts[1]["field_coercions"] == [
        {"field": "correct", "from": True, "to": "yes"},
        {"field": "confidence", "from": "95%", "to": "95"},
        {"field": "extracted_final_answer", "from": 4, "to": "4"},
    ]


def test_new_benchmark_descriptors_are_registered() -> None:
    livecodebench = BENCHMARKS.get("livecodebench").descriptor()
    hle_verified = BENCHMARKS.get("hle_verified").descriptor()
    assert livecodebench["runtime_defaults"]["trials_per_case"] == 1
    assert livecodebench["task_contracts"]["livecodebench"]["scoring"][
        "default_profile"
    ] == "official_release_v6"
    assert RELEASE_VERSION == "release_v6"
    assert hle_verified["capabilities"]["required"] == ["text_generation", "vision"]
