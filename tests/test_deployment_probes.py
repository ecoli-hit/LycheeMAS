from __future__ import annotations

import json
from urllib.error import HTTPError

import lychee_mas.eval.deployments.probes as probes
import pytest


def test_api_probe_explains_missing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: object, **_kwargs: object) -> None:
        raise HTTPError("https://example.test/v1/models", 401, "Unauthorized", {}, None)

    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.setattr(probes, "urlopen", reject)
    health = probes.probe_openai_endpoint("https://example.test/v1", "TEST_API_KEY")
    assert health["status"] == "auth_required"
    assert health["credential_present"] is False
    assert "TEST_API_KEY" in health["health_detail"]
    assert "not set" in health["health_detail"]


def test_api_probe_executes_minimal_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "你好，有什么可以帮你？"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def respond(request, **_kwargs: object):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    monkeypatch.setattr(probes, "urlopen", respond)
    health = probes.probe_openai_endpoint(
        "https://example.test/v1", "TEST_API_KEY", model_id="test-model"
    )

    assert health["status"] == "running"
    assert health["probe_response_preview"].startswith("你好")
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-value"
    assert captured["body"]["messages"] == [{"role": "user", "content": "你好"}]


def test_vllm_probe_verifies_native_thinking_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "brief reasoning",
                                "content": "你好",
                            }
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def respond(request, **_kwargs: object):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(probes, "urlopen", respond)
    health = probes.probe_openai_endpoint(
        "http://127.0.0.1:6100/v1",
        None,
        model_id="Qwen3.6-27B",
        auth_mode="none",
        thinking_budget_field="thinking_token_budget",
    )

    assert health["status"] == "running"
    assert health["thinking_budget_probe"] is True
    assert health["thinking_budget_parameter"] == "thinking_token_budget"
    assert health["reasoning_token_accounting"] == "client_retokenized_required"
    assert health["probe_reasoning_content_present"] is True
    assert health["provider_reasoning_tokens_reported"] is False
    assert captured["body"]["thinking_token_budget"] == 16
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_probe_prefers_provider_reasoning_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "brief reasoning",
                                "content": "323",
                            }
                        }
                    ],
                    "usage": {
                        "completion_tokens": 12,
                        "completion_tokens_details": {"reasoning_tokens": 9},
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr(probes, "urlopen", lambda *_args, **_kwargs: Response())
    health = probes.probe_openai_endpoint(
        "http://127.0.0.1:6100/v1",
        None,
        model_id="Qwen3.6-27B",
        auth_mode="none",
        thinking_budget_field="thinking_token_budget",
    )

    assert health["reasoning_token_accounting"] == "provider_usage"
    assert health["provider_reasoning_tokens_reported"] is True
    assert health["probe_completion_tokens"] == 12
    assert health["probe_reasoning_tokens"] == 9
