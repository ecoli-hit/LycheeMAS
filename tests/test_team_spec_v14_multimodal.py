from __future__ import annotations

from types import SimpleNamespace

from lychee_mas.core.types import TaskQuery
from lychee_mas.eval.teams.contracts import model_resource_requirements
from lychee_mas.runtime.adapters.frameworks.crewai.runtime import CrewAIRuntime
from lychee_mas.runtime.adapters.frameworks.langgraph.runtime import LangGraphRuntime


def test_model_gateway_runtime_preserves_benchmark_media() -> None:
    runtime = LangGraphRuntime()
    query = TaskQuery(
        "question",
        meta={
            "multimodal_content": [
                {"type": "text", "text": "question"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ]
        },
    )

    runtime._set_task_media(query)
    content = runtime._model_user_content("materialized question")

    assert content == [
        {"type": "text", "text": "materialized question"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]


def test_crewai_gateway_bridge_preserves_benchmark_media() -> None:
    captured = {}

    class Gateway:
        def invoke_sync(self, role, messages, **kwargs):
            captured.update(role=role, messages=messages, kwargs=kwargs)
            return SimpleNamespace(content="ok")

    runtime = CrewAIRuntime()
    GatewayLLM = runtime._gateway_llm_type()
    llm = GatewayLLM(
        Gateway(),
        {},
        set(),
        {},
        {},
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
    )

    assert llm.call("inspect this image") == "ok"
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "inspect this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]


def test_explicit_vision_capability_is_a_deployment_requirement() -> None:
    requirements = model_resource_requirements(
        {
            "nodes": [
                {
                    "id": "Solver",
                    "kind": "model_agent",
                    "behavior": {"type": "assistant"},
                    "capabilities": ["reason", "vision"],
                }
            ]
        }
    )

    assert requirements[0]["required_capabilities"] == ["text_generation", "vision"]
