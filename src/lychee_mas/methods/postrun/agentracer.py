"""AgenTracer failure-attribution adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from ...core.registry import REGISTRY
from ...core.types import Trajectory
from .base import Attribution

TracerLLM = Callable[[str], str]


def format_trajectory(trajectory: Trajectory) -> str:
    """Serialize messages using AgenTracer's zero-based ordered step format."""
    if not trajectory.messages:
        raise ValueError("agentracer: trajectory.messages 不能为空")
    return "\n\n".join(
        f"Step {step} - {message.sender} ({message.role}, round={message.round}):\n"
        f"{message.content}"
        for step, message in enumerate(trajectory.messages)
    )


def _context_value(context: Any, *names: str, default: Any = "") -> Any:
    if isinstance(context, Mapping):
        for name in names:
            if name in context:
                return context[name]
    elif context is not None:
        for name in names:
            if hasattr(context, name):
                return getattr(context, name)
    return default


def build_prompt(trajectory: Trajectory, context: Any = None) -> str:
    """Build a framework-neutral version of the public AgenTracer diagnosis task."""
    question = _context_value(context, "question", default=trajectory.meta.get("question", ""))
    ground_truth = _context_value(
        context, "ground_truth", "gold", default=trajectory.meta.get("ground_truth", "")
    )
    feedback = _context_value(
        context,
        "environment_feedback",
        "feedback",
        default=trajectory.meta.get("environment_feedback", ""),
    )
    prediction = trajectory.final_answer.content if trajectory.final_answer is not None else ""
    return f"""Identify the decisive error in this failed multi-agent trajectory.

Task ID: {trajectory.task_id}
Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {prediction}
Environment Feedback: {feedback}

Execution History:
{format_trajectory(trajectory)}

Return JSON only:
{{
  "mistake_step": <zero-based step number>,
  "mistake_agent": "<agent name exactly as shown in the history>",
  "reason": "<why this is the decisive error>",
  "suggested_fix": "<guidance, not a complete replacement solution>",
  "confidence": <optional number from 0 to 1>
}}

The step must exist in the history and the agent must be the sender of that step.
Focus on the earliest error that decisively caused the final failure."""


def parse_diagnosis(raw: str, trajectory: Trajectory) -> Attribution:
    """Parse and validate AgenTracer's public diagnosis JSON contract."""
    if not isinstance(raw, str):
        raise TypeError(f"agentracer: tracer_llm 必须返回 str，得到 {type(raw).__name__}")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("agentracer: tracer 输出中没有 JSON 对象")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"agentracer: tracer 输出不是合法 JSON: {exc.msg}") from exc
    try:
        step = int(data["mistake_step"])
        agent = str(data["mistake_agent"])
        reason = str(data["reason"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("agentracer: 输出需要 mistake_step/mistake_agent/reason") from exc
    if step < 0 or step >= len(trajectory.messages):
        raise ValueError(
            f"agentracer: mistake_step={step} 越界；合法范围 0..{len(trajectory.messages) - 1}"
        )
    actual_agent = trajectory.messages[step].sender
    if agent != actual_agent:
        raise ValueError(
            f"agentracer: step {step} 的实际 agent 是 {actual_agent!r}，不是 {agent!r}"
        )
    confidence = float(data.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("agentracer: confidence 必须在 [0, 1]")
    return Attribution(
        agent=agent,
        step=step,
        is_fault=True,
        reason=reason,
        confidence=confidence,
        meta={
            "suggested_fix": str(data.get("suggested_fix", "")),
            "message_id": trajectory.messages[step].id,
            "message_round": trajectory.messages[step].round,
        },
    )


@REGISTRY.register("attributor", "agentracer")
class AgenTracerAttributor:
    """Locate one decisive Who/When failure using an injected tracer model."""

    name = "agentracer"

    def __init__(self, tracer_llm: TracerLLM | None = None, **_: Any) -> None:
        if tracer_llm is None:
            raise ValueError("agentracer: 需要 tracer_llm(prompt) -> str 回调")
        self.tracer_llm = tracer_llm

    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]:
        return [parse_diagnosis(self.tracer_llm(build_prompt(trajectory, context)), trajectory)]


__all__ = [
    "AgenTracerAttributor",
    "TracerLLM",
    "build_prompt",
    "format_trajectory",
    "parse_diagnosis",
]
