"""AgenTracer attributor: public Who/When JSON contract, fully offline."""

from __future__ import annotations

import pytest
from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Message, TaskQuery, Trajectory
from lychee_mas.methods.postrun.agentracer import format_trajectory, parse_diagnosis
from lychee_mas.plugins.postrun import analyze_run


def make_trajectory() -> Trajectory:
    traj = Trajectory(task_id="q1", final_answer=Answer(content="wrong"))
    traj.add(Message(sender="Planner", role="assistant", round=0, content="plan"))
    traj.add(Message(sender="Coder", role="assistant", round=1, content="buggy code"))
    return traj


class ScriptedTracer:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_registry_and_public_contract():
    assert "agentracer" in REGISTRY.list("attributor")
    tracer = ScriptedTracer(
        '```json\n{"mistake_step":"1","mistake_agent":"Coder",'
        '"reason":"introduced the bug","suggested_fix":"repair the branch",'
        '"confidence":0.9}\n```'
    )
    attributor = REGISTRY.create("attributor", "agentracer", tracer_llm=tracer)
    result = attributor.attribute(make_trajectory(), TaskQuery(question="solve", gold="correct"))
    assert [(a.agent, a.step, a.is_fault) for a in result] == [("Coder", 1, True)]
    assert result[0].meta["suggested_fix"] == "repair the branch"
    assert "Question: solve" in tracer.prompts[0]
    assert "Ground Truth: correct" in tracer.prompts[0]


def test_trajectory_uses_ordered_zero_based_steps():
    text = format_trajectory(make_trajectory())
    assert "Step 0 - Planner" in text
    assert "Step 1 - Coder" in text


def test_step_agent_pair_is_validated():
    traj = make_trajectory()
    with pytest.raises(ValueError, match="实际 agent"):
        parse_diagnosis('{"mistake_step":1,"mistake_agent":"Planner","reason":"x"}', traj)
    with pytest.raises(ValueError, match="越界"):
        parse_diagnosis('{"mistake_step":9,"mistake_agent":"Coder","reason":"x"}', traj)


def test_invalid_json_is_explicit():
    with pytest.raises(ValueError, match="没有 JSON"):
        parse_diagnosis("not json", make_trajectory())


def test_analyze_run_can_run_attribution_before_credit_is_implemented():
    traj = make_trajectory()
    tracer = ScriptedTracer('{"mistake_step":1,"mistake_agent":"Coder","reason":"bug"}')
    credits = analyze_run(
        traj,
        method="agentracer",
        credit_assigner=None,
        tracer_llm=tracer,
        context={"question": "solve"},
    )
    assert credits == {}
    assert traj.meta["attribution"][0]["agent"] == "Coder"


def test_llm_callback_is_required():
    with pytest.raises(ValueError, match="tracer_llm"):
        REGISTRY.create("attributor", "agentracer")
