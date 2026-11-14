"""Counterfactual first-error localization for a replayable four-agent task.

The module deliberately keeps the replay contract small: every step has an agent,
input and output; replacing one output deterministically replays all later steps.
A repair proposer is injected so the same localizer works with HFBackend, an API,
or a deterministic oracle used by tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class ReplayStep:
    index: int
    agent: str
    input: str
    output: str


@dataclass
class ReplayTrace:
    task_id: str
    question: str
    gold_answer: int
    steps: list[ReplayStep]
    final_answer: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairProposal:
    replacement: str
    reason: str = ""
    confidence: float = 0.0


@dataclass
class LocalizationResult:
    task_id: str
    baseline_correct: bool
    first_effective_step: int | None
    responsible_agent: str | None
    repair_reason: str
    repaired_answer: str | None
    model_calls: int
    prompt_chars: int
    proposer_chars: int
    tried_steps: list[int]
    trace: ReplayTrace

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["trace"] = self.trace.to_dict()
        return result


class ArithmeticReplayTask:
    """Planner -> solver -> verifier -> output for an automatically checkable task."""

    agents = ("planner", "solver", "verifier", "output")

    def make_task(self, task_id: str, a: int, b: int, c: int) -> dict[str, Any]:
        return {"task_id": task_id, "a": int(a), "b": int(b), "c": int(c)}

    def expected(self, task: Mapping[str, Any]) -> int:
        return (int(task["a"]) + int(task["b"])) * int(task["c"])

    def _clean_output(self, task: Mapping[str, Any], index: int) -> str:
        a, b, c = int(task["a"]), int(task["b"]), int(task["c"])
        answer = self.expected(task)
        return (
            f"PLAN: a={a}; b={b}; c={c}; op=add_then_multiply"
            if index == 0
            else f"SOLVER_RESULT: {answer}"
            if index == 1
            else f"VERIFIER_RESULT: {answer}; status=OK"
            if index == 2
            else f"FINAL_ANSWER: {answer}"
        )

    def _parse_plan(self, text: str) -> tuple[int, int, int]:
        values = {name: int(value) for name, value in re.findall(r"([abc])\s*=\s*(-?\d+)", text)}
        return values.get("a", 0), values.get("b", 0), values.get("c", 0)

    @staticmethod
    def _parse_int(text: str) -> int:
        match = re.search(r"-?\d+", text)
        return int(match.group(0)) if match else 0

    def run(
        self, task: Mapping[str, Any], overrides: Mapping[int, str] | None = None
    ) -> ReplayTrace:
        overrides = dict(overrides or {})
        question = f"Compute (a+b)*c for a={task['a']}, b={task['b']}, c={task['c']}."
        plan_input = question
        plan = overrides.get(0, self._clean_output(task, 0))
        a, b, c = self._parse_plan(plan)
        solver_input = plan
        solver = overrides.get(1, f"SOLVER_RESULT: {(a + b) * c}")
        verifier_input = solver
        solver_value = self._parse_int(solver)
        verifier = overrides.get(2, f"VERIFIER_RESULT: {solver_value}; status=OK")
        output_input = verifier
        verified_value = self._parse_int(verifier)
        output = overrides.get(3, f"FINAL_ANSWER: {verified_value}")
        steps = [
            ReplayStep(0, "planner", plan_input, plan),
            ReplayStep(1, "solver", solver_input, solver),
            ReplayStep(2, "verifier", verifier_input, verifier),
            ReplayStep(3, "output", output_input, output),
        ]
        return ReplayTrace(
            task_id=str(task["task_id"]),
            question=question,
            gold_answer=self.expected(task),
            steps=steps,
            final_answer=output,
            meta={"task": dict(task), "overrides": dict(overrides)},
        )

    def validate(self, trace: ReplayTrace) -> bool:
        return self._parse_int(trace.final_answer) == trace.gold_answer

    def repair_prompt(self, trace: ReplayTrace, index: int) -> str:
        step = trace.steps[index]
        contract = (
            'Return JSON only: {"replacement": "exact replacement output", '
            '"reason": "local error and repair", "confidence": 0.0}. '
            "Use the target agent's output format; do not include a downstream answer."
        )
        return f"""You are repairing one step in a deterministic multi-agent computation.
Task: {trace.question}
Gold answer for validation is hidden from the repair decision.
Target step: {index}; agent: {step.agent}
Step input:
{step.input}
Current output:
{step.output}

{contract}"""

    def oracle_repair(self, task: Mapping[str, Any], index: int) -> RepairProposal:
        return RepairProposal(
            self._clean_output(task, index), "oracle correction for hand-verified smoke cases", 1.0
        )


def parse_repair(raw: str) -> RepairProposal:
    if not isinstance(raw, str):
        raise TypeError("repair proposer must return str")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("repair proposer output has no JSON object")
    data = json.loads(raw[start : end + 1])
    replacement = str(data["replacement"])
    if not replacement.strip():
        raise ValueError("replacement must be non-empty")
    confidence = float(data.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return RepairProposal(replacement, str(data.get("reason", "")), confidence)


def locate_first_effective_fix(
    task: ArithmeticReplayTask,
    failed_trace: ReplayTrace,
    proposer: Callable[[str, int, ReplayTrace], str | RepairProposal],
) -> LocalizationResult:
    """Try every step in order; first replay that validates is AgenTracer's key error."""
    baseline_correct = task.validate(failed_trace)
    calls = 0
    prompt_chars = 0
    proposer_chars = 0
    tried: list[int] = []
    if baseline_correct:
        return LocalizationResult(
            failed_trace.task_id,
            True,
            None,
            None,
            "baseline already correct",
            failed_trace.final_answer,
            0,
            0,
            0,
            tried,
            failed_trace,
        )
    for index, step in enumerate(failed_trace.steps):
        tried.append(index)
        prompt = task.repair_prompt(failed_trace, index)
        prompt_chars += len(prompt)
        raw = proposer(prompt, index, failed_trace)
        calls += 1
        if isinstance(raw, RepairProposal):
            proposal = raw
        else:
            proposer_chars += len(raw)
            proposal = parse_repair(raw)
        replay_overrides = dict(failed_trace.meta.get("overrides", {}))
        replay_overrides[index] = proposal.replacement
        replayed = task.run(failed_trace.meta["task"], replay_overrides)
        if task.validate(replayed):
            return LocalizationResult(
                failed_trace.task_id,
                False,
                index,
                step.agent,
                proposal.reason,
                replayed.final_answer,
                calls,
                prompt_chars,
                proposer_chars,
                tried,
                replayed,
            )
    return LocalizationResult(
        failed_trace.task_id,
        False,
        None,
        None,
        "no single-step repair produced a valid final answer",
        None,
        calls,
        prompt_chars,
        proposer_chars,
        tried,
        failed_trace,
    )


__all__ = [
    "ArithmeticReplayTask",
    "LocalizationResult",
    "RepairProposal",
    "ReplayStep",
    "ReplayTrace",
    "locate_first_effective_fix",
    "parse_repair",
]
