"""Framework-neutral task/progress-ledger orchestration protocol.

AutoGen owns its native Magentic-One implementation. LangGraph and CrewAI use
this protocol helper to compose the same declared TeamSpec operations inside
their own StateGraph and Flow execution kernels. The helper performs strict JSON
parsing and retry only; it never repairs or rewrites model output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from lychee_mas.core.types import Message
from lychee_mas.runtime.model.gateway import GatewayReply, ModelGateway

_PROGRESS_LEDGER_EXAMPLE = (
    '{"is_request_satisfied":{"reason":"...","answer":false},'
    '"is_in_loop":{"reason":"...","answer":false},'
    '"is_progress_being_made":{"reason":"...","answer":true},'
    '"next_speaker":{"reason":"...","answer":"NODE"},'
    '"instruction_or_question":{"reason":"...","answer":"..."}}'
)


@lru_cache(maxsize=1)
def _progress_ledger_schema():
    """Build the optional provider schema only when orchestration needs it."""

    from pydantic import ConfigDict, create_model

    boolean_entry = create_model(
        "_BooleanLedgerEntry",
        __config__=ConfigDict(extra="forbid"),
        reason=(str, ...),
        answer=(bool, ...),
    )
    string_entry = create_model(
        "_StringLedgerEntry",
        __config__=ConfigDict(extra="forbid"),
        reason=(str, ...),
        answer=(str, ...),
    )
    return create_model(
        "_ProgressLedgerSchema",
        __config__=ConfigDict(extra="forbid"),
        is_request_satisfied=(boolean_entry, ...),
        is_in_loop=(boolean_entry, ...),
        is_progress_being_made=(boolean_entry, ...),
        next_speaker=(string_entry, ...),
        instruction_or_question=(string_entry, ...),
    )


@dataclass
class LedgerDecision:
    satisfied: bool
    in_loop: bool
    making_progress: bool
    next_node: str
    instruction: str
    raw: dict[str, Any]


@dataclass
class LedgerState:
    facts: str = ""
    plan: str = ""
    progress: dict[str, Any] | None = None
    stalls: int = 0
    replans: int = 0


def _history_text(history: Sequence[Message]) -> str:
    return "\n\n".join(f"{item.sender}:\n{item.content}" for item in history)


def _answer(entry: Any, *, field: str, expected: type) -> Any:
    if not isinstance(entry, dict) or "answer" not in entry:
        raise ValueError(f"ledger field {field!r} must be an object containing reason and answer")
    if "reason" not in entry:
        raise ValueError(f"ledger field {field!r} must contain reason")
    value = entry["answer"]
    if expected is bool and not isinstance(value, bool):
        raise ValueError("ledger boolean answer must be true or false")
    if expected is str and not isinstance(value, str):
        raise ValueError("ledger string answer must be text")
    return value


def parse_progress_ledger(text: str, *, candidates: Sequence[str]) -> LedgerDecision:
    """Parse one unmodified progress-ledger response."""

    value = json.loads(str(text))
    if not isinstance(value, dict):
        raise ValueError("progress ledger must be one JSON object")
    required = {
        "is_request_satisfied",
        "is_in_loop",
        "is_progress_being_made",
        "next_speaker",
        "instruction_or_question",
    }
    if set(value) != required:
        raise ValueError("progress ledger keys do not match the declared schema")
    next_node = str(_answer(value["next_speaker"], field="next_speaker", expected=str))
    if next_node not in set(candidates):
        raise ValueError(f"progress ledger selected unknown Node {next_node!r}")
    return LedgerDecision(
        satisfied=bool(
            _answer(
                value["is_request_satisfied"],
                field="is_request_satisfied",
                expected=bool,
            )
        ),
        in_loop=bool(_answer(value["is_in_loop"], field="is_in_loop", expected=bool)),
        making_progress=bool(
            _answer(
                value["is_progress_being_made"],
                field="is_progress_being_made",
                expected=bool,
            )
        ),
        next_node=next_node,
        instruction=str(
            _answer(
                value["instruction_or_question"],
                field="instruction_or_question",
                expected=str,
            )
        ),
        raw=value,
    )


class LedgerOrchestrationProtocol:
    """Execute plan, delegate, monitor, stall, replan, and aggregate operations."""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        controller_node_id: str,
        task: str,
        candidates: Sequence[str],
        team_description: str,
        max_stalls: int = 3,
        max_replans: int = 3,
        parse_attempts: int = 3,
        operation_options: Mapping[str, Mapping[str, Any]] | None = None,
        event_logger: Callable[..., Any] | None = None,
        framework: str = "portable",
    ) -> None:
        self.gateway = gateway
        self.controller_node_id = controller_node_id
        self.task = task
        self.candidates = list(candidates)
        self.team_description = team_description
        self.max_stalls = max(1, int(max_stalls))
        self.max_replans = max(0, int(max_replans))
        self.parse_attempts = max(1, int(parse_attempts))
        self.operation_options = {
            str(kind): dict(options) for kind, options in dict(operation_options or {}).items()
        }
        self.event_logger = event_logger
        self.framework = str(framework)
        self.state = LedgerState()

    async def initialize(self) -> LedgerState:
        facts_prompt = (
            "Analyze the request before delegating work. Return four concise sections: "
            "GIVEN OR VERIFIED FACTS, FACTS TO LOOK UP, FACTS TO DERIVE, and "
            f"EDUCATED GUESSES.\n\nRequest:\n{self.task}"
        )
        facts = await self._invoke_text([{"role": "user", "content": facts_prompt}])
        plan_prompt = self._operation_prompt(
            "plan",
            f"Request:\n{self.task}\n\nTeam:\n{self.team_description}\n\n"
            f"Fact sheet:\n{facts.content}\n\nProduce a concise executable plan.",
            facts=facts.content,
        )
        plan = await self._invoke_text([{"role": "user", "content": plan_prompt}])
        self.state.facts = facts.content
        self.state.plan = plan.content
        self._emit(
            "plan",
            task_ledger={"facts": self.state.facts, "plan": self.state.plan},
            state=self._state_payload(),
        )
        return self.state

    async def decide(self, history: Sequence[Message]) -> LedgerDecision:
        candidates = self.candidates
        prompt = self._progress_prompt(history, candidates=candidates)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        failures: list[str] = []
        for attempt in range(self.parse_attempts):
            reply = await self._invoke_text(messages, json_output=_progress_ledger_schema())
            try:
                decision = parse_progress_ledger(reply.content, candidates=candidates)
            except (json.JSONDecodeError, ValueError) as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 >= self.parse_attempts:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": reply.content},
                        {
                            "role": "user",
                            "content": (
                                "The response did not match the declared JSON schema: "
                                f"{type(exc).__name__}: {exc}. Return exactly one JSON "
                                "object with this complete shape:\n"
                                f"{_PROGRESS_LEDGER_EXAMPLE}\n"
                                "Every top-level field value must be an object containing "
                                "both reason and answer. The example answer values are "
                                "placeholders, not required decisions. Do not use Markdown "
                                "fences or explanatory text."
                            ),
                        },
                    ]
                )
                continue
            stalled = decision.in_loop or not decision.making_progress
            self.state.stalls = self.state.stalls + 1 if stalled else 0
            self.state.progress = dict(decision.raw)
            self._emit(
                "monitor_progress",
                progress_ledger=dict(decision.raw),
                state=self._state_payload(),
            )
            self._emit(
                "detect_stall",
                stalled=stalled,
                in_loop=decision.in_loop,
                making_progress=decision.making_progress,
                consecutive_stalls=self.state.stalls,
                state=self._state_payload(),
            )
            self._emit(
                "delegate",
                next_node=decision.next_node,
                instruction=decision.instruction,
                request_satisfied=decision.satisfied,
                state=self._state_payload(),
            )
            return decision
        raise ValueError(
            "controller failed to produce a valid progress ledger after "
            f"{self.parse_attempts} attempts: {failures!r}"
        )

    def should_replan(self) -> bool:
        return self.state.stalls >= self.max_stalls and self.state.replans < self.max_replans

    async def replan(self, history: Sequence[Message]) -> LedgerState:
        prompt = self._operation_prompt(
            "replan",
            f"Request:\n{self.task}\n\nPrevious facts:\n{self.state.facts}\n\n"
            f"Previous plan:\n{self.state.plan}\n\nRecent work:\n{_history_text(history)}\n\n"
            "Update the fact sheet and produce a revised concise plan that avoids "
            "the observed stall.",
            history=_history_text(history),
        )
        reply = await self._invoke_text([{"role": "user", "content": prompt}])
        self.state.plan = reply.content
        self.state.stalls = 0
        self.state.replans += 1
        self._emit("replan", plan=self.state.plan, state=self._state_payload())
        return self.state

    async def final_answer(self, history: Sequence[Message]) -> GatewayReply:
        prompt = self._operation_prompt(
            "aggregate",
            f"Original request:\n{self.task}\n\nTeam conversation:\n"
            f"{_history_text(history)}\n\nProvide the final answer to the original request. "
            "Address the user directly and include no coordination metadata.",
            history=_history_text(history),
        )
        reply = await self._invoke_text([{"role": "user", "content": prompt}])
        self._emit("aggregate", result=reply.content, state=self._state_payload())
        return reply

    async def _invoke_text(
        self,
        messages: list[dict[str, Any]],
        *,
        json_output: Any = None,
    ) -> GatewayReply:
        return await self.gateway.invoke(
            self.controller_node_id,
            messages,
            controller=True,
            json_output=json_output,
        )

    def _progress_prompt(
        self,
        history: Sequence[Message],
        *,
        candidates: Sequence[str],
    ) -> str:
        names = ", ".join(candidates)
        default = (
            f"Original request:\n{self.task}\n\nTeam:\n{self.team_description}\n\n"
            f"Fact sheet:\n{self.state.facts}\n\nPlan:\n{self.state.plan}\n\n"
            f"Conversation:\n{_history_text(history)}\n\n"
            "Assess progress and choose the next Node. Return exactly one JSON object "
            "with this shape:\n"
            f"{_PROGRESS_LEDGER_EXAMPLE}\n"
            "The boolean values in the example are placeholders; choose the values that "
            "match the current progress. Every top-level field value must remain an object "
            "containing both reason and answer.\n"
            f"NODE must be one of: {names}."
        )
        return self._operation_prompt(
            "monitor_progress",
            default,
            history=_history_text(history),
            candidates=names,
        )

    def _operation_prompt(self, operation: str, default: str, **values: str) -> str:
        prompt = str((self.operation_options.get(operation) or {}).get("prompt") or default)
        replacements = {
            "task": self.task,
            "team_description": self.team_description,
            "facts": self.state.facts,
            "plan": self.state.plan,
            **values,
        }
        for name, value in replacements.items():
            prompt = prompt.replace("{" + name + "}", str(value))
        return prompt

    def _state_payload(self) -> dict[str, Any]:
        return {
            "facts": self.state.facts,
            "plan": self.state.plan,
            "progress": self.state.progress,
            "consecutive_stalls": self.state.stalls,
            "replans": self.state.replans,
        }

    def _emit(self, operation: str, **payload: Any) -> None:
        if self.event_logger is None:
            return
        self.event_logger(
            "coordination.operation.completed",
            operation=operation,
            actor=self.controller_node_id,
            framework=self.framework,
            operation_options=dict(self.operation_options.get(operation) or {}),
            **payload,
        )


__all__ = [
    "LedgerDecision",
    "LedgerOrchestrationProtocol",
    "LedgerState",
    "parse_progress_ledger",
]
