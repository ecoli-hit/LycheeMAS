"""LangGraph RuntimeAdapter for portable LycheeMAS TeamSpecs."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Message, TaskQuery
from lychee_mas.runtime.adapters.frameworks.base import ModelGatewayRuntimeBase
from lychee_mas.runtime.contracts.runtime import MASGraph
from lychee_mas.runtime.coordination.compiler import (
    operation_candidates as coordination_operation_candidates,
)
from lychee_mas.runtime.coordination.compiler import (
    operation_options,
    parse_next_node_decision,
)
from lychee_mas.runtime.coordination.ledger_orchestration import (
    LedgerOrchestrationProtocol,
)
from lychee_mas.runtime.model.gateway import ModelGateway
from lychee_mas.runtime.results.contract import result_messages


class _State(TypedDict, total=False):
    task_text: str
    messages: Annotated[list[Message], operator.add]
    rounds: Annotated[int, operator.add]
    steps: Annotated[int, operator.add]
    next_role: str
    next_instruction: str


@REGISTRY.register("runtime", "langgraph")
class LangGraphRuntime(ModelGatewayRuntimeBase):
    """Compile coordination to an official ``langgraph.graph.StateGraph``."""

    name = "langgraph"

    async def _execute_team(
        self,
        *,
        team: MASGraph,
        task_text: str,
        query: TaskQuery,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
    ) -> list[Message]:
        from langgraph.graph import END, START, StateGraph

        coordination = dict(team.meta["coordination_ir"])
        plan = dict((team.meta.get("adapter_plans") or {}).get("langgraph") or {})
        strategy = str(plan.get("strategy") or "")
        nodes = list(team.nodes)
        by_name = {node.name: node for node in nodes}
        maximum = self._effective_max_turns(coordination, len(nodes))
        builder = StateGraph(_State)

        async def participant(state: _State, *, role: str) -> _State:
            history = list(state.get("messages") or [])
            handoffs = list(by_name[role].meta.get("handoffs") or [])
            handoff_instruction = (
                "To transfer control, end with exactly HANDOFF: <role>, where <role> "
                f"is one of: {', '.join(handoffs)}. Publish a final answer instead "
                "when the Team completion contract is satisfied."
                if strategy == "conditional_handoff_edges" and handoffs
                else ""
            )
            node_instruction = str(state.get("next_instruction") or "")
            suffix = "\n\n".join(item for item in (handoff_instruction, node_instruction) if item)
            message = await self._invoke_node(
                node=by_name[role],
                history=history,
                task_text=task_text,
                gateway=gateway,
                tools=tools_by_role.get(role, {}),
                system_prompt_suffix=suffix,
            )
            updates: _State = {
                "messages": [message],
                "steps": 1,
            }
            if role == nodes[-1].name:
                updates["rounds"] = 1
            return updates

        for node in nodes:

            async def execute(state: _State, role=node.name):
                return await participant(state, role=role)

            builder.add_node(node.name, execute)

        if strategy in {"single_node", "deterministic_dependency_chain"}:
            builder.add_edge(START, nodes[0].name)
            for left, right in zip(nodes, nodes[1:]):
                builder.add_edge(left.name, right.name)

            def after_last(state: _State) -> str:
                if self._should_stop(team, state, maximum):
                    return "done"
                return "repeat"

            builder.add_conditional_edges(
                nodes[-1].name,
                after_last,
                {"repeat": nodes[0].name, "done": END},
            )
        elif strategy == "explicit_dependency_graph":
            incoming: dict[str, list[str]] = {name: [] for name in by_name}
            for source, targets in team.edges.items():
                for target in targets:
                    if source in by_name and target in by_name:
                        incoming[target].append(source)
            roots = [name for name, sources in incoming.items() if not sources]
            if not roots:
                raise ValueError("LangGraph graph coordination requires at least one root")
            for root in roots:
                builder.add_edge(START, root)
            for target, sources in incoming.items():
                if len(sources) == 1:
                    builder.add_edge(sources[0], target)
                elif len(sources) > 1:
                    builder.add_edge(sources, target)
            sinks = [name for name in by_name if not team.edges.get(name)]
            for sink in sinks:
                builder.add_edge(sink, END)
        elif strategy == "conditional_handoff_edges":
            builder.add_edge(START, nodes[0].name)
            for node in nodes:
                candidates = list(node.meta.get("handoffs") or team.edges.get(node.name) or [])
                mapping = {name: name for name in candidates if name in by_name}
                mapping["done"] = END

                mapping.setdefault(node.name, node.name)

                def route(
                    state: _State,
                    allowed=list(candidates),
                    source_role=node.name,
                ) -> str:
                    if self._should_stop(team, state, maximum):
                        return "done"
                    text = (state.get("messages") or [])[-1].content
                    for name in allowed:
                        if f"HANDOFF: {name}".lower() in text.lower():
                            return name
                    fallback = str(
                        operation_options(
                            coordination,
                            "handoff",
                            node_id=source_role,
                        ).get("fallback")
                        or "error"
                    )
                    if fallback == "next_priority":
                        return next(iter(allowed), "done")
                    if fallback == "retry":
                        return source_role
                    if fallback == "finish" and self._may_finish(
                        team, list(state.get("messages") or [])
                    ):
                        return "done"
                    raise ValueError(
                        f"Node {source_role!r} did not produce an explicit legal handoff"
                    )

                builder.add_conditional_edges(node.name, route, mapping)
        elif strategy in {"supervisor_conditional_edges", "supervisor_orchestration"}:
            operation = "select_next" if strategy == "supervisor_conditional_edges" else "delegate"
            controller_role = str(team.meta.get("control_node_id") or "")
            if not controller_role:
                raise ValueError(f"{strategy} requires an explicit operation-provider Node")
            candidates = coordination_operation_candidates(
                coordination,
                operation,
                node_id=controller_role,
            )
            candidates = [name for name in candidates if name in by_name]
            if not candidates:
                raise ValueError(f"{controller_role} has no explicit {operation} candidates")
            operation_binding: dict[str, Any] = next(
                (
                    item
                    for item in (coordination.get("operations") or {}).get(operation) or []
                    if str(item.get("node_id")) == controller_role
                ),
                {},
            )
            selector_options = dict(operation_binding.get("options") or {})
            attempts = int(selector_options.get("max_attempts", 3))
            finish = dict(selector_options.get("finish") or {})
            finish_allowed = bool(finish.get("allowed", strategy == "supervisor_conditional_edges"))
            finish_requires_result = bool(finish.get("requires_result", True))
            mode = str(selector_options.get("mode") or "model")
            protocol = None
            if strategy == "supervisor_orchestration":
                detect_options = next(
                    (
                        dict(item.get("options") or {})
                        for item in (coordination.get("operations") or {}).get("detect_stall") or []
                        if str(item.get("node_id")) == controller_role
                    ),
                    {},
                )
                replan_options = next(
                    (
                        dict(item.get("options") or {})
                        for item in (coordination.get("operations") or {}).get("replan") or []
                        if str(item.get("node_id")) == controller_role
                    ),
                    {},
                )
                team_description = "\n".join(
                    f"- {node.name}: {node.meta.get('description') or node.role}" for node in nodes
                )
                protocol = LedgerOrchestrationProtocol(
                    gateway=gateway,
                    controller_node_id=controller_role,
                    task=task_text,
                    candidates=candidates,
                    team_description=team_description,
                    max_stalls=int(detect_options.get("window") or self.max_stalls),
                    max_replans=int(replan_options.get("max_replans") or 3),
                    parse_attempts=attempts,
                    operation_options=dict(
                        (coordination.get("operations_by_node") or {}).get(controller_role) or {}
                    ),
                    event_logger=self.ctx.log_event,
                    framework=self.name,
                )
                await protocol.initialize()

            async def supervisor(state: _State) -> _State:
                if self._should_stop(team, state, maximum):
                    history = list(state.get("messages") or [])
                    if protocol is not None and not result_messages(team, history):
                        final = await self._publish_orchestration_result(
                            protocol=protocol,
                            history=history,
                            controller_role=controller_role,
                        )
                        return {"messages": [final], "next_role": "FINISH"}
                    return {"next_role": "FINISH"}
                history = list(state.get("messages") or [])
                if protocol is not None:
                    decision = await protocol.decide(history)
                    if decision.satisfied:
                        message = await self._publish_orchestration_result(
                            protocol=protocol,
                            history=history,
                            controller_role=controller_role,
                        )
                        return {"messages": [message], "next_role": "FINISH"}
                    if protocol.should_replan():
                        await protocol.replan(history)
                    self._emit_control_selection(
                        team,
                        source_node_id=controller_role,
                        target_node_id=decision.next_node,
                        operation="delegate",
                    )
                    return {
                        "next_role": decision.next_node,
                        "next_instruction": decision.instruction,
                    }
                if mode == "deterministic":
                    choice = candidates[int(state.get("steps", 0)) % len(candidates)]
                    self._emit_coordination_decision(
                        team,
                        actor=controller_role,
                        operation=operation,
                        selected_node=choice,
                        decision_source="deterministic",
                        raw_decision=choice,
                    )
                    return {"next_role": choice}
                rendered = "\n".join(f"{item.sender}: {item.content}" for item in history)
                may_finish = not finish_requires_result or self._may_finish(team, history)
                permitted = [*candidates, *(["FINISH"] if finish_allowed and may_finish else [])]
                finish_instruction = (
                    "Reply with exactly one participant name, or FINISH if the task is complete."
                    if finish_allowed and may_finish
                    else (
                        "Reply with exactly one participant name. FINISH is not "
                        "permitted until an approved result submitter has published "
                        "a non-empty result."
                        if finish_allowed
                        else "Reply with exactly one participant name."
                    )
                )
                prompt = (
                    f"Original task:\n{task_text}\n\nConversation:\n{rendered}\n\n"
                    f"Choose the next participant from: {', '.join(candidates)}. "
                    + finish_instruction
                )
                selector_messages = [{"role": "user", "content": prompt}]
                invalid_replies: list[str] = []
                for attempt in range(attempts):
                    reply = await gateway.invoke(
                        controller_role,
                        selector_messages,
                        controller=True,
                    )
                    choice = parse_next_node_decision(
                        reply.content,
                        candidates=candidates,
                    )
                    if choice == "FINISH" and finish_allowed and may_finish:
                        self._emit_coordination_decision(
                            team,
                            actor=controller_role,
                            operation=operation,
                            selected_node=choice,
                            decision_source="model",
                            raw_decision=reply.content,
                        )
                        return {"next_role": choice}
                    if choice is not None and choice in candidates:
                        self._emit_coordination_decision(
                            team,
                            actor=controller_role,
                            operation=operation,
                            selected_node=choice,
                            decision_source="model",
                            raw_decision=reply.content,
                        )
                        return {"next_role": choice}
                    invalid_replies.append(reply.content)
                    if attempt + 1 < attempts:
                        finish_constraint = (
                            " FINISH is not permitted until an approved result submitter "
                            "has published a non-empty result."
                            if choice == "FINISH"
                            else ""
                        )
                        selector_messages.extend(
                            [
                                {"role": "assistant", "content": reply.content},
                                {
                                    "role": "user",
                                    "content": (
                                        "That response is not one of the permitted values. "
                                        f"Reply with exactly one of: {', '.join(permitted)}. "
                                        "Do not add explanation or punctuation."
                                        f"{finish_constraint}"
                                    ),
                                },
                            ]
                        )
                rendered_invalid = ", ".join(repr(value) for value in invalid_replies)
                raise ValueError(
                    f"{controller_role} failed to select a valid participant after "
                    f"{attempts} attempts: {rendered_invalid}"
                )

            builder.add_node("__controller__", supervisor)
            builder.add_edge(START, "__controller__")

            def selected(state: _State) -> str:
                return str(state.get("next_role") or "FINISH")

            builder.add_conditional_edges(
                "__controller__",
                selected,
                {**{name: name for name in by_name}, "FINISH": END},
            )
            for node in nodes:
                builder.add_edge(node.name, "__controller__")
        else:  # pragma: no cover - normalized before adapter dispatch
            raise ValueError(f"unsupported LangGraph adapter strategy {strategy!r}")

        compiled = builder.compile()
        result = await compiled.ainvoke(
            {
                "task_text": task_text,
                "messages": [],
                "rounds": 0,
                "steps": 0,
                "next_instruction": "",
            },
            config={"recursion_limit": max(25, maximum * 3 + 5)},
        )
        return list(result.get("messages") or [])

    @staticmethod
    def _may_finish(team: MASGraph, history: list[Message]) -> bool:
        eligible = result_messages(team, history)
        return any(str(message.content or "").strip() for message in eligible)

    def _should_stop(self, team: MASGraph, state: _State, maximum: int) -> bool:
        if int(state.get("steps", 0)) >= maximum:
            return True
        messages = list(state.get("messages") or [])
        if not messages:
            return False
        return self._completion_condition_met(team, messages[-1])
