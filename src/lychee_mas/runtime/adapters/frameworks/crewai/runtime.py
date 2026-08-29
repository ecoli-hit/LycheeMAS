"""CrewAI RuntimeAdapter for portable LycheeMAS TeamSpecs."""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

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

# Benchmark execution must not emit third-party telemetry or wait for its exporter.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

_EMPTY_LLM_RESPONSE_ERROR = "Invalid response from LLM call - None or empty."


@REGISTRY.register("runtime", "crewai")
class CrewAIRuntime(ModelGatewayRuntimeBase):
    """Compile explicit TeamSpec facts with CrewAI-local Crew and Flow strategies."""

    name = "crewai"

    async def _execute_team(
        self,
        *,
        team: MASGraph,
        task_text: str,
        query: TaskQuery,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
    ) -> list[Message]:
        plan = dict((team.meta.get("adapter_plans") or {}).get("crewai") or {})
        strategy = str(plan.get("strategy") or "")
        if strategy in {"single_task_crew", "sequential_crew"}:
            return await self._run_crew(
                team,
                task_text,
                gateway,
                tools_by_role,
                strategy,
            )
        if strategy == "orchestration_flow":
            return await self._run_orchestration_flow(
                team,
                task_text,
                gateway,
                tools_by_role,
            )
        return await self._run_flow(
            team,
            task_text,
            gateway,
            tools_by_role,
            strategy,
        )

    def _gateway_llm_type(self):
        from crewai.llms.base_llm import BaseLLM
        from pydantic import PrivateAttr

        class GatewayLLM(BaseLLM):
            _gateway: ModelGateway = PrivateAttr()
            _tools_by_role: dict[str, dict[str, Callable]] = PrivateAttr()
            _controller_roles: set[str] = PrivateAttr()
            _max_tool_iterations_by_role: dict[str, int] = PrivateAttr()
            _on_tool_limit_by_role: dict[str, str] = PrivateAttr()

            def __init__(
                self,
                gateway,
                tools_by_role,
                controller_roles,
                max_tool_iterations_by_role,
                on_tool_limit_by_role,
                task_media,
            ):
                super().__init__(model="lychee-model-gateway")
                self._gateway = gateway
                self._tools_by_role = tools_by_role
                self._controller_roles = set(controller_roles)
                self._max_tool_iterations_by_role = dict(max_tool_iterations_by_role)
                self._on_tool_limit_by_role = dict(on_tool_limit_by_role)
                self._task_media = list(task_media)

            def call(
                self,
                messages: str | list[Any],
                tools: Sequence[Mapping[str, Any]] | None = None,
                callbacks=None,
                available_functions: Mapping[str, Callable] | None = None,
                from_task=None,
                from_agent=None,
                response_model=None,
            ) -> str:
                role = str(getattr(from_agent, "role", "") or "CrewManager")
                payload = (
                    [{"role": "user", "content": messages}]
                    if isinstance(messages, str)
                    else [dict(item) for item in messages]
                )
                if self._task_media:
                    for message in payload:
                        if str(message.get("role") or "") != "user":
                            continue
                        content = message.get("content")
                        if isinstance(content, str):
                            message["content"] = [
                                {"type": "text", "text": content},
                                *deepcopy(self._task_media),
                            ]
                controller = role in self._controller_roles
                reply = self._gateway.invoke_sync(
                    role,
                    payload,
                    tools=self._tools_by_role.get(role, {}),
                    external_tool_schemas=list(tools or []),
                    external_functions=dict(available_functions or {}),
                    controller=controller,
                    json_output=response_model,
                    max_tool_iterations=max(
                        1,
                        int(self._max_tool_iterations_by_role.get(role, 30)),
                    ),
                    on_tool_limit=str(
                        self._on_tool_limit_by_role.get(role) or "finalize"
                    ),
                )
                return reply.content

        return GatewayLLM

    async def _run_crew(
        self,
        team: MASGraph,
        task_text: str,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
        strategy: str,
    ) -> list[Message]:
        from crewai import Agent, Crew, Process, Task

        GatewayLLM = self._gateway_llm_type()
        controller_roles: set[str] = set()
        max_tool_iterations_by_role = {
            node.name: int(node.meta.get("max_tool_iterations", 30)) for node in team.nodes
        }
        on_tool_limit_by_role = {
            node.name: str(node.meta.get("on_tool_limit") or "finalize")
            for node in team.nodes
        }
        llm = GatewayLLM(
            gateway,
            tools_by_role,
            controller_roles,
            max_tool_iterations_by_role,
            on_tool_limit_by_role,
            self._current_task_media,
        )
        agents: list[Any] = []
        tasks: list[Any] = []
        previous: list[Any] = []
        for node in team.nodes:
            agent = Agent(
                role=node.name,
                goal=str(node.meta.get("description") or node.role or node.name),
                backstory=node.system_prompt or f"You are the {node.name} participant.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=max(1, int(node.meta.get("max_tool_iterations", 8))),
            )
            task = Task(
                description=(
                    f"Original task:\n{task_text}\n\n"
                    f"Perform the {node.name} responsibility. Preserve prior evidence and "
                    "follow the role instructions exactly."
                ),
                expected_output=f"A complete {node.name} contribution to the original task.",
                agent=agent,
                context=list(previous),
            )
            agents.append(agent)
            tasks.append(task)
            previous.append(task)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            manager_llm=None,
            verbose=False,
            max_rpm=None,
        )
        try:
            output = await crew.kickoff_async()
        except ValueError as exc:
            if str(exc) != _EMPTY_LLM_RESPONSE_ERROR:
                raise
            # CrewAI requires non-empty task text even when the benchmark result
            # is an artifact (for example, a SWE-bench workspace patch). Preserve
            # the model's empty final content and let the shared ResultProjection
            # apply the benchmark-owned collector instead of fabricating text.
            self.ctx.log_event(
                "framework.output_rejected",
                framework=self.name,
                reason="native_framework_rejected_empty_model_content",
                projection="empty",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return []
        messages = []
        for index, task_output in enumerate(list(getattr(output, "tasks_output", []) or [])):
            if index >= len(team.nodes):
                break
            framework_source = str(getattr(task_output, "agent", "") or "")
            # CrewAI's display label is not a stable TeamSpec Node identity and a
            # hierarchical manager may appear as the apparent author. The ordered
            # tasks were created from the ordered logical Nodes above, so preserve
            # that task provenance as the result-contract source.
            sender = team.nodes[index].name
            content = str(getattr(task_output, "raw", "") or task_output)
            message = Message(
                sender=sender,
                content=content,
                round=index,
                meta={
                    "framework": self.name,
                    "crewai_output": True,
                    "framework_source": framework_source or None,
                    "logical_task_index": index,
                },
            )
            messages.append(message)
            self.ctx.log_event(
                "agent.message.published",
                actor=sender,
                source=sender,
                framework=self.name,
                content=content,
                prompt_tokens=0,
                completion_tokens=0,
                is_tool_event=False,
                origin="crewai_task_output",
                framework_source=framework_source or None,
                logical_task_index=index,
            )
        if not messages:
            submitters = set((team.meta.get("result_contract") or {}).get("submitters") or [])
            final_node = next(
                (
                    node
                    for node in reversed(team.nodes)
                    if str(node.meta.get("node_id") or node.id) in submitters
                ),
                team.nodes[-1],
            )
            messages.append(
                Message(
                    sender=final_node.name,
                    content=str(getattr(output, "raw", "") or output),
                    meta={
                        "framework": self.name,
                        "crewai_output": True,
                        "framework_source": "CrewOutput.raw",
                        "logical_final_task": True,
                    },
                )
            )
        return messages

    async def _run_flow(
        self,
        team: MASGraph,
        task_text: str,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
        strategy: str,
    ) -> list[Message]:
        """Use an official CrewAI Flow for graph and explicit handoff coordination."""

        from crewai.flow.flow import Flow, start

        runtime = self

        class PortableTeamFlow(Flow):
            @start()
            async def execute_team(self):
                return await runtime._flow_schedule(
                    team,
                    task_text,
                    gateway,
                    tools_by_role,
                    strategy,
                )

        value = await PortableTeamFlow().kickoff_async()
        return list(value or [])

    async def _run_orchestration_flow(
        self,
        team: MASGraph,
        task_text: str,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
    ) -> list[Message]:
        """Run orchestration as native CrewAI Flow routes and listeners."""

        from crewai.flow.flow import Flow, listen, or_, router, start
        from pydantic import BaseModel, ConfigDict, Field

        coordination = dict(team.meta["coordination_ir"])
        controller_role = str(team.meta.get("control_node_id") or "")
        if not controller_role:
            raise ValueError("orchestration_flow requires an operation-provider Node")
        by_name = {node.name: node for node in team.nodes}
        candidates = coordination_operation_candidates(
            coordination,
            "delegate",
            node_id=controller_role,
        )
        candidates = [name for name in candidates if name in by_name]
        if not candidates:
            raise ValueError(f"{controller_role} has no explicit delegate candidates")
        operations = dict((coordination.get("operations_by_node") or {}).get(controller_role) or {})
        delegate_options = dict(operations.get("delegate") or {})
        detect_options = dict(operations.get("detect_stall") or {})
        replan_options = dict(operations.get("replan") or {})
        protocol = LedgerOrchestrationProtocol(
            gateway=gateway,
            controller_node_id=controller_role,
            task=task_text,
            candidates=candidates,
            team_description="\n".join(
                f"- {node.name}: {node.meta.get('description') or node.role}" for node in team.nodes
            ),
            max_stalls=int(detect_options.get("window") or self.max_stalls),
            max_replans=int(replan_options.get("max_replans") or 3),
            parse_attempts=int(delegate_options.get("max_attempts") or 3),
            operation_options=operations,
            event_logger=self.ctx.log_event,
            framework=self.name,
        )
        maximum = self._effective_max_turns(coordination, len(team.nodes))
        runtime = self

        class OrchestrationFlowState(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            messages: list[Any] = Field(default_factory=list)
            next_node: str = ""
            next_instruction: str = ""
            stop_requested: bool = False

        route_events = [*candidates, "FINISH"]

        class OrchestrationFlow(Flow[OrchestrationFlowState]):
            @start()
            async def initialize(self) -> None:
                await protocol.initialize()

            @router(or_("initialize", "execute_selected"), emit=route_events)
            async def choose_next(self) -> str:
                history = list(self.state.messages)
                if self.state.stop_requested or len(history) >= maximum:
                    await self._publish_final_if_needed(history)
                    return "FINISH"
                decision = await protocol.decide(history)
                if decision.satisfied:
                    await self._publish_final_if_needed(history)
                    return "FINISH"
                if protocol.should_replan():
                    await protocol.replan(history)
                runtime._emit_control_selection(
                    team,
                    source_node_id=controller_role,
                    target_node_id=decision.next_node,
                    operation="delegate",
                )
                self.state.next_node = decision.next_node
                self.state.next_instruction = decision.instruction
                return decision.next_node

            @listen(or_(*candidates))
            async def execute_selected(self) -> None:
                role = self.state.next_node
                message = await runtime._invoke_node(
                    node=by_name[role],
                    history=list(self.state.messages),
                    task_text=task_text,
                    gateway=gateway,
                    tools=tools_by_role.get(role, {}),
                    system_prompt_suffix=self.state.next_instruction,
                )
                self.state.messages.append(message)
                self.state.stop_requested = runtime._completion_condition_met(team, message)

            @listen("FINISH")
            def finish(self) -> list[Message]:
                return list(self.state.messages)

            async def _publish_final_if_needed(self, history: list[Message]) -> None:
                if result_messages(team, history):
                    return
                message = await runtime._publish_orchestration_result(
                    protocol=protocol,
                    history=history,
                    controller_role=controller_role,
                )
                history.append(message)
                self.state.messages.append(message)

        value = await OrchestrationFlow().kickoff_async()
        if value is None:
            return []
        if not isinstance(value, list):
            raise RuntimeError(
                "CrewAI orchestration Flow must return the final Message list; "
                f"received {type(value).__name__}"
            )
        return list(value)

    async def _flow_schedule(
        self,
        team: MASGraph,
        task_text: str,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
        strategy: str,
    ) -> list[Message]:
        by_name = {node.name: node for node in team.nodes}
        coordination = dict(team.meta["coordination_ir"])
        history: list[Message] = []
        maximum = self._effective_max_turns(coordination, len(team.nodes))
        if strategy == "dependency_flow":
            incoming: dict[str, set[str]] = {name: set() for name in by_name}
            for source, targets in team.edges.items():
                for target in targets:
                    if target in incoming and source in by_name:
                        incoming[target].add(source)
            pending = set(by_name)
            completed: set[str] = set()
            while pending:
                ready = sorted(
                    (name for name in pending if incoming[name] <= completed),
                    key=lambda name: [node.name for node in team.nodes].index(name),
                )
                if not ready:
                    raise ValueError("CrewAI Flow graph contains a cycle")
                replies = await asyncio.gather(
                    *(
                        self._invoke_node(
                            node=by_name[name],
                            history=list(history),
                            task_text=task_text,
                            gateway=gateway,
                            tools=tools_by_role.get(name, {}),
                        )
                        for name in ready
                    )
                )
                history.extend(replies)
                pending.difference_update(ready)
                completed.update(ready)
            return history

        if strategy == "selector_flow":
            coordination = dict(team.meta["coordination_ir"])
            operation = "select_next"
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
            binding: dict[str, Any] = next(
                (
                    item
                    for item in (coordination.get("operations") or {}).get(operation) or []
                    if str(item.get("node_id")) == controller_role
                ),
                {},
            )
            options = dict(binding.get("options") or {})
            attempts = int(options.get("max_attempts", 3))
            finish = dict(options.get("finish") or {})
            finish_allowed = bool(finish.get("allowed", True))
            finish_requires_result = bool(finish.get("requires_result", True))
            mode = str(options.get("mode") or "model")
            while len(history) < maximum:
                if mode == "deterministic":
                    choice = candidates[len(history) % len(candidates)]
                else:
                    rendered = "\n".join(f"{item.sender}: {item.content}" for item in history)
                    may_finish = not finish_requires_result or bool(result_messages(team, history))
                    permitted = [
                        *candidates,
                        *(["FINISH"] if finish_allowed and may_finish else []),
                    ]
                    finish_instruction = (
                        "Reply with exactly one participant name, or FINISH if "
                        "the task is complete."
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
                    choice = None
                    invalid: list[str] = []
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
                        if choice in candidates or (
                            choice == "FINISH" and finish_allowed and may_finish
                        ):
                            break
                        invalid.append(reply.content)
                        selector_messages.extend(
                            [
                                {"role": "assistant", "content": reply.content},
                                {
                                    "role": "user",
                                    "content": (
                                        "That response is not one of the currently "
                                        "permitted values. Reply with exactly one of: "
                                        f"{', '.join(permitted)}. Do not add explanation "
                                        "or punctuation."
                                    ),
                                },
                            ]
                        )
                    else:
                        raise ValueError(
                            f"{controller_role} failed to select a valid participant: {invalid!r}"
                        )
                if choice == "FINISH":
                    self._emit_coordination_decision(
                        team,
                        actor=controller_role,
                        operation=operation,
                        selected_node=choice,
                        decision_source=("deterministic" if mode == "deterministic" else "model"),
                        raw_decision=str(choice),
                    )
                    break
                self._emit_coordination_decision(
                    team,
                    actor=controller_role,
                    operation=operation,
                    selected_node=str(choice),
                    decision_source=("deterministic" if mode == "deterministic" else "model"),
                    raw_decision=str(choice),
                )
                message = await self._invoke_node(
                    node=by_name[str(choice)],
                    history=history,
                    task_text=task_text,
                    gateway=gateway,
                    tools=tools_by_role.get(str(choice), {}),
                )
                history.append(message)
                if self._completion_condition_met(team, message):
                    break
            return history

        current = team.nodes[0]
        while len(history) < maximum:
            targets = list(current.meta.get("handoffs") or team.edges.get(current.name) or [])
            handoff_instruction = (
                "To transfer control, end with exactly HANDOFF: <role>, where <role> "
                f"is one of: {', '.join(targets)}. Publish a final answer instead "
                "when the Team completion contract is satisfied."
                if targets
                else ""
            )
            message = await self._invoke_node(
                node=current,
                history=history,
                task_text=task_text,
                gateway=gateway,
                tools=tools_by_role.get(current.name, {}),
                system_prompt_suffix=handoff_instruction,
            )
            history.append(message)
            if self._completion_condition_met(team, message):
                break
            selected = next(
                (name for name in targets if f"HANDOFF: {name}".lower() in message.content.lower()),
                None,
            )
            # A portable handoff relation is an executable control edge, not only
            # a prompting hint. LangGraph already follows the first legal edge when
            # the model omits an explicit choice; apply the same deterministic
            # fallback here so CrewAI Flow does not silently terminate at a
            # non-submitter node.
            if selected is None:
                fallback = str(
                    operation_options(
                        coordination,
                        "handoff",
                        node_id=current.name,
                    ).get("fallback")
                    or "error"
                )
                if fallback == "next_priority":
                    selected = next((name for name in targets if name in by_name), None)
                elif fallback == "retry":
                    selected = current.name
                elif fallback == "finish" and result_messages(team, history):
                    break
                else:
                    raise ValueError(
                        f"Node {current.name!r} did not produce an explicit legal handoff"
                    )
            if selected is None:
                break
            current = by_name[selected]
        return history
