"""Lifecycle shell for adapters that invoke models through ModelGateway."""

from __future__ import annotations

import inspect
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from lychee_mas.core.types import Answer, Message, TaskQuery, Trajectory
from lychee_mas.runtime.conformance import evaluate_runtime_conformance
from lychee_mas.runtime.contracts.node import NodeInput, NodeOutput
from lychee_mas.runtime.contracts.runtime import BaseRuntime, MASGraph
from lychee_mas.runtime.events.store import exception_record
from lychee_mas.runtime.model.gateway import ModelGateway
from lychee_mas.runtime.results.contract import result_messages
from lychee_mas.runtime.results.projection import project_result
from lychee_mas.runtime.state import TeamMemoryRuntime
from lychee_mas.runtime.tools.code_execution import CodeExecutorFactory
from lychee_mas.runtime.tools.portable import portable_web_tools, portable_workspace_tools
from lychee_mas.runtime.workspaces.service import WorkspaceService

from .bindings import build_framework_binding_report


class ModelGatewayRuntimeBase(BaseRuntime):
    """Run a TeamSpec with shared services and a normalized ModelGateway."""

    name = "framework"

    def __init__(
        self,
        backend=None,
        ctx=None,
        max_new_tokens: int = 256,
        max_rounds: int = 2,
        model_id: str = "model",
        max_turns: Optional[int] = None,
        max_model_calls_per_case: Optional[int] = None,
        max_stalls: int = 3,
        work_root: str | Path = "runs/lychee_tool_workspaces",
        code_executor: str = "docker",
        docker_image: str = "python:3.11-slim",
        code_timeout: int = 60,
        web_headless: bool = True,
        save_screenshots: bool = False,
        web_proxy_url: Optional[str] = None,
        container_proxy_url: Optional[str] = None,
        proxy_no_proxy: str = "127.0.0.1,localhost,::1",
        trace_model_calls: Optional[bool] = None,
        backend_resolver=None,
        model_resolver=None,
        control_deployment_id: Optional[str] = None,
        max_new_tokens_resolver=None,
        control_max_new_tokens: Optional[int] = None,
        invocation_overrides_resolver=None,
        control_invocation_overrides: Optional[dict[str, Any]] = None,
        memory_method: str = "none",
        framework_options: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.ctx = ctx
        self.max_new_tokens = int(max_new_tokens)
        self.max_rounds = int(max_rounds)
        self.model_id = model_id
        self.max_turns = max_turns
        self.max_model_calls_per_case = max_model_calls_per_case
        self.max_stalls = int(max_stalls)
        self.work_root = Path(work_root)
        self.code_executor = code_executor
        self.docker_image = docker_image
        self.code_timeout = int(code_timeout)
        self.web_headless = web_headless
        self.save_screenshots = save_screenshots
        self.web_proxy_url = web_proxy_url
        self.container_proxy_url = container_proxy_url
        self.proxy_no_proxy = proxy_no_proxy
        self.trace_model_calls = trace_model_calls
        self.backend_resolver = backend_resolver
        self.model_resolver = model_resolver
        self.control_deployment_id = control_deployment_id
        self.max_new_tokens_resolver = max_new_tokens_resolver
        self.control_max_new_tokens = control_max_new_tokens
        self.invocation_overrides_resolver = invocation_overrides_resolver
        self.control_invocation_overrides = dict(control_invocation_overrides or {})
        self.memory_method = str(memory_method)
        self.framework_options = dict(framework_options or {})
        self.workspace_service = WorkspaceService(self.work_root)
        self.code_executor_factory = CodeExecutorFactory(
            executor=self.code_executor,
            docker_image=self.docker_image,
            timeout_s=self.code_timeout,
            container_proxy_url=self.container_proxy_url,
            proxy_no_proxy=self.proxy_no_proxy,
        )
        self._executor_consumed_messages: dict[str, set[tuple[str, int, int]]] = {}
        self._current_task_media: list[dict[str, Any]] = []

    def _set_task_media(self, query: TaskQuery) -> None:
        """Freeze model-visible non-text task parts for this Trial."""

        content = query.meta.get("multimodal_content")
        self._current_task_media = [
            deepcopy(part)
            for part in (content if isinstance(content, list) else [])
            if isinstance(part, dict) and str(part.get("type") or "") in {"image", "image_url"}
        ]

    def _model_user_content(self, text: str) -> str | list[dict[str, Any]]:
        """Compose one user message without dropping benchmark media."""

        if not self._current_task_media:
            return text
        return [{"type": "text", "text": text}, *deepcopy(self._current_task_media)]

    async def _publish_orchestration_result(
        self,
        *,
        protocol: Any,
        history: list[Message],
        controller_role: str,
    ) -> Message:
        """Run the TeamSpec aggregate operation and publish its logical result."""

        final = await protocol.final_answer(history)
        message = Message(
            sender=controller_role,
            content=final.content,
            round=len(history),
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            meta={
                "framework": self.name,
                "reasoning_content": final.reasoning_content,
                "orchestration_final": True,
            },
        )
        self.ctx.log_event(
            "agent.message.published",
            actor=controller_role,
            source=controller_role,
            framework=self.name,
            content=message.content,
            prompt_tokens=message.prompt_tokens,
            completion_tokens=message.completion_tokens,
            is_tool_event=False,
            orchestration_final=True,
        )
        return message

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if (self.backend is None and self.backend_resolver is None) or self.ctx is None:
            raise ValueError(f"{type(self).__name__}.run requires backend and RoutingContext")
        if self.memory_method != "none":
            raise ValueError(
                f"{self.name} currently supports method=none only; text/latent memory "
                "injection remains an AutoGen InjectionClient capability"
            )
        coordination = dict(team.meta.get("coordination_ir") or {})
        support = build_framework_binding_report(
            self.name,
            coordination,
            dict(team.meta.get("team_spec") or {}),
        )
        if not support["supported"]:
            raise ValueError(str(support["reason"]))
        self.ctx.configure_graph(team)
        self.ctx.max_model_calls_per_case = self.max_model_calls_per_case
        self.ctx.trace_model_calls = (
            True if self.trace_model_calls is None else bool(self.trace_model_calls)
        )
        for node in team.nodes:
            deployment_id = str(node.meta.get("deployment_id") or "")
            self.ctx.deployment_of_role[node.name] = deployment_id
            self.ctx.model_of_role[node.name] = (
                self.model_resolver(deployment_id) if self.model_resolver else self.model_id
            )

        from lychee_mas.eval.benchmarks import BENCHMARKS

        benchmark = BENCHMARKS.for_query(self.ctx.task, query.meta)
        uses_tools = self._uses_tools(team)
        workspace: Path | None = None
        task_text = query.question
        copied_files: list[str] = []
        workspace_info: dict[str, Any] = {}
        code_executor = None
        resources: list[Any] = []
        runtime_id = None
        group_id = None
        started = time.monotonic()
        self._current_tool_requests: list[dict[str, Any]] = []
        self._current_tool_executions: list[dict[str, Any]] = []
        self._executor_consumed_messages.clear()
        self._set_task_media(query)
        try:
            runtime_id = self.ctx.log_event(
                "runtime.started",
                runtime=self.name,
                framework=self.name,
                team=team.meta.get("team"),
                coordination=coordination,
                framework_implementation=support.get("implementation"),
                framework_binding_report=support,
                max_turns=self._effective_max_turns(coordination, len(team.nodes)),
                max_rounds=self.max_rounds,
                max_model_calls_per_case=self.max_model_calls_per_case,
                uses_tools=uses_tools,
            )
            self.ctx.current_runtime_event_id = runtime_id
            self.ctx.team_memory = TeamMemoryRuntime(
                dict(team.meta.get("team_spec") or {}),
                log_event=self.ctx.log_event,
            )
            if uses_tools:
                workspace, task_text, copied_files, workspace_info = self._prepare_workspace(
                    query, benchmark
                )
                self.ctx.log_event(
                    "workspace.prepared",
                    parent_event_id=runtime_id,
                    workspace=str(workspace),
                    copied_files=copied_files,
                    task_text=task_text,
                    **workspace_info,
                )
                if self._needs_code_executor(team):
                    code_executor = self._build_code_executor(workspace)
                    event_id = self.ctx.log_event(
                        "code_executor.started",
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        timeout_s=self.code_timeout,
                        workspace=str(workspace),
                    )
                    await code_executor.start()
                    self.ctx.log_event(
                        "code_executor.ready",
                        operation_id=event_id,
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        workspace=str(workspace),
                    )
            tools_by_role = self._tools_by_role(
                team,
                benchmark=benchmark,
                query=query,
                workspace=workspace,
                code_executor=code_executor,
                resources=resources,
            )
            gateway = ModelGateway(
                ctx=self.ctx,
                backend=self.backend,
                backend_resolver=self.backend_resolver,
                model_resolver=self.model_resolver,
                max_new_tokens=self.max_new_tokens,
                max_new_tokens_resolver=self.max_new_tokens_resolver,
                invocation_overrides_resolver=self.invocation_overrides_resolver,
                control_deployment_id=self.control_deployment_id,
                control_max_new_tokens=self.control_max_new_tokens,
                control_invocation_overrides=self.control_invocation_overrides,
                model_context_policy=team.meta.get("model_context"),
                max_inline_tool_result_chars=int(
                    ((team.meta.get("communication") or {}).get("tool_results") or {}).get(
                        "max_inline_chars", 50000
                    )
                ),
                framework=self.name,
            )
            group_id = self.ctx.log_event(
                "group_chat.started",
                framework=self.name,
                coordination=coordination,
                implementation=support.get("implementation"),
                framework_binding_report=support,
                member_nodes=[node.name for node in team.nodes],
                operation_bindings=list(team.meta.get("operation_bindings") or []),
                context_visibility=self.ctx.context_visibility,
            )
            self.ctx.current_group_chat_event_id = group_id
            messages = await self._execute_team(
                team=team,
                task_text=task_text,
                query=query,
                gateway=gateway,
                tools_by_role=tools_by_role,
            )
            self.ctx.log_event(
                "group_chat.completed",
                operation_id=group_id,
                framework=self.name,
                message_count=len(messages),
                stop_reason="coordination_completed",
            )
        except BaseException as exc:
            self.ctx.log_event(
                "group_chat.failed",
                operation_id=group_id,
                framework=self.name,
                **exception_record(exc),
            )
            self.ctx.log_event(
                "runtime.failed",
                operation_id=runtime_id,
                framework=self.name,
                **exception_record(exc),
            )
            raise
        finally:
            self.ctx.current_group_chat_event_id = None
            for resource in resources:
                close = getattr(resource, "close", None)
                if close is not None:
                    value = close()
                    if inspect.isawaitable(value):
                        await value
            if code_executor is not None:
                await code_executor.stop()
                self.ctx.log_event(
                    "code_executor.stopped",
                    executor=self.code_executor,
                    docker_image=self.docker_image,
                    workspace=str(workspace) if workspace else None,
                )
            self.ctx.current_runtime_event_id = None
            self._current_task_media = []

        elapsed = round(time.monotonic() - started, 6)
        projected = project_result(
            benchmark=benchmark,
            task=self.ctx.task,
            team=team,
            query=query,
            messages=messages,
            workspace=workspace,
            tool_requests=self._current_tool_requests,
        )
        self.ctx.log_event(
            (
                "result_contract.validated"
                if projected.validation.valid
                else "result_contract.failed"
            ),
            parent_event_id=runtime_id,
            **projected.validation.to_dict(),
        )
        conformance = evaluate_runtime_conformance(
            framework=self.name,
            team=team,
            result_validation=projected.validation,
            tool_requests=self._current_tool_requests,
            tool_executions=self._current_tool_executions,
            observation_coverage={
                "trial_lifecycle": "full",
                "model_calls": "full",
                "tool_calls": "full",
                "agent_messages": "full",
                "team_graph_lifecycle": "full",
            },
        )
        self.ctx.log_event(
            "runtime.conformance_evaluated",
            parent_event_id=runtime_id,
            **conformance.to_dict(),
        )
        trajectory = Trajectory(
            task_id=query.id,
            messages=messages,
            final_answer=Answer(
                content=projected.content,
                source=projected.source,
                meta={"n_messages": len(messages), "case_wall_time_s": elapsed},
            ),
            meta={
                "runtime": self.name,
                "framework": self.name,
                "coordination": coordination,
                "framework_implementation": support.get("implementation"),
                "task": self.ctx.task,
                "team": team.meta.get("team"),
                "workspace": str(workspace) if workspace else None,
                "copied_files": copied_files,
                "workspace_info": workspace_info,
                "case_wall_time_s": elapsed,
                "model_calls_started": self.ctx.model_calls_started,
                "decisions": list(self.ctx.decisions),
                "tool_requests": self._current_tool_requests,
                "tool_executions": self._current_tool_executions,
                "tool_request_count": len(self._current_tool_requests),
                "tool_execution_count": len(self._current_tool_executions),
                "tool_error_count": sum(
                    bool(item.get("is_error")) for item in self._current_tool_executions
                ),
                "result_submitters": list(
                    (team.meta.get("result_contract") or {}).get("submitters") or []
                ),
                "eligible_result_message_count": len(projected.eligible_messages),
                "result_contract": projected.validation.to_dict(),
                "runtime_conformance": conformance.to_dict(),
                "team_memory": (self.ctx.team_memory.snapshot() if self.ctx.team_memory else {}),
            },
        )
        trajectory.candidates = (
            [trajectory.final_answer] if trajectory.final_answer is not None else []
        )
        for message in messages:
            self._emit(message)
        self.ctx.log_event(
            "runtime.completed",
            operation_id=runtime_id,
            framework=self.name,
            message_count=len(messages),
            model_calls_started=self.ctx.model_calls_started,
            case_wall_time_s=elapsed,
            duration_s=elapsed,
        )
        return trajectory

    async def _execute_team(
        self,
        *,
        team: MASGraph,
        task_text: str,
        query: TaskQuery,
        gateway: ModelGateway,
        tools_by_role: dict[str, dict[str, Callable]],
    ) -> list[Message]:
        raise NotImplementedError

    async def _invoke_role(
        self,
        *,
        node: Any,
        history: list[Message],
        task_text: str,
        gateway: ModelGateway,
        tools: dict[str, Callable],
        system_prompt_suffix: str = "",
    ) -> Message:
        invocation_id = uuid.uuid4().hex
        policies = dict(node.meta.get("message_policies") or {})
        visible_history = self._visible_history(node, history, policies)
        memory_recall = (
            self.ctx.team_memory.recall(
                str(node.meta.get("node_id") or node.id),
                task=task_text,
                node_input="\n".join(str(item.content or "") for item in visible_history),
                invocation_id=invocation_id,
            )
            if self.ctx.team_memory is not None
            else None
        )
        node_input = NodeInput(
            node_id=str(node.meta.get("node_id") or node.id),
            invocation_id=invocation_id,
            task=task_text,
            messages=tuple(visible_history),
            state={
                "memory": {
                    "memory_ids": list(memory_recall.memory_ids),
                    "item_count": memory_recall.item_count,
                }
            }
            if memory_recall and not memory_recall.empty
            else {},
            metadata={"framework": self.name, "runtime_role": node.name},
        )
        invocation_event_id = self.ctx.log_event(
            "node_invocation.started",
            node_id=node_input.node_id,
            node_kind="executable",
            execution_kind=str(node.meta.get("execution_kind") or "model"),
            invocation_id=invocation_id,
            runtime_role=node.name,
            input_message_count=len(node_input.messages),
            input_ports=[
                "task",
                *(["messages"] if node_input.messages else []),
                *(["memory"] if node_input.state.get("memory") else []),
            ],
        )
        backend_messages: list[dict[str, Any]] = []
        system_prompt = "\n\n".join(
            item
            for item in (
                node.system_prompt,
                system_prompt_suffix,
                *(memory_recall.model_context if memory_recall else ()),
            )
            if item
        )
        if system_prompt:
            backend_messages.append({"role": "system", "content": system_prompt})
        effective_task = "\n\n".join(
            [node_input.task, *(memory_recall.node_input if memory_recall else ())]
        )
        backend_messages.append(
            {"role": "user", "content": self._model_user_content(effective_task)}
        )
        visible_history = list(node_input.messages)
        backend_messages.extend(
            {
                "role": "assistant",
                "name": item.sender,
                "content": item.content,
            }
            for item in visible_history
        )
        try:
            reply = await gateway.invoke(
                node.name,
                backend_messages,
                tools=tools,
                max_tool_iterations=int(node.meta.get("max_tool_iterations", 30)),
                on_tool_limit=str(node.meta.get("on_tool_limit") or "finalize"),
            )
        except BaseException as exc:
            self.ctx.log_event(
                "node_invocation.failed",
                operation_id=invocation_event_id,
                node_id=node_input.node_id,
                node_kind="executable",
                execution_kind=str(node.meta.get("execution_kind") or "model"),
                invocation_id=invocation_id,
                runtime_role=node.name,
                **exception_record(exc),
            )
            raise
        self._current_tool_requests.extend(reply.tool_requests)
        self._current_tool_executions.extend(reply.tool_executions)
        message = Message(
            sender=node.name,
            content=reply.content,
            round=len(history),
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            meta={
                "framework": self.name,
                "reasoning_content": reply.reasoning_content,
                "node_invocation_id": invocation_id,
            },
        )
        node_output = NodeOutput(
            node_id=node_input.node_id,
            invocation_id=invocation_id,
            status="completed",
            message=message,
            control_signals=("completed",),
            data_by_port={"response": message.content},
            metadata={"framework": self.name},
        )
        if self.ctx.team_memory is not None:
            self.ctx.team_memory.write_node_output(
                node_output.node_id,
                message.content,
                invocation_id=invocation_id,
                metadata={"framework": self.name, "runtime_role": node.name},
            )
        self.ctx.log_event(
            "node_invocation.completed",
            operation_id=invocation_event_id,
            node_id=node_output.node_id,
            node_kind="executable",
            execution_kind=str(node.meta.get("execution_kind") or "model"),
            invocation_id=node_output.invocation_id,
            runtime_role=node.name,
            status=node_output.status,
            output_ports=sorted(node_output.data_by_port),
            control_signals=list(node_output.control_signals),
        )
        self._emit_response_data_events(node, node_output)
        self.ctx.log_event(
            "agent.message.published",
            actor=node.name,
            source=node.name,
            framework=self.name,
            content=message.content,
            prompt_tokens=message.prompt_tokens,
            completion_tokens=message.completion_tokens,
            is_tool_event=False,
            node_invocation_id=node_output.invocation_id,
        )
        return node_output.message or message

    async def _invoke_node(
        self,
        *,
        node: Any,
        history: list[Message],
        task_text: str,
        gateway: ModelGateway,
        tools: dict[str, Callable],
        system_prompt_suffix: str = "",
    ) -> Message:
        """Invoke a model Node or execute a deterministic executor Node."""

        if str(node.meta.get("execution_kind") or "model") != "executor":
            return await self._invoke_role(
                node=node,
                history=history,
                task_text=task_text,
                gateway=gateway,
                tools=tools,
                system_prompt_suffix=system_prompt_suffix,
            )
        return await self._invoke_executor_node(node=node, history=history, tools=tools)

    async def _invoke_executor_node(
        self,
        *,
        node: Any,
        history: list[Message],
        tools: dict[str, Callable],
    ) -> Message:
        """Execute fenced code visible through the Node's Message Data Edges."""

        policies = dict(node.meta.get("message_policies") or {})
        visible_history = self._visible_history(node, history, policies)
        consumed = self._executor_consumed_messages.setdefault(node.name, set())
        pending_history = [
            item for item in visible_history if self._message_key(item) not in consumed
        ]
        invocation_id = uuid.uuid4().hex
        invocation_event_id = self.ctx.log_event(
            "node_invocation.started",
            node_id=str(node.meta.get("node_id") or node.id),
            node_kind="executable",
            execution_kind="executor",
            invocation_id=invocation_id,
            runtime_role=node.name,
            input_message_count=len(pending_history),
            input_ports=["messages"] if pending_history else [],
        )
        try:
            supported = "python|py|python3"
            pattern = re.compile(rf"```\s*({supported})?\s*\n([\s\S]*?)```", re.IGNORECASE)
            blocks = [
                code
                for item in pending_history
                for _language, code in pattern.findall(str(item.content or ""))
            ]
            consumed.update(self._message_key(item) for item in pending_history)
            executor = tools.get("python_code")
            if not blocks:
                content = (
                    "No executable Python code block was found in the messages visible "
                    "to this executor Node."
                )
            elif executor is None:
                raise RuntimeError(f"executor Node {node.name!r} has no python_code tool")
            else:
                outputs: list[str] = []
                for code in blocks:
                    outputs.append(str(await executor(code)))
                content = "\n\n".join(outputs)
        except BaseException as exc:
            self.ctx.log_event(
                "node_invocation.failed",
                operation_id=invocation_event_id,
                node_id=str(node.meta.get("node_id") or node.id),
                node_kind="executable",
                execution_kind="executor",
                invocation_id=invocation_id,
                runtime_role=node.name,
                **exception_record(exc),
            )
            raise
        message = Message(
            sender=node.name,
            content=content,
            round=len(history),
            meta={
                "framework": self.name,
                "executor_node": True,
                "code_blocks": len(blocks),
                "node_invocation_id": invocation_id,
            },
        )
        node_output = NodeOutput(
            node_id=str(node.meta.get("node_id") or node.id),
            invocation_id=invocation_id,
            status="completed",
            message=message,
            control_signals=("completed",),
            data_by_port={"response": message.content},
            metadata={"framework": self.name, "code_blocks": len(blocks)},
        )
        if self.ctx.team_memory is not None:
            self.ctx.team_memory.write_node_output(
                node_output.node_id,
                message.content,
                invocation_id=invocation_id,
                metadata={"framework": self.name, "runtime_role": node.name},
            )
        self.ctx.log_event(
            "node_invocation.completed",
            operation_id=invocation_event_id,
            node_id=node_output.node_id,
            node_kind="executable",
            execution_kind="executor",
            invocation_id=invocation_id,
            runtime_role=node.name,
            status=node_output.status,
            output_ports=sorted(node_output.data_by_port),
            control_signals=list(node_output.control_signals),
            code_block_count=len(blocks),
        )
        self._emit_response_data_events(node, node_output)
        self.ctx.log_event(
            "agent.message.published",
            actor=node.name,
            source=node.name,
            framework=self.name,
            content=message.content,
            prompt_tokens=0,
            completion_tokens=0,
            is_tool_event=True,
            executor_node=True,
            code_block_count=len(blocks),
        )
        return message

    @staticmethod
    def _message_key(message: Message) -> tuple[str, int, int]:
        return (str(message.sender), int(message.round), hash(str(message.content)))

    def _emit_response_data_events(self, node: Any, output: NodeOutput) -> None:
        if "response" not in output.data_by_port:
            return
        data_item_id = uuid.uuid4().hex
        self.ctx.log_event(
            "data_item.created",
            data_item_id=data_item_id,
            semantic_type="message",
            source_node_id=output.node_id,
            source_port="response",
            producer_invocation_id=output.invocation_id,
            value=output.data_by_port["response"],
        )
        for edge in node.meta.get("store_access") or []:
            if edge.get("edge_type") != "data":
                continue
            if str((edge.get("source") or {}).get("node") or "") != output.node_id:
                continue
            if str((edge.get("source") or {}).get("port") or "") != "response":
                continue
            operation = str((edge.get("data") or {}).get("operation") or "transfer")
            self.ctx.log_event(
                "data_edge.transferred",
                edge_id=edge.get("id"),
                data_item_id=data_item_id,
                source=edge.get("source"),
                target=edge.get("target"),
                operation=operation,
                producer_invocation_id=output.invocation_id,
            )
            if operation in {"initialize", "write", "append", "merge", "commit"}:
                self.ctx.log_event(
                    "data_item.stored",
                    data_item_id=data_item_id,
                    store_node_id=(edge.get("target") or {}).get("node"),
                    operation=operation,
                    producer_invocation_id=output.invocation_id,
                )

    def _emit_control_selection(
        self,
        team: MASGraph,
        *,
        source_node_id: str,
        target_node_id: str,
        operation: str,
    ) -> None:
        edge = next(
            (
                item
                for item in team.meta.get("typed_edges") or []
                if item.get("edge_type") == "control"
                and str((item.get("source") or {}).get("node") or "") == source_node_id
                and str((item.get("target") or {}).get("node") or "") == target_node_id
            ),
            None,
        )
        self.ctx.log_event(
            "control_edge.selected",
            edge_id=(edge or {}).get("id"),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            operation=operation,
            source=(edge or {}).get("source"),
            target=(edge or {}).get("target"),
        )

    def _emit_coordination_decision(
        self,
        team: MASGraph,
        *,
        actor: str,
        operation: str,
        selected_node: str,
        decision_source: str,
        raw_decision: str | None = None,
    ) -> None:
        """Project one framework decision into the portable coordination events."""

        valid_target = selected_node in team.names
        self.ctx.log_event(
            "coordination.operation.completed",
            operation=operation,
            actor=actor,
            framework=self.name,
            decision_source=decision_source,
            raw_decision=raw_decision,
            selected_node=selected_node,
            valid=valid_target or selected_node == "FINISH",
        )
        if valid_target:
            self._emit_control_selection(
                team,
                source_node_id=actor,
                target_node_id=selected_node,
                operation=operation,
            )

    @staticmethod
    def _visible_history(
        node: Any,
        history: list[Message],
        policies: dict[str, dict[str, Any]],
    ) -> list[Message]:
        """Materialize the message Data Edges for one Node activation."""

        visible = [
            item
            for item in history
            if not policies or item.sender in policies or item.sender == node.name
        ]
        latest_sources = {
            source
            for source, policy in policies.items()
            if str((policy or {}).get("history") or "all") == "latest"
        }
        if not latest_sources:
            return visible
        latest_index = {
            source: max(index for index, item in enumerate(visible) if item.sender == source)
            for source in latest_sources
            if any(item.sender == source for item in visible)
        }
        return [
            item
            for index, item in enumerate(visible)
            if item.sender not in latest_index or latest_index[item.sender] == index
        ]

    @staticmethod
    def _uses_tools(team: MASGraph) -> bool:
        return any(
            node.tools
            or node.meta.get("tools")
            or str(node.meta.get("execution_kind") or "model") == "executor"
            for node in team.nodes
        )

    @staticmethod
    def _needs_code_executor(team: MASGraph) -> bool:
        return any(
            "runtime:python_code" in {*node.tools, *(node.meta.get("tools") or [])}
            or (
                str(node.meta.get("execution_kind") or "model") == "executor"
                and str(node.meta.get("executor_type") or "") == "code"
            )
            for node in team.nodes
        )

    @staticmethod
    def _completion_condition_met(team: MASGraph, message: Message) -> bool:
        """Return whether one published message satisfies the Team completion contract."""

        return bool(str(message.content or "").strip()) and bool(result_messages(team, [message]))

    def _effective_max_turns(self, coordination: dict[str, Any], participants: int) -> int:
        if self.max_turns is not None:
            return int(self.max_turns)
        operations = coordination.get("operations") or {}
        if not operations and not coordination.get("handoff_relations"):
            return max(1, participants * self.max_rounds)
        return 20

    def _prepare_workspace(
        self, query: TaskQuery, benchmark: Any
    ) -> tuple[Path, str, list[str], dict[str, Any]]:
        prepared = self.workspace_service.materialize(query, benchmark)
        return (
            prepared.root,
            prepared.task_text,
            list(prepared.visible_paths),
            prepared.details,
        )

    def _build_code_executor(self, workspace: Path):
        return self.code_executor_factory.build(workspace)

    def _tools_by_role(
        self,
        team: MASGraph,
        *,
        benchmark: Any,
        query: TaskQuery,
        workspace: Path | None,
        code_executor: Any,
        resources: list[Any],
    ) -> dict[str, dict[str, Callable]]:
        result: dict[str, dict[str, Callable]] = {}

        async def python_code(code: str) -> str:
            """Execute Python code in the case-local benchmark sandbox."""
            if code_executor is None:
                raise RuntimeError("this team has no active code executor")
            from autogen_core import CancellationToken
            from autogen_core.code_executor import CodeBlock

            execution = await code_executor.execute_code_blocks(
                [CodeBlock(code=code, language="python")],
                cancellation_token=CancellationToken(),
            )
            return str(execution.output)

        bundles: dict[str, Any] = {}
        web_tools = portable_web_tools(self.web_proxy_url)
        workspace_tools = portable_workspace_tools(workspace)
        for node in team.nodes:
            names = [*node.tools, *(node.meta.get("tools") or [])]
            if (
                str(node.meta.get("execution_kind") or "model") == "executor"
                and str(node.meta.get("executor_type") or "") == "code"
            ):
                names.append("runtime:python_code")
            tools: dict[str, Callable] = {}
            for name in names:
                if name == "list_workspace":
                    tools["list_workspace"] = workspace_tools["list_workspace"]
                elif name == "read_text_file":
                    tools["read_text_file"] = workspace_tools["read_text_file"]
                elif name == "replace_text_file":
                    tools["replace_text_file"] = workspace_tools["replace_text_file"]
                elif name == "runtime:python_code":
                    tools[python_code.__name__] = python_code
                elif name == "runtime:web_search":
                    tools["search_web"] = web_tools["search_web"]
                    tools["web_search"] = web_tools["web_search"]
                elif name == "runtime:fetch_url":
                    tools["fetch_url"] = web_tools["fetch_url"]
                elif isinstance(name, str) and name.startswith("benchmark:"):
                    bundle = bundles.get(name)
                    if bundle is None:
                        bundle = benchmark.create_tools(name, query.meta)
                        bundles[name] = bundle
                        resources.extend(bundle.resources)
                    tools.update({tool.__name__: tool for tool in bundle.tools})
                elif isinstance(name, str) and ":" in name:
                    module_name, function_name = name.split(":", 1)
                    module = __import__(module_name, fromlist=[function_name])
                    function = getattr(module, function_name)
                    tools[function.__name__] = function
            result[node.name] = tools
        return result
