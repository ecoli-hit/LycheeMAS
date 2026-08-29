"""Framework-neutral model invocation, tool loop, and canonical EventLog output."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from lychee_mas.memory.routing.base import RouteDecision
from lychee_mas.runtime.events.store import exception_record, to_jsonable

from ..tools.invocation import ToolInvocationError, bind_tool_arguments
from .telemetry import model_timing_fields
from .token_budget import bound_text_for_model_context, prepare_model_request


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation in {int, "int"}:
        return {"type": "integer"}
    if annotation in {float, "float"}:
        return {"type": "number"}
    if annotation in {bool, "bool"}:
        return {"type": "boolean"}
    return {"type": "string"}


def callable_tool_schema(function: Callable, *, name: str | None = None) -> dict[str, Any]:
    signature = inspect.signature(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name.startswith("_") or parameter.kind in {
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }:
            continue
        properties[parameter_name] = _annotation_schema(parameter.annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)
    return {
        "type": "function",
        "function": {
            "name": name or function.__name__,
            "description": inspect.getdoc(function) or function.__name__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def normalize_tool_schema(value: Any) -> dict[str, Any] | None:
    if callable(value):
        return callable_tool_schema(value)
    if not isinstance(value, Mapping):
        return None
    item = to_jsonable(dict(value))
    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        return item
    if "name" in item and "parameters" in item:
        return {"type": "function", "function": item}
    return None


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function") or {}
    return str(function.get("name") or "") if isinstance(function, Mapping) else ""


def _preview(text: str, head: int = 60, tail: int = 60) -> str:
    compact = str(text).replace("\n", "\\n")
    if len(compact) <= head + tail + 5:
        return compact
    return f"{compact[:head]} ... {compact[-tail:]}"


@dataclass
class GatewayReply:
    content: str
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)


class ModelGateway:
    """Keep framework adapters on one model/tool/observability implementation."""

    def __init__(
        self,
        *,
        ctx: Any,
        backend: Any,
        backend_resolver: Callable[[str | None], Any] | None = None,
        model_resolver: Callable[[str | None], str] | None = None,
        max_new_tokens: int,
        max_new_tokens_resolver: Callable[[str, int], int] | None = None,
        invocation_overrides_resolver: Callable[[str], dict[str, Any]] | None = None,
        control_deployment_id: str | None = None,
        control_max_new_tokens: int | None = None,
        control_invocation_overrides: Mapping[str, Any] | None = None,
        model_context_policy: Mapping[str, Any] | None = None,
        max_inline_tool_result_chars: int = 50000,
        framework: str,
    ) -> None:
        self.ctx = ctx
        self.backend = backend
        self.backend_resolver = backend_resolver
        self.model_resolver = model_resolver
        self.max_new_tokens = int(max_new_tokens)
        self.max_new_tokens_resolver = max_new_tokens_resolver
        self.invocation_overrides_resolver = invocation_overrides_resolver
        self.control_deployment_id = control_deployment_id
        self.control_max_new_tokens = control_max_new_tokens
        self.control_invocation_overrides = dict(control_invocation_overrides or {})
        self.model_context_policy = dict(model_context_policy or {"type": "unbounded"})
        self.max_inline_tool_result_chars = max(1000, int(max_inline_tool_result_chars))
        self.framework = framework

    def _binding(self, role: str, *, controller: bool) -> tuple[Any, str, str, int, dict]:
        role_deployment_id = str(self.ctx.deployment_of_role.get(role) or "")
        if role_deployment_id:
            backend = (
                self.backend_resolver(role_deployment_id)
                if self.backend_resolver
                else self.backend
            )
            model_id = (
                self.model_resolver(role_deployment_id)
                if self.model_resolver
                else str(getattr(backend, "model_name", "model"))
            )
            maximum = (
                self.max_new_tokens_resolver(role, self.max_new_tokens)
                if self.max_new_tokens_resolver
                else self.max_new_tokens
            )
            overrides = (
                self.invocation_overrides_resolver(role)
                if self.invocation_overrides_resolver
                else {}
            )
            return (
                backend,
                role_deployment_id,
                model_id,
                int(maximum),
                dict(overrides or {}),
            )
        # Fallback for callers that have not populated a control Node binding.
        if controller:
            deployment_id = str(self.control_deployment_id or "")
            backend = (
                self.backend_resolver(deployment_id) if self.backend_resolver else self.backend
            )
            model_id = (
                self.model_resolver(deployment_id)
                if self.model_resolver
                else str(getattr(backend, "model_name", "model"))
            )
            maximum = int(self.control_max_new_tokens or self.max_new_tokens)
            return (
                backend,
                deployment_id,
                model_id,
                maximum,
                dict(self.control_invocation_overrides),
            )
        deployment_id = str(self.ctx.deployment_of_role.get(role) or "")
        backend = self.backend_resolver(deployment_id) if self.backend_resolver else self.backend
        model_id = (
            self.model_resolver(deployment_id)
            if self.model_resolver
            else str(getattr(backend, "model_name", "model"))
        )
        maximum = (
            self.max_new_tokens_resolver(role, self.max_new_tokens)
            if self.max_new_tokens_resolver
            else self.max_new_tokens
        )
        overrides = (
            self.invocation_overrides_resolver(role) if self.invocation_overrides_resolver else {}
        )
        return backend, deployment_id, model_id, int(maximum), dict(overrides or {})

    async def invoke(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Mapping[str, Callable] | None = None,
        external_tool_schemas: Sequence[Mapping[str, Any]] = (),
        external_functions: Mapping[str, Callable] | None = None,
        controller: bool = False,
        json_output: Any = None,
        max_tool_iterations: int = 1,
        on_tool_limit: str = "finalize",
    ) -> GatewayReply:
        tool_functions = {**dict(tools or {}), **dict(external_functions or {})}
        schemas = [callable_tool_schema(value, name=name) for name, value in (tools or {}).items()]
        existing_tool_names = {_tool_name(item) for item in schemas}
        schemas.extend(
            schema
            for schema in (normalize_tool_schema(item) for item in external_tool_schemas)
            if schema is not None and _tool_name(schema) not in existing_tool_names
        )
        conversation = [dict(item) for item in messages]
        reply = GatewayReply(content="")
        tool_iteration_limit = max(1, int(max_tool_iterations))
        tool_limit_policy = str(on_tool_limit or "finalize")
        if tool_limit_policy not in {"finalize", "return_tool_result", "error"}:
            raise ValueError(f"unsupported on_tool_limit policy {tool_limit_policy!r}")
        for tool_iteration in range(tool_iteration_limit):
            result, start_id, allocation = await self._model_call(
                role,
                conversation,
                tool_schemas=schemas,
                controller=controller,
                json_output=json_output,
            )
            reply.prompt_tokens += int(result.n_prompt_pos)
            reply.completion_tokens += int(result.n_gen_tokens)
            reply.reasoning_content = str(getattr(result, "reasoning_content", "") or "")
            reply.content = str(getattr(result, "text", "") or "")
            calls = list(getattr(result, "tool_calls", None) or [])
            if not calls:
                return await self._recover_empty_final(
                    role,
                    conversation,
                    reply,
                    controller=controller,
                    json_output=json_output,
                )
            assistant_calls = []
            for raw in calls:
                call = dict(raw)
                call_id = str(call.get("id") or f"tool_{uuid.uuid4().hex}")
                name = str(call.get("name") or "")
                arguments_text = str(call.get("arguments") or "{}")
                request = {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "arguments": arguments_text,
                    "role": role,
                    "tool_iteration": tool_iteration,
                }
                request_id = self.ctx.log_event(
                    "tool_call.requested",
                    parent_event_id=start_id,
                    correlation_id=call_id,
                    origin=f"{self.framework}_model_gateway",
                    **request,
                )
                reply.tool_requests.append(request)
                self.ctx.remember_tool_request_event(
                    call_id,
                    request_id,
                    {
                        **request,
                        "source": role,
                        "requested_at_monotonic_s": time.monotonic(),
                    },
                )
                assistant_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments_text},
                    }
                )
                execution = await self._execute_tool(
                    role=role,
                    call_id=call_id,
                    name=name,
                    arguments_text=arguments_text,
                    function=tool_functions.get(name),
                    request_event_id=request_id,
                )
                reply.tool_executions.append(execution)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": str(
                            execution.get("delivered_output")
                            or execution.get("output")
                            or execution.get("error_message")
                            or ""
                        ),
                    }
                )
            conversation.insert(
                len(conversation) - len(calls),
                {
                    "role": "assistant",
                    "content": reply.content or None,
                    "tool_calls": assistant_calls,
                },
            )
        self.ctx.log_event(
            "tool_loop.limit_reached",
            role=role,
            actor=role,
            framework=self.framework,
            max_tool_iterations=tool_iteration_limit,
            tool_request_count=len(reply.tool_requests),
            on_tool_limit=tool_limit_policy,
        )
        if tool_limit_policy == "error":
            raise RuntimeError(
                f"Node {role!r} reached max_tool_iterations={tool_iteration_limit}"
            )
        if tool_limit_policy == "return_tool_result":
            latest = reply.tool_executions[-1] if reply.tool_executions else {}
            reply.content = str(
                latest.get("delivered_output")
                or latest.get("output")
                or latest.get("error_message")
                or reply.content
                or ""
            )
            return reply
        conversation.append(
            {
                "role": "user",
                "content": (
                    "The tool-use limit for this turn has been reached. Do not request "
                    "another tool. Return the best final response supported by the "
                    "evidence and tool results already present in the conversation."
                ),
            }
        )
        result, _, _ = await self._model_call(
            role,
            conversation,
            tool_schemas=[],
            controller=controller,
            json_output=json_output,
        )
        reply.prompt_tokens += int(result.n_prompt_pos)
        reply.completion_tokens += int(result.n_gen_tokens)
        reply.reasoning_content = str(getattr(result, "reasoning_content", "") or "")
        reply.content = str(getattr(result, "text", "") or "")
        return await self._recover_empty_final(
            role,
            conversation,
            reply,
            controller=controller,
            json_output=json_output,
        )

    async def _recover_empty_final(
        self,
        role: str,
        conversation: list[dict[str, Any]],
        reply: GatewayReply,
        *,
        controller: bool,
        json_output: Any,
    ) -> GatewayReply:
        """Retry a reasoning-only response without rewriting model output.

        Some reasoning parsers can return a valid reasoning field with no final
        content and no tool call. The framework cannot consume that as a message,
        so request the missing deliverable again under the same role/model/sampling
        policy. Every retry remains a normal ``model_call`` in the EventLog.
        """

        if reply.content.strip() or not reply.reasoning_content.strip():
            return reply
        recovery_messages = list(conversation)
        for _recovery_index in range(1, 4):
            recovery_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response contained reasoning but no final "
                        "deliverable. Return the final deliverable now. Do not request "
                        "another tool and do not return reasoning alone."
                    ),
                }
            )
            result, _, _ = await self._model_call(
                role,
                recovery_messages,
                tool_schemas=[],
                controller=controller,
                json_output=json_output,
            )
            reply.prompt_tokens += int(result.n_prompt_pos)
            reply.completion_tokens += int(result.n_gen_tokens)
            reply.reasoning_content = str(getattr(result, "reasoning_content", "") or "")
            reply.content = str(getattr(result, "text", "") or "")
            if reply.content.strip() or not reply.reasoning_content.strip():
                break
        return reply

    def invoke_sync(self, *args: Any, **kwargs: Any) -> GatewayReply:
        return asyncio.run(self.invoke(*args, **kwargs))

    async def _model_call(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        tool_schemas: list[dict[str, Any]],
        controller: bool,
        json_output: Any,
    ) -> tuple[Any, str | None, Any]:
        backend, deployment_id, model_id, configured_maximum, overrides = self._binding(
            role, controller=controller
        )
        budget_policy = dict(overrides.pop("token_budget_policy", {}) or {})
        prepared = prepare_model_request(
            backend,
            messages,
            tools=tool_schemas,
            max_new_tokens=configured_maximum,
            budget_policy=budget_policy,
            model_context_policy=self.model_context_policy,
        )
        allocation = prepared.allocation
        turn = self.ctx.turn_of(role)
        sender = next(
            (
                str(item.get("name") or item.get("role") or "")
                for item in reversed(prepared.messages)
                if item.get("role") != "system"
            ),
            None,
        )
        model_call_index = self.ctx.reserve_model_call(role, controller=controller)
        event_scope = {
            "case_id": self.ctx.current_case_id,
            "dataset_index": self.ctx.current_dataset_index,
            "worker_id": self.ctx.current_worker_id,
            "trial_index": self.ctx.current_trial_index,
            "attempt": self.ctx.current_attempt,
        }
        start_id = self.ctx.log_event(
            "model_call.started",
            role=role,
            actor=role,
            controller=controller,
            framework=self.framework,
            deployment_instance_id=deployment_id,
            model_id=model_id,
            model_call_index=model_call_index,
            max_model_calls_per_case=self.ctx.max_model_calls_per_case,
            message_count=len(prepared.messages),
            tool_count=len(tool_schemas),
            json_output_requested=bool(json_output),
            max_new_tokens=allocation.effective_max_new_tokens,
            backend_messages=to_jsonable(prepared.messages),
            **event_scope,
            **allocation.event_fields(),
        )
        if bool(getattr(self.ctx, "trace_model_calls", True)):
            input_chars = sum(len(str(item.get("content") or "")) for item in prepared.messages)
            print(
                f"[model:{role}] start turn={turn} framework={self.framework} "
                f"messages={len(prepared.messages)} chars={input_chars} "
                f"tools={len(tool_schemas)} "
                f"max_new_tokens={allocation.effective_max_new_tokens}",
                flush=True,
            )
        started = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "max_new_tokens": allocation.effective_max_new_tokens,
                "seed": self.ctx.generation_seed,
                "request_overrides": overrides,
            }
            if self.ctx.case_deadline_monotonic_s is not None:
                overrides["_request_deadline_monotonic_s"] = (
                    self.ctx.case_deadline_monotonic_s
                )
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"
            if json_output is not None:
                kwargs["json_output"] = json_output
            result = await asyncio.to_thread(backend.generate_chat, prepared.messages, **kwargs)
        except BaseException as exc:
            self.ctx.log_event(
                "model_call.failed",
                operation_id=start_id,
                role=role,
                actor=role,
                controller=controller,
                framework=self.framework,
                deployment_instance_id=deployment_id,
                model_call_index=model_call_index,
                **event_scope,
                **exception_record(exc),
            )
            raise
        elapsed = round(time.monotonic() - started, 6)
        reasoning = str(getattr(result, "reasoning_content", "") or "")
        final = str(getattr(result, "text", "") or "")
        timing_fields = model_timing_fields(result)
        if timing_fields["client_model_call_wall_time_s"] is None:
            timing_fields["client_model_call_wall_time_s"] = elapsed
        if bool(getattr(self.ctx, "trace_model_calls", True)):
            print(
                f"[model:{role}] done turn={turn} input_pos={int(result.n_prompt_pos)} "
                f"output_tokens={int(result.n_gen_tokens)} latency_s={elapsed} "
                f"tool_calls={len(list(getattr(result, 'tool_calls', []) or []))} "
                f"preview={_preview(final)!r}",
                flush=True,
            )
        self.ctx.bump_turn(role)
        self.ctx.last_model_exchange = {
            "backend_messages": to_jsonable(prepared.messages),
            "output": final,
        }
        self.ctx.log_decision(
            role,
            turn,
            sender,
            RouteDecision(
                channel="none",
                reason=("framework_controller" if controller else "framework_model_gateway"),
            ),
            {
                "controller": controller,
                "framework": self.framework,
                "model_call_index": model_call_index,
                "deployment_instance_id": deployment_id,
                "input_positions": int(result.n_prompt_pos),
                "text_input_tokens": int(result.n_prompt_pos),
                "latent_prefix_positions": 0,
                "output_tokens": int(result.n_gen_tokens),
                "output_reasoning_tokens": getattr(result, "reasoning_tokens", None),
                "output_answer_tokens": getattr(result, "answer_tokens", None),
                "output_reasoning_tokens_source": getattr(result, "reasoning_tokens_source", None),
                "output_answer_tokens_source": getattr(result, "answer_tokens_source", None),
                "input_cached_tokens": getattr(result, "cached_input_tokens", None),
                "input_cached_tokens_source": getattr(result, "cached_input_tokens_source", None),
                "input_cached_tokens_status": getattr(
                    result, "cached_input_tokens_status", "not_reported"
                ),
                "model_generation_latency_s": float(getattr(result, "latency_s", elapsed)),
                **timing_fields,
                "finish_reason": str(getattr(result, "finish_reason", "stop")),
                "prompt_pos": int(result.n_prompt_pos),
                "gen_tokens": int(result.n_gen_tokens),
                "prefix_len": 0,
                "latency_s": float(getattr(result, "latency_s", elapsed)),
                **allocation.event_fields(),
            },
        )
        self.ctx.log_event(
            "model_call.completed",
            operation_id=start_id,
            role=role,
            actor=role,
            controller=controller,
            framework=self.framework,
            deployment_instance_id=deployment_id,
            model_id=model_id,
            model_call_index=model_call_index,
            **event_scope,
            input_total_positions=int(result.n_prompt_pos),
            input_text_tokens=int(result.n_prompt_pos),
            input_latent_positions=0,
            output_text_tokens=int(result.n_gen_tokens),
            output_total_tokens=int(result.n_gen_tokens),
            output_reasoning_tokens=getattr(result, "reasoning_tokens", None),
            output_answer_tokens=getattr(result, "answer_tokens", None),
            output_reasoning_tokens_source=getattr(result, "reasoning_tokens_source", None),
            output_answer_tokens_source=getattr(result, "answer_tokens_source", None),
            input_cached_tokens=getattr(result, "cached_input_tokens", None),
            input_cached_tokens_source=getattr(result, "cached_input_tokens_source", None),
            input_cached_tokens_status=getattr(
                result, "cached_input_tokens_status", "not_reported"
            ),
            model_latency_s=float(getattr(result, "latency_s", elapsed)),
            **timing_fields,
            finish_reason=str(getattr(result, "finish_reason", "stop")),
            provider_request_payload=getattr(result, "provider_request_payload", {}),
            provider_response_payload=getattr(result, "provider_response_payload", {}),
            response_payload_origin=getattr(result, "response_payload_origin", None),
            reasoning_content=reasoning,
            final_content=final,
            delivered_content=(list(getattr(result, "tool_calls", []) or []) or final),
            tool_call_count=len(list(getattr(result, "tool_calls", []) or [])),
            **allocation.event_fields(),
        )
        return result, start_id, allocation

    async def _execute_tool(
        self,
        *,
        role: str,
        call_id: str,
        name: str,
        arguments_text: str,
        function: Callable | None,
        request_event_id: str | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        event_id = self.ctx.log_event(
            "tool_execution.started",
            parent_event_id=request_event_id,
            correlation_id=call_id,
            tool_call_id=call_id,
            tool_name=name,
            role=role,
            framework=self.framework,
        )
        try:
            arguments = bind_tool_arguments(
                tool_name=name,
                arguments_text=arguments_text,
                function=function,
            )
            assert function is not None
            value = function(**arguments)
            if inspect.isawaitable(value):
                value = await value
            output = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        except BaseException as exc:
            record = {
                "tool_call_id": call_id,
                "tool_name": name,
                "role": role,
                "is_error": True,
                "duration_s": round(time.monotonic() - started, 6),
                **(
                    {
                        "failure_kind": exc.reason,
                        "invalid_tool_arguments": exc.reason != "unknown_tool",
                    }
                    if isinstance(exc, ToolInvocationError)
                    else {}
                ),
                **exception_record(exc),
            }
            self.ctx.log_event(
                "tool_execution.failed",
                operation_id=event_id,
                correlation_id=call_id,
                framework=self.framework,
                **record,
            )
            return record
        delivered_output, output_truncated = bound_text_for_model_context(
            output, self.max_inline_tool_result_chars
        )
        record = {
            "tool_call_id": call_id,
            "tool_name": name,
            "role": role,
            "is_error": False,
            "output": output,
            "delivered_output": delivered_output,
            "output_chars": len(output),
            "delivered_output_chars": len(delivered_output),
            "output_truncated_for_model_context": output_truncated,
            "duration_s": round(time.monotonic() - started, 6),
        }
        self.ctx.log_event(
            "tool_execution.completed",
            operation_id=event_id,
            correlation_id=call_id,
            framework=self.framework,
            **record,
        )
        return record
