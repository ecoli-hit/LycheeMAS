"""InjectionClient —— 一个「会路由 + 注入记忆」的 AutoGen ChatCompletionClient（迁移自
models/injection_client.py）。

【它在 AutoGen 里的位置】
AutoGen 的 AssistantAgent 持有一个 `model_client`（必须是 `ChatCompletionClient`），每当轮到该 agent
发言，AgentChat 就调用 `model_client.create(...)`。我们通过**继承 ChatCompletionClient** 接管这次
调用，
完成「路由 + 记忆注入」再真正生成。

【每次 create() 做的事（整条流水线的汇合点）】
  1. 把 LLMMessage 列表转成 [{role, content}] 给 HF chat 模板
  2. memory.observe(history)                       —— 更新记忆库（方法接缝）
  3. router.decide(role, task, turn, sender, ...)  —— 决定本轮记忆通道（触发接缝）
  4. memory.recall(decision, query) -> MemoryBundle—— 物化出要注入的 NL 文本 / latent prefix
  5. 注入：NL 文本进 prompt（system 消息）；latent prefix 进 backend 的 embedding 层
  6. 用共享 HF backend 生成，包成 AutoGen 期望的 CreateResult 返回

【实例关系】每个 agent 独占一个 InjectionClient（绑定各自 role）；所有 client 共享 backend +
RoutingContext。

⚠️ 这是【唯一允许 import autogen 的文件之一】（CLAUDE.md 黄金法则 1：runtime/backends/autogen_*.py
）。
autogen_core 的导入与 ChatCompletionClient 子类的「构造」都被惰性化到工厂内部，保证无 autogen 时本
模块
仍可被 import（注册一个延迟构造的 model_client/injection 工厂）。
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Mapping, Optional, Sequence

from ...core.registry import REGISTRY
from ..token_budget import count_request_tokens, prepare_model_request

# 注：NL 注入内容的来源标志由 NLMemory.recall 产出（见 memory/channels/nl.py 的
# PREV_OUTPUT_HEADER），
# 本文件把 bundle.NL_Channel 原样作为 system 消息插入。


async def _call_backend(
    backend,
    method: str,
    *args,
    cancellation_token=None,
    **kwargs,
):
    """Call a backend without blocking AutoGen's event loop.

    AutoGen cancellation is linked to the worker future. Local HF additionally
    receives the token and checks it during token generation; synchronous HTTP
    clients may finish their in-flight request in the worker after the awaiting
    AutoGen task has been cancelled.
    """
    fn = getattr(backend, method)
    request_seed = kwargs.pop("request_seed", None)
    if bool(getattr(backend, "supports_request_seed", False)):
        kwargs["seed"] = request_seed
    if bool(getattr(backend, "supports_cancellation_token", False)):
        kwargs["cancellation_token"] = cancellation_token
    if cancellation_token is not None and cancellation_token.is_cancelled():
        raise asyncio.CancelledError

    use_worker = bool(
        getattr(backend, "supports_concurrent_requests", False)
        or getattr(backend, "runs_in_worker_thread", False)
    )
    if use_worker:

        def invoke():
            lock = getattr(backend, "_call_lock", None)
            if lock is None:
                return fn(*args, **kwargs)
            with lock:
                return fn(*args, **kwargs)

        future = asyncio.create_task(asyncio.to_thread(invoke))
        if cancellation_token is not None:
            cancellation_token.link_future(future)
        return await future
    return fn(*args, **kwargs)


def _add_usage(left, right):
    from autogen_core.models import RequestUsage

    return RequestUsage(
        prompt_tokens=int(left.prompt_tokens) + int(right.prompt_tokens),
        completion_tokens=int(left.completion_tokens) + int(right.completion_tokens),
    )


def _backend_extra_create_args(backend, extra_create_args: Mapping[str, Any]) -> dict[str, Any]:
    extra = dict(extra_create_args or {})
    if not extra:
        return {}
    if not bool(getattr(backend, "supports_extra_create_args", False)):
        raise ValueError(f"{type(backend).__name__} does not support AutoGen extra_create_args")
    return {"extra_create_args": extra}


def _content_to_text(content: Any) -> str:
    """Best-effort text view used for routing queries and unknown message types."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    if hasattr(content, "model_dump"):
        try:
            return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
        except Exception:
            return str(content)
    if hasattr(content, "__dict__"):
        try:
            return json.dumps(vars(content), ensure_ascii=False, default=str)
        except Exception:
            return str(content)
    return str(content)


def _content_to_backend(content: Any) -> Any:
    """Preserve AutoGen text/image parts in an OpenAI-compatible content shape.

    AutoGen ``UserMessage.content`` is either text or ``list[str | Image]``.
    Keeping image parts here allows both the OpenAI-compatible backend and the
    local Transformers backend to perform their own provider-specific mapping.
    """

    if not isinstance(content, list):
        return _content_to_text(content)
    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        to_openai = getattr(item, "to_openai_format", None)
        if callable(to_openai):
            parts.append(dict(to_openai()))
            continue
        if isinstance(item, Mapping):
            part = dict(item)
            if part.get("type") in {"text", "image", "image_url"}:
                parts.append(part)
                continue
        parts.append({"type": "text", "text": _content_to_text(item)})
    return parts


def _trace_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages for spans without sharing mutable nested containers."""

    from ...runtime.spans import to_jsonable

    return [to_jsonable(dict(message)) for message in messages]


def _model_call_start_message_fields(
    ctx: Any,
    *,
    autogen_model_messages: list[dict[str, Any]],
    role_visible_messages: list[dict[str, Any]],
    backend_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the sole compact/full schema difference for model_call_start."""

    if str(getattr(ctx, "trace_detail_level", "compact")) != "full":
        return {}
    return {
        "autogen_model_messages": autogen_model_messages,
        "role_visible_messages": role_visible_messages,
        "backend_messages": backend_messages,
    }


def _thinking_budget_span_fields(request_overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Expose provider-specific reasoning budgets under one trace schema."""

    extra_body = request_overrides.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return {}
    if extra_body.get("thinking_token_budget") is not None:
        return {
            "thinking_budget_requested": int(extra_body["thinking_token_budget"]),
            "thinking_budget_parameter": "thinking_token_budget",
            "thinking_budget_enforced_by": "vllm",
        }
    if extra_body.get("thinking_budget") is not None:
        return {
            "thinking_budget_requested": int(extra_body["thinking_budget"]),
            "thinking_budget_parameter": "thinking_budget",
            "thinking_budget_enforced_by": "provider",
        }
    return {}


def _request_transport_span_fields(backend: Any, max_new_tokens: int) -> dict[str, Any]:
    """Describe the timeout/retry decision without changing the provider payload."""

    planner = getattr(backend, "request_timeout_plan", None)
    if not callable(planner):
        return {}
    plan = planner(max_new_tokens)
    return {
        "request_timeout_mode": plan.get("mode"),
        "request_timeout_s": plan.get("timeout_s"),
        "request_max_retries": plan.get("max_retries"),
        "request_timeout_planning_tokens_per_second": plan.get(
            "planning_generation_tokens_per_second"
        ),
        "request_timeout_observed_tokens_per_second": plan.get(
            "observed_generation_tokens_per_second"
        ),
        "request_timeout_policy": plan.get("policy"),
    }


def _completed_request_transport_span_fields(
    result: Any, planned: Mapping[str, Any]
) -> dict[str, Any]:
    fields = dict(planned)
    if getattr(result, "request_timeout_s", None) is not None:
        fields["request_timeout_s"] = float(result.request_timeout_s)
    fields["request_max_retries"] = int(getattr(result, "request_max_retries", 0))
    if getattr(result, "observed_generation_tokens_per_second", None) is not None:
        fields["request_observed_generation_tokens_per_second"] = float(
            result.observed_generation_tokens_per_second
        )
    return fields


def _optional_seconds(result: Any, name: str) -> float | None:
    value = getattr(result, name, None)
    return round(float(value), 6) if value is not None else None


def _model_timing_fields(result: Any) -> dict[str, Any]:
    """Expose client and provider timings with explicit availability."""

    provider_metrics = dict(getattr(result, "provider_request_metrics", {}) or {})
    return {
        "client_rate_limiter_wait_s": _optional_seconds(result, "client_rate_limiter_wait_s"),
        "client_http_request_latency_s": _optional_seconds(result, "client_http_request_latency_s"),
        "client_response_postprocess_latency_s": _optional_seconds(
            result, "client_response_postprocess_latency_s"
        ),
        "client_model_call_wall_time_s": _optional_seconds(result, "client_model_call_wall_time_s"),
        "provider_request_metrics_available": bool(provider_metrics),
        "provider_request_queue_latency_s": _optional_seconds(
            result, "provider_request_queue_latency_s"
        ),
        "provider_scheduled_to_first_token_s": _optional_seconds(
            result, "provider_scheduled_to_first_token_s"
        ),
        "provider_generation_latency_s": _optional_seconds(result, "provider_generation_latency_s"),
        "provider_mean_inter_token_latency_s": _optional_seconds(
            result, "provider_mean_inter_token_latency_s"
        ),
        "provider_output_tokens_per_second": _optional_seconds(
            result, "provider_output_tokens_per_second"
        ),
        "provider_request_metrics": provider_metrics,
    }


def _prepare_backend_request(
    backend: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Any],
    configured_max_new_tokens: int,
    request_overrides: Mapping[str, Any],
    model_context_policy: Mapping[str, Any] | None,
):
    """Resolve one call's adaptive context/output/reasoning allocation."""

    provider_overrides = dict(request_overrides or {})
    prepared_messages = [dict(message) for message in messages]
    if provider_overrides.get("preserve_thinking") is False:
        for message in prepared_messages:
            message.pop("reasoning_content", None)
    provider_overrides.pop("preserve_thinking", None)
    budget_policy = dict(provider_overrides.pop("token_budget_policy", {}) or {})
    counted_tools = tools if getattr(backend, "supports_native_tools", False) else ()
    prepared = prepare_model_request(
        backend,
        prepared_messages,
        tools=[_tool_schema(tool) for tool in counted_tools],
        max_new_tokens=configured_max_new_tokens,
        budget_policy=budget_policy,
        model_context_policy=model_context_policy,
    )
    effective_thinking = prepared.allocation.effective_max_thinking_budget_tokens
    if effective_thinking is not None:
        extra_body = dict(provider_overrides.get("extra_body") or {})
        if "thinking_token_budget" in extra_body:
            extra_body["thinking_token_budget"] = effective_thinking
        elif "thinking_budget" in extra_body:
            extra_body["thinking_budget"] = effective_thinking
        provider_overrides["extra_body"] = extra_body
    return prepared.messages, provider_overrides, prepared.allocation


def _to_chat(messages: Sequence[Any]):
    """把 AutoGen 的 LLMMessage 列表转成 HF chat 模板要的 [{role, content}] 字典列表。

    autogen_core 在函数内惰性导入；按子类拆出 role/content，并保留 source（发言者名）以识别 sender。
    """
    from autogen_core.models import (
        AssistantMessage,
        FunctionExecutionResultMessage,
        SystemMessage,
        UserMessage,
    )

    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, AssistantMessage):
            if isinstance(m.content, list):
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": m.thought,
                        "source": m.source,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                            for call in m.content
                        ],
                    }
                )
            else:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "reasoning_content": m.thought,
                        "source": m.source,
                    }
                )
        elif isinstance(m, UserMessage):
            c = _content_to_backend(m.content)
            out.append({"role": "user", "content": c, "source": getattr(m, "source", "user")})
        elif isinstance(m, FunctionExecutionResultMessage):
            for result in m.content:
                out.append(
                    {
                        "role": "tool",
                        "content": result.content,
                        "tool_call_id": result.call_id,
                        "source": result.name,
                    }
                )
        else:
            source = getattr(m, "source", type(m).__name__)
            content = _content_to_text(getattr(m, "content", ""))
            out.append({"role": "user", "content": f"[{source}]\n{content}", "source": source})
    return out


def _tool_schema(tool: Any) -> dict[str, Any]:
    if isinstance(tool, Mapping):
        schema = dict(tool)
        function = schema.get("function")
        if isinstance(function, Mapping):
            return {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        return schema
    schema = getattr(tool, "schema", None)
    if isinstance(schema, Mapping):
        return dict(schema)
    if hasattr(tool, "model_dump"):
        try:
            dumped = tool.model_dump(mode="json")
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass
    return {
        "name": getattr(tool, "name", type(tool).__name__),
        "description": getattr(tool, "description", getattr(tool, "__doc__", "") or ""),
    }


def _tool_prompt(tools: Sequence[Any]) -> str:
    schemas = [_tool_schema(tool) for tool in tools]
    return (
        "You may call tools when needed. If you need a tool, respond with ONLY a JSON "
        "object in one of these forms:\n"
        '{"tool": "<tool_name>", "arguments": {...}}\n'
        '{"tool_calls": [{"name": "<tool_name>", "arguments": {...}}]}\n'
        "Do not wrap tool-call JSON in markdown. Available tools:\n"
        f"{json.dumps(schemas, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_tool_calls(text: str, tools: Sequence[Any]):
    if not tools:
        return None
    data = _extract_json_object(text)
    if not data:
        return None
    available = {_tool_schema(tool).get("name") for tool in tools}
    raw_calls: list[dict[str, Any]] = []
    if isinstance(data.get("tool_calls"), list):
        raw_calls = [call for call in data["tool_calls"] if isinstance(call, dict)]
    elif data.get("tool") or data.get("name"):
        raw_calls = [data]
    calls = []
    from autogen_core import FunctionCall

    for raw in raw_calls:
        function = raw.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            args = function.get("arguments", raw.get("arguments", raw.get("args", {})))
        else:
            name = raw.get("name") or raw.get("tool") or raw.get("function")
            args = raw.get("arguments", raw.get("args", {}))
        if not name or name not in available:
            continue
        if isinstance(args, str):
            args_text = args
        else:
            args_text = json.dumps(args or {}, ensure_ascii=False)
        calls.append(
            FunctionCall(
                id=raw.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=str(name),
                arguments=args_text,
            )
        )
    return calls or None


def _native_tool_calls(raw_calls: Sequence[Mapping[str, Any]], tools: Sequence[Any]):
    if not raw_calls or not tools:
        return None
    available = {_tool_schema(tool).get("name") for tool in tools}
    from autogen_core import FunctionCall

    calls = []
    for raw in raw_calls:
        name = str(raw.get("name") or "")
        if not name or name not in available:
            continue
        calls.append(
            FunctionCall(
                id=str(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                name=name,
                arguments=str(raw.get("arguments") or "{}"),
            )
        )
    return calls or None


def _preview_text(text: str, head: int = 60, tail: int = 60) -> str:
    compact = text.replace("\n", "\\n")
    if len(compact) <= head + tail + 5:
        return compact
    return f"{compact[:head]} ... {compact[-tail:]}"


def _build_injection_client_class():
    """惰性构造 InjectionClient/PlainClient 类（继承 autogen_core 的 ChatCompletionClient）。

    autogen_core 仅在调用本工厂时导入，从而 import 本模块不需要 autogen。
    """
    from autogen_core import CancellationToken  # noqa: F401
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        ModelFamily,
        ModelInfo,
        RequestUsage,
    )

    # RouterInputs 是纯 dataclass（无 autogen/torch），从触发接缝直接复用；
    # LatentMemory.FUSION_STRATEGIES = latent 走 KV 融合(而非 prefix 拼接)的策略集（单一事实来源）。
    from ...memory.channels.latent import LatentMemory
    from ...memory.routing.base import RouteDecision, RouterInputs

    class InjectionClient(ChatCompletionClient):
        """实现 AutoGen 的 ChatCompletionClient 接口。一个 agent 一个实例（绑定其 role）。"""

        def __init__(
            self,
            backend,
            role: str,
            ctx,
            max_new_tokens: int = 256,
            model_id: str = "qwen-3-4b",
            request_overrides=None,
            model_context_policy=None,
            deployment_id: str | None = None,
        ):
            self.backend = backend  # 共享的 HF 后端（模型只加载一次，所有 agent 复用）
            self.role = role  # 本 client 服务的角色（路由按此条件化）
            self.ctx = ctx  # 共享的 RoutingContext：turn 计数 / 决策日志 / 同模型对约束
            self.max_new_tokens = max_new_tokens
            self.model_id = model_id  # 模型标识（用于 constraint #2 的同模型对判断）
            self.deployment_id = deployment_id
            self.request_overrides = dict(request_overrides or {})
            self.model_context_policy = dict(model_context_policy or {"type": "unbounded"})
            self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
            self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(
            self,
            messages,
            *,
            tools=(),
            tool_choice="auto",
            json_output=None,
            extra_create_args: Mapping[str, Any] | None = None,
            cancellation_token: Optional[Any] = None,
        ) -> CreateResult:
            chat = _to_chat(messages)
            autogen_model_messages = _trace_messages(chat)
            role_visible_messages = _trace_messages(chat)
            # sender = 最近一条非 system 消息的发言者；query = 其内容
            sender, query = None, ""
            for m in reversed(chat):
                if m["role"] != "system":
                    sender = m.get("source")
                    query = m.get("content", "")
                    break

            # ① 观察：把当前历史交给记忆库更新
            self.ctx.memory.observe(chat)
            # ② 记忆通道决策：按 (role×task×turn×sender×可用性×同模型对) 决定本轮通道
            turn = self.ctx.turn_of(self.role)
            decision = self.ctx.router.decide(
                RouterInputs(
                    role=self.role,
                    task=self.ctx.task,
                    turn=turn,
                    sender=sender,
                    query=query if isinstance(query, str) else str(query),
                    availability=self.ctx.availability,
                    same_model_pair=self.ctx.same_model_pair(sender, self.role),
                )
            )
            # ③ 召回：按决策物化要注入的 NL 文本 / latent prefix
            bundle = self.ctx.memory.recall(decision, query if isinstance(query, str) else "")

            # ---- 注入 ----
            send_msgs = list(chat)
            native_tools = bool(tools and getattr(self.backend, "supports_native_tools", False))
            if tools and not native_tools:
                insert_at = 0
                while insert_at < len(send_msgs) and send_msgs[insert_at]["role"] == "system":
                    insert_at += 1
                send_msgs.insert(insert_at, {"role": "system", "content": _tool_prompt(tools)})

            if bundle.NL_Channel:
                # NL 通道：把记忆作为一条 system 消息，插在开头 system 提示之后、对话之前
                insert_at = 0
                while insert_at < len(send_msgs) and send_msgs[insert_at]["role"] == "system":
                    insert_at += 1
                send_msgs.insert(insert_at, {"role": "system", "content": bundle.NL_Channel})

            send_msgs, effective_overrides, allocation = _prepare_backend_request(
                self.backend,
                send_msgs,
                tools=tools,
                configured_max_new_tokens=self.max_new_tokens,
                request_overrides=self.request_overrides,
                model_context_policy=self.model_context_policy,
            )
            effective_max_new_tokens = allocation.effective_max_new_tokens
            request_transport_fields = _request_transport_span_fields(
                self.backend, effective_max_new_tokens
            )

            trace_model_calls = bool(getattr(self.ctx, "trace_model_calls", True))
            input_chars = sum(len(str(m.get("content", ""))) for m in send_msgs)
            backend_messages = _trace_messages(send_msgs)
            start_message_fields = _model_call_start_message_fields(
                self.ctx,
                autogen_model_messages=autogen_model_messages,
                role_visible_messages=role_visible_messages,
                backend_messages=backend_messages,
            )
            model_call_index = self.ctx.reserve_model_call(self.role)
            span_id = self.ctx.log_span(
                "model_call_start",
                deployment_instance_id=self.deployment_id,
                role=self.role,
                turn=turn,
                sender=sender,
                model_call_index=model_call_index,
                request_seed=getattr(self.ctx, "generation_seed", None),
                max_model_calls_per_case=self.ctx.max_model_calls_per_case,
                message_count=len(send_msgs),
                input_chars=input_chars,
                tool_count=len(tools),
                tool_choice=tool_choice,
                json_output_requested=bool(json_output),
                max_new_tokens=effective_max_new_tokens,
                invocation_policy=effective_overrides,
                memory_channel=decision.channel,
                routing_reason=decision.reason,
                nl_memory_chars=len(bundle.NL_Channel or ""),
                trace_detail_level=str(getattr(self.ctx, "trace_detail_level", "compact")),
                **allocation.span_fields(),
                **_thinking_budget_span_fields(effective_overrides),
                **request_transport_fields,
                **start_message_fields,
            )
            if trace_model_calls:
                print(
                    f"[model:{self.role}] start call={model_call_index}/"
                    f"{self.ctx.max_model_calls_per_case or '∞'} turn={turn} "
                    f"sender={sender or '-'} "
                    f"messages={len(send_msgs)} chars={input_chars} tools={len(tools)} "
                    f"max_new_tokens={effective_max_new_tokens}/"
                    f"{self.max_new_tokens} "
                    f"timeout_s={request_transport_fields.get('request_timeout_s', '-')} "
                    f"retries={request_transport_fields.get('request_max_retries', '-')}",
                    flush=True,
                )

            try:
                if bundle.Latent_Channel is None:
                    # 无 latent：走普通文本生成（none / nl_only 都走这里）
                    backend_kwargs = {
                        "max_new_tokens": effective_max_new_tokens,
                        "request_seed": getattr(self.ctx, "generation_seed", None),
                    }
                    if effective_overrides and getattr(
                        self.backend, "supports_request_overrides", False
                    ):
                        backend_kwargs["request_overrides"] = effective_overrides
                    backend_kwargs.update(
                        _backend_extra_create_args(self.backend, extra_create_args or {})
                    )
                    if json_output is not None:
                        backend_kwargs["json_output"] = json_output
                    if native_tools:
                        backend_kwargs["tools"] = [
                            {"type": "function", "function": _tool_schema(tool)} for tool in tools
                        ]
                        backend_kwargs["tool_choice"] = tool_choice
                    g = await _call_backend(
                        self.backend,
                        "generate_chat",
                        send_msgs,
                        cancellation_token=cancellation_token,
                        **backend_kwargs,
                    )
                elif bundle.Latent_strategy in LatentMemory.FUSION_STRATEGIES:
                    if extra_create_args:
                        raise ValueError(
                            "extra_create_args are unavailable for latent C2C generation"
                        )
                    # 融合类（c2c）：Latent_Channel=projector 栈；source=上一个 agent 的输入+输出
                    # （从 ctx 取），经 projector 把其 KV 融进本 agent 生成；
                    # 首个 agent 无前驱则退回普通生成。
                    src_msgs = self._c2c_source()
                    if src_msgs is not None:
                        g = await _call_backend(
                            self.backend,
                            "generate_chat_with_c2c",
                            send_msgs,
                            src_msgs,
                            bundle.Latent_Channel,
                            max_new_tokens=effective_max_new_tokens,
                            json_output=json_output,
                            request_seed=getattr(self.ctx, "generation_seed", None),
                            request_overrides=effective_overrides,
                            cancellation_token=cancellation_token,
                        )
                    else:
                        g = await _call_backend(
                            self.backend,
                            "generate_chat",
                            send_msgs,
                            max_new_tokens=effective_max_new_tokens,
                            json_output=json_output,
                            request_seed=getattr(self.ctx, "generation_seed", None),
                            request_overrides=effective_overrides,
                            cancellation_token=cancellation_token,
                        )
                else:
                    if extra_create_args:
                        raise ValueError(
                            "extra_create_args are unavailable for latent-prefix generation"
                        )
                    # prefix 类（soft_token 等）：Latent_Channel=(1,P,H) 张量，拼到 embedding 层最前
                    g = await _call_backend(
                        self.backend,
                        "generate_chat_with_prefix",
                        send_msgs,
                        bundle.Latent_Channel,
                        max_new_tokens=effective_max_new_tokens,
                        json_output=json_output,
                        request_seed=getattr(self.ctx, "generation_seed", None),
                        request_overrides=effective_overrides,
                        cancellation_token=cancellation_token,
                    )
            except Exception as exc:
                from ...runtime.spans import exception_record

                self.ctx.log_span(
                    "model_call_error",
                    parent_span_id=span_id,
                    deployment_instance_id=self.deployment_id,
                    role=self.role,
                    turn=turn,
                    sender=sender,
                    model_call_index=model_call_index,
                    **request_transport_fields,
                    **exception_record(exc),
                )
                raise

            # Preserve the model client's output exactly. AutoGen's group-chat
            # implementation owns structured-output parsing, validation, retry,
            # and failure behavior; this adapter must not rewrite controller decisions.
            output_text = g.text

            reasoning_content = str(getattr(g, "reasoning_content", "") or "")

            # ④ 记账与记录
            self.ctx.bump_turn(self.role)
            latent_prefix_positions = int(g.prefix_len or 0)
            input_positions = int(g.n_prompt_pos)
            output_tokens = int(g.n_gen_tokens)
            generation_latency_s = round(g.latency_s, 3)
            request_queue_latency_s = _optional_seconds(g, "request_queue_latency_s")
            timing_fields = _model_timing_fields(g)
            finish_reason = str(getattr(g, "finish_reason", None) or "stop")
            usage = RequestUsage(prompt_tokens=input_positions, completion_tokens=output_tokens)
            self._actual_usage = _add_usage(self._actual_usage, usage)
            self._total_usage = _add_usage(self._total_usage, usage)
            self.ctx.last_model_exchange = {
                "backend_messages": backend_messages,
                "output": output_text,
            }
            self.ctx.log_decision(
                self.role,
                turn,
                sender,
                decision,
                {
                    "model_call_index": model_call_index,
                    "deployment_instance_id": self.deployment_id,
                    "input_positions": input_positions,
                    "text_input_tokens": max(0, input_positions - latent_prefix_positions),
                    "latent_prefix_positions": latent_prefix_positions,
                    "output_tokens": output_tokens,
                    "output_reasoning_tokens": getattr(g, "reasoning_tokens", None),
                    "output_answer_tokens": getattr(g, "answer_tokens", None),
                    "output_reasoning_tokens_source": getattr(g, "reasoning_tokens_source", None),
                    "output_answer_tokens_source": getattr(g, "answer_tokens_source", None),
                    "input_cached_tokens": getattr(g, "cached_input_tokens", None),
                    "input_cached_tokens_source": getattr(g, "cached_input_tokens_source", None),
                    "input_cached_tokens_status": getattr(
                        g, "cached_input_tokens_status", "not_reported"
                    ),
                    "model_generation_latency_s": generation_latency_s,
                    "request_queue_latency_s": request_queue_latency_s,
                    **timing_fields,
                    "finish_reason": finish_reason,
                    "invocation_policy": effective_overrides,
                    "original_prompt_positions": int(
                        getattr(g, "original_prompt_pos", None) or input_positions
                    ),
                    "prompt_truncated": bool(getattr(g, "prompt_truncated", False)),
                    "dropped_messages": int(getattr(g, "dropped_messages", 0)),
                    "nl_memory_chars": len(bundle.NL_Channel or ""),
                    # CDM 通道方法（供消融分析：本轮用了哪种 NL / latent 策略）
                    "nl_strategy": bundle.NL_strategy,
                    "latent_strategy": bundle.Latent_strategy,
                    # Compact metric aliases retained for the existing case aggregator.
                    "prompt_pos": input_positions,
                    "gen_tokens": output_tokens,
                    "prefix_len": latent_prefix_positions,
                    "nl_chars": len(bundle.NL_Channel or ""),
                    "latency_s": generation_latency_s,
                },
            )
            tool_calls = _native_tool_calls(
                getattr(g, "tool_calls", None) or [], tools
            ) or _parse_tool_calls(output_text, tools)
            self.ctx.log_span(
                "model_call_end",
                parent_span_id=span_id,
                deployment_instance_id=self.deployment_id,
                role=self.role,
                turn=turn,
                sender=sender,
                model_call_index=model_call_index,
                input_total_positions=input_positions,
                input_text_tokens=max(0, input_positions - latent_prefix_positions),
                input_latent_positions=latent_prefix_positions,
                output_text_tokens=output_tokens,
                model_latency_s=generation_latency_s,
                request_queue_latency_s=request_queue_latency_s,
                **timing_fields,
                finish_reason=finish_reason,
                invocation_policy=effective_overrides,
                original_prompt_positions=int(
                    getattr(g, "original_prompt_pos", None) or input_positions
                ),
                prompt_truncated=bool(getattr(g, "prompt_truncated", False)),
                dropped_messages=int(getattr(g, "dropped_messages", 0)),
                tool_call_count=len(tool_calls or []),
                tool_call_request=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in (tool_calls or [])
                ],
                json_output_requested=bool(json_output),
                provider_request_payload=getattr(g, "provider_request_payload", {}),
                provider_response_payload=getattr(g, "provider_response_payload", {}),
                response_payload_origin=getattr(g, "response_payload_origin", None),
                output_total_tokens=output_tokens,
                input_cached_tokens=getattr(g, "cached_input_tokens", None),
                input_cached_tokens_source=getattr(g, "cached_input_tokens_source", None),
                input_cached_tokens_status=getattr(g, "cached_input_tokens_status", "not_reported"),
                output_reasoning_tokens=getattr(g, "reasoning_tokens", None),
                output_answer_tokens=getattr(g, "answer_tokens", None),
                output_reasoning_tokens_source=getattr(g, "reasoning_tokens_source", None),
                output_answer_tokens_source=getattr(g, "answer_tokens_source", None),
                output_reasoning_chars=len(reasoning_content),
                output_answer_chars=len(output_text),
                **allocation.span_fields(),
                **_thinking_budget_span_fields(effective_overrides),
                **_completed_request_transport_span_fields(g, request_transport_fields),
            )
            if trace_model_calls:
                preview = _preview_text(output_text)
                print(
                    f"[model:{self.role}] done turn={turn} input_pos={input_positions} "
                    f"output_tokens={output_tokens} latency_s={generation_latency_s} "
                    f"client_limiter_wait_s={timing_fields['client_rate_limiter_wait_s']} "
                    f"provider_queue_s={timing_fields['provider_request_queue_latency_s']} "
                    f"provider_ttft_s={timing_fields['provider_scheduled_to_first_token_s']} "
                    f"finish_reason={finish_reason} "
                    f"truncated={bool(getattr(g, 'prompt_truncated', False))} "
                    f"dropped_messages={int(getattr(g, 'dropped_messages', 0))} "
                    f"tool_calls={len(tool_calls or [])} "
                    f"preview={preview!r}",
                    flush=True,
                )
            if tool_calls:
                self.ctx.decisions[-1]["tool_call_request"] = [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in tool_calls
                ]
                return CreateResult(
                    finish_reason="function_calls",
                    content=tool_calls,
                    usage=usage,
                    cached=False,
                    thought=reasoning_content or None,
                )
            return CreateResult(
                finish_reason=finish_reason,
                content=output_text,
                usage=usage,
                cached=False,
                thought=reasoning_content or None,
            )

        def _c2c_source(self):
            """Build C2C source from the previous in-memory model exchange."""
            previous = getattr(self.ctx, "last_model_exchange", None)
            if not previous:
                return None
            inp = previous.get("backend_messages")
            if not inp:
                return None
            src = [{"role": m["role"], "content": m["content"]} for m in inp]
            out = previous.get("output")
            if isinstance(out, str) and out.strip():
                src.append({"role": "assistant", "content": out})  # 带上前驱的实际输出
            return src

        async def create_stream(
            self,
            messages,
            *,
            tools=(),
            tool_choice="auto",
            json_output=None,
            extra_create_args=None,
            cancellation_token=None,
        ):
            raise NotImplementedError(
                "LycheeMAS InjectionClient does not implement token streaming; "
                "set model_client_streaming=false"
            )
            yield  # pragma: no cover - keeps this method an async generator

        async def close(self) -> None:
            return  # 共享 backend 不在此释放，故空实现

        def actual_usage(self) -> RequestUsage:
            return self._actual_usage

        def total_usage(self) -> RequestUsage:
            return self._total_usage

        def count_tokens(self, messages, *, tools=[]) -> int:
            chat = _to_chat(messages)
            return count_request_tokens(
                self.backend,
                chat,
                [_tool_schema(tool) for tool in tools],
            )[0]

        def remaining_tokens(self, messages, *, tools=[]) -> int:
            context_window = getattr(self.backend, "context_window", None)
            if context_window in (None, "", 0, "0"):
                return 2**63 - 1
            return int(context_window) - self.count_tokens(messages, tools=tools)

        @property
        def capabilities(self):
            return self.model_info

        @property
        def model_info(self) -> ModelInfo:
            configured = getattr(self.backend, "model_info", {}) or {}
            return ModelInfo(
                vision=bool(configured.get("vision", getattr(self.backend, "vision", False))),
                function_calling=bool(
                    configured.get(
                        "function_calling", getattr(self.backend, "supports_native_tools", False)
                    )
                    and getattr(self.backend, "supports_native_tools", False)
                ),
                json_output=bool(
                    configured.get(
                        "json_output", getattr(self.backend, "supports_json_output", False)
                    )
                    and getattr(self.backend, "supports_json_output", False)
                ),
                family=ModelFamily.UNKNOWN,
                structured_output=bool(
                    configured.get(
                        "structured_output",
                        getattr(self.backend, "supports_structured_output", False),
                    )
                    and getattr(self.backend, "supports_structured_output", False)
                ),
            )

    class PlainClient(InjectionClient):
        """SelectorGroupChat 选下一个发言者用的「无编排/无注入」客户端：直接喂对话给 backend 生成
        。"""

        def __init__(
            self,
            backend,
            ctx=None,
            max_new_tokens: int = 64,
            model_id: str = "qwen-3-4b",
            request_overrides=None,
            model_context_policy=None,
            deployment_id: str | None = None,
        ):
            super().__init__(
                backend,
                role="Selector",
                ctx=ctx,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
                request_overrides=request_overrides,
                model_context_policy=model_context_policy,
                deployment_id=deployment_id,
            )

        async def create(
            self,
            messages,
            *,
            tools=(),
            tool_choice="auto",
            json_output=None,
            extra_create_args: Mapping[str, Any] | None = None,
            cancellation_token: Optional[Any] = None,
        ) -> CreateResult:
            chat = _to_chat(messages)
            sender = next(
                (
                    message.get("source")
                    for message in reversed(chat)
                    if message.get("role") != "system"
                ),
                None,
            )
            turn = self.ctx.turn_of(self.role)
            autogen_model_messages = _trace_messages(chat)
            backend_messages = _trace_messages(chat)
            chat, effective_overrides, allocation = _prepare_backend_request(
                self.backend,
                chat,
                tools=tools,
                configured_max_new_tokens=self.max_new_tokens,
                request_overrides=self.request_overrides,
                model_context_policy=self.model_context_policy,
            )
            effective_max_new_tokens = allocation.effective_max_new_tokens
            request_transport_fields = _request_transport_span_fields(
                self.backend, effective_max_new_tokens
            )
            backend_messages = _trace_messages(chat)
            input_chars = sum(len(str(message.get("content", ""))) for message in chat)
            start_message_fields = _model_call_start_message_fields(
                self.ctx,
                autogen_model_messages=autogen_model_messages,
                role_visible_messages=autogen_model_messages,
                backend_messages=backend_messages,
            )
            model_call_index = self.ctx.reserve_model_call(self.role, controller=True)
            span_id = self.ctx.log_span(
                "model_call_start",
                deployment_instance_id=self.deployment_id,
                role=self.role,
                turn=turn,
                sender=sender,
                controller=True,
                model_call_index=model_call_index,
                request_seed=getattr(self.ctx, "generation_seed", None),
                max_model_calls_per_case=self.ctx.max_model_calls_per_case,
                message_count=len(chat),
                input_chars=input_chars,
                tool_count=0,
                json_output_requested=bool(json_output),
                max_new_tokens=effective_max_new_tokens,
                invocation_policy=effective_overrides,
                memory_channel="none",
                routing_reason="group_chat_controller",
                trace_detail_level=str(getattr(self.ctx, "trace_detail_level", "compact")),
                **allocation.span_fields(),
                **_thinking_budget_span_fields(effective_overrides),
                **request_transport_fields,
                **start_message_fields,
            )
            trace_model_calls = bool(getattr(self.ctx, "trace_model_calls", True))
            if trace_model_calls:
                print(
                    f"[model:{self.role}] start call={model_call_index}/"
                    f"{self.ctx.max_model_calls_per_case or '∞'} turn={turn} "
                    f"sender={sender or '-'} "
                    f"messages={len(chat)} chars={input_chars} tools=0 "
                    f"max_new_tokens={effective_max_new_tokens}/"
                    f"{self.max_new_tokens} "
                    f"timeout_s={request_transport_fields.get('request_timeout_s', '-')} "
                    f"retries={request_transport_fields.get('request_max_retries', '-')}",
                    flush=True,
                )
            backend_kwargs = {
                "max_new_tokens": effective_max_new_tokens,
                "request_seed": getattr(self.ctx, "generation_seed", None),
            }
            if effective_overrides and getattr(self.backend, "supports_request_overrides", False):
                backend_kwargs["request_overrides"] = effective_overrides
            backend_kwargs.update(_backend_extra_create_args(self.backend, extra_create_args or {}))
            if json_output is not None:
                backend_kwargs["json_output"] = json_output
            try:
                g = await _call_backend(
                    self.backend,
                    "generate_chat",
                    chat,
                    cancellation_token=cancellation_token,
                    **backend_kwargs,
                )
            except Exception as exc:
                from ...runtime.spans import exception_record

                self.ctx.log_span(
                    "model_call_error",
                    parent_span_id=span_id,
                    deployment_instance_id=self.deployment_id,
                    role=self.role,
                    turn=turn,
                    sender=sender,
                    controller=True,
                    model_call_index=model_call_index,
                    **request_transport_fields,
                    **exception_record(exc),
                )
                raise
            usage = RequestUsage(prompt_tokens=g.n_prompt_pos, completion_tokens=g.n_gen_tokens)
            self._actual_usage = _add_usage(self._actual_usage, usage)
            self._total_usage = _add_usage(self._total_usage, usage)
            self.ctx.bump_turn(self.role)
            finish_reason = str(getattr(g, "finish_reason", None) or "stop")
            reasoning_content = str(getattr(g, "reasoning_content", "") or "")
            request_queue_latency_s = _optional_seconds(g, "request_queue_latency_s")
            timing_fields = _model_timing_fields(g)
            self.ctx.log_decision(
                self.role,
                turn,
                sender,
                RouteDecision(channel="none", reason="group_chat_controller"),
                {
                    "controller": True,
                    "model_call_index": model_call_index,
                    "deployment_instance_id": self.deployment_id,
                    "input_positions": int(g.n_prompt_pos),
                    "text_input_tokens": int(g.n_prompt_pos),
                    "latent_prefix_positions": 0,
                    "output_tokens": int(g.n_gen_tokens),
                    "output_reasoning_tokens": getattr(g, "reasoning_tokens", None),
                    "output_answer_tokens": getattr(g, "answer_tokens", None),
                    "output_reasoning_tokens_source": getattr(g, "reasoning_tokens_source", None),
                    "output_answer_tokens_source": getattr(g, "answer_tokens_source", None),
                    "input_cached_tokens": getattr(g, "cached_input_tokens", None),
                    "input_cached_tokens_source": getattr(g, "cached_input_tokens_source", None),
                    "input_cached_tokens_status": getattr(
                        g, "cached_input_tokens_status", "not_reported"
                    ),
                    "model_generation_latency_s": round(float(g.latency_s), 3),
                    "request_queue_latency_s": request_queue_latency_s,
                    **timing_fields,
                    "finish_reason": finish_reason,
                    "invocation_policy": effective_overrides,
                    "prompt_pos": int(g.n_prompt_pos),
                    "gen_tokens": int(g.n_gen_tokens),
                    "prefix_len": 0,
                    "latency_s": round(float(g.latency_s), 3),
                },
            )
            self.ctx.log_span(
                "model_call_end",
                parent_span_id=span_id,
                deployment_instance_id=self.deployment_id,
                role=self.role,
                turn=turn,
                sender=sender,
                controller=True,
                model_call_index=model_call_index,
                input_total_positions=int(g.n_prompt_pos),
                input_text_tokens=int(g.n_prompt_pos),
                input_latent_positions=0,
                output_text_tokens=int(g.n_gen_tokens),
                model_latency_s=round(float(g.latency_s), 3),
                request_queue_latency_s=request_queue_latency_s,
                **timing_fields,
                finish_reason=finish_reason,
                invocation_policy=effective_overrides,
                provider_request_payload=getattr(g, "provider_request_payload", {}),
                provider_response_payload=getattr(g, "provider_response_payload", {}),
                response_payload_origin=getattr(g, "response_payload_origin", None),
                output_total_tokens=int(g.n_gen_tokens),
                input_cached_tokens=getattr(g, "cached_input_tokens", None),
                input_cached_tokens_source=getattr(g, "cached_input_tokens_source", None),
                input_cached_tokens_status=getattr(g, "cached_input_tokens_status", "not_reported"),
                output_reasoning_tokens=getattr(g, "reasoning_tokens", None),
                output_answer_tokens=getattr(g, "answer_tokens", None),
                output_reasoning_tokens_source=getattr(g, "reasoning_tokens_source", None),
                output_answer_tokens_source=getattr(g, "answer_tokens_source", None),
                output_reasoning_chars=len(reasoning_content),
                output_answer_chars=len(g.text),
                **allocation.span_fields(),
                **_thinking_budget_span_fields(effective_overrides),
                **_completed_request_transport_span_fields(g, request_transport_fields),
            )
            if trace_model_calls:
                print(
                    f"[model:{self.role}] done turn={turn} input_pos={g.n_prompt_pos} "
                    f"output_tokens={g.n_gen_tokens} latency_s={round(float(g.latency_s), 3)} "
                    f"finish_reason={finish_reason} preview={_preview_text(g.text)!r}",
                    flush=True,
                )
            return CreateResult(
                finish_reason=finish_reason,
                content=g.text,
                usage=usage,
                cached=False,
                thought=reasoning_content or None,
            )

    return InjectionClient, PlainClient


def make_injection_client(
    backend,
    role: str,
    ctx,
    max_new_tokens: int = 256,
    model_id: str = "qwen-3-4b",
    request_overrides=None,
    model_context_policy=None,
    deployment_id: str | None = None,
):
    """工厂：构造一个 InjectionClient 实例（首次调用时才导入 autogen）。"""
    InjectionClient, _ = _build_injection_client_class()
    return InjectionClient(
        backend,
        role=role,
        ctx=ctx,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
        request_overrides=request_overrides,
        model_context_policy=model_context_policy,
        deployment_id=deployment_id,
    )


def make_plain_client(
    backend,
    ctx=None,
    max_new_tokens: int = 64,
    model_id: str = "qwen-3-4b",
    request_overrides=None,
    model_context_policy=None,
    deployment_id: str | None = None,
):
    """工厂：构造一个 PlainClient（selector 用，无注入）。"""
    _, PlainClient = _build_injection_client_class()
    return PlainClient(
        backend,
        ctx=ctx,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
        request_overrides=request_overrides,
        model_context_policy=model_context_policy,
        deployment_id=deployment_id,
    )


@REGISTRY.register("model_client", "injection")
class InjectionClientFactory:
    """`model_client/injection` 的注册入口：延迟到 .build() 时才构造 autogen 子类实例。

    这样 import / REGISTRY.snapshot() 不触发 autogen 导入（黄金法则 1+4）。
    """

    name = "injection"

    def __init__(self, **defaults):
        self.defaults = defaults

    def build(self, backend, role: str, ctx, **kwargs):
        params = {**self.defaults, **kwargs}
        return make_injection_client(backend, role, ctx, **params)
