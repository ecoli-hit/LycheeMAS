"""AutoGenRuntime -- map one explicit TeamSpec GroupChat to AutoGen and run it.

Each graph must declare exactly one of ``round_robin``, ``selector`` or
``magentic_one`` in ``graph.meta["group_chat"]``.  The runtime does not infer a
chat implementation from participant count or legacy scheduler/protocol fields.

路由信号经共享 RoutingContext 流动——每个 agent 的 InjectionClient 都读它。答案提取按任务可定制
（见 eval/task_config.extractor_for_task）。`run` 返回统一的 `Trajectory`，并把每条消息 + 决策喂给
intercept hook（写 TraceStore / 抽取）。

⚠️ 这是【唯一允许 import autogen 的文件之一】（CLAUDE.md 黄金法则 1：runtime/backends/autogen_*.py
）。
autogen_agentchat 的导入全部惰性化到方法内部，保证 import 本模块（与注册）不需要 autogen。
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import urlsplit

from ...core.registry import REGISTRY
from ...core.types import Answer, Message, TaskQuery, Trajectory
from ..base import BaseRuntime, MASGraph
from ..group_chat import normalize_group_chat_config


class _DockerContainerUserProxy:
    """Run Docker exec commands as the host user for bind-mounted workspaces."""

    def __init__(self, container: Any, user: str):
        self._container = container
        self._user = user

    def __getattr__(self, name: str) -> Any:
        return getattr(self._container, name)

    def exec_run(self, command: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("user", self._user)
        return self._container.exec_run(command, *args, **kwargs)

GAIA_FINAL_ANSWER_PROMPT = "\n".join(
    [
        ",",
        "We have completed the following task:",
        "",
        "{task}",
        "",
        "The above messages contain the conversation that took place to complete the task.",
        "Read the above conversation and output a FINAL ANSWER to the question.",
        (
            "To output the final answer, use the following template: "
            "FINAL ANSWER: [YOUR FINAL ANSWER]"
        ),
        (
            "Your FINAL ANSWER should be a number OR as few words as possible OR a comma "
            "separated list of numbers and/or strings."
        ),
        (
            "ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions "
            "specified in the original question (e.g., alphabetization, sequencing, units, "
            "rounding, decimal places, etc.)"
        ),
        (
            "If you are asked for a number, express it numerically (i.e., with digits rather "
            "than words), don't use commas, and don't include units such as $ or percent signs "
            "unless specified otherwise."
        ),
        (
            "If you are asked for a string, don't use articles or abbreviations (e.g. for "
            "cities), unless specified otherwise. Don't output any final sentence punctuation "
            "such as '.', '!', or '?'."
        ),
        (
            "If you are asked for a comma separated list, apply the above rules depending on "
            "whether the elements are numbers or strings."
        ),
    ]
)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "case"


def _content_to_text(content: Any) -> str:
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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _structured_tool_records(event: Any, event_index: int) -> tuple[list[dict], list[dict]]:
    """Normalize only typed AutoGen tool events; never infer calls from role names."""

    source = str(getattr(event, "source", "?") or "?")
    msg_type = str(getattr(event, "type", type(event).__name__))
    requests: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    if msg_type == "ToolCallRequestEvent":
        for call in getattr(event, "content", []) or []:
            requests.append(
                {
                    "record_type": "tool_request",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": str(getattr(call, "id", "") or ""),
                    "tool_name": str(getattr(call, "name", "") or ""),
                    "arguments": str(getattr(call, "arguments", "") or "{}"),
                    "origin": "autogen_tool_call_request_event",
                }
            )
    elif msg_type == "ToolCallExecutionEvent":
        for result in getattr(event, "content", []) or []:
            executions.append(
                {
                    "record_type": "tool_execution",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": str(getattr(result, "call_id", "") or ""),
                    "tool_name": str(getattr(result, "name", "") or ""),
                    "is_error": bool(getattr(result, "is_error", False)),
                    "output": str(getattr(result, "content", "") or ""),
                    "origin": "autogen_tool_call_execution_event",
                }
            )
    elif msg_type == "CodeExecutionEvent":
        result = getattr(event, "result", None)
        if result is not None:
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            executions.append(
                {
                    "record_type": "tool_execution",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": "",
                    "tool_name": "code_executor",
                    "exit_code": exit_code,
                    "is_error": exit_code != 0,
                    "output": str(getattr(result, "output", "") or ""),
                    "origin": "autogen_code_execution_event",
                }
            )
    return requests, executions


def _tool_requests_from_decisions(decisions: Sequence[dict]) -> list[dict[str, Any]]:
    records = []
    for decision in decisions:
        for call in decision.get("tool_call_request") or []:
            records.append(
                {
                    "record_type": "tool_request",
                    "source": str(decision.get("role") or "?"),
                    "turn": decision.get("turn_index", decision.get("turn")),
                    "tool_call_id": str(call.get("id") or ""),
                    "tool_name": str(call.get("name") or ""),
                    "arguments": str(call.get("arguments") or "{}"),
                    "origin": "model_client_function_call",
                }
            )
    return records


def _deduplicate_tool_records(records: Sequence[dict]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for record in records:
        call_id = str(record.get("tool_call_id") or "")
        key = (
            record.get("record_type"),
            call_id or None,
            record.get("source"),
            record.get("tool_name"),
            None if call_id else record.get("event_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(record))
    return unique


def _tool_operations(requests: Sequence[dict], executions: Sequence[dict]) -> list[dict]:
    """Pair request/result records by call id and retain standalone code executions."""

    operations = [dict(request, execution_status="requested") for request in requests]
    by_id = {
        str(operation.get("tool_call_id")): operation
        for operation in operations
        if operation.get("tool_call_id")
    }
    for execution in executions:
        call_id = str(execution.get("tool_call_id") or "")
        if call_id and call_id in by_id:
            operation = by_id[call_id]
            operation["execution_status"] = (
                "error" if execution.get("is_error") else "completed"
            )
            operation["execution"] = dict(execution)
        else:
            operations.append(
                {
                    "record_type": "tool_operation",
                    "source": execution.get("source"),
                    "tool_call_id": call_id,
                    "tool_name": execution.get("tool_name"),
                    "execution_status": (
                        "error" if execution.get("is_error") else "completed"
                    ),
                    "execution": dict(execution),
                    "origin": execution.get("origin"),
                }
            )
    return operations


def _is_web_surfer_operation_error(event: Any) -> bool:
    """Recognize AutoGen WebSurfer's documented exception response separately."""

    return bool(
        getattr(event, "type", type(event).__name__) == "TextMessage"
        and str(getattr(event, "content", "")).startswith("Web surfing error:\n\n")
    )


def _web_surfer_error_details(content: Any) -> dict[str, Any]:
    """Extract diagnostic facts without changing WebSurfer failure behavior."""

    text = str(content or "")
    error_codes = sorted(set(re.findall(r"\bERR_[A-Z0-9_]+\b", text)))
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s<>\]\[\"']+", text)))
    lowered = text.lower()
    if "err_network_changed" in lowered:
        category = "network_changed"
    elif "timeout" in lowered or "timed out" in lowered:
        category = "timeout"
    elif "font" in lowered:
        category = "font_or_rendering"
    elif "dns" in lowered or "name_not_resolved" in lowered:
        category = "dns"
    elif "proxy" in lowered or "tunnel" in lowered:
        category = "proxy"
    else:
        category = "other"
    return {
        "error_category": category,
        "browser_error_codes": error_codes,
        "referenced_urls": urls[:10],
        "referenced_domains": list(
            dict.fromkeys(urlsplit(url).hostname for url in urls if urlsplit(url).hostname)
        )[:10],
    }


def _safe_proxy_endpoint(proxy_url: str | None) -> str | None:
    if not proxy_url:
        return None
    parsed = urlsplit(proxy_url)
    if not parsed.hostname:
        return "configured"
    return f"{parsed.scheme or 'http'}://{parsed.hostname}:{parsed.port or 80}"


class _ObservedCodeExecutor:
    """Record typed CodeResult objects while preserving the executor interface."""

    def __init__(self, wrapped: Any, ctx: Any, records: list[dict[str, Any]]):
        self._wrapped = wrapped
        self._ctx = ctx
        self._records = records

    async def start(self) -> None:
        await self._wrapped.start()

    async def stop(self) -> None:
        await self._wrapped.stop()

    async def restart(self) -> None:
        await self._wrapped.restart()

    async def execute_code_blocks(self, code_blocks, cancellation_token):
        call_id = f"code_{_safe_id(str(time.time_ns()))}"
        started = time.time()
        span_id = self._ctx.log_span(
            "tool_execution_start",
            tool_call_id=call_id,
            tool_name="code_executor",
            origin="code_executor_adapter",
            code_blocks=_jsonable(code_blocks),
        )
        try:
            result = await self._wrapped.execute_code_blocks(
                code_blocks,
                cancellation_token=cancellation_token,
            )
        except BaseException as exc:
            from ..spans import exception_record

            record = {
                "record_type": "tool_execution",
                "source": "ComputerTerminal",
                "tool_call_id": call_id,
                "tool_name": "code_executor",
                "is_error": True,
                "duration_s": round(time.time() - started, 3),
                "origin": "code_executor_adapter",
                **exception_record(exc),
            }
            self._records.append(record)
            self._ctx.log_span(
                "tool_execution_error",
                parent_span_id=span_id,
                **record,
            )
            raise
        exit_code = int(getattr(result, "exit_code", 0) or 0)
        record = {
            "record_type": "tool_execution",
            "source": "ComputerTerminal",
            "tool_call_id": call_id,
            "tool_name": "code_executor",
            "exit_code": exit_code,
            "is_error": exit_code != 0,
            "output": str(getattr(result, "output", "") or ""),
            "duration_s": round(time.time() - started, 3),
            "origin": "code_executor_adapter",
        }
        self._records.append(record)
        self._ctx.log_span(
            "tool_execution_end",
            parent_span_id=span_id,
            **record,
        )
        return result


def _extract_final_answer(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", type(msg).__name__) == "ThoughtEvent":
            continue
        text = _content_to_text(getattr(msg, "content", ""))
        match = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _agent_type(node) -> str:
    return str(node.meta.get("agent_type") or node.meta.get("type") or "assistant").lower()


def _has_tool_agents(graph: MASGraph) -> bool:
    return any(
        _agent_type(node) != "assistant" or node.tools or node.meta.get("tools")
        for node in graph.order()
    )


def _group_chat_config(graph: MASGraph) -> dict[str, Any]:
    """Return the explicit GroupChat config carried by the TeamSpec graph."""
    return normalize_group_chat_config(graph.meta.get("group_chat"))


def topology_selector(graph: MASGraph) -> Callable:
    """Select a valid downstream role from the TeamGraph deterministically."""

    order = list(graph.meta.get("speaking_order") or graph.names)
    indexes = {name: index for index, name in enumerate(order)}
    state = {"fallback": 0}

    def _source(message) -> str:
        return str(getattr(message, "source", "") or getattr(message, "name", ""))

    def _selector(messages) -> str:
        previous = next((_source(item) for item in reversed(messages) if _source(item)), "")
        allowed = list(graph.edges.get(previous) or [])
        if allowed:
            return min(
                allowed,
                key=lambda name: (indexes.get(name, len(order)), name),
            )
        if previous in indexes and order:
            return order[(indexes[previous] + 1) % len(order)]
        name = order[state["fallback"] % len(order)]
        state["fallback"] += 1
        return name

    return _selector


def topology_candidates(graph: MASGraph) -> Callable:
    """Return an AutoGen candidate_func constrained by outgoing TeamGraph edges."""

    order = list(graph.meta.get("speaking_order") or graph.names)
    indexes = {name: index for index, name in enumerate(order)}

    def _source(message) -> str:
        return str(getattr(message, "source", "") or getattr(message, "name", ""))

    def _candidates(messages) -> list[str]:
        previous = next((_source(item) for item in reversed(messages) if _source(item)), "")
        allowed = list(graph.edges.get(previous) or [])
        if allowed:
            return sorted(allowed, key=lambda name: (indexes.get(name, len(order)), name))
        if previous in indexes and order:
            return [order[(indexes[previous] + 1) % len(order)]]
        return list(order)

    return _candidates


def _termination_condition(team: MASGraph):
    """Compile TeamSpec text conditions to AutoGen TerminationCondition objects."""

    from autogen_agentchat.conditions import TextMentionTermination

    conditions = list((team.meta.get("termination") or {}).get("conditions") or [])
    compiled = []
    for raw in conditions:
        item = dict(raw) if isinstance(raw, dict) else {"type": str(raw)}
        condition_type = str(item.get("type") or "").lower()
        if condition_type in {"approve", "terminate"}:
            item = {"type": "text_mention", "text": condition_type.upper()}
            condition_type = "text_mention"
        if condition_type != "text_mention":
            raise ValueError(f"unsupported termination condition {condition_type!r}")
        text = str(item.get("text") or "")
        if not text:
            raise ValueError("text_mention termination requires non-empty text")
        sources = item.get("sources")
        compiled.append(
            TextMentionTermination(
                text,
                sources=[str(source) for source in sources] if sources else None,
            )
        )
    if not compiled:
        return None
    condition = compiled[0]
    for extra in compiled[1:]:
        condition = condition | extra
    return condition


def _model_call_limit_termination(ctx: Any, limit: Optional[int]):
    """Stop cleanly between GroupChat turns once the per-case call limit is reached."""

    if limit is None:
        return None
    from autogen_agentchat.base import TerminatedException, TerminationCondition
    from autogen_agentchat.messages import StopMessage

    class ModelCallLimitTermination(TerminationCondition):
        def __init__(self) -> None:
            self._terminated = False

        @property
        def terminated(self) -> bool:
            return self._terminated

        async def __call__(self, messages):
            if self._terminated:
                raise TerminatedException("Model-call limit termination already reached")
            if int(getattr(ctx, "model_calls_started", 0)) >= limit:
                self._terminated = True
                ctx.model_call_budget_exhausted = True
                ctx.log_span(
                    "model_call_budget_stop",
                    model_calls_started=ctx.model_calls_started,
                    max_model_calls_per_case=limit,
                )
                return StopMessage(
                    content=f"Maximum model calls per case {limit} reached.",
                    source="ModelCallLimitTermination",
                )
            return None

        async def reset(self) -> None:
            self._terminated = False

    return ModelCallLimitTermination()


def _effective_max_turns(
    group_chat_type: str,
    participant_count: int,
    *,
    max_rounds: int,
    max_turns: Optional[int],
) -> int:
    """Resolve the one AutoGen turn limit used by a concrete GroupChat."""

    if max_turns is not None:
        return int(max_turns)
    if group_chat_type == "round_robin":
        return max(1, participant_count) * int(max_rounds)
    return 20


def _builtin_workspace_tools(workspace: Optional[Path]) -> dict[str, Callable]:
    if workspace is None:
        return {}

    def list_workspace() -> str:
        """List files available in the current workspace."""
        return "\n".join(sorted(path.name for path in workspace.iterdir()))

    def read_text_file(path: str) -> str:
        """Read a UTF-8 text file from the current workspace."""
        target = (workspace / path).resolve()
        if workspace.resolve() not in target.parents and target != workspace.resolve():
            raise ValueError("path must stay inside the workspace")
        return target.read_text(encoding="utf-8", errors="replace")

    return {"list_workspace": list_workspace, "read_text_file": read_text_file}


def _resolve_function_tools(
    tool_names: Sequence[Any],
    workspace: Optional[Path],
    *,
    query_meta: Optional[dict[str, Any]] = None,
    resources: Optional[list[Any]] = None,
    benchmark: Any = None,
) -> list[Any]:
    builtins = _builtin_workspace_tools(workspace)
    tools: list[Any] = []
    seen: set[str] = set()
    for item in tool_names:
        key = getattr(item, "__name__", str(item))
        if key in seen:
            continue
        seen.add(key)
        if callable(item):
            tools.append(item)
        elif isinstance(item, str) and item.startswith("benchmark:"):
            if benchmark is None:
                raise ValueError(f"benchmark tool slot {item!r} has no active adapter")
            bundle = benchmark.create_tools(item, query_meta or {})
            tools.extend(bundle.tools)
            if resources is not None:
                resources.extend(bundle.resources)
        elif isinstance(item, str) and item in builtins:
            tools.append(builtins[item])
        elif isinstance(item, str) and ":" in item:
            module_name, func_name = item.split(":", 1)
            module = __import__(module_name, fromlist=[func_name])
            tools.append(getattr(module, func_name))
    return tools


def _configure_pydub_ffmpeg() -> Optional[str]:
    """Bind pydub to the benchmark extra's bundled ffmpeg binary when needed."""

    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Couldn't find ffmpeg or avconv.*",
            category=RuntimeWarning,
        )
        from pydub import AudioSegment

    AudioSegment.converter = executable
    return executable


def _apply_official_message_filter(agent: Any, node: Any, context_visibility: str):
    """Wrap one participant with AutoGen's official source-based message filter."""

    if context_visibility != "topology_filtered":
        return agent
    from autogen_agentchat.agents import (
        MessageFilterAgent,
        MessageFilterConfig,
        PerSourceFilter,
    )

    sources = ["user"]
    sources.extend(str(source) for source in (node.meta.get("receives_from") or []))
    sources.append(str(node.name))
    unique_sources = list(dict.fromkeys(sources))
    return MessageFilterAgent(
        name=node.name,
        wrapped_agent=agent,
        filter=MessageFilterConfig(
            per_source=[PerSourceFilter(source=source) for source in unique_sources]
        ),
    )


def _build_agents(
    graph: MASGraph,
    backend,
    ctx,
    max_new_tokens: int = 256,
    model_id: str = "qwen-3-4b",
    workspace: Optional[Path] = None,
    code_executor=None,
    web_headless: bool = True,
    save_screenshots: bool = False,
    web_contexts: Optional[dict[str, dict[str, Any]]] = None,
    backend_resolver=None,
    model_resolver=None,
    max_new_tokens_resolver=None,
    invocation_overrides_resolver=None,
    query_meta: Optional[dict[str, Any]] = None,
    resources: Optional[list[Any]] = None,
    benchmark: Any = None,
) -> List[Any]:
    """把 MASGraph 的节点实例化成 AutoGen agents.

    默认仍是 AssistantAgent + InjectionClient；当 AgentSpec.meta.agent_type
    声明工具型 participant 时，构造对应的 Coder/FileSurfer/WebSurfer/Terminal。
    """
    from autogen_agentchat.agents import AssistantAgent

    from .autogen_injection_client import make_injection_client

    agents = []
    for node in graph.order():
        deployment_id = node.meta.get("deployment_id")
        node_backend = backend_resolver(deployment_id) if backend_resolver else backend
        resolved_model = model_resolver(deployment_id) if model_resolver else None
        mid = node.model or resolved_model or model_id
        node_max_new_tokens = (
            max_new_tokens_resolver(node.name, max_new_tokens)
            if max_new_tokens_resolver
            else max_new_tokens
        )
        node_invocation_overrides = (
            invocation_overrides_resolver(node.name)
            if invocation_overrides_resolver
            else None
        )
        model_context_policy = dict(
            node.meta.get("model_context")
            or graph.meta.get("model_context")
            or {"type": "unbounded"}
        )
        ctx.model_of_role[node.name] = mid  # 登记角色->模型（驱动约束 #2 的同模型对判断）
        ctx.deployment_of_role[node.name] = str(deployment_id or "")
        client = make_injection_client(
            node_backend,
            role=node.name,
            ctx=ctx,
            max_new_tokens=node_max_new_tokens,
            model_id=mid,
            request_overrides=node_invocation_overrides,
            model_context_policy=model_context_policy,
            deployment_id=str(deployment_id or ""),
        )
        desc = node.meta.get("description") or node.name
        kind = _agent_type(node)
        if kind == "assistant":
            tools = _resolve_function_tools(
                [*(node.tools or []), *(node.meta.get("tools") or [])],
                workspace,
                query_meta=query_meta,
                resources=resources,
                benchmark=benchmark,
            )
            agent = AssistantAgent(
                node.name,
                model_client=client,
                tools=tools or None,
                system_message=node.system_prompt,
                description=desc,
                model_context=_build_model_context(model_context_policy, client),
                max_tool_iterations=int(node.meta.get("max_tool_iterations", 1)),
            )
        elif kind in {"coder", "magentic_one_coder"}:
            from autogen_ext.agents.magentic_one._magentic_one_coder_agent import (
                MAGENTIC_ONE_CODER_DESCRIPTION,
                MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
            )

            agent = AssistantAgent(
                node.name,
                model_client=client,
                description=MAGENTIC_ONE_CODER_DESCRIPTION,
                system_message=MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
                model_context=_build_model_context(model_context_policy, client),
            )
        elif kind in {"computer_terminal", "terminal", "code_executor"}:
            from autogen_agentchat.agents import CodeExecutorAgent

            if code_executor is None:
                raise ValueError("computer_terminal agent requires a code_executor")
            sources = node.meta.get("sources") if "sources" in node.meta else None
            agent = CodeExecutorAgent(
                node.name,
                code_executor=code_executor,
                sources=[str(source) for source in sources] if sources is not None else None,
            )
        elif kind == "file_surfer":
            _configure_pydub_ffmpeg()
            from autogen_ext.agents.file_surfer import FileSurfer

            if workspace is None:
                raise ValueError("file_surfer agent requires a workspace")
            agent = FileSurfer(node.name, model_client=client, base_path=str(workspace))
        elif kind == "web_surfer":
            from autogen_ext.agents.web_surfer import MultimodalWebSurfer

            if workspace is None:
                raise ValueError("web_surfer agent requires a workspace")
            web_runtime = (web_contexts or {}).get(node.name, {})
            agent = MultimodalWebSurfer(
                node.name,
                model_client=client,
                downloads_folder=str(workspace),
                debug_dir=str(workspace / "web_logs"),
                headless=web_headless,
                to_save_screenshots=save_screenshots,
                playwright=web_runtime.get("playwright"),
                context=web_runtime.get("context"),
            )
        else:
            raise ValueError(f"unknown agent_type {kind!r} for node {node.name!r}")
        agents.append(
            _apply_official_message_filter(agent, node, ctx.context_visibility)
        )
    return agents


def _build_model_context(policy: dict[str, Any] | None, model_client):
    """Instantiate the corresponding official AutoGen model-context class."""

    from autogen_core.model_context import (
        BufferedChatCompletionContext,
        TokenLimitedChatCompletionContext,
        UnboundedChatCompletionContext,
    )

    from ..token_budget import normalize_model_context_policy

    normalized = normalize_model_context_policy(policy)
    if normalized["type"] == "buffered":
        return BufferedChatCompletionContext(buffer_size=normalized["buffer_size"])
    if normalized["type"] == "token_limited":
        return TokenLimitedChatCompletionContext(
            model_client,
            token_limit=normalized.get("token_limit"),
        )
    return UnboundedChatCompletionContext()


@REGISTRY.register("runtime", "autogen")
class AutoGenRuntime(BaseRuntime):
    """Instantiate the TeamSpec's explicit AutoGen GroupChat and return a Trajectory."""

    name = "autogen"

    def __init__(
        self,
        backend=None,
        ctx=None,
        max_new_tokens: int = 256,
        max_rounds: int = 2,
        model_id: str = "qwen-3-4b",
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
    ):
        super().__init__()
        self.backend = backend  # 共享 HFBackend（None 时延迟到 run 前必须由调用方注入）
        self.ctx = ctx  # 共享 RoutingContext（携带 router/memory/task）
        self.max_new_tokens = max_new_tokens
        self.max_rounds = max_rounds
        self.model_id = model_id
        self.max_turns = max_turns
        self.max_model_calls_per_case = max_model_calls_per_case
        self.max_stalls = max_stalls
        self.work_root = Path(work_root)
        self.code_executor = code_executor
        self.docker_image = docker_image
        self.code_timeout = code_timeout
        self.web_headless = web_headless
        self.save_screenshots = save_screenshots
        self.web_proxy_url = str(web_proxy_url or "").strip() or None
        self.container_proxy_url = str(container_proxy_url or "").strip() or None
        self.proxy_no_proxy = str(proxy_no_proxy or "127.0.0.1,localhost,::1")
        self.trace_model_calls = trace_model_calls
        self.backend_resolver = backend_resolver
        self.model_resolver = model_resolver
        self.control_deployment_id = control_deployment_id
        self.max_new_tokens_resolver = max_new_tokens_resolver
        self.control_max_new_tokens = control_max_new_tokens
        self.invocation_overrides_resolver = invocation_overrides_resolver
        self.control_invocation_overrides = dict(control_invocation_overrides or {})

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if (self.backend is None and self.backend_resolver is None) or self.ctx is None:
            raise ValueError("AutoGenRuntime.run 需要 backend 与 ctx（构造时或运行前注入）")
        configure_graph = getattr(self.ctx, "configure_graph", None)
        if configure_graph is not None:
            configure_graph(team)

        from ...eval.benchmarks import BENCHMARKS

        benchmark = BENCHMARKS.for_query(self.ctx.task, query.meta)

        uses_tools = _has_tool_agents(team)
        workspace: Optional[Path] = None
        task_text = query.question
        copied_files: list[str] = []
        workspace_info: dict[str, Any] = {}
        code_executor = None
        code_executor_started = False
        code_executor_t0: float | None = None
        runtime_span_id: str | None = None
        code_executor_span_id: str | None = None
        agents: list[Any] = []
        code_execution_records: list[dict[str, Any]] = []
        stream_tool_requests: list[dict[str, Any]] = []
        stream_tool_executions: list[dict[str, Any]] = []
        tool_agent_errors: list[dict[str, Any]] = []
        web_contexts: dict[str, dict[str, Any]] = {}
        tool_resources: list[Any] = []
        t0 = time.time()

        try:
            group_chat = _group_chat_config(team)
            effective_max_turns = int(
                group_chat.get("max_turns")
                or _effective_max_turns(
                    str(group_chat.get("type") or "round_robin"),
                    len(team.order()),
                    max_rounds=self.max_rounds,
                    max_turns=self.max_turns,
                )
            )
            self.ctx.max_model_calls_per_case = self.max_model_calls_per_case
            runtime_span_id = self.ctx.log_span(
                "runtime_start",
                runtime=self.name,
                team=team.meta.get("team"),
                group_chat=group_chat,
                group_chat_type=group_chat["type"],
                uses_tools=uses_tools,
                max_turns=effective_max_turns,
                max_rounds=self.max_rounds,
                max_model_calls_per_case=self.max_model_calls_per_case,
                code_executor=self.code_executor,
                docker_image=self.docker_image if self.code_executor == "docker" else None,
                model_context=team.meta.get("model_context") or {"type": "unbounded"},
                controller_model_context=team.meta.get("controller_model_context")
                or {"type": "unbounded"},
                network={
                    "web_proxy_enabled": bool(self.web_proxy_url),
                    "container_proxy_enabled": bool(self.container_proxy_url),
                },
            )
            if uses_tools:
                workspace, task_text, copied_files, workspace_info = self._prepare_workspace(
                    query,
                    benchmark,
                )
                print(
                    f"[autogen:tools] workspace={workspace} copied_files={len(copied_files)}",
                    flush=True,
                )
                self.ctx.log_span(
                    "workspace_prepared",
                    parent_span_id=runtime_span_id,
                    workspace=str(workspace),
                    copied_files=copied_files,
                    task_text=task_text,
                    **workspace_info,
                )
                if self._needs_code_executor(team):
                    code_executor = _ObservedCodeExecutor(
                        self._build_code_executor(workspace),
                        self.ctx,
                        code_execution_records,
                    )
                    print(
                        f"[autogen:tools] starting code_executor={self.code_executor} "
                        f"image={self.docker_image}",
                        flush=True,
                    )
                    code_executor_t0 = time.time()
                    code_executor_span_id = self.ctx.log_span(
                        "code_executor_start",
                        parent_span_id=runtime_span_id,
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        timeout_s=self.code_timeout,
                        workspace=str(workspace),
                    )
                    await code_executor.start()
                    code_executor_started = True
                    print("[autogen:tools] code_executor ready", flush=True)
                    self.ctx.log_span(
                        "code_executor_ready",
                        parent_span_id=code_executor_span_id,
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        workspace=str(workspace),
                        startup_latency_s=round(time.time() - code_executor_t0, 3),
                    )

            self.ctx.trace_model_calls = (
                True if self.trace_model_calls is None else self.trace_model_calls
            )
            if any(
                _agent_type(node) == "web_surfer" for node in team.order()
            ):
                web_contexts = await self._build_web_contexts(team)
            if uses_tools:
                print(
                    f"[autogen:tools] building agents team={team.meta.get('team')} "
                    f"max_turns={effective_max_turns} "
                    f"max_model_calls_per_case={self.max_model_calls_per_case or 'unlimited'}",
                    flush=True,
                )
            agents = _build_agents(
                team,
                self.backend,
                self.ctx,
                max_new_tokens=self.max_new_tokens,
                model_id=self.model_id,
                workspace=workspace,
                code_executor=code_executor,
                web_headless=self.web_headless,
                save_screenshots=self.save_screenshots,
                web_contexts=web_contexts,
                backend_resolver=self.backend_resolver,
                model_resolver=self.model_resolver,
                max_new_tokens_resolver=self.max_new_tokens_resolver,
                invocation_overrides_resolver=self.invocation_overrides_resolver,
                query_meta=query.meta,
                resources=tool_resources,
                benchmark=benchmark,
            )
            task_payload = self._build_task_payload(query, task_text)
            gc = self._build_chat(team, agents, task_text)
            if uses_tools:
                print("[autogen:tools] running group chat", flush=True)
            (
                result,
                stream_tool_requests,
                stream_tool_executions,
                tool_agent_errors,
            ) = await self._run_group_chat_stream(gc, task_payload, team)
        except Exception as exc:
            from ..spans import exception_record

            self.ctx.log_span(
                "runtime_error",
                parent_span_id=runtime_span_id,
                runtime_elapsed_s=round(time.time() - t0, 3),
                **exception_record(exc),
            )
            raise
        finally:
            for agent in agents:
                close_target = getattr(agent, "_wrapped_agent", agent)
                close = getattr(close_target, "close", None)
                if close is not None and uses_tools:
                    await close()
            await self._close_web_contexts(web_contexts)
            for resource in tool_resources:
                close = getattr(resource, "close", None)
                if close is None:
                    continue
                result_close = close()
                if inspect.isawaitable(result_close):
                    await result_close
            if code_executor_started and code_executor is not None:
                await code_executor.stop()
                self.ctx.log_span(
                    "code_executor_stop",
                    parent_span_id=code_executor_span_id,
                    executor=self.code_executor,
                    docker_image=self.docker_image,
                    workspace=str(workspace) if workspace else None,
                    executor_wall_time_s=(
                        round(time.time() - code_executor_t0, 3)
                        if code_executor_t0 is not None
                        else None
                    ),
                )
        elapsed = time.time() - t0
        msgs = result.messages

        # 组装统一 Trajectory：逐条 AutoGen 消息 -> core.Message，并 emit 给 hook
        traj = Trajectory(
            task_id=query.id,
            meta={
                "runtime": self.name,
                "task": self.ctx.task,
                "team": team.meta.get("team"),
                "group_chat": _group_chat_config(team),
                "workspace": str(workspace) if workspace else None,
                "copied_files": copied_files,
                "workspace_info": workspace_info,
                "stop_reason": getattr(result, "stop_reason", None),
                "max_model_calls_per_case": self.max_model_calls_per_case,
                "model_calls_started": self.ctx.model_calls_started,
                "model_call_budget_exhausted": self.ctx.model_call_budget_exhausted,
                "case_wall_time_s": round(elapsed, 3),
            },
        )
        for r, m in enumerate(msgs):
            text = _content_to_text(getattr(m, "content", ""))
            usage = getattr(m, "models_usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            source = getattr(m, "source", "?")
            msg_type = getattr(m, "type", type(m).__name__)
            msg = Message(
                sender=source,
                content=text,
                round=r,
                role="assistant",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                meta={"autogen_type": msg_type},
            )
            traj.add(msg)
            self._emit(msg)

        default_text = _extract_final_answer(list(msgs)) or benchmark.extract_messages(
            self.ctx.task, msgs
        )
        final_text = benchmark.collect_prediction(
            query=query,
            messages=list(msgs),
            workspace=workspace,
            default_text=default_text,
        )
        traj.final_answer = Answer(
            content=final_text,
            source="verifier",
            meta={"n_messages": len(msgs), "case_wall_time_s": round(elapsed, 3)},
        )
        traj.candidates = [traj.final_answer]
        decisions = list(getattr(self.ctx, "decisions", []))
        traj.meta["decisions"] = decisions
        tool_requests = _deduplicate_tool_records(
            [*_tool_requests_from_decisions(decisions), *stream_tool_requests]
        )
        tool_executions = _deduplicate_tool_records(
            [*stream_tool_executions, *code_execution_records]
        )
        tool_calls = _tool_operations(tool_requests, tool_executions)
        traj.meta["tool_calls"] = tool_calls
        traj.meta["tool_requests"] = tool_requests
        traj.meta["tool_executions"] = tool_executions
        traj.meta["tool_agent_errors"] = tool_agent_errors
        traj.meta["tool_call_count"] = len(tool_calls)
        traj.meta["tool_request_count"] = len(tool_requests)
        traj.meta["tool_execution_count"] = len(tool_executions)
        traj.meta["tool_error_count"] = sum(
            1 for execution in tool_executions if execution.get("is_error")
        )
        traj.meta["tool_agent_error_count"] = len(tool_agent_errors)
        self.ctx.log_span(
            "runtime_end",
            parent_span_id=runtime_span_id,
            stop_reason=getattr(result, "stop_reason", None),
            max_model_calls_per_case=self.max_model_calls_per_case,
            model_calls_started=self.ctx.model_calls_started,
            model_call_budget_exhausted=self.ctx.model_call_budget_exhausted,
            message_count=len(msgs),
            tool_call_count=len(tool_calls),
            tool_request_count=len(tool_requests),
            tool_execution_count=len(tool_executions),
            tool_error_count=traj.meta["tool_error_count"],
            tool_agent_error_count=len(tool_agent_errors),
            case_wall_time_s=round(elapsed, 3),
            duration_s=round(elapsed, 3),
        )
        return traj

    async def _run_group_chat_stream(self, group_chat, task_text: str, team: MASGraph):
        from autogen_agentchat.base import TaskResult

        result = None
        event_index = 0
        streamed_events: list[Any] = []
        tool_requests: list[dict[str, Any]] = []
        tool_executions: list[dict[str, Any]] = []
        tool_agent_errors: list[dict[str, Any]] = []
        self.ctx.log_group_chat(
            "group_chat_start",
            group_chat_class=type(group_chat).__name__,
            group_chat_config=_group_chat_config(team),
            participants=[node.name for node in team.order()],
            context_visibility=getattr(self.ctx, "context_visibility", "shared"),
            dynamic_topology=bool(team.meta.get("dynamic_topology", False)),
        )
        try:
            async for event in group_chat.run_stream(task=task_text):
                if isinstance(event, TaskResult):
                    result = event
                    self.ctx.log_group_chat(
                        "group_chat_end",
                        group_chat_class=type(group_chat).__name__,
                        message_count=len(event.messages),
                        stop_reason=getattr(event, "stop_reason", None),
                    )
                    continue
                streamed_events.append(event)
                event_requests, event_executions, agent_error = self._log_autogen_event(
                    event, event_index
                )
                tool_requests.extend(event_requests)
                tool_executions.extend(event_executions)
                if agent_error is not None:
                    tool_agent_errors.append(agent_error)
                event_index += 1
        except Exception as exc:
            from ..spans import exception_record

            if bool(getattr(self.ctx, "model_call_budget_exhausted", False)):
                limit = self.max_model_calls_per_case
                stop_reason = f"Maximum model calls per case {limit} reached."
                self.ctx.log_group_chat(
                    "group_chat_end",
                    group_chat_class=type(group_chat).__name__,
                    message_count=len(streamed_events),
                    stop_reason=stop_reason,
                    model_call_budget_exhausted=True,
                )
                self.ctx.log_span(
                    "model_call_budget_stop",
                    model_calls_started=self.ctx.model_calls_started,
                    max_model_calls_per_case=limit,
                )
                result = TaskResult(messages=streamed_events, stop_reason=stop_reason)
                return result, tool_requests, tool_executions, tool_agent_errors

            self.ctx.log_group_chat(
                "group_chat_error",
                group_chat_class=type(group_chat).__name__,
                event_index=event_index,
                **exception_record(exc),
            )
            raise
        if result is None:
            raise RuntimeError("AutoGen group chat ended without TaskResult")
        return result, tool_requests, tool_executions, tool_agent_errors

    def _log_autogen_event(
        self, event: Any, event_index: int
    ) -> tuple[list[dict], list[dict], Optional[dict]]:
        source = getattr(event, "source", "?")
        msg_type = getattr(event, "type", type(event).__name__)
        content = _jsonable(getattr(event, "content", ""))
        usage = getattr(event, "models_usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        tool_requests, tool_executions = _structured_tool_records(event, event_index)
        is_tool_event = bool(tool_requests or tool_executions)
        agent_error = None
        if _is_web_surfer_operation_error(event):
            agent_error = {
                "event_index": event_index,
                "source": source,
                "autogen_type": msg_type,
                "error_type": "web_surfer_operation_error",
                "content": content,
                "proxy_enabled": bool(self.web_proxy_url),
                "proxy_endpoint": _safe_proxy_endpoint(self.web_proxy_url),
                **_web_surfer_error_details(content),
            }
        event_fields = {
            "event_index": event_index,
            "source": source,
            "autogen_type": msg_type,
            "content": content,
            "models_usage": _jsonable(usage),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "is_tool_event": is_tool_event,
            "is_error": any(record.get("is_error") for record in tool_executions),
            "metadata": _jsonable(getattr(event, "metadata", {})),
            "tool_requests": tool_requests,
            "tool_executions": tool_executions,
            "agent_operation_error": agent_error,
        }
        self.ctx.log_group_chat("message", **event_fields)
        self.ctx.log_span(
            "tool_event" if is_tool_event else "autogen_message",
            event_index=event_index,
            source=source,
            autogen_type=msg_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_error=event_fields["is_error"],
            tool_request_count=len(tool_requests),
            tool_execution_count=len(tool_executions),
            agent_operation_error=agent_error,
        )
        return tool_requests, tool_executions, agent_error

    def _build_chat(self, team: MASGraph, agents: list[Any], task_text: str):
        group_chat = _group_chat_config(team)
        group_chat_type = str(group_chat.get("type") or "round_robin")
        max_turns = int(
            group_chat.get("max_turns")
            or _effective_max_turns(
                group_chat_type,
                len(agents),
                max_rounds=self.max_rounds,
                max_turns=self.max_turns,
            )
        )
        self.ctx.log_span(
            "group_chat_selected",
            group_chat_type=group_chat_type,
            group_chat=group_chat,
            speaking_order=team.meta.get("speaking_order"),
        )
        termination = _termination_condition(team)
        model_call_termination = _model_call_limit_termination(
            self.ctx, self.max_model_calls_per_case
        )
        if model_call_termination is not None:
            termination = (
                model_call_termination
                if termination is None
                else termination | model_call_termination
            )
        if group_chat_type == "magentic_one":
            from autogen_agentchat.teams import MagenticOneGroupChat

            from .autogen_injection_client import make_injection_client

            control_deployment_id = (
                team.meta.get("control_deployment_id") or self.control_deployment_id
            )
            control_backend = (
                self.backend_resolver(control_deployment_id)
                if self.backend_resolver
                else self.backend
            )
            control_model = (
                self.model_resolver(control_deployment_id) if self.model_resolver else self.model_id
            )
            orchestrator = make_injection_client(
                control_backend,
                role="Orchestrator",
                ctx=self.ctx,
                max_new_tokens=self.control_max_new_tokens or self.max_new_tokens,
                model_id=control_model,
                request_overrides=self.control_invocation_overrides,
                model_context_policy=team.meta.get("controller_model_context"),
                deployment_id=str(control_deployment_id or ""),
            )
            final_answer_prompt = group_chat.get("final_answer_prompt")
            kwargs = {
                "model_client": orchestrator,
                "max_turns": max_turns,
                "max_stalls": int(group_chat.get("max_stalls", self.max_stalls)),
                "termination_condition": termination,
            }
            kwargs["final_answer_prompt"] = (
                str(final_answer_prompt).format(task=task_text)
                if final_answer_prompt
                else GAIA_FINAL_ANSWER_PROMPT.format(task=task_text)
            )
            return MagenticOneGroupChat(agents, **kwargs)
        if group_chat_type == "round_robin":
            from autogen_agentchat.teams import RoundRobinGroupChat

            return RoundRobinGroupChat(
                agents,
                max_turns=max_turns,
                termination_condition=termination,
            )
        if group_chat_type != "selector":
            raise ValueError(
                f"unknown group_chat type {group_chat_type!r}; choose round_robin, "
                "selector, or magentic_one"
            )
        if len(agents) < 2:
            raise ValueError("SelectorGroupChat requires at least two participants")
        from autogen_agentchat.teams import SelectorGroupChat

        from .autogen_injection_client import make_plain_client

        control_deployment_id = team.meta.get("control_deployment_id") or self.control_deployment_id
        control_backend = (
            self.backend_resolver(control_deployment_id) if self.backend_resolver else self.backend
        )

        selector_factory = group_chat.get("selector_func_factory")
        candidate_factory = group_chat.get("candidate_func_factory")
        selector_func = (
            topology_selector(team) if selector_factory == "topology_selector" else None
        )
        candidate_func = None
        if candidate_factory == "topology_candidates":
            candidate_func = topology_candidates(team)
        selector_client = make_plain_client(
            control_backend,
            ctx=self.ctx,
            max_new_tokens=self.control_max_new_tokens or self.max_new_tokens,
            model_id=(
                self.model_resolver(control_deployment_id)
                if self.model_resolver
                else self.model_id
            ),
            request_overrides=self.control_invocation_overrides,
            model_context_policy=team.meta.get("controller_model_context"),
            deployment_id=str(control_deployment_id or ""),
        )
        kwargs = {
            "model_client": selector_client,
            "model_context": _build_model_context(
                team.meta.get("controller_model_context"), selector_client
            ),
            "selector_func": selector_func,
            "candidate_func": candidate_func,
            "termination_condition": termination,
            "max_turns": max_turns,
            "allow_repeated_speaker": bool(
                group_chat.get("allow_repeated_speaker", False)
            ),
            "max_selector_attempts": int(group_chat.get("max_selector_attempts", 3)),
            "model_client_streaming": bool(
                group_chat.get("model_client_streaming", False)
            ),
        }
        if group_chat.get("selector_prompt"):
            kwargs["selector_prompt"] = str(group_chat["selector_prompt"])
        return SelectorGroupChat(
            agents,
            **kwargs,
        )

    def _needs_code_executor(self, team: MASGraph) -> bool:
        return any(
            _agent_type(node) in {"computer_terminal", "terminal", "code_executor"}
            for node in team.order()
        )

    async def _build_web_contexts(
        self, team: MASGraph
    ) -> dict[str, dict[str, Any]]:
        """Create official Playwright contexts with an explicit experiment proxy."""

        from playwright.async_api import async_playwright

        contexts: dict[str, dict[str, Any]] = {}
        try:
            for node in team.order():
                if _agent_type(node) != "web_surfer":
                    continue
                context_id = f"browser_{uuid.uuid4().hex}"
                started = time.monotonic()
                span_id = self.ctx.log_span(
                    "web_context_start",
                    role=node.name,
                    browser_context_id=context_id,
                    headless=self.web_headless,
                    proxy_enabled=bool(self.web_proxy_url),
                    proxy_endpoint=_safe_proxy_endpoint(self.web_proxy_url),
                )
                playwright = await async_playwright().start()
                launch_kwargs: dict[str, Any] = {"headless": self.web_headless}
                if self.web_proxy_url:
                    launch_kwargs["proxy"] = {"server": self.web_proxy_url}
                browser = await playwright.chromium.launch(**launch_kwargs)
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
                    )
                )
                contexts[node.name] = {
                    "playwright": playwright,
                    "browser": browser,
                    "context": context,
                    "browser_context_id": context_id,
                }
                self.ctx.log_span(
                    "web_context_ready",
                    parent_span_id=span_id,
                    role=node.name,
                    browser_context_id=context_id,
                    startup_latency_s=round(time.monotonic() - started, 3),
                    proxy_enabled=bool(self.web_proxy_url),
                    proxy_endpoint=_safe_proxy_endpoint(self.web_proxy_url),
                )
            return contexts
        except Exception as exc:
            from ..spans import exception_record

            self.ctx.log_span(
                "web_context_error",
                proxy_enabled=bool(self.web_proxy_url),
                proxy_endpoint=_safe_proxy_endpoint(self.web_proxy_url),
                **exception_record(exc),
            )
            await self._close_web_contexts(contexts)
            raise

    @staticmethod
    async def _close_web_contexts(contexts: dict[str, dict[str, Any]]) -> None:
        for value in contexts.values():
            for key in ("context", "browser"):
                target = value.get(key)
                close = getattr(target, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass
            playwright = value.get("playwright")
            stop = getattr(playwright, "stop", None)
            if stop is not None:
                try:
                    await stop()
                except Exception:
                    pass

    def _build_code_executor(self, workspace: Path):
        if self.code_executor == "local":
            from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

            return LocalCommandLineCodeExecutor(timeout=self.code_timeout, work_dir=workspace)
        if self.code_executor == "docker":
            from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

            host_user = f"{os.getuid()}:{os.getgid()}"

            class HostUserDockerExecutor(DockerCommandLineCodeExecutor):
                async def start(self):
                    await super().start()
                    if self._container is not None:
                        self._container = _DockerContainerUserProxy(
                            self._container,
                            host_user,
                        )

            executor_type = HostUserDockerExecutor
            kwargs: dict[str, Any] = {}
            if self.container_proxy_url:
                proxy_url = self.container_proxy_url
                no_proxy = self.proxy_no_proxy

                class ProxyAwareDockerExecutor(HostUserDockerExecutor):
                    async def _execute_command(self, command, cancellation_token):
                        environment = [
                            f"HTTP_PROXY={proxy_url}",
                            f"HTTPS_PROXY={proxy_url}",
                            f"http_proxy={proxy_url}",
                            f"https_proxy={proxy_url}",
                            f"NO_PROXY={no_proxy}",
                            f"no_proxy={no_proxy}",
                        ]
                        return await super()._execute_command(
                            ["env", *environment, *command], cancellation_token
                        )

                executor_type = ProxyAwareDockerExecutor
                kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
            return executor_type(
                image=self.docker_image,
                timeout=self.code_timeout,
                work_dir=workspace,
                **kwargs,
            )
        raise ValueError(f"unknown code_executor {self.code_executor!r}; choose docker or local")

    def _create_isolated_workspace(self) -> Path:
        """Atomically create an opaque workspace for one runtime attempt."""

        self.work_root.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            workspace = self.work_root / f"ws_{uuid.uuid4().hex}"
            try:
                workspace.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return workspace
        raise RuntimeError("failed to allocate a unique tool workspace after 8 attempts")

    def _prepare_workspace(
        self,
        query: TaskQuery,
        benchmark: Any = None,
    ) -> tuple[Path, str, list[str], dict[str, Any]]:
        workspace = self._create_isolated_workspace()
        text = query.question
        copied: list[str] = []
        materialized = None
        if benchmark is not None:
            materialized = benchmark.materialize_case(query, workspace, text)
            text = materialized.task_text
            copied.extend(materialized.visible_paths)
        for src_text in self._referenced_paths(query):
            src = Path(src_text).expanduser()
            if not src.exists():
                continue
            dest = workspace / src.name
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            copied.append(str(dest))
            text = text.replace(str(src), dest.name)
        if copied:
            names = ", ".join(Path(path).name for path in copied)
            text = (
                f"{text}\n\nThe referenced file(s) are available in the current workspace: {names}"
            )
        info: dict[str, Any] = {
            "opaque_workspace_id": workspace.name,
            "workspace_policy": "uuid4_isolated_per_attempt_v1",
            "workspace_is_new": True,
            "preexisting_entry_count": 0,
            "attachment_filename_policy": "preserve_official_filename",
            "visible_attachment_names": [Path(path).name for path in copied],
        }
        if benchmark is not None and materialized is not None:
            info["benchmark_id"] = benchmark.id
            info["benchmark_materialization"] = materialized.details
        return workspace, text, copied, info

    @staticmethod
    def _build_task_payload(query: TaskQuery, fallback_text: str):
        content = query.meta.get("multimodal_content")
        if not isinstance(content, list):
            return fallback_text
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image

        parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    parts.append(Image.from_uri(str(url)))
        return MultiModalMessage(content=parts, source="user")

    def _referenced_paths(self, query: TaskQuery) -> list[str]:
        haystack = "\n".join(
            part for part in [query.question, query.context or ""] if isinstance(part, str)
        )
        refs: list[str] = []
        for match in re.finditer(r"Referenced file path:\s*(.+)", haystack):
            ref = match.group(1).strip()
            if ref not in refs:
                refs.append(ref)
        return refs
