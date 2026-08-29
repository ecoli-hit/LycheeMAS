"""AutoGenRuntime -- compile one explicit TeamSpec graph to AutoGen and run it.

The AutoGen adapter reads its own plan from ``graph.meta["adapter_plans"]``.
TeamSpec itself stores Nodes, Relations, and function bindings rather than an
AutoGen-shaped GroupChat name.

路由信号经共享 RoutingContext 流动——每个 agent 的 InjectionClient 都读它。答案提取按任务可定制
（见 eval/benchmarks/task_config.py）。`run` 返回统一的 `Trajectory`，并把每条消息 + 决策喂给
intercept hook（写 TraceStore / 抽取）。

⚠️ 这是【唯一允许 import autogen 的文件之一】（当前路径：
runtime/adapters/frameworks/autogen/*.py
）。
autogen_agentchat 的导入全部惰性化到方法内部，保证 import 本模块（与注册）不需要 autogen。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from lychee_mas.core.registry import REGISTRY
from lychee_mas.core.types import Answer, Message, TaskQuery, Trajectory
from lychee_mas.runtime.conformance import evaluate_runtime_conformance
from lychee_mas.runtime.contracts.runtime import BaseRuntime, MASGraph
from lychee_mas.runtime.results.contract import result_messages
from lychee_mas.runtime.results.projection import project_result
from lychee_mas.runtime.state import TeamMemoryRuntime
from lychee_mas.runtime.tools.code_execution import CodeExecutorFactory
from lychee_mas.runtime.workspaces.service import WorkspaceService

from ..bindings import binding_for_node
from .coordination import (
    bounded_handoff_selector,
    operation_candidates,
    operation_selector,
    topology_candidates,
    topology_selector,
)
from .coordination import effective_max_turns as _effective_max_turns
from .coordination import group_chat_config as _group_chat_config
from .coordination import model_call_limit_termination as _model_call_limit_termination
from .coordination import termination_condition as _termination_condition
from .events import (
    ObservedCodeExecutor as _ObservedCodeExecutor,
)
from .events import (
    content_to_text as _content_to_text,
)
from .events import (
    deduplicate_tool_records as _deduplicate_tool_records,
)
from .events import (
    is_web_surfer_operation_error as _is_web_surfer_operation_error,
)
from .events import jsonable as _jsonable
from .events import safe_proxy_endpoint as _safe_proxy_endpoint
from .events import structured_tool_records as _structured_tool_records
from .events import tool_operations as _tool_operations
from .events import tool_requests_from_decisions as _tool_requests_from_decisions
from .events import web_surfer_error_details as _web_surfer_error_details

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


def _render_magentic_one_final_answer_prompt(template: str, task: str) -> str:
    """Translate the framework-neutral aggregate prompt to AutoGen's transcript context."""

    return template.format(
        task=task,
        history=(
            "Use the complete team transcript already present in the Magentic-One "
            "context as the team evidence."
        ),
    )


def _extract_final_answer(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", type(msg).__name__) == "ThoughtEvent":
            continue
        text = _content_to_text(getattr(msg, "content", ""))
        match = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _magentic_one_protocol_diagnostic(
    ctx: Any,
    participant_names: Sequence[str],
) -> dict[str, Any] | None:
    """Classify the last official ledger rejection without rewriting model text."""

    exchange = dict(getattr(ctx, "last_controller_exchange", None) or {})
    output = exchange.get("output")
    if not isinstance(output, str) or not output.strip():
        return {
            "protocol": "autogen_magentic_one_progress_ledger",
            "violation": "missing_controller_output",
            "allowed_next_speakers": list(participant_names),
        }
    candidate = output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        ledger = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "protocol": "autogen_magentic_one_progress_ledger",
            "violation": "invalid_json",
            "parse_error": str(exc),
            "allowed_next_speakers": list(participant_names),
            "model_call_index": exchange.get("model_call_index"),
        }
    if not isinstance(ledger, dict):
        return {
            "protocol": "autogen_magentic_one_progress_ledger",
            "violation": "non_object_json",
            "allowed_next_speakers": list(participant_names),
            "model_call_index": exchange.get("model_call_index"),
        }
    required = (
        "is_request_satisfied",
        "is_progress_being_made",
        "is_in_loop",
        "instruction_or_question",
        "next_speaker",
    )
    invalid_fields = [
        key
        for key in required
        if not isinstance(ledger.get(key), dict)
        or "answer" not in ledger[key]
        or "reason" not in ledger[key]
    ]
    if invalid_fields:
        return {
            "protocol": "autogen_magentic_one_progress_ledger",
            "violation": "missing_or_invalid_fields",
            "invalid_fields": invalid_fields,
            "allowed_next_speakers": list(participant_names),
            "model_call_index": exchange.get("model_call_index"),
        }
    satisfied = bool(ledger["is_request_satisfied"]["answer"])
    next_speaker = str(ledger["next_speaker"]["answer"])
    if not satisfied and next_speaker not in participant_names:
        return {
            "protocol": "autogen_magentic_one_progress_ledger",
            "violation": "invalid_next_speaker",
            "next_speaker": next_speaker,
            "allowed_next_speakers": list(participant_names),
            "model_call_index": exchange.get("model_call_index"),
        }
    return {
        "protocol": "autogen_magentic_one_progress_ledger",
        "violation": "official_validator_rejected_output",
        "next_speaker": next_speaker,
        "request_satisfied": satisfied,
        "allowed_next_speakers": list(participant_names),
        "model_call_index": exchange.get("model_call_index"),
    }


def _autogen_specialization(node) -> str:
    return str(binding_for_node(node, "autogen").get("specialization") or "assistant")


def _control_node(team: MASGraph) -> Any | None:
    control_node_id = str(team.meta.get("control_node_id") or "")
    return next(
        (
            node
            for node in team.nodes
            if str(node.id) == control_node_id or node.name == control_node_id
        ),
        None,
    )


def _has_tool_agents(graph: MASGraph) -> bool:
    return any(
        _autogen_specialization(node) in {"code_executor", "file_surfer", "multimodal_web_surfer"}
        or node.tools
        or node.meta.get("tools")
        or str(node.meta.get("execution_kind") or "model") == "executor"
        for node in graph.order()
    )


class _NativeToolObserver:
    """Correlate tools hidden inside official AutoGen specialist agents."""

    def __init__(self, ctx: Any, source: str, origin: str) -> None:
        self.ctx = ctx
        self.source = source
        self.origin = origin
        self.started: dict[str, float] = {}
        self.started_event_ids: dict[str, str | None] = {}
        self.finished: set[str] = set()

    def request_ids(self) -> set[str]:
        return {
            str(call_id)
            for call_id, record in self.ctx.tool_request_records.items()
            if str(record.get("source") or "") == self.source
        }

    def _fields(self, call_or_id: Any) -> tuple[str, dict[str, Any]]:
        call_id = (
            str(call_or_id)
            if isinstance(call_or_id, str)
            else str(getattr(call_or_id, "id", "") or "")
        )
        request = self.ctx.tool_request_record(call_id)
        if not request and not isinstance(call_or_id, str):
            request = {
                "tool_call_id": call_id,
                "tool_name": str(getattr(call_or_id, "name", "") or ""),
                "arguments": str(getattr(call_or_id, "arguments", "") or "{}"),
                "source": self.source,
            }
        return call_id, request

    def on_started(self, call: Any) -> None:
        call_id, request = self._fields(call)
        if not call_id or call_id in self.started:
            return
        self.started[call_id] = time.monotonic()
        self.started_event_ids[call_id] = self.ctx.log_event(
            "tool_execution.started",
            parent_event_id=self.ctx.tool_request_event_id(call_id),
            correlation_id=call_id,
            source=self.source,
            tool_call_id=call_id,
            tool_name=request.get("tool_name"),
            arguments=request.get("arguments"),
            origin=self.origin,
            timing_scope="tool_execution",
        )

    def on_completed(
        self,
        call_or_id: Any,
        output: Any,
        duration_s: float | None = None,
        *,
        timing_scope: str = "tool_execution",
    ) -> None:
        call_id, request = self._fields(call_or_id)
        if not call_id or call_id in self.finished:
            return
        self.finished.add(call_id)
        observable_output = (
            output[1] if isinstance(output, tuple) and len(output) == 2 else output
        )
        text = _content_to_text(observable_output)
        measured_duration = duration_s
        if measured_duration is None and timing_scope == "tool_execution":
            started = self.started.get(call_id)
            if started is not None:
                measured_duration = time.monotonic() - started
        record = {
            "record_type": "tool_execution",
            "source": self.source,
            "tool_call_id": call_id,
            "tool_name": str(request.get("tool_name") or ""),
            "arguments": request.get("arguments"),
            "is_error": False,
            "status": "completed",
            "output": text,
            "delivered_output": text,
            "output_chars": len(text),
            "delivered_output_chars": len(text),
            "output_truncated_for_model_context": False,
            "origin": self.origin,
            "timing_scope": timing_scope,
        }
        if measured_duration is not None:
            record["duration_s"] = round(float(measured_duration), 6)
        event_type = (
            "tool_execution.completed"
            if timing_scope == "tool_execution"
            else "tool_execution.observed"
        )
        event_link = (
            {"operation_id": self.started_event_ids[call_id]}
            if self.started_event_ids.get(call_id)
            else {"parent_event_id": self.ctx.tool_request_event_id(call_id)}
        )
        self.ctx.log_event(event_type, correlation_id=call_id, **event_link, **record)
        self.ctx.record_tool_execution(record)

    def on_failed(
        self,
        call_or_id: Any,
        exc: BaseException,
        duration_s: float | None = None,
        *,
        timing_scope: str = "tool_execution",
    ) -> None:
        from lychee_mas.runtime.events.store import exception_record

        call_id, request = self._fields(call_or_id)
        if not call_id or call_id in self.finished:
            return
        self.finished.add(call_id)
        record = {
            "record_type": "tool_execution",
            "source": self.source,
            "tool_call_id": call_id,
            "tool_name": str(request.get("tool_name") or ""),
            "arguments": request.get("arguments"),
            "is_error": True,
            "status": "error",
            "output": str(exc),
            "delivered_output": str(exc),
            "output_chars": len(str(exc)),
            "delivered_output_chars": len(str(exc)),
            "output_truncated_for_model_context": False,
            "origin": self.origin,
            "timing_scope": timing_scope,
            **exception_record(exc),
        }
        if duration_s is not None:
            record["duration_s"] = round(float(duration_s), 6)
        event_link = (
            {"operation_id": self.started_event_ids[call_id]}
            if self.started_event_ids.get(call_id)
            else {"parent_event_id": self.ctx.tool_request_event_id(call_id)}
        )
        self.ctx.log_event(
            "tool_execution.failed",
            correlation_id=call_id,
            **event_link,
            **record,
        )
        self.ctx.record_tool_execution(record)


def _builtin_workspace_tools(workspace: Optional[Path]) -> dict[str, Callable]:
    from lychee_mas.runtime.tools.portable import portable_workspace_tools

    return portable_workspace_tools(workspace)


def _resolve_function_tools(
    tool_names: Sequence[Any],
    workspace: Optional[Path],
    *,
    query_meta: Optional[dict[str, Any]] = None,
    resources: Optional[list[Any]] = None,
    benchmark: Any = None,
    code_executor: Any = None,
    tool_bundles: Optional[dict[str, Any]] = None,
    web_proxy_url: Optional[str] = None,
) -> list[Any]:
    builtins = _builtin_workspace_tools(workspace)
    from lychee_mas.runtime.tools.portable import portable_web_tools

    web_tools = portable_web_tools(web_proxy_url)
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
            bundle = (tool_bundles or {}).get(item)
            if bundle is None:
                bundle = benchmark.create_tools(item, query_meta or {})
                if tool_bundles is not None:
                    tool_bundles[item] = bundle
                if resources is not None:
                    resources.extend(bundle.resources)
            tools.extend(bundle.tools)
        elif item == "runtime:python_code":
            if code_executor is None:
                raise ValueError("runtime:python_code requires an active code executor")
            from autogen_ext.tools.code_execution import PythonCodeExecutionTool

            tools.append(PythonCodeExecutionTool(code_executor))
        elif item == "runtime:web_search":
            tools.append(web_tools["search_web"])
        elif item == "runtime:fetch_url":
            tools.append(web_tools["fetch_url"])
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

    policies = dict(node.meta.get("message_policies") or {})
    has_latest = any(
        str((policy or {}).get("history") or "all") == "latest" for policy in policies.values()
    )
    if context_visibility != "topology_filtered" and not has_latest:
        return agent
    from autogen_agentchat.agents import (
        MessageFilterAgent,
        MessageFilterConfig,
        PerSourceFilter,
    )

    filters = [PerSourceFilter(source="user")]
    for source in node.meta.get("receives_from") or []:
        policy = policies.get(str(source)) or {}
        filters.append(
            PerSourceFilter(
                source=str(source),
                position=("last" if str(policy.get("history") or "all") == "latest" else None),
                count=(1 if str(policy.get("history") or "all") == "latest" else None),
            )
        )
    filters.append(PerSourceFilter(source=str(node.name)))
    return MessageFilterAgent(
        name=node.name,
        wrapped_agent=agent,
        filter=MessageFilterConfig(per_source=filters),
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
    web_proxy_url: Optional[str] = None,
) -> List[Any]:
    """把 MASGraph 的节点实例化成 AutoGen agents.

    普通模型 Node 使用 AssistantAgent + InjectionClient；框架适配器根据
    TeamSpec 的显式 kind/capabilities/tools 编译出的 NodeBinding，选择
    Coder/FileSurfer/WebSurfer/CodeExecutorAgent 等 AutoGen 原生组件。
    """
    from autogen_agentchat.agents import AssistantAgent

    from .client import make_injection_client

    agents: list[Any] = []
    tool_bundles: dict[str, Any] = {}
    for node in graph.order():
        agent: Any
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
            invocation_overrides_resolver(node.name) if invocation_overrides_resolver else None
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
            max_inline_tool_result_chars=int(
                ((graph.meta.get("communication") or {}).get("tool_results") or {}).get(
                    "max_inline_chars", 50000
                )
            ),
        )
        desc = node.meta.get("description") or node.name
        kind = _autogen_specialization(node)
        if kind == "assistant":
            tools = _resolve_function_tools(
                [*(node.tools or []), *(node.meta.get("tools") or [])],
                workspace,
                query_meta=query_meta,
                resources=resources,
                benchmark=benchmark,
                code_executor=code_executor,
                tool_bundles=tool_bundles,
                web_proxy_url=web_proxy_url,
            )
            agent = AssistantAgent(
                node.name,
                model_client=client,
                tools=tools or None,
                system_message=node.system_prompt,
                description=desc,
                model_context=_build_model_context(model_context_policy, client),
                max_tool_iterations=int(node.meta.get("max_tool_iterations", 1)),
                reflect_on_tool_use=(
                    bool(tools)
                    and str(node.meta.get("on_tool_limit") or "finalize") == "finalize"
                ),
                handoffs=[str(item) for item in (node.meta.get("handoffs") or [])] or None,
            )
        elif kind == "magentic_one_coder":
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
        elif kind == "code_executor":
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

            from .file_surfer import observed_file_surfer_class

            if workspace is None:
                raise ValueError("file_surfer agent requires a workspace")
            observer = _NativeToolObserver(
                ctx,
                node.name,
                "autogen_native_file_surfer",
            )
            file_surfer_class = observed_file_surfer_class()
            agent = file_surfer_class(
                node.name,
                model_client=client,
                base_path=str(workspace),
                request_ids=observer.request_ids,
                on_tool_completed=lambda call_id, output, observer=observer: observer.on_completed(
                    call_id,
                    output,
                    timing_scope="agent_turn",
                ),
                on_tool_failed=lambda call_id, exc, observer=observer: observer.on_failed(
                    call_id,
                    exc,
                    timing_scope="agent_turn",
                ),
            )
        elif kind == "multimodal_web_surfer":
            if workspace is None:
                raise ValueError("web_surfer agent requires a workspace")
            from lychee_mas.runtime.events.store import exception_record

            from .web_surfer import resilient_multimodal_web_surfer_class

            web_runtime = (web_contexts or {}).get(node.name, {})

            def log_reset_recovery(exc: BaseException, *, agent_name: str = node.name) -> None:
                ctx.log_event(
                    "web.reset.recovered",
                    parent_event_id=getattr(ctx, "current_runtime_event_id", None),
                    agent=agent_name,
                    fallback_url="about:blank",
                    **exception_record(exc),
                )

            web_surfer_class = resilient_multimodal_web_surfer_class()
            observer = _NativeToolObserver(
                ctx,
                node.name,
                "autogen_native_web_surfer",
            )
            agent = web_surfer_class(
                node.name,
                model_client=client,
                downloads_folder=str(workspace),
                debug_dir=str(workspace / "web_logs"),
                headless=web_headless,
                to_save_screenshots=save_screenshots,
                playwright=web_runtime.get("playwright"),
                context=web_runtime.get("context"),
                on_reset_recovered=log_reset_recovery,
                on_tool_started=observer.on_started,
                on_tool_completed=observer.on_completed,
                on_tool_failed=observer.on_failed,
            )
        else:
            raise ValueError(
                f"unsupported AutoGen NodeBinding specialization {kind!r} for node {node.name!r}"
            )
        agents.append(_apply_official_message_filter(agent, node, ctx.context_visibility))
    return agents


def _build_model_context(policy: dict[str, Any] | None, model_client):
    """Instantiate the corresponding official AutoGen model-context class."""

    from autogen_core.model_context import (
        BufferedChatCompletionContext,
        TokenLimitedChatCompletionContext,
        UnboundedChatCompletionContext,
    )

    from lychee_mas.runtime.model.token_budget import normalize_model_context_policy

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
        self._tool_request_started_at: dict[str, float] = {}
        self._max_inline_tool_result_chars = 50000
        self.workspace_service = WorkspaceService(self.work_root)
        self.code_executor_factory = CodeExecutorFactory(
            executor=self.code_executor,
            docker_image=self.docker_image,
            timeout_s=self.code_timeout,
            container_proxy_url=self.container_proxy_url,
            proxy_no_proxy=self.proxy_no_proxy,
        )

    async def _run_cleanup_steps(
        self,
        steps: list[tuple[str, Any]],
        *,
        operation_id: str | None,
        timeout_s: float = 20.0,
    ) -> dict[str, bool]:
        """Run independent cleanup callbacks with one bounded grace period."""

        async def invoke(callback):
            result = callback()
            if inspect.isawaitable(result):
                await result

        tasks = {asyncio.create_task(invoke(callback)): label for label, callback in steps}
        if not tasks:
            return {}
        try:
            done, pending = await asyncio.wait(tasks, timeout=max(0.1, float(timeout_s)))
        except BaseException:
            for task in tasks:
                task.cancel()
                task.add_done_callback(
                    lambda completed: None
                    if completed.cancelled()
                    else completed.exception()
                )
            raise
        outcomes: dict[str, bool] = {}
        for task in done:
            label = tasks[task]
            try:
                task.result()
            except BaseException as exc:
                message = str(exc)
                already_closed = type(exc).__name__ == "TargetClosedError" or any(
                    marker in message
                    for marker in (
                        "Target page, context or browser has been closed",
                        "Failed to find context with id",
                    )
                )
                if already_closed:
                    outcomes[label] = True
                    continue
                outcomes[label] = False
                from lychee_mas.runtime.events.store import exception_record

                self.ctx.log_event(
                    "runtime.cleanup_failed",
                    parent_event_id=operation_id,
                    cleanup_target=label,
                    **exception_record(exc),
                )
            else:
                outcomes[label] = True
        for task in pending:
            label = tasks[task]
            outcomes[label] = False
            task.cancel()
            task.add_done_callback(
                lambda completed: None if completed.cancelled() else completed.exception()
            )
            self.ctx.log_event(
                "runtime.cleanup_timed_out",
                parent_event_id=operation_id,
                cleanup_target=label,
                cleanup_timeout_s=timeout_s,
            )
        return outcomes

    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        """Run one AutoGen Trial with a terminal-event guard around every phase."""

        if self.ctx is not None:
            self.ctx.current_runtime_event_id = None
        started_at = time.time()
        try:
            return await self._run_impl(team, query)
        except asyncio.CancelledError:
            if self.ctx is not None:
                self.ctx.log_event(
                    "runtime.cancelled",
                    operation_id=self.ctx.current_runtime_event_id,
                    runtime_elapsed_s=round(time.time() - started_at, 3),
                )
            raise
        except Exception as exc:
            if self.ctx is not None:
                from lychee_mas.runtime.events.store import exception_record

                self.ctx.log_event(
                    "runtime.failed",
                    operation_id=self.ctx.current_runtime_event_id,
                    runtime_elapsed_s=round(time.time() - started_at, 3),
                    **exception_record(exc),
                )
            raise
        finally:
            if self.ctx is not None:
                self.ctx.current_runtime_event_id = None

    async def _run_impl(self, team: MASGraph, query: TaskQuery) -> Trajectory:
        if (self.backend is None and self.backend_resolver is None) or self.ctx is None:
            raise ValueError("AutoGenRuntime.run 需要 backend 与 ctx（构造时或运行前注入）")
        configure_graph = getattr(self.ctx, "configure_graph", None)
        if configure_graph is not None:
            configure_graph(team)
        self._max_inline_tool_result_chars = int(
            ((team.meta.get("communication") or {}).get("tool_results") or {}).get(
                "max_inline_chars", 50000
            )
        )

        from lychee_mas.eval.benchmarks import BENCHMARKS

        benchmark = BENCHMARKS.for_query(self.ctx.task, query.meta)
        self._tool_request_started_at.clear()

        uses_tools = _has_tool_agents(team)
        workspace: Optional[Path] = None
        task_text = query.question
        copied_files: list[str] = []
        workspace_info: dict[str, Any] = {}
        code_executor = None
        code_executor_started = False
        code_executor_t0: float | None = None
        runtime_operation_id: str | None = None
        code_executor_operation_id: str | None = None
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
            runtime_operation_id = self.ctx.log_event(
                "runtime.started",
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
            self.ctx.current_runtime_event_id = runtime_operation_id
            self.ctx.team_memory = TeamMemoryRuntime(
                dict(team.meta.get("team_spec") or {}),
                log_event=self.ctx.log_event,
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
                self.ctx.log_event(
                    "workspace.prepared",
                    parent_event_id=runtime_operation_id,
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
                        max_inline_tool_result_chars=self._max_inline_tool_result_chars,
                    )
                    print(
                        f"[autogen:tools] starting code_executor={self.code_executor} "
                        f"image={self.docker_image}",
                        flush=True,
                    )
                    code_executor_t0 = time.time()
                    code_executor_operation_id = self.ctx.log_event(
                        "code_executor.started",
                        parent_event_id=runtime_operation_id,
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        timeout_s=self.code_timeout,
                        workspace=str(workspace),
                    )
                    await code_executor.start()
                    code_executor_started = True
                    print("[autogen:tools] code_executor ready", flush=True)
                    self.ctx.log_event(
                        "code_executor.ready",
                        parent_event_id=code_executor_operation_id,
                        executor=self.code_executor,
                        docker_image=self.docker_image,
                        workspace=str(workspace),
                        startup_latency_s=round(time.time() - code_executor_t0, 3),
                    )

            self.ctx.trace_model_calls = (
                True if self.trace_model_calls is None else self.trace_model_calls
            )
            if any(
                _autogen_specialization(node) == "multimodal_web_surfer" for node in team.order()
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
                web_proxy_url=self.web_proxy_url,
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
        finally:
            cleanup_steps: list[tuple[str, Any]] = []
            for agent in agents:
                close_target = getattr(agent, "_wrapped_agent", agent)
                close = getattr(close_target, "close", None)
                if close is not None and uses_tools:
                    cleanup_steps.append(
                        (f"agent:{getattr(agent, 'name', type(agent).__name__)}", close)
                    )
            cleanup_steps.append(("web_contexts", lambda: self._close_web_contexts(web_contexts)))
            for resource in tool_resources:
                close = getattr(resource, "close", None)
                if close is None:
                    continue
                cleanup_steps.append((f"tool_resource:{type(resource).__name__}", close))
            await self._run_cleanup_steps(
                cleanup_steps,
                operation_id=runtime_operation_id,
            )
            if code_executor_started and code_executor is not None:
                code_cleanup = await self._run_cleanup_steps(
                    [("code_executor", code_executor.stop)],
                    operation_id=runtime_operation_id,
                )
                if code_cleanup.get("code_executor"):
                    self.ctx.log_event(
                        "code_executor.stopped",
                        operation_id=code_executor_operation_id,
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

        decisions = list(getattr(self.ctx, "decisions", []))
        tool_requests = _deduplicate_tool_records(
            [*_tool_requests_from_decisions(decisions), *stream_tool_requests]
        )
        tool_executions = _deduplicate_tool_records(
            [
                *stream_tool_executions,
                *code_execution_records,
                *getattr(self.ctx, "observed_tool_executions", []),
            ]
        )
        tool_calls = _tool_operations(tool_requests, tool_executions)
        eligible_messages = result_messages(team, msgs)
        default_text = _extract_final_answer(eligible_messages) or benchmark.extract_messages(
            self.ctx.task, eligible_messages
        )
        projected = project_result(
            benchmark=benchmark,
            task=self.ctx.task,
            team=team,
            query=query,
            messages=list(msgs),
            workspace=workspace,
            default_text=default_text,
            tool_requests=tool_requests,
        )
        traj.final_answer = Answer(
            content=projected.content,
            source=projected.source,
            meta={"n_messages": len(msgs), "case_wall_time_s": round(elapsed, 3)},
        )
        traj.candidates = [traj.final_answer]
        traj.meta["decisions"] = decisions
        traj.meta["result_submitters"] = list(
            (team.meta.get("result_contract") or {}).get("submitters") or []
        )
        traj.meta["eligible_result_message_count"] = len(eligible_messages)
        self.ctx.log_event(
            (
                "result_contract.validated"
                if projected.validation.valid
                else "result_contract.failed"
            ),
            parent_event_id=runtime_operation_id,
            **projected.validation.to_dict(),
        )
        traj.meta["result_contract"] = projected.validation.to_dict()
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
        traj.meta["team_memory"] = (
            self.ctx.team_memory.snapshot() if self.ctx.team_memory is not None else {}
        )
        conformance = evaluate_runtime_conformance(
            framework=self.name,
            team=team,
            result_validation=projected.validation,
            tool_requests=tool_requests,
            tool_executions=tool_executions,
            observation_coverage={
                "trial_lifecycle": "full",
                "model_calls": "full",
                "tool_calls": "full",
                "agent_messages": "full",
                "team_graph_lifecycle": "native_group_chat_unavailable",
            },
        )
        traj.meta["runtime_conformance"] = conformance.to_dict()
        self.ctx.log_event(
            "runtime.conformance_evaluated",
            parent_event_id=runtime_operation_id,
            **conformance.to_dict(),
        )
        self.ctx.log_event(
            "runtime.completed",
            operation_id=runtime_operation_id,
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
        from autogen_core import CancellationToken

        result = None
        cancellation_token = CancellationToken()
        event_index = 0
        streamed_events: list[Any] = []
        tool_requests: list[dict[str, Any]] = []
        tool_executions: list[dict[str, Any]] = []
        tool_agent_errors: list[dict[str, Any]] = []
        group_chat_event_id = self.ctx.log_event(
            "group_chat.started",
            group_chat_class=type(group_chat).__name__,
            group_chat_config=_group_chat_config(team),
            participants=[node.name for node in team.order()],
            member_nodes=[node.name for node in team.order()],
            operation_bindings=list(team.meta.get("operation_bindings") or []),
            context_visibility=getattr(self.ctx, "context_visibility", "shared"),
            dynamic_topology=bool(team.meta.get("dynamic_topology", False)),
        )
        previous_group_chat_event_id = self.ctx.current_group_chat_event_id
        self.ctx.current_group_chat_event_id = group_chat_event_id
        try:
            try:
                async for event in group_chat.run_stream(
                    task=task_text,
                    cancellation_token=cancellation_token,
                ):
                    if isinstance(event, TaskResult):
                        result = event
                        self.ctx.log_event(
                            "group_chat.completed",
                            operation_id=group_chat_event_id,
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
            except asyncio.CancelledError:
                cancellation_token.cancel()
                self.ctx.log_event(
                    "group_chat.cancelled",
                    operation_id=group_chat_event_id,
                    group_chat_class=type(group_chat).__name__,
                    event_index=event_index,
                    message_count=len(streamed_events),
                )
                raise
            except Exception as exc:
                from lychee_mas.runtime.events.store import exception_record

                if bool(getattr(self.ctx, "model_call_budget_exhausted", False)):
                    limit = self.max_model_calls_per_case
                    stop_reason = f"Maximum model calls per case {limit} reached."
                    self.ctx.log_event(
                        "group_chat.completed",
                        operation_id=group_chat_event_id,
                        group_chat_class=type(group_chat).__name__,
                        message_count=len(streamed_events),
                        stop_reason=stop_reason,
                        model_call_budget_exhausted=True,
                    )
                    self.ctx.log_event(
                        "model_call.budget_stopped",
                        model_calls_started=self.ctx.model_calls_started,
                        max_model_calls_per_case=limit,
                    )
                    result = TaskResult(messages=streamed_events, stop_reason=stop_reason)
                    return result, tool_requests, tool_executions, tool_agent_errors

                if (
                    type(group_chat).__name__ == "MagenticOneGroupChat"
                    and "Failed to parse ledger information" in str(exc)
                ):
                    self.ctx.log_event(
                        "coordination.protocol_violation",
                        **_magentic_one_protocol_diagnostic(
                            self.ctx,
                            [node.name for node in team.order()],
                        ),
                    )
                self.ctx.log_event(
                    "group_chat.failed",
                    operation_id=group_chat_event_id,
                    group_chat_class=type(group_chat).__name__,
                    event_index=event_index,
                    **exception_record(exc),
                )
                raise
            if result is None:
                raise RuntimeError("AutoGen group chat ended without TaskResult")
            return result, tool_requests, tool_executions, tool_agent_errors
        finally:
            self.ctx.current_group_chat_event_id = previous_group_chat_event_id

    def _log_autogen_event(
        self, event: Any, event_index: int
    ) -> tuple[list[dict], list[dict], Optional[dict]]:
        source = getattr(event, "source", "?")
        msg_type = getattr(event, "type", type(event).__name__)
        content = _jsonable(getattr(event, "content", ""))
        usage = getattr(event, "models_usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        tool_requests, tool_executions = _structured_tool_records(
            event,
            event_index,
            max_inline_tool_result_chars=getattr(self, "_max_inline_tool_result_chars", 50000),
        )
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
        self.ctx.log_event("agent.message.published", **event_fields)
        if not tool_requests and getattr(self.ctx, "team_memory", None) is not None:
            memory_content = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, default=str)
            )
            self.ctx.team_memory.write_node_output(
                str(source),
                memory_content,
                invocation_id=f"autogen:event:{event_index}",
                metadata={"framework": "autogen", "autogen_type": msg_type},
            )
        tool_request_started_at = getattr(self, "_tool_request_started_at", None)
        if tool_request_started_at is None:
            tool_request_started_at = {}
            self._tool_request_started_at = tool_request_started_at
        request_event_ids: dict[str, str] = {}
        for request in tool_requests:
            tool_call_id = str(request.get("tool_call_id") or "")
            if tool_call_id:
                tool_request_started_at.setdefault(tool_call_id, time.monotonic())
            request_event_id = self.ctx.tool_request_event_id(tool_call_id)
            if request_event_id is None:
                request_fields = {
                    **request,
                    "event_index": event_index,
                    "source": request.get("source") or source,
                    "correlation_id": tool_call_id or None,
                }
                request_event_id = self.ctx.log_event(
                    "tool_call.requested",
                    **request_fields,
                )
                self.ctx.remember_tool_request_event(
                    tool_call_id,
                    request_event_id,
                    {
                        **request_fields,
                        "requested_at_monotonic_s": time.monotonic(),
                    },
                )
            if tool_call_id and request_event_id:
                request_event_ids[tool_call_id] = request_event_id
        for execution in tool_executions:
            tool_call_id = str(execution.get("tool_call_id") or "")
            started_at = tool_request_started_at.pop(tool_call_id, None)
            execution_fields = {
                **execution,
                "event_index": event_index,
                "source": execution.get("source") or source,
                "correlation_id": tool_call_id or None,
                "parent_event_id": (
                    request_event_ids.get(tool_call_id)
                    or self.ctx.tool_request_event_id(tool_call_id)
                ),
                "status": "error" if execution.get("is_error") else "completed",
                **(
                    {"duration_s": round(time.monotonic() - started_at, 6)}
                    if started_at is not None
                    else {}
                ),
            }
            self.ctx.log_event(
                "tool_execution.observed",
                **execution_fields,
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
        self.ctx.log_event(
            "group_chat.selected",
            group_chat_type=group_chat_type,
            group_chat=group_chat,
            speaking_order=team.meta.get("speaking_order"),
        )
        termination = _termination_condition(team, self.ctx)
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

            from .client import make_plain_client

            control_node = _control_node(team)
            control_role = str(
                getattr(control_node, "name", None)
                or team.meta.get("control_node_id")
                or "Orchestrator"
            )
            control_deployment_id = (
                (control_node.meta.get("deployment_id") if control_node else None)
                or team.meta.get("control_deployment_id")
                or self.control_deployment_id
            )
            control_backend = (
                self.backend_resolver(control_deployment_id)
                if self.backend_resolver
                else self.backend
            )
            control_model = (
                self.model_resolver(control_deployment_id) if self.model_resolver else self.model_id
            )
            orchestrator = make_plain_client(
                control_backend,
                role=control_role,
                ctx=self.ctx,
                max_new_tokens=(
                    self.max_new_tokens_resolver(control_role, self.max_new_tokens)
                    if self.max_new_tokens_resolver
                    else self.control_max_new_tokens or self.max_new_tokens
                ),
                model_id=control_model,
                request_overrides=(
                    self.invocation_overrides_resolver(control_role)
                    if self.invocation_overrides_resolver
                    else self.control_invocation_overrides
                ),
                model_context_policy=(
                    control_node.meta.get("model_context")
                    if control_node
                    else team.meta.get("controller_model_context")
                ),
                deployment_id=str(control_deployment_id or ""),
                controller=True,
                max_inline_tool_result_chars=int(
                    ((team.meta.get("communication") or {}).get("tool_results") or {}).get(
                        "max_inline_chars", 50000
                    )
                ),
            )
            final_answer_prompt = group_chat.get("final_answer_prompt")
            kwargs = {
                "model_client": orchestrator,
                "max_turns": max_turns,
                "max_stalls": int(group_chat.get("max_stalls", self.max_stalls)),
                "termination_condition": termination,
            }
            kwargs["final_answer_prompt"] = (
                _render_magentic_one_final_answer_prompt(
                    str(final_answer_prompt),
                    task_text,
                )
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
        if group_chat_type == "swarm":
            from autogen_agentchat.teams import Swarm

            return Swarm(
                agents,
                max_turns=max_turns,
                termination_condition=termination,
            )
        if group_chat_type == "graph_flow":
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

            agents_by_name = {agent.name: agent for agent in agents}
            builder = DiGraphBuilder()
            for agent in agents:
                builder.add_node(agent)
            edge_count = 0
            for source, targets in team.edges.items():
                if source not in agents_by_name:
                    continue
                for target in targets:
                    if target not in agents_by_name:
                        continue
                    builder.add_edge(source, target)
                    edge_count += 1
            if edge_count == 0:
                raise ValueError("GraphFlow requires at least one topology edge")
            return GraphFlow(
                participants=agents,
                graph=builder.build(),
                max_turns=max_turns,
                termination_condition=termination,
            )
        if group_chat_type != "selector":
            raise ValueError(
                f"unknown group_chat type {group_chat_type!r}; choose round_robin, "
                "selector, magentic_one, swarm, or graph_flow"
            )
        if len(agents) < 2:
            raise ValueError("SelectorGroupChat requires at least two participants")
        from autogen_agentchat.teams import SelectorGroupChat

        from .client import make_plain_client

        control_node = _control_node(team)
        control_role = str(
            getattr(control_node, "name", None)
            or team.meta.get("control_node_id")
            or "Selector"
        )
        control_deployment_id = (
            (control_node.meta.get("deployment_id") if control_node else None)
            or team.meta.get("control_deployment_id")
            or self.control_deployment_id
        )
        control_backend = (
            self.backend_resolver(control_deployment_id) if self.backend_resolver else self.backend
        )

        selector_factory = group_chat.get("selector_func_factory")
        candidate_factory = group_chat.get("candidate_func_factory")
        candidate_node_ids = list(group_chat.get("candidate_node_ids") or [])
        selector_func = (
            operation_selector(team, candidate_node_ids)
            if selector_factory == "operation_selector"
            else bounded_handoff_selector(team)
            if selector_factory == "bounded_handoff_selector"
            else topology_selector(team)
            if selector_factory == "topology_selector"
            else None
        )
        candidate_func = None
        if candidate_factory == "operation_candidates":
            candidate_func = operation_candidates(team, candidate_node_ids)
        elif candidate_factory == "topology_candidates":
            candidate_func = topology_candidates(team)
        selector_candidates = [agent.name for agent in agents]

        def emit_selected_edge(selected_node: str) -> None:
            self._emit_control_selection(
                team,
                source_node_id=control_role,
                target_node_id=selected_node,
                operation="select_next",
            )

        if selector_func is not None:
            raw_selector_func = selector_func

            def observed_selector_func(messages):
                selected_node = raw_selector_func(messages)
                self._emit_coordination_decision(
                    team,
                    actor=control_role,
                    operation="select_next",
                    decision_source="selector_func",
                    selected_node=selected_node,
                    raw_decision=str(selected_node),
                )
                return selected_node

            selector_func = observed_selector_func
        selector_client = make_plain_client(
            control_backend,
            role=control_role,
            ctx=self.ctx,
            max_new_tokens=(
                self.max_new_tokens_resolver(control_role, self.max_new_tokens)
                if self.max_new_tokens_resolver
                else self.control_max_new_tokens or self.max_new_tokens
            ),
            model_id=(
                self.model_resolver(control_deployment_id) if self.model_resolver else self.model_id
            ),
            request_overrides=(
                self.invocation_overrides_resolver(control_role)
                if self.invocation_overrides_resolver
                else self.control_invocation_overrides
            ),
            model_context_policy=(
                control_node.meta.get("model_context")
                if control_node
                else team.meta.get("controller_model_context")
            ),
            deployment_id=str(control_deployment_id or ""),
            controller=True,
            controller_operation="select_next",
            controller_candidates=selector_candidates,
            controller_decision_callback=emit_selected_edge,
            max_inline_tool_result_chars=int(
                ((team.meta.get("communication") or {}).get("tool_results") or {}).get(
                    "max_inline_chars", 50000
                )
            ),
        )
        kwargs = {
            "model_client": selector_client,
            "model_context": _build_model_context(
                (
                    control_node.meta.get("model_context")
                    if control_node
                    else team.meta.get("controller_model_context")
                ),
                selector_client,
            ),
            "selector_func": selector_func,
            "candidate_func": candidate_func,
            "termination_condition": termination,
            "max_turns": max_turns,
            "allow_repeated_speaker": bool(group_chat.get("allow_repeated_speaker", False)),
            "max_selector_attempts": int(group_chat.get("max_selector_attempts", 3)),
            "model_client_streaming": bool(group_chat.get("model_client_streaming", False)),
        }
        if group_chat.get("selector_prompt"):
            kwargs["selector_prompt"] = str(group_chat["selector_prompt"])
        return SelectorGroupChat(
            agents,
            **kwargs,
        )

    def _needs_code_executor(self, team: MASGraph) -> bool:
        return any(
            (
                str(node.meta.get("execution_kind") or "model") == "executor"
                and str(node.meta.get("executor_type") or "") == "code"
            )
            or "runtime:python_code" in {*(node.tools or []), *(node.meta.get("tools") or [])}
            for node in team.order()
        )

    async def _build_web_contexts(self, team: MASGraph) -> dict[str, dict[str, Any]]:
        """Create official Playwright contexts with an explicit experiment proxy."""

        from playwright.async_api import async_playwright

        contexts: dict[str, dict[str, Any]] = {}
        startup_operation_id: str | None = None
        try:
            for node in team.order():
                if _autogen_specialization(node) != "multimodal_web_surfer":
                    continue
                context_id = f"browser_{uuid.uuid4().hex}"
                started = time.monotonic()
                startup_operation_id = self.ctx.log_event(
                    "web.context_startup.started",
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
                self.ctx.log_event(
                    "web.context_startup.completed",
                    operation_id=startup_operation_id,
                    role=node.name,
                    browser_context_id=context_id,
                    startup_latency_s=round(time.monotonic() - started, 3),
                    proxy_enabled=bool(self.web_proxy_url),
                    proxy_endpoint=_safe_proxy_endpoint(self.web_proxy_url),
                )
            return contexts
        except Exception as exc:
            from lychee_mas.runtime.events.store import exception_record

            self.ctx.log_event(
                "web.context_startup.failed",
                operation_id=startup_operation_id,
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
        return self.code_executor_factory.build(workspace)

    def _prepare_workspace(
        self,
        query: TaskQuery,
        benchmark: Any = None,
    ) -> tuple[Path, str, list[str], dict[str, Any]]:
        prepared = self.workspace_service.materialize(query, benchmark)
        return (
            prepared.root,
            prepared.task_text,
            list(prepared.visible_paths),
            prepared.details,
        )

    @staticmethod
    def _build_task_payload(query: TaskQuery, fallback_text: str):
        content = query.meta.get("multimodal_content")
        if not isinstance(content, list):
            return fallback_text
        from autogen_agentchat.messages import MultiModalMessage
        from autogen_core import Image

        from lychee_mas.runtime.model.multimodal import pil_image_from_data_uri

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
                    parts.append(Image.from_pil(pil_image_from_data_uri(url)))
        return MultiModalMessage(content=parts, source="user")
