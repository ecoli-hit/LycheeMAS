"""InjectionClient —— 一个「会路由 + 注入记忆」的 AutoGen ChatCompletionClient（迁移自
models/injection_client.py）。

【它在 AutoGen 里的位置】
AutoGen 的 AssistantAgent 持有一个 `model_client`（必须是 `ChatCompletionClient`），每当轮到该 agent
发言，AgentChat 就调用 `model_client.create(...)`。我们通过**继承 ChatCompletionClient** 接管这次
调用，
完成「路由 + 记忆注入」再真正生成。

【每次 create() 做的事】
  1. 把 LLMMessage 列表转成 [{role, content}] 给 HF chat 模板（AutoGen 专属转换）
  2. 组装 AutoGen 专属的额外 system 段（JSON 指令 / 工具提示）与后处理回调
  3. 调共享注入引擎 `runtime/injection.py::run_injection_step`（记忆六步的唯一实现）
  4. 把引擎产出转换成 AutoGen 期望的 CreateResult（工具调用转 FunctionCall）

【实例关系】每个 agent 独占一个 InjectionClient（绑定各自 role）；所有 client 共享 backend +
RoutingContext。

⚠️ 这是【唯一允许 import autogen 的文件之一】（CLAUDE.md 黄金法则 1：runtime/backends/autogen_*.py
）。
autogen_core 的导入与 ChatCompletionClient 子类的「构造」都被惰性化到工厂内部，保证无 autogen 时本
模块
仍可被 import（注册一个延迟构造的 model_client/injection 工厂）。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Mapping, Optional, Sequence

from ...core.registry import REGISTRY
from ..injection import (
    InjectionRequest,
    PostProcessed,
    ToolCallRequest,
    run_injection_step,
)

# 注：NL 注入内容由共享注入引擎（runtime/injection.py）作为 system 消息插入；
# 本文件只负责 AutoGen 专属的类型转换、工具调用格式与 JSON 修复。

_JSON_SYSTEM_INSTRUCTION = (
    "Return exactly one valid JSON object and nothing else. "
    "Do not wrap JSON in markdown fences. If a JSON string "
    "needs to mention code, do not use triple-backtick code fences."
)


def _content_to_text(content: Any) -> str:
    """Best-effort conversion for AutoGen message content.

    Tool and multimodal messages may carry pydantic objects/lists. The HF/API
    backends currently consume text chat messages, so preserve structured
    content as compact JSON when possible instead of dropping it.
    """
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


def _to_chat(messages: Sequence[Any]):
    """把 AutoGen 的 LLMMessage 列表转成 HF chat 模板要的 [{role, content}] 字典列表。

    autogen_core 在函数内惰性导入；按子类拆出 role/content，并保留 source（发言者名）以识别 sender。
    """
    from autogen_core.models import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )

    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, AssistantMessage):
            c = _content_to_text(m.content)
            out.append({"role": "assistant", "content": c, "source": m.source})
        elif isinstance(m, UserMessage):
            c = _content_to_text(m.content)
            out.append({"role": "user", "content": c, "source": getattr(m, "source", "user")})
        else:  # FunctionExecutionResultMessage 等 -> 压平成 user 文本
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


def _extract_balanced_json(text: str) -> Optional[Any]:
    """Extract the first balanced JSON value without treating markdown fences
    inside JSON strings as delimiters.

    AutoGen's Magentic-One ledger parser accepts fenced JSON, but its helper
    uses a simple markdown-code-block regex. If the model writes a Python code
    fence inside ``instruction_or_question.answer``, that regex truncates the
    outer JSON block. Scanning braces while respecting string escapes avoids
    that failure mode.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _normalise_json_response(text: str) -> tuple[str, bool]:
    """Return a parser-friendly JSON string when the model was asked for JSON."""
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        unfenced = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        unfenced = re.sub(r"\s*```$", "", unfenced)
        candidates.append(unfenced)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        return json.dumps(value, ensure_ascii=False), candidate != stripped
    value = _extract_balanced_json(stripped)
    if value is not None:
        return json.dumps(value, ensure_ascii=False), True
    return text, False


def _extract_magentic_one_names(messages: Sequence[dict[str, Any]]) -> list[str]:
    for msg in reversed(messages):
        text = str(msg.get("content", ""))
        match = re.search(r"Who should speak next\?\s*\(select from:\s*([^)]+)\)", text)
        if not match:
            continue
        names = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if names:
            return names
    return []


def _coerce_bool_answer(entry: Any) -> None:
    if not isinstance(entry, dict):
        return
    value = entry.get("answer")
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            entry["answer"] = True
        elif lowered in {"false", "no"}:
            entry["answer"] = False


def _choose_next_speaker(ledger: dict[str, Any], names: list[str]) -> str | None:
    if not names:
        return None
    lower_to_name = {name.lower(): name for name in names}
    raw_next = str((ledger.get("next_speaker") or {}).get("answer") or "").strip()
    if raw_next.lower() in lower_to_name:
        return lower_to_name[raw_next.lower()]

    text = json.dumps(ledger, ensure_ascii=False).lower()
    preferences = [
        ("file", ["file", "spreadsheet", "xlsx", "csv", "pdf", "image", "attachment"]),
        ("web", ["web", "search", "look up", "lookup", "website", "internet",
                 "zip code", "verify"]),
        ("coder", ["code", "python", "compute", "calculate", "script", "program"]),
        ("computer", ["terminal", "execute", "run command", "shell"]),
    ]
    for prefix, needles in preferences:
        if any(needle in text for needle in needles):
            for name in names:
                if name.lower().startswith(prefix):
                    return name
    return names[0]


def _ledger_indicates_no_more_work(ledger: dict[str, Any]) -> bool:
    text = json.dumps(ledger, ensure_ascii=False).lower()
    markers = [
        "no additional action",
        "no further action",
        "no further search",
        "further search or analysis is unnecessary",
        "no additional work",
        "no more work",
        "not required",
        "not needed",
        "final answer",
        "the answer is",
        "therefore, the answer",
    ]
    return any(marker in text for marker in markers)


def _repair_magentic_one_ledger(
    text: str, messages: Sequence[dict[str, Any]]
) -> tuple[str, bool, str]:
    try:
        ledger = json.loads(text)
    except Exception:
        return text, False, ""
    if not isinstance(ledger, dict):
        return text, False, ""
    required = [
        "is_request_satisfied",
        "is_progress_being_made",
        "is_in_loop",
        "instruction_or_question",
        "next_speaker",
    ]
    if not all(isinstance(ledger.get(key), dict) for key in required):
        return text, False, ""

    repaired_reasons: list[str] = []
    for key in ("is_request_satisfied", "is_progress_being_made", "is_in_loop"):
        before = (ledger.get(key) or {}).get("answer")
        _coerce_bool_answer(ledger.get(key))
        after = (ledger.get(key) or {}).get("answer")
        if before != after:
            repaired_reasons.append(f"coerced {key}.answer to boolean")

    names = _extract_magentic_one_names(messages)
    satisfied = bool((ledger.get("is_request_satisfied") or {}).get("answer"))
    next_entry = ledger.get("next_speaker") or {}
    next_answer = str(next_entry.get("answer") or "").strip()
    if not satisfied:
        valid_names = {name.lower() for name in names}
        if not next_answer or next_answer.lower() not in valid_names:
            instruction = ledger.get("instruction_or_question") or {}
            if _ledger_indicates_no_more_work(ledger):
                satisfied_entry = ledger.get("is_request_satisfied") or {}
                satisfied_entry["answer"] = True
                satisfied_entry["reason"] = (
                    str(satisfied_entry.get("reason") or "").strip()
                    or str(instruction.get("reason") or "").strip()
                    or str(next_entry.get("reason") or "").strip()
                    or "The ledger indicates no further action is needed."
                )
                ledger["is_request_satisfied"] = satisfied_entry
                if not next_answer:
                    next_entry["answer"] = names[0] if names else ""
                    next_entry["reason"] = (
                        str(next_entry.get("reason") or "").strip()
                        or "The task is being marked complete; next speaker will not be used."
                    )
                    ledger["next_speaker"] = next_entry
                if not str(instruction.get("answer") or "").strip():
                    instruction["answer"] = (
                        "Prepare the final answer based on the completed investigation."
                    )
                    instruction["reason"] = (
                        str(instruction.get("reason") or "").strip()
                        or "The ledger indicates no further action is needed."
                    )
                    ledger["instruction_or_question"] = instruction
                repaired_reasons.append(
                    "marked request satisfied because ledger indicates no further action"
                )
                return json.dumps(ledger, ensure_ascii=False), True, "; ".join(repaired_reasons)

            speaker = _choose_next_speaker(ledger, names)
            if speaker:
                next_entry["answer"] = speaker
                next_entry["reason"] = (
                    str(next_entry.get("reason") or "").strip()
                    or "The previous ledger did not name a valid next speaker."
                )
                ledger["next_speaker"] = next_entry
                repaired_reasons.append(
                    f"replaced invalid next_speaker {next_answer!r} with {speaker!r}"
                )
                if not str(instruction.get("answer") or "").strip():
                    instruction["answer"] = (
                        "Continue investigating the original request using your "
                        "available capabilities, "
                        "then report concise findings needed for the final answer."
                    )
                    instruction["reason"] = (
                        str(instruction.get("reason") or "").strip()
                        or "The previous ledger left the instruction empty."
                    )
                    ledger["instruction_or_question"] = instruction
                    repaired_reasons.append("filled empty instruction_or_question.answer")

    if not repaired_reasons:
        return text, False, ""
    return json.dumps(ledger, ensure_ascii=False), True, "; ".join(repaired_reasons)


def _parse_tool_call_requests(text: str, tools: Sequence[Any]) -> Optional[list[ToolCallRequest]]:
    """从输出文本解析工具调用，产出运行时无关的 ToolCallRequest（不构造 autogen 类型）。"""
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
    calls: list[ToolCallRequest] = []
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
        calls.append(ToolCallRequest(id=raw.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                     name=str(name), arguments=args_text))
    return calls or None


def _build_injection_client_class():
    """惰性构造 InjectionClient/PlainClient 类（继承 autogen_core 的 ChatCompletionClient）。

    autogen_core 仅在调用本工厂时导入，从而 import 本模块不需要 autogen。
    """
    from autogen_core import CancellationToken, FunctionCall  # noqa: F401
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        ModelFamily,
        ModelInfo,
        RequestUsage,
    )

    class InjectionClient(ChatCompletionClient):
        """实现 AutoGen 的 ChatCompletionClient 接口。一个 agent 一个实例（绑定其 role）。"""

        def __init__(self, backend, role: str, ctx, max_new_tokens: int = 256,
                     model_id: str = "qwen-3-4b"):
            self.backend = backend  # 共享的 HF 后端（模型只加载一次，所有 agent 复用）
            self.role = role  # 本 client 服务的角色（路由按此条件化）
            self.ctx = ctx  # 共享的 RoutingContext：turn 计数 / 决策日志 / 同模型对约束
            self.max_new_tokens = max_new_tokens
            self.model_id = model_id  # 模型标识（用于 constraint #2 的同模型对判断）
            self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(self, messages, *, tools=[], tool_choice="auto",
                         json_output=None, extra_create_args: Mapping[str, Any] = {},
                         cancellation_token: Optional[Any] = None) -> CreateResult:
            chat = _to_chat(messages)

            # AutoGen 专属的额外 system 段（引擎按序插到 leading system 段末尾，记忆文本恒在其后）
            extra_system: list[str] = []
            if json_output:
                extra_system.append(_JSON_SYSTEM_INSTRUCTION)
            if tools:
                extra_system.append(_tool_prompt(tools))

            def _postprocess(raw_text: str, send_msgs: list[dict[str, Any]]) -> PostProcessed:
                if not json_output:
                    return PostProcessed(text=raw_text)
                text, normalized = _normalise_json_response(raw_text)
                text, repaired, repair_reason = _repair_magentic_one_ledger(text, send_msgs)
                return PostProcessed(text=text, normalized=normalized,
                                     repaired=repaired, repair_reason=repair_reason)

            def _parse(text: str) -> Optional[list[ToolCallRequest]]:
                return _parse_tool_call_requests(text, tools)

            res = run_injection_step(self.backend, self.ctx, InjectionRequest(
                role=self.role,
                chat=chat,
                max_new_tokens=self.max_new_tokens,
                extra_system_messages=extra_system,
                json_output=bool(json_output),
                tool_count=len(tools),
                tool_choice=tool_choice,
                postprocess=_postprocess,
                parse_tool_calls=_parse if tools else None,
            ))

            self._usage = RequestUsage(
                prompt_tokens=res.input_positions, completion_tokens=res.output_tokens)
            if res.tool_calls:
                calls = [FunctionCall(id=c.id, name=c.name, arguments=c.arguments)
                         for c in res.tool_calls]
                return CreateResult(finish_reason="function_calls", content=calls,
                                    usage=self._usage, cached=False)
            return CreateResult(finish_reason="stop", content=res.text, usage=self._usage,
                                cached=False)

        async def create_stream(self, messages, *, tools=[], tool_choice="auto",
                                json_output=None, extra_create_args={}, cancellation_token=None):
            res = await self.create(messages, tools=tools, tool_choice=tool_choice,
                                    json_output=json_output, extra_create_args=extra_create_args,
                                    cancellation_token=cancellation_token)
            yield res

        async def close(self) -> None:
            return  # 共享 backend 不在此释放，故空实现

        def actual_usage(self) -> RequestUsage:
            return self._usage

        def total_usage(self) -> RequestUsage:
            return self._usage

        def count_tokens(self, messages, *, tools=[]) -> int:
            chat = _to_chat(messages)
            try:
                ids = self.backend.tok.apply_chat_template(
                    [{"role": m["role"], "content": m["content"]} for m in chat],
                    tokenize=True, add_generation_prompt=True)
                return len(ids)
            except Exception:
                return sum(len(str(m["content"])) // 4 for m in chat)

        def remaining_tokens(self, messages, *, tools=[]) -> int:
            return max(0, 32768 - self.count_tokens(messages, tools=tools))

        @property
        def capabilities(self):
            return self.model_info

        @property
        def model_info(self) -> ModelInfo:
            configured = getattr(self.backend, "model_info", {}) or {}
            return ModelInfo(
                vision=bool(configured.get("vision", True)),
                function_calling=bool(configured.get("function_calling", True)),
                json_output=bool(configured.get("json_output", True)),
                family=ModelFamily.UNKNOWN,
                structured_output=bool(configured.get("structured_output", False)),
            )

    class PlainClient(InjectionClient):
        """SelectorGroupChat 选下一个发言者用的「无编排/无注入」客户端：直接喂对话给 backend 生成
        。"""

        def __init__(self, backend, max_new_tokens: int = 64, model_id: str = "qwen-3-4b"):
            super().__init__(backend, role="__selector__", ctx=None,
                             max_new_tokens=max_new_tokens, model_id=model_id)

        async def create(self, messages, *, tools=[], tool_choice="auto",
                         json_output=None, extra_create_args: Mapping[str, Any] = {},
                         cancellation_token: Optional[Any] = None) -> CreateResult:
            g = self.backend.generate_chat(_to_chat(messages), max_new_tokens=self.max_new_tokens)
            self._usage = RequestUsage(
                prompt_tokens=g.n_prompt_pos, completion_tokens=g.n_gen_tokens)
            return CreateResult(
                finish_reason="stop", content=g.text, usage=self._usage, cached=False)

    return InjectionClient, PlainClient


def make_injection_client(backend, role: str, ctx, max_new_tokens: int = 256,
                          model_id: str = "qwen-3-4b"):
    """工厂：构造一个 InjectionClient 实例（首次调用时才导入 autogen）。"""
    InjectionClient, _ = _build_injection_client_class()
    return InjectionClient(backend, role=role, ctx=ctx, max_new_tokens=max_new_tokens,
                           model_id=model_id)


def make_plain_client(backend, max_new_tokens: int = 64, model_id: str = "qwen-3-4b"):
    """工厂：构造一个 PlainClient（selector 用，无注入）。"""
    _, PlainClient = _build_injection_client_class()
    return PlainClient(backend, max_new_tokens=max_new_tokens, model_id=model_id)


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
