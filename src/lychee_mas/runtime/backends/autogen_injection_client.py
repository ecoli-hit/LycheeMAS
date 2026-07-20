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

import json
import re
import uuid
from typing import Any, Mapping, Optional, Sequence

from ...core.registry import REGISTRY

# 注：NL 注入内容的来源标志由 NLMemory.recall 产出（见 memory/channels/nl.py 的
# PREV_OUTPUT_HEADER），
# 本文件把 bundle.NL_Channel 原样作为 system 消息插入。


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
        calls.append(FunctionCall(id=raw.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                  name=str(name), arguments=args_text))
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
    from ...memory.routing.base import RouterInputs

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
            decision = self.ctx.router.decide(RouterInputs(
                role=self.role, task=self.ctx.task, turn=turn, sender=sender,
                query=query if isinstance(query, str) else str(query),
                availability=self.ctx.availability,
                same_model_pair=self.ctx.same_model_pair(sender, self.role)))
            # ③ 召回：按决策物化要注入的 NL 文本 / latent prefix
            bundle = self.ctx.memory.recall(decision, query if isinstance(query, str) else "")

            # ---- 注入 ----
            send_msgs = list(chat)
            if json_output:
                insert_at = 0
                while insert_at < len(send_msgs) and send_msgs[insert_at]["role"] == "system":
                    insert_at += 1
                send_msgs.insert(insert_at, {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object and nothing else. "
                        "Do not wrap JSON in markdown fences. If a JSON string "
                        "needs to mention code, do not use triple-backtick code fences."
                    ),
                })

            if tools:
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

            trace_model_calls = bool(getattr(self.ctx, "trace_model_calls", True))
            input_chars = sum(len(str(m.get("content", ""))) for m in send_msgs)
            input_messages = [{"role": m["role"], "content": m["content"]} for m in send_msgs]
            span_id = self.ctx.log_span(
                "model_call_start",
                role=self.role,
                turn=turn,
                sender=sender,
                message_count=len(send_msgs),
                input_chars=input_chars,
                tool_count=len(tools),
                tool_choice=tool_choice,
                json_output_requested=bool(json_output),
                max_new_tokens=self.max_new_tokens,
                memory_channel=decision.channel,
                routing_reason=decision.reason,
                nl_memory_chars=len(bundle.NL_Channel or ""),
                input_messages=input_messages,
            )
            if trace_model_calls:
                print(
                    f"[model:{self.role}] start turn={turn} sender={sender or '-'} "
                    f"messages={len(send_msgs)} chars={input_chars} tools={len(tools)} "
                    f"max_new_tokens={self.max_new_tokens}",
                    flush=True,
                )

            try:
                if bundle.Latent_Channel is None:
                    # 无 latent：走普通文本生成（none / nl_only 都走这里）
                    g = self.backend.generate_chat(send_msgs, max_new_tokens=self.max_new_tokens)
                elif bundle.Latent_strategy in LatentMemory.FUSION_STRATEGIES:
                    # 融合类（c2c）：Latent_Channel=projector 栈；source=上一个 agent 的输入+输出
                    # （从 ctx 取），经 projector 把其 KV 融进本 agent 生成；
                    # 首个 agent 无前驱则退回普通生成。
                    src_msgs = self._c2c_source()
                    if src_msgs is not None:
                        g = self.backend.generate_chat_with_c2c(
                            send_msgs, src_msgs, bundle.Latent_Channel,
                            max_new_tokens=self.max_new_tokens)
                    else:
                        g = self.backend.generate_chat(
                            send_msgs, max_new_tokens=self.max_new_tokens)
                else:
                    # prefix 类（soft_token 等）：Latent_Channel=(1,P,H) 张量，拼到 embedding 层最前
                    g = self.backend.generate_chat_with_prefix(send_msgs, bundle.Latent_Channel,
                                                               max_new_tokens=self.max_new_tokens)
            except Exception as exc:
                from ...runtime.spans import exception_record

                self.ctx.log_span(
                    "model_call_error",
                    parent_span_id=span_id,
                    role=self.role,
                    turn=turn,
                    sender=sender,
                    **exception_record(exc),
                )
                raise

            output_text = g.text
            json_output_normalized = False
            json_output_repaired = False
            json_output_repair_reason = ""
            if json_output:
                output_text, json_output_normalized = _normalise_json_response(g.text)
                output_text, json_output_repaired, json_output_repair_reason = (
                    _repair_magentic_one_ledger(output_text, send_msgs)
                )

            # ④ 记账与记录
            self.ctx.bump_turn(self.role)
            latent_prefix_positions = int(g.prefix_len or 0)
            input_positions = int(g.n_prompt_pos)
            output_tokens = int(g.n_gen_tokens)
            generation_latency_s = round(g.latency_s, 3)
            self._usage = RequestUsage(
                prompt_tokens=input_positions, completion_tokens=output_tokens)
            self.ctx.log_decision(self.role, turn, sender, decision, {
                "input_positions": input_positions,
                "text_input_tokens": max(0, input_positions - latent_prefix_positions),
                "latent_prefix_positions": latent_prefix_positions,
                "output_tokens": output_tokens,
                "model_generation_latency_s": generation_latency_s,
                "original_prompt_positions": int(
                    getattr(g, "original_prompt_pos", None) or input_positions),
                "prompt_truncated": bool(getattr(g, "prompt_truncated", False)),
                "dropped_messages": int(getattr(g, "dropped_messages", 0)),
                "nl_memory_chars": len(bundle.NL_Channel or ""),
                # CDM 通道方法（供消融分析：本轮用了哪种 NL / latent 策略）
                "nl_strategy": bundle.NL_strategy, "latent_strategy": bundle.Latent_strategy,
                "input_chat_messages": input_messages,
                "output_text": output_text,
                "json_output_normalized": json_output_normalized,
                "json_output_repaired": json_output_repaired,
                "json_output_repair_reason": json_output_repair_reason,
                # Backward-compatible aliases for older analysis scripts.
                "prompt_pos": input_positions, "gen_tokens": output_tokens,
                "prefix_len": latent_prefix_positions, "nl_chars": len(bundle.NL_Channel or ""),
                "latency_s": generation_latency_s,
                "input_messages": input_messages,
                "output": output_text,
                **({"raw_output_text": g.text}
                   if (json_output_normalized or json_output_repaired) else {})})
            tool_calls = _parse_tool_calls(output_text, tools)
            self.ctx.log_span(
                "model_call_end",
                parent_span_id=span_id,
                role=self.role,
                turn=turn,
                sender=sender,
                input_total_positions=input_positions,
                input_text_tokens=max(0, input_positions - latent_prefix_positions),
                input_latent_positions=latent_prefix_positions,
                output_text_tokens=output_tokens,
                model_latency_s=generation_latency_s,
                original_prompt_positions=int(
                    getattr(g, "original_prompt_pos", None) or input_positions),
                prompt_truncated=bool(getattr(g, "prompt_truncated", False)),
                dropped_messages=int(getattr(g, "dropped_messages", 0)),
                tool_call_count=len(tool_calls or []),
                tool_call_request=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in (tool_calls or [])
                ],
                json_output_requested=bool(json_output),
                json_output_normalized=json_output_normalized,
                json_output_repaired=json_output_repaired,
                json_output_repair_reason=json_output_repair_reason,
                output_text=output_text,
                **({"raw_output_text": g.text}
                   if (json_output_normalized or json_output_repaired) else {}),
            )
            if trace_model_calls:
                preview = _preview_text(output_text)
                print(
                    f"[model:{self.role}] done turn={turn} input_pos={input_positions} "
                    f"output_tokens={output_tokens} latency_s={generation_latency_s} "
                    f"truncated={bool(getattr(g, 'prompt_truncated', False))} "
                    f"dropped_messages={int(getattr(g, 'dropped_messages', 0))} "
                    f"tool_calls={len(tool_calls or [])} "
                    f"json_normalized={json_output_normalized} "
                    f"json_repaired={json_output_repaired} preview={preview!r}",
                    flush=True,
                )
            if tool_calls:
                self.ctx.decisions[-1]["tool_call_request"] = [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in tool_calls
                ]
                return CreateResult(finish_reason="function_calls", content=tool_calls,
                                    usage=self._usage, cached=False)
            return CreateResult(finish_reason="stop", content=output_text, usage=self._usage,
                                cached=False)

        def _c2c_source(self):
            """C2C 的 source = 上一个 agent 的「输入消息 + 其输出」（从 ctx.decisions 取）。

            ctx.log_decision 在每个 agent 生成后落了 input_messages 与 output；本 agent 发言时
            decisions[-1] 即上一个发言者。无前驱（首个 agent）返回 None ⇒ 注入 client 退回普通生成。
            """
            decs = getattr(self.ctx, "decisions", None)
            if not decs:
                return None
            prev = decs[-1]
            inp = prev.get("input_messages")
            if not inp:
                return None
            src = [{"role": m["role"], "content": m["content"]} for m in inp]
            out = prev.get("output")
            if isinstance(out, str) and out.strip():
                src.append({"role": "assistant", "content": out})  # 带上前驱的实际输出
            return src

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
