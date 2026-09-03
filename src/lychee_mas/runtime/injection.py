"""共享注入引擎 —— 「记忆注入六步」的唯一实现（运行时无关）。

每次 agent 发言（无论底层是 AutoGen 的 ChatCompletionClient 还是 LangGraph 的节点函数）
都汇合到 `run_injection_step`：

  ① memory.observe(chat)                        更新记忆库（方法接缝）
  ② router.decide(RouterInputs) -> RouteDecision 决定本轮记忆通道（触发接缝）
  ③ memory.recall(decision, query) -> MemoryBundle 物化要注入的内容
  ④ system 段注入（保序：原始 system → extra_system_messages → 记忆文本 → 对话）
  ⑤ 按 bundle 内容分支生成（KV 融合 > prefix 拼接 > 普通生成）
  ⑥ 记账：bump_turn + log_decision + log_span

引擎只依赖 memory 层公共契约（MemoryManager / MemoryRouter / MemoryBundle / RoutingContext）
与 backend 生成原语（generate_chat / generate_chat_with_prefix / generate_chat_with_c2c）。
引擎专属之外的逻辑（AutoGen 的工具调用格式、JSON 修复等）以回调传入：
`postprocess` 后处理生成文本、`parse_tool_calls` 解析工具调用（返回中立 ToolCallRequest）。

⚠️ 行为契约（回归基线对拍依赖，勿改动）：
- system 段最终顺序 = [原始 system…, extra_system_messages…, 记忆文本, 对话…]。
- log_decision 的记录键集合（含 prompt_pos/gen_tokens 等兼容别名）与
  model_call_start/end/error 的 span 字段逐一保持。
- 融合类 latent 无前驱 source 时走普通生成（算法语义：首个发言者没有可融合的前驱 KV，
  并非配置回退）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..backends.spans import exception_record
from ..methods.memory.channels.latent import LatentMemory
from ..methods.memory.routing.base import RouteDecision, RouterInputs

# chat 消息：{"role": str, "content": str, 可选 "source": 发言者名}
ChatMessage = Dict[str, Any]


@dataclass
class ToolCallRequest:
    """运行时无关的工具调用请求（由 parse_tool_calls 回调产出，调用方自行转换成引擎类型）。"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class PostProcessed:
    """postprocess 回调的返回：后处理文本 + 是否发生规范化/修复（进决策日志与 span）。"""

    text: str
    normalized: bool = False
    repaired: bool = False
    repair_reason: str = ""


# (raw_text, send_msgs) -> PostProcessed；send_msgs 供需要上下文的修复逻辑使用
PostProcessor = Callable[[str, List[ChatMessage]], PostProcessed]
# (post_text) -> ToolCallRequest 列表（无工具调用返回 None）
ToolCallParser = Callable[[str], Optional[List[ToolCallRequest]]]


@dataclass
class InjectionRequest:
    """一次注入生成的全部输入。"""

    role: str
    chat: List[ChatMessage]
    max_new_tokens: int
    extra_system_messages: List[str] = field(default_factory=list)
    json_output: bool = False  # 仅影响 span 字段与后处理记录
    tool_count: int = 0
    tool_choice: Any = "auto"
    postprocess: Optional[PostProcessor] = None
    parse_tool_calls: Optional[ToolCallParser] = None


@dataclass
class InjectionResult:
    """一次注入生成的全部产出。"""

    text: str  # 后处理后的输出文本
    raw_text: str  # 模型原始输出
    gen: Any  # backend 的 GenResult / APIGenResult
    decision: RouteDecision
    bundle: Any  # MemoryBundle
    turn: int
    sender: Optional[str]
    input_positions: int
    output_tokens: int
    tool_calls: Optional[List[ToolCallRequest]]
    post: PostProcessed


def _preview_text(text: str, head: int = 60, tail: int = 60) -> str:
    compact = text.replace("\n", "\\n")
    if len(compact) <= head + tail + 5:
        return compact
    return f"{compact[:head]} ... {compact[-tail:]}"


def _insert_after_leading_system(send_msgs: List[ChatMessage], content: str) -> None:
    """把一条 system 消息插到 leading system 段末尾（对话消息之前）。"""
    insert_at = 0
    while insert_at < len(send_msgs) and send_msgs[insert_at]["role"] == "system":
        insert_at += 1
    send_msgs.insert(insert_at, {"role": "system", "content": content})


def c2c_source_messages(ctx: Any) -> Optional[List[ChatMessage]]:
    """融合类 latent 的 source = 上一个 agent 的「输入消息 + 其输出」（从 ctx.decisions 取）。

    ctx.log_decision 在每个 agent 生成后落了 input_messages 与 output；本 agent 发言时
    decisions[-1] 即上一个发言者。无前驱（首个 agent）返回 None ⇒ 调用方走普通生成。
    """
    decs = getattr(ctx, "decisions", None)
    if not decs:
        return None
    prev = decs[-1]
    inp = prev.get("input_messages")
    if not inp:
        return None
    src: List[ChatMessage] = [{"role": m["role"], "content": m["content"]} for m in inp]
    out = prev.get("output")
    if isinstance(out, str) and out.strip():
        src.append({"role": "assistant", "content": out})  # 带上前驱的实际输出
    return src


def run_injection_step(backend: Any, ctx: Any, req: InjectionRequest) -> InjectionResult:
    """执行一次「记忆注入 + 生成 + 记账」。行为契约见模块 docstring。"""
    chat = req.chat
    # sender = 最近一条非 system 消息的发言者；query = 其内容
    sender: Optional[str] = None
    query: Any = ""
    for m in reversed(chat):
        if m["role"] != "system":
            sender = m.get("source")
            query = m.get("content", "")
            break

    # ① 观察：把当前历史交给记忆库更新
    ctx.memory.observe(chat)
    # ② 记忆通道决策：按 (role×task×turn×sender×可用性×同模型对) 决定本轮通道
    turn = ctx.turn_of(req.role)
    decision = ctx.router.decide(RouterInputs(
        role=req.role, task=ctx.task, turn=turn, sender=sender,
        query=query if isinstance(query, str) else str(query),
        availability=ctx.availability,
        same_model_pair=ctx.same_model_pair(sender, req.role)))
    # ③ 召回：按决策物化要注入的内容
    bundle = ctx.memory.recall(decision, query if isinstance(query, str) else "")

    # ---- ④ 注入（保序：原始 system → extra → 记忆文本 → 对话）----
    send_msgs = list(chat)
    for content in req.extra_system_messages:
        _insert_after_leading_system(send_msgs, content)
    if bundle.NL_Channel:
        _insert_after_leading_system(send_msgs, bundle.NL_Channel)

    trace_model_calls = bool(getattr(ctx, "trace_model_calls", True))
    input_chars = sum(len(str(m.get("content", ""))) for m in send_msgs)
    input_messages = [{"role": m["role"], "content": m["content"]} for m in send_msgs]
    span_id = ctx.log_span(
        "model_call_start",
        role=req.role,
        turn=turn,
        sender=sender,
        message_count=len(send_msgs),
        input_chars=input_chars,
        tool_count=req.tool_count,
        tool_choice=req.tool_choice,
        json_output_requested=bool(req.json_output),
        max_new_tokens=req.max_new_tokens,
        memory_channel=decision.channel,
        routing_reason=decision.reason,
        nl_memory_chars=len(bundle.NL_Channel or ""),
        input_messages=input_messages,
    )
    if trace_model_calls:
        print(
            f"[model:{req.role}] start turn={turn} sender={sender or '-'} "
            f"messages={len(send_msgs)} chars={input_chars} tools={req.tool_count} "
            f"max_new_tokens={req.max_new_tokens}",
            flush=True,
        )

    # ---- ⑤ 生成（KV 融合 > prefix 拼接 > 普通）----
    try:
        if bundle.Latent_Channel is None:
            # 无 latent：走普通文本生成
            g = backend.generate_chat(send_msgs, max_new_tokens=req.max_new_tokens)
        elif bundle.Latent_strategy in LatentMemory.FUSION_STRATEGIES:
            # 融合类：Latent_Channel=projector 栈；source=上一个 agent 的输入+输出（从 ctx 取），
            # 经 projector 把其 KV 融进本 agent 生成；首个 agent 无前驱则走普通生成。
            src_msgs = c2c_source_messages(ctx)
            if src_msgs is not None:
                g = backend.generate_chat_with_c2c(
                    send_msgs, src_msgs, bundle.Latent_Channel,
                    max_new_tokens=req.max_new_tokens)
            else:
                g = backend.generate_chat(send_msgs, max_new_tokens=req.max_new_tokens)
        else:
            # prefix 类：Latent_Channel=(1,P,H) 张量，拼到 embedding 层最前
            g = backend.generate_chat_with_prefix(send_msgs, bundle.Latent_Channel,
                                                  max_new_tokens=req.max_new_tokens)
    except Exception as exc:
        ctx.log_span(
            "model_call_error",
            parent_span_id=span_id,
            role=req.role,
            turn=turn,
            sender=sender,
            **exception_record(exc),
        )
        raise

    if req.postprocess is not None:
        post = req.postprocess(g.text, send_msgs)
    else:
        post = PostProcessed(text=g.text)
    output_text = post.text

    # ---- ⑥ 记账与记录 ----
    ctx.bump_turn(req.role)
    latent_prefix_positions = int(g.prefix_len or 0)
    input_positions = int(g.n_prompt_pos)
    output_tokens = int(g.n_gen_tokens)
    generation_latency_s = round(g.latency_s, 3)
    ctx.log_decision(req.role, turn, sender, decision, {
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
        # 通道方法（供消融分析：本轮用了哪种 NL / latent 策略）
        "nl_strategy": bundle.NL_strategy, "latent_strategy": bundle.Latent_strategy,
        "input_chat_messages": input_messages,
        "output_text": output_text,
        "json_output_normalized": post.normalized,
        "json_output_repaired": post.repaired,
        "json_output_repair_reason": post.repair_reason,
        # Backward-compatible aliases for older analysis scripts.
        "prompt_pos": input_positions, "gen_tokens": output_tokens,
        "prefix_len": latent_prefix_positions, "nl_chars": len(bundle.NL_Channel or ""),
        "latency_s": generation_latency_s,
        "input_messages": input_messages,
        "output": output_text,
        **({"raw_output_text": g.text}
           if (post.normalized or post.repaired) else {})})
    tool_calls = req.parse_tool_calls(output_text) if req.parse_tool_calls else None
    ctx.log_span(
        "model_call_end",
        parent_span_id=span_id,
        role=req.role,
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
        json_output_requested=bool(req.json_output),
        json_output_normalized=post.normalized,
        json_output_repaired=post.repaired,
        json_output_repair_reason=post.repair_reason,
        output_text=output_text,
        **({"raw_output_text": g.text}
           if (post.normalized or post.repaired) else {}),
    )
    if trace_model_calls:
        preview = _preview_text(output_text)
        print(
            f"[model:{req.role}] done turn={turn} input_pos={input_positions} "
            f"output_tokens={output_tokens} latency_s={generation_latency_s} "
            f"truncated={bool(getattr(g, 'prompt_truncated', False))} "
            f"dropped_messages={int(getattr(g, 'dropped_messages', 0))} "
            f"tool_calls={len(tool_calls or [])} "
            f"json_normalized={post.normalized} "
            f"json_repaired={post.repaired} preview={preview!r}",
            flush=True,
        )
    if tool_calls:
        ctx.decisions[-1]["tool_call_request"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in tool_calls
        ]
    return InjectionResult(
        text=output_text,
        raw_text=g.text,
        gen=g,
        decision=decision,
        bundle=bundle,
        turn=turn,
        sender=sender,
        input_positions=input_positions,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        post=post,
    )
