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

from typing import Any, Mapping, Optional, Sequence

from ...core.registry import REGISTRY

# 注：NL 注入内容的来源标志由 NLMemory.recall 产出（见 memory/channels/nl.py 的
# PREV_OUTPUT_HEADER），
# 本文件把 bundle.nl_text 原样作为 system 消息插入。


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
            c = m.content if isinstance(m.content, str) else str(m.content)
            out.append({"role": "assistant", "content": c, "source": m.source})
        elif isinstance(m, UserMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            out.append({"role": "user", "content": c, "source": getattr(m, "source", "user")})
        else:  # FunctionExecutionResultMessage 等 -> 压平成 user 文本
            out.append({"role": "user", "content": str(getattr(m, "content", ""))})
    return out


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

    # RouterInputs 是纯 dataclass（无 autogen/torch），从 L3 触发接缝直接复用
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
            if bundle.nl_text:
                # NL 通道：把记忆作为一条 system 消息，插在开头 system 提示之后、对话之前
                insert_at = 0
                while insert_at < len(send_msgs) and send_msgs[insert_at]["role"] == "system":
                    insert_at += 1
                send_msgs.insert(insert_at, {"role": "system", "content": bundle.nl_text})

            if bundle.latent_c2c is not None:
                # C2C latent 通道：source=上一个 agent 的输入+输出（从 ctx 取），经 projector 把其
                # KV 融进本 agent 生成；首个 agent 无前驱则退回普通生成。
                src_msgs = self._c2c_source()
                if src_msgs is not None:
                    g = self.backend.generate_chat_with_c2c(
                        send_msgs, src_msgs, bundle.latent_c2c,
                        max_new_tokens=self.max_new_tokens)
                else:
                    g = self.backend.generate_chat(send_msgs, max_new_tokens=self.max_new_tokens)
            elif bundle.latent_prefix is not None:
                # latent 通道（soft_token）：把 prefix 张量拼到 token embedding 前
                g = self.backend.generate_chat_with_prefix(send_msgs, bundle.latent_prefix,
                                                           max_new_tokens=self.max_new_tokens)
            else:
                # 无 latent：走普通文本生成（none / nl_only 都走这里）
                g = self.backend.generate_chat(send_msgs, max_new_tokens=self.max_new_tokens)

            input_messages = [{"role": m["role"], "content": m["content"]} for m in send_msgs]

            # ④ 记账与记录
            self.ctx.bump_turn(self.role)
            self._usage = RequestUsage(
                prompt_tokens=g.n_prompt_pos, completion_tokens=g.n_gen_tokens)
            self.ctx.log_decision(self.role, turn, sender, decision, {
                "prompt_pos": g.n_prompt_pos, "gen_tokens": g.n_gen_tokens,
                "prefix_len": g.prefix_len, "nl_chars": len(bundle.nl_text or ""),
                "latency_s": round(g.latency_s, 3),
                "input_messages": input_messages,
                "output": g.text})
            return CreateResult(finish_reason="stop", content=g.text, usage=self._usage,
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
            return ModelInfo(vision=False, function_calling=False, json_output=False,
                             family=ModelFamily.UNKNOWN, structured_output=False)

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
