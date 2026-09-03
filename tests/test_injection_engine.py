"""共享注入引擎（runtime/injection.py）的离线测试：FakeBackend/FakeMemory，零重依赖。

覆盖行为契约：system 段保序、三分支生成选择、融合 source 取自 ctx.decisions[-1]、
首 agent 无前驱走普通生成、decision 记录键全集（含兼容别名）、span 序列与异常路径。
"""
from __future__ import annotations

import pytest
from lychee_mas.methods.memory.base import MemoryBundle
from lychee_mas.methods.memory.channels.latent import LatentMemory
from lychee_mas.methods.memory.context import RoutingContext
from lychee_mas.methods.memory.routing.base import MemoryRouter, RouteDecision, RouterInputs
from lychee_mas.runtime.injection import (
    InjectionRequest,
    PostProcessed,
    ToolCallRequest,
    c2c_source_messages,
    run_injection_step,
)

FUSION = next(iter(LatentMemory.FUSION_STRATEGIES))


class FakeGen:
    def __init__(self, text: str = "ok", prefix_len: int = 0):
        self.text = text
        self.n_prompt_pos = 10
        self.n_gen_tokens = 5
        self.prefix_len = prefix_len
        self.latency_s = 0.0123


class FakeBackend:
    """记录走了哪个生成分支与收到的消息。"""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple] = []
        self.fail = fail

    def generate_chat(self, msgs, max_new_tokens):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(("plain", list(msgs)))
        return FakeGen()

    def generate_chat_with_prefix(self, msgs, prefix, max_new_tokens):
        self.calls.append(("prefix", list(msgs), prefix))
        return FakeGen(prefix_len=4)

    def generate_chat_with_c2c(self, msgs, src_msgs, projectors, max_new_tokens):
        self.calls.append(("c2c", list(msgs), list(src_msgs)))
        return FakeGen()


class FakeRouter(MemoryRouter):
    name = "fake"

    def __init__(self, channel: str = "nl"):
        self.channel = channel

    def decide(self, x: RouterInputs) -> RouteDecision:
        return RouteDecision(channel=self.channel, reason="test")


class FakeMemory:
    """recall 恒返回构造时给定的 bundle；observe 记录调用。"""

    def __init__(self, bundle: MemoryBundle):
        self.bundle = bundle
        self.observed: list[list] = []

    def observe(self, messages):
        self.observed.append(list(messages))

    def recall(self, decision, query):
        return self.bundle

    def reset(self):
        pass


class FakeSpanLogger:
    def __init__(self):
        self.spans: list[tuple[str, dict]] = []

    def set_case(self, case_id, sample_index=None):
        pass

    def log(self, span_type, **fields):
        self.spans.append((span_type, fields))
        return f"sid-{len(self.spans)}"


def make_ctx(bundle: MemoryBundle, channel: str = "nl", spans: bool = False) -> RoutingContext:
    ctx = RoutingContext(task="testtask", router=FakeRouter(channel), memory=FakeMemory(bundle))
    ctx.trace_model_calls = False
    if spans:
        ctx.span_logger = FakeSpanLogger()
    return ctx


CHAT = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "task question", "source": "user"},
]


def run(backend, ctx, **kw):
    req = InjectionRequest(role="worker", chat=list(CHAT), max_new_tokens=32, **kw)
    return run_injection_step(backend, ctx, req)


def test_system_segment_order_and_plain_branch():
    ctx = make_ctx(MemoryBundle(NL_Channel="MEM", NL_strategy="prev_output"))
    backend = FakeBackend()
    res = run(backend, ctx, extra_system_messages=["JSON-INS", "TOOL-INS"])
    branch, msgs = backend.calls[0][0], backend.calls[0][1]
    assert branch == "plain"
    # 保序：原始 system → extra（按给定顺序）→ 记忆文本 → 对话
    assert [m["content"] for m in msgs[:4]] == ["SYS", "JSON-INS", "TOOL-INS", "MEM"]
    assert msgs[4]["role"] == "user"
    assert res.text == "ok"
    assert res.input_positions == 10 and res.output_tokens == 5


def test_prefix_branch_for_non_fusion_latent():
    prefix = object()
    ctx = make_ctx(MemoryBundle(Latent_Channel=prefix, Latent_strategy="soft_token"),
                   channel="latent")
    backend = FakeBackend()
    run(backend, ctx)
    assert backend.calls[0][0] == "prefix"
    assert backend.calls[0][2] is prefix


def test_fusion_without_predecessor_uses_plain_generation():
    ctx = make_ctx(MemoryBundle(Latent_Channel=object(), Latent_strategy=FUSION),
                   channel="latent")
    backend = FakeBackend()
    run(backend, ctx)
    # 首个发言者没有可融合的前驱 KV（算法语义），走普通生成
    assert backend.calls[0][0] == "plain"


def test_fusion_source_comes_from_previous_decision():
    ctx = make_ctx(MemoryBundle(Latent_Channel=object(), Latent_strategy=FUSION),
                   channel="latent")
    ctx.decisions.append({
        "input_messages": [{"role": "user", "content": "prev input"}],
        "output": "prev output",
    })
    backend = FakeBackend()
    run(backend, ctx)
    branch, _, src = backend.calls[0]
    assert branch == "c2c"
    assert src == [{"role": "user", "content": "prev input"},
                   {"role": "assistant", "content": "prev output"}]


def test_c2c_source_messages_empty_cases():
    ctx = make_ctx(MemoryBundle())
    assert c2c_source_messages(ctx) is None
    ctx.decisions.append({"output": "no input recorded"})
    assert c2c_source_messages(ctx) is None


def test_decision_record_keys_and_turn_bump():
    ctx = make_ctx(MemoryBundle(NL_Channel="MEM", NL_strategy="prev_output"))
    run(FakeBackend(), ctx)
    assert ctx.turns["worker"] == 1
    rec = ctx.decisions[-1]
    expected = {
        "role", "turn", "sender", "channel", "reason",
        "turn_index", "sender_role", "memory_channel", "routing_reason",
        "input_positions", "text_input_tokens", "latent_prefix_positions",
        "output_tokens", "model_generation_latency_s", "original_prompt_positions",
        "prompt_truncated", "dropped_messages", "nl_memory_chars",
        "nl_strategy", "latent_strategy", "input_chat_messages", "output_text",
        "json_output_normalized", "json_output_repaired", "json_output_repair_reason",
        # 兼容别名
        "prompt_pos", "gen_tokens", "prefix_len", "nl_chars", "latency_s",
        "input_messages", "output",
    }
    assert expected <= set(rec)
    assert rec["prompt_pos"] == rec["input_positions"] == 10
    assert rec["gen_tokens"] == rec["output_tokens"] == 5
    assert rec["output"] == rec["output_text"] == "ok"


def test_postprocess_and_raw_text_recorded():
    ctx = make_ctx(MemoryBundle())

    def post(raw, send_msgs):
        return PostProcessed(text="FIXED", normalized=True)

    res = run(FakeBackend(), ctx, json_output=True, postprocess=post)
    assert res.text == "FIXED" and res.raw_text == "ok"
    rec = ctx.decisions[-1]
    assert rec["output"] == "FIXED"
    assert rec["raw_output_text"] == "ok"
    assert rec["json_output_normalized"] is True


def test_tool_calls_parsed_and_appended_to_decision():
    ctx = make_ctx(MemoryBundle())
    call = ToolCallRequest(id="c1", name="calc", arguments="{}")
    res = run(FakeBackend(), ctx, tool_count=1, parse_tool_calls=lambda text: [call])
    assert res.tool_calls == [call]
    assert ctx.decisions[-1]["tool_call_request"] == [
        {"id": "c1", "name": "calc", "arguments": "{}"}]


def test_span_sequence_success_and_error():
    ctx = make_ctx(MemoryBundle(), spans=True)
    run(FakeBackend(), ctx)
    types = [t for t, _ in ctx.span_logger.spans]
    assert types == ["model_call_start", "model_call_end"]
    start_fields = ctx.span_logger.spans[0][1]
    assert start_fields["memory_channel"] == "nl"
    assert start_fields["input_messages"][0] == {"role": "system", "content": "SYS"}

    ctx2 = make_ctx(MemoryBundle(), spans=True)
    with pytest.raises(RuntimeError):
        run(FakeBackend(fail=True), ctx2)
    types2 = [t for t, _ in ctx2.span_logger.spans]
    assert types2 == ["model_call_start", "model_call_error"]
