"""DualChannelMemoryManager —— CDM 默认记忆方法（在研，CLAUDE.md §5：memory_manager/cdm）。

维护**单一**记忆 source（seed 进来的基础上下文，如 LoCoMo 对话历史 + 运行中的 MAS transcript），
并从这同一个 source 物化出路由器选定的任意通道——保证 apples-to-apples（对比时只变「注入/路由」）。

  channel none   -> 空 bundle（什么都不注入）
  channel nl     -> bundle.nl_text       = 上一个 agent 的输出（NLMemory.recall(query)，不截断）
  channel latent -> bundle.latent_prefix = LatentMemory.build_prefix(source, P)
  channel both   -> 上面两个都产

编码出来的 latent prefix 在单次对话内按 (source 长度, P) 缓存，避免每轮重复编码同一 source。

注册为 `memory_manager/cdm`（自然语言/隐空间动态通道选择，CDM 在研重点）。import 本模块不触发 torch
（NLMemory/LatentMemory 的 torch 已惰性化），满足离线可注册（黄金法则 4）。
"""
from __future__ import annotations

from typing import Any, List

from ....core.registry import REGISTRY
from ..base import MemoryBundle, MemoryManager
from ..channels.latent import LatentMemory
from ..channels.nl import NLMemory
from ..routing.base import RouteDecision


@REGISTRY.register("memory_manager", "cdm")
class DualChannelMemoryManager(MemoryManager):
    name = "cdm"

    def __init__(self, backend, latent_strategy: str = "soft_token",
                 nl_strategy: str = "prev_output",
                 max_encode_tokens: int = 4096, include_transcript: bool = True):
        self.backend = backend
        # NL 通道：按 nl_strategy 选方法
        self.nl = NLMemory(getattr(backend, "tok", None), strategy=nl_strategy)
        # latent 通道：按 latent_strategy 选方法
        self.latent = LatentMemory(backend, latent_strategy, max_encode_tokens)
        self.include_transcript = include_transcript  # 是否把运行中的对话也并入 source
        self._seed = ""  # 基础上下文（任务历史），seed() 预载
        self._transcript: List[str] = []  # 运行中累积的对话行
        self._prefix_cache: dict = {}  # (len(source), P) -> latent prefix 缓存

    # ---- 生命周期 ----
    def reset(self) -> None:
        # 每个样本开始前清空（由 run_one_task 的 ctx.reset 间接驱动 / 显式调用）
        self._seed = ""
        self._transcript = []
        self._prefix_cache = {}

    def seed(self, context_text: str) -> None:
        """预载基础上下文（任务自带的历史），manager 全程持有。长程记忆任务在此灌入对话历史。"""
        self._seed = context_text or ""
        self._prefix_cache.clear()  # source 变了，缓存失效

    def observe(self, messages: List[Any]) -> None:
        """把最新一条消息追加进运行 transcript。

        messages: {role, content} 字典列表（最新在末尾）。只取最后一条，避免每轮把整段历史重复追加。
        内容相同则去重；transcript 一变就清 latent 缓存。
        """
        if not self.include_transcript or not messages:
            return
        last = messages[-1]
        # 兼容 dict 与对象两种消息形态
        is_dict = isinstance(last, dict)
        role = last.get("role", "?") if is_dict else getattr(last, "source", "?")
        content = last.get("content", "") if is_dict else getattr(last, "content", "")
        if isinstance(content, str) and content.strip():
            line = f"[{role}]: {content.strip()}"
            if not self._transcript or self._transcript[-1] != line:  # 与上一行不同才追加（去重）
                self._transcript.append(line)
                self._prefix_cache.clear()  # source 变了，缓存失效

    # ---- 各通道召回所依据的统一 source ----
    def _source(self) -> str:
        # seed 基础上下文 + 运行 transcript 拼成单一来源，NL 与 latent 都从它产出（
        # apples-to-apples）
        parts = []
        if self._seed:
            parts.append(self._seed)
        if self.include_transcript and self._transcript:
            parts.append("\n".join(self._transcript))
        return "\n\n".join(parts)

    # ---- 物化通道 ----
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle:
        if decision.channel == "none":
            return MemoryBundle(meta={"channel": "none"})  # 什么都不注入
        bundle = MemoryBundle(meta={"channel": decision.channel})
        if decision.uses_nl():  # nl / both
            # NL 通道默认 = 向下传递上一个 agent 的输出（query），原样不截断
            bundle.nl_text = self.nl.recall(query)
        if decision.uses_latent():  # latent / both
            source = self._source()  # latent 仍压缩完整 source（seed + transcript）
            if source:
                key = (len(source), decision.P)  # 同 source 同 P 复用缓存，省重复编码
                if key not in self._prefix_cache:
                    self._prefix_cache[key] = self.latent.build_prefix(source, decision.P)
                bundle.latent_prefix = self._prefix_cache[key]
        return bundle
