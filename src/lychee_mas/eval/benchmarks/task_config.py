"""Task-specific answer extraction policies.

Benchmark loaders and scorers must not select a TeamSpec. Experiment configs
choose role profiles/teams independently; this module only controls how a
completed trajectory exposes its candidate answer.

未登记的 task 用 DEFAULT_EXTRACTOR 兜底。纯标准库（仅 re）。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional, cast


# ====================== 答案提取策略（按任务分发） ======================
def _content_text(content) -> str:
    """Render AutoGen text, multimodal, and structured message content safely."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False, default=str)
    if hasattr(content, "model_dump"):
        return _content_text(content.model_dump(mode="json"))
    if is_dataclass(content):
        return _content_text(asdict(cast(Any, content)))
    return str(content)


def _extract_default(messages) -> str:
    """默认策略：优先取最后一条含 'APPROVE:' 的内容；没有则回退最后一条消息。"""
    visible_messages = [
        message
        for message in messages
        if (
            message.get("type") if isinstance(message, dict)
            else getattr(message, "type", type(message).__name__)
        )
        != "ThoughtEvent"
    ]
    for m in reversed(visible_messages):  # 从后往前找第一条 APPROVE
        raw = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
        c = _content_text(raw)
        if "APPROVE" in c:
            matches = list(re.finditer(r"(?im)^\s*APPROVE:\s*(.+?)\s*$", c))
            if matches:
                return matches[-1].group(1).strip()
    last = visible_messages[-1] if visible_messages else None  # 没 APPROVE 则用末条可见消息
    if last is None:
        return ""
    raw = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
    return _content_text(raw).strip()


def _extract_boxed(messages) -> str:
    r"""数学/AIME 策略：在 default 基础上，抽出 APPROVE 行里 \boxed{...} 内的答案。

    答案可能是整数（含负数）或带符号/分式/根式的式子，故取 \boxed{} 里的原始内容、不强转整数。
    """
    ans = _extract_default(messages)
    m = re.search(r"\\boxed\{(.*)\}", ans, re.DOTALL)
    return m.group(1).strip() if m else ans  # 没有 \boxed 就退回原文本


# 命名策略表
EXTRACTORS: dict[str, Callable[[list], str]] = {
    "default": _extract_default,
    "boxed": _extract_boxed,
}


# ====================== 每个 task 的默认配置 ======================
DEFAULT_EXTRACTOR = "default"  # 未登记 task 的默认答案提取策略

def _registered_task_config() -> dict[str, dict[str, str]]:
    from .registry import BENCHMARKS

    return {
        task: {"extractor": benchmark.extractor_names.get(task, DEFAULT_EXTRACTOR)}
        for benchmark in BENCHMARKS.all()
        for task in benchmark.runnable_tasks
    }


# Compatibility view for callers that inspect task configuration directly.
# Benchmark objects are the single source of truth.
TASK_CONFIG: dict[str, dict[str, str]] = _registered_task_config()


def extractor_for_task(task: Optional[str]) -> Callable[[list], str]:
    """按任务名选出答案提取策略；未登记回退 DEFAULT_EXTRACTOR。"""
    name = TASK_CONFIG.get(task or "", {}).get("extractor", DEFAULT_EXTRACTOR)
    return EXTRACTORS[name]
