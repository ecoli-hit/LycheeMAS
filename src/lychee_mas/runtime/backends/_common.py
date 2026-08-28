"""运行时后端共用的纯函数（autogen / langgraph 两后端共享，避免行为漂移）。

零重依赖：消息既可能是引擎对象（带 .source/.content 属性）也可能是 dict，
统一用 duck-typing 读取。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def _msg_field(msg: Any, name: str, default: Any = "") -> Any:
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def content_to_text(content: Any) -> str:
    """把消息 content（str / 列表 / pydantic 对象 / 任意对象）尽力转成文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(content_to_text(item) for item in content)
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


def jsonable(value: Any) -> Any:
    """把任意值转成可 JSON 序列化的结构（落盘 spans / meta 用）。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def extract_final_answer(messages: list[Any], *, prefer_code_from: Optional[str] = None) -> str:
    """从消息列表提取 'FINAL ANSWER: ...'（可选优先取某来源的代码块）。"""
    if prefer_code_from:
        for msg in reversed(messages):
            source = _msg_field(msg, "source")
            text = content_to_text(_msg_field(msg, "content"))
            if source == prefer_code_from and "```" in text:
                return text.strip()

    for msg in reversed(messages):
        text = content_to_text(_msg_field(msg, "content"))
        match = re.search(r"FINAL ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""
