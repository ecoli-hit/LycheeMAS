"""L3 通道实现：NL（自然语言）+ Latent（隐空间 soft prefix）。

- nl.py     NL 通道 + PREV_OUTPUT_HEADER（全包唯一来源）
- latent.py Latent 通道：把上下文压成 (1,P,H) soft prefix（torch 惰性导入）
"""
from __future__ import annotations

from .latent import LatentMemory, compress_hidden, soft_token_prefix
from .nl import PREV_OUTPUT_HEADER, NLMemory

__all__ = [
    "NLMemory",
    "PREV_OUTPUT_HEADER",
    "LatentMemory",
    "compress_hidden",
    "soft_token_prefix",
]
