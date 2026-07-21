"""AIME 2025 loader and data preparation.

数据布局与 aime_2024 相同：`<prepared_root>/aime_2025/data/train-*.parquet`，列含
problem/answer（本仓库数据根下已有 30 题：Part I + Part II 各 15 题的合并 split）。
上游来源：HuggingFace `opencompass/AIME2025`（part1/part2 两个 config 合并）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .common import load_parquet, parquet_ready, prepared_root


def prepare_aime_2025(force: bool = False, source: Optional[str] = None) -> str:
    """AIME 2025 只支持本地已就绪的 parquet（无自动下载源）。"""
    out_dir = os.path.join(prepared_root(), "aime_2025", "data")
    if parquet_ready("aime_2025/data", "train", required_columns=("problem", "answer")):
        return out_dir
    raise RuntimeError(
        f"AIME 2025 parquet not found or malformed under {out_dir}. "
        "Place opencompass/AIME2025 (part1+part2 merged, columns problem/answer) as "
        "train-*.parquet there; automatic download is not configured for this benchmark."
    )


def load_aime_2025(n: Optional[int] = None) -> List[Dict]:
    prepare_aime_2025()
    out = []
    for it in load_parquet("aime_2025/data", "train"):
        out.append(
            {
                "task": "aime_2025",
                "kind": "aime",
                "question": it["problem"].strip() + "\nGive the final integer answer.",
                "gold": str(it["answer"]).strip(),
                "context": None,
            }
        )
        if n and len(out) >= n:
            break
    return out
