"""数据 loaders（迁移自 benchs/loaders.py）+ benchmark 注册。

每个 task 注册为 `benchmark/<name>`（惰性加载数据，避免 import 时读盘，CLAUDE.md §9）。
`datasets` 库在用到时才惰性导入；数据根目录走环境变量（CDM_DATA_ROOT / CDM_PROCESSED_ROOT）。

统一返回格式：每条 = {task, kind, question, gold, context}。
- kind ∈ {exact, aime, mc, f1}（驱动 eval.metrics.score）
- gold：exact/aime 为 str；mc 为 [text, letter]；f1 为 [str, ...]
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional

from ...core.registry import REGISTRY

RAW = os.environ.get("CDM_DATA_ROOT", os.path.join("data", "raw"))
PROCESSED = os.environ.get("CDM_PROCESSED_ROOT", os.path.join("data", "processed"))

# poles for the communication-channel probe
REASONING_POLE = ("gsm8k", "aime2024")  # expect latent to do relatively well
FACT_POLE = ("medqa", "openbookqa", "arc_easy")  # expect NL to do relatively well
MEMORY_TASKS = ("locomo10",)  # long-memory probe


def _parquet(subdir: str, split: str):
    from datasets import load_dataset  # 惰性导入

    base = os.path.join(RAW, subdir)
    files = glob.glob(os.path.join(base, f"{split}-*.parquet")) or \
        glob.glob(os.path.join(base, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {base}")
    return load_dataset("parquet", data_files={split: files}, split=split)


def _fmt_choices(stem: str, texts, labels) -> str:
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts)]
    return stem.strip() + "\n" + "\n".join(lines) + "\nAnswer with the option letter."


# ---------- reasoning pole ----------
def load_gsm8k(n: Optional[int] = None) -> List[Dict]:
    out = []
    for it in _parquet("gsm8k/main", "test"):
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"task": "gsm8k", "kind": "exact",
                    "question": it["question"].strip() + "\nGive the final numeric answer.",
                    "gold": gold, "context": None})
        if n and len(out) >= n:
            break
    return out


def load_aime2024(n: Optional[int] = None) -> List[Dict]:
    out = []
    for it in _parquet("aime_2024/data", "train"):
        out.append({"task": "aime2024", "kind": "aime",
                    "question": it["problem"].strip() + "\nGive the final integer answer.",
                    "gold": str(it["answer"]).strip(), "context": None})
        if n and len(out) >= n:
            break
    return out


# ---------- fact pole ----------
def _load_arc(subset: str, task: str, n: Optional[int]) -> List[Dict]:
    out = []
    for it in _parquet(f"ai2_arc/{subset}", "test"):
        ch = it["choices"]
        labels = [str(label) for label in ch["label"]]
        gold_letter = str(it["answerKey"])
        gold_text = dict(zip(labels, ch["text"])).get(gold_letter, "")
        out.append({"task": task, "kind": "mc",
                    "question": _fmt_choices(it["question"], ch["text"], labels),
                    "gold": [gold_text, gold_letter], "context": None})
        if n and len(out) >= n:
            break
    return out


def load_arc_easy(n=None):
    return _load_arc("ARC-Easy", "arc_easy", n)


def load_openbookqa(n: Optional[int] = None) -> List[Dict]:
    out = []
    for it in _parquet("openbookqa/main", "test"):
        ch = it["choices"]
        labels = [str(label) for label in ch["label"]]
        gold_letter = str(it["answerKey"])
        gold_text = dict(zip(labels, ch["text"])).get(gold_letter, "")
        out.append({"task": "openbookqa", "kind": "mc",
                    "question": _fmt_choices(it["question_stem"], ch["text"], labels),
                    "gold": [gold_text, gold_letter], "context": None})
        if n and len(out) >= n:
            break
    return out


def load_medqa(n: Optional[int] = None) -> List[Dict]:
    with open(os.path.join(RAW, "medqa.json")) as f:
        data = json.load(f)
    out = []
    for it in data:
        opts = it["options"]  # ["A. ...", ...]
        ans = it["answer"].strip()
        letter = None
        for o in opts:
            m = re.match(r"\s*([A-D])\.\s*(.*)", o)
            if m and m.group(2).strip() == ans:
                letter = m.group(1)
                break
        out.append({"task": "medqa", "kind": "mc",
                    "question": it["question"].strip(),
                    "gold": [ans, letter], "context": None})
        if n and len(out) >= n:
            break
    return out


# ---------- long-memory probe ----------
def load_locomo10(n: Optional[int] = None, max_qa_per_conv: int = 10) -> List[Dict]:
    """Flatten LoCoMo: each record = one question + the full dialogue history text."""
    with open(os.path.join(PROCESSED, "locomo10.json")) as f:
        data = json.load(f)
    out = []
    for conv in data:
        parts = []
        for sid, date, sess in zip(conv["sessions_ids"], conv["sessions_dates"], conv["sessions"]):
            parts.append(f"=== {sid} ({date}) ===")
            parts.extend(sess if isinstance(sess, list) else [str(sess)])
        history = "\n".join(parts)
        for qa in conv["qa"][:max_qa_per_conv]:
            ans = qa.get("answer")
            if ans is None:
                continue
            out.append({"task": "locomo10", "kind": "f1",
                        "question": qa["question"].strip() + "\nAnswer concisely.",
                        "gold": [str(ans)], "context": history})
            if n and len(out) >= n:
                return out
    return out


LOADERS = {
    "gsm8k": load_gsm8k, "aime2024": load_aime2024,
    "arc_easy": load_arc_easy, "openbookqa": load_openbookqa, "medqa": load_medqa,
    "locomo10": load_locomo10,
}


def load(task: str, n: Optional[int] = None) -> List[Dict]:
    return LOADERS[task](n=n)


# ---------- benchmark 注册（惰性：构造不读盘，load() 时才加载） ----------
class _Benchmark:
    """统一基准接口：load(n) 调对应 loader。构造时不触碰磁盘/datasets（CLAUDE.md §9）。"""

    task: str = ""

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def load(self, n: Optional[int] = None) -> List[Dict]:
        return load(self.task, n=self.n if n is None else n)


def _register_benchmarks():
    # 为每个 task 动态生成一个 _Benchmark 子类并注册为 benchmark/<task>
    for _task in LOADERS:
        cls = type(f"Benchmark_{_task}", (_Benchmark,), {"task": _task, "name": _task})
        REGISTRY.register("benchmark", _task)(cls)


_register_benchmarks()

__all__ = ["load", "LOADERS", "REASONING_POLE", "FACT_POLE", "MEMORY_TASKS"]
