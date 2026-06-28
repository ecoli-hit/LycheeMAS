"""评分 + 结果落盘（迁移自 benchs/metrics.py + math_parsing_util.py）。

评分类型（kind）：
  - "mc":    选择题；预测里出现 gold 选项字母 或 gold 文本即算对。用于 medqa/arc/openbookqa。
  - "exact": 数值/短答规整后精确匹配。用于 gsm8k。
  - "aime":  数学答案等价比较（数值 + sympy 符号），支持整数/分式/根式/带符号式子。用于 aime。
  - "f1":    token 级 F1（对 gold 列表取最大），MemGAS 口径。用于 locomo/longmemeval。

CLAUDE.md §9：所有跑分必须同时报性能与成本——`result_dir`/`write_results` 落 metrics+outputs+config。

⚠️ 惰性导入：`score_aime` 用到的 math_parsing_util（依赖 sympy/regex/latex2sympy2/word2number）
与 `write_results` 用到的 yaml 都在函数内部 import，保证本模块在无这些库时也可被 import（黄金法则
4）。
"""
from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import List, Optional


# ---- 规整 / 评分 ----
def _norm(s: str) -> str:
    # 统一小写、去标点、去冠词(a/an/the)、压空白——SQuAD 式规整，使匹配更鲁棒
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _last_number(s: str) -> Optional[str]:
    # 抽取字符串里最后一个数字（去千分位逗号）；数学题答案通常在末尾
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def score_exact(pred: str, gold: str) -> float:
    g = _last_number(gold) or _norm(gold)
    p = _last_number(pred)
    if p is not None and g is not None and p == g:
        return 1.0  # 数值匹配优先
    return 1.0 if _norm(gold) and _norm(gold) in _norm(pred) else 0.0  # 否则退化为子串包含


def score_mc(pred: str, gold_text: str, gold_letter: Optional[str]) -> float:
    np_ = _norm(pred)
    if gold_letter:
        # 在预测开头 40 字符内匹配独立的选项字母，如 "A" / "(A)" / "answer: b"
        if re.search(rf"\b{re.escape(gold_letter.lower())}\b", np_[:40]):
            return 1.0
    return 1.0 if _norm(gold_text) and _norm(gold_text) in np_ else 0.0  # 或选项文本出现在预测里


def score_f1(pred: str, golds: List[str]) -> float:
    # token 级 F1，对多个 gold 取最大值（locomo 一题可有多个可接受答案）
    pt = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)  # 词频交集 = 命中词数
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_aime(pred: str, gold: str) -> float:
    """AIME/数学评分：两边各自归一化后做数学等价比较（数值 + sympy 符号）。

    math_parsing_util 在此惰性导入（它依赖 sympy/regex/latex2sympy2/word2number）。
    """
    # 数学答案抽取/等价比较工具（math_parsing_util 里函数名无 math_ 前缀）
    from .math_parsing_util import extract_answer, math_equal, strip_answer_string

    def _math_norm(s: str) -> str:
        r"""归一化数学答案。

        含 \boxed/"answer is" 标记时用 extract_answer 抽取（它能正确取出 \boxed{} 里的表达式）；
        否则只做 strip_answer_string 归一化。aime 提取器已剥掉 \boxed，pred 多是裸表达式，故走
        strip 分支。
        """
        s = str(s)
        return extract_answer(s) if "boxed" in s else strip_answer_string(s)

    p = _math_norm(pred)
    g = _math_norm(gold)
    return 1.0 if math_equal(p, g) else 0.0


def score(kind: str, pred: str, gold) -> float:
    # 统一评分入口：按 kind 分发；gold 可能是 str / (text, letter) / list
    if kind == "exact":
        return score_exact(pred, gold if isinstance(gold, str) else gold[0])
    if kind == "aime":
        return score_aime(pred, gold if isinstance(gold, str) else gold[0])
    if kind == "mc":
        gt, gl = gold if isinstance(gold, (list, tuple)) else (gold, None)  # (选项文本, 选项字母)
        return score_mc(pred, gt, gl)
    if kind == "f1":
        return score_f1(pred, gold if isinstance(gold, list) else [gold])
    raise ValueError(kind)


# ---- 落盘 ----
def result_dir(model: str, method: str, task: str, root: Optional[str] = None) -> str:
    # 按 CLAUDE.md §9 布局拼出叶子目录并创建：root/[model]/[method]/[task]/
    root = root or os.path.join("runs", "autogen_base")
    d = os.path.join(root, model, method, task)
    os.makedirs(d, exist_ok=True)
    return d


def write_results(out_dir: str, samples: List[dict], metrics: dict, config: dict):
    # 三个产物：每样本明细 / 汇总指标 / 可复现配置。yaml 惰性导入（写 config 时才需要）。
    with open(os.path.join(out_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            # 每行一个样本（含 routing_trace+transcript）
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)  # Pareto 主图的数据源
    _dump_config(os.path.join(out_dir, "config.yaml"), config)


def _dump_config(path: str, config: dict) -> None:
    """落配置快照：优先 yaml（可读），无 yaml 时退回 JSON（保证离线也能落盘）。"""
    try:
        import yaml  # 惰性导入

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
