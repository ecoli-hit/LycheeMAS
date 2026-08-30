"""基于原生 LangGraph 的独立基线运行脚本（不使用 lychee_mas 运行时/记忆/插件套件）。

只复用**评测模块**（`lychee_mas.eval`：benchmark 加载 + 答案抽取 + 评分 + 落盘），
执行侧全部用原生 LangGraph API + transformers 直连实现，作为框架无关的对照基线。

五个环节（对应本文件五个区块）：
  ① Benchmark 选择   eval.benchmarks.load(task, n)（19 个已接入任务按名选择）
  ② 网络构建         原生 StateGraph：单 agent 或 planner→solver→verifier 顺序链，
                     条件边终止（输出含 "APPROVE" 或达 len(agents)×max_rounds）
  ③ 执行             app.ainvoke 逐 case 跑（deterministic：do_sample=False）
  ④ 评测             eval.task_config.extractor_for_task 抽答案 + eval.metrics.score 打分
  ⑤ 落盘             outputs.jsonl + metrics.json + config.yaml
                     （eval.metrics.write_results / aggregate_samples，字段与 run_mas 同口径）

用法：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_langgraph_baseline.py --task gsm8k --n 5 \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B
  python scripts/run_langgraph_baseline.py --task aime_2024 --n 3 --team single

依赖：langgraph + torch/transformers（conda env CDM）；无 GPU/数据时显式报错，不静默降级。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import LOADERS  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark
from lychee_mas.eval.task_config import extractor_for_task  # noqa: E402

# ====================== ② 网络构建：agent 画像（脚本内自包含，不用套件模板） ======================

APPROVE_RULE = (
    'When you are confident in the final answer, end your reply with a line '
    '"APPROVE: <final answer>" (for math, put the answer inside \\boxed{...}).'
)

CHAIN3_AGENTS: List[tuple[str, str]] = [
    ("planner",
     "You are the planner. Restate the problem precisely and lay out a short, "
     "numbered solution plan. Do not compute the final answer yourself."),
    ("solver",
     "You are the solver. Follow the planner's plan step by step and derive the answer. "
     "Show the key computations compactly."),
    ("verifier",
     "You are the verifier. Check the solver's derivation for errors. If it is correct, "
     "confirm the answer; if not, correct it. " + APPROVE_RULE),
]

SINGLE_AGENT: List[tuple[str, str]] = [
    ("solver",
     "You are a careful problem solver. Solve the task step by step, then conclude. "
     + APPROVE_RULE),
]

TEAMS: Dict[str, List[tuple[str, str]]] = {"chain3": CHAIN3_AGENTS, "single": SINGLE_AGENT}


class BaselineState(TypedDict):
    """LangGraph 状态：history 条目 = {source, content, prompt_tokens, completion_tokens}。"""

    history: List[Dict[str, Any]]
    agent_turns: int
    done: bool


# ================== 生成后端：transformers 直连（不经 lychee_mas runtime） ==================

@dataclass
class GenOut:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class HFChat:
    """最小 HF 聊天封装：chat 模板 + 贪心生成 + token/时延记账。"""

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device

    def _chat_ids(self, messages: List[Dict[str, str]]):
        try:  # Qwen3 支持 enable_thinking；其他模型不认这个参数则回退
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.device)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenOut:
        import torch

        inputs = self._chat_ids(messages)
        n_prompt = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        latency = time.time() - t0
        gen_ids = out[0][n_prompt:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return GenOut(text=text.strip(), prompt_tokens=n_prompt,
                      completion_tokens=int(gen_ids.shape[0]), latency_s=round(latency, 3))


# ====================== ② 网络构建：原生 StateGraph ======================

def build_app(agents: List[tuple[str, str]], chat: HFChat, max_turns: int,
              max_new_tokens: int) -> Any:
    """把 agent 列表连成顺序轮转的 StateGraph；APPROVE 或轮数上限终止。"""
    from langgraph.graph import END, START, StateGraph

    names = [name for name, _ in agents]

    def make_node(name: str, system_prompt: str) -> Callable[[BaselineState],
                                                             Awaitable[Dict[str, Any]]]:
        async def node(state: BaselineState) -> Dict[str, Any]:
            msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            for entry in state["history"]:
                role = "assistant" if entry["source"] == name else "user"
                msgs.append({"role": role, "content": entry["content"]})
            g = chat.generate(msgs, max_new_tokens=max_new_tokens)
            entry = {"source": name, "content": g.text,
                     "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                     "latency_s": g.latency_s}
            return {"history": state["history"] + [entry],
                    "agent_turns": state["agent_turns"] + 1,
                    "done": state["done"] or ("APPROVE" in g.text)}

        return node

    def route(state: BaselineState) -> str:
        return "end" if (state["done"] or state["agent_turns"] >= max_turns) else "continue"

    graph = StateGraph(BaselineState)
    for name, prompt in agents:
        graph.add_node(name, make_node(name, prompt))
    graph.add_edge(START, names[0])
    for i, name in enumerate(names):
        graph.add_conditional_edges(name, route,
                                    {"continue": names[(i + 1) % len(names)], "end": END})
    return graph.compile()


# ====================== ③ 执行 + ④ 评测（单 case） ======================

async def run_case(app: Any, task: str, record: Dict[str, Any], case_id: str,
                   max_turns: int) -> Dict[str, Any]:
    question = record["question"]
    if record.get("context"):
        question = f"{record['context']}\n\n{question}"  # 无记忆套件：上下文直接并入任务文本
    state: BaselineState = {
        "history": [{"source": "user", "content": question,
                     "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}],
        "agent_turns": 0,
        "done": False,
    }
    t0 = time.time()
    final = await app.ainvoke(state, config={"recursion_limit": 2 * max_turns + 10})
    wall = time.time() - t0

    history = final["history"]
    prediction = extractor_for_task(task)(history)  # eval 模块的答案抽取（兼容 dict 消息）
    kind = record.get("kind") or "exact"
    score = M.score(kind, prediction, record.get("gold"))

    return {
        # 与 run_mas / aggregate_samples 同口径的字段名
        "case_id": case_id,
        "task": task,
        "kind": kind,
        "method": "langgraph_baseline",
        "question": record["question"],
        "gold": record.get("gold"),
        "prediction": prediction,
        "score": score,
        "is_correct": bool(score == 1.0) if M.is_binary_scorer(kind) else None,
        "stop_reason": "APPROVE" if final["done"] else "max_turns",
        "model_call_count": final["agent_turns"],
        "message_count": len(history),
        "input_positions_total": sum(int(e.get("prompt_tokens", 0)) for e in history),
        "gen_tokens": sum(int(e.get("completion_tokens", 0)) for e in history),
        "latency_s": round(sum(float(e.get("latency_s", 0.0)) for e in history), 3),
        "case_wall_time_s": round(wall, 3),
        "messages": [{"source": e["source"], "content": e["content"]} for e in history],
    }


# ====================== ⑤ 主流程：选择 → 构建 → 执行 → 评测 → 落盘 ======================

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=sorted(LOADERS),
                    help="benchmark 名（eval/benchmarks 已接入的 19 个）")
    ap.add_argument("--n", type=int, default=3, help="样本数（0=全量）")
    ap.add_argument("--team", default="chain3", choices=sorted(TEAMS),
                    help="网络形态：single=单 agent；chain3=planner→solver→verifier")
    ap.add_argument("--model-path", default=os.environ.get("LYCHEE_HF_MODEL"),
                    help="HF 模型路径（默认 $LYCHEE_HF_MODEL）")
    ap.add_argument("--model-tag", default=None, help="落盘目录用的模型标签（默认取路径末段）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-rounds", type=int, default=2, help="每 agent 最多发言轮数")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-root", default=None,
                    help="结果根目录（默认 eval.benchmarks.common.runs_root()）")
    args = ap.parse_args()

    if not args.model_path:
        raise SystemExit("需要 --model-path 或环境变量 LYCHEE_HF_MODEL（不做静默兜底）")

    # ① Benchmark 选择
    records = load_benchmark(args.task, n=(args.n or None))
    if not records:
        raise SystemExit(f"benchmark {args.task!r} 加载为空：请先准备数据（benchmarks.prepare）")
    print(f"[baseline] task={args.task} cases={len(records)} team={args.team}")

    # ② 网络构建
    agents = TEAMS[args.team]
    max_turns = len(agents) * args.max_rounds
    chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)
    app = build_app(agents, chat, max_turns=max_turns, max_new_tokens=args.max_new_tokens)

    # ③ 执行 + ④ 评测（逐 case）
    samples: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        case_id = str(record.get("id", i))
        sample = asyncio.run(run_case(app, args.task, record, case_id, max_turns))
        samples.append(sample)
        print(f"  [{i + 1}/{len(records)}] score={sample['score']:.3f} "
              f"stop={sample['stop_reason']} calls={sample['model_call_count']} "
              f"pred={sample['prediction'][:60]!r}")

    # ⑤ 落盘：outputs.jsonl + metrics.json + config.yaml（评测模块的统一产物）
    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    out_dir = M.result_dir(model_tag, "langgraph_baseline", args.task, root=args.out_root)
    run_info = {"task": args.task, "method": "langgraph_baseline",
                "scorer_kind": samples[0]["kind"], "model": model_tag}
    summary = M.aggregate_samples(samples, run_info)
    config = {
        "script": "run_langgraph_baseline.py",
        "git_sha": git_sha(),
        "task": args.task, "n": len(samples), "team": args.team,
        "agents": [name for name, _ in agents],
        "model_path": args.model_path, "model_tag": model_tag,
        "device": args.device, "dtype": args.dtype,
        "max_rounds": args.max_rounds, "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    M.write_results(out_dir, samples, summary, config)

    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[baseline] done: accuracy={acc} out_dir={out_dir}")
    print("[baseline] artifacts: outputs.jsonl / metrics.json / config.yaml")


if __name__ == "__main__":
    main()
