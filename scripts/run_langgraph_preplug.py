"""LangGraph 运行脚本（运行前挂载版）：构建层产图 + pre_run_plugin 插件链变换图。

递进关系（三个脚本逐级验证框架组件的增益）：
  run_langgraph_baseline.py   裸 LangGraph（agent 内联）+ eval
  run_langgraph_construct.py  + 构建层（StaticTopology 按 team 模板产图）
  本脚本                      + **运行前挂载**：图在执行前依序过 pre_run_plugin 链
                                （挂载语义与 Orchestrator 完全一致：按名从 REGISTRY 取、
                                逐个 before_run(graph, query, ctx)、返回值必须是 MASGraph）

仍然**不接**：记忆/路由/注入引擎、runtime 后端、post_run 插件、optimizer、Orchestrator。
执行是原生 StateGraph + transformers 直连；评测/落盘走 eval 模块（与 run_mas 同口径）。

插件指定（可重复，按序执行）：
  --pre-plugin keep_first                       # 无参：注册名
  --pre-plugin 'keep_first:{"k": 2}'            # 带参：名字 + 冒号 + JSON kwargs
  --pre-plugin 'prune:{"pruner": "agentprune"}' # 套件适配器（被包装 pruner 为桩时显式报错）

本脚本自带一个示例插件 `pre_run_plugin/keep_first`（只保留前 k 个 agent），
用于在剪枝算法桩落地前演示挂载链路真实生效（agents 数量变化会体现在落盘记录里）。

用法：
  CDM_DATA_ROOT=/data/.../raw CUDA_VISIBLE_DEVICES=0 \
      python scripts/run_langgraph_preplug.py --task aime_2024 --n 3 \
      --model-path /data/mxy/Models/Qwen/Qwen3-4B --pre-plugin 'keep_first:{"k": 2}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from lychee_mas.core.registry import REGISTRY  # noqa: E402
from lychee_mas.core.types import TaskQuery  # noqa: E402
from lychee_mas.eval import metrics as M  # noqa: E402
from lychee_mas.eval.benchmarks import LOADERS  # noqa: E402
from lychee_mas.eval.benchmarks import load as load_benchmark  # noqa: E402
from lychee_mas.eval.task_config import extractor_for_task, team_name_for_task  # noqa: E402
from lychee_mas.layers.construct.templates import TEAMS, StaticTopology  # noqa: E402
from lychee_mas.plugins.base import RunContext  # noqa: E402
from lychee_mas.runtime.base import MASGraph  # noqa: E402


class RunState(TypedDict):
    """LangGraph 状态：history 条目 = {source, content, prompt_tokens, completion_tokens}。"""

    history: List[Dict[str, Any]]
    agent_turns: int
    done: bool


# ====================== 示例运行前插件（本脚本自带，演示挂载机制） ======================

@REGISTRY.register("pre_run_plugin", "keep_first")
class KeepFirstAgents:
    """只保留前 k 个 agent（产新图，不原地改）。剪枝算法桩落地前的挂载演示件。"""

    name = "keep_first"

    def __init__(self, k: int = 2):
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"keep_first 需要 k >= 1，得到 {self.k}")

    def before_run(self, graph: MASGraph, query: TaskQuery, ctx: RunContext) -> MASGraph:
        return MASGraph(nodes=list(graph.nodes[: self.k]), edges=dict(graph.edges),
                        rounds=graph.rounds, meta=dict(graph.meta))


# ====================== 运行前挂载：解析与执行（语义对齐 Orchestrator） ======================

def parse_plugin_specs(values: List[str]) -> List[tuple[str, dict]]:
    """把 CLI 的 'name' / 'name:{json kwargs}' 解析成 [(name, kwargs)]；非法即显式报错。"""
    specs: List[tuple[str, dict]] = []
    for raw in values:
        name, _, rest = raw.partition(":")
        name = name.strip()
        if not name:
            raise SystemExit(f"非法插件指定 {raw!r}：应为 name 或 name:{{json kwargs}}")
        kwargs: dict = {}
        if rest.strip():
            try:
                kwargs = json.loads(rest)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"插件 {name!r} 的 kwargs 不是合法 JSON：{rest!r}（{exc}）")
            if not isinstance(kwargs, dict):
                raise SystemExit(f"插件 {name!r} 的 kwargs 必须是 JSON 对象，得到 {rest!r}")
        specs.append((name, kwargs))
    return specs


def apply_pre_plugins(graph: MASGraph, query: TaskQuery, ctx: RunContext,
                      plugins: List[tuple[str, Any]]) -> MASGraph:
    """按序执行 pre_run_plugin 链；返回值非 MASGraph 显式 TypeError（与 pipeline 同约定）。"""
    for name, plugin in plugins:
        graph = plugin.before_run(graph, query, ctx)
        if not isinstance(graph, MASGraph):
            raise TypeError(
                f"pre_run_plugin/{name}.before_run 必须返回 MASGraph，"
                f"得到 {type(graph).__name__}")
        if not graph.order():
            raise ValueError(f"pre_run_plugin/{name} 把图剪空了：没有 agent 可执行")
    return graph


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


# ====================== ② 网络构建（构建层）→ ②' 运行前挂载 → StateGraph ======================

def build_masgraph(task: str, team: str | None, max_rounds: int) -> MASGraph:
    """构建层封装函数产基础图：team 名默认按任务选，显式 --team 覆盖。"""
    profile = team or team_name_for_task(task)
    if profile not in TEAMS:
        raise SystemExit(f"未知 team {profile!r}；可用：{sorted(TEAMS)}")
    graph = StaticTopology(team=profile, rounds=max_rounds).build()
    for node in graph.order():
        kind = str(node.meta.get("agent_type") or node.meta.get("type") or "assistant").lower()
        if kind != "assistant" or node.tools or node.meta.get("tools"):
            raise SystemExit(
                f"team {profile!r} 含工具型节点 {node.name!r}（{kind}）；"
                "本脚本仅支持纯文本团队，请换 --team 或用套件的 run_mas.py")
    return graph


def build_app(graph: MASGraph, chat: HFChat, max_turns: int, max_new_tokens: int) -> Any:
    """把 MASGraph 的 AgentSpec 节点连成顺序轮转的 StateGraph（原生 API）。"""
    from langgraph.graph import END, START, StateGraph

    specs = list(graph.order())
    names = [s.name for s in specs]

    def make_node(name: str, system_prompt: str) -> Callable[[RunState],
                                                             Awaitable[Dict[str, Any]]]:
        async def node(state: RunState) -> Dict[str, Any]:
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

    def route(state: RunState) -> str:
        return "end" if (state["done"] or state["agent_turns"] >= max_turns) else "continue"

    sg = StateGraph(RunState)
    for spec in specs:
        sg.add_node(spec.name, make_node(spec.name, spec.system_prompt))
    sg.add_edge(START, names[0])
    for i, name in enumerate(names):
        sg.add_conditional_edges(name, route,
                                 {"continue": names[(i + 1) % len(names)], "end": END})
    return sg.compile()


# ====================== ③ 执行 + ④ 评测（单 case，图经插件链后逐题构建） ======================

async def run_case(base_graph: MASGraph, plugins: List[tuple[str, Any]], chat: HFChat,
                   task: str, record: Dict[str, Any], case_id: str, max_rounds: int,
                   max_new_tokens: int, team: str, run_ctx: RunContext) -> Dict[str, Any]:
    query = TaskQuery(question=record["question"], context=record.get("context"),
                      gold=record.get("gold"), meta={"task": task})
    # ②' 运行前挂载：与 Orchestrator 相同——每条 query 都过一遍插件链（插件可按 query 变换图）
    graph = apply_pre_plugins(base_graph, query, run_ctx, plugins)
    agents_used = [s.name for s in graph.order()]
    max_turns = len(agents_used) * max_rounds
    app = build_app(graph, chat, max_turns=max_turns, max_new_tokens=max_new_tokens)

    question = record["question"]
    if record.get("context"):
        question = f"{record['context']}\n\n{question}"  # 无记忆套件：上下文直接并入任务文本
    state: RunState = {
        "history": [{"source": "user", "content": question,
                     "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}],
        "agent_turns": 0,
        "done": False,
    }
    t0 = time.time()
    final = await app.ainvoke(state, config={"recursion_limit": 2 * max_turns + 10})
    wall = time.time() - t0

    history = final["history"]
    prediction = extractor_for_task(task)(history)
    kind = record.get("kind") or "exact"
    score = M.score(kind, prediction, record.get("gold"))

    return {
        "case_id": case_id,
        "task": task,
        "kind": kind,
        "method": f"langgraph_preplug_{team}",
        "question": record["question"],
        "gold": record.get("gold"),
        "prediction": prediction,
        "score": score,
        "is_correct": bool(score == 1.0) if M.is_binary_scorer(kind) else None,
        "stop_reason": "APPROVE" if final["done"] else "max_turns",
        "agents_used": agents_used,  # 插件链作用后的实际团队（可与 config.agents 对比看剪枝效果）
        "model_call_count": final["agent_turns"],
        "message_count": len(history),
        "input_positions_total": sum(int(e.get("prompt_tokens", 0)) for e in history),
        "gen_tokens": sum(int(e.get("completion_tokens", 0)) for e in history),
        "latency_s": round(sum(float(e.get("latency_s", 0.0)) for e in history), 3),
        "case_wall_time_s": round(wall, 3),
        "messages": [{"source": e["source"], "content": e["content"]} for e in history],
    }


# ====================== ⑤ 主流程：选择 → 构建 → 挂载 → 执行 → 评测 → 落盘 ======================

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
    ap.add_argument("--team", default=None,
                    help="构建层 team 模板名（默认按任务选：task_config.team_name_for_task）")
    ap.add_argument("--pre-plugin", dest="pre_plugins", action="append", default=[],
                    help="运行前插件，可重复按序挂载：name 或 'name:{\"k\": 2}'（JSON kwargs）")
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

    # ② 网络构建（构建层封装函数）
    base_graph = build_masgraph(args.task, args.team, args.max_rounds)
    team = str(base_graph.meta.get("team"))
    base_agents = [s.name for s in base_graph.order()]

    # ②' 运行前挂载：按名从 REGISTRY 实例化（与 Orchestrator._create_plugins 同语义）
    specs = parse_plugin_specs(args.pre_plugins)
    plugins = [(name, REGISTRY.create("pre_run_plugin", name, **kwargs))
               for name, kwargs in specs]
    print(f"[preplug] task={args.task} cases={len(records)} team={team} "
          f"agents={base_agents} pre_plugins={[n for n, _ in plugins] or '无'}")

    chat = HFChat(args.model_path, device=args.device, dtype=args.dtype)
    run_ctx = RunContext(task=args.task)

    # ③ 执行 + ④ 评测
    samples: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        case_id = str(record.get("id", i))
        sample = asyncio.run(run_case(
            base_graph, plugins, chat, args.task, record, case_id,
            args.max_rounds, args.max_new_tokens, team, run_ctx))
        samples.append(sample)
        print(f"  [{i + 1}/{len(records)}] score={sample['score']:.3f} "
              f"stop={sample['stop_reason']} agents={sample['agents_used']} "
              f"pred={sample['prediction'][:60]!r}")

    # ⑤ 落盘
    model_tag = args.model_tag or os.path.basename(os.path.normpath(args.model_path))
    method = f"langgraph_preplug_{team}"
    out_dir = M.result_dir(model_tag, method, args.task, root=args.out_root)
    run_info = {"task": args.task, "method": method,
                "scorer_kind": samples[0]["kind"], "model": model_tag, "team": team}
    summary = M.aggregate_samples(samples, run_info)
    config = {
        "script": "run_langgraph_preplug.py",
        "git_sha": git_sha(),
        "task": args.task, "n": len(samples),
        "team": team, "agents": base_agents,
        "pre_plugins": [{"name": n, "kwargs": k} for n, k in specs],
        "model_path": args.model_path, "model_tag": model_tag,
        "device": args.device, "dtype": args.dtype,
        "max_rounds": args.max_rounds, "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    M.write_results(out_dir, samples, summary, config)

    acc = summary.get("accuracy", summary.get("mean_score"))
    print(f"[preplug] done: accuracy={acc} out_dir={out_dir}")
    print("[preplug] artifacts: outputs.jsonl / metrics.json / config.yaml")


if __name__ == "__main__":
    main()
