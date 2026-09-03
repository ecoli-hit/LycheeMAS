"""AgentDropout —— 动态节点/边淘汰（graph_pruner/agentdropout 的真实现）。

复现自：AgentDropout: Dynamic Agent Elimination for Token-Efficient and
High-Performance LLM-Based Multi-Agent Collaboration（ACL 2025，arXiv:2503.18891）；
参考实现 https://github.com/wangzx1219/AgentDropout（graph/graph.py 与
experiments/run_gsm8k.py 的机制逐一对应）。与 AgentPrune 同族代码基，增量在于
**逐轮独立参数（diff）+ 两阶段淘汰**：

- **阶段一 Node Dropout**：逐轮持有可训练度权重 ``deg_logits[r]``（原版 spatial_logits_1）。
  每次 rollout：按固定掩码确定性实现全图（贪心去环，DAG），对每个节点求「实现边上的
  logits 加权度」，softmax 后**多项式采样**一个节点本轮跳过（其输出置 'None.'）；
  skip-REINFORCE：loss = −utility × [4·Σ_{被跳节点边} log(1−σ) + Σ_{其余节点边} log σ]
  （逐节点循环使非跳边被两端各计一次，忠实保留该双计语义），Adam lr=0.1。
  训练后 ``node_dropout()``（原版 update_masks_dec）：每轮取「全行列 logits 和 / 固定
  度数」最小的节点，整轮淘汰——空间行列清零 + 该轮入向/出向时间边清零，记入 skip_nodes。
- **阶段二 Edge Dropout**：在淘汰后掩码上做逐轮 AgentPrune 式伯努利采样 + REINFORCE
  （参数 ``spatial_logits[r]`` / ``temporal_logits[r]``），后 ``edge_dropout(rate)``
  （原版 update_masks_diff）逐轮按 logit 升序置零 round(存活×rate) 条边（强制 ≥1 条）。
- **评测**：threshold 确定性实现（σ(logit)>0.5 且未被剪）逐轮产邻接；被淘汰节点
  该轮不执行（runner 依据 ``skip_nodes``）。

与原版的声明差异：
1. 原版发布代码中核范数/Frobenius 正则在构造后立即 ``add_loss=0`` 禁用——本移植
   跟随**实际执行路径**（纯 skip-REINFORCE），不实现该正则；
2. 纯标准库逐边字典参数 + 自实现 Adam（与 AgentPrune 移植同款）；n 个 agent 通用
   （原版硬编码 5）；采样用实例内 seeded RNG（可复现）。

时间边索引约定：``temporal_*[r]``（r ∈ 1..rounds-1）= 第 r-1 轮 → 第 r 轮的边
（原版 ``temporal_logits[r-1]``）。图级挂载见 ``plugins/prerun/agentdropout_lg.py``；
训练循环由实验脚本驱动，本类只负责「采样 / 更新 / 淘汰 / 产出实现矩阵」。
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ...core.registry import REGISTRY
from .agentprune import Edge, Realization, _Adam, full_connected_masks


def acyclic_realization(n: int, alive: set[Edge]) -> set[Edge]:
    """按固定 i-主序贪心保留不成环的边（原版 check_cycle 的确定性实现语义）。"""
    kept: set[Edge] = set()

    def reaches(src: int, dst: int) -> bool:
        stack, seen = [src], set()
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for a, b in kept:
                if a == cur and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return False

    for i in range(n):
        for j in range(n):
            if (i, j) in alive and not reaches(j, i):
                kept.add((i, j))
    return kept


@REGISTRY.register("graph_pruner", "agentdropout")
class AgentDropoutOptimizer:
    """AgentDropout 两阶段淘汰器（逐轮参数；训练接口 + 确定性实现）。"""

    name = "agentdropout"

    def __init__(self, n_agents: int, rounds: int = 2, lr: float = 0.1,
                 initial_probability: float = 0.5, temperature: float = 1.0,
                 seed: int = 0, state_file: Optional[str] = None) -> None:
        if n_agents < 2 or rounds < 1:
            raise ValueError(f"需要 n_agents>=2 且 rounds>=1，得到 {n_agents}/{rounds}")
        self.n = int(n_agents)
        self.rounds = int(rounds)
        self.temperature = float(temperature)
        self.rng = random.Random(seed)
        fs, ft = full_connected_masks(self.n)
        logit0 = math.log(initial_probability / (1.0 - initial_probability))
        all_edges = [(i, j) for i in range(self.n) for j in range(self.n)]
        # 阶段一：逐轮度权重（原版 spatial_logits_1，init=log(p/(1-p)) 同款）
        self.deg_logits: List[Dict[Edge, float]] = [
            {e: logit0 for e in all_edges} for _ in range(self.rounds)]
        # 阶段二：逐轮边参数 + 掩码（temporal[r] 存在于 r>=1）
        self.spatial_logits: List[Dict[Edge, float]] = [
            {e: logit0 for e in all_edges} for _ in range(self.rounds)]
        self.spatial_masks: List[Dict[Edge, int]] = [
            {(i, j): fs[i][j] for i, j in all_edges} for _ in range(self.rounds)]
        self.temporal_logits: Dict[int, Dict[Edge, float]] = {
            r: {e: logit0 for e in all_edges} for r in range(1, self.rounds)}
        self.temporal_masks: Dict[int, Dict[Edge, int]] = {
            r: {(i, j): ft[i][j] for i, j in all_edges} for r in range(1, self.rounds)}
        self.skip_nodes: Dict[int, int] = {}  # round -> 被整轮淘汰的节点下标
        self._adam_deg = _Adam(lr)
        self._adam_s = _Adam(lr)
        self._adam_t = _Adam(lr)
        if state_file:
            self.load(state_file)

    # ------------------ 阶段一：Node Dropout（skip 采样 + REINFORCE） ------------------

    def realized_full(self) -> set[Edge]:
        """固定掩码全图的确定性无环实现（阶段一 rollout 用；对角不参与）。"""
        alive = {(i, j) for i in range(self.n) for j in range(self.n)
                 if i != j and self.spatial_masks[0][(i, j)] == 1}
        return acyclic_realization(self.n, alive)

    def sample_skip(self, r: int, edges: Optional[set[Edge]] = None
                    ) -> Tuple[int, set[Edge]]:
        """按「实现边上的 deg_logits 加权度」softmax 多项式采样本轮跳过的节点。"""
        edges = self.realized_full() if edges is None else edges
        weights = []
        for i in range(self.n):
            w = sum(self.deg_logits[r][e] for e in edges if i in e)
            weights.append(w)
        mx = max(weights)
        probs = [math.exp(w - mx) for w in weights]
        total = sum(probs)
        pick, acc = self.rng.uniform(0.0, total), 0.0
        skip = self.n - 1
        for i, p in enumerate(probs):
            acc += p
            if pick <= acc:
                skip = i
                break
        return skip, edges

    def skip_reinforce(self, batch: List[Tuple[List[Tuple[int, set[Edge]]], float]]) -> None:
        """阶段一 Adam 步：loss = mean_query(−u × Σ_rounds[4Σ_跳边 log(1−σ) + Σ_余边 log σ])。

        每个 batch 项 = ([(每轮 skip_idx, 该轮实现边集)], utility)。梯度（忠实含双计）：
        边 (a,b) 触及被跳节点 s 的端点次数 k_s、非跳端点次数 k_o →
        ∂loss/∂l = −u·[k_s·4·(−σ(l)) + k_o·(1−σ(l))]。
        """
        if not batch:
            return
        grads: List[Dict[Edge, float]] = [dict() for _ in range(self.rounds)]
        for per_round, u in batch:
            if len(per_round) != self.rounds:
                raise ValueError(f"每题需 {self.rounds} 轮 skip 记录，得到 {len(per_round)}")
            for r, (skip, edges) in enumerate(per_round):
                for e in edges:
                    sig = 1.0 / (1.0 + math.exp(-self.deg_logits[r][e]))
                    k_s = (1 if e[0] == skip else 0) + (1 if e[1] == skip else 0)
                    k_o = 2 - k_s - (1 if e[0] == e[1] else 0)
                    g = -u * (k_s * 4.0 * (-sig) + k_o * (1.0 - sig))
                    grads[r][e] = grads[r].get(e, 0.0) + g
        k = float(len(batch))
        flat_grads = {(r, e): g / k for r in range(self.rounds)
                      for e, g in grads[r].items()}
        if flat_grads:  # (r, e) 扁平键：各轮参数的 Adam 矩量互不串扰
            flat = {key: self.deg_logits[key[0]][key[1]] for key in flat_grads}
            self._adam_deg.step(flat, flat_grads)
            for (r, e), v in flat.items():
                self.deg_logits[r][e] = v

    def node_dropout(self) -> Dict[int, int]:
        """原版 update_masks_dec：每轮淘汰「全行列 logits 和 / 固定度数」最小的节点。"""
        fs, _ft = full_connected_masks(self.n)
        for r in range(self.rounds):
            best, best_score = -1, float("inf")
            for j in range(self.n):
                total = sum(self.deg_logits[r][(j, k)] + self.deg_logits[r][(k, j)]
                            for k in range(self.n))
                count = sum(fs[j][k] + fs[k][j] for k in range(self.n))
                score = total / count
                if score < best_score:
                    best_score, best = score, j
            self.skip_nodes[r] = best
            for k in range(self.n):
                self.spatial_masks[r][(best, k)] = 0
                self.spatial_masks[r][(k, best)] = 0
                if r >= 1:
                    self.temporal_masks[r][(k, best)] = 0   # 上一轮 → 本轮被淘汰者
                if r + 1 <= self.rounds - 1:
                    self.temporal_masks[r + 1][(best, k)] = 0  # 被淘汰者 → 下一轮
        return dict(self.skip_nodes)

    # ------------------ 阶段二：Edge Dropout（逐轮采样 + REINFORCE） ------------------

    def _prob(self, logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    def sample_round(self, r: int) -> Realization:
        """第 r 轮伯努利实现（空间对本轮存活边；r>=1 另采上一轮→本轮时间边）。"""
        real = Realization(spatial_edges=set(), temporal_edges=set())
        for e, m in self.spatial_masks[r].items():
            if m == 0 or e[0] == e[1]:
                continue
            p = self._prob(self.spatial_logits[r][e])
            b = 1 if self.rng.random() < p else 0
            real.spatial_samples[e] = (b, p)
            if b:
                real.spatial_edges.add(e)
        if r >= 1:
            for e, m in self.temporal_masks[r].items():
                if m == 0:
                    continue
                p = self._prob(self.temporal_logits[r][e])
                b = 1 if self.rng.random() < p else 0
                real.temporal_samples[e] = (b, p)
                if b:
                    real.temporal_edges.add(e)
        return real

    def edge_reinforce(self, batch: List[Tuple[List[Realization], float]]) -> None:
        """阶段二 Adam 步（AgentPrune 同款公式，参数逐轮独立）。"""
        if not batch:
            return
        gs: Dict[Tuple[int, Edge], float] = {}
        gt: Dict[Tuple[int, Edge], float] = {}
        for reals, u in batch:
            if len(reals) != self.rounds:
                raise ValueError(f"每题需 {self.rounds} 轮实现，得到 {len(reals)}")
            for r, real in enumerate(reals):
                for e, (b, p) in real.spatial_samples.items():
                    gs[(r, e)] = gs.get((r, e), 0.0) - u * (b - p) / self.temperature
                for e, (b, p) in real.temporal_samples.items():
                    gt[(r, e)] = gt.get((r, e), 0.0) - u * (b - p) / self.temperature
        k = float(len(batch))
        if gs:
            flat = {key: self.spatial_logits[key[0]][key[1]] for key in gs}
            self._adam_s.step(flat, {key: g / k for key, g in gs.items()})
            for (r, e), v in flat.items():
                self.spatial_logits[r][e] = v
        if gt:
            flat = {key: self.temporal_logits[key[0]][key[1]] for key in gt}
            self._adam_t.step(flat, {key: g / k for key, g in gt.items()})
            for (r, e), v in flat.items():
                self.temporal_logits[r][e] = v

    def edge_dropout(self, pruning_rate: float) -> Dict[str, int]:
        """原版 update_masks_diff：逐轮按 logit 升序置零 round(存活×rate) 条边（强制 ≥1）。"""
        def prune_round(logits: Dict[Edge, float], masks: Dict[Edge, int],
                        skip_diag: bool) -> int:
            alive = [e for e, m in masks.items() if m == 1 and not (skip_diag and e[0] == e[1])]
            if not alive:
                return 0
            k = int(round(len(alive) * pruning_rate)) or 1  # 原版：至少剪 1 条
            for e in sorted(alive, key=lambda e: logits[e])[:k]:
                masks[e] = 0
            return k

        ks = sum(prune_round(self.spatial_logits[r], self.spatial_masks[r], True)
                 for r in range(self.rounds))
        kt = sum(prune_round(self.temporal_logits[r], self.temporal_masks[r], False)
                 for r in range(1, self.rounds))
        return {"spatial_pruned": ks, "temporal_pruned": kt}

    # ------------------------- 确定性实现（评测 / 挂载） -------------------------

    def realized_matrices(self, r: int, mode: str = "threshold"
                          ) -> tuple[List[List[int]], List[List[int]]]:
        """第 r 轮 (spatial, temporal) 邻接矩阵。threshold: σ>0.5 且未被剪；sample: 伯努利。"""
        if not 0 <= r < self.rounds:
            raise ValueError(f"round 必须在 [0,{self.rounds})，得到 {r}")
        if mode == "sample":
            real = self.sample_round(r)
            spatial, temporal = real.spatial_edges, real.temporal_edges
        elif mode == "threshold":
            spatial = {e for e, m in self.spatial_masks[r].items()
                       if m == 1 and e[0] != e[1] and self._prob(self.spatial_logits[r][e]) > 0.5}
            temporal = set() if r == 0 else {
                e for e, m in self.temporal_masks[r].items()
                if m == 1 and self._prob(self.temporal_logits[r][e]) > 0.5}
        else:
            raise ValueError(f"未知 realized 模式 {mode!r}（threshold|sample）")
        sm = [[1 if (i, j) in spatial else 0 for j in range(self.n)] for i in range(self.n)]
        tm = [[1 if (i, j) in temporal else 0 for j in range(self.n)] for i in range(self.n)]
        return sm, tm

    # ------------------------------- 状态持久化 -------------------------------

    def state_dict(self) -> dict:
        def pack(d: Dict[Edge, Any]) -> Dict[str, Any]:
            return {f"{i},{j}": v for (i, j), v in d.items()}

        return {"n_agents": self.n, "rounds": self.rounds, "temperature": self.temperature,
                "deg_logits": [pack(d) for d in self.deg_logits],
                "spatial_logits": [pack(d) for d in self.spatial_logits],
                "spatial_masks": [pack(d) for d in self.spatial_masks],
                "temporal_logits": {str(r): pack(d) for r, d in self.temporal_logits.items()},
                "temporal_masks": {str(r): pack(d) for r, d in self.temporal_masks.items()},
                "skip_nodes": {str(r): v for r, v in self.skip_nodes.items()}}

    def load_state_dict(self, state: dict) -> None:
        if int(state["n_agents"]) != self.n or int(state["rounds"]) != self.rounds:
            raise ValueError(
                f"状态 n_agents/rounds={state['n_agents']}/{state['rounds']} "
                f"与当前 {self.n}/{self.rounds} 不符")

        def unpack(d: Dict[str, Any], cast) -> Dict[Edge, Any]:
            return {(int(k.split(",")[0]), int(k.split(",")[1])): cast(v)
                    for k, v in d.items()}

        self.deg_logits = [unpack(d, float) for d in state["deg_logits"]]
        self.spatial_logits = [unpack(d, float) for d in state["spatial_logits"]]
        self.spatial_masks = [unpack(d, int) for d in state["spatial_masks"]]
        self.temporal_logits = {int(r): unpack(d, float)
                                for r, d in state["temporal_logits"].items()}
        self.temporal_masks = {int(r): unpack(d, int)
                               for r, d in state["temporal_masks"].items()}
        self.skip_nodes = {int(r): int(v) for r, v in state["skip_nodes"].items()}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.load_state_dict(json.load(f))
