# `plugins` — 插件系统（运行前 / 运行后 / 离线优化）

## 功能

把「优化器」挂在执行外部的三种生命周期，三接口分立：

- **PreRunPlugin**（类别 `pre_run_plugin`）：每次执行前变换图 G（如剪枝）。
- **PostRunPlugin**（类别 `post_run_plugin`）：每次执行后消费轨迹 τ（如归因/信用）。
- **Optimizer**（类别 `optimizer`）：GEPA 式离线 compile 循环，迭代改进系统的可变异文本组件。

挂载：`Orchestrator(pre_plugins=[...], post_plugins=[...])`，每项为名字或 `(名字, kwargs)`；不配置时行为与无插件完全一致。Optimizer 由驱动脚本直接 `REGISTRY.create("optimizer", ...)` 调用。

## 接口（`base.py`）

```python
@dataclass
class RunContext:                       # 插件可见的运行上下文
    task: str; trace_store: Any; routing_ctx: Any; backend: Any; meta: dict

class PreRunPlugin(Protocol):
    def before_run(self, graph: MASGraph, query: TaskQuery, ctx: RunContext) -> MASGraph: ...

class PostRunPlugin(Protocol):
    def after_run(self, trajectory: Trajectory, score: Optional[float], ctx: RunContext) -> None: ...

Metric = Callable[[Trajectory, TaskQuery], float]

class Optimizer(Protocol):
    def optimize(self, system: MASProgram, trainset: Sequence[TaskQuery], metric: Metric) -> MASProgram: ...
```

`before_run` 必须返回 MASGraph——编排器校验，非法返回显式 TypeError（显式错误原则）。

## `program.py` — `MASProgram`

系统的「可变异文本组件」视图（Optimizer 的操作对象）：`components: dict[str, str]`（键如 `agent:<name>:system_prompt`、`topology:description`）+ `mutable_keys`。方法：`from_graph(graph)` / `apply_to(graph) -> 新图` / `mutated(key, text) -> 新 program`（不原地改；不可变异组件 / 图中不存在的 agent 显式 KeyError）。

## 已注册组件

| 类别/名字 | 说明 |
|---|---|
| `pre_run_plugin/prune` | 适配器：包装任意已注册 `graph_pruner`（构造参数 `pruner=<name>`） |
| `post_run_plugin/attribution` | 适配器：串 attributor + credit_assigner，写 `trajectory.meta["attribution"/"credits"]` + TraceStore |
| `optimizer/gepa` | GEPA 反思式提示演化（`gepa/`）：候选池 + per-instance Pareto 采样（`gepa/pareto.py` 纯函数）→ 轮换选可变组件 → minibatch rollout 收反馈 → 反思变异（`reflector` 可注入，默认走 `backend.generate_chat`）→ minibatch 提升才全量评估入池 → `max_metric_calls` 硬预算耗尽返回均分最优。`rollout: (MASProgram, TaskQuery) -> Trajectory` 必须注入 |

## `prerun/` — LangGraph 原生运行前优化（类别 `pre_run_optimizer`）

与上面三接缝并存的**第四个接缝**：算法直接在 LangGraph 图上做运行前优化，图进图出。

```python
from lychee_mas.plugins.prerun import optimize_langgraph

sg = optimize_langgraph(sg, method="maspo", mode="apply", prompt_file="p.json")   # 即插即用
sg = optimize_langgraph(sg, method="maspo", mode="optimize", trainset=[...],
                        agent_llm=..., evaluator_llm=..., prompt_file="p.json")   # 离线优化
sg = optimize_langgraph(sg, method="agentprune", state_file="state.json")         # 剪边
app = sg.compile()
```

- **节点契约**（`graphview.py`）：`sg.add_node(name, fn, metadata={"agent_spec": spec})`；`spec.system_prompt` = 可变异提示模板（`{question}`/`{context}`）；`spec.meta["predecessors"]` = 通信前驱；节点函数运行时从 spec 读——换 spec 即换行为，无需重建节点。只支持静态 DAG + 唯一终端，违反显式报错。
- **`pre_run_optimizer/maspo`**（`maspo/`，MASPO ICML 2026）：多粒度成对评估（Local/Lookahead/Global，免 gold）+ 错位驱动采样 + 进化 beam search + fixed-rounds 坐标上升 + Beam Refresh；optimize 落 `{"prompts": {节点名: 提示}}` JSON，apply 加载注入。提示资产逐字 vendored（`maspo/prompts.py`，保留出处引用）。
- **`pre_run_optimizer/agentprune`**：复用 `graph_pruner/agentprune` 的 threshold 实现剪 LangGraph 边（与 MASGraph 路径对拍一致，见 `tests/test_prerun.py`）。
- langgraph 在本包内**一律惰性导入**（`make selfcheck` 仍须 HEAVY LOADED: NONE）。

## 约定

- 纯标准库、零重依赖；LLM 只经由传入的 backend 触达。
- 插件对不支持的输入显式 raise，不静默降级；被包装的桩组件（如 `agentdropout`）的 `NotImplementedError` 如实上抛。
- 新增插件：实现对应协议 + `@REGISTRY.register("<category>", name)` + 本包 `__init__` import 触发 + config + test。
