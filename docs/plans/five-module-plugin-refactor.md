# 五模块全插件化重构方案

> 状态：**方案 + Phase 0 已执行**（2026-09-03）。用户决策来源："我们目前还是整体五个模块，
> 但是都要用 plugin 的形式。不属于这个设计逻辑要么删除要么进行结构化迁移。"
> 基准形态 = 当前挂载（`plugins/lg_prerun`）：统一入口 + method 按名分发 + 图进图出/包裹执行
> + optimize/apply 两段式 + AgentSpec 节点契约。

---

## 1. 目标形态：五个模块 = 五个挂载式接缝

全部 LangGraph 原生；每个接缝一个统一入口、一个 REGISTRY 类别、换算法 = 换 `method`：

```
src/lychee_mas/plugins/
├── build/        构建   build_langgraph(method, query, **kw) -> StateGraph
│                 [graph_builder]  产出契约图（metadata=agent_spec + meta.predecessors）
├── lg_prerun/    运行前 optimize_langgraph(sg, method, **kw) -> sg          ✅ 已就位（范式样板）
│                 [pre_run_optimizer]  agentprune / maspo（/ 未来 gepa）
├── memory/       运行时 attach_memory(sg, method, backend, **kw) -> sg
│                 [memory_manager + memory_router]  注入六步包裹进每个 agent 节点
├── processing/   执行   run_processed(sg, method, k, aggregator, **kw) -> 结果
│                 [processor + aggregator]  serial / parallel×K + 归约（pass@K 承载点）
└── postrun/      运行后 analyze_run(result, method, **kw) -> 归因/信用
                  [post_run_optimizer]  attribution（attributor+credit）/ 未来训练信号
```

生命周期：`build → prerun(可选) → memory(可选) → compile → processing 包裹执行 → postrun(可选)`。
每一环不配置时行为与不挂载完全一致（零回归原则沿用）。

## 2. 现存模块处置表

| 现位置 | 处置 | 去向/理由 |
|---|---|---|
| `plugins/lg_prerun/` | **保留** | 范式样板，五接缝之"运行前" |
| `layers/construct/` | **迁移 P1** | → `plugins/build/`（templates→`static`，agentinit→`agentinit`；四个脚本重复建图代码归此） |
| `layers/prune/` | **保留原位** | 算法库（被 lg_prerun 挂载）；桩照旧占名 |
| `memory/` + `runtime/injection.py` | **迁移 P3** | memory/ 作算法库保留；注入六步 → `plugins/memory/` 的节点包裹挂载 |
| `layers/processing/` + aggregator | **迁移 P2** | → `plugins/processing/` |
| `trace/` + `train/` | **迁移 P2** | 协议与桩 → `plugins/postrun/`；train 留离线训练协议壳 |
| `plugins/gepa/ + program.py` | **迁移候选（P4 定）** | 以 `method="gepa"` 并入 prerun，或维持独立 optimizer 接缝 |
| `eval/`、`core/` | **保留** | 两线共用地基，不属五模块 |
| `runtime/backends/{hf,openai_api}` | **保留** | 生成原语（被各接缝注入使用） |
| `runtime/backends/langgraph_runtime.py` | **P3 吸收** | 其 MASState/终止语义并入 build+memory 挂载后退役 |
| **旧接缝组**：`pipeline.py`、`plugins/adapters.py`、`plugins/base.py` 三协议、`mock_runtime`、`run_experiment.py`、`examples/01`、`configs/config.yaml`、`configs/plugins/`、旧接缝测试×3 | **删除 P4** | 被五接缝完全取代；替代物（build/processing/postrun + lg 原生 demo/测试）就位后一次性下线 |
| **AutoGen 执行线**：`autogen_runtime.py`、`autogen_injection_client.py`、`configs/runtime/autogen.yaml` | **删除 P4** | 新指令下不再等双后端对拍门槛；随旧接缝退役 |
| `run_mas.py` + `default.yaml` + `aime_*.yaml` | **迁移 P3** | 重写为 lg 原生（build+memory+processing 挂载版）；配置随之收编 |
| `run_langgraph_{baseline,construct,preplug}.py` | **合并 P1** | 三合一 `run_langgraph.py`（`--pre-optimize` 走统一接口）；preplug 的旧接缝演示退场 |
| `scripts/test_generate_team.py` | **迁移 P1** | 改名 `smoke_agentinit_generate.py`（在线冒烟，非单测） |
| `src/autogen/`（1831 文件/56MB vendored） | **删除 P0 ✅** | 零引用死重 |
| 本地垃圾（DOCUMENTS/HybridMem/site/空 config 目录/.DS_Store/egg-info） | **删除 P0 ✅** | 未跟踪杂物 |
| `requirements/benchmark.txt`、`docker/` | **保留** | 评测环境基建 |

## 3. 分阶段执行

- **P0 ✅（本次）**：死重与垃圾清理；pyproject 陈旧排除项清理；本方案文档。
- **P1 build 接缝**：`plugins/build/` + 四脚本建图去重 + 三合一运行脚本 + preplug 退场。
- **P2 processing / postrun 接缝**（小）：两个包裹接缝 + pass@K 承接 + 桩迁移。
- **P3 memory 挂载**（最大）：`attach_memory` 节点包裹语义 + run_mas 的 lg 原生重写 + 双实现对拍回归。
- **P4 旧接缝退役**：删除表中"删除 P4"整组；CLAUDE.md / DESIGN.md / README / make demo 全面改写为五接缝口径。

每阶段收尾：三件套全绿 + 小样本回归对拍（P3 另需与 run_mas 旧实现对拍 predictions 一致）。

## 4. 待签字的接缝设计决策

1. **memory 挂载语义**：推荐 `attach_memory` 重包节点 runnable（对建图方零侵入，与 lg_prerun 同风格）；备选：建图方显式接受 memory 回调（侵入建图契约）。
2. **processing 入口**：推荐吃未编译 `sg`（内部按 K 编译/并发实例化）；备选：吃编译后 app。
3. **GEPA 去向**：并入 prerun（`method="gepa"`，MASProgram 收敛到 AgentSpec 提示视图）或保持独立 optimizer 接缝。
4. **AutoGen 退役提前**（P4 随旧接缝下线，不再等双后端对拍门槛）——请确认。
