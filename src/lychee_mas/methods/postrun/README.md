# `trace` — 归因 / 信用 + 轨迹落点（顶层包，读侧）

## 功能

「**读**」执行轨迹的一侧：回答「一条轨迹里谁该为成败负责」。三件事：错误归因（`attributor`）、信用分配（`credit_assigner`）、统一落点（`TraceStore`）。经 `post_run_plugin/attribution` 适配器以运行后插件形式挂载（见 `plugins/README.md`）；产出的信用喂给 `train`。

## 接口（`base.py`）

```python
@dataclass
class Attribution:            # 一次归因结果
    agent: str; step: int; is_fault: bool; reason: str; confidence: float; meta: dict

class FailureAttributor(Protocol):
    def attribute(self, trajectory: Trajectory, context: Any = None) -> list[Attribution]: ...

class CreditAssigner(Protocol):
    def credits(self, attributions: list[Attribution], reward: float) -> dict[str, float]: ...
```

## `store.py` — `TraceStore`

```python
TraceStore(path=None)         # path 给定则同步 JSONL 落盘，否则仅内存累积
  .hook(message)              # 接 Runtime.intercept：逐条 Message 写入
  .log_decision(record)       # 决策/归因记录
  .reset(); len(store)
```

纯标准库。这是记忆/处理/训练的数据来源之一。

## 已注册组件

| 类别 | 注册名 |
|---|---|
| `attributor` | `agentracer`（可跑）/ `all_at_once` / `step_by_step` / `binary_search`（桩） |
| `credit_assigner` | `attribution_guided` |

接真实实现：实现协议 + 保留注册名 + 保留原始引用。import 本包触发注册，零重依赖。

`agentracer` 适配官方公开代码的 `mistake_agent` / `mistake_step` / `reason` / `suggested_fix` JSON 协议。由于官方未公开 AgenTracer-8B 权重，构造时必须注入同步 `tracer_llm(prompt) -> str` 回调；第一版可令 `analyze_run(..., credit_assigner=None)` 只做归因。

反事实扩展提供 `replacement → replay → final validation` 流程，并输出最早有效修复步骤、责任 agent、修正依据和模拟最终答案。Who&When 的 replay 明确是 `model_simulated`，不恢复原始网页、终端、文件或 MetaGPT 运行环境；固定 JSON 样例仅用于离线验证 replay 逻辑。
