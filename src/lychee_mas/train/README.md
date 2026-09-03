# `train` — 训练（顶层包，写侧）

## 功能

「**写**」参数/提示的一侧：`Trainer` 把信用（来自 `trace` 的 CreditAssigner）与轨迹喂给优化过程，产出新参数/提示（RL 路线）。提示级离线优化走插件系统的 `Optimizer`（如 `optimizer/gepa`，见 `plugins/README.md`）。

## 接口（`base.py`）

```python
class Trainer(Protocol):
    def credits(self, attrs: list[Attribution], reward: float) -> dict[str, float]: ...
    def train(self, generator, policies, mem_policies, traces, credits) -> Any: ...
```

（`Attribution` 仅作类型注解引用 trace 包，运行期无跨包硬依赖。）

## 已注册组件

| 注册名 | 状态 |
|---|---|
| —（RL 训练器待接） | MASPO 已按其本义（联合提示优化）落地为 `pre_run_optimizer/maspo`（`plugins/prerun/`），原 `trainer/maspo` 桩迁出 |

## 约定

- RL 库（TRL / veRL / OpenRLHF）只在 `trainer/*` 具体实现里依赖，放 optional extra，不污染基座。
- import 本包触发注册，零重依赖。
