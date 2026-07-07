# 组件注册与配置

LycheeMAS 的可插拔性建立在**一个注册表**上：每个算法 = 注册一个类，由 config/CLI 按名字选择，**绝不硬编码、绝不改 `pipeline.py`**。

## 注册表 REGISTRY

```python
from lychee_mas.core.registry import REGISTRY

@REGISTRY.register("aggregator", "my_agg")
class MyAgg: ...

agg = REGISTRY.create("aggregator", "my_agg", k=5)   # 按名实例化
REGISTRY.list("aggregator")                          # -> 已注册的名字
REGISTRY.snapshot()                                   # -> {category: [names]}
```

详见 API：[`lychee_mas.core.registry`](../reference/lychee_mas/core/registry.md)。

## 组件类别（CATEGORIES）

注册表按**类别**组织；新增类别须在 `core/registry.py` 的 `CATEGORIES` 同步登记：

`runtime, model_client, agent_selector, topology_generator, graph_pruner, vocab_adapter, memory_manager, memory_router, aggregator, processor, attributor, credit_assigner, trainer, benchmark`

`make snapshot` 打印每个类别下已注册的实现（含「桩」——占好名字、`NotImplementedError`，让消融矩阵在代码里可见）。

## 触发注册

组件类所在模块被 import 时，装饰器执行注册。子包 `__init__.py` import 各实现模块；`lychee_mas/__init__.py` 再链式 import 各子包 —— 于是 `import lychee_mas` 触发**全部**注册，但**不触发** torch/transformers/autogen（重依赖惰性导入，`make selfcheck` = `HEAVY LOADED: NONE`）。

## 由配置选择

做对照实验只改 config / CLI，不动代码。例如记忆通道消融只换一个变量：

```bash
python scripts/run_mas.py --config configs/aime_latent_c2c.yaml --method none
python scripts/run_mas.py --config configs/aime_latent_c2c.yaml --method nl_only
python scripts/run_mas.py --config configs/aime_latent_c2c.yaml --method latent_only
```

## 新增一个组件 = 六步配方

1. 读对应 `base.py` 协议 → 2. 写实现类 + `@REGISTRY.register(category, name)` → 3. 子包 `__init__.py` import 触发注册 → 4. 加 `configs/<类别>/<name>.yaml` → 5. 加 `tests/test_<name>.py` → 6. `make lint && make test && make selfcheck` + `make demo`。完整说明见 **[开发指南](../contributing.md)**。
