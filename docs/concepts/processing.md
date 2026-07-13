# 处理层（serial / parallel）

处理层（`lychee_mas.layers.processing`）决定「**跑几次 MAS + 如何得到最终 Answer**」。给定一个 **runner**（`async () -> Trajectory`：把 MAS 在当前 query 上跑一次、产一条轨迹），[`Processor`](../reference/lychee_mas/layers/processing/base.md) 决定调用它几次、以及如何归约成一个 `Answer`。

两个子模块 = 两种模式：

- **serial（串行）**：`processor/serial` —— 只执行**一次**，产 **1 条**轨迹，直接返回其 `final_answer`（无聚合）。
- **parallel（并行）**：`processor/parallel` —— 并发执行 **K 次**，产 **K 条**轨迹，再用 `aggregator` 归约成一个 `Answer`。

```python
# K=1 → serial；K>1 → parallel（默认 self_consistency 多数投票）
proc = REGISTRY.create("processor", "parallel", k=5, aggregator="self_consistency")
res = await proc.run(runner)      # -> ProcessingResult(answer, trajectories)
```

## 归约策略 = `aggregator` 类别

并行处理器的可插拔归约由 [`TrajectoryAggregator`](../reference/lychee_mas/layers/processing/base.md) 承担：

- `aggregator/self_consistency`：纯标准库多数投票（对候选答案按归一化内容取众数）。
- `aggregator/dynamicagg`：动态聚合（在研，桩）。

## 在实验驱动里

`scripts/run_mas.py` 据 `run.samples=K` 自动选 serial / parallel，把「reset→run」封成 runner 交给处理器；pass@1 仍对每条轨迹单独评分，parallel 的聚合答案在配了 `aggregator` 时作为 `vote_acc` 上报。详见 **[快速上手](../quickstart.md)**。

相关 API：[`processing.base`](../reference/lychee_mas/layers/processing/base.md)、[`processing.serial`](../reference/lychee_mas/layers/processing/serial/index.md)、[`processing.parallel`](../reference/lychee_mas/layers/processing/parallel/index.md)。
