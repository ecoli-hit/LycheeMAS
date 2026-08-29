# 快速上手

## 离线端到端 demo（零重依赖）

```bash
make demo
# 等价：PYTHONPATH=src python examples/01_static_chain_e2e.py
```

用 `runtime=mock` + 一个静态团队 + 一个 `TaskQuery` 跑通 `Orchestrator`，打印最终答案、`TraceStore` 收到的消息条数与 `REGISTRY.snapshot()`。**无需 GPU / API key。**

## 看现有组件 / 校验零重依赖

```bash
make snapshot     # 打印 REGISTRY.snapshot()：每个类别下已注册的实现名
make selfcheck    # 应打印 HEAVY LOADED: NONE（注册路径不触发 torch/autogen）
make test         # 离线单元测试（mock runtime，无需 GPU/API）
```

## 实验入口（CLI + 落盘，默认 `runtime=mock` 可离线）

```bash
PYTHONPATH=src python scripts/run_experiment.py \
    --runtime mock --team default --aggregator self_consistency \
    --questions "2 plus 2 is 4" "answer is 7"
```

## 带 CDM 记忆通道的真实 AIME 实验（需 `.[all]` + GPU）

`scripts/run_mas.py` 把 backend + CDM 记忆 + 固定通道路由 + 处理层接成端到端实验。四种记忆通道消融：`none | nl_only | latent_only | both`；latent 走 `memory.latent_strategy`（`soft_token` / `c2c`）；NL 走 `memory.nl_strategy`（`prev_output` / `simplemem`）。

```bash
export LYCHEE_HF_MODEL=/path/to/Qwen3-4B     # HF 后端模型路径（不写进代码）
CDM_DATA_ROOT=/path/to/Data/raw CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_mas.py --config configs/aime_latent_c2c.yaml
```

**独立执行次数由 Trial 数决定**：同一个 Case 可以有 K 个 Trial，每个 Trial 有独立的 `trial_index` 与派生 seed；并发只改变调度，不改变 Trial 身份。运行时持续写唯一事实源 `events/run_events*.jsonl` 和配置快照，不在推理阶段读取 gold。完成后使用 `scripts/analyze_benchmark_run.py <run_dir>` 调用 Benchmark scorer，追加 Evaluation Event，并派生 Result Projection、Execution Trace、指标和 Evidence。

## 下一步

- 想懂目录职责与数据流 → **[架构与设计](DEVELOPMENT.md)**
- 想加一个组件（六步配方）→ **[开发指南](contributing.md)** 与 **[组件开发](dev/README.md)**
- 想查某个类/函数 → **[API Reference](reference/lychee_mas/index.md)**
