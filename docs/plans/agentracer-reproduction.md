# AgenTracer 复现证据（公开数据上的配对评估）

本文档记录本仓库 AgenTracer 适配层之外的**复现实验**：在公开 TracerTraj 数据上，
对"多粒度 RL 是否提升失败归因"这一主张做的可复核评估。

> 结论先行：**在公开数据与单卡条件下，没有观察到稳定提升**。两个独立 run、7 个 checkpoint，
> 平均效应分别为 +0.67pp 与 −0.13pp，置信区间均跨 0。
> 进一步剔除单 agent 数据重训（下方第 5 节）也没有把指标推过平凡基线。
> 因此本仓库只声称"适配层可用"，不声称"复现了论文指标"。

## 1. 复现边界（为什么不能对标论文）

| 项 | 论文 | 本仓库 |
|---|---|---|
| 基座权重 | AgenTracer-8B | **未公开**，改用 Qwen3-8B + LoRA(r16, α32) |
| 训练数据 | TracerTraj（论文版） | 公开 `data-v1.0.0`（作者自述与论文版不一致、含额外样本） |
| 推理协议 | `<think>…</think><answer>agent \| step</answer>` | 同左；与本仓库适配层使用的 **JSON 协议不同** |
| 方法版本 | v2（NeurIPS 2026）为工具增强多跳 agent | 本仓库适配 v1 的单次直出协议 |

因此本实验只能做"同数据自比"，不能用来验证论文的 Who&When 数字。

## 2. 评估设置

- 数据：`validation_canonical.jsonl`，790 条（agentic 600 / coding 127 / math 63）
- 解码：greedy（temperature 0），prompt ≤24576，生成 ≤16384
- 打分：论文口径的四项指标（parse / agent_exact / step_exact / joint_exact），配对按 `(task_id, source_benchmark)`
- 噪声底：同一策略在同一批 200 条上重跑一次（A/A），用于给出可比噪声

## 3. 主结果（790 条，配对）

| 指标 | 裸 Qwen3-8B | GRPO 12 步 | 差值 | 95% CI（配对 bootstrap） |
|---|---:|---:|---:|---|
| parse | 89.9% | 87.5% | −2.4pp | [−4.68, −0.38] |
| agent_exact | 77.8% | 76.8% | −1.0pp | [−3.16, +1.14] |
| **step_exact** | **37.3%** | **40.4%** | **+3.04pp** | **[+0.63, +5.44]** |
| **joint_exact** | **34.9%** | **37.7%** | **+2.78pp** | **[+0.38, +5.32]** |
| 截断率 | 4.6% | 6.6% | +2.0pp | — |

分 subset（step_exact）：agentic 42.5% → 46.5%（+4.0pp, p=0.011）；coding 10.2% → 9.4%（−0.8pp, p=1.0）；
math 42.9% → 44.4%（+1.6pp, p=1.0）。

## 4. 剂量-反应（两个独立 run，同一批 200 条，逐条配对）

**Run A（12 步，`grpo_trend_4gpu_full`）**

| policy | step_exact | joint_exact | 相对裸模型 | p |
|---|---:|---:|---:|---:|
| 裸模型 | 44.5% | 42.0% | — | — |
| step-5 | 45.5% | 43.0% | +1.0pp | 0.864 |
| step-10 | 43.0% | 40.0% | −1.5pp | 0.728 |
| step-12 | 47.0% | 44.0% | +2.5pp | 0.473 |

三个 checkpoint 平均效应：step **+0.67pp**（95% CI [−3.67, +5.33]），joint +0.33pp（CI [−4.00, +4.67]）。

**Run B（20 updates，`repro_full_20261111`，全量数据重训）**

| policy | step_exact | joint_exact | parse | 相对裸模型 | p |
|---|---:|---:|---:|---:|---:|
| 裸模型 | 44.5% | 42.0% | 92.0% | — | — |
| step-5 | 43.5% | 41.5% | 91.5% | −1.0pp | 0.851 |
| step-10 | 46.5% | 43.0% | 90.5% | +2.0pp | 0.585 |
| step-15 | 43.0% | 38.5% | 92.0% | −1.5pp | 0.743 |
| step-20 | 44.5% | 42.0% | 92.0% | +0.0pp | 1.000 |

四个 checkpoint 平均效应：step **−0.13pp**（95% CI [−4.50, +4.13]），joint −0.75pp（CI [−5.00, +3.50]）。
**起点与终点的 step 完全相同（44.5% → 44.5%）**，中间步骤在 ±2pp 内摆动。

相邻 checkpoint 的两两比较也都不显著：s5→s10 +3.0pp（p=0.345）、s10→s15 −3.5pp（p=0.296）、
s15→s20 +1.5pp（p=0.728）。checkpoint 间跨度 3.5pp，**不单调**。

两个 run 合起来共 7 个 checkpoint，没有一个点显著高于裸模型。

## 5. 对照实验：剔除单 agent 数据后重训

### 5.1 动机

公开训练集 3173 条中，**1500 条（47.3%）来自单 agent 的 `agentic` 子集**，
其 `mistake_agent` 只有一个取值。数据包 README 明确说明
"Some agentic trajectories use a generic `assistant` label rather than a named agent"。

在这部分数据上，`r_agent` 只剩两种取值（是否为唯一允许的那个名字），拿 1 的条件是**复述名字**
而不是"判断谁该负责"。一旦模型学会复述，`r_agent` 在组内成为常数，而组内归一化会把常数消掉：

```
R_k = c + λ·r_step_k  →  A_k = (r_step_k − mean_step) / std_step
```

即 λ 被完全约掉。于是 λ 的消融（多粒度 vs 单粒度）有一半样本从一开始就没有区分力。

### 5.2 设置

- 训练集：剔除 `agentic` 后剩 **1673 条**（coding 1141 + math 532，其中 1638 条轨迹确实有 >1 个 sender）
- 其余配置与 Run B 完全一致（λ=0.5、σ=1、group 8、lr 1e-6、HF rollout、20 updates、同一 base 模型）
- 评测集：`validation_multiagent.jsonl`，**190 条**（coding 127 + math 63），与训练同域但样本不重叠
- 另加一个同 step 对照：未过滤版 step-20 在同一 190 条上评测

### 5.3 结果（190 条，逐条配对）

| policy | parse | agent | step | joint | 截断 |
|---|---:|---:|---:|---:|---:|
| 常数预测器（永远答 step 4） | — | — | **22.6%** | — | — |
| 裸 Qwen3-8B | 82.6% | 58.9% | 21.1% | 19.5% | 25 |
| 未过滤 step-20 | 77.4% | 54.7% | 21.1% | 19.5% | 34 |
| 过滤 step-10 | 74.7% | 56.3% | 21.1% | 20.5% | 37 |
| **过滤 step-20** | 76.8% | 57.9% | **22.6%** | 21.6% | 33 |

配对检验：

| 比较 | step | joint | agent |
|---|---:|---:|---:|
| 过滤 step-20 vs 裸模型 | +1.6pp（p=0.607） | +2.1pp（p=0.481） | −1.1pp（p=0.875） |
| **过滤 step-20 vs 未过滤 step-20** | **+1.6pp（p=0.549）** | +2.1pp（p=0.388） | +3.2pp（p=0.392） |

分 subset（step）：coding 裸模型 10.2% → 过滤 11.0%；math 42.9% → 46.0%（均不显著）。

### 5.4 解读

1. **稀释假设未获支持。** 点估计方向与假设一致（step +1.6pp、agent +3.2pp），但 n=190、
   不一致对有 4 vs 7 对，统计上讲不出任何东西；p 值全部在 0.39–0.88。
2. **过滤后最好的点也只是与平凡基线打平。** 22.6% 恰好等于"永远答众数 step"的 22.6%，
   仍然没有任何证据表明模型在做真正的定位而非猜众数。
3. **代价确定存在。** 所有训练变体的 parse 都低于裸模型（82.6% → 76.8%），
   截断从 25 条涨到 33–37 条。训练换来的唯一稳定效果是"更不会收尾"。
4. **这个实验的检验力不足。** 按 190 条与观察到的 3–4 对不一致对，只能检出 10pp 量级以上的差异。
   因此正确表述是"没有证据支持"，而不是"证明无效"。要真正判定需要更大的评测集与更多 seed。

## 6. 噪声底（A/A）

同一策略（裸模型）、同一批 200 条、同一 8 分片并行重跑：step 44.5% → 44.5%（+0.0pp），
逐条翻转 13 对 13 完全对称。说明批处理带来的是无偏抖动，测量噪声约 0.5pp，
**不足以解释 checkpoint 间 3.5–4pp 的差异**——策略确实在变，但没有朝一个方向走。

## 7. 指标本身的缺陷（重要）

| 现象 | 实测 |
|---|---|
| agentic 子集只有一个 agent 名（`assistant`） | 该子集 600/790，`agent_exact` 恒为 100% |
| 金标 step 高度集中 | 众数占比：agentic 22.5%、coding 28.7%、math 43.0% |
| 常数预测器（永远答众数 step） | **joint 23.8%**（全量）；多 agent 190 条上 22.6% |
| 模型 vs 常数预测器 | agentic +19.2pp；**coding −22.9pp；math −6.3pp** |

即：在当前指标下，coding 与 math 两个子集的模型表现**低于"永远答众数 step"**。
任何在该数据上报告的提升，都必须同时给出相对常数基线的增量，否则会被基率吞掉。

## 8. 与适配层相关的三个注意事项

1. **prompt 中含 `Ground Truth` 字段**（`agentracer.build_prompt`）。评测时必须置空，
   否则等于把答案喂给模型，指标失效。
2. **步号基准**：公开 TracerTraj 中 agentic 与 math 的 `mistake_step` 从 1 开始，
   而 `format_trajectory` 按零基枚举，存在整体偏一位的风险。
3. **协议不同**：适配层解析 JSON（`mistake_step/mistake_agent/reason/suggested_fix`），
   本实验的模型输出 `<answer>agent | step</answer>`，两者不能混报。

## 9. 复现方式

```bash
# 1) 用训练侧脚本产出预测（略），得到 predictions/*.jsonl
# 2) 重算配对表
python scripts/reproduction/agentracer_paired_eval.py \
  --records  /path/validation_canonical.jsonl \
  --baseline predictions/baseline.predictions.jsonl \
  --adapter  predictions/trained.predictions.jsonl \
  --extra    s5=predictions/s5.predictions.jsonl \
             s10=predictions/s10.predictions.jsonl \
             s12=predictions/trained.predictions.jsonl

# 3) 重算金标分布与常数基线
python scripts/reproduction/agentracer_base_rates.py --records /path/validation_canonical.jsonl

# 4) 重算多 checkpoint 曲线（Run B 的四个点）
python scripts/reproduction/agentracer_checkpoint_curve.py \
  --records /path/validation_canonical.jsonl \
  --baseline predictions/baseline.predictions.jsonl \
  --checkpoints s5=predictions/full20_s5.predictions.jsonl \
              s10=predictions/full20_s10.predictions.jsonl \
              s15=predictions/full20_s15.predictions.jsonl \
              s20=predictions/full20_s20.predictions.jsonl

# 5) 多 agent 子集上的对照（剔除单 agent 数据重训）
python scripts/reproduction/agentracer_paired_eval.py \
  --records  /path/validation_multiagent.jsonl \
  --baseline predictions/baseline.predictions.jsonl \
  --adapter  predictions/multi_f_s20.predictions.jsonl \
  --extra    unfiltered_s20=predictions/multi_u_s20.predictions.jsonl \
             filtered_s10=predictions/multi_f_s10.predictions.jsonl
```

完整产物（逐条预测、重算表格、sha256 清单）保存在实验机
`/data3/xhy/codes/work/AgenTracer/runs/reproduction_bundle/`，
因 `*.jsonl` 被 `.gitignore` 排除，未纳入仓库。

## 10. 结论

- 适配层可用：6 个契约测试通过，归因链路可在框架内正常运行。
- 方法主张未获支持：两个 run 共 7 个 checkpoint，平均效应 +0.67pp 与 −0.13pp，置信区间均跨 0；
  20 updates 那轮的起点与终点完全相同（44.5% → 44.5%）。
- **剔除单 agent 数据重训也没有改变结论**：多 agent 子集上 22.6%，仅与平凡基线打平，
  与未过滤版的差异 +1.6pp（p=0.549）。
- 指标需要修正后才能用于任何"复现成功"的声明（见第 7 节）。
