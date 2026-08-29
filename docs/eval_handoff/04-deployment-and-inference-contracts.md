# 4.6 Deployment、模型网关与生成参数

[上一部分：跨框架 Runtime](04-framework-runtime-contracts.md) · [返回第 4 章索引](04-implementation-contracts.md) · [下一部分：Experiment 与 Scheduler](04-experiment-and-scheduler-contracts.md)

### 4.6.1 三种后端

| kind | 运行形式 | 主要特点 |
|---|---|---|
| `local_hf` | 实验进程直接加载 Transformers | 支持 hidden states/latent；单进程并发和吞吐较弱 |
| `vllm` | 独立 OpenAI-compatible 服务 | 连续批处理、prefix cache、服务级并发和 Prometheus 指标 |
| `api` | 云端 OpenAI-compatible 服务 | 无本地 GPU；受 provider quota、限流和账单约束 |

这里的 `vllm` 是 **Deployment kind**，不是第三种 completion 协议。vLLM 与云 API 的模型请求都进入 `runtime/adapters/inference/openai_compatible.py`；区别位于 Deployment 的服务生命周期、endpoint、健康探测、Prometheus telemetry 和本地 GPU 成本。旧的独立 `runtime/adapters/inference/vllm.py` 未接入正式链路且容易制造“两套 vLLM client”的错觉，现已删除。

### 4.6.2 参数归属

| 层级 | 典型参数 |
|---|---|
| DeploymentSpec | model path/API source、TP、DP、max model len、gpu memory utilization、max num seqs、dtype、parser、服务端缓存 |
| DeploymentInstance | endpoint、进程、GPU、健康状态、PricingInstance |
| TeamInstance Node binding | 某 Node 使用哪个 DeploymentInstance；Node 级 thinking 和生成覆盖 |
| ExperimentSpec | cases、trials、case concurrency、turn/round/model-call limits、默认生成参数、工具、网络、EventLog |
| ExperimentInstance | benchmark/team 实例、队列优先级、下一执行段 case 数、累计完成上限、run dir |

所有 Studio 可编辑字段必须进入 API payload、规范化配置和最终命令；后端支持但未展示的字段应进入高级设置；既不展示也不执行的字段必须删除。ExperimentSpec 注册表现在拒绝未知 runtime 字段，防止“写进 JSON 但运行不生效”。

### 4.6.3 长度与采样

- `max_model_len`：部署能够容纳的输入加输出总上下文，仅在模型服务启动时设置。
- `max_input_tokens`：本次调用允许保留的输入上限。
- `max_new_tokens`：本次调用允许生成的最大输出，不代表一定生成这么长。
- token budget 层按模型窗口、输入、safety margin、最小输出和 thinking/final reserve 计算实际输出上限。
- `do_sample=false` 使用后端的确定性路径；`temperature/top_p/top_k/min_p` 只有采样开启时才共同决定候选分布。
- seed 由 run seed、case、trial 和 attempt 派生，确保并发顺序变化不改变某个 Trial 的随机流。

### 4.6.4 gpu_memory_utilization

`gpu_memory_utilization` 是 vLLM 可占用显存比例，不是 GPU 算力利用率。提高可增加 KV cache、支持更长上下文或更多并发序列，但会压缩 CUDA graph、临时 workspace、其他进程和碎片化余量，增加 OOM 风险。

建议：

- 共享服务器可以从 `0.85` 开始；独占 GPU 且模型加载后无 preemption/OOM 时再验证 `0.88`、`0.90`。
- 冷启动和最小请求只证明部署可用，不证明正式负载稳定。
- 是否提高由 KV cache 峰值、等待队列和 preemption 决定，而不是看到显存“没占满”就提高。

### 4.6.5 max_num_seqs

`max_num_seqs` 是每个 DP replica 可同时处于调度批次中的序列硬上限。开大可以让服务有更大的连续批处理空间，但可能提高 KV cache、TTFT、排队和尾延迟；开小则让上层运行池即使还有资源也无法提高吞吐。

上层 Scheduler 不把 vLLM 序列容量误写成 Trial 槽位，而是在各 ExperimentInstance 并发约束内逐个接纳 Trial，并以 vLLM、GPU、API 各自的压力安全门决定是否继续。`gpu_memory_utilization` 和 `max_num_seqs` 只能通过记录完整配置的独立负载实验更新；至少比较总 Trial 吞吐、Queue/TTFT P50/P95/P99、KV 峰值、preemption、错误率和单 Trial wall time。

### 4.6.6 自适应请求超时

`request_timeout` 是故障保护，不是限制模型正常思考时间的实验预算。`adaptive` 模式根据 `max_new_tokens` 估算最坏生成时间，并限制在 `minimum_s` 与 `maximum_s` 之间。

规划速度取以下可用值中的最小值：冷启动基线速度、配置的观测速度上限、历史生成速度 EWMA。因此：

- 历史调用比基线快时，不能缩短保守超时；共享 vLLM 在高并发下可能突然变慢。
- 历史调用比基线慢时，可以延长后续超时。
- 只有达到 `maximum_s` 或真正超过规划时间才终止，不把 API timeout 当作正常停止条件。

规划所使用的观测速度和最终 timeout 必须写入不可变运行快照，不能只保存当前服务的 EWMA。

### 4.6.7 前后端参数一致性

这里的“全部参数”指 LycheeMAS 合同正式声明并校验的参数，不是把 vLLM、Transformers 和每家云 API 的所有上游开关无筛选地搬进 Studio。

| 参数组 | Studio 入口 | 后端落点 |
|---|---|---|
| HF Python、GPU、模型资源 | DeploymentSpec/Instance | Local HF backend 初始化 |
| vLLM host/port、TP/DP、max model len、GPU memory、max num seqs、dtype | DeploymentSpec | 托管 vLLM 启动命令 |
| reasoning/tool parser、prefix cache、eager、per-request metrics、prompt token details | vLLM 高级设置 | 对应 vLLM CLI 参数；不支持时 auto 降级或 enabled 拒绝部署 |
| endpoint、鉴权引用、client concurrency、请求间隔 | API/Deployment | OpenAI-compatible client 和共享 limiter |
| timeout policy、SDK retries、extra request body | DeploymentSpec | 每次 API/vLLM 请求 |
| thinking、输入/输出与保留量、sampling、Top-K/Min-P、penalty | TeamInstance Node binding 或 ExperimentSpec 默认值 | ModelGateway 按 Node 合并后传给真实模型调用 |
| case/turn/round/model-call/tool/network/EventLog | ExperimentSpec | `run_mas.py` 和 RuntimeAdapter |

保存时会经过前端 payload、API normalizer、配置快照和最终命令四层。ExperimentSpec 未知 runtime 字段会直接拒绝；DeploymentSpec 中只有已声明字段可生成服务命令。`extra request body` 是 OpenAI-compatible provider 扩展的受控出口，但不能替代服务启动参数。

### 4.6.8 Pricing 与成本绑定

PricingSpec 保存完整价格合同，PricingInstance 只保存 `id`、`pricing_spec_id`、Spec 指纹、创建时间和来源。价格字段不能塞到 Instance 临时覆盖；Spec 变化后旧 Instance 失效，需要重新实例化。

| basis | billing mode | 必需 rate | 适用部署 | 含义 |
|---|---|---|---|---|
| `token_usage` | `provider_token_usage` | input、output；可选 cached/reasoning/thinking | API、外部 vLLM、本地部署的 API 等价价格 | 按 provider usage 计费 |
| `allocated_gpu_time` | `on_demand_gpu_hour` | gpu_hour | 本地 HF、托管 vLLM | 按分配 GPU 数 × 服务/Run 墙钟 |
| `allocated_gpu_time` | `monthly_node_amortized` | gpu_hour | 本地 HF、托管 vLLM | 整机月价按 GPU 份额和月小时摊销 |
| `allocated_gpu_time` | `internal_gpu_hour` | gpu_hour | 本地 HF、托管 vLLM | 内部参考单价 |

DeploymentSpec 声明所需的 actual PricingSpec，并可为本地 HF/托管 vLLM 声明 API-equivalent PricingSpec；DeploymentInstance 实例化时必须绑定相匹配的 PricingInstance。API 与外部 vLLM 的实际价格本身就是 token pricing，不再重复绑定另一个 API-equivalent 规则。

`metrics.json.costing` 区分 `actual_cost`、`api_equivalent_cost`、同币种 comparison 和 `failed_model_calls_without_usage`。provider 返回 cached input usage 时按缓存价格计算；只提供服务级 cache 指标时不能伪造到单次调用。Local HF 暂不提供 per-request cached token，文档与 Observation 均保留 missing evidence。

### 4.6.9 CDM 通信消融与后端边界

`method` 是整个 Run 的通信/记忆消融条件，不是 Benchmark、Team topology 或 Runtime framework。每次是否真正注入由 RoutingContext、router、Relation 和 backend 能力共同决定：

| method | NL memory | latent prefix | 后端边界 |
|---|---:|---:|---|
| `none` | 否 | 否 | HF、vLLM、API 均可 |
| `nl_only` | 是 | 否 | HF、vLLM、API 均可 |
| `latent_only` | 否 | 是 | 当前只支持 Local HF hidden-state 路径 |
| `both` | 是 | 是 | 当前只支持 Local HF hidden-state 路径 |

vLLM/API 可以正常运行同一 TeamSpec 和工具链，但当前不能承载 latent prefix；这不表示它们“不支持 MAS”或“不支持工具”。`memory_channel`、`routing_reason` 和实际注入长度写入 model-call Event，Study 必须按 method 分组，不能把 latent 与纯文本运行混在同一系统配置中。
