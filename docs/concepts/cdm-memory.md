# CDM 双通道记忆（当前主线）

**CDM** = 双通道记忆 + 动态通道选择。用**单一 source**（seed 基础上下文 + 运行中 transcript）按路由器决策物化出某个通道，保证 apples-to-apples（对比时只变「注入/路由」一个变量）。两个对称的可替换接缝：

- **方法接缝（manager）**：记忆怎么存/召回/物化各通道 —— `memory_manager` 类别（`cdm` 默认 / `mem0` / `ama`）。
- **触发接缝（router）**：当前 agent 本轮用哪个通道 —— `memory_router` 类别（`static` / `fixed` / `learned` / `soft_gate`）。

两个**主接口**：[`NLMemory`](../reference/lychee_mas/memory/channels/nl.md)（自然语言）与 [`LatentMemory`](../reference/lychee_mas/memory/channels/latent.md)（隐空间）。manager 只依赖这两个；latent 的子实现（`c2c_channel` / `c2c_projector`）都在 `latent.py` 内部按需调用。

## 一次 agent 发言的数据流

在 `model_client/injection` 的 `create()` 里汇合：

```
① memory.observe(chat)                       更新记忆库（transcript 去重 + 失效 latent 缓存）
② router.decide(RouterInputs) -> RouteDecision(channel)
     └ _enforce_availability：latent 不可用 / 非同模型对 ⇒ 回退 nl
③ memory.recall(decision, query) -> MemoryBundle(NL_Channel?, Latent_Channel?, + 各自 strategy)
     none   → 空
     nl     → NL_Channel = NLMemory.recall(query)（prev_output 原样转发 / simplemem 检索问答）
     latent → LatentMemory.materialize(bundle, source)：soft_token ⇒ prefix；c2c ⇒ projector 栈
④/⑤ 注入 + 生成（据 Latent_strategy 分派）：
     Latent_Channel 为 projector（c2c）→ generate_chat_with_c2c（逐层 KV 融合）
     Latent_Channel 为 prefix（soft_token）→ generate_chat_with_prefix（embedding 拼接）
     否则 → generate_chat（none / nl_only）
⑥ 记账：bump_turn + ctx.log_decision（成本/通道/strategy，可选写 trace.TraceStore）
```

## 两种 latent 物化策略

`memory.latent_strategy` 选，接口对上层一致（都产出 `MemoryBundle.Latent_Channel`）：

- **`soft_token`**（免训练）：源文本 → HF 末层 hidden `(1,T,2560)` → `softmax(h@Eᵀ/τ)@E` 投回**输入嵌入空间** → `segment_mean` 到 P → `(1,P,2560)` prefix。
- **`c2c`**（训练好的 Cache-to-Cache 融合器）：懒加载逐层 `C2CProjector` 栈；把上一个 agent 的 KV 按最长公共 token 块对齐后逐层融进本 agent 生成。训练/评测见 `scripts/train_c2c_projector.py` / `scripts/eval_c2c_aime.py`。

!!! note "两个可扩展的命名集"
    - `LatentMemory.FUSION_STRATEGIES`：走 KV 融合（而非 prefix 拼接）的策略集 —— 新增此类方法在此登记。
    - `NLMemory.STATEFUL_STRATEGIES`：需有状态记忆后端（observe 喂对话 + recall 检索）的策略集，如 `simplemem`。

## 硬约束

- `soft_token` prefix 形状必须 `(1,P,2560)`；单次对话内按 `(len(source), P)` 缓存 prefix。
- `c2c` projector 维度须匹配 `backend.kv_dims()`。
- 两种策略下 latent 都**仅同模型对**可跨 agent 传，否则路由器回退 NL。

相关 API：[`memory.base`](../reference/lychee_mas/memory/base.md)（`MemoryBundle` / `MemoryManager`）、[`memory.managers.DualChannelMemory`](../reference/lychee_mas/memory/managers/DualChannelMemory.md)、[`memory.routing.base`](../reference/lychee_mas/memory/routing/base.md)。
