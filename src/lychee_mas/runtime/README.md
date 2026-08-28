# `runtime` — 运行时抽象、共享注入引擎与执行后端

## 功能

把「执行一个 MAS」抽象成最小 `Runtime` 协议，隔离底层执行引擎（autogen / langgraph 双后端，可替换）。业务层只依赖本包协议，**绝不**直接 import 引擎库。每次 agent 发言的「记忆注入六步」统一实现在 `injection.py`，两个真实后端共用。

## 接口

### `base.py` — 运行时协议

```python
class Runtime(Protocol):
    async def run(self, team: MASGraph, query: TaskQuery) -> Trajectory: ...
    def intercept(self, hook: Callable[[Message], None]) -> None: ...   # 消息级拦截
```

- `MASGraph`：图 G 的轻量容器——`nodes: list[AgentSpec]`、`edges`（role→下游，空则按声明顺序链）、`rounds`、`meta`；`order()`、`.names`。`MASTeam` 为其别名。
- `BaseRuntime`：可选基类，实现 `intercept` 样板（hook 列表 + `_emit(message)` 逐消息回调）。intercept 是轨迹落盘与记忆抽取的数据来源。

### `injection.py` — 共享注入引擎（记忆六步的唯一实现）

```python
run_injection_step(backend, ctx, InjectionRequest) -> InjectionResult
# ① memory.observe → ② router.decide → ③ memory.recall → ④ system 段注入（保序）
# → ⑤ 按 bundle 分支生成（KV 融合 > prefix 拼接 > 普通）→ ⑥ bump_turn/log_decision/log_span
```

- `InjectionRequest`：role / chat / max_new_tokens / extra_system_messages / tool 参数 / `postprocess` / `parse_tool_calls` 回调（引擎专属逻辑以回调注入，引擎本身运行时无关）。
- `InjectionResult`：text / raw_text / gen / decision / bundle / turn / sender / token 计数 / tool_calls。
- 行为契约（回归对拍依赖）：system 段顺序 = `[原始 system…, extra…, 记忆文本, 对话…]`；decision 记录键与 span 字段固定。

### `spans.py`

`JsonlSpanLogger`：增量 JSONL 运行事件日志——`set_case(case_id, sample_index)`、`log(span_type, **fields) -> span_id`；`exception_record(exc)` 把异常转成可落盘 dict。

### `backends/` — 执行后端与 model client

| 注册名/文件 | 职责 |
|---|---|
| `runtime/mock`（`mock_runtime.py`） | 确定性离线后端（测试/CI/示例默认，纯标准库） |
| `runtime/autogen`（`autogen_runtime.py`） | AutoGen 群聊执行（含工具型团队预设：magentic_one / coder_executor 等） |
| `runtime/langgraph`（`langgraph_runtime.py`） | LangGraph StateGraph 执行：每 AgentSpec 一节点、条件边控终止（APPROVE / max_turns）；仅支持纯文本团队，工具型团队显式报错 |
| `model_client/injection`（`autogen_injection_client.py`） | AutoGen 的 ChatCompletionClient 适配：类型转换 + 工具调用格式 + JSON 修复，核心走共享注入引擎 |
| `model_client/vllm`（`vllm_client.py`） | 桩 |
| `hf_backend.py` / `openai_api_backend.py` | 生成后端（不注册，被运行时组合）：`generate_chat` / `encode_hidden` / `generate_chat_with_prefix` / KV 原语；API 后端仅文本生成 |
| `_common.py` | 两后端共享纯函数：`content_to_text` / `jsonable` / `extract_final_answer` |

## 约定

- **`autogen_*` 只允许出现在 `backends/autogen_*.py`；`langgraph` 只允许出现在 `langgraph_runtime.py` 的 `_build_app` 内**；均惰性导入。import 本包只触发注册，零重依赖。
- 新增执行引擎 = 新增 `backends/<engine>_runtime.py` 实现 `Runtime` 协议（节点内调 `run_injection_step`）并注册，业务层零改动。
