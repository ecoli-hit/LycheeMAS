# MAD-M² static graph plugin

This plugin runs four GSM8K methods from the [MAD-M² reproduction](https://github.com/Sophia-xq/mad-m2-local-reproduction): CoT, CoT-SC, MAD, and MAD-M²(S). The paper implementation is [MAD-MM](https://github.com/tmlr-group/MAD-MM). Its prompts and answer rules were adapted from commit `f02069add08280b764d059a2f06ca0043aa093e2`; explicit per-call SHA256 seeds and Qwen2.5 single-user chat rendering are local reproducibility choices.

## Install and run one case

Install into an isolated environment: `pip install -e '.[langgraph,config]'`. Start a compatible `llama-server` separately with the chosen GGUF model, then edit [`examples/config_mad_m2.yaml`](../examples/config_mad_m2.yaml) for its URL, variant, question, gold answer and generation settings. The example is **not** a formal benchmark configuration. Run:

```bash
python examples/run_mad_m2.py --config examples/config_mad_m2.yaml
```

The runner calls `build_langgraph(method="mad_m2", backend=LlamaCppBackend(...), variant="mad_m2s", ...)`, compiles the graph and invokes it on a fresh `MadM2State`. Valid variants are `cot`, `cot_sc`, `mad`, and `mad_m2s`. `LlamaCppBackend.complete(...)` sends a raw prompt to `/completion`; it does not launch or stop the server. The backend accepts a distinct 31-bit positive `seed` on every call, along with `temperature`, `top_p`, `max_tokens`, `stop`, `ignore_eos`, and the additional llama.cpp sampling controls used by this reproduction. It returns text, token counts, stop type and the echoed server seed. A missing or different echoed seed fails explicitly.

## Graph and protocol

All graphs are static, linear DAGs with `AgentSpec` node metadata and one terminal `vote` node. CoT has one generation call. CoT-SC has six **sequential** paths and numeric majority voting. MAD has three first-round agent calls, a first-round vote/context node, three second-round agent calls, then a final vote. MAD-M²(S) inserts three subjective mask evaluations and a filter/fallback node between rounds. Each agent is a role in a single loaded model; no parallel model copies are required.

Every graph state records raw responses, extracted answers, per-call prompt and seed, masks, fallback, votes and score. The `<label>NO</label>` decision alone removes a first-round context; `YES`, `NOT SURE` and unparseable labels keep it. If all three contexts are removed, the filter records `empty_context_fallback` and second-round nodes use the base CoT prompt. Voting extracts the last numeric substring as `float`, uses first appearance on a tie, and marks consensus at at least half of the paths. GSM8K scores within absolute tolerance `1e-6`.

The protocol uses an explicit Qwen2.5 single-user chat rendering. Switching to `/v1/chat/completions` or the framework's `OpenAICompatibleBackend` may change prompt bytes and available sampling fields, so it is outside this plugin's parity guarantee. The graph does not call the framework's `attach_memory()` seam, which is currently unimplemented; subjective masking lives in explicit nodes. The graph also locks prompts in Python helpers and is not intended for prompt-mutating prerun optimizers.

## Testing and scope

`PYTHONPATH=src pytest -q tests/test_mad_m2_plugin.py` runs offline with a scripted completion backend and no model download. To compare the old runner directly, set `MAD_M2_LEGACY_ROOT` to a local checkout of the independent reproduction; the optional tests compare one fixed question under seed 41 for CoT and MAD-M²(S), including prompt bytes, per-call seeds, raw text, extraction and score. The regular offline tests remain standalone for upstream CI.

For an opt-in real-model parity run, also set `MAD_M2_REAL_URL` to an **already-running** llama-server URL. The live test invokes both runners sequentially with the same 256-token output cap and compares every call. This cap is a local test setting, separate from any formal experiment configuration. The test does not start or stop the server.

This plugin does not select a dataset, manage the model service, persist benchmark results or replace the original reproduction's checkpoint/resume machinery. Users running a full benchmark must supply those pieces and lock data/model hashes separately. Existing LycheeMAS methods and backends are unchanged.
