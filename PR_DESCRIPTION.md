# Add MAD-M² static graph and llama.cpp completion backend

## Background

This PR brings four methods from our [MAD-M² local reproduction](https://github.com/Sophia-xq/mad-m2-local-reproduction) into LycheeMAS: CoT, CoT-SC, MAD and MAD-M²(S). The paper implementation is [MAD-MM](https://github.com/tmlr-group/MAD-MM), pinned to commit `f02069add08280b764d059a2f06ca0043aa093e2` for the adapted GSM8K prompts and answer rules. Our explicit Qwen2.5 single-user prompt rendering and per-call SHA256 seed schedule are local reproducibility choices, documented as such.

## Changes

- Add `LlamaCppBackend`, a standard-library `/completion` client with explicit per-call seed and sampler fields. It returns generated text, token counts, stop type and the echoed server seed.
- Register `graph_builder/mad_m2`. Its `variant` selects `cot`, `cot_sc`, `mad` or `mad_m2s`; all calls are static sequential `StateGraph` nodes with `AgentSpec` metadata and one terminal vote node.
- Isolate protocol helpers for prompt rendering, answer extraction, numeric voting, subjective masks, all-masked fallback and GSM8K scoring.
- Add an offline example, one-case YAML configuration, tests and usage documentation.

## A100 benchmark

We also exercised the protocol in a five-seed GSM8K benchmark on an NVIDIA A100-PCIE-40GB. The run used Qwen2.5-7B-Instruct Q4_K_M through the llama.cpp CUDA backend, with 100 fixed questions for each of seeds 41–45. These results validate the integrated algorithm and execution path under sustained load; they are a local quantized-model reproduction rather than a strict reproduction of the paper's original inference stack.

| Method | Mean accuracy |
|---|---:|
| CoT | 63.6% |
| CoT-SC | 83.8% |
| MAD | 87.8% |
| MAD-M²(S) | 88.2% |

MAD-M²(S) was 0.4 percentage points above MAD overall. A paired two-sided exact McNemar test gave `p=0.86794`, so this small difference is not statistically significant and is consistent with expected run-to-run variation in this quantized-model setting. We do not interpret the benchmark as evidence that subjective masking always improves accuracy.

The full benchmark completed 11,000 model calls with zero runner failures. The explicit `empty_context_fallback` path was triggered 32 times when all three subjective contexts were masked; every event was logged and completed using the base-question CoT prompt. Peak GPU memory was 5,792 MiB (about 5.7 GiB), and total wall-clock time was approximately 4.5 hours.

## Tests

- Offline scripted test suite: `PYTHONPATH=src pytest -q` — **191 passed, 7 skipped**. The skips are optional external-checkout and live-server parity checks.
- Legacy parity test: `MAD_M2_LEGACY_ROOT=<checkout> PYTHONPATH=src pytest -q tests/test_mad_m2_plugin.py` — fixed GSM8K case 600, seed 41, CoT and MAD-M²(S), compares prompt bytes, seeds, raw responses, parsing and scores using scripted model responses.
- Live llama-server parity: **2 passed in 168.82 s**, using local CPU Qwen2.5-7B-Instruct Q4_K_M, fixed GSM8K case 600 and seed 41. The old runner and new graph agreed on every prompt byte, per-call seed, raw model response, extracted answer and final score for CoT and MAD-M²(S). Both sides used a 256-token output cap solely for this local interface test; it does not replace the formal experiment's 1024-token setting. The server was stopped after the test.
- `ruff check src`, offline five-seam demo and zero-heavy-import selfcheck passed.

## Compatibility and limits

Existing methods, backend APIs and evaluation scorers are unchanged. The plugin uses an injected backend and does not modify LycheeMAS's unfinished `attach_memory()` seam. Its subjective memory masking is represented by explicit evaluation/filter nodes. It does not launch a server, download models or datasets, persist formal results, or replace the original reproduction's checkpoint/resume workflow. Prompt-mutating graph optimizers are outside this protocol-locked plugin's equivalence guarantee.
