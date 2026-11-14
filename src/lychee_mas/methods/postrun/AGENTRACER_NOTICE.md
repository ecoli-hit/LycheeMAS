# AgenTracer adaptation notice

This adapter follows the public input/output contract from:

- Paper: *AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?*
- Code: <https://github.com/bingreeky/AgenTracer>
- The external repository was consulted during development but is not vendored.

The public repository currently releases a MetaGPT data-curation pipeline and
sample TracerTraj data, but not the AgenTracer-8B weights. LycheeMAS therefore
uses an injected model callback and does not vendor the MetaGPT repository.

The adapted fields are `mistake_step`, `mistake_agent`, `reason`, and
`suggested_fix`. No source file is copied verbatim.

LycheeMAS does not redistribute AgenTracer or MetaGPT source files. This
adapter only implements the public input/output contract, so the external
repository's source-code license is not copied into this repository.

## LycheeMAS runner

The integration runner accepts the released Who&When parquet files:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agentracer_attribution.py \
  --input /path/to/Algorithm-Generated.parquet \
  --output runs/agentracer/algorithm.jsonl \
  --model-path /path/to/model
```

`--smoke` uses gold labels only to validate the adapter and must not be used as
a model-quality result. The runner writes `agent_exact`, `step_exact`,
`joint_exact`, and mean `step_distance` to the companion metrics JSON.

## Counterfactual replay boundary

The counterfactual extension proposes a local replacement, simulates downstream
steps, and validates the simulated final answer. Who&When replay is explicitly
`model_simulated`: it does not restore the original browser, terminal,
filesystem, or MetaGPT runtime environment. The fixed JSON fixture is only for
offline replay validation.

## Validation

The current contribution was checked in the `LycheeMAS` conda environment:

```text
make lint: passed
make test: 220 passed, 1 skipped
make selfcheck: passed
make demo: passed
counterfactual smoke: joint_exact=1.0
```
