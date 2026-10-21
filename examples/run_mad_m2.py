"""Run one demonstration question against an already-running llama-server."""

from __future__ import annotations

import argparse
import json

import yaml
from lychee_mas.backends.llama_cpp_backend import LlamaCppBackend
from lychee_mas.methods.build.mad_m2_builder import new_mad_m2_state
from lychee_mas.plugins.build import build_langgraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/config_mad_m2.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as source:
        config = yaml.safe_load(source)
    backend = LlamaCppBackend(base_url=config["server_url"])
    graph = build_langgraph(
        method="mad_m2",
        backend=backend,
        variant=config["variant"],
        sampling=config["sampling"],
        max_tokens=config["max_tokens"],
        stop=config["stop"],
        ignore_eos=config["ignore_eos"],
    )
    initial = new_mad_m2_state(
        config["question"], config["question_id"], config["seed"], config["gold"]
    )
    state = graph.compile().invoke(initial)
    print(
        json.dumps(
            {
                "variant": config["variant"],
                "prediction": state["prediction"],
                "score": state["score"],
                "vote_counts": state["vote_counts"],
                "masks": state["masks"],
                "empty_context_fallback": state["empty_context_fallback"],
                "calls": state["calls"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
