"""Configuration resolution for benchmark runs.

This module contains no model or runtime imports. It is shared by the CLI,
Studio launch compiler, and focused configuration tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_config_value(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a dotted key without coupling callers to a YAML library."""

    current: Any = config
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def apply_benchmark_contract(
    config: dict[str, Any],
    args: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Apply task-owned runtime and network defaults to one run configuration."""

    from lychee_mas.eval.benchmarks.assets import BenchmarkRegistry

    resolved = json.loads(json.dumps(config))
    configured_task = str(get_config_value(resolved, "run.task", "") or "")
    task = str(args.task or configured_task).strip()
    if not task:
        return resolved
    try:
        benchmark_spec, benchmark_instance = BenchmarkRegistry(Path(repo_root)).resolve_task(task)
    except ValueError:
        return resolved

    run_config = resolved.setdefault("run", {})
    run_config["task"] = task
    run_config["benchmark_spec_id"] = benchmark_spec["id"]
    if benchmark_instance is not None:
        run_config["benchmark_instance_id"] = benchmark_instance["id"]

    task_changed = bool(args.task and configured_task and task != configured_task)
    runtime_config = resolved.setdefault("runtime", {})
    for key, value in dict(benchmark_spec.get("runtime_defaults") or {}).items():
        if task_changed or runtime_config.get(key) is None:
            runtime_config[key] = value

    network_defaults = dict(benchmark_spec.get("network_defaults") or {})
    network_config = dict(resolved.get("network") or {})
    if task_changed or not network_config or network_config.get("mode") == "benchmark_default":
        network_config = {**network_config, **network_defaults}
    else:
        for key, value in network_defaults.items():
            if network_config.get(key) is None:
                network_config[key] = value
    resolved["network"] = network_config

    if network_config.get("mode") == "proxy":
        relay_port = int(network_config.get("container_proxy_port") or 17897)
        network_config.setdefault(
            "container_proxy_url",
            f"http://host.docker.internal:{relay_port}",
        )
        network_config.setdefault("proxy_relay_upstream_url", network_config.get("proxy_url"))
    return resolved


def resolve_prefix_length(config: dict[str, Any], cli_value: int | None) -> int:
    """Resolve latent-prefix length from CLI, current config, then legacy config."""

    if cli_value is not None:
        return int(cli_value)
    return int(get_config_value(config, "router.P", get_config_value(config, "memory.P", 16)))


def as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def optional_positive_limit(value: Any, *, name: str) -> int | None:
    """Parse a positive integer limit; None/unlimited means no limit."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "unlimited",
        "unbounded",
    }:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer or 'unlimited'") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1 or 'unlimited'")
    return parsed
