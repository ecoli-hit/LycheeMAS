"""Build one managed vLLM command and isolated process environment."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def prepare_vllm_launch(
    value: dict[str, Any],
    *,
    python: Path,
    model_path: Path,
    host: str,
    port: int,
    supports_flag: Callable[[Path, str, dict[str, str]], bool],
) -> dict[str, Any]:
    """Return command, environment, and version-gated metrics capabilities."""

    command = [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        str(value["model_id"]),
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(value.get("tensor_parallel_size") or 1),
        "--data-parallel-size",
        str(value.get("data_parallel_size") or 1),
        "--max-model-len",
        str(value.get("max_model_len") or 32768),
        "--gpu-memory-utilization",
        str(value.get("gpu_memory_utilization") or 0.9),
    ]
    if value.get("dtype"):
        command.extend(["--dtype", str(value["dtype"])])
    if value.get("max_num_seqs"):
        command.extend(["--max-num-seqs", str(value["max_num_seqs"])])
    if value.get("reasoning_parser"):
        command.extend(["--reasoning-parser", str(value["reasoning_parser"])])
    if value.get("reasoning_config"):
        command.extend(
            ["--reasoning-config", json.dumps(value["reasoning_config"], ensure_ascii=False)]
        )
    if value.get("enable_auto_tool_choice"):
        command.append("--enable-auto-tool-choice")
    if value.get("tool_call_parser"):
        command.extend(["--tool-call-parser", str(value["tool_call_parser"])])
    if value.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    if value.get("enforce_eager"):
        command.append("--enforce-eager")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(value.get("cuda_visible_devices") or "0")
    environment_lib = python.parent.parent / "lib"
    site_packages = sorted(
        {path.resolve() for path in environment_lib.glob("python*/site-packages")}
    )
    cuda_roots = [
        path
        for site_packages_root in site_packages
        for path in sorted((site_packages_root / "nvidia").glob("cu*"))
        if (path / "bin/nvcc").is_file()
    ]
    nvidia_library_paths = [
        path
        for site_packages_root in site_packages
        for path in sorted((site_packages_root / "nvidia").glob("*/lib"))
        if path.is_dir()
    ]
    configured_library_paths = [
        str(Path(item).expanduser())
        for item in value.get("library_paths", [])
        if str(item).strip()
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(
        item
        for item in (
            str(environment_lib),
            *(str(path) for path in nvidia_library_paths),
            *configured_library_paths,
            environment.get("LD_LIBRARY_PATH", ""),
        )
        if item
    )
    environment["PATH"] = ":".join(
        item
        for item in (
            str(python.parent),
            *(str(path / "bin") for path in cuda_roots),
            environment.get("PATH", ""),
        )
        if item
    )
    if cuda_roots:
        cuda_root = cuda_roots[0]
        include_path = str(cuda_root / "include")
        environment["CUDA_HOME"] = str(cuda_root)
        environment["CUDACXX"] = str(cuda_root / "bin/nvcc")
        environment["CPATH"] = ":".join(
            item for item in (include_path, environment.get("CPATH", "")) if item
        )
        environment["CPLUS_INCLUDE_PATH"] = ":".join(
            item
            for item in (include_path, environment.get("CPLUS_INCLUDE_PATH", ""))
            if item
        )
    if value.get("use_flashinfer_sampler") is not None:
        environment["VLLM_USE_FLASHINFER_SAMPLER"] = (
            "1" if bool(value["use_flashinfer_sampler"]) else "0"
        )

    metrics_mode = str(value.get("per_request_metrics_mode") or "auto")
    metrics_flag = "--enable-per-request-metrics"
    metrics_supported = supports_flag(python, metrics_flag, environment)
    if metrics_mode == "enabled" and not metrics_supported:
        raise ValueError(
            f"selected vLLM Python does not support {metrics_flag}; "
            "use per_request_metrics_mode=auto/disabled or upgrade vLLM"
        )
    metrics_enabled = metrics_supported and metrics_mode != "disabled"
    if metrics_enabled:
        command.append(metrics_flag)

    prompt_details_mode = str(value.get("prompt_tokens_details_mode") or "auto")
    prompt_details_flag = "--enable-prompt-tokens-details"
    prompt_details_supported = supports_flag(python, prompt_details_flag, environment)
    if prompt_details_mode == "enabled" and not prompt_details_supported:
        raise ValueError(
            f"selected vLLM Python does not support {prompt_details_flag}; "
            "use prompt_tokens_details_mode=auto/disabled or upgrade vLLM"
        )
    prompt_details_enabled = (
        prompt_details_supported and prompt_details_mode != "disabled"
    )
    if prompt_details_enabled:
        command.append(prompt_details_flag)

    return {
        "command": command,
        "environment": environment,
        "per_request_metrics_supported": metrics_supported,
        "per_request_metrics_enabled": metrics_enabled,
        "prompt_tokens_details_supported": prompt_details_supported,
        "prompt_tokens_details_enabled": prompt_details_enabled,
    }


__all__ = ["prepare_vllm_launch"]
