"""Non-binding benchmark capability requirements and experiment suggestions.

These records help users assemble a valid ExperimentSpec.  They never select
or construct a TeamSpec, so benchmark data/evaluation remains independent from
team design.
"""

from __future__ import annotations

from typing import Any

DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "required_capabilities": ["text_generation"],
    "suggested_role_profile": "default",
}

BENCHMARK_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "gsm8k": {"required_capabilities": ["text_generation"], "suggested_role_profile": "reason"},
    "aime_2024": {
        "required_capabilities": ["text_generation"],
        "suggested_role_profile": "aime",
    },
    "medqa": {"required_capabilities": ["text_generation"], "suggested_role_profile": "fact"},
    "arc_easy": {"required_capabilities": ["text_generation"], "suggested_role_profile": "fact"},
    "openbookqa": {
        "required_capabilities": ["text_generation"],
        "suggested_role_profile": "fact",
    },
    "locomo10": {
        "required_capabilities": ["text_generation", "long_context"],
        "suggested_role_profile": "memory",
    },
    "human_eval": {
        "required_capabilities": ["code_generation", "code_execution"],
        "suggested_role_profile": "human_eval",
    },
    "gaia_validation": {
        "required_capabilities": ["file_access", "web_browsing", "code_execution"],
        "suggested_role_profile": "gaia",
    },
    "bbeh": {
        "required_capabilities": ["text_generation"],
        "suggested_role_profile": "reason",
    },
    "hle": {
        "required_capabilities": ["text_generation", "vision"],
        "suggested_role_profile": "hle",
    },
    "swe_bench_verified": {
        "required_capabilities": ["code_generation", "file_access", "code_execution"],
        "suggested_role_profile": "swe_bench_verified",
    },
    "workbench": {
        "required_capabilities": ["text_generation", "native_tool_calls"],
        "suggested_role_profile": "workbench",
    },
}


def requirements_for_task(task: str | None) -> dict[str, Any]:
    key = str(task or "")
    if key.startswith("gaia_validation"):
        key = "gaia_validation"
    return {**DEFAULT_REQUIREMENTS, **BENCHMARK_REQUIREMENTS.get(key, {})}
