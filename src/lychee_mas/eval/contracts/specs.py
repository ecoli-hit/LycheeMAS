"""Pure-data contracts for Eval Studio experiment assembly.

The contracts deliberately keep logical collaboration separate from model
deployment.  A :class:`TeamSpec` can therefore be reused across local HF,
vLLM, and cloud API experiments without changing the team definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...runtime.model.token_budget import (
    BUDGET_POLICY_FIELDS,
    normalize_token_budget_policy,
)
from ..teams.contracts import TEAM_SPEC_SCHEMA_VERSION, normalize_team_spec_document


def normalize_invocation_overrides(
    value: dict[str, Any] | None, *, label: str = "invocation"
) -> dict[str, Any]:
    """Validate fixed per-call generation and thinking policy overrides."""

    overrides = dict(value or {})
    allowed = {
        "max_new_tokens",
        "do_sample",
        *BUDGET_POLICY_FIELDS,
        "top_k",
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "thinking_mode",
        "preserve_thinking",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {', '.join(sorted(unknown))}")
    integer_fields = (
        "max_new_tokens",
        "max_input_tokens",
        "max_thinking_budget_tokens",
        "top_k",
    )
    for field_name in integer_fields:
        if overrides.get(field_name) is None:
            overrides.pop(field_name, None)
            continue
        parsed = int(overrides[field_name])
        if parsed < 1:
            raise ValueError(f"{label} {field_name} must be positive")
        overrides[field_name] = parsed
    for field_name in (
        "min_output_reserve_tokens",
        "min_thinking_reserve_tokens",
        "min_final_reserve_tokens",
        "safety_margin_tokens",
    ):
        if overrides.get(field_name) is None:
            overrides.pop(field_name, None)
            continue
        parsed = int(overrides[field_name])
        if parsed < 0:
            raise ValueError(f"{label} {field_name} must be non-negative")
        overrides[field_name] = parsed
    if "max_new_tokens" in overrides:
        budget = normalize_token_budget_policy(
            {key: overrides[key] for key in BUDGET_POLICY_FIELDS if key in overrides},
            max_new_tokens=overrides["max_new_tokens"],
        )
        for key, parsed in budget.to_dict().items():
            if key in overrides:
                overrides[key] = parsed
    float_fields = (
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    )
    for field_name in float_fields:
        if overrides.get(field_name) is None:
            overrides.pop(field_name, None)
            continue
        overrides[field_name] = float(overrides[field_name])
    if overrides.get("temperature", 0.0) < 0.0:
        raise ValueError(f"{label} temperature must be non-negative")
    if "top_p" in overrides and not 0.0 < overrides["top_p"] <= 1.0:
        raise ValueError(f"{label} top_p must be greater than 0 and at most 1")
    if "min_p" in overrides and not 0.0 <= overrides["min_p"] <= 1.0:
        raise ValueError(f"{label} min_p must be between 0 and 1")
    if "presence_penalty" in overrides and not -2.0 <= overrides["presence_penalty"] <= 2.0:
        raise ValueError(f"{label} presence_penalty must be between -2 and 2")
    if "repetition_penalty" in overrides and overrides["repetition_penalty"] <= 0.0:
        raise ValueError(f"{label} repetition_penalty must be positive")
    thinking_mode = str(overrides.get("thinking_mode") or "inherit")
    if thinking_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError(f"{label} thinking_mode must be inherit, enabled, or disabled")
    if thinking_mode == "inherit":
        overrides.pop("thinking_mode", None)
    else:
        overrides["thinking_mode"] = thinking_mode
    if overrides.get("preserve_thinking") is None:
        overrides.pop("preserve_thinking", None)
    else:
        overrides["preserve_thinking"] = bool(overrides["preserve_thinking"])
    if overrides.get("do_sample") is None:
        overrides.pop("do_sample", None)
    else:
        overrides["do_sample"] = bool(overrides["do_sample"])
    return overrides


@dataclass
class TeamSpec:
    """Canonical framework-neutral TeamSpec v14 contract."""

    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    shared_state: list[dict[str, Any]] = field(default_factory=list)
    lifecycle: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TEAM_SPEC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return normalize_team_spec_document(
            {
                "schema_version": self.schema_version,
                "id": self.id,
                "metadata": self.metadata,
                "nodes": self.nodes,
                "relations": self.relations,
                "shared_state": self.shared_state,
                "lifecycle": self.lifecycle,
            }
        )


@dataclass
class ModelInstance:
    """One concrete local materialization of a registered ModelSpec."""

    id: str
    model_spec_id: str
    model_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    path: str
    acquisition: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"


@dataclass
class APIInstance:
    """One concrete endpoint and credential binding for a registered APISpec."""

    id: str
    api_spec_id: str
    api_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    model_id: str
    base_url: str
    auth_mode: str = "env"
    credential_source: str = "environment"


@dataclass
class DeploymentSpec:
    """One reusable model service or local model runtime."""

    id: str
    kind: str
    source_spec: dict[str, str]
    actual_pricing_spec_id: str
    api_equivalent_pricing_spec_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_spec": dict(self.source_spec),
            "actual_pricing_spec_id": self.actual_pricing_spec_id,
            "api_equivalent_pricing_spec_id": self.api_equivalent_pricing_spec_id,
            **self.options,
        }


@dataclass
class DeploymentInstance:
    """Materialized, health-checked deployment that a role can invoke."""

    id: str
    deployment_spec_id: str
    source_instance: dict[str, str]
    kind: str
    model_id: str
    status: str
    actual_pricing_instance_id: str
    deployment_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    api_equivalent_pricing_instance_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "deployment_spec_id": self.deployment_spec_id,
            "source_instance": dict(self.source_instance),
            "kind": self.kind,
            "model_id": self.model_id,
            "status": self.status,
            "actual_pricing_instance_id": self.actual_pricing_instance_id,
            "deployment_spec_fingerprint": self.deployment_spec_fingerprint,
            "created_at_utc": self.created_at_utc,
            "creation_source": self.creation_source,
            "api_equivalent_pricing_instance_id": self.api_equivalent_pricing_instance_id,
            **self.options,
        }


@dataclass
class PricingSpec:
    """Complete, versionable price configuration used by one or more deployments."""

    id: str
    basis: str
    billing_mode: str
    rate_card_mode: str
    rate_unit: str
    supported_deployment_kinds: list[str]
    required_rates: list[str]
    currency: str
    rates: dict[str, float | None]
    optional_rates: list[str] = field(default_factory=list)
    rate_tiers: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    amortization_policy: str | None = None
    amortization_hours_per_month: float | None = None
    notes: str = ""


@dataclass
class PricingInstance:
    """Immutable evidence that a particular PricingSpec version was materialized."""

    id: str
    pricing_spec_id: str
    pricing_spec_fingerprint: str
    created_at_utc: str
    creation_source: str = "studio"


@dataclass
class BenchmarkSpec:
    """Minimal registry identity for one code-registered Benchmark."""

    id: str
    name: str
    category: str
    schema_version: int = 4


@dataclass
class BenchmarkInstance:
    """One prepared and integrity-checked materialization of a BenchmarkSpec."""

    id: str
    benchmark_spec_id: str
    benchmark_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    prepared_path: str
    acquisition: dict[str, Any] = field(default_factory=dict)
    raw_path: str | None = None
    supported_tasks: list[str] = field(default_factory=list)
    status: str = "unknown"
    manifest_path: str | None = None
    provenance: list[dict[str, Any]] = field(default_factory=list)
    managed: bool = True


@dataclass
class NodeResourceBinding:
    """Bind one Node requirement to a concrete resource Instance."""

    node_id: str
    requirement: str
    resource_instance_type: str
    resource_instance_id: str
    generation_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamInstance:
    """A TeamSpec compiled for one runtime framework plus resource bindings."""

    id: str
    team_spec_id: str
    team_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    runtime_framework: str
    framework_options: dict[str, Any] = field(default_factory=dict)
    resource_bindings: list[NodeResourceBinding] = field(default_factory=list)
    framework_binding_report: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """Reusable benchmark/team experiment recipe; it references Specs only."""

    id: str
    benchmark: dict[str, Any]
    team_spec_id: str
    runtime: dict[str, Any]
    network: dict[str, Any]
    environment: dict[str, Any]
    evaluation: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": dict(self.environment),
            "benchmark": dict(self.benchmark),
            "id": self.id,
            "team_spec_id": self.team_spec_id,
            "runtime": dict(self.runtime),
            "network": dict(self.network),
            "evaluation": dict(self.evaluation),
        }


@dataclass
class ExperimentInstance:
    """Persistent queue entry for one concrete execution of an ExperimentSpec."""

    id: str
    experiment_spec_id: str
    experiment_spec_fingerprint: str
    created_at_utc: str
    creation_source: str
    benchmark_instance_id: str
    team_instance_id: str
    status: str = "ready"
    priority: int = 100
    launcher: dict[str, Any] = field(default_factory=lambda: {"type": "subprocess"})
    launch_id: str | None = None
    launch_dir: str | None = None
    run_dir: str | None = None


def validate_project_document(value: dict[str, Any]) -> dict[str, Any]:
    """Reject obsolete Studio schemas instead of silently changing semantics."""

    project = dict(value)
    version = int(project.get("schema_version") or 0)
    if version != 4:
        raise ValueError(
            f"unsupported project schema_version {version}; recreate it as schema_version 4"
        )
    team = project.get("team")
    if not isinstance(team, dict):
        raise ValueError("project requires a team object")
    project["team"] = normalize_team_spec_document(team)
    return project
