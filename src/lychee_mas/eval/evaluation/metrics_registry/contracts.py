"""Contracts and observations for the current evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

METRIC_CONTRACT_SCHEMA_VERSION = 1
METRIC_OBSERVATION_SCHEMA_VERSION = 1
METRIC_OBSERVATION_STATUSES = frozenset(
    {
        "measured",
        "not_applicable",
        "missing_evidence",
        "unsupported",
        "evaluator_error",
    }
)

_METRIC_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_CATEGORIES = {"task", "coordination", "efficiency", "reliability"}
_LEVELS = {"trial", "case", "run", "study"}
_VALIDATION_STATUSES = {"candidate", "calibrated", "validated", "rejected"}


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"MetricContract {field_name} must be a list of strings")
    cleaned = tuple(item.strip() for item in value if item.strip())
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"MetricContract {field_name} contains duplicates")
    return cleaned


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricContract:
    """Declarative identity, evidence requirements, and interpretation limits."""

    metric_id: str
    name: str
    description: str
    category: str
    construct: str
    unit: str
    level: str
    scope: str
    required_evidence: tuple[str, ...]
    optional_evidence: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    applicable_systems: tuple[str, ...] = ("any",)
    known_confounders: tuple[str, ...] = ()
    na_policy: Mapping[str, str] = field(default_factory=dict)
    validation_status: str = "candidate"
    evaluator: Mapping[str, Any] | None = None
    schema_version: int = METRIC_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"MetricContract requires schema_version {METRIC_CONTRACT_SCHEMA_VERSION}"
            )
        if not _METRIC_ID.fullmatch(self.metric_id):
            raise ValueError(f"invalid metric_id {self.metric_id!r}")
        if self.category not in _CATEGORIES:
            raise ValueError(f"unsupported metric category {self.category!r}")
        if self.level not in _LEVELS:
            raise ValueError(f"unsupported metric level {self.level!r}")
        if self.validation_status not in _VALIDATION_STATUSES:
            raise ValueError(
                f"unsupported metric validation_status {self.validation_status!r}"
            )
        for field_name in ("name", "description", "construct", "unit", "scope"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"MetricContract {field_name} cannot be empty")
        overlap = set(self.required_evidence) & set(self.optional_evidence)
        if overlap:
            raise ValueError(
                "MetricContract evidence cannot be both required and optional: "
                + ", ".join(sorted(overlap))
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MetricContract:
        allowed = {
            "schema_version",
            "metric_id",
            "name",
            "description",
            "category",
            "construct",
            "unit",
            "level",
            "scope",
            "required_evidence",
            "optional_evidence",
            "required_capabilities",
            "applicable_systems",
            "known_confounders",
            "na_policy",
            "validation_status",
            "evaluator",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "MetricContract has unsupported fields: " + ", ".join(sorted(unknown))
            )
        required = {
            "metric_id",
            "name",
            "description",
            "category",
            "construct",
            "unit",
            "level",
            "scope",
            "required_evidence",
        }
        missing = required - set(value)
        if missing:
            raise ValueError("MetricContract is missing fields: " + ", ".join(sorted(missing)))
        na_policy = value.get("na_policy") or {}
        if not isinstance(na_policy, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in na_policy.items()
        ):
            raise ValueError("MetricContract na_policy must be an object of string values")
        evaluator = value.get("evaluator")
        if evaluator is not None and not isinstance(evaluator, dict):
            raise ValueError("MetricContract evaluator must be an object or null")
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            metric_id=str(value["metric_id"]),
            name=str(value["name"]),
            description=str(value["description"]),
            category=str(value["category"]),
            construct=str(value["construct"]),
            unit=str(value["unit"]),
            level=str(value["level"]),
            scope=str(value["scope"]),
            required_evidence=_strings(value["required_evidence"], "required_evidence"),
            optional_evidence=_strings(value.get("optional_evidence"), "optional_evidence"),
            required_capabilities=_strings(
                value.get("required_capabilities"), "required_capabilities"
            ),
            applicable_systems=_strings(
                value.get("applicable_systems", ["any"]), "applicable_systems"
            ),
            known_confounders=_strings(
                value.get("known_confounders"), "known_confounders"
            ),
            na_policy=dict(na_policy),
            validation_status=str(value.get("validation_status", "candidate")),
            evaluator=dict(evaluator) if evaluator is not None else None,
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "construct": self.construct,
            "unit": self.unit,
            "level": self.level,
            "scope": self.scope,
            "required_evidence": list(self.required_evidence),
            "optional_evidence": list(self.optional_evidence),
            "required_capabilities": list(self.required_capabilities),
            "applicable_systems": list(self.applicable_systems),
            "known_confounders": list(self.known_confounders),
            "na_policy": dict(self.na_policy),
            "validation_status": self.validation_status,
            "evaluator": dict(self.evaluator) if self.evaluator is not None else None,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class MetricObservation:
    """One metric result with explicit status and evidence provenance."""

    metric_id: str
    status: str
    unit: str
    level: str
    value: Any = None
    case_id: str | None = None
    dataset_index: int | None = None
    trial_index: int = 0
    reason: str | None = None
    evidence_event_ids: tuple[str, ...] = ()
    evaluator_fingerprint: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = METRIC_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"MetricObservation requires schema_version {METRIC_OBSERVATION_SCHEMA_VERSION}"
            )
        if not _METRIC_ID.fullmatch(self.metric_id):
            raise ValueError(f"invalid metric_id {self.metric_id!r}")
        if self.status not in METRIC_OBSERVATION_STATUSES:
            raise ValueError(f"unsupported MetricObservation status {self.status!r}")
        if self.level not in _LEVELS:
            raise ValueError(f"unsupported metric level {self.level!r}")
        if self.status == "measured":
            if self.value is None:
                raise ValueError("measured MetricObservation requires a value")
            if not self.evaluator_fingerprint:
                raise ValueError("measured MetricObservation requires evaluator_fingerprint")
        else:
            if self.value is not None:
                raise ValueError(f"{self.status} MetricObservation must not contain a value")
            if not self.reason:
                raise ValueError(f"{self.status} MetricObservation requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "level": self.level,
            "case_id": self.case_id,
            "dataset_index": self.dataset_index,
            "trial_index": self.trial_index,
            "reason": self.reason,
            "evidence_event_ids": list(self.evidence_event_ids),
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "attributes": dict(self.attributes),
        }
