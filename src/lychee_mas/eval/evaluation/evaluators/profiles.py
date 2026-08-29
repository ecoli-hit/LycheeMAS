"""Evaluation Profiles select the current Metric Contracts for one analysis."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

EVALUATION_PROFILE_SCHEMA_VERSION = 1

_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricReference:
    metric_id: str

    def __post_init__(self) -> None:
        if not _METRIC_ID.fullmatch(self.metric_id):
            raise ValueError(f"invalid metric_id {self.metric_id!r}")
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MetricReference:
        unknown = set(value) - {"metric_id"}
        if unknown:
            raise ValueError("MetricReference has unsupported fields: " + ", ".join(unknown))
        return cls(metric_id=str(value["metric_id"]))

    def to_dict(self) -> dict[str, str]:
        return {"metric_id": self.metric_id}


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    name: str
    description: str
    metric_refs: tuple[MetricReference, ...]
    strict_evidence_gate: bool = True
    schema_version: int = EVALUATION_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"EvaluationProfile requires schema_version {EVALUATION_PROFILE_SCHEMA_VERSION}"
            )
        if not _ID.fullmatch(self.profile_id):
            raise ValueError(f"invalid profile_id {self.profile_id!r}")
        if not self.name.strip() or not self.description.strip():
            raise ValueError("EvaluationProfile name and description cannot be empty")
        if not self.metric_refs:
            raise ValueError("EvaluationProfile must reference at least one metric")
        identities = [item.metric_id for item in self.metric_refs]
        if len(set(identities)) != len(identities):
            raise ValueError("EvaluationProfile contains duplicate metric references")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationProfile:
        allowed = {
            "schema_version",
            "profile_id",
            "name",
            "description",
            "strict_evidence_gate",
            "metric_refs",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "EvaluationProfile has unsupported fields: " + ", ".join(sorted(unknown))
            )
        required = {"profile_id", "name", "description", "metric_refs"}
        missing = required - set(value)
        if missing:
            raise ValueError(
                "EvaluationProfile is missing fields: " + ", ".join(sorted(missing))
            )
        refs = value["metric_refs"]
        if not isinstance(refs, list) or any(not isinstance(item, dict) for item in refs):
            raise ValueError("EvaluationProfile metric_refs must be a list of objects")
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            profile_id=str(value["profile_id"]),
            name=str(value["name"]),
            description=str(value["description"]),
            strict_evidence_gate=bool(value.get("strict_evidence_gate", True)),
            metric_refs=tuple(MetricReference.from_mapping(item) for item in refs),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "strict_evidence_gate": self.strict_evidence_gate,
            "metric_refs": [item.to_dict() for item in self.metric_refs],
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


class EvaluationProfileRegistry:
    """Load one current profile per id from LycheeMAS or supplied directories."""

    def __init__(
        self,
        profile_dirs: Iterable[str | Path] | None = None,
        *,
        include_builtin: bool = True,
    ) -> None:
        directories: list[Path] = []
        if include_builtin:
            directories.append(Path(__file__).with_name("profiles"))
        directories.extend(Path(item).expanduser().resolve() for item in profile_dirs or ())
        self._profiles: dict[str, EvaluationProfile] = {}
        for directory in directories:
            self._load_directory(directory)

    def _load_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            documents = value if isinstance(value, list) else [value]
            if any(not isinstance(item, dict) for item in documents):
                raise ValueError(f"evaluation profile file must contain object(s): {path}")
            for document in documents:
                profile = EvaluationProfile.from_mapping(document)
                key = profile.profile_id
                if key in self._profiles:
                    raise ValueError(f"duplicate EvaluationProfile {profile.profile_id}")
                self._profiles[key] = profile

    def get(self, profile_id: str) -> EvaluationProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown EvaluationProfile {profile_id}") from exc

    def all(self) -> list[EvaluationProfile]:
        return sorted(self._profiles.values(), key=lambda item: item.profile_id)

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(item.fingerprint for item in self.all())
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        profiles = self.all()
        return {
            "schema_version": 1,
            "registry_fingerprint": self.fingerprint,
            "num_profiles": len(profiles),
            "profiles": [profile.to_dict() for profile in profiles],
        }
