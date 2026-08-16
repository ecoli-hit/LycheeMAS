"""Public benchmark contract used by data preparation, inference, and scoring."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class PrepareHandler(Protocol):
    def __call__(self, force: bool = False, source: str | None = None) -> str: ...


class Loader(Protocol):
    def __call__(self, n: int | None = None) -> list[dict[str, Any]]: ...


ScoreHandler = Callable[[str, Any, Mapping[str, Any]], Mapping[str, Any]]

STANDARD_SOURCE_PROVIDERS = ("modelscope", "huggingface", "github")


def resolve_provider_ids(sources: Mapping[str, Any], provider: str) -> list[str]:
    """Resolve one provider's environment override and registered defaults."""

    section = sources.get(provider, {})
    if not isinstance(section, Mapping):
        return []
    ids: list[str] = []
    env_name = section.get("env")
    if env_name:
        override = os.environ.get(str(env_name))
        if override:
            ids.append(override)
    ids.extend(str(item) for item in section.get("default_ids", []) if item)
    return ids


def resolve_source_backend_order(
    benchmark_id: str,
    sources: Mapping[str, Any],
    source: str | None,
    *,
    providers: tuple[str, ...] = STANDARD_SOURCE_PROVIDERS,
) -> list[str]:
    """Resolve ``auto`` or one explicitly requested data provider."""

    requested = (source or os.environ.get("LYCHEE_DATA_SOURCE") or "auto").lower()
    if requested == "auto":
        return [provider for provider in providers if resolve_provider_ids(sources, provider)]
    if requested in providers:
        if not resolve_provider_ids(sources, requested):
            raise ValueError(f"benchmark {benchmark_id!r} has no registered {requested} source")
        return [requested]
    valid = ", ".join(("auto", *providers))
    raise ValueError(f"unknown {benchmark_id} data source {requested!r}; choose {valid}")


class BenchmarkEvaluationError(RuntimeError):
    """Raised when a benchmark-owned evaluator cannot produce a valid score."""


@dataclass(frozen=True)
class BenchmarkCase:
    """Canonical representation of one case returned by every benchmark."""

    task: str
    kind: str
    question: str
    gold: Any
    context: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
        *,
        expected_task: str,
        expected_kind: str,
        benchmark_id: str,
    ) -> "BenchmarkCase":
        missing = [key for key in ("task", "kind", "question", "gold") if key not in value]
        if missing:
            raise ValueError(
                f"benchmark {benchmark_id!r} returned a case without " + ", ".join(missing)
            )
        task = str(value["task"])
        if task != expected_task:
            raise ValueError(
                f"benchmark {benchmark_id!r} loader for {expected_task!r} returned task={task!r}"
            )
        kind = str(value["kind"])
        if kind != expected_kind:
            raise ValueError(
                f"benchmark {benchmark_id!r} loader for {expected_task!r} "
                f"returned kind={kind!r}, expected {expected_kind!r}"
            )
        metadata = dict(value.get("metadata") or {})
        metadata["benchmark_id"] = benchmark_id
        standard = {"task", "kind", "question", "gold", "context", "metadata"}
        return cls(
            task=task,
            kind=kind,
            question=str(value["question"]),
            gold=value.get("gold"),
            context=value.get("context"),
            metadata=metadata,
            extensions={key: item for key, item in value.items() if key not in standard},
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "kind": self.kind,
            "question": self.question,
            "gold": self.gold,
            "context": self.context,
            "metadata": dict(self.metadata),
            **self.extensions,
        }


@dataclass
class CaseMaterialization:
    """Benchmark contribution to one isolated runtime workspace."""

    task_text: str
    visible_paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolBundle:
    """Runtime tools plus resources that must be closed after a case."""

    tools: list[Any] = field(default_factory=list)
    resources: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationContext:
    """Stable input passed to benchmark-owned external evaluators."""

    run_dir: Path
    run_info: Mapping[str, Any]
    options: Mapping[str, Any]
    external_evaluator: str = "auto"


class Benchmark:
    """One benchmark source exposed through prepare/load/score/aggregate."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        full_prepare_target: str,
        prepare_handlers: Mapping[str, PrepareHandler],
        loaders: Mapping[str, Loader],
        scorer_kinds: Mapping[str, str],
        score_handlers: Mapping[str, ScoreHandler],
        name: str | None = None,
        category: str = "other",
        sources: Mapping[str, Any] | None = None,
        binary_kinds: Sequence[str] = (),
        prepare_aliases: Mapping[str, str] | None = None,
        extractor_names: Mapping[str, str] | None = None,
        scoring_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        sandbox_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        runtime_defaults: Mapping[str, Any] | None = None,
        network_defaults: Mapping[str, Any] | None = None,
        contamination_audit_default: bool = False,
    ) -> None:
        self.id = benchmark_id
        self.name = name or benchmark_id
        self.category = category
        self.sources = dict(sources or {})
        self.full_prepare_target = full_prepare_target
        self.prepare_handlers = dict(prepare_handlers)
        self.loaders = dict(loaders)
        self.scorer_kinds = {str(task): str(kind) for task, kind in scorer_kinds.items()}
        self.score_handlers = dict(score_handlers)
        self.binary_kinds = frozenset(str(kind) for kind in binary_kinds)
        self.prepare_aliases = dict(prepare_aliases or {})
        self.extractor_names = {
            task: str((extractor_names or {}).get(task, "default")) for task in self.loaders
        }
        self.capabilities = deepcopy(
            dict(capabilities) if capabilities is not None else {"required": ["text_generation"]}
        )
        self.sandbox_profiles = deepcopy(dict(sandbox_profiles or {}))
        self.runtime_defaults = deepcopy(
            dict(runtime_defaults)
            if runtime_defaults is not None
            else {"docker_image": "lychee-python-sandbox:local"}
        )
        self.network_defaults = deepcopy(
            dict(network_defaults)
            if network_defaults is not None
            else {"access": "optional", "mode": "direct"}
        )
        declared_profiles = dict(scoring_profiles or {})
        self.scoring_profiles = {
            task: deepcopy(
                declared_profiles.get(task)
                or {
                    "default_profile": "official",
                    "profiles": {
                        "official": {
                            "scorer_id": kind,
                            "parameters": {},
                        }
                    },
                }
            )
            for task, kind in self.scorer_kinds.items()
        }
        self.contamination_audit_default = contamination_audit_default
        if full_prepare_target not in self.prepare_handlers:
            raise ValueError(
                f"benchmark {benchmark_id!r} has no handler for {full_prepare_target!r}"
            )
        if set(self.scorer_kinds) != set(self.loaders):
            raise ValueError(
                f"benchmark {benchmark_id!r} scorer_kinds must cover exactly its tasks"
            )
        missing_scorers = set(self.scorer_kinds.values()) - set(self.score_handlers)
        if missing_scorers:
            raise ValueError(
                f"benchmark {benchmark_id!r} has no score handlers for {sorted(missing_scorers)}"
            )
        self._validate_contract_metadata()

    def _validate_contract_metadata(self) -> None:
        unknown_profiles = set(self.scoring_profiles) - set(self.loaders)
        if unknown_profiles:
            raise ValueError(
                f"benchmark {self.id!r} has scoring profiles for unknown tasks "
                f"{sorted(unknown_profiles)}"
            )
        for task, scoring in self.scoring_profiles.items():
            default_profile = str(scoring.get("default_profile") or "")
            profiles = dict(scoring.get("profiles") or {})
            if not default_profile or default_profile not in profiles:
                raise ValueError(
                    f"benchmark {self.id!r} task {task!r} has no valid default scoring profile"
                )
            for profile_id, profile in profiles.items():
                scorer_id = str(dict(profile or {}).get("scorer_id") or "")
                if scorer_id != self.scorer_kinds[task]:
                    raise ValueError(
                        f"benchmark {self.id!r} task {task!r} scoring profile "
                        f"{profile_id!r} must use scorer {self.scorer_kinds[task]!r}"
                    )
        default_image = str(self.runtime_defaults.get("docker_image") or "")
        profile_images = {
            str(dict(profile or {}).get("docker_image") or "")
            for profile in self.sandbox_profiles.values()
        }
        if self.sandbox_profiles and default_image not in profile_images:
            raise ValueError(
                f"benchmark {self.id!r} default Docker image must reference a sandbox profile"
            )

    @property
    def runnable_tasks(self) -> tuple[str, ...]:
        return tuple(self.loaders)

    @property
    def direct_prepare_targets(self) -> tuple[str, ...]:
        return tuple(self.prepare_handlers)

    @property
    def accepted_prepare_targets(self) -> tuple[str, ...]:
        return (*self.prepare_handlers, *self.prepare_aliases)

    @property
    def task_contracts(self) -> dict[str, dict[str, Any]]:
        return {
            task: {
                "kind": self.scorer_kinds[task],
                "extractor": self.extractor_names[task],
                "scoring": deepcopy(self.scoring_profiles[task]),
            }
            for task in self.runnable_tasks
        }

    def descriptor(self) -> dict[str, Any]:
        """Return the executable contract exposed as read-only Studio metadata."""

        return {
            "schema_version": 1,
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "prepare_target": self.full_prepare_target,
            "prepare_targets": list(self.accepted_prepare_targets),
            "runnable_tasks": list(self.runnable_tasks),
            "kinds": sorted(set(self.scorer_kinds.values())),
            "task_contracts": self.task_contracts,
            "capabilities": deepcopy(self.capabilities),
            "sandbox_profiles": deepcopy(self.sandbox_profiles),
            "runtime_defaults": deepcopy(self.runtime_defaults),
            "network_defaults": deepcopy(self.network_defaults),
            "sources": deepcopy(self.sources),
            "contamination_audit_default": self.contamination_audit_default,
        }

    def provider_ids(self, provider: str) -> list[str]:
        """Return configured source IDs, with an environment override first."""

        return resolve_provider_ids(self.sources, provider)

    def source_backend_order(
        self,
        source: str | None,
        *,
        providers: tuple[str, ...] = STANDARD_SOURCE_PROVIDERS,
    ) -> list[str]:
        """Resolve ``auto`` or one explicitly requested data provider."""

        return resolve_source_backend_order(
            self.id,
            self.sources,
            source,
            providers=providers,
        )

    def other_defaults(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.sources.get("other_defaults", [])]

    def fallback_specs(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.sources.get("fallback_files", [])]

    def prepare(
        self,
        target: str | None = None,
        *,
        force: bool = False,
        source: str | None = None,
    ) -> str:
        target = target or self.full_prepare_target
        resolved = self.prepare_aliases.get(target, target)
        try:
            handler = self.prepare_handlers[resolved]
        except KeyError as exc:
            raise KeyError(f"benchmark {self.id!r} cannot prepare target {target!r}") from exc
        return str(handler(force, source))

    def load(self, task: str | None = None, n: int | None = None) -> list[dict[str, Any]]:
        if task is None:
            if len(self.loaders) != 1:
                raise ValueError(f"benchmark {self.id!r} requires an explicit runnable task")
            task = next(iter(self.loaders))
        try:
            values = self.loaders[task](n=n)
        except KeyError as exc:
            raise KeyError(f"benchmark {self.id!r} has no task {task!r}") from exc
        return [
            BenchmarkCase.from_record(
                value,
                expected_task=task,
                expected_kind=self.scorer_kinds[task],
                benchmark_id=self.id,
            ).to_record()
            for value in values
        ]

    def score(
        self,
        prediction: str,
        case: Mapping[str, Any],
        *,
        record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = str(case.get("kind") or "")
        try:
            handler = self.score_handlers[kind]
        except KeyError as exc:
            raise KeyError(f"benchmark {self.id!r} has no scorer for kind {kind!r}") from exc
        details = dict(handler(prediction, case.get("gold"), record or {}))
        if "score" not in details:
            raise ValueError(f"benchmark {self.id!r} scorer {kind!r} did not return a score")
        details["score"] = float(details["score"])
        return details

    def is_binary(self, kind: str) -> bool:
        return kind in self.binary_kinds

    def materialize_case(
        self,
        query: Any,
        workspace: Path,
        task_text: str,
    ) -> CaseMaterialization:
        return CaseMaterialization(task_text=task_text)

    def create_tools(self, slot: str, query_meta: Mapping[str, Any]) -> ToolBundle:
        raise ValueError(f"benchmark {self.id!r} does not provide tool slot {slot!r}")

    def collect_prediction(
        self,
        *,
        query: Any,
        messages: Sequence[Any],
        workspace: Path | None,
        default_text: str,
    ) -> str:
        return default_text

    def extract_messages(self, task: str, messages: Sequence[Any]) -> str:
        from ..task_config import EXTRACTORS

        name = self.extractor_names.get(task, "default")
        try:
            extractor = EXTRACTORS[name]
        except KeyError as exc:
            raise KeyError(f"benchmark {self.id!r} declares unknown extractor {name!r}") from exc
        return extractor(list(messages))

    def add_analysis_arguments(self, parser: Any) -> None:
        return None

    def evaluation_options(self, args: Any) -> dict[str, Any]:
        return {}

    def prepare_evaluation(
        self,
        predictions: list[dict[str, Any]],
        gold_by_id: Mapping[str, dict[str, Any]],
        context: EvaluationContext,
    ) -> None:
        return None

    def aggregate(
        self,
        samples: Sequence[Mapping[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return metrics
