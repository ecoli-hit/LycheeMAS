"""Read-only repository catalog used by Eval queries and command compilation."""

from __future__ import annotations

import base64
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from lychee_mas.runtime.events.store import run_event_paths, run_events_path


def repository_root() -> Path:
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "lychee_mas"
        ).is_dir():
            return candidate
    raise RuntimeError(f"cannot locate LycheeMAS repository root from {source}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _asset_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "status": "missing", "file_count": 0, "size_bytes": 0}
    files = []
    for item in path.rglob("*") if path.is_dir() else [path]:
        if not item.is_file():
            continue
        relative = item.relative_to(path) if path.is_dir() else Path(item.name)
        if (
            ".cache" in relative.parts
            or item.name in {".lychee_source.json", "manifest.json"}
            or item.name.endswith((".lock", ".incomplete", ".part"))
        ):
            continue
        files.append(item)
    return {
        "exists": True,
        "status": "present" if files else "metadata_only",
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def _path_id(path: Path, root: Path) -> str:
    relative = str(path.resolve().relative_to(root.resolve()))
    return base64.urlsafe_b64encode(relative.encode()).decode().rstrip("=")


def _external_path_id(path: Path) -> str:
    value = base64.urlsafe_b64encode(str(path.resolve()).encode()).decode().rstrip("=")
    return f"external.{value}"


def decode_path_id(value: str, root: Path) -> Path:
    padding = "=" * (-len(value) % 4)
    try:
        relative = base64.urlsafe_b64decode(value + padding).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid path id") from exc
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path id escapes configured root") from exc
    return path


class EvalCatalog:
    """Expose benchmark compatibility metadata and prior runs."""

    def __init__(
        self,
        repo_root: str | os.PathLike | None = None,
        *,
        raw_root: str | os.PathLike | None = None,
        prepared_root: str | os.PathLike | None = None,
        models_root: str | os.PathLike | None = None,
        runs_root: str | os.PathLike | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or repository_root()).resolve()
        self.raw_root = Path(
            raw_root
            or os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT")
            or self.repo_root / "data/benchmarks/raw"
        ).resolve()
        self.prepared_root = Path(
            prepared_root
            or os.environ.get("LYCHEE_BENCHMARK_PREPARED_ROOT")
            or self.repo_root / "data/benchmarks/prepared"
        ).resolve()
        self.models_root = Path(
            models_root or os.environ.get("LYCHEE_MODEL_ROOT") or self.repo_root / "models"
        ).resolve()
        self.runs_root = Path(
            runs_root
            or os.environ.get("LYCHEE_BENCHMARK_RUNS_ROOT")
            or self.repo_root / "runs/benchmarks"
        ).resolve()
        self._runs_cache: dict[tuple[int, tuple[str, ...]], tuple[float, list[dict[str, Any]]]] = {}

    def paths(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "raw_root": str(self.raw_root),
            "prepared_root": str(self.prepared_root),
            "models_root": str(self.models_root),
            "runs_root": str(self.runs_root),
        }

    def benchmarks(self, *, include_asset_stats: bool = True) -> list[dict[str, Any]]:
        from ...benchmarks import BENCHMARK_STRUCTURE, BENCHMARKS
        from ...benchmarks.manifest import benchmark_key_for_target, verify_prepared_manifest

        rows: list[dict[str, Any]] = []
        for source in BENCHMARK_STRUCTURE:
            full_target = str(source["full_prepare_target"])
            key = benchmark_key_for_target(full_target)
            benchmark = BENCHMARKS.get(key)
            raw_path = self.raw_root / key
            prepared_path = self.prepared_root / key
            manifest_path = prepared_path / "manifest.json"
            prepared_stats = (
                _asset_stats(prepared_path)
                if include_asset_stats
                else {
                    "exists": prepared_path.exists(),
                    "status": "present" if manifest_path.is_file() else "missing",
                    "file_count": None,
                    "size_bytes": None,
                }
            )
            prepared_status = "not_downloaded"
            manifest: dict[str, Any] = {}
            if manifest_path.is_file() and prepared_stats["file_count"]:
                try:
                    verified = verify_prepared_manifest(manifest_path, verify_hashes=False)
                    prepared_status = str(verified["status"])
                    manifest = _read_json(manifest_path)
                except Exception:
                    prepared_status = "invalid"
            elif prepared_stats["file_count"]:
                prepared_status = "unverified"
            elif prepared_path.exists():
                prepared_status = "incomplete"

            raw_assets = []
            if raw_path.is_dir():
                for provider_dir in sorted(path for path in raw_path.iterdir() if path.is_dir()):
                    for source_dir in sorted(
                        path for path in provider_dir.iterdir() if path.is_dir()
                    ):
                        metadata = _read_json(source_dir / ".lychee_source.json")
                        stats = (
                            _asset_stats(source_dir)
                            if include_asset_stats
                            else {
                                "exists": True,
                                "status": "present"
                                if any(
                                    item.name != ".lychee_source.json"
                                    for item in source_dir.iterdir()
                                )
                                else "metadata_only",
                                "file_count": None,
                                "size_bytes": None,
                            }
                        )
                        raw_assets.append(
                            {
                                "provider": metadata.get("provider") or provider_dir.name,
                                "source_id": metadata.get("source_id")
                                or source_dir.name.replace("--", "/", 1),
                                "path": str(source_dir),
                                **stats,
                            }
                        )
            raw_status = (
                "ready"
                if any(item["status"] == "present" for item in raw_assets)
                else "incomplete"
                if raw_assets
                else "not_downloaded"
            )
            registered_download_sources: dict[str, list[Any]] = {
                "modelscope": benchmark.provider_ids("modelscope"),
                "huggingface": benchmark.provider_ids("huggingface"),
                "github": benchmark.provider_ids("github"),
            }
            reference_sources = benchmark.other_defaults()
            for reference in reference_sources:
                provider = str(reference.get("provider") or "other")
                if reference.get("selectable", False):
                    registered_download_sources.setdefault(provider, []).append(reference)
            rows.append(
                {
                    **source,
                    "benchmark_key": key,
                    "scenario": benchmark.category,
                    "status": prepared_status,
                    "raw_status": raw_status,
                    "prepared_status": prepared_status,
                    "raw_path": str(raw_path),
                    "prepared_path": str(prepared_path),
                    "manifest_path": str(manifest_path),
                    "raw_assets": raw_assets,
                    "download_sources": registered_download_sources,
                    "reference_sources": reference_sources,
                    "file_count": manifest.get("integrity", {}).get("file_count"),
                    "size_bytes": manifest.get("integrity", {}).get("total_size_bytes"),
                    "record_count": manifest.get("integrity", {}).get("known_record_count"),
                    "raw_sources": manifest.get("raw_sources", []),
                }
            )
        return rows

    def teams(self) -> dict[str, Any]:
        from ...teams.registry import TeamSpecRegistry

        return {"specs": TeamSpecRegistry(self.repo_root).all()}

    def runs(
        self,
        limit: int = 100,
        *,
        roots: list[str | os.PathLike] | None = None,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        run_roots = self._run_roots(roots)
        cache_key = (max(1, int(limit)), tuple(str(path) for path in run_roots))
        cached = self._runs_cache.get(cache_key)
        if not refresh and cached and time.monotonic() - cached[0] < 2.0:
            return deepcopy(cached[1])
        rows = []
        seen = set()
        for run_root in run_roots:
            if not run_root.exists():
                continue
            for status_path in run_root.rglob("run_status.json"):
                run_dir = status_path.parent.resolve()
                events_path = run_events_path(run_dir)
                if not events_path.is_file():
                    continue
                run_event_segments = run_event_paths(run_dir)
                if run_dir in seen:
                    continue
                seen.add(run_dir)
                status = _read_json(status_path)
                metrics = _read_json(run_dir / "metrics.json")
                contamination_summary = _read_json(run_dir / "contamination_summary.json")
                evidence_coverage = _read_json(run_dir / "evidence_coverage.json")
                metric_applicability = _read_json(run_dir / "metric_applicability.json")
                metric_evaluation = _read_json(run_dir / "metric_evaluation.json")
                evidence_path = run_dir / "evidence.jsonl"
                observations_path = run_dir / "metric_observations.jsonl"
                rows.append(
                    {
                        "id": (
                            _path_id(run_dir, self.runs_root)
                            if run_root == self.runs_root
                            else _external_path_id(run_dir)
                        ),
                        "run_dir": str(run_dir),
                        "runs_root": str(run_root),
                        "relative_path": str(run_dir.relative_to(run_root)),
                        "status": _run_status_summary(status),
                        "metrics": _run_metrics_summary(metrics),
                        "contamination_summary": contamination_summary,
                        "evidence_coverage": evidence_coverage.get("summary", {}),
                        "metric_applicability": metric_applicability.get("summary", {}),
                        "metric_evaluation": metric_evaluation.get("summary", {}),
                        "has_run_events": True,
                        "run_event_bytes": sum(path.stat().st_size for path in run_event_segments),
                        "run_event_segment_count": len(run_event_segments),
                        "has_result_projection": True,
                        "has_execution_trace": True,
                        "has_evidence": evidence_path.is_file(),
                        "has_metric_applicability": bool(metric_applicability),
                        "has_metric_observations": observations_path.is_file(),
                        "metric_observation_bytes": (
                            observations_path.stat().st_size if observations_path.is_file() else 0
                        ),
                        "evidence_bytes": (
                            evidence_path.stat().st_size if evidence_path.is_file() else 0
                        ),
                        "updated_at": status_path.stat().st_mtime,
                    }
                )
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        result = rows[: max(1, int(limit))]
        self._runs_cache[cache_key] = (time.monotonic(), deepcopy(result))
        return result

    def run_dir(
        self,
        run_id: str,
        *,
        roots: list[str | os.PathLike] | None = None,
    ) -> Path:
        run_roots = self._run_roots(roots)
        if run_id.startswith("external."):
            encoded = run_id.partition(".")[2]
            padding = "=" * (-len(encoded) % 4)
            try:
                path = Path(base64.urlsafe_b64decode(encoded + padding).decode()).resolve()
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("invalid external run id") from exc
            if not any(path == root or root in path.parents for root in run_roots):
                raise ValueError("external run id is outside registered result roots")
        else:
            path = decode_path_id(run_id, self.runs_root)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    def _run_roots(self, roots: list[str | os.PathLike] | None) -> list[Path]:
        values = [self.runs_root, *(roots or [])]
        result = []
        for value in values:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.repo_root / path
            resolved = path.resolve()
            if resolved not in result:
                result.append(resolved)
        # A recursive scan of a parent already includes every nested custom
        # run directory. Keeping both caused the same tree to be walked once
        # per ExperimentInstance during Studio bootstrap.
        minimal: list[Path] = []
        for path in sorted(result, key=lambda item: (len(item.parts), str(item))):
            if any(parent == path or parent in path.parents for parent in minimal):
                continue
            minimal.append(path)
        return minimal


def _run_status_summary(status: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "status",
        "task",
        "kind",
        "expected_trials",
        "completed_trials",
        "successful_trials",
        "failed_trials",
        "completed_distinct_cases",
        "started_at_utc",
        "updated_at_utc",
        "finished_at_utc",
    }
    return {key: value for key, value in status.items() if key in keys}


def _run_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "accuracy",
        "score_mean",
        "score_sum",
        "num_predictions",
        "successful_trials",
        "failed_trials",
        "evidence_coverage",
    }
    return {key: value for key, value in metrics.items() if key in keys}
