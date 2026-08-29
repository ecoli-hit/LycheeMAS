"""Project-owned Docker image cache with bounded, lease-aware eviction."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

GIB = 1024**3
MANIFEST_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DockerImageCachePolicy:
    mode: str = "bounded"
    max_images: int = 24
    min_free_gib: float = 12.0
    target_free_gib: float = 20.0

    def __post_init__(self) -> None:
        if self.mode not in {"bounded", "keep", "remove_after_use"}:
            raise ValueError("cache mode must be bounded, keep, or remove_after_use")
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.min_free_gib < 0 or self.target_free_gib < self.min_free_gib:
            raise ValueError("invalid free-space thresholds")


@dataclass
class DockerImageLease:
    lease_id: str
    owner: str
    images: tuple[str, ...]
    reports: list[dict[str, Any]]


class ProjectDockerImageCache:
    """Track and evict only Docker images explicitly adopted by this project."""

    def __init__(
        self,
        *,
        manifest_path: str | Path | None = None,
        policy: DockerImageCachePolicy | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        configured = os.environ.get("LYCHEE_SWEBENCH_IMAGE_CACHE_MANIFEST")
        self.manifest_path = Path(
            manifest_path
            or configured
            or "runs/eval_studio/docker_cache/swe_bench_verified.json"
        )
        self.lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
        self.policy = policy or DockerImageCachePolicy()
        self._run = command_runner

    @contextmanager
    def lease(self, images: Sequence[str], *, owner: str) -> Iterator[DockerImageLease]:
        refs = tuple(dict.fromkeys(str(item).strip() for item in images if str(item).strip()))
        lease = DockerImageLease(uuid.uuid4().hex, owner, refs, [])
        with self._locked_state() as state:
            self._drop_stale_leases(state)
            now = _utc_now()
            for image in refs:
                record = state["images"].setdefault(image, self._new_record(image, now))
                record["last_used_at_utc"] = now
                record["last_owner"] = owner
            state["leases"][lease.lease_id] = {
                "owner": owner,
                "images": list(refs),
                "pid": os.getpid(),
                "created_at_utc": now,
            }
            lease.reports.append(self._trim_locked(state, protected=set(refs), phase="before"))
        try:
            yield lease
        finally:
            with self._locked_state() as state:
                now = _utc_now()
                for image in refs:
                    record = state["images"].setdefault(image, self._new_record(image, now))
                    record.update(self._inspect_image(image))
                    record["last_used_at_utc"] = now
                    record["last_owner"] = owner
                state["leases"].pop(lease.lease_id, None)
                lease.reports.append(self._trim_locked(state, protected=set(), phase="after"))

    def adopt(self, image_usage: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Adopt images backed by LycheeMAS run evidence."""

        adopted = 0
        with self._locked_state() as state:
            for image, evidence in image_usage.items():
                inspected = self._inspect_image(image)
                if not inspected.get("present"):
                    continue
                used_at = str(evidence.get("last_used_at_utc") or _utc_now())
                record = state["images"].setdefault(image, self._new_record(image, used_at))
                record.update(inspected)
                record["last_used_at_utc"] = max(
                    str(record.get("last_used_at_utc") or ""), used_at
                )
                record["evidence"] = dict(evidence)
                adopted += 1
            trim = self._trim_locked(state, protected=set(), phase="adopt")
        return {"adopted": adopted, "trim": trim}

    def status(self) -> dict[str, Any]:
        with self._locked_state() as state:
            docker_root = self._docker_root()
            usage = shutil.disk_usage(docker_root)
            present = [item for item in state["images"].values() if item.get("present")]
            return {
                "manifest_path": str(self.manifest_path.resolve()),
                "mode": self.policy.mode,
                "managed_images": len(state["images"]),
                "present_images": len(present),
                "active_leases": len(state["leases"]),
                "docker_root": str(docker_root),
                "free_bytes": usage.free,
                "free_gib": usage.free / GIB,
                "policy": {
                    "max_images": self.policy.max_images,
                    "min_free_gib": self.policy.min_free_gib,
                    "target_free_gib": self.policy.target_free_gib,
                },
            }

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self._load_state()
            try:
                yield state
            finally:
                self._write_state(state)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._empty_state()
        try:
            state = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = self._empty_state()
        if state.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported Docker cache manifest: {self.manifest_path}")
        state.setdefault("images", {})
        state.setdefault("leases", {})
        state.setdefault("evictions", [])
        return state

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "images": {},
            "leases": {},
            "evictions": [],
            "updated_at_utc": _utc_now(),
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        state["updated_at_utc"] = _utc_now()
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _new_record(image: str, now: str) -> dict[str, Any]:
        return {
            "image": image,
            "present": False,
            "image_id": None,
            "first_used_at_utc": now,
            "last_used_at_utc": now,
            "last_owner": None,
        }

    def _drop_stale_leases(self, state: dict[str, Any]) -> None:
        for lease_id, lease in list(state["leases"].items()):
            pid = int(lease.get("pid") or 0)
            if pid <= 0 or not Path(f"/proc/{pid}").exists():
                state["leases"].pop(lease_id, None)

    def _inspect_image(self, image: str) -> dict[str, Any]:
        result = self._run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {"present": False, "image_id": None}
        try:
            record = json.loads(result.stdout)[0]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return {"present": False, "image_id": None}
        return {
            "present": True,
            "image_id": record.get("Id"),
            "created_at": record.get("Created"),
        }

    def _docker_root(self) -> Path:
        result = self._run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip() if result.returncode == 0 else ""
        return Path(value or "/var/lib/docker")

    def _container_ids(self, image: str) -> list[str]:
        result = self._run(
            ["docker", "ps", "-aq", "--filter", f"ancestor={image}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.split() if result.returncode == 0 else ["docker-query-failed"]

    @staticmethod
    def _protected_images(state: dict[str, Any]) -> set[str]:
        protected: set[str] = set()
        for lease in state["leases"].values():
            protected.update(str(image) for image in lease.get("images") or [])
        return protected

    def _trim_locked(
        self,
        state: dict[str, Any],
        *,
        protected: set[str],
        phase: str,
    ) -> dict[str, Any]:
        self._drop_stale_leases(state)
        protected.update(self._protected_images(state))
        for image, record in state["images"].items():
            record.update(self._inspect_image(image))

        docker_root = self._docker_root()
        free_before = shutil.disk_usage(docker_root).free
        present = [record for record in state["images"].values() if record.get("present")]
        report: dict[str, Any] = {
            "phase": phase,
            "mode": self.policy.mode,
            "free_bytes_before": free_before,
            "managed_present_before": len(present),
            "removed": [],
            "skipped": [],
        }
        if self.policy.mode == "keep":
            report["free_bytes_after"] = free_before
            report["managed_present_after"] = len(present)
            return report

        low_space = free_before < int(self.policy.min_free_gib * GIB)
        max_images = 0 if self.policy.mode == "remove_after_use" else self.policy.max_images
        target_free = int(self.policy.target_free_gib * GIB) if low_space else free_before
        candidates = sorted(
            (record for record in present if record["image"] not in protected),
            key=lambda item: (
                str(item.get("last_used_at_utc") or ""),
                str(item.get("first_used_at_utc") or ""),
                str(item["image"]),
            ),
        )
        present_count = len(present)
        free_now = free_before
        for record in candidates:
            over_count = present_count > max_images
            below_target = low_space and free_now < target_free
            if not over_count and not below_target:
                break
            image = str(record["image"])
            containers = self._container_ids(image)
            if containers:
                report["skipped"].append(
                    {"image": image, "reason": "container", "ids": containers}
                )
                continue
            result = self._run(
                ["docker", "image", "rm", image],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                report["skipped"].append(
                    {"image": image, "reason": "remove_failed", "detail": result.stderr.strip()}
                )
                continue
            record["present"] = False
            record["evicted_at_utc"] = _utc_now()
            record["eviction_reason"] = "low_space" if below_target else "max_images"
            report["removed"].append(image)
            state["evictions"].append(
                {
                    "image": image,
                    "at_utc": record["evicted_at_utc"],
                    "reason": record["eviction_reason"],
                    "phase": phase,
                }
            )
            state["evictions"] = state["evictions"][-500:]
            present_count -= 1
            free_now = shutil.disk_usage(docker_root).free

        report["free_bytes_after"] = free_now
        report["managed_present_after"] = present_count
        return report


def swebench_image_usage_from_runs(
    *,
    dataset_path: str | Path,
    runs_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Build image ownership evidence from LycheeMAS official prediction files."""

    import pyarrow.parquet as parquet

    table = parquet.read_table(dataset_path, columns=["instance_id", "image"])
    case_to_image = {
        str(case_id): str(image)
        for case_id, image in zip(
            table.column("instance_id").to_pylist(),
            table.column("image").to_pylist(),
            strict=True,
        )
    }
    usage: dict[str, dict[str, Any]] = {}
    pattern = "**/official_evaluation/swe_bench_verified/predictions.jsonl"
    for path in Path(runs_root).glob(pattern):
        used_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                case_id = str(json.loads(line).get("instance_id") or "")
            except json.JSONDecodeError:
                continue
            image = case_to_image.get(case_id)
            if not image:
                continue
            record = usage.setdefault(
                image,
                {
                    "source": "official_predictions",
                    "case_ids": [],
                    "prediction_files": [],
                    "last_used_at_utc": used_at,
                },
            )
            if case_id not in record["case_ids"]:
                record["case_ids"].append(case_id)
            if str(path) not in record["prediction_files"]:
                record["prediction_files"].append(str(path))
            record["last_used_at_utc"] = max(record["last_used_at_utc"], used_at)
    return usage


__all__ = [
    "DockerImageCachePolicy",
    "DockerImageLease",
    "ProjectDockerImageCache",
    "swebench_image_usage_from_runs",
]
