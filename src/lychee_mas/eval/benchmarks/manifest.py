"""Prepared benchmark provenance and integrity manifests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import prepared_root, raw_root


def benchmark_key_for_target(target: str) -> str:
    prefixes = {
        "gaia": "gaia",
        "agent_collab": "agent_collab",
        "aftraj": "aftraj",
        "mast": "mast_data",
    }
    for prefix, key in prefixes.items():
        if target == prefix or target.startswith(f"{prefix}_"):
            return key
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_count(path: Path) -> int | None:
    name = path.name.lower()
    try:
        if name.endswith(".jsonl"):
            with path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        if name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        if name.endswith(".json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            return len(value) if isinstance(value, list) else None
        if name.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                return None
    except Exception:
        return None
    return None


def _files(root: Path, *, hash_files: bool) -> list[dict[str, Any]]:
    if root.is_file():
        candidates = [root]
        base = root.parent
    else:
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    records = []
    for path in candidates:
        relative = path.relative_to(base)
        if (
            ".cache" in relative.parts
            or path.name == "manifest.json"
            or path.name == ".lychee_source.json"
        ):
            continue
        stat = path.stat()
        record = {
            "path": str(relative),
            "size_bytes": stat.st_size,
            "mtime_unix_s": round(stat.st_mtime, 6),
            "record_count": _record_count(path),
        }
        if hash_files:
            record["sha256"] = _sha256(path)
        records.append(record)
    return records


def _source_entries(benchmark_key: str) -> list[dict[str, Any]]:
    root = Path(raw_root()) / benchmark_key
    if not root.exists():
        return []
    entries = []
    for provider_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for source_dir in sorted(path for path in provider_dir.iterdir() if path.is_dir()):
            metadata_path = source_dir / ".lychee_source.json"
            metadata = {}
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}
            entries.append(
                {
                    "provider": metadata.get("provider") or provider_dir.name,
                    "source_id": metadata.get("source_id") or source_dir.name.replace("--", "/", 1),
                    "revision": metadata.get("revision"),
                    "recorded_at_utc": metadata.get("recorded_at_utc"),
                    "raw_path": str(source_dir),
                    "exists": source_dir.exists(),
                }
            )
    return entries


def write_prepared_manifest(
    target: str,
    location: str | os.PathLike,
    *,
    source_mode: str,
    hash_files: bool = True,
) -> Path:
    """Write one canonical manifest next to the prepared benchmark tree."""
    benchmark_key = benchmark_key_for_target(target)
    prepared_location = Path(location)
    benchmark_dir = Path(prepared_root()) / benchmark_key
    if not benchmark_dir.exists():
        benchmark_dir = (
            prepared_location if prepared_location.is_dir() else prepared_location.parent
        )
    files = _files(prepared_location, hash_files=hash_files)
    source_entries = _source_entries(benchmark_key)
    manifest = {
        "schema_version": 1,
        "benchmark_key": benchmark_key,
        "prepare_target": target,
        "source_mode": source_mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(Path(raw_root())),
        "prepared_root": str(Path(prepared_root())),
        "prepared_location": str(prepared_location),
        "integrity": {
            "status": "ready" if prepared_location.exists() and files else "incomplete",
            "hash_algorithm": "sha256" if hash_files else None,
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "known_record_count": sum(
                item["record_count"] for item in files if item["record_count"] is not None
            ),
        },
        "raw_sources": source_entries,
        "prepared_files": files,
    }
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = benchmark_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def verify_prepared_manifest(
    path: str | os.PathLike, *, verify_hashes: bool = True
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_location = Path(manifest["prepared_location"])
    base = prepared_location if prepared_location.is_dir() else prepared_location.parent
    missing = []
    changed = []
    for record in manifest.get("prepared_files", []):
        file_path = base / record["path"]
        if not file_path.is_file():
            missing.append(record["path"])
            continue
        if file_path.stat().st_size != int(record["size_bytes"]):
            changed.append(record["path"])
            continue
        if verify_hashes and record.get("sha256") and _sha256(file_path) != record["sha256"]:
            changed.append(record["path"])
    declared_status = manifest.get("integrity", {}).get("status")
    return {
        "manifest": str(manifest_path),
        "status": (
            "ready"
            if declared_status == "ready" and not missing and not changed
            else "invalid"
        ),
        "missing_files": missing,
        "changed_files": changed,
        "raw_sources_available": [
            entry for entry in manifest.get("raw_sources", []) if Path(entry["raw_path"]).exists()
        ],
    }
