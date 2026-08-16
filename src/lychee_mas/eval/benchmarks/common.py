"""Shared helpers for LycheeMAS benchmark loaders.

The individual benchmark modules own their dataset-specific choices. This
module only keeps the common mechanics: data roots, provider order, provider
cache/Git checkout helpers, parquet IO, and multiple-choice formatting.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import STANDARD_SOURCE_PROVIDERS

DATA_BACKENDS = STANDARD_SOURCE_PROVIDERS

DEFAULT_RAW_ROOT = os.path.join("data", "benchmarks", "raw")
DEFAULT_PREPARED_ROOT = os.path.join("data", "benchmarks", "prepared")
DEFAULT_RUNS_ROOT = os.path.join("runs", "benchmarks")


def raw_root() -> str:
    return (
        os.environ.get("LYCHEE_BENCHMARK_RAW_ROOT")
        or os.environ.get("CDM_DATA_ROOT")
        or DEFAULT_RAW_ROOT
    )


def prepared_root() -> str:
    return (
        os.environ.get("LYCHEE_BENCHMARK_PREPARED_ROOT")
        or os.environ.get("CDM_DATA_ROOT")
        or DEFAULT_PREPARED_ROOT
    )


def processed_root() -> str:
    """Backward-compatible alias for the prepared benchmark cache root."""
    return prepared_root()


def runs_root() -> str:
    return os.environ.get("LYCHEE_BENCHMARK_RUNS_ROOT", DEFAULT_RUNS_ROOT)


def safe_source_id(identifier: str) -> str:
    """Make a provider source id safe as one local directory name."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "--", str(identifier).strip()).strip("._-")
    return value or "source"


def raw_source_dir(benchmark: str, provider: str, identifier: str) -> Path:
    try:
        overrides = json.loads(os.environ.get("LYCHEE_BENCHMARK_RAW_OVERRIDES", "[]"))
    except ValueError:
        overrides = []
    matching = [
        item
        for item in overrides
        if isinstance(overrides, list)
        if item.get("benchmark_key") == benchmark and item.get("path")
    ]
    for item in matching:
        if item.get("provider") == provider and item.get("source_id") == identifier:
            return Path(str(item["path"])).expanduser().resolve()
    if len(matching) == 1:
        return Path(str(matching[0]["path"])).expanduser().resolve()
    return Path(raw_root()) / benchmark / provider / safe_source_id(identifier)


def conversion_only() -> bool:
    return os.environ.get("LYCHEE_BENCHMARK_CONVERSION_ONLY", "").lower() in {
        "1",
        "true",
        "yes",
    }


def load_local_dataset_split(source: str | os.PathLike, split: str):
    """Load one split from a downloaded dataset tree without provider network calls."""
    from datasets import load_dataset

    root = Path(source)
    patterns = {
        "parquet": ("*.parquet",),
        "json": ("*.json", "*.jsonl"),
        "csv": ("*.csv",),
    }
    for builder, suffixes in patterns.items():
        files = [path for suffix in suffixes for path in root.rglob(suffix)]
        split_files = [path for path in files if split.lower() in path.name.lower()]
        selected = split_files or files
        if selected:
            return load_dataset(
                builder,
                data_files={split: [str(path) for path in selected]},
                split=split,
            )
    raise FileNotFoundError(
        "conversion-only could not find parquet/json/jsonl/csv files "
        f"for split={split} under {root}"
    )


def prepared_benchmark_dir(benchmark: str) -> Path:
    try:
        overrides = json.loads(os.environ.get("LYCHEE_BENCHMARK_PREPARED_OVERRIDES", "{}"))
    except ValueError:
        overrides = {}
    override = overrides.get(benchmark) if isinstance(overrides, dict) else None
    if override:
        return Path(str(override)).expanduser().resolve()
    return Path(prepared_root()) / benchmark


def remove_path(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)


def copy_raw_to_prepared(raw_dir: str | os.PathLike, prepared_dir: str | os.PathLike) -> Path:
    """Copy raw source data into prepared so loaders always read real files."""
    raw_path = Path(raw_dir).resolve()
    prepared_path = Path(prepared_dir)
    if prepared_path.exists():
        remove_path(prepared_path)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.is_dir():
        shutil.copytree(raw_path, prepared_path)
    else:
        shutil.copy2(raw_path, prepared_path)
    try:
        relative = raw_path.relative_to(Path(raw_root()).resolve())
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) >= 3:
        benchmark, provider, source_id = relative.parts[:3]
        metadata_path = (
            prepared_path / ".lychee_source.json"
            if prepared_path.is_dir()
            else Path(f"{prepared_path}.lychee_source.json")
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark": benchmark,
                    "provider": provider,
                    "source_id": source_id.replace("--", "/", 1),
                    "raw_path": str(raw_path),
                    "copied_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return prepared_path


def clone_git_source(
    benchmark: str,
    repo_url: str,
    destination: str | os.PathLike,
    *,
    revision: str,
    force: bool = False,
) -> Path:
    """Materialize one pinned official Git repository in the Raw tree.

    A temporary sibling directory prevents an interrupted clone from looking
    complete. Existing clones are reused only when their HEAD matches the
    requested revision; benchmark preparation never performs an implicit pull.
    """

    destination = Path(destination)
    if destination.is_dir() and not force:
        result = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == revision:
            return destination
    if conversion_only():
        raise FileNotFoundError(
            f"conversion-only requires the pinned source {revision} at {destination}"
        )

    temporary = destination.with_name(f".{destination.name}.clone-part")
    remove_path(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_download_source(
        benchmark,
        "github",
        repo_url,
        temporary,
        revision=revision,
    )
    try:
        subprocess.run(["git", "init", str(temporary)], check=True)
        subprocess.run(
            ["git", "-C", str(temporary), "remote", "add", "origin", repo_url],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary),
                "fetch",
                "--depth",
                "1",
                "origin",
                revision,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
        remove_path(destination)
        os.replace(temporary, destination)
    except Exception:
        remove_path(temporary)
        raise

    metadata_path = destination / ".lychee_source.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": benchmark,
                "provider": "github",
                "source_id": repo_url,
                "revision": revision,
                "destination": str(destination),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def raw_source_candidates(
    benchmark: str,
    providers: Sequence[str] = DATA_BACKENDS,
) -> list[tuple[str, str, Path]]:
    """Return raw cache locations for configured provider ids in preferred order."""
    from .registry import BENCHMARKS

    candidates: list[tuple[str, str, Path]] = []
    benchmark_impl = BENCHMARKS.get(benchmark)
    for provider in providers:
        for identifier in benchmark_impl.provider_ids(provider):
            candidates.append(
                (provider, identifier, raw_source_dir(benchmark, provider, identifier))
            )
    return candidates


def restore_prepared_from_raw(
    benchmark: str,
    prepared_path: str | os.PathLike,
    candidates: Sequence[tuple[str, str, Path]],
    *,
    ready: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Restore prepared data from an existing raw source when possible.

    This intentionally copies data into prepared rather than linking it, so
    loaders always see a standalone, inspectable prepared tree.
    """
    for provider, identifier, raw_path in candidates:
        if not raw_path.exists():
            continue
        if ready is not None and not ready(raw_path):
            continue
        restored = copy_raw_to_prepared(raw_path, prepared_path)
        print(
            f"[prepare] {benchmark}: restored prepared from raw "
            f"provider={provider} id={identifier} -> {restored}",
            flush=True,
        )
        return restored
    return None


def parquet_files(
    subdir: str, split: Optional[str] = None, root: Optional[str] = None
) -> list[str]:
    relative = Path(subdir)
    base = (
        Path(root) / relative
        if root is not None
        else prepared_benchmark_dir(relative.parts[0]).joinpath(*relative.parts[1:])
    )
    if split:
        return glob.glob(os.path.join(str(base), f"{split}-*.parquet"))
    return glob.glob(os.path.join(str(base), "*.parquet"))


def has_parquet(subdir: str, split: Optional[str] = None, root: Optional[str] = None) -> bool:
    return bool(parquet_files(subdir, split=split, root=root))


def parquet_ready(
    subdir: str,
    split: str,
    *,
    required_columns: Sequence[str] = (),
    min_rows: int = 1,
    root: Optional[str] = None,
) -> bool:
    """Return True only if cached parquet files are readable and minimally shaped."""
    files = parquet_files(subdir, split=split, root=root)
    if not files:
        return False
    try:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files={split: files}, split=split)
        if min_rows and len(dataset) < min_rows:
            return False
        missing = [name for name in required_columns if name not in dataset.column_names]
        return not missing
    except Exception:
        return False


def json_file_ready(path: str | os.PathLike, *, min_rows: int = 1) -> bool:
    """Return True when a cached JSON/JSONL file can be parsed and is non-empty."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        stripped = text.lstrip()
        if not stripped:
            return False
        if stripped.startswith("["):
            data = json.loads(text)
            return isinstance(data, list) and len(data) >= min_rows
        count = 0
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
                count += 1
                if count >= min_rows:
                    return True
        return min_rows == 0
    except Exception:
        return False


def backend_order(
    source: Optional[str] = None,
    *,
    backends: Sequence[str] = DATA_BACKENDS,
    source_label: str = "data source",
) -> list[str]:
    source = (source or os.environ.get("LYCHEE_DATA_SOURCE") or "auto").lower()
    if source == "auto":
        return list(backends)
    if source not in backends:
        valid = ", ".join(("auto", *backends))
        raise ValueError(f"unknown {source_label} {source!r}; choose {valid}")
    return [source]


def log_download_source(
    benchmark: str,
    provider: str,
    identifier: str,
    destination: str | os.PathLike | None = None,
    *,
    revision: str | None = None,
) -> None:
    """Print a stable, grep-friendly line showing which provider is being used."""
    msg = f"[download] {benchmark}: provider={provider} id={identifier}"
    if destination is not None:
        msg += f" -> {destination}"
        path = Path(destination)
        metadata_path = (
            path / ".lychee_source.json"
            if path.suffix == ""
            else Path(f"{path}.lychee_source.json")
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark": benchmark,
                    "provider": provider,
                    "source_id": identifier,
                    "revision": revision,
                    "destination": str(path),
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(msg, flush=True)


def raw_source_matches(
    path: str | os.PathLike,
    *,
    provider: str,
    source_id: str,
    revision: str | None,
) -> bool:
    """Return whether a Raw source has matching provenance and real payload files."""

    root = Path(path)
    metadata_path = root / ".lychee_source.json"
    if not root.is_dir() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    if (
        metadata.get("provider") != provider
        or metadata.get("source_id") != source_id
        or metadata.get("revision") != revision
    ):
        return False
    return any(
        candidate.is_file()
        and candidate.name != ".lychee_source.json"
        and not candidate.name.endswith((".lock", ".part", ".incomplete"))
        for candidate in root.rglob("*")
    )


def load_parquet(subdir: str, split: str):
    from datasets import load_dataset

    relative = Path(subdir)
    base = prepared_benchmark_dir(relative.parts[0]).joinpath(*relative.parts[1:])
    files = parquet_files(subdir, split=split) or parquet_files(subdir)
    if not files:
        raise FileNotFoundError(f"no parquet under {base}")
    return load_dataset("parquet", data_files={split: files}, split=split)


def save_hf_dataset(
    repo_id: str,
    subset: Optional[str],
    split: str,
    out_dir: str,
    *,
    benchmark: str = "hf_dataset",
    raw_subset_dir: Optional[str] = None,
):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    src = raw_source_dir(benchmark, "huggingface", repo_id)
    if conversion_only():
        source_tree = src / raw_subset_dir if raw_subset_dir else src
        dataset = load_local_dataset_split(source_tree, split)
        _write_parquet_split(dataset, out_dir, split)
        return dataset
    log_download_source(benchmark, "huggingface", repo_id, src, revision="main")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(src))
    if raw_subset_dir:
        dataset = load_local_dataset_split(src / raw_subset_dir, split)
    else:
        dataset = (
            load_dataset(repo_id, subset, split=split)
            if subset
            else load_dataset(repo_id, split=split)
        )
    _write_parquet_split(dataset, out_dir, split)
    return dataset


def msdataset_to_hf_dataset(dataset):
    from datasets import Dataset

    if hasattr(dataset, "to_hf_dataset"):
        return dataset.to_hf_dataset()
    if hasattr(dataset, "hf_ds"):
        return dataset.hf_ds
    if hasattr(dataset, "to_dataset"):
        converted = dataset.to_dataset()
        if hasattr(converted, "to_hf_dataset"):
            return converted.to_hf_dataset()
        return converted
    if hasattr(dataset, "to_list"):
        return Dataset.from_list(dataset.to_list())
    return Dataset.from_list(list(dataset))


def load_modelscope_dataset(dataset_id: str, subset: Optional[str], split: str):
    from modelscope.msdatasets import MsDataset

    kwargs = {"split": split}
    if subset:
        kwargs["subset_name"] = subset
    dataset = MsDataset.load(dataset_id, **kwargs)
    return msdataset_to_hf_dataset(dataset)


def save_modelscope_dataset(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    out_dir: str,
    *,
    benchmark: str = "modelscope_dataset",
    raw_subset_dir: Optional[str] = None,
):
    from modelscope.hub.snapshot_download import snapshot_download

    src = raw_source_dir(benchmark, "modelscope", dataset_id)
    if conversion_only():
        source_tree = src / raw_subset_dir if raw_subset_dir else src
        dataset = load_local_dataset_split(source_tree, split)
        _write_parquet_split(dataset, out_dir, split)
        return dataset
    log_download_source(benchmark, "modelscope", dataset_id, src)
    snapshot_download(dataset_id, repo_type="dataset", local_dir=str(src))
    dataset = (
        load_local_dataset_split(src / raw_subset_dir, split)
        if raw_subset_dir
        else load_modelscope_dataset(dataset_id, subset, split)
    )
    _write_parquet_split(dataset, out_dir, split)
    return dataset


def _write_parquet_split(dataset, out_dir: str, split: str) -> None:
    """Atomically replace one prepared split and remove stale shards."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / f"{split}-00000-of-00001.parquet"
    temporary_path = destination / f".{split}-00000-of-00001.tmp.parquet"
    if temporary_path.exists():
        temporary_path.unlink()
    dataset.to_parquet(str(temporary_path))
    for old_path in destination.glob(f"{split}-*.parquet"):
        old_path.unlink()
    temporary_path.replace(final_path)


def hf_resolve_url(repo_id: str, path: str, revision: str = "main") -> str:
    quoted_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_path}"


def download_url(url: str, dest: str, *, timeout: int = 120) -> str:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    resume_from = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    try:
        request = urllib.request.Request(url)
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        response = urllib.request.urlopen(request, timeout=timeout)
        mode = "ab" if resume_from and getattr(response, "status", None) == 206 else "wb"
        if mode == "wb":
            resume_from = 0
        with response, open(tmp, mode) as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        # Keep the .part file so the next attempt can resume if the server
        # supports HTTP Range requests.
        raise
    return dest


def download_hf_files(
    repo_id: str, files: Sequence[str], out_dir: str, *, revision: str = "main"
) -> list[str]:
    log_download_source("hf_files", "huggingface", repo_id, out_dir, revision=revision)
    downloaded: list[str] = []
    errors: list[str] = []
    for path in files:
        dest = os.path.join(out_dir, path)
        try:
            downloaded.append(download_url(hf_resolve_url(repo_id, path, revision=revision), dest))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError("failed to download HuggingFace files; " + " | ".join(errors))
    return downloaded


def list_hf_dataset_files(repo_id: str, *, revision: str = "main") -> list[str]:
    try:
        from huggingface_hub import HfApi

        return list(
            HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        )
    except Exception:
        pass
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    files: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("type") == "file" and item.get("path"):
                files.append(str(item["path"]))
    return files


def format_choices(stem: str, texts, labels) -> str:
    lines = [f"{lab}. {txt}" for lab, txt in zip(labels, texts)]
    return stem.strip() + "\n" + "\n".join(lines) + "\nAnswer with the option letter."
