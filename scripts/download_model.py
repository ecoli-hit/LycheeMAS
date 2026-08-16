#!/usr/bin/env python3
"""Download a registered ModelSpec snapshot as verified real files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _safe_source(value: str) -> str:
    return value.replace("/", "--").replace("\\", "--")


def _materialize_symlinks(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        resolved = path.resolve(strict=True)
        temporary = path.with_name(path.name + ".materialized")
        shutil.copy2(resolved, temporary)
        path.unlink()
        os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model(root: Path) -> dict[str, object]:
    config = root / "config.json"
    weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in root.rglob(pattern)
        if ".cache" not in path.parts
    ]
    tokenizer = [
        root / name
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        if (root / name).is_file()
    ]
    incomplete = list(root.rglob("*.incomplete")) + list(root.rglob("*.part"))
    missing = []
    if not config.is_file():
        missing.append("config.json")
    if not weights:
        missing.append("model weights (*.safetensors or *.bin)")
    if not tokenizer:
        missing.append("tokenizer metadata")
    if incomplete:
        missing.append("unfinished download files")
    return {
        "ready": not missing,
        "missing": missing,
        "weight_files": len(weights),
        "tokenizer_files": len(tokenizer),
    }


def _manifest(
    root: Path,
    *,
    provider: str,
    source_id: str,
    revision: str | None,
    model_spec_id: str | None,
) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "model_manifest.json" or ".cache" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    value = {
        "schema_version": 2,
        "model_spec_id": model_spec_id,
        "provider": provider,
        "source_id": source_id,
        "revision": revision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": {
            "status": "ready",
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        },
    }
    path = root / "model_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _download(provider: str, source_id: str, target: Path, revision: str | None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=source_id, revision=revision, local_dir=target)
        return
    from modelscope import snapshot_download

    kwargs = {"model_id": source_id, "local_dir": str(target)}
    if revision:
        kwargs["revision"] = revision
    snapshot_download(**kwargs)


def _registered_request(args) -> tuple[Path, list[dict[str, object]]]:
    from lychee_mas.eval.studio.models import ModelRegistry

    registry = ModelRegistry(ROOT)
    entry = registry.get_spec(args.model_spec_id)
    root = Path(args.models_root).expanduser().resolve()
    target = Path(args.target_dir).expanduser().resolve() if args.target_dir else (
        root / str(entry["name"])
    )
    sources = registry.sources(args.model_spec_id, args.source)
    if not sources:
        raise SystemExit(
            f"ModelSpec {args.model_spec_id!r} has no {args.source!r} download source"
        )
    return target, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a registered ModelSpec")
    parser.add_argument("--list-models", action="store_true", help="List registered ModelSpecs")
    parser.add_argument("--model-spec-id", default=None, help="ID from the ModelSpec registry")
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="Registered source selection; auto follows catalog priority",
    )
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        from lychee_mas.eval.studio.models import ModelRegistry

        print("MODEL SPEC ID\tDISPLAY NAME\tPARAMS\tPROVIDERS\tGATED")
        for item in ModelRegistry(ROOT).specs():
            providers = ",".join(item.get("sources", {}))
            print(
                f"{item['id']}\t{item['name']}\t"
                f"{item.get('parameter_size', '-')}\t{providers}\t"
                f"{'yes' if item.get('gated') else 'no'}"
            )
        return

    if not args.model_spec_id:
        raise SystemExit("registered model download requires --model-spec-id")
    target, sources = _registered_request(args)
    current = _validate_model(target)
    if current["ready"] and not args.force:
        print(f"[model] already ready at {target}; use --force to replace", flush=True)
        return

    staging_root = target.parent / ".downloads" / target.name
    errors: list[str] = []
    for source in sources:
        provider = str(source["provider"])
        source_id = str(source["id"])
        revision = str(source.get("revision") or args.revision or "") or None
        staging = staging_root / provider / _safe_source(source_id)
        print(
            f"[model] model_spec_id={args.model_spec_id} provider={provider} "
            f"source_id={source_id} staging={staging} target={target}",
            flush=True,
        )
        try:
            _download(provider, source_id, staging, revision)
            _materialize_symlinks(staging)
            status = _validate_model(staging)
            if not status["ready"]:
                raise RuntimeError("incomplete model snapshot: " + ", ".join(status["missing"]))
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(target))
            _manifest(
                target,
                provider=provider,
                source_id=source_id,
                revision=revision,
                model_spec_id=args.model_spec_id,
            )
            print(f"[progress] percent=100 model={target.name}", flush=True)
            print(f"[model] ready at {target}", flush=True)
            return
        except Exception as exc:
            message = f"{provider}:{source_id}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[model] source failed: {message}", flush=True)
    raise SystemExit("all model sources failed:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
