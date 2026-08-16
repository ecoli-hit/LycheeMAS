"""Build and runtime contracts for project-owned benchmark Docker images."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

SANDBOX_SCHEMA_VERSION = "1"
PROFILE_LABEL = "org.lychee-mas.sandbox.profile"
SCHEMA_LABEL = "org.lychee-mas.sandbox.schema"
FINGERPRINT_LABEL = "org.lychee-mas.sandbox.fingerprint"

LOCAL_SANDBOX_SPECS: dict[str, dict[str, Any]] = {
    "python_sandbox": {
        "image": "lychee-python-sandbox:local",
        "profile": "python_sandbox",
        "dockerfile": "docker/python-sandbox.Dockerfile",
        "deps": [],
        "fingerprint_files": ["docker/python-sandbox.Dockerfile"],
    },
    "human_eval": {
        "image": "lychee-human-eval:local",
        "profile": "human_eval",
        "dockerfile": "docker/human_eval.Dockerfile",
        "deps": ["python_sandbox"],
        "fingerprint_files": [
            "docker/python-sandbox.Dockerfile",
            "docker/human_eval.Dockerfile",
        ],
    },
    "agbench_base": {
        "image": "lychee-agbench-base:local",
        "profile": "agbench_base",
        "dockerfile": "docker/agbench-base.Dockerfile",
        "deps": [],
        "fingerprint_files": [
            "docker/agbench-base.Dockerfile",
            "docker/agbench-base.requirements.txt",
            "docker/agbench-base-capabilities.json",
            "docker/verify_benchmark_sandbox.py",
            "src/autogen/python/packages/agbench/src/agbench/res/Dockerfile",
        ],
        "verifier": "/opt/lychee-sandbox/verify_benchmark_sandbox.py",
    },
    "agbench_gaia": {
        "image": "lychee-agbench-gaia:local",
        "profile": "agbench_gaia",
        "dockerfile": "docker/agbench-gaia.Dockerfile",
        "deps": ["agbench_base"],
        "fingerprint_files": [
            "docker/agbench-base.Dockerfile",
            "docker/agbench-base.requirements.txt",
            "docker/agbench-base-capabilities.json",
            "docker/agbench-gaia.Dockerfile",
            "docker/agbench-gaia.requirements.txt",
            "docker/agbench-gaia-capabilities.json",
            "docker/verify_benchmark_sandbox.py",
            "src/autogen/python/packages/agbench/src/agbench/res/Dockerfile",
            "src/autogen/python/packages/agbench/benchmarks/GAIA/Templates/MagenticOne/requirements.txt",
            "src/autogen/python/packages/autogen-core/pyproject.toml",
            "src/autogen/python/packages/autogen-core/README.md",
            "src/autogen/python/packages/autogen-core/src",
            "src/autogen/python/packages/autogen-ext/pyproject.toml",
            "src/autogen/python/packages/autogen-ext/README.md",
            "src/autogen/python/packages/autogen-ext/src",
            "src/autogen/python/packages/autogen-agentchat/pyproject.toml",
            "src/autogen/python/packages/autogen-agentchat/README.md",
            "src/autogen/python/packages/autogen-agentchat/src",
        ],
        "verifier": "/opt/lychee-sandbox/verify_benchmark_sandbox.py",
    },
}

DOCKER_BUILD_ORDER = (
    "python_sandbox",
    "human_eval",
    "agbench_base",
    "agbench_gaia",
)
_SPEC_BY_IMAGE = {str(spec["image"]): spec for spec in LOCAL_SANDBOX_SPECS.values()}


class SandboxVerificationError(RuntimeError):
    """Raised when a local sandbox is absent, stale, or missing capabilities."""


def source_fingerprint(repo_root: str | Path, spec: dict[str, Any]) -> str:
    """Hash every source artifact that defines one sandbox image."""

    root = Path(repo_root)
    digest = hashlib.sha256()
    for relative in sorted(str(item) for item in spec.get("fingerprint_files") or []):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"sandbox fingerprint source is missing: {path}")
        files = (
            [path]
            if path.is_file()
            else sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix not in {".pyc", ".pyo"}
            )
        )
        for source in files:
            source_relative = source.relative_to(root).as_posix()
            digest.update(source_relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def inspect_image(image: str) -> dict[str, Any] | None:
    """Return normalized Docker image identity and labels, or ``None`` if absent."""

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SandboxVerificationError("Docker CLI was not found") from exc
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        record = payload[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SandboxVerificationError(f"Docker returned invalid inspect data for {image}") from exc
    labels = dict((record.get("Config") or {}).get("Labels") or {})
    return {
        "image": image,
        "image_id": record.get("Id"),
        "profile": labels.get(PROFILE_LABEL),
        "schema_version": labels.get(SCHEMA_LABEL),
        "fingerprint": labels.get(FINGERPRINT_LABEL),
        "labels": labels,
    }


def _run_capability_verifier(image: str, verifier: str) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        image,
        "python",
        verifier,
        "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout.strip()
    try:
        report = json.loads(output) if output else {}
    except json.JSONDecodeError:
        report = {"status": "invalid_output", "stdout": output}
    if result.returncode != 0 or report.get("status") != "ready":
        detail = report or result.stderr.strip() or f"exit code {result.returncode}"
        raise SandboxVerificationError(
            f"sandbox capability verification failed for {image}: {detail}"
        )
    return report


def verify_local_sandbox(
    image: str,
    *,
    repo_root: str | Path,
    run_capability_check: bool = True,
) -> dict[str, Any]:
    """Verify existence, source fingerprint, profile labels, and declared capabilities."""

    metadata = inspect_image(image)
    if metadata is None:
        raise SandboxVerificationError(f"local Docker image {image!r} was not found")

    spec = _SPEC_BY_IMAGE.get(image)
    if spec is None:
        return metadata

    expected_fingerprint = source_fingerprint(repo_root, spec)
    mismatches = {}
    expected = {
        "profile": str(spec["profile"]),
        "schema_version": SANDBOX_SCHEMA_VERSION,
        "fingerprint": expected_fingerprint,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": metadata.get(key),
            }
    if mismatches:
        raise SandboxVerificationError(
            f"local Docker image {image!r} is stale or incompatible: {mismatches}"
        )

    verifier = spec.get("verifier")
    if verifier and run_capability_check:
        metadata["capabilities"] = _run_capability_verifier(image, str(verifier))
    return metadata
