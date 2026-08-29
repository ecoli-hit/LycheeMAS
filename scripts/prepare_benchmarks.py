"""Prepare benchmark data and runtime resources for LycheeMAS benchmarks.

This command downloads project-owned benchmark data. It does not copy from other
users' workspaces. It also prepares benchmark-owned runtime resources such as
benchmark Docker images. Use --source modelscope, --source huggingface, or
--source github to force one registered provider; the default auto mode uses
each benchmark's preferred provider order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lychee_mas.runtime.adapters.infrastructure.docker import (  # noqa: E402
    DOCKER_BUILD_ORDER,
    LOCAL_SANDBOX_SPECS,
    SANDBOX_SCHEMA_VERSION,
    SandboxVerificationError,
    docker_targets_for_prepare_targets,
    inspect_image,
    source_fingerprint,
    verify_local_sandbox,
)

DOCKER_IMAGE_SPECS = LOCAL_SANDBOX_SPECS


def _parse_prepare_targets(
    raw: list[str] | None, preparers: dict, *, all_full: bool, full_targets
) -> list[str]:
    if all_full:
        return list(full_targets)
    if not raw:
        return sorted(preparers)
    targets: list[str] = []
    for item in raw:
        targets.extend(part.strip() for part in item.split(",") if part.strip())
    return targets


def _print_names(names) -> None:
    print("\n".join(sorted(names)))


def _print_structure(rows, aliases: dict[str, str]) -> None:
    for row in rows:
        print(f"Benchmark source: {row['benchmark_source']}")
        print(f"  full prepare target: {row['full_prepare_target']}")
        print("  prepare target -> runnable task(s) -> kind/scorer(s)")
        for mapping in row.get("mappings", []):
            prepare_target = mapping["prepare_target"]
            actual = aliases.get(prepare_target)
            alias_suffix = f" (alias -> {actual})" if actual else ""
            tasks = ", ".join(mapping.get("runnable_tasks") or ["-"])
            kinds = ", ".join(mapping.get("kinds") or ["-"])
            note = mapping.get("note")
            note_suffix = f" [{note}]" if note else ""
            print(f"    {prepare_target}{alias_suffix} -> {tasks} -> {kinds}{note_suffix}")
        print()


def _format_list(values) -> str:
    return ", ".join(str(value) for value in values if value) or "-"


def _print_download_sources(benchmarks) -> None:
    for benchmark in benchmarks:
        name = benchmark.id
        entry = benchmark.sources
        print(f"Benchmark source: {benchmark.name} ({name})")
        for provider in ("modelscope", "huggingface", "github"):
            section = entry.get(provider, {})
            env_name = section.get("env") or "-"
            defaults = _format_list(section.get("default_ids") or [])
            revision = section.get("revision") or "provider default"
            print(f"  {provider}: env={env_name}; default_ids={defaults}; revision={revision}")
        others = entry.get("other_defaults") or []
        if others:
            print("  other defaults:")
            for item in others:
                provider = item.get("provider", "-")
                source_id = item.get("id", "-")
                purpose = item.get("purpose", "-")
                revision = item.get("revision") or "provider default"
                selectable = "download" if item.get("selectable", False) else "reference"
                print(f"    {provider}: {source_id} (revision={revision}; {selectable}; {purpose})")
        else:
            print("  other defaults: -")
        fallbacks = entry.get("fallback_files") or []
        if fallbacks:
            print("  direct-file fallback:")
            for item in fallbacks:
                files = _format_list(item.get("files") or item.get("patterns") or [])
                strict = item.get("strict", "-")
                provider = item.get("provider", "-")
                purpose = item.get("purpose", "-")
                print(f"    {provider}: {files}; strict={strict}; {purpose}")
        else:
            print("  direct-file fallback: -")
        print()


def _expand_docker_targets(targets: set[str]) -> list[str]:
    expanded = set(targets)
    pending = list(targets)
    while pending:
        target = pending.pop()
        for dep in DOCKER_IMAGE_SPECS[target]["deps"]:
            if dep not in expanded:
                expanded.add(dep)
                pending.append(dep)
    return [name for name in DOCKER_BUILD_ORDER if name in expanded]


def _build_docker_images(
    targets: set[str],
    *,
    apt_mirror: str | None,
    pip_index_url: str,
    build_proxy_url: str | None,
    no_cache: bool,
    pull_base: bool,
    force: bool,
) -> None:
    selected = _expand_docker_targets(targets)
    rebuilt: set[str] = set()
    for name in selected:
        spec = DOCKER_IMAGE_SPECS[name]
        image = spec["image"]
        dockerfile = spec["dockerfile"]
        fingerprint = source_fingerprint(_ROOT, spec)
        try:
            existing = inspect_image(image)
        except SandboxVerificationError as exc:
            raise SystemExit(f"{exc}. Install Docker or use --docker-images never.") from exc
        current = bool(
            existing
            and existing.get("profile") == spec["profile"]
            and existing.get("schema_version") == SANDBOX_SCHEMA_VERSION
            and existing.get("fingerprint") == fingerprint
        )
        refresh_base = pull_base and not spec["deps"]
        dependency_changed = any(dep in rebuilt for dep in spec["deps"])
        if not force and not refresh_base and not dependency_changed and current:
            print(
                f"[prepare:docker] {image} current "
                f"profile={spec['profile']} fingerprint={fingerprint[:12]}",
                flush=True,
            )
            continue
        reason = (
            "forced"
            if force
            else "pull_base"
            if refresh_base
            else "dependency_changed"
            if dependency_changed
            else "missing"
            if existing is None
            else "source_changed"
        )

        cmd = ["docker", "build", "--network=host"]
        if no_cache:
            cmd.append("--no-cache")
        if build_proxy_url:
            for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                cmd.extend(["--build-arg", f"{proxy_name}={build_proxy_url}"])
            cmd.extend(["--build-arg", "NO_PROXY=127.0.0.1,localhost,::1"])
            cmd.extend(["--build-arg", "no_proxy=127.0.0.1,localhost,::1"])
        if refresh_base:
            cmd.append("--pull")
        if name in {
            "python_sandbox",
            "agbench_base",
            "agbench_gaia",
        }:
            cmd.extend(["--build-arg", f"PIP_INDEX_URL={pip_index_url}"])
        if name == "python_sandbox" and apt_mirror:
            cmd.extend(["--build-arg", f"APT_MIRROR={apt_mirror}"])
        cmd.extend(["--build-arg", f"LYCHEE_SANDBOX_FINGERPRINT={fingerprint}"])
        cmd.extend(["-f", dockerfile, "-t", image, "."])

        print(
            f"[prepare:docker] building {image} from {dockerfile} "
            f"reason={reason} fingerprint={fingerprint[:12]}",
            flush=True,
        )
        try:
            subprocess.run(cmd, cwd=_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"[prepare:docker] failed to build {image}") from exc
        rebuilt.add(name)

    for name in selected:
        image = DOCKER_IMAGE_SPECS[name]["image"]
        try:
            report = verify_local_sandbox(image, repo_root=_ROOT)
        except SandboxVerificationError as exc:
            raise SystemExit(f"[prepare:docker] {exc}") from exc
        capabilities = report.get("capabilities") or {}
        print(
            f"[prepare:docker] {image} ready profile={report.get('profile')} "
            f"fingerprint={str(report.get('fingerprint') or '')[:12]} "
            f"capabilities={capabilities.get('status', 'not_applicable')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LycheeMAS benchmark data and runtime resources"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help=(
            "Prepare target names, comma-separated or space-separated. "
            "Use --list-prepare-targets to inspect accepted names."
        ),
    )
    parser.add_argument(
        "--all-full-benchmarks",
        action="store_true",
        help="Prepare one full/canonical prepare target for every benchmark source.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Set LYCHEE_BENCHMARK_RAW_ROOT for upstream/raw benchmark files",
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="Set LYCHEE_BENCHMARK_PREPARED_ROOT for loader-ready benchmark files",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface", "github"),
        default="auto",
        help=(
            "Download provider. auto uses benchmark-specific order: confirmed "
            "ModelScope mirrors first when registered, otherwise HuggingFace or official GitHub."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if target exists")
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help=(
            "Build Prepared only from an existing managed Raw source or a path override; "
            "disable downloads."
        ),
    )
    parser.add_argument(
        "--manifest-hash-mode",
        choices=("full", "metadata"),
        default="full",
        help="Manifest integrity mode: full writes SHA-256; metadata records size/mtime only.",
    )
    parser.add_argument(
        "--verify-manifests",
        action="store_true",
        help="Verify existing manifests for selected targets and exit without downloading.",
    )
    parser.add_argument(
        "--docker-images",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Prepare benchmark Docker images. auto builds images required by selected targets "
            "(human_eval/gaia); GAIA prepares the AgBench-aligned profile; "
            "always builds every registered benchmark image; never skips Docker."
        ),
    )
    parser.add_argument(
        "--force-docker-images",
        action="store_true",
        help="Rebuild selected benchmark Docker images even if the local image already exists.",
    )
    parser.add_argument(
        "--docker-apt-mirror",
        default=os.environ.get("LYCHEE_DOCKER_APT_MIRROR"),
        help="Ubuntu apt mirror used while building benchmark Docker images.",
    )
    parser.add_argument(
        "--docker-pip-index-url",
        default=os.environ.get(
            "LYCHEE_DOCKER_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ),
        help="Python package index used while building benchmark Docker images.",
    )
    parser.add_argument(
        "--docker-build-proxy-url",
        default=os.environ.get("LYCHEE_DOCKER_BUILD_PROXY_URL"),
        help=(
            "Optional HTTP proxy used only by Docker build RUN steps, for example "
            "http://127.0.0.1:7897 with --network=host."
        ),
    )
    parser.add_argument(
        "--docker-no-cache",
        action="store_true",
        help="Build benchmark Docker images without Docker layer cache.",
    )
    parser.add_argument(
        "--docker-pull-base",
        action="store_true",
        help="Ask Docker to refresh every root base image and rebuild its dependent images.",
    )
    parser.add_argument(
        "--list-benchmark-sources",
        action="store_true",
        help="List canonical benchmark sources and exit",
    )
    parser.add_argument(
        "--list-prepare-targets",
        action="store_true",
        help="List every name accepted by --tasks, including source names and aliases, and exit",
    )
    parser.add_argument(
        "--list-runnable-tasks",
        action="store_true",
        help="List runnable benchmark tasks accepted by run_mas.py --task and exit",
    )
    parser.add_argument(
        "--list-benchmark-structure",
        action="store_true",
        help=(
            "Show benchmark source with prepare target -> runnable task(s) -> "
            "kind/scorer mapping and exit"
        ),
    )
    parser.add_argument(
        "--list-download-sources",
        action="store_true",
        help=(
            "Show ModelScope/HuggingFace/GitHub source candidates, other defaults, "
            "and direct-file fallback rules"
        ),
    )
    args = parser.parse_args()

    if args.raw_root:
        os.environ["LYCHEE_BENCHMARK_RAW_ROOT"] = args.raw_root
    if args.prepared_root:
        os.environ["LYCHEE_BENCHMARK_PREPARED_ROOT"] = args.prepared_root
    os.environ["LYCHEE_DATA_SOURCE"] = args.source
    if args.conversion_only:
        os.environ.update(
            LYCHEE_BENCHMARK_CONVERSION_ONLY="1",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
    if args.all_full_benchmarks:
        os.environ.setdefault("LYCHEE_MAST_DOWNLOAD_FULL", "1")

    from lychee_mas.eval.benchmarks import (
        BENCHMARK_STRUCTURE,
        BENCHMARKS,
        FULL_PREPARE_TARGETS,
        LOADERS,
        PREPARE_ALIASES,
        PREPARERS,
        prepare,
    )
    from lychee_mas.eval.benchmarks.common import prepared_root, raw_root
    from lychee_mas.eval.benchmarks.manifest import (
        benchmark_key_for_target,
        verify_prepared_manifest,
        write_prepared_manifest,
    )

    if args.list_benchmark_sources:
        _print_names(row["benchmark_source"] for row in BENCHMARK_STRUCTURE)
        return
    if args.list_prepare_targets:
        _print_names(PREPARERS)
        return
    if args.list_runnable_tasks:
        _print_names(LOADERS)
        return
    if args.list_benchmark_structure:
        _print_structure(BENCHMARK_STRUCTURE, PREPARE_ALIASES)
        return
    if args.list_download_sources:
        _print_download_sources(BENCHMARKS.all())
        return

    targets = _parse_prepare_targets(
        args.tasks, PREPARERS, all_full=args.all_full_benchmarks, full_targets=FULL_PREPARE_TARGETS
    )

    if args.verify_manifests:
        failures = []
        for target in targets:
            key = benchmark_key_for_target(PREPARE_ALIASES.get(target, target))
            manifest_path = os.path.join(prepared_root(), key, "manifest.json")
            if not os.path.isfile(manifest_path):
                failures.append({"target": target, "status": "missing_manifest"})
                print(f"[manifest] {target}: missing {manifest_path}", flush=True)
                continue
            result = verify_prepared_manifest(
                manifest_path,
                verify_hashes=args.manifest_hash_mode == "full",
            )
            print(f"[manifest] {target}: {result['status']} ({manifest_path})", flush=True)
            if result["status"] != "ready":
                failures.append({"target": target, **result})
        if failures:
            raise SystemExit(f"manifest verification failed: {failures}")
        return

    print(f"[prepare] raw_root={raw_root()}")
    print(f"[prepare] prepared_root={prepared_root()}")
    print(f"[prepare] source={args.source}")
    if args.all_full_benchmarks:
        print(f"[prepare] all_full_benchmarks={','.join(FULL_PREPARE_TARGETS)}")
    for target in targets:
        print(f"[prepare] {target} ...", flush=True)
        if args.conversion_only:
            key = benchmark_key_for_target(PREPARE_ALIASES.get(target, target))
            managed_raw = os.path.join(raw_root(), key)
            try:
                overrides = json.loads(os.environ.get("LYCHEE_BENCHMARK_RAW_OVERRIDES", "[]"))
            except ValueError:
                overrides = []
            overridden = [
                str(item.get("path"))
                for item in overrides
                if isinstance(item, dict) and item.get("benchmark_key") == key and item.get("path")
            ]
            available = [path for path in [managed_raw, *overridden] if os.path.exists(path)]
            if not available:
                raise SystemExit(
                    f"[prepare] conversion-only requires existing Raw for {target}; "
                    f"checked {managed_raw} and this run's Raw path overrides"
                )
            print(f"[prepare] conversion_only=true raw={available[0]}", flush=True)
        try:
            location = prepare(target, force=args.force, source=args.source)
        except Exception as exc:
            raise SystemExit(f"[prepare] failed for {target}: {exc}") from exc
        print(f"[prepare] {target} ready at {location}", flush=True)
        manifest_path = write_prepared_manifest(
            PREPARE_ALIASES.get(target, target),
            location,
            source_mode=args.source,
            hash_files=args.manifest_hash_mode == "full",
        )
        print(f"[manifest] {target} ready at {manifest_path}", flush=True)

    if args.conversion_only or args.docker_images == "never":
        docker_targets: set[str] = set()
    elif args.docker_images == "always":
        docker_targets = {"human_eval", "agbench_gaia"}
    else:
        docker_targets = docker_targets_for_prepare_targets(targets, PREPARE_ALIASES)
    if docker_targets:
        selected = ",".join(_expand_docker_targets(docker_targets))
        print(f"[prepare:docker] selected={selected}", flush=True)
        _build_docker_images(
            docker_targets,
            apt_mirror=args.docker_apt_mirror,
            pip_index_url=args.docker_pip_index_url,
            build_proxy_url=args.docker_build_proxy_url,
            no_cache=args.docker_no_cache,
            pull_base=args.docker_pull_base,
            force=args.force_docker_images,
        )


if __name__ == "__main__":
    main()
