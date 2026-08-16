"""Verify a benchmark sandbox capability contract using only the standard library."""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import os
import shutil
import site
import sys
import tempfile
from pathlib import Path

DEFAULT_MANIFEST = "/opt/lychee-sandbox/capabilities.json"


def verify(manifest_path: str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    missing_commands = [name for name in manifest["commands"] if shutil.which(name) is None]
    missing_modules = [
        name for name in manifest["python_modules"] if importlib.util.find_spec(name) is None
    ]
    distribution_versions = {}
    distribution_mismatches = {}
    for name, expected_version in manifest.get("python_distributions", {}).items():
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        distribution_versions[name] = actual_version
        if actual_version != expected_version:
            distribution_mismatches[name] = {
                "expected": expected_version,
                "actual": actual_version,
            }
    unexpected_distributions = []
    for name in manifest.get("absent_python_distributions", []):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        unexpected_distributions.append(name)

    browser_matches = {
        pattern: sorted(glob.glob(pattern)) for pattern in manifest.get("browser_globs", [])
    }
    missing_browser_globs = [pattern for pattern, paths in browser_matches.items() if not paths]

    browser_launch = False
    browser_error = None
    if not missing_browser_globs and "playwright" not in missing_modules:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content("<main>lychee-sandbox-ready</main>")
                browser_launch = page.text_content("main") == "lychee-sandbox-ready"
                browser.close()
        except Exception as exc:  # pragma: no cover - exercised inside the image build
            browser_error = f"{type(exc).__name__}: {exc}"

    prefix = str(manifest["writable_python_prefix"])
    prefix_matches = os.path.realpath(sys.prefix) == os.path.realpath(prefix)
    site_packages = next(
        (path for path in site.getsitepackages() if os.path.realpath(path).startswith(prefix)),
        None,
    )
    writable_python_prefix = False
    write_error = None
    if site_packages:
        try:
            with tempfile.NamedTemporaryFile(
                dir=site_packages,
                prefix=".lychee-write-",
                delete=True,
            ):
                writable_python_prefix = True
        except OSError as exc:
            write_error = str(exc)

    status = (
        "ready"
        if not missing_commands
        and not missing_modules
        and not distribution_mismatches
        and not unexpected_distributions
        and not missing_browser_globs
        and browser_launch
        and prefix_matches
        and writable_python_prefix
        else "failed"
    )
    return {
        "status": status,
        "schema_version": manifest["schema_version"],
        "profile": manifest["profile"],
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "prefix_matches": prefix_matches,
        "writable_python_prefix": writable_python_prefix,
        "write_error": write_error,
        "missing_commands": missing_commands,
        "missing_modules": missing_modules,
        "distribution_versions": distribution_versions,
        "distribution_mismatches": distribution_mismatches,
        "unexpected_distributions": unexpected_distributions,
        "missing_browser_globs": missing_browser_globs,
        "browser_matches": browser_matches,
        "browser_launch": browser_launch,
        "browser_error": browser_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify(args.manifest)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    else:
        print(f"Benchmark sandbox: {report['status']}")
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
