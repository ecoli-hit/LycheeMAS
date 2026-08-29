"""Environment discovery and isolated Eval Studio readiness checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_child_environment(payload: dict[str, Any]) -> dict[str, str]:
    """Build validated proxy and package-mirror variables for one child process."""

    env = os.environ.copy()
    proxy = _mapping(payload.get("proxy"))
    if proxy.get("enabled"):
        mappings = {
            "http_proxy": ("HTTP_PROXY", "http_proxy"),
            "https_proxy": ("HTTPS_PROXY", "https_proxy"),
            "all_proxy": ("ALL_PROXY", "all_proxy"),
            "no_proxy": ("NO_PROXY", "no_proxy"),
        }
        for field, names in mappings.items():
            value = str(proxy.get(field) or "").strip()
            if not value:
                continue
            if field != "no_proxy":
                parsed = urlparse(value)
                if (
                    parsed.scheme not in {"http", "https", "socks5", "socks5h"}
                    or not parsed.hostname
                ):
                    raise ValueError(f"invalid {field} URL")
            for name in names:
                env[name] = value
    mirrors = _mapping(payload.get("mirrors"))
    if mirrors.get("enabled"):
        index_url = str(mirrors.get("pip_index_url") or "").strip()
        extra_index_url = str(mirrors.get("pip_extra_index_url") or "").strip()
        for value, field in (
            (index_url, "pip_index_url"),
            (extra_index_url, "pip_extra_index_url"),
        ):
            if value:
                parsed = urlparse(value)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError(f"invalid {field} URL")
        if index_url:
            env.update(
                PIP_INDEX_URL=index_url,
                UV_INDEX_URL=index_url,
                UV_DEFAULT_INDEX=index_url,
            )
        if extra_index_url:
            env["PIP_EXTRA_INDEX_URL"] = extra_index_url
    return env

_PROBE = r"""
import importlib.util
import json
import os
import platform
import sys
import traceback

MODULE_GROUPS = {
    "core": ["lychee_mas"],
    "benchmark": ["yaml", "datasets", "pandas", "pyarrow", "sympy", "requests"],
    "runtime": ["torch", "transformers", "autogen_core", "autogen_agentchat", "openai"],
    "tools": ["PIL", "pypdf", "pdfplumber", "playwright", "markitdown"],
    "agentinit": ["numpy", "vendi_score"],
}

groups = {}
for group, modules in MODULE_GROUPS.items():
    items = []
    for module in modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except Exception:
            available = False
        items.append({"name": module, "available": available})
    groups[group] = items

checks = []
try:
    import lychee_mas
    checks.append({"id": "core_import", "status": "pass", "detail": "import lychee_mas"})
except Exception as exc:
    checks.append({"id": "core_import", "status": "fail", "detail": repr(exc)})

try:
    from lychee_mas.eval.evaluation.metrics import score
    value = score("exact", "42", "42")
    checks.append({
        "id": "scorer",
        "status": "pass" if value == 1.0 else "fail",
        "detail": f"exact score={value}",
    })
except Exception as exc:
    checks.append({"id": "scorer", "status": "fail", "detail": repr(exc)})

try:
    from lychee_mas.eval.benchmarks import LOADERS
    checks.append({
        "id": "benchmark_registry",
        "status": "pass",
        "detail": f"{len(LOADERS)} runnable tasks",
    })
except Exception as exc:
    checks.append({"id": "benchmark_registry", "status": "fail", "detail": repr(exc)})

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        browser_version = browser.version
        browser.close()
    checks.append({
        "id": "playwright_chromium",
        "status": "pass",
        "detail": f"Chromium {browser_version}",
    })
except Exception as exc:
    checks.append({
        "id": "playwright_chromium",
        "status": "warn",
        "detail": repr(exc),
    })

cuda = {"available": False, "device_count": 0, "devices": []}
try:
    import torch
    cuda["available"] = bool(torch.cuda.is_available())
    cuda["device_count"] = int(torch.cuda.device_count())
    if cuda["available"]:
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(cuda["device_count"])]
except Exception as exc:
    cuda["error"] = repr(exc)

print(json.dumps({
    "python": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "prefix": sys.prefix,
    "groups": groups,
    "checks": checks,
    "cuda": cuda,
}, ensure_ascii=False))
"""

INSTALL_PROFILES: dict[str, dict[str, Any]] = {
    "basic": {
        "label": "基础开发",
        "description": "可编辑安装 LycheeMAS，并安装测试和静态检查工具",
        "extras": ["dev"],
    },
    "eval": {
        "label": "Eval / Benchmark",
        "description": "数据准备、评分、HumanEval/GAIA 工具链和 Eval Studio",
        "extras": ["benchmark", "studio"],
    },
    "runtime": {
        "label": "本地推理 Runtime",
        "description": "AutoGen、Transformers、Torch 与配置系统",
        "extras": ["runtime", "config"],
    },
    "agentinit": {
        "label": "AgentInit",
        "description": "AgentInit 多样性与相关性选队依赖",
        "extras": ["construct"],
    },
    "docs": {
        "label": "文档站",
        "description": "MkDocs Material 和 API 文档生成工具",
        "extras": ["docs"],
    },
    "full": {
        "label": "完整环境",
        "description": "全部运行、评测、开发和 Studio 依赖",
        "extras": ["all", "dev", "studio"],
    },
}


def installation_profiles() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in INSTALL_PROFILES.items()]


def recommended_install_profiles(result: dict[str, Any]) -> list[str]:
    if result.get("status") == "missing":
        return ["basic"]
    groups = result.get("groups") or {}
    missing = {
        name for name, items in groups.items() if any(not item.get("available") for item in items)
    }
    recommendations = []
    if "core" in missing:
        recommendations.append("basic")
    if missing & {"benchmark", "tools"}:
        recommendations.append("eval")
    if "runtime" in missing:
        recommendations.append("runtime")
    if "agentinit" in missing:
        recommendations.append("agentinit")
    return list(dict.fromkeys(recommendations))


def installation_command(
    repo_root: Path,
    python: str,
    profiles: list[str],
) -> list[str]:
    unknown = [profile for profile in profiles if profile not in INSTALL_PROFILES]
    if unknown:
        raise ValueError(f"unknown installation profiles: {unknown}")
    extras = list(
        dict.fromkeys(
            extra for profile in profiles for extra in INSTALL_PROFILES[profile]["extras"]
        )
    )
    target = f"{repo_root}[{','.join(extras)}]" if extras else str(repo_root)
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", python, "-e", target]
    return [python, "-m", "pip", "install", "-e", target]


def _conda_sh(conda_exe: str | None) -> str:
    if not conda_exe:
        return ""
    path = Path(conda_exe).expanduser().absolute().parent.parent / "etc/profile.d/conda.sh"
    return str(path) if path.is_file() else ""


def conda_executable() -> str:
    """Find Conda even when Eval Studio was started without shell initialization."""
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    executable = Path(sys.executable).resolve()
    parents = executable.parents
    if len(parents) >= 4 and parents[1].parent.name == "envs":
        candidates.append(str(parents[1].parent.parent / "bin/conda"))
    candidates.extend(
        str(Path.home() / directory / "bin/conda")
        for directory in ("miniconda3", "anaconda3", "miniforge3")
    )
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser().resolve())
    return ""


def readme_environment(repo_root: Path) -> dict[str, Any]:
    """Return the canonical environment documented in README.md."""
    conda_exe = conda_executable()
    return {
        "id": "readme-default",
        "name": "LycheeMAS (README default)",
        "kind": "project_venv",
        "python": str(repo_root / ".venv/bin/python"),
        "venv": str(repo_root / ".venv"),
        "conda_env": "LycheeMAS",
        "conda_sh": _conda_sh(conda_exe),
        "is_default": True,
        "exists": (repo_root / ".venv/bin/python").is_file(),
    }


def discover_environments(repo_root: Path) -> dict[str, Any]:
    default = readme_environment(repo_root)
    rows: list[dict[str, Any]] = [default]
    seen = {str(Path(default["python"]).expanduser().absolute())}

    current_python = str(Path(sys.executable).absolute())
    if current_python not in seen:
        rows.append(
            {
                "id": "studio-server",
                "name": "Eval Studio server environment",
                "kind": "current",
                "python": current_python,
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
                "conda_sh": _conda_sh(os.environ.get("CONDA_EXE")),
                "is_default": False,
                "exists": True,
            }
        )
        seen.add(current_python)

    conda_exe = conda_executable()
    if conda_exe:
        try:
            output = subprocess.run(
                [conda_exe, "env", "list", "--json"],
                text=True,
                capture_output=True,
                check=True,
                timeout=15,
            )
            for prefix_text in json.loads(output.stdout).get("envs", []):
                prefix = Path(prefix_text)
                python = prefix / "bin/python"
                key = str(python.absolute())
                if not python.is_file() or key in seen:
                    continue
                rows.append(
                    {
                        "id": f"conda-{prefix.name}",
                        "name": f"Conda: {prefix.name}",
                        "kind": "conda",
                        "python": key,
                        "conda_env": prefix.name,
                        "conda_sh": _conda_sh(conda_exe),
                        "is_default": False,
                        "exists": True,
                    }
                )
                seen.add(key)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            pass
    return {"default": default, "environments": rows}


def inspect_environment(
    repo_root: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    python = Path(str(spec.get("python") or "")).expanduser().absolute()
    if not python.is_file():
        return {
            "status": "missing",
            "environment": {**spec, "python": str(python)},
            "error": f"Python executable not found: {python}",
            "checks": [],
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    try:
        result = subprocess.run(
            [str(python), "-c", _PROBE],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "environment": {**spec, "python": str(python)},
            "error": repr(exc),
            "checks": [],
        }

    payload: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {}
    external = []
    for command in ("docker", "tmux", "ffmpeg", "node", "uv"):
        path = shutil.which(command)
        item: dict[str, Any] = {
            "name": command,
            "available": bool(path),
            "path": path,
            "detail": path,
        }
        if command == "node" and path:
            try:
                node_probe = subprocess.run(
                    [path, "--version"],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                version = node_probe.stdout.strip().lstrip("v")
                parts = version.split(".")
                major = int(parts[0]) if parts else 0
                minor = int(parts[1]) if len(parts) > 1 else 0
                compatible = (
                    (major == 20 and minor >= 19)
                    or (major == 22 and minor >= 12)
                    or major > 22
                )
                item.update(
                    {
                        "available": compatible,
                        "installed": True,
                        "version": version,
                        "compatible": compatible,
                        "detail": (
                            f"Node.js {version} · {path}"
                            if compatible
                            else f"Node.js {version} 不兼容（需要 ^20.19 或 >=22.12） · {path}"
                        ),
                    }
                )
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                item.update(
                    {
                        "available": False,
                        "installed": True,
                        "compatible": False,
                        "detail": f"Node.js 版本检测失败：{exc!r} · {path}",
                    }
                )
        external.append(item)
    checks = list(payload.get("checks") or [])
    required_failed = any(
        item.get("status") == "fail"
        for item in checks
        if item.get("id") in {"core_import", "scorer", "benchmark_registry"}
    )
    required_packages_missing = [
        item["name"]
        for group in ("core", "benchmark", "runtime")
        for item in payload.get("groups", {}).get(group, [])
        if not item.get("available")
    ]
    response = {
        "status": (
            "failed"
            if result.returncode or required_failed
            else "incomplete"
            if required_packages_missing
            else "ready"
        ),
        "environment": {**spec, "python": str(python)},
        "return_code": result.returncode,
        "stderr": result.stderr[-4000:],
        "external": external,
        "required_packages_missing": required_packages_missing,
        **payload,
    }
    response["recommended_install_profiles"] = recommended_install_profiles(response)
    return response
