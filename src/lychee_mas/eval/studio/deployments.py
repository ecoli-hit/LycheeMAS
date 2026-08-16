"""File-backed DeploymentSpec and DeploymentInstance registries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .lifecycle import spec_fingerprint
from .pricing import PricingRegistry

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SESSION_KEY_PREFIX = "LYCHEE_STUDIO_API_KEY_"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(command: str) -> str:
    return re.sub(
        r"((?:--api-key|--token)(?:=|\s+))[^\s]+",
        r"\1<redacted>",
        command,
        flags=re.IGNORECASE,
    )


def _session_key_name(deployment_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", deployment_id).upper()
    return f"{_SESSION_KEY_PREFIX}{normalized}"


def _reasoning_token_accounting_probe(payload: Any) -> dict[str, Any]:
    """Inspect one OpenAI-compatible response without changing its content."""

    choices = payload.get("choices") if isinstance(payload, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    message = message if isinstance(message, dict) else {}
    reasoning_content = message.get("reasoning_content") or message.get("reasoning")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    reasoning_tokens = completion_details.get("reasoning_tokens")
    if reasoning_tokens is not None:
        accounting = "provider_usage"
    elif reasoning_content:
        accounting = "client_retokenized_required"
    else:
        accounting = "inconclusive"
    return {
        "reasoning_token_accounting": accounting,
        "reasoning_token_accounting_probe": accounting,
        "provider_reasoning_tokens_reported": reasoning_tokens is not None,
        "probe_reasoning_content_present": bool(reasoning_content),
        "probe_completion_tokens": usage.get("completion_tokens"),
        "probe_reasoning_tokens": reasoning_tokens,
    }


def _deployment_spec_fingerprint(spec: dict[str, Any], *, include_notes: bool = False) -> str:
    """Hash behavior-bearing Spec fields while ignoring registry bookkeeping."""
    if include_notes:
        value = {
            key: item
            for key, item in spec.items()
            if key not in {"registry_path", "updated_at_utc"}
        }
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return spec_fingerprint(spec)


def _vllm_supports_flag(python: Path, flag: str, environment: dict[str, str]) -> bool:
    """Inspect the selected vLLM CLI instead of assuming a package version."""

    if not python.is_file() or not os.access(python, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [
                str(python),
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in f"{result.stdout}\n{result.stderr}"


class DeploymentRegistry:
    """Store DeploymentSpecs and inspect or manage DeploymentInstances."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/deployments"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"
        self.log_root = self.repo_root / "runs/eval_studio/deployment_instances"
        self.pricing = PricingRegistry(self.repo_root)

    def specs(self) -> list[dict[str, Any]]:
        if not self.spec_root.is_dir():
            return []
        rows = []
        for path in sorted(self.spec_root.glob("*.json")):
            try:
                value = self.validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            rows.append({**value, "registry_path": str(path)})
        return rows

    def instances(self, *, probe: bool = False) -> list[dict[str, Any]]:
        """Read declared instance files and optionally verify their availability."""

        if not self.instance_root.is_dir():
            return []
        specs = {item["id"]: item for item in self.specs()}
        rows = []
        for path in sorted(self.instance_root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    continue
                value = self._check_instance(dict(value), probe=probe)
            except (OSError, ValueError):
                continue
            if probe:
                self._write_instance(value)
            value["deployment_spec"] = specs.get(value["deployment_spec_id"])
            if value["deployment_spec"] is None:
                value.update(
                    observed_status="invalid",
                    status="invalid",
                    available=False,
                    health_detail="referenced DeploymentSpec is missing",
                )
            try:
                actual_pricing = self.pricing.get_instance(value["actual_pricing_instance_id"])
                value["actual_pricing_instance"] = actual_pricing
                value["actual_pricing_status"] = self.pricing.completeness(actual_pricing)
                equivalent_id = value.get("api_equivalent_pricing_instance_id")
                if equivalent_id:
                    equivalent = self.pricing.get_instance(str(equivalent_id))
                    value["api_equivalent_pricing_instance"] = equivalent
                    value["api_equivalent_pricing_status"] = self.pricing.completeness(equivalent)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                value.update(
                    observed_status="invalid",
                    status="invalid",
                    available=False,
                    actual_pricing_instance=None,
                    actual_pricing_status={"status": "invalid", "detail": str(exc)},
                )
            rows.append({**value, "registry_path": str(path)})
        return sorted(
            rows,
            key=lambda item: (
                item.get("status") not in {"running", "ready_on_run"},
                item["id"],
            ),
        )

    def instance(self, instance_id: str, *, probe: bool = False) -> dict[str, Any]:
        self._validate_id(instance_id)
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid DeploymentInstance document {path}")
        checked = self._check_instance(dict(value), probe=probe)
        checked["deployment_spec"] = next(
            (item for item in self.specs() if item["id"] == checked["deployment_spec_id"]),
            None,
        )
        if checked["deployment_spec"] is None:
            checked.update(
                observed_status="invalid",
                status="invalid",
                available=False,
                health_detail="referenced DeploymentSpec is missing",
            )
        try:
            actual_pricing = self.pricing.get_instance(checked["actual_pricing_instance_id"])
            checked["actual_pricing_instance"] = actual_pricing
            checked["actual_pricing_status"] = self.pricing.completeness(actual_pricing)
            equivalent_id = checked.get("api_equivalent_pricing_instance_id")
            if equivalent_id:
                equivalent = self.pricing.get_instance(str(equivalent_id))
                checked["api_equivalent_pricing_instance"] = equivalent
                checked["api_equivalent_pricing_status"] = self.pricing.completeness(equivalent)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            checked.update(
                observed_status="invalid",
                status="invalid",
                available=False,
                actual_pricing_instance=None,
                actual_pricing_status={"status": "invalid", "detail": str(exc)},
            )
        if probe:
            self._write_instance(checked)
        return {**checked, "registry_path": str(path)}

    def deploy(
        self,
        spec: dict[str, Any],
        *,
        instance_id: str,
        source_instance: dict[str, str],
        source_fields: dict[str, Any],
        actual_pricing_instance_id: str,
        api_equivalent_pricing_instance_id: str | None = None,
        api_key: str = "",
    ) -> dict[str, Any]:
        """Materialize a DeploymentSpec with one compatible resource Instance."""

        value_spec = self.validate(spec)
        self._validate_id(instance_id)
        actual_pricing = self.pricing.get_instance(actual_pricing_instance_id)
        if actual_pricing["pricing_spec_id"] != value_spec["actual_pricing_spec_id"]:
            raise ValueError(
                f"PricingInstance {actual_pricing_instance_id!r} is not an instance of "
                f"PricingSpec {value_spec['actual_pricing_spec_id']!r}"
            )
        if not self.pricing.compatible(actual_pricing, value_spec["kind"]):
            raise ValueError(
                f"PricingInstance {actual_pricing_instance_id!r} does not support "
                f"deployment kind {value_spec['kind']!r}"
            )
        self.pricing.validate_actual_binding(actual_pricing, value_spec)
        equivalent_spec_id = value_spec.get("api_equivalent_pricing_spec_id")
        if equivalent_spec_id and not api_equivalent_pricing_instance_id:
            raise ValueError(
                "DeploymentInstance requires api_equivalent_pricing_instance_id for "
                f"PricingSpec {equivalent_spec_id!r}"
            )
        equivalent = None
        if api_equivalent_pricing_instance_id:
            equivalent = self.pricing.get_instance(api_equivalent_pricing_instance_id)
            if equivalent["pricing_spec_id"] != equivalent_spec_id:
                raise ValueError(
                    f"PricingInstance {api_equivalent_pricing_instance_id!r} is not an "
                    f"instance of PricingSpec {equivalent_spec_id!r}"
                )
            if equivalent.get("pricing_spec", {}).get("basis") != "token_usage":
                raise ValueError("API-equivalent pricing must use token_usage basis")
        payload = {
            **{key: item for key, item in value_spec.items() if key != "source_spec"},
            **source_fields,
        }
        self.pricing.validate_model_binding(
            actual_pricing,
            payload,
            purpose="actual-cost",
            source_spec_id=str(value_spec["source_spec"]["id"]),
            require_source_scope=(
                actual_pricing.get("pricing_spec", {}).get("basis") == "token_usage"
            ),
        )
        if equivalent is not None:
            self.pricing.validate_model_binding(
                equivalent,
                payload,
                purpose="API-equivalent",
            )
        manual_api_key = str(api_key or "")
        credential_source = "environment"
        if manual_api_key:
            if payload.get("kind") not in {"api", "vllm"}:
                raise ValueError("manual API key input is only supported for API/vLLM deployments")
            key_name = _session_key_name(instance_id)
            os.environ[key_name] = manual_api_key
            payload["api_key_env"] = key_name
            payload["auth_mode"] = "env"
            credential_source = "manual_session"
        value = self._validate_materialized(payload)
        kind = value["kind"]
        instance = {
            **value,
            "id": instance_id,
            "deployment_spec_id": value_spec["id"],
            "deployment_spec_fingerprint": _deployment_spec_fingerprint(value_spec),
            "actual_pricing_instance_id": actual_pricing_instance_id,
            "api_equivalent_pricing_instance_id": api_equivalent_pricing_instance_id,
            "source_instance": dict(source_instance),
            "created_at_utc": _utc_now(),
            "managed": bool(value.get("managed", kind != "api")),
            "creation_source": "studio",
            "credential_source": credential_source if kind in {"api", "vllm"} else None,
        }
        if kind == "vllm":
            host = str(value.get("host") or "127.0.0.1")
            port = int(value.get("port") or 8000)
            base_url = str(value.get("base_url") or f"http://{host}:{port}/v1").rstrip("/")
            metrics_mode = str(value.get("per_request_metrics_mode") or "auto")
            prompt_details_mode = str(value.get("prompt_tokens_details_mode") or "auto")
            instance.update(
                per_request_metrics_mode=metrics_mode,
                prompt_tokens_details_mode=prompt_details_mode,
                metrics_url=f"{base_url[:-3] if base_url.endswith('/v1') else base_url}/metrics",
            )
            health = self._probe_endpoint(
                base_url,
                value.get("api_key_env"),
                model_id=value.get("model_id"),
                auth_mode=value.get("auth_mode", "env"),
                trust_env=bool(value.get("trust_env", True)),
                **(
                    {"thinking_budget_field": "thinking_token_budget"}
                    if bool((value.get("capabilities") or {}).get("native_thinking_budget"))
                    else {}
                ),
            )
            if not value.get("managed", True):
                instance.update(
                    health,
                    managed=False,
                    instance_type=(
                        "external_shared_server" if value.get("shared") else "external_server"
                    ),
                    lifecycle=(
                        "External service is probed and used, never started or stopped "
                        "by Eval Studio"
                    ),
                    base_url=base_url,
                )
            elif health["status"] == "running":
                instance.update(
                    health,
                    managed=False,
                    instance_type="external_server",
                    lifecycle="Existing endpoint is used but not managed by Eval Studio",
                    base_url=base_url,
                )
            else:
                model_path = Path(str(value.get("model_path") or "")).expanduser()
                python = Path(str(value.get("python") or "")).expanduser()
                if not model_path.exists():
                    raise ValueError("managed vLLM deployment requires an existing model_path")
                if not python.is_file():
                    raise ValueError(
                        "managed vLLM deployment requires an existing Python executable"
                    )
                self.log_root.mkdir(parents=True, exist_ok=True)
                log_path = self.log_root / f"{value['id']}.log"
                command = [
                    str(python),
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    str(model_path),
                    "--served-model-name",
                    str(value["model_id"]),
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--tensor-parallel-size",
                    str(value.get("tensor_parallel_size") or 1),
                    "--data-parallel-size",
                    str(value.get("data_parallel_size") or 1),
                    "--max-model-len",
                    str(value.get("max_model_len") or 32768),
                    "--gpu-memory-utilization",
                    str(value.get("gpu_memory_utilization") or 0.9),
                ]
                if value.get("dtype"):
                    command.extend(["--dtype", str(value["dtype"])])
                if value.get("max_num_seqs"):
                    command.extend(["--max-num-seqs", str(value["max_num_seqs"])])
                if value.get("reasoning_parser"):
                    command.extend(["--reasoning-parser", str(value["reasoning_parser"])])
                if value.get("reasoning_config"):
                    command.extend(
                        [
                            "--reasoning-config",
                            json.dumps(value["reasoning_config"], ensure_ascii=False),
                        ]
                    )
                if value.get("enable_auto_tool_choice"):
                    command.append("--enable-auto-tool-choice")
                if value.get("tool_call_parser"):
                    command.extend(["--tool-call-parser", str(value["tool_call_parser"])])
                if value.get("enable_prefix_caching"):
                    command.append("--enable-prefix-caching")
                if value.get("enforce_eager"):
                    command.append("--enforce-eager")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(value.get("cuda_visible_devices") or "0")
                environment_lib = python.parent.parent / "lib"
                site_packages = sorted(
                    {path.resolve() for path in environment_lib.glob("python*/site-packages")}
                )
                cuda_roots = [
                    path
                    for site_packages_root in site_packages
                    for path in sorted((site_packages_root / "nvidia").glob("cu*"))
                    if (path / "bin/nvcc").is_file()
                ]
                nvidia_library_paths = [
                    path
                    for site_packages_root in site_packages
                    for path in sorted((site_packages_root / "nvidia").glob("*/lib"))
                    if path.is_dir()
                ]
                existing_library_path = environment.get("LD_LIBRARY_PATH", "")
                configured_library_paths = [
                    str(Path(item).expanduser())
                    for item in value.get("library_paths", [])
                    if str(item).strip()
                ]
                environment["LD_LIBRARY_PATH"] = ":".join(
                    item
                    for item in (
                        str(environment_lib),
                        *(str(path) for path in nvidia_library_paths),
                        *configured_library_paths,
                        existing_library_path,
                    )
                    if item
                )
                environment["PATH"] = ":".join(
                    item
                    for item in (
                        str(python.parent),
                        *(str(path / "bin") for path in cuda_roots),
                        environment.get("PATH", ""),
                    )
                    if item
                )
                if cuda_roots:
                    cuda_root = cuda_roots[0]
                    include_path = str(cuda_root / "include")
                    environment["CUDA_HOME"] = str(cuda_root)
                    environment["CUDACXX"] = str(cuda_root / "bin/nvcc")
                    environment["CPATH"] = ":".join(
                        item for item in (include_path, environment.get("CPATH", "")) if item
                    )
                    environment["CPLUS_INCLUDE_PATH"] = ":".join(
                        item
                        for item in (
                            include_path,
                            environment.get("CPLUS_INCLUDE_PATH", ""),
                        )
                        if item
                    )
                if value.get("use_flashinfer_sampler") is not None:
                    environment["VLLM_USE_FLASHINFER_SAMPLER"] = (
                        "1" if bool(value["use_flashinfer_sampler"]) else "0"
                    )
                metrics_flag = "--enable-per-request-metrics"
                supports_per_request_metrics = _vllm_supports_flag(
                    python, metrics_flag, environment
                )
                if metrics_mode == "enabled" and not supports_per_request_metrics:
                    raise ValueError(
                        f"selected vLLM Python does not support {metrics_flag}; "
                        "use per_request_metrics_mode=auto/disabled or upgrade vLLM"
                    )
                per_request_metrics_enabled = (
                    supports_per_request_metrics and metrics_mode != "disabled"
                )
                if per_request_metrics_enabled:
                    command.append(metrics_flag)
                prompt_details_flag = "--enable-prompt-tokens-details"
                supports_prompt_tokens_details = _vllm_supports_flag(
                    python, prompt_details_flag, environment
                )
                if prompt_details_mode == "enabled" and not supports_prompt_tokens_details:
                    raise ValueError(
                        f"selected vLLM Python does not support {prompt_details_flag}; "
                        "use prompt_tokens_details_mode=auto/disabled or upgrade vLLM"
                    )
                prompt_tokens_details_enabled = (
                    supports_prompt_tokens_details and prompt_details_mode != "disabled"
                )
                if prompt_tokens_details_enabled:
                    command.append(prompt_details_flag)
                log = log_path.open("wb", buffering=0)
                process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
                wait_for_exit = getattr(process, "wait", None)
                if callable(wait_for_exit):
                    threading.Thread(
                        target=wait_for_exit,
                        name=f"vllm-reaper-{instance_id}",
                        daemon=True,
                    ).start()
                instance.update(
                    status="starting",
                    instance_type="managed_server",
                    pid=process.pid,
                    host=host,
                    port=port,
                    base_url=base_url,
                    python=str(python),
                    log_path=str(log_path),
                    command=_redact(" ".join(command)),
                    per_request_metrics_supported=supports_per_request_metrics,
                    per_request_metrics_enabled=per_request_metrics_enabled,
                    prompt_tokens_details_supported=supports_prompt_tokens_details,
                    prompt_tokens_details_enabled=prompt_tokens_details_enabled,
                )
        elif kind == "api":
            base_url = str(value.get("base_url") or "").rstrip("/")
            if not base_url:
                raise ValueError("API deployment requires base_url")
            instance.update(
                self._probe_endpoint(
                    base_url,
                    value.get("api_key_env"),
                    model_id=value.get("model_id"),
                    auth_mode=value.get("auth_mode", "env"),
                    trust_env=bool(value.get("trust_env", True)),
                ),
                instance_type="remote_api_connection",
                base_url=base_url,
            )
        else:
            model_path = Path(str(value.get("model_path") or "")).expanduser()
            python = Path(str(value.get("python") or "")).expanduser()
            if not model_path.exists():
                raise ValueError("HF deployment requires an existing model_path")
            if not python.is_file():
                raise ValueError("HF deployment requires an existing Python executable")
            instance.update(
                self._probe_hf(value),
                instance_type="in_process_backend",
                lifecycle="HF backend is instantiated inside each experiment process",
                model_path=str(model_path),
                python=str(python),
            )
        self._write_instance(instance)
        return instance

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        self._validate_id(instance_id)
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        match = json.loads(path.read_text(encoding="utf-8"))
        pid = match.get("pid") if match.get("managed") else None
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        key_name = str(match.get("api_key_env") or "")
        if match.get("credential_source") == "manual_session" and key_name.startswith(
            _SESSION_KEY_PREFIX
        ):
            os.environ.pop(key_name, None)
        path.unlink()
        return {
            "id": instance_id,
            "deleted": True,
            "termination_requested": bool(pid),
        }

    def save(self, spec: dict[str, Any]) -> dict[str, Any]:
        value = self.validate(spec)
        self.spec_root.mkdir(parents=True, exist_ok=True)
        path = self.spec_root / f"{value['id']}.json"
        value.update(updated_at_utc=_utc_now())
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
        return {**value, "registry_path": str(path)}

    def delete(self, deployment_id: str) -> dict[str, Any]:
        self._validate_id(deployment_id)
        path = self.spec_root / f"{deployment_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": deployment_id, "deleted": True, "path": str(path)}

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Validate a reusable DeploymentSpec that references only a resource Spec."""

        value = dict(spec)
        if value.get("api_key"):
            raise ValueError("api_key is session-only and cannot be stored in a DeploymentSpec")
        value.pop("api_key", None)
        deployment_id = str(value.get("id") or "")
        self._validate_id(deployment_id)
        kind = str(value.get("kind") or "")
        if kind not in {"hf", "vllm", "api"}:
            raise ValueError("deployment kind must be hf, vllm or api")
        source_spec = dict(value.get("source_spec") or {})
        source_type = str(source_spec.get("type") or "")
        source_id = str(source_spec.get("id") or "")
        if source_type not in {"model", "api"} or not _SAFE_ID.fullmatch(source_id):
            raise ValueError("DeploymentSpec requires source_spec.type=model/api and a valid id")
        managed_vllm = kind == "vllm" and value.get("managed", True) is not False
        if kind != "vllm" and value.get("managed") is not None:
            expected_managed = kind == "hf"
            if bool(value["managed"]) != expected_managed:
                raise ValueError(f"{kind} DeploymentSpec has invalid managed={value['managed']!r}")
        value["managed"] = managed_vllm if kind == "vllm" else kind == "hf"
        expected_type = "model" if kind == "hf" or managed_vllm else "api"
        if source_type != expected_type:
            raise ValueError(f"{kind} DeploymentSpec requires source_spec.type={expected_type!r}")
        value["id"] = deployment_id
        value["kind"] = kind
        value["source_spec"] = {"type": source_type, "id": source_id}
        default_actual_pricing_spec_id = (
            "allocated-gpu-time" if kind == "hf" or managed_vllm else "api-token-usage"
        )
        actual_pricing_spec_id = str(
            value.get("actual_pricing_spec_id") or default_actual_pricing_spec_id
        )
        self._validate_id(actual_pricing_spec_id)
        pricing_spec = self.pricing.get_spec(actual_pricing_spec_id)
        required_basis = self.pricing.required_actual_basis(value)
        if pricing_spec["basis"] != required_basis:
            raise ValueError(
                f"DeploymentSpec {deployment_id!r} requires actual PricingSpec basis "
                f"{required_basis!r}, got {pricing_spec['basis']!r}"
            )
        if kind not in set(pricing_spec.get("supported_deployment_kinds") or []):
            raise ValueError(
                f"PricingSpec {actual_pricing_spec_id!r} does not support deployment kind {kind!r}"
            )
        value["actual_pricing_spec_id"] = actual_pricing_spec_id
        equivalent_spec_id = value.get("api_equivalent_pricing_spec_id")
        if equivalent_spec_id in {None, ""}:
            value["api_equivalent_pricing_spec_id"] = None
        else:
            if required_basis != "allocated_gpu_time":
                raise ValueError(
                    "API-equivalent pricing is only valid for local HF or managed vLLM "
                    "deployments; API and external vLLM already use token pricing as "
                    "their actual cost"
                )
            equivalent_spec_id = str(equivalent_spec_id)
            self._validate_id(equivalent_spec_id)
            equivalent_spec = self.pricing.get_spec(equivalent_spec_id)
            if equivalent_spec["basis"] != "token_usage":
                raise ValueError("API-equivalent PricingSpec must use token_usage basis")
            value["api_equivalent_pricing_spec_id"] = equivalent_spec_id
        forbidden = {
            "model_instance_id",
            "api_instance_id",
            "model_id",
            "model_path",
            "base_url",
            "api_key_env",
            "auth_mode",
        }
        present = sorted(key for key in forbidden if value.get(key) not in {None, ""})
        if present:
            raise ValueError(
                "DeploymentSpec cannot bind resource Instance fields: " + ", ".join(present)
            )
        if kind == "vllm" and value.get("reasoning_config") is not None:
            reasoning_config = value["reasoning_config"]
            if not isinstance(reasoning_config, dict):
                raise ValueError("reasoning_config must be an object")
            unknown = set(reasoning_config) - {
                "reasoning_start_str",
                "reasoning_end_str",
            }
            if unknown:
                raise ValueError(
                    "reasoning_config contains unsupported fields: " + ", ".join(sorted(unknown))
                )
            normalized_reasoning_config: dict[str, str | None] = {}
            for field_name in ("reasoning_start_str", "reasoning_end_str"):
                field_value = reasoning_config.get(field_name)
                if field_value is not None and not isinstance(field_value, str):
                    raise ValueError(f"reasoning_config.{field_name} must be a string or null")
                normalized_reasoning_config[field_name] = field_value
            if not normalized_reasoning_config.get("reasoning_end_str"):
                raise ValueError("reasoning_config.reasoning_end_str is required")
            value["reasoning_config"] = normalized_reasoning_config
        if kind == "vllm":
            tensor_parallel_size = int(value.get("tensor_parallel_size", 1) or 1)
            data_parallel_size = int(value.get("data_parallel_size", 1) or 1)
            max_num_seqs = int(value.get("max_num_seqs", 1) or 1)
            if tensor_parallel_size < 1 or data_parallel_size < 1 or max_num_seqs < 1:
                raise ValueError(
                    "tensor_parallel_size, data_parallel_size and max_num_seqs must be positive"
                )
            value["tensor_parallel_size"] = tensor_parallel_size
            value["data_parallel_size"] = data_parallel_size
            value["max_num_seqs"] = max_num_seqs
            if managed_vllm:
                visible_devices = [
                    item.strip()
                    for item in str(value.get("cuda_visible_devices") or "0").split(",")
                    if item.strip()
                ]
                required_devices = tensor_parallel_size * data_parallel_size
                if len(visible_devices) < required_devices:
                    raise ValueError(
                        "managed vLLM requires at least "
                        f"tensor_parallel_size * data_parallel_size = {required_devices} "
                        "CUDA visible devices"
                    )
            metrics_mode = str(value.get("per_request_metrics_mode") or "auto")
            if metrics_mode not in {"auto", "enabled", "disabled"}:
                raise ValueError("per_request_metrics_mode must be auto, enabled or disabled")
            value["per_request_metrics_mode"] = metrics_mode
            prompt_details_mode = str(value.get("prompt_tokens_details_mode") or "auto")
            if prompt_details_mode not in {"auto", "enabled", "disabled"}:
                raise ValueError("prompt_tokens_details_mode must be auto, enabled or disabled")
            value["prompt_tokens_details_mode"] = prompt_details_mode
            if value.get("use_flashinfer_sampler") is not None:
                value["use_flashinfer_sampler"] = bool(value["use_flashinfer_sampler"])
        else:
            stale_vllm_fields = sorted(
                field
                for field in (
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "max_num_seqs",
                    "per_request_metrics_mode",
                    "prompt_tokens_details_mode",
                    "reasoning_parser",
                    "reasoning_config",
                    "tool_call_parser",
                    "enable_auto_tool_choice",
                    "enable_prefix_caching",
                    "use_flashinfer_sampler",
                    "enforce_eager",
                )
                if value.get(field) not in {None, ""}
            )
            if stale_vllm_fields:
                raise ValueError(
                    f"{kind} DeploymentSpec contains vLLM-only fields: "
                    + ", ".join(stale_vllm_fields)
                )
        if kind in {"api", "vllm"}:
            from ...runtime.backends.openai_api_backend import _normalize_timeout_policy

            value["request_timeout"] = _normalize_timeout_policy(
                value.get("request_timeout"),
                minimum_timeout_s=float(value.get("timeout", 120.0)),
            )
            request_limits = dict(value.get("request_limits") or {})
            max_concurrency = int(request_limits.get("max_concurrency", 0) or 0)
            min_interval_s = float(request_limits.get("min_interval_s", 0.0) or 0.0)
            scope = str(request_limits.get("scope") or "process")
            if max_concurrency < 0 or min_interval_s < 0:
                raise ValueError("request limits must be non-negative")
            if scope not in {"process", "host"}:
                raise ValueError("request limit scope must be process or host")
            value["request_limits"] = {
                "max_concurrency": max_concurrency,
                "min_interval_s": min_interval_s,
                "scope": scope,
            }
            value["max_retries"] = int(value.get("max_retries", 0))
            if value["max_retries"] < 0:
                raise ValueError("max_retries must be non-negative")
            extra_body = value.get("extra_body")
            if extra_body is not None and not isinstance(extra_body, dict):
                raise ValueError("extra_body must be an object")
        library_paths = value.get("library_paths")
        if library_paths is not None:
            if not isinstance(library_paths, list) or not all(
                isinstance(item, str) and item.strip() for item in library_paths
            ):
                raise ValueError("library_paths must be a list of non-empty paths")
            value["library_paths"] = [item.strip() for item in library_paths]
        value.pop("registry_path", None)
        value.pop("storage", None)
        return value

    def _validate_materialized(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate the resolved runtime document written to a DeploymentInstance."""

        value = dict(payload)
        deployment_id = str(value.get("id") or "")
        self._validate_id(deployment_id)
        kind = str(value.get("kind") or "")
        if kind not in {"hf", "vllm", "api"}:
            raise ValueError("deployment kind must be hf, vllm or api")
        if not str(value.get("model_id") or "").strip():
            raise ValueError("resolved deployment model_id is required")
        if kind in {"api", "vllm"}:
            auth_mode = str(value.get("auth_mode") or "env")
            if auth_mode not in {"env", "none"}:
                raise ValueError("auth_mode must be env or none")
            value["auth_mode"] = auth_mode
            value["trust_env"] = bool(value.get("trust_env", True))
            if auth_mode == "env":
                key_name = str(
                    value.get("api_key_env")
                    or ("DASHSCOPE_API_KEY" if kind == "api" else "VLLM_API_KEY")
                )
                if not _ENV_NAME.fullmatch(key_name):
                    raise ValueError(
                        "api_key_env must be an environment variable name, not a secret value"
                    )
                value["api_key_env"] = key_name
            else:
                value.pop("api_key_env", None)
        if kind == "vllm" and value.get("reasoning_config") is not None:
            reasoning_config = value["reasoning_config"]
            if not isinstance(reasoning_config, dict):
                raise ValueError("reasoning_config must be an object")
            unknown = set(reasoning_config) - {"reasoning_start_str", "reasoning_end_str"}
            if unknown:
                raise ValueError(
                    "reasoning_config contains unsupported fields: " + ", ".join(sorted(unknown))
                )
            if not reasoning_config.get("reasoning_end_str"):
                raise ValueError("reasoning_config.reasoning_end_str is required")
        if kind == "vllm":
            tensor_parallel_size = int(value.get("tensor_parallel_size", 1) or 1)
            data_parallel_size = int(value.get("data_parallel_size", 1) or 1)
            max_num_seqs = int(value.get("max_num_seqs", 1) or 1)
            if tensor_parallel_size < 1 or data_parallel_size < 1 or max_num_seqs < 1:
                raise ValueError(
                    "tensor_parallel_size, data_parallel_size and max_num_seqs must be positive"
                )
            value["tensor_parallel_size"] = tensor_parallel_size
            value["data_parallel_size"] = data_parallel_size
            value["max_num_seqs"] = max_num_seqs
            metrics_mode = str(value.get("per_request_metrics_mode") or "auto")
            if metrics_mode not in {"auto", "enabled", "disabled"}:
                raise ValueError("per_request_metrics_mode must be auto, enabled or disabled")
            value["per_request_metrics_mode"] = metrics_mode
            prompt_details_mode = str(value.get("prompt_tokens_details_mode") or "auto")
            if prompt_details_mode not in {"auto", "enabled", "disabled"}:
                raise ValueError("prompt_tokens_details_mode must be auto, enabled or disabled")
            value["prompt_tokens_details_mode"] = prompt_details_mode
        value.pop("registry_path", None)
        value.pop("storage", None)
        return value

    def _probe_api(self, instance: dict[str, Any]) -> dict[str, Any]:
        capabilities = dict(instance.get("capabilities") or {})
        budget_field = None
        if str(instance.get("kind") or "") == "vllm" and bool(
            capabilities.get("native_thinking_budget")
        ):
            budget_field = "thinking_token_budget"
        return self._probe_endpoint(
            str(instance.get("base_url") or ""),
            instance.get("api_key_env"),
            model_id=instance.get("model_id"),
            auth_mode=instance.get("auth_mode", "env"),
            trust_env=bool(instance.get("trust_env", True)),
            thinking_budget_field=budget_field,
        )

    @staticmethod
    def _probe_endpoint(
        base_url: str,
        api_key_env: Any,
        *,
        model_id: Any = None,
        auth_mode: Any = "env",
        trust_env: bool = True,
        thinking_budget_field: str | None = None,
    ) -> dict[str, Any]:
        if not base_url:
            return {"status": "invalid", "health_detail": "base_url is empty"}
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        mode = str(auth_mode or "env")
        key_name = str(api_key_env or "")
        if mode not in {"env", "none"}:
            return {"status": "invalid", "health_detail": "auth_mode must be env or none"}
        if mode == "env" and not _ENV_NAME.fullmatch(key_name):
            return {
                "status": "invalid",
                "health_detail": "api_key_env must contain an environment variable name",
                "api_key_env": "<invalid>",
                "credential_present": False,
            }
        key = os.environ.get(key_name) if mode == "env" else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resolved_model = str(model_id or "").strip()
        if resolved_model:
            endpoint = f"{base_url.rstrip('/')}/chat/completions"
            probe_payload: dict[str, Any] = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 64 if thinking_budget_field else 8,
                "temperature": 0,
                "stream": False,
            }
            if thinking_budget_field:
                probe_payload[thinking_budget_field] = 16
                probe_payload["chat_template_kwargs"] = {"enable_thinking": True}
            body = json.dumps(probe_payload, ensure_ascii=False).encode("utf-8")
            request = Request(endpoint, data=body, headers=headers, method="POST")
        else:
            endpoint = f"{base_url.rstrip('/')}/models"
            request = Request(endpoint, headers=headers)
        started = time.monotonic()
        try:
            opener = build_opener(ProxyHandler({})) if not trust_env else None
            open_request = opener.open if opener is not None else urlopen
            with open_request(request, timeout=30) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                preview = ""
                if resolved_model:
                    choices = payload.get("choices") if isinstance(payload, dict) else None
                    if not isinstance(choices, list) or not choices:
                        raise ValueError("chat completion response has no choices")
                    message = choices[0].get("message") or {}
                    preview = str(message.get("content") or "").strip()[:120]
                accounting_probe = (
                    _reasoning_token_accounting_probe(payload) if resolved_model else {}
                )
                return {
                    "status": "running",
                    "health_status": response.status,
                    "health_detail": (
                        "minimal chat completion succeeded"
                        if resolved_model
                        else "model endpoint succeeded"
                    ),
                    "auth_mode": mode,
                    "api_key_env": key_name or None,
                    "credential_present": bool(key),
                    "probe_model": resolved_model or None,
                    "probe_prompt": "你好" if resolved_model else None,
                    "probe_response_preview": preview or None,
                    "probe_latency_s": round(time.monotonic() - started, 3),
                    "thinking_budget_probe": bool(thinking_budget_field),
                    "thinking_budget_parameter": thinking_budget_field,
                    **accounting_probe,
                }
        except HTTPError as exc:
            status = "auth_required" if exc.code in {401, 403} else "unhealthy"
            detail = f"deployment probe returned HTTP {exc.code}"
            if status == "auth_required":
                if mode == "none":
                    detail = "endpoint requires authentication although auth_mode is none"
                else:
                    detail = (
                        f"environment variable {key_name!r} is not set in the Eval Studio process"
                        if not key
                        else f"credential from environment variable {key_name!r} was rejected"
                    )
            return {
                "status": status,
                "health_status": exc.code,
                "health_detail": detail,
                "auth_mode": mode,
                "api_key_env": key_name or None,
                "credential_present": bool(key),
            }
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "unhealthy",
                "health_detail": f"invalid deployment probe response: {exc}",
                "auth_mode": mode,
                "api_key_env": key_name or None,
                "credential_present": bool(key),
            }
        except (OSError, URLError) as exc:
            return {
                "status": "unreachable",
                "health_detail": str(exc),
                "auth_mode": mode,
                "api_key_env": key_name or None,
                "credential_present": bool(key),
            }

    def _probe_hf(self, value: dict[str, Any]) -> dict[str, Any]:
        python = str(Path(str(value["python"])).expanduser())
        model_path = str(Path(str(value["model_path"])).expanduser())
        device = str(value.get("device") or "cuda:0")
        program = (
            "import json,sys\n"
            "from lychee_mas.runtime.backends.hf_backend import HFBackend\n"
            "backend=HFBackend(model_name=sys.argv[1],device=sys.argv[2],"
            "strict_hidden=False,enable_thinking=False,max_input_tokens=512)\n"
            "result=backend.generate_chat([{'role':'user','content':'你好'}],"
            "max_new_tokens=8)\n"
            "print('__LYCHEE_PROBE__'+json.dumps({'text':result.text,"
            "'latency_s':result.latency_s},ensure_ascii=False))\n"
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(value.get("cuda_visible_devices") or "0")
        source_root = str(self.repo_root / "src")
        environment["PYTHONPATH"] = ":".join(
            item for item in (source_root, environment.get("PYTHONPATH", "")) if item
        )
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        timeout = int(value.get("probe_timeout_s") or 900)
        try:
            result = subprocess.run(
                [python, "-c", program, model_path, device],
                cwd=str(self.repo_root),
                env=environment,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"HF minimal generation timed out after {timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "HF probe failed").strip().splitlines()[-1]
            raise ValueError(f"HF minimal generation failed: {detail}") from exc
        marker = next(
            (
                line
                for line in reversed(result.stdout.splitlines())
                if line.startswith("__LYCHEE_PROBE__")
            ),
            None,
        )
        if marker is None:
            raise ValueError("HF minimal generation returned no probe result")
        payload = json.loads(marker.removeprefix("__LYCHEE_PROBE__"))
        return {
            "status": "ready_on_run",
            "health_detail": "minimal local HF generation succeeded",
            "probe_model": value.get("model_id"),
            "probe_prompt": "你好",
            "probe_response_preview": str(payload.get("text") or "").strip()[:120] or None,
            "probe_latency_s": round(float(payload.get("latency_s") or 0.0), 3),
        }

    def _check_instance(self, instance: dict[str, Any], *, probe: bool) -> dict[str, Any]:
        instance_id = str(instance.get("id") or "")
        self._validate_id(instance_id)
        deployment_spec_id = str(instance.get("deployment_spec_id") or "")
        self._validate_id(deployment_spec_id)
        actual_pricing_instance_id = str(instance.get("actual_pricing_instance_id") or "")
        self._validate_id(actual_pricing_instance_id)
        source_instance = dict(instance.get("source_instance") or {})
        if source_instance.get("type") not in {"model", "api"} or not _SAFE_ID.fullmatch(
            str(source_instance.get("id") or "")
        ):
            raise ValueError(f"DeploymentInstance {instance_id!r} requires a valid source_instance")
        kind = str(instance.get("kind") or "")
        if kind not in {"hf", "vllm", "api"}:
            raise ValueError(f"DeploymentInstance {instance_id!r} has invalid kind {kind!r}")
        if not str(instance.get("model_id") or "").strip():
            raise ValueError(f"DeploymentInstance {instance_id!r} requires model_id")
        deployment_spec = next(
            (item for item in self.specs() if item["id"] == deployment_spec_id),
            None,
        )
        if deployment_spec is None:
            return {
                **instance,
                "observed_status": "invalid",
                "status": "invalid",
                "available": False,
                "health_detail": (f"referenced DeploymentSpec {deployment_spec_id!r} is missing"),
            }
        expected_source_type = str((deployment_spec.get("source_spec") or {}).get("type") or "")
        if (
            kind != deployment_spec.get("kind")
            or source_instance.get("type") != expected_source_type
        ):
            return {
                **instance,
                "observed_status": "invalid",
                "status": "invalid",
                "available": False,
                "health_detail": "DeploymentInstance lifecycle no longer matches DeploymentSpec",
            }
        expected_fingerprint = _deployment_spec_fingerprint(deployment_spec)
        legacy_fingerprint = _deployment_spec_fingerprint(
            deployment_spec,
            include_notes=True,
        )
        stored_fingerprint = str(instance.get("deployment_spec_fingerprint") or "")
        if (
            not stored_fingerprint
            or not str(instance.get("created_at_utc") or "")
            or not str(instance.get("creation_source") or "")
        ):
            return {
                **instance,
                "observed_status": "invalid",
                "status": "invalid",
                "available": False,
                "health_detail": "DeploymentInstance lifecycle metadata is incomplete",
            }
        if stored_fingerprint and stored_fingerprint not in {
            expected_fingerprint,
            legacy_fingerprint,
        }:
            return {
                **instance,
                "observed_status": "invalid",
                "status": "invalid",
                "available": False,
                "health_detail": (
                    "DeploymentSpec changed after this instance was created; "
                    "instantiate a new DeploymentInstance"
                ),
            }
        instance["deployment_spec_fingerprint"] = expected_fingerprint
        try:
            actual_pricing = self.pricing.get_instance(actual_pricing_instance_id)
            if actual_pricing["pricing_spec_id"] != deployment_spec["actual_pricing_spec_id"]:
                raise ValueError(
                    f"PricingInstance {actual_pricing_instance_id!r} is not an instance of "
                    f"PricingSpec {deployment_spec['actual_pricing_spec_id']!r}"
                )
            if not self.pricing.compatible(actual_pricing, kind):
                raise ValueError(
                    f"PricingInstance {actual_pricing_instance_id!r} does not support "
                    f"deployment kind {kind!r}"
                )
            self.pricing.validate_actual_binding(actual_pricing, deployment_spec)
            self.pricing.validate_model_binding(
                actual_pricing,
                instance,
                purpose="actual-cost",
                source_spec_id=str(deployment_spec["source_spec"]["id"]),
                require_source_scope=(
                    actual_pricing.get("pricing_spec", {}).get("basis") == "token_usage"
                ),
            )
            equivalent_id = instance.get("api_equivalent_pricing_instance_id")
            equivalent_spec_id = deployment_spec.get("api_equivalent_pricing_spec_id")
            if equivalent_spec_id and not equivalent_id:
                raise ValueError(
                    f"DeploymentInstance {instance_id!r} requires an API-equivalent PricingInstance"
                )
            if equivalent_id:
                equivalent = self.pricing.get_instance(str(equivalent_id))
                if equivalent["pricing_spec_id"] != equivalent_spec_id:
                    raise ValueError(
                        f"PricingInstance {equivalent_id!r} is not an instance of "
                        f"PricingSpec {equivalent_spec_id!r}"
                    )
                self.pricing.validate_model_binding(
                    equivalent,
                    instance,
                    purpose="API-equivalent",
                )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return {
                **instance,
                "observed_status": "invalid",
                "status": "invalid",
                "available": False,
                "health_detail": f"invalid pricing binding: {exc}",
            }

        value = dict(instance)
        if kind == "hf":
            model_path = Path(str(value.get("model_path") or "")).expanduser()
            python = Path(str(value.get("python") or "")).expanduser()
            missing = []
            if not model_path.exists():
                missing.append(f"model_path={model_path}")
            if not python.is_file():
                missing.append(f"python={python}")
            if missing:
                value.update(
                    status="unavailable",
                    available=False,
                    health_detail="missing " + ", ".join(missing),
                )
            else:
                value.update(
                    status="ready_on_run",
                    available=True,
                    health_detail=value.get("health_detail")
                    or "model path and Python executable are available",
                )
        elif probe:
            health = self._probe_api(value)
            previous_accounting = str(value.get("reasoning_token_accounting") or "")
            if health.get(
                "reasoning_token_accounting_probe"
            ) == "inconclusive" and previous_accounting in {
                "provider_usage",
                "client_retokenized_required",
            }:
                health["reasoning_token_accounting"] = previous_accounting
                health["reasoning_token_accounting_source"] = "previous_definitive_probe"
            else:
                health["reasoning_token_accounting_source"] = "latest_probe"
            value.update(health, checked_at_utc=_utc_now())
            value["available"] = value.get("status") == "running"
        else:
            value["available"] = value.get("status") == "running"
        value["observed_status"] = value.get("status")
        value["actual_pricing_status"] = self.pricing.completeness(actual_pricing)
        return value

    def _write_instance(self, instance: dict[str, Any]) -> None:
        instance_id = str(instance.get("id") or "")
        self._validate_id(instance_id)
        value = dict(instance)
        value.pop("registry_path", None)
        value.pop("deployment_spec", None)
        value.pop("actual_pricing_instance", None)
        value.pop("api_equivalent_pricing_instance", None)
        if not value.get("deployment_spec_fingerprint"):
            deployment_spec = next(
                (item for item in self.specs() if item["id"] == value.get("deployment_spec_id")),
                None,
            )
            if deployment_spec is None:
                raise ValueError("DeploymentInstance references an unknown DeploymentSpec")
            value["deployment_spec_fingerprint"] = _deployment_spec_fingerprint(deployment_spec)
        value.setdefault("created_at_utc", _utc_now())
        value.setdefault("creation_source", "studio")
        self.instance_root.mkdir(parents=True, exist_ok=True)
        path = self.instance_root / f"{instance_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _failure_reason(log_path: Any) -> str | None:
        path = Path(str(log_path or ""))
        if not path.is_file():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        markers = (
            "ImportError:",
            "AttributeError:",
            "RuntimeError:",
            "CUDA out of memory",
            "Address already in use",
            "EngineCore failed to start",
        )
        recent_lines = lines[-400:]
        for marker in markers:
            for line in reversed(recent_lines):
                if marker in line:
                    return line.strip()[-1000:]
        return None

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("deployment id may contain only letters, numbers, '.', '_' and '-'")
