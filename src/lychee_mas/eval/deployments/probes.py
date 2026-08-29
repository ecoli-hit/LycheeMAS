"""Runtime health probes for materialized DeploymentInstances.

The registry owns persistence and validation. This module owns the side
effects required to prove that one deployment can answer a minimal request.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def reasoning_token_accounting_probe(payload: Any) -> dict[str, Any]:
    """Describe whether a provider reports reasoning-token usage directly."""

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


def probe_openai_endpoint(
    base_url: str,
    api_key_env: Any,
    *,
    model_id: Any = None,
    auth_mode: Any = "env",
    trust_env: bool = True,
    thinking_budget_field: str | None = None,
) -> dict[str, Any]:
    """Run one minimal OpenAI-compatible models or chat-completion request."""

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
                reasoning_token_accounting_probe(payload) if resolved_model else {}
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


def probe_local_hf(repo_root: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Start a short-lived process and prove one local HF generation succeeds."""

    python = str(Path(str(value["python"])).expanduser())
    model_path = str(Path(str(value["model_path"])).expanduser())
    device = str(value.get("device") or "cuda:0")
    program = (
        "import json,sys\n"
        "from lychee_mas.runtime.adapters.inference.hf import HFBackend\n"
        "backend=HFBackend(model_name=sys.argv[1],device=sys.argv[2],"
        "strict_hidden=False,enable_thinking=False,max_input_tokens=512)\n"
        "result=backend.generate_chat([{'role':'user','content':'你好'}],"
        "max_new_tokens=8)\n"
        "print('__LYCHEE_PROBE__'+json.dumps({'text':result.text,"
        "'latency_s':result.latency_s},ensure_ascii=False))\n"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(value.get("cuda_visible_devices") or "0")
    source_root = str(repo_root / "src")
    environment["PYTHONPATH"] = ":".join(
        item for item in (source_root, environment.get("PYTHONPATH", "")) if item
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    timeout = int(value.get("probe_timeout_s") or 900)
    try:
        result = subprocess.run(
            [python, "-c", program, model_path, device],
            cwd=str(repo_root),
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
