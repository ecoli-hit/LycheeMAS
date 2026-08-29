"""Materialize one model backend from frozen Deployment and generation settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendAssemblyRequest:
    """Resolved inputs required to construct the Run's model backend."""

    backend_provider: str
    method: str
    model_tag: str
    api_model: str | None
    model_path: str | None
    device: str
    enable_thinking: bool
    dtype_name: str
    vision: Any
    deployment_document: dict[str, Any] | None
    graph: Any
    generation: dict[str, Any]
    api_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendAssembly:
    """Backend plus optional role-aware pool and startup observations."""

    backend: Any
    deployment_pool: Any | None
    startup_reports: list[dict[str, Any]]


def assemble_backend(request: BackendAssemblyRequest) -> BackendAssembly:
    """Build HF, API/vLLM, or role-bound DeploymentPool without Runner policy."""

    generation = dict(request.generation)
    backend: Any
    deployment_document = request.deployment_document
    if deployment_document:
        from lychee_mas.runtime.adapters.inference.deployment_pool import DeploymentPool

        pool = DeploymentPool(deployment_document, generation=generation)
        pool.bind_graph(request.graph)
        if request.method in {"latent_only", "both"} and (
            len(pool.configs) != 1 or pool.has_kind("api") or pool.has_kind("vllm")
        ):
            raise SystemExit(
                "latent_only/both currently require one shared HF deployment for every role"
            )
        return BackendAssembly(
            backend=pool.resolve(),
            deployment_pool=pool,
            startup_reports=pool.prepare_observability(),
        )

    if request.backend_provider == "api":
        if not request.api_model:
            raise SystemExit("API backend requires backend.model or --api-model")
        from lychee_mas.runtime.adapters.inference.openai_compatible import OpenAICompatibleBackend

        options = dict(request.api_options)
        extra_body = dict(options.pop("extra_body", {}) or {})
        extra_body.update(
            {
                key: value
                for key, value in {
                    "top_k": generation.get("top_k"),
                    "min_p": generation.get("min_p"),
                    "presence_penalty": generation.get("presence_penalty"),
                    "repetition_penalty": generation.get("repetition_penalty"),
                }.items()
                if value is not None
            }
        )
        backend = OpenAICompatibleBackend(
            request.api_model,
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.7)),
            top_p=float(generation.get("top_p", 0.8)),
            seed=int(generation.get("seed", 0)),
            extra_body=extra_body,
            **options,
        )
        report = {
            "deployment_instance_id": "cli-api-backend",
            "kind": "api",
            "model_id": request.api_model,
            **backend.startup_observability(),
        }
        return BackendAssembly(backend=backend, deployment_pool=None, startup_reports=[report])

    if request.backend_provider == "hf":
        if not request.model_path:
            raise SystemExit("HF backend requires a model path")
        import torch

        from lychee_mas.runtime.adapters.inference.hf import HFBackend

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        backend = HFBackend(
            request.model_path,
            device=request.device,
            dtype=dtypes.get(request.dtype_name, torch.bfloat16),
            enable_thinking=request.enable_thinking,
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.7)),
            top_p=float(generation.get("top_p", 0.8)),
            top_k=generation.get("top_k"),
            min_p=generation.get("min_p"),
            presence_penalty=generation.get("presence_penalty"),
            seed=int(generation.get("seed", 0)),
            vision=request.vision,
            max_input_tokens=generation.get("max_input_tokens"),
            max_repeated_token_run=int(generation.get("max_repeated_token_run", 0)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        )
        report = {
            "deployment_instance_id": "cli-hf-backend",
            "kind": "hf",
            "model_id": request.model_tag,
            "reasoning_token_accounting": "backend_native",
            "tokenizer_warmup_status": "not_applicable",
        }
        return BackendAssembly(backend=backend, deployment_pool=None, startup_reports=[report])

    raise SystemExit(
        f"unknown backend.provider {request.backend_provider!r}; choose hf or api"
    )
