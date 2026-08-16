"""Lazy backend pool for per-agent Eval Studio deployment bindings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..token_budget import BUDGET_POLICY_FIELDS


class DeploymentPool:
    """Instantiate each configured backend once and resolve it by deployment ID."""

    def __init__(self, document: dict[str, Any], *, generation: dict[str, Any] | None = None):
        deployments = document.get("deployments") or []
        self.configs = {str(item["id"]): dict(item) for item in deployments}
        if not self.configs:
            raise ValueError("deployment config must contain at least one deployment")
        bindings = dict(document.get("team_deployment_bindings") or {})
        default_id = str(bindings.get("default_deployment_id") or next(iter(self.configs)))
        self.control_deployment_id = str(
            bindings.get("control_deployment_id")
            or document.get("control_deployment_id")
            or default_id
        )
        if self.control_deployment_id not in self.configs:
            raise ValueError(f"unknown control deployment {self.control_deployment_id!r}")
        if default_id not in self.configs:
            raise ValueError(f"unknown default deployment {default_id!r}")
        self.default_deployment_id = default_id
        self.role_bindings: dict[str, str] = {}
        self.role_generation_overrides: dict[str, dict[str, Any]] = {}
        for item in bindings.get("role_bindings") or []:
            role_id = str(item.get("role_id") or "")
            deployment_id = str(item.get("deployment_id") or "")
            if not role_id:
                raise ValueError("role deployment binding requires role_id")
            if deployment_id not in self.configs:
                raise ValueError(
                    f"role {role_id!r} references unknown deployment {deployment_id!r}"
                )
            self.role_bindings[role_id] = deployment_id
            self.role_generation_overrides[role_id] = dict(item.get("generation_overrides") or {})
        self.control_generation_overrides = dict(bindings.get("control_generation_overrides") or {})
        self.generation = dict(generation or {})
        self._instances: dict[str, Any] = {}

    @classmethod
    def from_path(
        cls, path: str | Path, *, generation: dict[str, Any] | None = None
    ) -> "DeploymentPool":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("deployment config must be a JSON object")
        return cls(document, generation=generation)

    def resolve(self, deployment_id: str | None = None):
        deployment_id = str(deployment_id or self.default_deployment_id)
        if deployment_id not in self.configs:
            raise ValueError(f"unknown deployment {deployment_id!r}")
        if deployment_id not in self._instances:
            self._instances[deployment_id] = self._build(self.configs[deployment_id])
        return self._instances[deployment_id]

    def model_id(self, deployment_id: str | None = None) -> str:
        config = self.configs[str(deployment_id or self.default_deployment_id)]
        return str(config.get("model_id") or Path(str(config.get("model_path") or "model")).name)

    def kind(self, deployment_id: str | None = None) -> str:
        return str(self.configs[str(deployment_id or self.default_deployment_id)]["kind"])

    def deployment_for_role(self, role_id: str) -> str:
        """Return a role-specific binding or the team's default deployment."""

        return self.role_bindings.get(str(role_id), self.default_deployment_id)

    def max_new_tokens_for_role(self, role_id: str, fallback: int) -> int:
        """Resolve one participant's output budget, falling back to Runtime."""

        value = self.role_generation_overrides.get(str(role_id), {}).get("max_new_tokens", fallback)
        return max(1, int(value))

    def control_max_new_tokens(self, fallback: int) -> int:
        """Resolve the Selector/Orchestrator output budget."""

        return max(
            1,
            int(self.control_generation_overrides.get("max_new_tokens", fallback)),
        )

    def invocation_overrides_for_role(self, role_id: str) -> dict[str, Any]:
        """Translate one participant binding into provider request options."""

        deployment_id = self.deployment_for_role(role_id)
        return self._invocation_overrides(
            self.configs[deployment_id],
            self.role_generation_overrides.get(str(role_id), {}),
        )

    def control_invocation_overrides(self) -> dict[str, Any]:
        """Translate the GroupChat controller binding into request options."""

        return self._invocation_overrides(
            self.configs[self.control_deployment_id],
            self.control_generation_overrides,
        )

    def bind_graph(self, graph) -> None:
        """Attach runtime-only deployment IDs without mutating TeamSpec on disk."""

        for node in graph.nodes:
            node.meta["deployment_id"] = self.role_bindings.get(
                node.name,
                self.role_bindings.get(str(node.id), self.default_deployment_id),
            )
        graph.meta["control_deployment_id"] = self.control_deployment_id

    def has_kind(self, kind: str) -> bool:
        return any(str(config.get("kind")) == kind for config in self.configs.values())

    def prepare_observability(self) -> list[dict[str, Any]]:
        """Warm only backends whose deployment probe requires client token splitting."""

        reports: list[dict[str, Any]] = []
        for deployment_id, config in self.configs.items():
            kind = str(config.get("kind") or "")
            accounting = str(config.get("reasoning_token_accounting") or "inconclusive")
            if kind not in {"api", "vllm"}:
                reports.append(
                    {
                        "deployment_instance_id": deployment_id,
                        "kind": kind,
                        "model_id": self.model_id(deployment_id),
                        "reasoning_token_accounting": "backend_native",
                        "tokenizer_warmup_status": "not_applicable",
                    }
                )
                continue
            if accounting == "client_retokenized_required":
                backend = self.resolve(deployment_id)
                report = dict(backend.startup_observability())
            else:
                report = {
                    "reasoning_token_accounting": accounting,
                    "tokenizer_path": config.get("tokenizer_path") or config.get("model_path"),
                    "tokenizer_warmup_status": "not_required",
                    "tokenizer_warmup_latency_s": 0.0,
                }
            reports.append(
                {
                    "deployment_instance_id": deployment_id,
                    "kind": kind,
                    "model_id": self.model_id(deployment_id),
                    **report,
                }
            )
        return reports

    @staticmethod
    def _thinking_protocol(config: dict[str, Any]) -> str:
        explicit = str(config.get("thinking_protocol") or "").strip()
        if explicit:
            return explicit
        model = str(config.get("model_id") or "").lower()
        base_url = str(config.get("base_url") or "").lower()
        if str(config.get("kind")) == "vllm" and "qwen" in model:
            return "qwen_chat_template"
        if "dashscope" in base_url or "modelstudio" in base_url or "maas" in base_url:
            return "dashscope"
        if "api.deepseek.com" in base_url:
            return "deepseek"
        return "unsupported"

    def _invocation_overrides(self, config: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        policy = {
            **{
                key: self.generation[key]
                for key in (
                    *BUDGET_POLICY_FIELDS,
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
                if self.generation.get(key) is not None
            },
            **dict(raw or {}),
        }
        policy.pop("max_new_tokens", None)
        request: dict[str, Any] = {}
        token_budget_policy = {
            key: policy[key] for key in BUDGET_POLICY_FIELDS if policy.get(key) is not None
        }
        if token_budget_policy:
            request["token_budget_policy"] = token_budget_policy
        for key in ("do_sample", "temperature", "top_p"):
            if policy.get(key) is not None:
                request[key] = policy[key]
        provider_sampling = {
            key: policy[key]
            for key in ("top_k", "min_p", "presence_penalty", "repetition_penalty")
            if policy.get(key) is not None
        }
        extra_body: dict[str, Any] = {}
        if str(config.get("kind")) == "hf":
            request.update(provider_sampling)
        else:
            extra_body.update(provider_sampling)
        mode = str(policy.get("thinking_mode") or "inherit")
        if mode not in {"inherit", "enabled", "disabled"}:
            raise ValueError(f"unsupported thinking_mode {mode!r}")
        preserve = policy.get("preserve_thinking")
        budget = policy.get("max_thinking_budget_tokens")
        protocol = self._thinking_protocol(config)
        if mode != "inherit" or preserve is not None or budget is not None:
            if str(config.get("kind")) == "hf":
                if mode != "inherit":
                    request["enable_thinking"] = mode == "enabled"
                if preserve is not None:
                    request["preserve_thinking"] = bool(preserve)
                if budget is not None:
                    raise ValueError(
                        "local HF cannot enforce max_thinking_budget_tokens; use a "
                        "vLLM/API deployment with native thinking budget support"
                    )
            elif protocol == "qwen_chat_template":
                template: dict[str, Any] = {}
                if mode != "inherit":
                    template["enable_thinking"] = mode == "enabled"
                if preserve is not None:
                    template["preserve_thinking"] = bool(preserve)
                if template:
                    extra_body["chat_template_kwargs"] = template
                if budget is not None:
                    capabilities = dict(config.get("capabilities") or {})
                    if not bool(capabilities.get("native_thinking_budget")):
                        raise ValueError(
                            "local Qwen vLLM deployment does not declare native "
                            "thinking budget support; use vLLM with reasoning_config "
                            "and thinking_token_budget support"
                        )
                    extra_body["thinking_token_budget"] = int(budget)
            elif protocol == "dashscope":
                if mode != "inherit":
                    extra_body["enable_thinking"] = mode == "enabled"
                if preserve is not None:
                    extra_body["preserve_thinking"] = bool(preserve)
                if budget is not None:
                    extra_body["thinking_budget"] = int(budget)
            elif protocol == "deepseek":
                if mode != "inherit":
                    extra_body["thinking"] = {"type": mode}
                if preserve is not None:
                    raise ValueError(
                        "DeepSeek reasoning_content history preservation is controlled "
                        "by the agent message adapter, not a provider request flag"
                    )
                if budget is not None:
                    raise ValueError(
                        "DeepSeek V4 supports reasoning_effort, not a token thinking budget"
                    )
            else:
                raise ValueError("deployment does not declare a supported thinking_protocol")
        if extra_body:
            request["extra_body"] = extra_body
        return request

    def _build(self, config: dict[str, Any]):
        kind = str(config["kind"])
        generation = {**self.generation, **dict(config.get("generation") or {})}
        if kind == "hf":
            import torch

            from .hf_backend import HFBackend

            dtype_name = str(config.get("dtype", "bfloat16"))
            dtypes = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            return HFBackend(
                config.get("model_path"),
                device=str(config.get("runtime_device") or config.get("device", "cuda:0")),
                dtype=dtypes.get(dtype_name, torch.bfloat16),
                enable_thinking=bool(config.get("enable_thinking", False)),
                do_sample=bool(generation.get("do_sample", False)),
                temperature=float(generation.get("temperature", 0.7)),
                top_p=float(generation.get("top_p", 0.8)),
                top_k=generation.get("top_k"),
                min_p=generation.get("min_p"),
                presence_penalty=generation.get("presence_penalty"),
                seed=int(generation.get("seed", 0)),
                vision=config.get("vision", (config.get("model_info") or {}).get("vision")),
                max_input_tokens=generation.get("max_input_tokens"),
                max_repeated_token_run=int(generation.get("max_repeated_token_run", 0)),
                repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
            )
        if kind in {"api", "vllm"}:
            from .openai_api_backend import OpenAICompatibleBackend

            return OpenAICompatibleBackend(
                str(config["model_id"]),
                base_url=config.get("base_url"),
                api_key_env=str(config.get("api_key_env") or "VLLM_API_KEY"),
                auth_mode=str(config.get("auth_mode") or "env"),
                trust_env=bool(config.get("trust_env", True)),
                proxy_url=config.get("proxy_url"),
                timeout=float(config.get("timeout", 120.0)),
                do_sample=bool(generation.get("do_sample", False)),
                temperature=float(generation.get("temperature", 0.7)),
                top_p=float(generation.get("top_p", 0.8)),
                seed=int(generation.get("seed", 0)),
                model_info={
                    **dict(config.get("model_info") or {}),
                    **(
                        {"context_window": int(config["max_model_len"])}
                        if config.get("max_model_len") is not None
                        else {}
                    ),
                },
                request_limits=dict(config.get("request_limits") or {}),
                request_timeout=dict(config.get("request_timeout") or {}),
                max_retries=int(config.get("max_retries", 0)),
                extra_body=dict(config.get("extra_body") or {}),
                tokenizer_path=config.get("tokenizer_path") or config.get("model_path"),
                reasoning_token_accounting=config.get("reasoning_token_accounting"),
            )
        raise ValueError(f"unsupported deployment kind {kind!r}")
