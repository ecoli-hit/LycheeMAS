"""File-backed PricingSpec and PricingInstance registries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..contracts.lifecycle import spec_fingerprint, utc_now
from ..infrastructure.json_store import atomic_write_json as _write_json

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_BASES = {"token_usage", "allocated_gpu_time"}
_DEPLOYMENT_KINDS = {"api", "vllm", "hf"}
_BILLING_MODES = {
    "token_usage": {"provider_token_usage"},
    "allocated_gpu_time": {
        "on_demand_gpu_hour",
        "monthly_node_amortized",
        "internal_gpu_hour",
    },
}
_DEFAULT_BILLING_MODE = {
    "token_usage": "provider_token_usage",
    "allocated_gpu_time": "internal_gpu_hour",
}
_REQUIRED_SPEC_METADATA = {
    "provider_token_usage": ["model_ids"],
    "on_demand_gpu_hour": ["accelerator", "provider", "quoted_gpu_hour"],
    "monthly_node_amortized": [
        "accelerator",
        "provider",
        "quoted_node_month",
        "node_gpu_count",
    ],
    "internal_gpu_hour": [],
}
_RATE_CONTRACTS = {
    "token_usage": {
        "rate_unit": "per_million_tokens",
        "required": {"input", "output"},
        "allowed": {
            "input",
            "output",
            "cached_input",
            "reasoning_output",
            "thinking_output",
        },
    },
    "allocated_gpu_time": {
        "rate_unit": "per_gpu_hour",
        "required": {"gpu_hour"},
        "allowed": {"gpu_hour"},
    },
}


def _utc_now() -> str:
    return utc_now()


def _pricing_spec_fingerprint(spec: dict[str, Any]) -> str:
    """Hash every price-bearing field while ignoring registry presentation data."""

    return spec_fingerprint(
        spec,
        ignored_fields={"missing_rates", "missing_model_scope"},
    )


class PricingRegistry:
    """Store complete price configurations and their materialized bindings."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        root = self.repo_root / "configs/eval_studio/pricing"
        self.spec_root = root / "specs"
        self.instance_root = root / "instances"

    def specs(self) -> list[dict[str, Any]]:
        rows = self._all(self.spec_root, self.validate_spec)
        return [{**item, **self.completeness(item)} for item in rows]

    def instances(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.instance_root.is_dir():
            return rows
        for path in sorted(self.instance_root.glob("*.json")):
            try:
                instance = self.validate_instance(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, TypeError, ValueError):
                continue
            try:
                spec = self.get_spec(instance["pricing_spec_id"])
            except (FileNotFoundError, ValueError) as exc:
                rows.append(
                    {
                        **instance,
                        "registry_path": str(path),
                        "status": "invalid",
                        "observed_status": "invalid",
                        "available": False,
                        "validation_error": str(exc),
                    }
                )
                continue
            expected = _pricing_spec_fingerprint(spec)
            if instance["pricing_spec_fingerprint"] != expected:
                rows.append(
                    {
                        **instance,
                        "pricing_spec": spec,
                        "registry_path": str(path),
                        "status": "invalid",
                        "observed_status": "invalid",
                        "available": False,
                        "validation_error": (
                            "PricingSpec changed after this PricingInstance was created; "
                            "instantiate a new PricingInstance"
                        ),
                    }
                )
                continue
            rows.append(self._resolve(instance, spec, path))
        return rows

    def get_spec(self, spec_id: str) -> dict[str, Any]:
        self._validate_id(spec_id)
        path = self.spec_root / f"{spec_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = self.validate_spec(json.loads(path.read_text(encoding="utf-8")))
        return {**value, "registry_path": str(path)}

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        self._validate_id(instance_id)
        path = self.instance_root / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        instance = self.validate_instance(json.loads(path.read_text(encoding="utf-8")))
        spec = self.get_spec(instance["pricing_spec_id"])
        if instance["pricing_spec_fingerprint"] != _pricing_spec_fingerprint(spec):
            raise ValueError(
                f"PricingInstance {instance_id!r} is stale because its PricingSpec changed"
            )
        return self._resolve(instance, spec, path)

    def save_spec(self, spec_id: str, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("id") and str(value["id"]) != spec_id:
            raise ValueError("PricingSpec id does not match URL")
        normalized = self.validate_spec({**value, "id": spec_id})
        normalized["updated_at_utc"] = _utc_now()
        path = self.spec_root / f"{spec_id}.json"
        _write_json(path, normalized)
        return {**normalized, **self.completeness(normalized), "registry_path": str(path)}

    def instantiate_instance(
        self,
        spec_id: str,
        instance_id: str,
        value: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Materialize the current version of one complete PricingSpec."""

        self._validate_id(spec_id)
        self._validate_id(instance_id)
        supplied = dict(value or {})
        unknown = set(supplied) - {"id", "pricing_spec_id"}
        if unknown:
            raise ValueError(
                "PricingInstance accepts only id and pricing_spec_id; move fields to "
                "PricingSpec: " + ", ".join(sorted(unknown))
            )
        if supplied.get("id") not in {None, "", instance_id}:
            raise ValueError("PricingInstance id does not match request")
        if supplied.get("pricing_spec_id") not in {None, "", spec_id}:
            raise ValueError("PricingInstance pricing_spec_id does not match PricingSpec")
        spec = self.get_spec(spec_id)
        path = self.instance_root / f"{instance_id}.json"
        if path.exists():
            raise ValueError(
                f"PricingInstance {instance_id!r} already exists; use a new ID"
            )
        instance = self.validate_instance(
            {
                "schema_version": 2,
                "id": instance_id,
                "pricing_spec_id": spec_id,
                "pricing_spec_fingerprint": _pricing_spec_fingerprint(spec),
                "created_at_utc": _utc_now(),
                "creation_source": "studio",
            }
        )
        _write_json(path, instance)
        return self._resolve(instance, spec, path)

    def delete_spec(self, spec_id: str) -> dict[str, Any]:
        if any(item.get("pricing_spec_id") == spec_id for item in self.instances()):
            raise ValueError("PricingSpec is referenced by a PricingInstance")
        return self._delete(self.spec_root, spec_id)

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        return self._delete(self.instance_root, instance_id)

    def validate_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("PricingSpec must be a JSON object")
        normalized = dict(value)
        spec_id = str(normalized.get("id") or "")
        self._validate_id(spec_id)
        basis = str(normalized.get("basis") or "")
        if basis not in _BASES:
            raise ValueError("PricingSpec basis must be token_usage or allocated_gpu_time")
        billing_mode = str(
            normalized.get("billing_mode") or _DEFAULT_BILLING_MODE[basis]
        )
        if billing_mode not in _BILLING_MODES[basis]:
            raise ValueError(
                f"PricingSpec basis {basis!r} does not support billing_mode "
                f"{billing_mode!r}"
            )
        contract = _RATE_CONTRACTS[basis]
        rate_unit = str(normalized.get("rate_unit") or contract["rate_unit"])
        if rate_unit != contract["rate_unit"]:
            raise ValueError(
                f"PricingSpec basis {basis!r} requires rate_unit "
                f"{contract['rate_unit']!r}"
            )
        kinds = [str(item) for item in normalized.get("supported_deployment_kinds") or []]
        if not kinds or any(item not in _DEPLOYMENT_KINDS for item in kinds):
            raise ValueError(
                "PricingSpec supported_deployment_kinds must contain api, vllm, or hf"
            )
        required_rates = [str(item) for item in normalized.get("required_rates") or []]
        optional_rates = [str(item) for item in normalized.get("optional_rates") or []]
        if not required_rates:
            raise ValueError("PricingSpec requires at least one required rate")
        if len(set(required_rates + optional_rates)) != len(required_rates + optional_rates):
            raise ValueError("PricingSpec rate names must be unique")
        if set(required_rates) != contract["required"]:
            raise ValueError(
                f"PricingSpec basis {basis!r} requires rates "
                f"{sorted(contract['required'])}"
            )
        unknown_rates = set(required_rates + optional_rates) - set(contract["allowed"])
        if unknown_rates:
            raise ValueError(
                "PricingSpec has unsupported rate names: "
                + ", ".join(sorted(unknown_rates))
            )
        currency = str(normalized.get("currency") or "").upper()
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("PricingSpec currency must be a three-letter ISO code")
        metadata = self._normalize_metadata(normalized.get("metadata") or {})
        rates = self._normalize_rates(normalized.get("rates") or {}, "rates")
        rate_tiers = self._normalize_rate_tiers(normalized.get("rate_tiers") or [])
        normalized.update(
            schema_version=2,
            id=spec_id,
            basis=basis,
            billing_mode=billing_mode,
            rate_card_mode=(
                "flat_or_input_token_tiers" if basis == "token_usage" else "flat"
            ),
            rate_unit=rate_unit,
            supported_deployment_kinds=kinds,
            required_rates=required_rates,
            optional_rates=optional_rates,
            currency=currency,
            rates=rates,
            rate_tiers=rate_tiers,
            metadata=metadata,
        )
        if billing_mode == "monthly_node_amortized":
            policy = str(
                normalized.get("amortization_policy") or "allocated_gpu_share"
            )
            if policy != "allocated_gpu_share":
                raise ValueError(
                    "monthly GPU PricingSpec currently requires "
                    "amortization_policy='allocated_gpu_share'"
                )
            hours = float(normalized.get("amortization_hours_per_month") or 0)
            if hours <= 0:
                raise ValueError(
                    "monthly GPU PricingSpec requires a positive "
                    "amortization_hours_per_month"
                )
            normalized.update(
                amortization_policy=policy,
                amortization_hours_per_month=hours,
            )
        else:
            normalized.pop("amortization_policy", None)
            normalized.pop("amortization_hours_per_month", None)
        for field in (
            "registry_path",
            "available",
            "status",
            "complete",
            "missing_rates",
            "missing_model_scope",
        ):
            normalized.pop(field, None)
        self._validate_spec_rates(normalized)
        return normalized

    def validate_instance(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("PricingInstance must be a JSON object")
        allowed = {
            "schema_version",
            "id",
            "pricing_spec_id",
            "pricing_spec_fingerprint",
            "created_at_utc",
            "creation_source",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "PricingInstance contains Spec-owned fields: "
                + ", ".join(sorted(unknown))
            )
        instance_id = str(value.get("id") or "")
        pricing_spec_id = str(value.get("pricing_spec_id") or "")
        self._validate_id(instance_id)
        self._validate_id(pricing_spec_id)
        fingerprint = str(value.get("pricing_spec_fingerprint") or "")
        if not _FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("PricingInstance requires a valid pricing_spec_fingerprint")
        created_at = str(value.get("created_at_utc") or "")
        if not created_at:
            raise ValueError("PricingInstance requires created_at_utc")
        creation_source = str(value.get("creation_source") or "")
        if not creation_source:
            raise ValueError("PricingInstance requires creation_source")
        return {
            "schema_version": 2,
            "id": instance_id,
            "pricing_spec_id": pricing_spec_id,
            "pricing_spec_fingerprint": fingerprint,
            "created_at_utc": created_at,
            "creation_source": creation_source,
        }

    def _resolve(
        self,
        instance: dict[str, Any],
        spec: dict[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        concrete = {
            key: item
            for key, item in spec.items()
            if key not in {"id", "registry_path", "updated_at_utc"}
        }
        completeness = self.completeness(spec)
        return {
            **concrete,
            **instance,
            "pricing_spec": spec,
            **completeness,
            "observed_status": completeness["status"],
            "available": True,
            "registry_path": str(path),
        }

    @staticmethod
    def compatible(pricing: dict[str, Any], deployment_kind: str) -> bool:
        spec = dict(pricing.get("pricing_spec") or pricing)
        return deployment_kind in set(spec.get("supported_deployment_kinds") or [])

    @staticmethod
    def required_actual_basis(deployment: dict[str, Any]) -> str:
        """Return the accounting basis implied by one deployment lifecycle."""

        kind = str(deployment.get("kind") or "")
        managed_vllm = kind == "vllm" and deployment.get("managed", True) is not False
        if kind == "hf" or managed_vllm:
            return "allocated_gpu_time"
        if kind == "api" or kind == "vllm":
            return "token_usage"
        raise ValueError(f"unsupported deployment kind {kind!r}")

    @classmethod
    def validate_actual_binding(
        cls, pricing: dict[str, Any], deployment: dict[str, Any]
    ) -> None:
        spec = dict(pricing.get("pricing_spec") or pricing)
        actual = str(spec.get("basis") or "")
        required = cls.required_actual_basis(deployment)
        if actual != required:
            lifecycle = (
                "managed local service"
                if str(deployment.get("kind")) == "vllm"
                and deployment.get("managed", True) is not False
                else "external/provider service"
                if str(deployment.get("kind")) in {"api", "vllm"}
                else "in-process local backend"
            )
            raise ValueError(
                f"{deployment.get('kind')} {lifecycle} requires actual pricing basis "
                f"{required!r}, got {actual!r}"
            )

    @staticmethod
    def applies_to_model(pricing: dict[str, Any], model_id: Any) -> bool:
        """Return whether a token rate card explicitly covers one served model."""

        spec = dict(pricing.get("pricing_spec") or pricing)
        if spec.get("basis") != "token_usage":
            return True
        configured = (pricing.get("metadata") or spec.get("metadata") or {}).get(
            "model_ids"
        ) or []
        expected = str(model_id or "").strip().casefold()
        return bool(expected) and any(
            str(item).strip() == "*" or str(item).strip().casefold() == expected
            for item in configured
        )

    @classmethod
    def validate_model_binding(
        cls,
        pricing: dict[str, Any],
        deployment: dict[str, Any],
        *,
        purpose: str,
        source_spec_id: str | None = None,
        require_source_scope: bool = False,
    ) -> None:
        """Prevent a provider/model rate card from being bound to another model."""

        spec = dict(pricing.get("pricing_spec") or pricing)
        if spec.get("basis") != "token_usage":
            return
        metadata = dict(pricing.get("metadata") or spec.get("metadata") or {})
        model_ids = metadata.get("model_ids") or []
        if not model_ids:
            raise ValueError(
                f"token PricingSpec {spec.get('id')!r} requires metadata.model_ids"
            )
        model_id = str(deployment.get("model_id") or "").strip()
        if not cls.applies_to_model(pricing, model_id):
            raise ValueError(
                f"{purpose} PricingInstance {pricing.get('id')!r} does not apply to "
                f"deployment model_id {model_id!r}"
            )
        source_ids = metadata.get("source_spec_ids") or []
        if require_source_scope and not source_ids:
            raise ValueError(
                f"{purpose} token PricingSpec {spec.get('id')!r} requires "
                "metadata.source_spec_ids"
            )
        if source_ids and source_spec_id and not any(
            str(item).strip() == "*"
            or str(item).strip().casefold() == source_spec_id.strip().casefold()
            for item in source_ids
        ):
            raise ValueError(
                f"{purpose} PricingInstance {pricing.get('id')!r} does not apply to "
                f"source Spec {source_spec_id!r}"
            )

    @staticmethod
    def completeness(pricing: dict[str, Any]) -> dict[str, Any]:
        spec = dict(pricing.get("pricing_spec") or pricing)
        required = list(spec.get("required_rates") or [])
        rate_tiers = list(spec.get("rate_tiers") or [])
        if rate_tiers:
            missing = [
                f"tier<={tier.get('up_to_input_tokens')}:{key}"
                for tier in rate_tiers
                for key in required
                if (tier.get("rates") or {}).get(key) is None
            ]
        else:
            rates = dict(spec.get("rates") or {})
            missing = [key for key in required if rates.get(key) is None]
        missing_model_scope = bool(
            spec.get("basis") == "token_usage"
            and not (spec.get("metadata") or {}).get("model_ids")
        )
        status = (
            "needs_model_scope"
            if missing_model_scope
            else "needs_rate"
            if missing
            else "ready"
        )
        return {
            "status": status,
            "complete": not missing and not missing_model_scope,
            "missing_rates": missing,
            "missing_model_scope": missing_model_scope,
        }

    def _validate_spec_rates(self, spec: dict[str, Any]) -> None:
        allowed = set(spec.get("required_rates") or []) | set(
            spec.get("optional_rates") or []
        )
        rate_sets = [("rates", spec.get("rates") or {})]
        rate_sets.extend(
            (f"rate_tiers[{index}].rates", tier.get("rates") or {})
            for index, tier in enumerate(spec.get("rate_tiers") or [])
        )
        for label, rates in rate_sets:
            unknown = set(rates) - allowed
            if unknown:
                raise ValueError(
                    f"PricingSpec {label} has undeclared rates: "
                    + ", ".join(sorted(unknown))
                )
        metadata = dict(spec.get("metadata") or {})
        required_metadata = _REQUIRED_SPEC_METADATA[spec["billing_mode"]]
        missing_metadata = [
            key
            for key in required_metadata
            if metadata.get(key) is None
            or metadata.get(key) == ""
            or metadata.get(key) == []
        ]
        if missing_metadata:
            raise ValueError(
                "PricingSpec is missing required metadata: "
                + ", ".join(missing_metadata)
            )
        if spec["basis"] == "allocated_gpu_time":
            self._validate_allocated_gpu_spec(spec)

    @staticmethod
    def _validate_allocated_gpu_spec(spec: dict[str, Any]) -> None:
        metadata = dict(spec.get("metadata") or {})
        billing_mode = spec.get("billing_mode")
        rate = (spec.get("rates") or {}).get("gpu_hour")
        if billing_mode == "on_demand_gpu_hour":
            quoted = float(metadata["quoted_gpu_hour"])
            if rate is None or abs(quoted - float(rate)) > 1e-9:
                raise ValueError(
                    "on-demand rates.gpu_hour must equal metadata.quoted_gpu_hour"
                )
            return
        if billing_mode != "monthly_node_amortized":
            return
        node_month = float(metadata["quoted_node_month"])
        node_gpu_count = int(metadata["node_gpu_count"])
        month_hours = float(spec["amortization_hours_per_month"])
        if node_month < 0 or node_gpu_count < 1 or month_hours <= 0:
            raise ValueError("monthly GPU pricing values must be positive")
        derived = node_month / node_gpu_count / month_hours
        if rate is None or abs(float(rate) - derived) > 1e-9:
            raise ValueError(
                "monthly rates.gpu_hour must equal quoted_node_month / "
                "node_gpu_count / amortization_hours_per_month"
            )

    @staticmethod
    def _normalize_rates(value: Any, label: str) -> dict[str, float | None]:
        if not isinstance(value, dict):
            raise ValueError(f"PricingSpec {label} must be an object")
        rates: dict[str, float | None] = {}
        for key, item in value.items():
            if item is None:
                rates[str(key)] = None
                continue
            rate = float(item)
            if rate < 0:
                raise ValueError(
                    f"PricingSpec {label} rate {key!r} must be non-negative"
                )
            rates[str(key)] = rate
        return rates

    def _normalize_rate_tiers(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise ValueError("PricingSpec rate_tiers must be a list")
        tiers: list[dict[str, Any]] = []
        previous_limit = 0
        for index, raw_tier in enumerate(value):
            if not isinstance(raw_tier, dict):
                raise ValueError(f"PricingSpec rate_tiers[{index}] must be an object")
            limit = int(raw_tier.get("up_to_input_tokens") or 0)
            if limit <= previous_limit:
                raise ValueError(
                    "PricingSpec rate_tiers must use strictly increasing positive "
                    "up_to_input_tokens values"
                )
            previous_limit = limit
            tiers.append(
                {
                    "up_to_input_tokens": limit,
                    "rates": self._normalize_rates(
                        raw_tier.get("rates") or {}, f"rate_tiers[{index}].rates"
                    ),
                }
            )
        return tiers

    @staticmethod
    def _normalize_metadata(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("PricingSpec metadata must be a JSON object")
        metadata = dict(value)
        for key in ("model_ids", "source_spec_ids"):
            items = metadata.get(key)
            if items is None:
                continue
            if (
                not isinstance(items, list)
                or not items
                or not all(isinstance(item, str) and item.strip() for item in items)
            ):
                raise ValueError(
                    f"PricingSpec metadata.{key} must be a non-empty string list"
                )
            metadata[key] = [str(item).strip() for item in items]
        return metadata

    @staticmethod
    def _all(root: Path, validator) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")):
            try:
                value = validator(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError):
                continue
            rows.append({**value, "registry_path": str(path)})
        return rows

    def _delete(self, root: Path, item_id: str) -> dict[str, Any]:
        self._validate_id(item_id)
        path = root / f"{item_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.unlink()
        return {"id": item_id, "deleted": True, "path": str(path)}

    @staticmethod
    def _validate_id(item_id: str) -> None:
        if not _SAFE_ID.fullmatch(str(item_id or "")):
            raise ValueError(
                "registry id may contain only letters, numbers, '.', '_' and '-'"
            )
