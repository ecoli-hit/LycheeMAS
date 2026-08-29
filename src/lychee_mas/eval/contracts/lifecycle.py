"""Shared Spec-to-Instance lifecycle metadata for Eval Studio registries."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_IGNORED_SPEC_FIELDS = {
    "registry_path",
    "updated_at_utc",
    "notes",
    "description",
    "status",
    "complete",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for immutable creation metadata."""

    return datetime.now(timezone.utc).isoformat()


def spec_fingerprint(
    spec: dict[str, Any], *, ignored_fields: set[str] | None = None
) -> str:
    """Hash behavior-bearing Spec fields, excluding registry presentation data."""

    ignored = _IGNORED_SPEC_FIELDS | set(ignored_fields or ())
    payload = {
        key: value
        for key, value in spec.items()
        if key not in ignored
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bind_instance_to_spec(
    value: dict[str, Any],
    *,
    spec: dict[str, Any],
    prefix: str,
    creation_source: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Attach the common immutable lifecycle envelope to a new Instance."""

    return {
        **value,
        f"{prefix}_spec_id": str(spec["id"]),
        f"{prefix}_spec_fingerprint": spec_fingerprint(spec),
        "created_at_utc": str(created_at_utc or utc_now()),
        "creation_source": str(creation_source or "studio"),
    }


def normalize_instance_lifecycle(
    value: dict[str, Any], *, prefix: str
) -> dict[str, str]:
    """Validate and return the common immutable lifecycle envelope."""

    spec_id = str(value.get(f"{prefix}_spec_id") or "")
    fingerprint = str(value.get(f"{prefix}_spec_fingerprint") or "")
    created_at = str(value.get("created_at_utc") or "")
    creation_source = str(value.get("creation_source") or "")
    if not spec_id:
        raise ValueError(f"Instance requires {prefix}_spec_id")
    if not _FINGERPRINT.fullmatch(fingerprint):
        raise ValueError(f"Instance requires a valid {prefix}_spec_fingerprint")
    if not created_at:
        raise ValueError("Instance requires created_at_utc")
    if not creation_source:
        raise ValueError("Instance requires creation_source")
    return {
        f"{prefix}_spec_id": spec_id,
        f"{prefix}_spec_fingerprint": fingerprint,
        "created_at_utc": created_at,
        "creation_source": creation_source,
    }


def lifecycle_error(
    instance: dict[str, Any], *, spec: dict[str, Any] | None, prefix: str
) -> str | None:
    """Explain why an Instance no longer matches its referenced Spec."""

    if spec is None:
        return f"referenced {prefix.title()}Spec is missing"
    stored = str(instance.get(f"{prefix}_spec_fingerprint") or "")
    if stored != spec_fingerprint(spec):
        return (
            f"{prefix.title()}Spec changed after this Instance was created; "
            "instantiate a new Instance"
        )
    return None
