"""Stable random-seed derivation for benchmark predictions."""

from __future__ import annotations

import hashlib
import json

PREDICTION_SEED_DERIVATION = "sha256_case_k_31bit_v1"
_MAX_PROVIDER_SEED_EXCLUSIVE = 2**31


def derive_prediction_seed(
    *,
    base_seed: int,
    benchmark_id: str,
    task: str,
    case_id: str,
    k_index: int,
) -> int:
    """Derive a provider-safe seed from stable prediction identity.

    Dataset position, worker id, concurrency and retry attempt are deliberately
    excluded. The same case/sample therefore keeps the same seed across subsets,
    reordering, resume and infrastructure retries.
    """

    material = {
        "derivation": PREDICTION_SEED_DERIVATION,
        "base_seed": int(base_seed),
        "benchmark_id": str(benchmark_id),
        "task": str(task),
        "case_id": str(case_id),
        "k_index": int(k_index),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % _MAX_PROVIDER_SEED_EXCLUSIVE
