"""Load and query the current metric contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .contracts import MetricContract


class MetricRegistry:
    """Immutable-in-use registry assembled from one current contract per metric."""

    def __init__(
        self,
        contract_dirs: Iterable[str | Path] | None = None,
        *,
        include_builtin: bool = True,
    ) -> None:
        directories: list[Path] = []
        if include_builtin:
            directories.append(Path(__file__).with_name("contracts"))
        directories.extend(Path(item).expanduser().resolve() for item in contract_dirs or ())
        self._contracts: dict[str, MetricContract] = {}
        self._sources: dict[str, str] = {}
        for directory in directories:
            self._load_directory(directory)

    def _load_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot load metric contract {path}: {exc}") from exc
            documents = payload if isinstance(payload, list) else [payload]
            if any(not isinstance(item, dict) for item in documents):
                raise ValueError(f"metric contract file must contain object(s): {path}")
            for document in documents:
                contract = MetricContract.from_mapping(document)
                key = contract.metric_id
                if key in self._contracts:
                    raise ValueError(
                        f"duplicate MetricContract {contract.metric_id}: "
                        f"{self._sources[key]} and {path}"
                    )
                self._contracts[key] = contract
                self._sources[key] = str(path)

    def get(self, metric_id: str) -> MetricContract:
        try:
            return self._contracts[metric_id]
        except KeyError as exc:
            raise KeyError(f"unknown MetricContract {metric_id}") from exc

    def all(
        self,
        *,
        category: str | None = None,
    ) -> list[MetricContract]:
        values = list(self._contracts.values())
        if category is not None:
            values = [contract for contract in values if contract.category == category]
        return sorted(
            values,
            key=lambda contract: (
                contract.category,
                contract.metric_id,
            ),
        )

    @property
    def fingerprint(self) -> str:
        values = [contract.fingerprint for contract in self.all()]
        return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()

    def descriptor(self) -> dict:
        contracts = self.all()
        return {
            "schema_version": 1,
            "registry_fingerprint": self.fingerprint,
            "num_contracts": len(contracts),
            "categories": sorted({contract.category for contract in contracts}),
            "contracts": [contract.to_dict() for contract in contracts],
        }
