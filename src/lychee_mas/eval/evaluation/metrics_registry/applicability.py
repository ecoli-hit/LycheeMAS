"""Resolve whether registered metrics are measurable for one run."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import MetricContract
from .registry import MetricRegistry

METRIC_APPLICABILITY_SCHEMA_VERSION = 1
METRIC_APPLICABILITY_FILENAME = "metric_applicability.json"


def _assessment(
    contract: MetricContract,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    raw_summary = coverage.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    total_events = int(summary.get("total_events") or 0)
    capabilities = summary.get("capabilities_observed")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    requirements = {
        str(item.get("requirement")): item
        for item in coverage.get("requirements") or []
        if isinstance(item, dict) and item.get("requirement")
    }
    reasons: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    if total_events == 0:
        reasons.append({"code": "no_evidence_events"})
        status = "missing_evidence"
    else:
        status = "measurable"
        for capability in contract.required_capabilities:
            if capability not in capabilities:
                reasons.append(
                    {"code": "unknown_capability", "capability": capability}
                )
                status = "unsupported"
            elif not bool(capabilities[capability]) and status != "unsupported":
                reasons.append(
                    {"code": "capability_not_observed", "capability": capability}
                )
                status = "not_applicable"

        if status == "measurable":
            for requirement in contract.required_evidence:
                row = requirements.get(requirement)
                if row is None:
                    evidence_rows.append(
                        {"requirement": requirement, "coverage_status": "unknown"}
                    )
                    reasons.append(
                        {"code": "unknown_evidence_requirement", "requirement": requirement}
                    )
                    status = "unsupported"
                    continue
                coverage_status = str(row.get("status") or "unknown")
                evidence_rows.append(
                    {
                        "requirement": requirement,
                        "coverage_status": coverage_status,
                        "coverage_ratio": row.get("coverage_ratio"),
                        "eligible_events": row.get("eligible_events", 0),
                        "present_events": row.get("present_events", 0),
                    }
                )
                if coverage_status != "complete" and status != "unsupported":
                    reasons.append(
                        {
                            "code": "required_evidence_incomplete",
                            "requirement": requirement,
                            "coverage_status": coverage_status,
                        }
                    )
                    status = "missing_evidence"

    optional_rows = []
    for requirement in contract.optional_evidence:
        row = requirements.get(requirement)
        optional_rows.append(
            {
                "requirement": requirement,
                "coverage_status": str(row.get("status") or "unknown") if row else "unknown",
                "coverage_ratio": row.get("coverage_ratio") if row else None,
            }
        )
    return {
        "metric_id": contract.metric_id,
        "contract_fingerprint": contract.fingerprint,
        "category": contract.category,
        "name": contract.name,
        "unit": contract.unit,
        "level": contract.level,
        "status": status,
        "reasons": reasons,
        "required_evidence": evidence_rows,
        "optional_evidence": optional_rows,
        "validation_status": contract.validation_status,
    }


def assess_metric_applicability(
    coverage: dict[str, Any],
    *,
    registry: MetricRegistry | None = None,
) -> dict[str, Any]:
    """Assess all latest contracts against one Evidence Coverage Report."""

    registry = registry or MetricRegistry()
    assessments = [_assessment(contract, coverage) for contract in registry.all()]
    statuses = Counter(item["status"] for item in assessments)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assessments:
        by_category[item["category"]][item["status"]] += 1
    return {
        "schema_version": METRIC_APPLICABILITY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_fingerprint": registry.fingerprint,
        "coverage_schema_version": coverage.get("schema_version"),
        "coverage_generated_at_utc": coverage.get("generated_at_utc"),
        "summary": {
            "total_contracts": len(assessments),
            "status_counts": dict(sorted(statuses.items())),
            "by_category": {
                category: dict(sorted(counts.items()))
                for category, counts in sorted(by_category.items())
            },
        },
        "assessments": assessments,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_run_metric_applicability(
    run_dir: str | os.PathLike,
    *,
    coverage_report: dict[str, Any] | None = None,
    registry: MetricRegistry | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Materialize the metric gate for one run without evaluating metric values."""

    root = Path(run_dir).expanduser().resolve()
    if coverage_report is None:
        coverage_path = root / "evidence_coverage.json"
        if coverage_path.is_file():
            coverage_report = json.loads(coverage_path.read_text(encoding="utf-8"))
        else:
            from ..evidence import normalize_run_evidence

            coverage_report = normalize_run_evidence(root, write=write)
    report = assess_metric_applicability(coverage_report, registry=registry)
    report["run_dir"] = str(root)
    report["artifacts"] = {
        "applicability": METRIC_APPLICABILITY_FILENAME if write else None,
    }
    if write:
        _atomic_write(root / METRIC_APPLICABILITY_FILENAME, report)
    return report
