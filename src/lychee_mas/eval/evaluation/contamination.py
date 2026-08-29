"""Non-intervening contamination audit for open-web benchmark runs.

The auditor never changes model inputs, browser results, predictions, or scores.
It derives reviewable evidence from the append-only runtime logs after inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

SCHEMA_VERSION = 1
AUDIT_POLICY = "open_web_audit_only_v1"
SUSPECTED_FLAGS = {
    "benchmark_artifact_access",
    "gold_like_text_in_web_evidence",
    "known_solution_page",
    "prior_run_artifact_access",
    "task_id_used_in_external_tool",
    "workspace_case_id_exposed",
    "workspace_reused",
}
_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
_BENCHMARK_ARTIFACT_RE = re.compile(
    r"(?:^|[/\\])(?:prepared|raw)(?:[/\\])|"
    r"(?:run_events|evidence|metric_observations|metrics)\.jsonl?",
    re.IGNORECASE,
)
_PRIOR_RUN_RE = re.compile(r"(?:^|[/\\])runs(?:[/\\])", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return "\n".join(_text(item) for key, item in value.items() if key != "image")
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(item) for item in value)
    return ""


def _urls(text: str) -> list[str]:
    return sorted({match.rstrip(".,;:!?)`") for match in _URL_RE.findall(text)})


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _known_solution_url(url: str, case_id: str) -> bool:
    value = unquote(url).casefold()
    solution_marker = any(
        marker in value
        for marker in ("solution", "solutions", "solve.sh", "answer", "answers")
    )
    benchmark_marker = any(
        marker in value for marker in (case_id.casefold(), "gaia", "harbor-datasets")
    )
    return solution_marker and benchmark_marker


def _gold_like_web_evidence(text: str, gold: Any, *, high_confidence_context: bool) -> bool:
    expected = _normalized(gold)
    if not expected:
        return False
    evidence = _normalized(text)
    if expected not in evidence:
        return False
    compact = re.sub(r"[^\w]+", "", expected)
    if len(compact) >= 8:
        return True
    return high_confidence_context or any(
        marker in evidence for marker in ("final answer", "correct answer", "solution:")
    )


def _trial_keys(records: Iterable[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for record in records:
        case_id = str(record.get("case_id") or "").strip()
        if case_id:
            keys.add((case_id, int(record.get("trial_index", 0) or 0)))
    return keys


def audit_run(
    run_dir: str | Path,
    *,
    trials: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    write: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit one run and write per-case evidence plus a compact summary."""

    root = Path(run_dir)
    from lychee_mas.runtime.events.store import event_data, iter_run_events

    audit_event_types = {
        "workspace.prepared",
        "agent.message.published",
        "tool_execution.started",
        "tool_execution.completed",
        "tool_execution.failed",
        "tool_execution.observed",
    }
    events = [
        event_data(event) | {"seq": event.get("seq")}
        for event in iter_run_events(root, event_types=audit_event_types)
    ]
    trial_keys = _trial_keys(trials)
    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for case_id, trial_index in sorted(trial_keys):
        records_by_key[(case_id, trial_index)] = {
            "schema_version": SCHEMA_VERSION,
            "audit_policy": AUDIT_POLICY,
            "case_id": case_id,
            "trial_index": trial_index,
            "audit_status": "complete",
            "contamination_suspected": False,
            "contamination_flags": [],
            "risk_flags": [],
            "evidence": [],
        }

    def record_for(value: dict[str, Any]) -> dict[str, Any] | None:
        key = (
            str(value.get("case_id") or ""),
            int(value.get("trial_index", 0) or 0),
        )
        return records_by_key.get(key)

    workspace_counts: dict[str, int] = {}
    workspace_events: dict[tuple[str, int], list[dict[str, Any]]] = {}
    workspaces_by_trial: dict[tuple[str, int], set[str]] = {}
    for event in events:
        if event.get("event_type") != "workspace.prepared":
            continue
        record = record_for(event)
        if record is None:
            continue
        key = (record["case_id"], record["trial_index"])
        workspace_events.setdefault(key, []).append(event)
        workspace = str(event.get("workspace") or "")
        if workspace:
            workspace_counts[workspace] = workspace_counts.get(workspace, 0) + 1
            workspaces_by_trial.setdefault(key, set()).add(workspace)

    for key, record in records_by_key.items():
        case_id = record["case_id"]
        case_workspace_events = workspace_events.get(key, [])
        if not case_workspace_events:
            record["audit_status"] = "partial"
            record["evidence"].append(
                {"flag": "workspace_audit_unavailable", "reason": "workspace event is absent"}
            )
        for event in case_workspace_events:
            workspace = str(event.get("workspace") or "")
            copied_names = list(event.get("visible_attachment_names") or [])
            if not copied_names:
                copied_names = [Path(path).name for path in event.get("copied_files") or []]
            if case_id.casefold() in workspace.casefold():
                record["contamination_flags"].append("workspace_case_id_exposed")
                record["evidence"].append(
                    {
                        "flag": "workspace_case_id_exposed",
                        "event_seq": event.get("seq"),
                        "workspace": workspace,
                        "reason": "agent-visible workspace path contains the benchmark case id",
                    }
                )
            if any(case_id.casefold() in name.casefold() for name in copied_names):
                record["risk_flags"].append("official_attachment_task_id_exposed")
                record["evidence"].append(
                    {
                        "flag": "official_attachment_task_id_exposed",
                        "event_seq": event.get("seq"),
                        "visible_attachment_names": copied_names,
                        "reason": "official attachment filename contains the GAIA task id",
                    }
                )
            reused = (
                event.get("workspace_is_new") is False
                or int(event.get("preexisting_entry_count", 0) or 0) > 0
                or (workspace and workspace_counts.get(workspace, 0) > 1)
            )
            if reused:
                record["contamination_flags"].append("workspace_reused")
                record["evidence"].append(
                    {
                        "flag": "workspace_reused",
                        "event_seq": event.get("seq"),
                        "workspace": workspace,
                        "reason": "workspace was not fresh or was assigned more than once",
                    }
                )

    for event in events:
        if event.get("event_type") != "agent.message.published":
            continue
        record = record_for(event)
        if record is None:
            continue
        source = str(event.get("source") or "")
        content = _text(event.get("content"))
        if not content:
            continue
        event_urls = _urls(content)
        is_web = source.casefold() == "websurfer"
        if is_web and record["case_id"].casefold() in content.casefold():
            record["contamination_flags"].append("task_id_used_in_external_tool")
            record["evidence"].append(
                {
                    "flag": "task_id_used_in_external_tool",
                    "event_seq": event.get("seq"),
                    "urls": event_urls,
                    "reason": "WebSurfer evidence contains the exact benchmark task id",
                }
            )
        solution_urls = [
            url for url in event_urls if _known_solution_url(url, record["case_id"])
        ]
        if is_web and solution_urls:
            record["contamination_flags"].append("known_solution_page")
            record["evidence"].append(
                {
                    "flag": "known_solution_page",
                    "event_seq": event.get("seq"),
                    "urls": solution_urls,
                    "reason": "external URL combines benchmark identity with a solution marker",
                }
            )
        gold_item = gold_by_id.get(record["case_id"], {})
        if is_web and _gold_like_web_evidence(
            content,
            gold_item.get("gold"),
            high_confidence_context=bool(solution_urls),
        ):
            record["contamination_flags"].append("gold_like_text_in_web_evidence")
            record["evidence"].append(
                {
                    "flag": "gold_like_text_in_web_evidence",
                    "event_seq": event.get("seq"),
                    "urls": event_urls,
                    "reason": "external web evidence contains gold-like answer text",
                }
            )

    for event in events:
        if event.get("event_type") not in {
            "tool_execution.started",
            "tool_execution.completed",
            "tool_execution.observed",
        }:
            continue
        record = record_for(event)
        if record is None:
            continue
        content = "\n".join(
            _text(event.get(key)) for key in ("code_blocks", "arguments", "output")
        )
        if _BENCHMARK_ARTIFACT_RE.search(content):
            record["contamination_flags"].append("benchmark_artifact_access")
            record["evidence"].append(
                {
                    "flag": "benchmark_artifact_access",
                    "event_seq": event.get("seq"),
                    "reason": "tool request references benchmark data or result artifacts",
                }
            )
        trial_key = (record["case_id"], record["trial_index"])
        external_run_content = content
        for workspace in workspaces_by_trial.get(trial_key, set()):
            external_run_content = external_run_content.replace(workspace, "<current_workspace>")
        if _PRIOR_RUN_RE.search(external_run_content):
            record["contamination_flags"].append("prior_run_artifact_access")
            record["evidence"].append(
                {
                    "flag": "prior_run_artifact_access",
                    "event_seq": event.get("seq"),
                    "reason": "tool request references a runs directory",
                }
            )
        external_urls = _urls(content)
        solution_urls = [
            url for url in external_urls if _known_solution_url(url, record["case_id"])
        ]
        if record["case_id"].casefold() in content.casefold() and external_urls:
            record["contamination_flags"].append("task_id_used_in_external_tool")
            record["evidence"].append(
                {
                    "flag": "task_id_used_in_external_tool",
                    "event_seq": event.get("seq"),
                    "urls": external_urls,
                    "reason": "code/tool execution combines the task id with an external URL",
                }
            )
        if solution_urls:
            record["contamination_flags"].append("known_solution_page")
            record["evidence"].append(
                {
                    "flag": "known_solution_page",
                    "event_seq": event.get("seq"),
                    "urls": solution_urls,
                    "reason": "external URL combines benchmark identity with a solution marker",
                }
            )
        gold_item = gold_by_id.get(record["case_id"], {})
        if external_urls and _gold_like_web_evidence(
            content,
            gold_item.get("gold"),
            high_confidence_context=bool(solution_urls),
        ):
            record["contamination_flags"].append("gold_like_text_in_web_evidence")
            record["evidence"].append(
                {
                    "flag": "gold_like_text_in_web_evidence",
                    "event_seq": event.get("seq"),
                    "urls": external_urls,
                    "reason": "external tool evidence contains gold-like answer text",
                }
            )

    for record in records_by_key.values():
        record["contamination_flags"] = sorted(set(record["contamination_flags"]))
        record["risk_flags"] = sorted(set(record["risk_flags"]))
        record["contamination_suspected"] = bool(
            SUSPECTED_FLAGS.intersection(record["contamination_flags"])
        )

    audits = sorted(
        records_by_key.values(), key=lambda item: (item["case_id"], item["trial_index"])
    )
    audited_case_ids = {
        item["case_id"] for item in audits if item["audit_status"] == "complete"
    }
    all_case_ids = {item["case_id"] for item in audits}
    suspected_case_ids = {
        item["case_id"] for item in audits if item["contamination_suspected"]
    }
    flag_counts: dict[str, set[str]] = {}
    risk_counts: dict[str, set[str]] = {}
    for item in audits:
        for flag in item["contamination_flags"]:
            flag_counts.setdefault(flag, set()).add(item["case_id"])
        for flag in item["risk_flags"]:
            risk_counts.setdefault(flag, set()).add(item["case_id"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_policy": AUDIT_POLICY,
        "num_cases": len(all_case_ids),
        "num_audited_cases": len(audited_case_ids),
        "audit_coverage": round(len(audited_case_ids) / len(all_case_ids), 4)
        if all_case_ids
        else 0.0,
        "num_suspected_cases": len(suspected_case_ids),
        "suspected_case_rate": round(len(suspected_case_ids) / len(all_case_ids), 4)
        if all_case_ids
        else 0.0,
        "suspected_case_ids": sorted(suspected_case_ids),
        "flag_case_counts": {key: len(value) for key, value in sorted(flag_counts.items())},
        "risk_flag_case_counts": {
            key: len(value) for key, value in sorted(risk_counts.items())
        },
    }
    if write:
        with (root / "contamination_audit.jsonl").open("w", encoding="utf-8") as handle:
            for item in audits:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (root / "contamination_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    return audits, summary


def attach_audit_to_trials(
    trials: list[dict[str, Any]], audits: list[dict[str, Any]]
) -> None:
    by_key = {
        (str(item.get("case_id")), int(item.get("trial_index", 0) or 0)): item
        for item in audits
    }
    for trial in trials:
        key = (str(trial.get("case_id")), int(trial.get("trial_index", 0) or 0))
        audit = by_key.get(key)
        if audit is not None:
            trial["contamination_audit"] = audit


def attach_audit_to_metrics(
    metrics: dict[str, Any], trials: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    suspected = set(summary.get("suspected_case_ids") or [])
    scored = [trial for trial in trials if isinstance(trial.get("score"), (int, float))]
    clean = [trial for trial in scored if str(trial.get("case_id")) not in suspected]
    official = sum(float(trial["score"]) for trial in scored) / len(scored) if scored else 0.0
    clean_score = (
        sum(float(trial["score"]) for trial in clean) / len(clean) if clean else None
    )
    conservative = (
        sum(
            0.0 if str(trial.get("case_id")) in suspected else float(trial["score"])
            for trial in scored
        )
        / len(scored)
        if scored
        else 0.0
    )
    metrics["contamination_audit"] = {
        **summary,
        "official_protocol_score": round(official, 4),
        "score_excluding_suspected_cases": (
            round(clean_score, 4) if clean_score is not None else None
        ),
        "score_treating_suspected_as_incorrect": round(conservative, 4),
        "score_note": (
            "official_protocol_score remains the primary result; the other scores are "
            "non-intervening sensitivity analyses"
        ),
    }
    return metrics
