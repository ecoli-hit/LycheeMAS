"""AutoGen event and tool-observation normalization."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence
from urllib.parse import urlsplit

from lychee_mas.runtime.events.store import exception_record
from lychee_mas.runtime.model.token_budget import bound_text_for_model_context


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "case"


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(content_to_text(item) for item in content)
    if hasattr(content, "model_dump"):
        try:
            return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
        except Exception:
            return str(content)
    if hasattr(content, "__dict__"):
        try:
            return json.dumps(vars(content), ensure_ascii=False, default=str)
        except Exception:
            return str(content)
    return str(content)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def structured_tool_records(
    event: Any,
    event_index: int,
    *,
    max_inline_tool_result_chars: int = 50000,
) -> tuple[list[dict], list[dict]]:
    """Normalize typed AutoGen tool events without inferring calls from role names."""

    source = str(getattr(event, "source", "?") or "?")
    message_type = str(getattr(event, "type", type(event).__name__))
    requests: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    if message_type == "ToolCallRequestEvent":
        for call in getattr(event, "content", []) or []:
            requests.append(
                {
                    "record_type": "tool_request",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": str(getattr(call, "id", "") or ""),
                    "tool_name": str(getattr(call, "name", "") or ""),
                    "arguments": str(getattr(call, "arguments", "") or "{}"),
                    "origin": "autogen_tool_call_request_event",
                }
            )
    elif message_type == "ToolCallExecutionEvent":
        for result in getattr(event, "content", []) or []:
            output = str(getattr(result, "content", "") or "")
            delivered, truncated = bound_text_for_model_context(
                output, max_inline_tool_result_chars
            )
            executions.append(
                {
                    "record_type": "tool_execution",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": str(getattr(result, "call_id", "") or ""),
                    "tool_name": str(getattr(result, "name", "") or ""),
                    "is_error": bool(getattr(result, "is_error", False)),
                    "output": output,
                    "delivered_output": delivered,
                    "output_chars": len(output),
                    "delivered_output_chars": len(delivered),
                    "output_truncated_for_model_context": truncated,
                    "origin": "autogen_tool_call_execution_event",
                }
            )
    elif message_type == "CodeExecutionEvent":
        result = getattr(event, "result", None)
        if result is not None:
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            output = str(getattr(result, "output", "") or "")
            delivered, truncated = bound_text_for_model_context(
                output, max_inline_tool_result_chars
            )
            executions.append(
                {
                    "record_type": "tool_execution",
                    "event_index": event_index,
                    "source": source,
                    "tool_call_id": "",
                    "tool_name": "code_executor",
                    "exit_code": exit_code,
                    "is_error": exit_code != 0,
                    "output": output,
                    "delivered_output": delivered,
                    "output_chars": len(output),
                    "delivered_output_chars": len(delivered),
                    "output_truncated_for_model_context": truncated,
                    "origin": "autogen_code_execution_event",
                }
            )
    return requests, executions


def tool_requests_from_decisions(decisions: Sequence[dict]) -> list[dict[str, Any]]:
    return [
        {
            "record_type": "tool_request",
            "source": str(decision.get("role") or "?"),
            "turn": decision.get("turn_index", decision.get("turn")),
            "tool_call_id": str(call.get("id") or ""),
            "tool_name": str(call.get("name") or ""),
            "arguments": str(call.get("arguments") or "{}"),
            "origin": "model_client_function_call",
        }
        for decision in decisions
        for call in decision.get("tool_call_request") or []
    ]


def deduplicate_tool_records(records: Sequence[dict]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        call_id = str(record.get("tool_call_id") or "")
        key = (
            record.get("record_type"),
            call_id or None,
            record.get("source"),
            record.get("tool_name"),
            None if call_id else record.get("event_index"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(dict(record))
    return unique


def tool_operations(requests: Sequence[dict], executions: Sequence[dict]) -> list[dict]:
    operations = [dict(request, execution_status="requested") for request in requests]
    by_id = {
        str(operation.get("tool_call_id")): operation
        for operation in operations
        if operation.get("tool_call_id")
    }
    for execution in executions:
        call_id = str(execution.get("tool_call_id") or "")
        if call_id and call_id in by_id:
            operation = by_id[call_id]
            operation["execution_status"] = "error" if execution.get("is_error") else "completed"
            operation["execution"] = dict(execution)
        else:
            operations.append(
                {
                    "record_type": "tool_operation",
                    "source": execution.get("source"),
                    "tool_call_id": call_id,
                    "tool_name": execution.get("tool_name"),
                    "execution_status": ("error" if execution.get("is_error") else "completed"),
                    "execution": dict(execution),
                    "origin": execution.get("origin"),
                }
            )
    return operations


def is_web_surfer_operation_error(event: Any) -> bool:
    return bool(
        getattr(event, "type", type(event).__name__) == "TextMessage"
        and str(getattr(event, "content", "")).startswith("Web surfing error:\n\n")
    )


def web_surfer_error_details(content: Any) -> dict[str, Any]:
    text = str(content or "")
    error_codes = sorted(set(re.findall(r"\bERR_[A-Z0-9_]+\b", text)))
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s<>\]\[\"']+", text)))
    lowered = text.lower()
    if "err_network_changed" in lowered:
        category = "network_changed"
    elif "timeout" in lowered or "timed out" in lowered:
        category = "timeout"
    elif "font" in lowered:
        category = "font_or_rendering"
    elif "dns" in lowered or "name_not_resolved" in lowered:
        category = "dns"
    elif "proxy" in lowered or "tunnel" in lowered:
        category = "proxy"
    else:
        category = "other"
    return {
        "error_category": category,
        "browser_error_codes": error_codes,
        "referenced_urls": urls[:10],
        "referenced_domains": list(
            dict.fromkeys(urlsplit(url).hostname for url in urls if urlsplit(url).hostname)
        )[:10],
    }


def safe_proxy_endpoint(proxy_url: str | None) -> str | None:
    if not proxy_url:
        return None
    parsed = urlsplit(proxy_url)
    if not parsed.hostname:
        return "configured"
    return f"{parsed.scheme or 'http'}://{parsed.hostname}:{parsed.port or 80}"


class ObservedCodeExecutor:
    """Record typed CodeResult objects while preserving the executor interface."""

    def __init__(
        self,
        wrapped: Any,
        ctx: Any,
        records: list[dict[str, Any]],
        *,
        max_inline_tool_result_chars: int = 50000,
    ):
        self._wrapped = wrapped
        self._ctx = ctx
        self._records = records
        self._max_inline_tool_result_chars = max(1000, int(max_inline_tool_result_chars))
        self._last_code_fingerprint: str | None = None
        self._last_code_output = ""
        self._consecutive_identical_calls = 0

    async def start(self) -> None:
        await self._wrapped.start()

    async def stop(self) -> None:
        await self._wrapped.stop()

    async def restart(self) -> None:
        await self._wrapped.restart()

    async def execute_code_blocks(self, code_blocks, cancellation_token):
        fingerprint = json.dumps(
            jsonable(code_blocks),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint == self._last_code_fingerprint:
            self._consecutive_identical_calls += 1
        else:
            self._last_code_fingerprint = fingerprint
            self._consecutive_identical_calls = 1

        call_id = f"code_{safe_id(str(time.time_ns()))}"
        started = time.time()
        operation_event_id = self._ctx.log_event(
            "tool_execution.started",
            tool_call_id=call_id,
            tool_name="code_executor",
            origin="code_executor_adapter",
            code_blocks=jsonable(code_blocks),
        )
        if self._consecutive_identical_calls > 1:
            from autogen_core.code_executor import CodeResult

            previous = self._last_code_output[-1000:] or "<empty output>"
            result = CodeResult(
                exit_code=1,
                output=(
                    "Blocked a consecutive identical code execution because it cannot "
                    "produce new evidence. Do not retry the same code. Inspect the prior "
                    "result, change the command, or hand off. The executor working directory "
                    "is the trial workspace: use relative paths such as repo/...; /repo is "
                    "not the repository path.\n\nPrevious output:\n"
                    f"{previous}"
                ),
            )
        else:
            try:
                result = await self._wrapped.execute_code_blocks(
                    code_blocks,
                    cancellation_token=cancellation_token,
                )
            except BaseException as exc:
                record = {
                    "record_type": "tool_execution",
                    "source": "ComputerTerminal",
                    "tool_call_id": call_id,
                    "tool_name": "code_executor",
                    "is_error": True,
                    "duration_s": round(time.time() - started, 3),
                    "origin": "code_executor_adapter",
                    **exception_record(exc),
                }
                self._records.append(record)
                self._ctx.log_event(
                    "tool_execution.failed",
                    operation_id=operation_event_id,
                    correlation_id=call_id,
                    **record,
                )
                raise
            self._last_code_output = str(getattr(result, "output", "") or "")
        from autogen_core.code_executor import CodeResult

        exit_code = int(getattr(result, "exit_code", 0) or 0)
        output = str(getattr(result, "output", "") or "")
        delivered_output, output_truncated = bound_text_for_model_context(
            output,
            self._max_inline_tool_result_chars,
        )
        record = {
            "record_type": "tool_execution",
            "source": "ComputerTerminal",
            "tool_call_id": call_id,
            "tool_name": "code_executor",
            "exit_code": exit_code,
            "is_error": exit_code != 0,
            "output": output,
            "delivered_output": delivered_output,
            "output_chars": len(output),
            "delivered_output_chars": len(delivered_output),
            "output_truncated_for_model_context": output_truncated,
            "duration_s": round(time.time() - started, 3),
            "origin": "code_executor_adapter",
            "repeated_identical_call_blocked": self._consecutive_identical_calls > 1,
        }
        self._records.append(record)
        self._ctx.log_event(
            "tool_execution.completed",
            operation_id=operation_event_id,
            correlation_id=call_id,
            **record,
        )
        if not output_truncated:
            return result
        return CodeResult(exit_code=exit_code, output=delivered_output)


__all__ = [
    "ObservedCodeExecutor",
    "content_to_text",
    "deduplicate_tool_records",
    "is_web_surfer_operation_error",
    "jsonable",
    "safe_id",
    "safe_proxy_endpoint",
    "structured_tool_records",
    "tool_operations",
    "tool_requests_from_decisions",
    "web_surfer_error_details",
]
