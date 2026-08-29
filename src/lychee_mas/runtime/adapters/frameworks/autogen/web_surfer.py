"""Narrow lifecycle adapter for AutoGen's official WebSurfer.

The official Magentic-One outer loop resets participants between replans. A
transient navigation failure while WebSurfer returns to its start page should
not invalidate an otherwise recoverable Trial, so this adapter falls back to a
blank page and records the recovery. All normal browsing remains owned by the
official ``MultimodalWebSurfer`` implementation.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable


def resilient_multimodal_web_surfer_class(
    *,
    base_class: type[Any] | None = None,
    recoverable_error_type: type[BaseException] | tuple[type[BaseException], ...] | None = None,
) -> type[Any]:
    """Return the official WebSurfer with recoverable reset navigation."""

    if base_class is None:
        from autogen_ext.agents.web_surfer import MultimodalWebSurfer

        base_class = MultimodalWebSurfer
    if recoverable_error_type is None:
        from playwright.async_api import Error as PlaywrightError

        recoverable_error_type = PlaywrightError

    class ResilientMultimodalWebSurfer(base_class):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            on_reset_recovered: Callable[[BaseException], Any] | None = None,
            on_tool_started: Callable[[Any], Any] | None = None,
            on_tool_completed: Callable[[Any, Any, float], Any] | None = None,
            on_tool_failed: Callable[[Any, BaseException, float], Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._lychee_on_reset_recovered = on_reset_recovered
            self._lychee_on_tool_started = on_tool_started
            self._lychee_on_tool_completed = on_tool_completed
            self._lychee_on_tool_failed = on_tool_failed
            super().__init__(*args, **kwargs)

        async def _execute_tool(self, message: list[Any], *args: Any, **kwargs: Any) -> Any:
            call = message[0] if message else None
            started = time.monotonic()
            callback = self._lychee_on_tool_started
            if callback is not None and call is not None:
                result = callback(call)
                if inspect.isawaitable(result):
                    await result
            try:
                output = await super()._execute_tool(message, *args, **kwargs)
            except BaseException as exc:
                callback = self._lychee_on_tool_failed
                if callback is not None and call is not None:
                    result = callback(call, exc, time.monotonic() - started)
                    if inspect.isawaitable(result):
                        await result
                raise
            callback = self._lychee_on_tool_completed
            if callback is not None and call is not None:
                result = callback(call, output, time.monotonic() - started)
                if inspect.isawaitable(result):
                    await result
            return output

        async def on_reset(self, cancellation_token: Any) -> None:
            try:
                await super().on_reset(cancellation_token)
                return
            except recoverable_error_type as exc:  # type: ignore[misc]
                page = getattr(self, "_page", None)
                if page is None:
                    raise
                try:
                    await page.goto(
                        "about:blank",
                        wait_until="domcontentloaded",
                        timeout=5000,
                    )
                except recoverable_error_type as fallback_exc:  # type: ignore[misc]
                    raise exc from fallback_exc

                history = getattr(self, "_chat_history", None)
                if history is not None and hasattr(history, "clear"):
                    history.clear()
                if hasattr(self, "_last_download"):
                    self._last_download = None
                if hasattr(self, "_prior_metadata_hash"):
                    self._prior_metadata_hash = None
                callback = self._lychee_on_reset_recovered
                if callback is not None:
                    result = callback(exc)
                    if inspect.isawaitable(result):
                        await result

    ResilientMultimodalWebSurfer.__name__ = "ResilientMultimodalWebSurfer"
    return ResilientMultimodalWebSurfer


__all__ = ["resilient_multimodal_web_surfer_class"]
