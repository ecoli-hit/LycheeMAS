"""Lifecycle observation for AutoGen's official FileSurfer."""

from __future__ import annotations

import inspect
from typing import Any, Callable


def observed_file_surfer_class(
    *,
    base_class: type[Any] | None = None,
) -> type[Any]:
    """Return FileSurfer with non-invasive request/execution correlation.

    AutoGen's FileSurfer executes its file action inside ``_generate_reply`` and
    publishes only the resulting page text. The wrapper observes request IDs
    created by LycheeMAS's model client before and after that official method;
    it does not alter tool selection, execution, or the returned message.
    """

    if base_class is None:
        from autogen_ext.agents.file_surfer import FileSurfer

        base_class = FileSurfer

    class ObservedFileSurfer(base_class):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            request_ids: Callable[[], set[str]],
            on_tool_completed: Callable[[str, Any], Any] | None = None,
            on_tool_failed: Callable[[str, BaseException], Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._lychee_request_ids = request_ids
            self._lychee_on_tool_completed = on_tool_completed
            self._lychee_on_tool_failed = on_tool_failed
            super().__init__(*args, **kwargs)

        async def _generate_reply(self, *args: Any, **kwargs: Any) -> Any:
            before = self._lychee_request_ids()
            try:
                output = await super()._generate_reply(*args, **kwargs)
            except BaseException as exc:
                for call_id in sorted(self._lychee_request_ids() - before):
                    callback = self._lychee_on_tool_failed
                    if callback is not None:
                        result = callback(call_id, exc)
                        if inspect.isawaitable(result):
                            await result
                raise
            for call_id in sorted(self._lychee_request_ids() - before):
                callback = self._lychee_on_tool_completed
                if callback is not None:
                    result = callback(call_id, output)
                    if inspect.isawaitable(result):
                        await result
            return output

    ObservedFileSurfer.__name__ = "ObservedFileSurfer"
    return ObservedFileSurfer


__all__ = ["observed_file_surfer_class"]
