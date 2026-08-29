"""RuntimeAdapter factory used by CLI and Eval Studio launches."""

from __future__ import annotations

from typing import Any

RUNTIME_FRAMEWORKS = ("autogen", "langgraph", "crewai")


def build_runtime(framework: str, **kwargs: Any):
    framework = str(framework).strip().lower()
    if framework == "autogen":
        from .autogen.runtime import AutoGenRuntime

        kwargs.pop("memory_method", None)
        kwargs.pop("framework_options", None)
        return AutoGenRuntime(**kwargs)
    if framework == "langgraph":
        from .langgraph.runtime import LangGraphRuntime

        return LangGraphRuntime(**kwargs)
    if framework == "crewai":
        from .crewai.runtime import CrewAIRuntime

        return CrewAIRuntime(**kwargs)
    raise ValueError(
        f"unknown runtime framework {framework!r}; choose {', '.join(RUNTIME_FRAMEWORKS)}"
    )
