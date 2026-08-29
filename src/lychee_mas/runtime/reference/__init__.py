"""Framework-neutral executable semantics for normalized TeamSpec graphs."""

from .interpreter import (
    ReferenceDataItem,
    ReferenceExecutionResult,
    TeamSpecReferenceInterpreter,
)

__all__ = [
    "ReferenceDataItem",
    "ReferenceExecutionResult",
    "TeamSpecReferenceInterpreter",
]
