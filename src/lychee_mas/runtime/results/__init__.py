"""Runtime result contracts and deterministic result projection."""

from .contract import ResultContractValidation, validate_result_contract
from .projection import ProjectedResult, project_result

__all__ = [
    "ProjectedResult",
    "ResultContractValidation",
    "project_result",
    "validate_result_contract",
]
