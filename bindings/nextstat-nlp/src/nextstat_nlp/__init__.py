from .priors import extract_prior_candidates
from .regimens import extract_regimens
from .schemas import (
    ExtractedSpan,
    PriorBundle,
    PriorCandidate,
    RegimenRecord,
    RegimenTable,
    SurvivalDataset,
    SurvivalRecord,
)
from .survival import extract_survival_records

try:
    # Keep version discoverable without importing packaging tooling.
    from importlib.metadata import version as _dist_version  # type: ignore

    __version__ = _dist_version("nextstat-nlp")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Schemas
    "ExtractedSpan",
    "SurvivalRecord",
    "SurvivalDataset",
    "PriorCandidate",
    "PriorBundle",
    "RegimenRecord",
    "RegimenTable",
    # Pipelines
    "extract_survival_records",
    "extract_prior_candidates",
    "extract_regimens",
]
