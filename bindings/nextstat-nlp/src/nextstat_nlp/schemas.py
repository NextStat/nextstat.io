"""Canonical dataclasses for all nextstat-nlp extraction outputs.

Every public dataclass is frozen for immutability and audit safety.
Domain modules (survival, priors, regimens) re-export relevant types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared span
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedSpan:
    """A single entity span produced by any backend."""
    label: str
    text: str
    start: int
    end: int
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurvivalRecord:
    subject_id: str
    time: float
    event: int  # 1=event, 0=censored
    covariates: Dict[str, Any] = field(default_factory=dict)
    spans: List[ExtractedSpan] = field(default_factory=list)
    document_id: Optional[str] = None
    text_hash: Optional[str] = None


@dataclass(frozen=True)
class SurvivalDataset:
    records: List[SurvivalRecord]
    backend_env: Dict[str, Any]

    def to_design_matrix(self) -> Tuple[List[float], List[int], List[List[float]], List[str]]:
        feature_names: List[str] = []
        for r in self.records:
            for k, v in r.covariates.items():
                if isinstance(v, (int, float)) and k not in feature_names:
                    feature_names.append(k)
        times = [float(r.time) for r in self.records]
        events = [int(r.event) for r in self.records]
        X: List[List[float]] = []
        for r in self.records:
            row: List[float] = []
            for k in feature_names:
                v = r.covariates.get(k, 0.0)
                row.append(float(v) if isinstance(v, (int, float)) else 0.0)
            X.append(row)
        return times, events, X, feature_names


# ---------------------------------------------------------------------------
# Priors (RFC section 6.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorCandidate:
    """A candidate informative prior extracted from literature text.

    Always ``status="candidate"`` — requires explicit human approval
    before use in any statistical model.
    """
    param_name: str
    dist: str  # "normal", "lognormal", "half_normal", "uniform", "gamma"
    location: float
    scale: float
    units: str = ""
    support: Tuple[Optional[float], Optional[float]] = (None, None)
    confidence: Optional[float] = None
    citation_span: str = ""
    document_id: Optional[str] = None
    text_hash: Optional[str] = None
    status: Literal["candidate"] = "candidate"


@dataclass(frozen=True)
class PriorBundle:
    candidates: List[PriorCandidate]
    backend_env: Dict[str, Any]
    extraction_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regimens / IOV (RFC section 6.3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimenRecord:
    """A single dosing event extracted from protocol text."""
    subject_id: str
    dose: float
    route: str  # "IV", "oral", "SC", "IM", "topical"
    start_time: Optional[float] = None  # in days
    # Course duration in days (e.g., "for 14 days"). This is *not* infusion duration.
    duration: Optional[float] = None
    # Infusion duration in days (e.g., "over 2 hours" -> 2/24). Used for IV infusion events.
    infusion_duration: Optional[float] = None
    amount_units: str = "mg"
    frequency: str = ""     # "QD", "BID", "TID", "Q12H", etc.
    occasion_id: Optional[int] = None
    spans: List[ExtractedSpan] = field(default_factory=list)
    document_id: Optional[str] = None
    text_hash: Optional[str] = None


@dataclass(frozen=True)
class RegimenTable:
    records: List[RegimenRecord]
    backend_env: Dict[str, Any]
