"""Survival-data extraction pipeline.

Extracts time-to-event records from clinical text via GLiNER2 or
heuristic NER, deterministic parsing, and validation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ._errors import ExtractionFailed
from ._hashing import sha256_text
from ._parsers import parse_event, parse_numeric, parse_time
from .backends import EntitySpan, get_backend
from .backends.base import ExtractorBackend
from .backends.base import BackendName
# Canonical types live in schemas; re-export for backward compatibility
from .schemas import ExtractedSpan, SurvivalDataset, SurvivalRecord

__all__ = [
    "ExtractedSpan",
    "SurvivalRecord",
    "SurvivalDataset",
    "extract_survival_records",
]

_SURVIVAL_SCHEMA: Dict[str, str] = {
    "subject_id": "Subject identifier (e.g., patient ID).",
    "time": "Time-to-event or time-to-censoring (include units in text).",
    "event": "Event status: progressed/died=event, withdrew/lost=censored.",
    "age": "Age in years.",
    "dose": "Dose amount (e.g., 20 mg, 5 mg/kg).",
    "sex": "Sex or gender (male, female, M, F).",
    "stage": "Disease stage (I, II, III, IV).",
    "ecog_ps": "ECOG performance status (0-4).",
    "weight": "Body weight in kg.",
}

# Covariates to try extracting — label -> covariate key
_COVARIATE_LABELS = ("age", "dose", "sex", "stage", "ecog_ps", "weight")


def extract_survival_records(
    texts: Sequence[str],
    *,
    backend: BackendName = "gliner2",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    providers: Optional[Sequence[str]] = None,
    num_threads: Optional[int] = None,
    document_id: Optional[str] = None,
    backend_obj: Optional[ExtractorBackend] = None,
) -> SurvivalDataset:
    """Extract survival records from clinical text passages.

    Each element of *texts* should describe one subject's outcome.
    """
    be = backend_obj or get_backend(
        backend,
        model_id=model_id,
        device=device,
        providers=providers,
        num_threads=num_threads,
    )
    env = be.environment()

    records: List[SurvivalRecord] = []
    for i, text in enumerate(texts):
        try:
            ents = be.extract_entities(text, _SURVIVAL_SCHEMA)
        except Exception as e:
            raise ExtractionFailed(f"Backend '{backend}' failed on record {i}: {e}") from e

        spans = [
            ExtractedSpan(label=ent.label, text=ent.text, start=ent.start, end=ent.end, score=ent.score)
            for ent in ents
        ]

        # --- subject_id ---
        subj = None
        for s in spans:
            if s.label == "subject_id":
                subj = s.text
                break
        if subj is None:
            subj = f"{i:04d}"

        # --- time (from spans first, fallback to full text) ---
        time_v = None
        for s in spans:
            if s.label == "time":
                time_v = parse_time(s.text)
                if time_v is not None:
                    break
        if time_v is None:
            time_v = parse_time(text)

        # --- event (from spans first, fallback to full text) ---
        event_v = None
        for s in spans:
            if s.label == "event":
                event_v = parse_event(s.text)
                if event_v is not None:
                    break
        if event_v is None:
            event_v = parse_event(text)

        # --- covariates ---
        cov: Dict[str, Any] = {}
        for label in _COVARIATE_LABELS:
            for s in spans:
                if s.label == label:
                    if label in ("sex", "stage"):
                        cov[label] = s.text.strip()
                    else:
                        val = parse_numeric(s.text, label)
                        if val is not None:
                            cov[label] = val
                    break

        # --- validation ---
        if time_v is None or event_v is None:
            raise ExtractionFailed(
                f"Missing required survival fields for subject={subj}: "
                f"time={time_v} event={event_v}. "
                "Provide clearer text or use a review-mode pipeline."
            )
        if time_v <= 0:
            raise ExtractionFailed(
                f"Invalid time={time_v} for subject={subj}: must be > 0."
            )
        if event_v not in (0, 1):
            raise ExtractionFailed(
                f"Invalid event={event_v} for subject={subj}: must be 0 or 1."
            )

        records.append(SurvivalRecord(
            subject_id=subj,
            time=float(time_v),
            event=int(event_v),
            covariates=cov,
            spans=spans,
            document_id=document_id,
            text_hash=sha256_text(text),
        ))

    return SurvivalDataset(records=records, backend_env=env)
