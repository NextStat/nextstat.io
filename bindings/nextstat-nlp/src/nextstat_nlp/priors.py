"""Prior-candidate extraction pipeline.

Extracts informative-prior candidates from published literature text
(abstracts, methods sections) via GLiNER2 or heuristic NER.

Every output is ``PriorCandidate(status="candidate")`` — **never auto-approved**.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ._errors import ExtractionFailed
from ._hashing import sha256_text
from ._parsers import parse_numeric
from .backends import EntitySpan, get_backend
from .backends.base import ExtractorBackend
from .backends.base import BackendName
from .schemas import ExtractedSpan, PriorBundle, PriorCandidate

_PRIOR_SCHEMA: Dict[str, str] = {
    "parameter_name": "Pharmacokinetic or statistical parameter name (CL, V1, V2, Q, ka, ke, EC50, Emax, etc.).",
    "distribution": "Statistical distribution type (normal, lognormal, half_normal, uniform, gamma).",
    "mean": "Mean or location parameter value.",
    "sd": "Standard deviation or scale parameter value.",
    "units": "Parameter units (L/h, L, 1/h, mg, etc.).",
    "constraint": "Parameter constraint or support (e.g., >0, 0-1).",
    "citation": "Citation or source reference for this prior.",
}

# Canonical PK parameter name mapping
_PARAM_ALIASES: Dict[str, str] = {
    "clearance": "CL", "cl": "CL",
    "volume": "V1", "v": "V1", "v1": "V1", "vd": "V1",
    "v2": "V2", "peripheral volume": "V2",
    "intercompartmental clearance": "Q", "q": "Q",
    "absorption rate": "ka", "ka": "ka",
    "elimination rate": "ke", "ke": "ke",
    "ec50": "EC50", "emax": "Emax",
    "hill": "Hill", "baseline": "E0", "e0": "E0",
}

_DIST_ALIASES: Dict[str, str] = {
    "normal": "normal", "gaussian": "normal",
    "lognormal": "lognormal", "log-normal": "lognormal", "log normal": "lognormal",
    "half-normal": "half_normal", "half_normal": "half_normal", "halfnormal": "half_normal",
    "uniform": "uniform", "flat": "uniform",
    "gamma": "gamma",
    "exponential": "exponential",
}


def _canonicalize_param(raw: str) -> str:
    key = raw.strip().lower()
    return _PARAM_ALIASES.get(key, raw.strip())


def _canonicalize_dist(raw: str) -> str:
    key = raw.strip().lower()
    return _DIST_ALIASES.get(key, raw.strip().lower())


def _parse_support(text: str) -> tuple[Optional[float], Optional[float]]:
    """Parse constraint text like '>0', '0-1', '(0, inf)'."""
    text = text.strip()
    m = re.match(r">\s*(\d+(?:\.\d+)?)", text)
    if m:
        return (float(m.group(1)), None)
    m = re.match(r"<\s*(\d+(?:\.\d+)?)", text)
    if m:
        return (None, float(m.group(1)))
    m = re.match(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", text)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (None, None)


def extract_prior_candidates(
    texts: Sequence[str],
    *,
    backend: BackendName = "gliner2",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    providers: Optional[Sequence[str]] = None,
    num_threads: Optional[int] = None,
    document_id: Optional[str] = None,
    backend_obj: Optional[ExtractorBackend] = None,
) -> PriorBundle:
    """Extract prior candidates from literature text(s).

    Returns a ``PriorBundle`` where every candidate has
    ``status="candidate"`` and requires explicit approval.
    """
    be = backend_obj or get_backend(
        backend,
        model_id=model_id,
        device=device,
        providers=providers,
        num_threads=num_threads,
    )
    env = be.environment()

    candidates: List[PriorCandidate] = []

    for text in texts:
        try:
            ents = be.extract_entities(text, _PRIOR_SCHEMA)
        except Exception as e:
            raise ExtractionFailed(f"Backend '{backend}' failed on prior extraction: {e}") from e

        # Group spans by label
        spans_by_label: Dict[str, List[EntitySpan]] = {}
        for ent in ents:
            spans_by_label.setdefault(ent.label, []).append(ent)

        param_spans = spans_by_label.get("parameter_name", [])
        dist_spans = spans_by_label.get("distribution", [])
        mean_spans = spans_by_label.get("mean", [])
        sd_spans = spans_by_label.get("sd", [])
        unit_spans = spans_by_label.get("units", [])
        constraint_spans = spans_by_label.get("constraint", [])
        citation_spans = spans_by_label.get("citation", [])

        # Build candidates — pair param with nearest distribution/values
        for idx, ps in enumerate(param_spans):
            param = _canonicalize_param(ps.text)
            dist = _canonicalize_dist(dist_spans[idx].text) if idx < len(dist_spans) else "normal"
            loc = parse_numeric(mean_spans[idx].text) if idx < len(mean_spans) else None
            sc = parse_numeric(sd_spans[idx].text) if idx < len(sd_spans) else None
            units = unit_spans[idx].text.strip() if idx < len(unit_spans) else ""
            support = _parse_support(constraint_spans[idx].text) if idx < len(constraint_spans) else (None, None)
            citation = citation_spans[idx].text.strip() if idx < len(citation_spans) else ""

            if loc is None or sc is None:
                continue  # Need at least location and scale

            candidates.append(PriorCandidate(
                param_name=param,
                dist=dist,
                location=loc,
                scale=sc,
                units=units,
                support=support,
                confidence=ps.score,
                citation_span=citation,
                document_id=document_id,
                text_hash=sha256_text(text),
            ))

    return PriorBundle(
        candidates=candidates,
        backend_env=env,
        extraction_config={"schema": list(_PRIOR_SCHEMA.keys())},
    )
