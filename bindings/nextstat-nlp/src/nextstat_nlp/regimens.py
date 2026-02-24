"""Regimen / IOV extraction pipeline.

Extracts dosing regimen records from protocol text via GLiNER2 or
heuristic NER — dose, route, frequency, timing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ._errors import ExtractionFailed
from ._hashing import sha256_text
from ._parsers import parse_numeric, parse_time
from .backends import EntitySpan, get_backend
from .backends.base import ExtractorBackend
from .backends.base import BackendName
from .schemas import ExtractedSpan, RegimenRecord, RegimenTable

_REGIMEN_SCHEMA: Dict[str, str] = {
    "subject_id": "Subject or patient identifier.",
    "dose_amount": "Dose amount (e.g., 20, 5.5, 100).",
    "dose_unit": "Dose unit (mg, mg/kg, mcg, g, IU).",
    "route": "Administration route (IV, oral, SC, IM, topical, inhaled).",
    "frequency": "Dosing frequency (QD, BID, TID, QID, Q12H, Q8H, once, weekly).",
    "start_time": "When dosing begins (e.g., day 1, week 0).",
    "duration": "Duration of dosing (e.g., 14 days, 6 weeks).",
    "occasion": "Dosing occasion or cycle number.",
}

_ROUTE_ALIASES: Dict[str, str] = {
    "iv": "IV", "intravenous": "IV",
    "oral": "oral", "po": "oral", "by mouth": "oral",
    "sc": "SC", "subcutaneous": "SC", "subq": "SC",
    "im": "IM", "intramuscular": "IM",
    "topical": "topical",
    "inhaled": "inhaled", "inhalation": "inhaled",
}

_FREQ_ALIASES: Dict[str, str] = {
    "qd": "QD", "once daily": "QD", "daily": "QD", "od": "QD",
    "bid": "BID", "twice daily": "BID", "b.i.d.": "BID",
    "tid": "TID", "three times daily": "TID", "t.i.d.": "TID",
    "qid": "QID", "four times daily": "QID", "q.i.d.": "QID",
    "q12h": "Q12H", "every 12 hours": "Q12H", "q 12h": "Q12H",
    "q8h": "Q8H", "every 8 hours": "Q8H", "q 8h": "Q8H",
    "q6h": "Q6H", "every 6 hours": "Q6H",
    "q24h": "Q24H", "every 24 hours": "Q24H",
    "once": "once", "single dose": "once", "stat": "once",
    "weekly": "weekly", "qw": "weekly", "once weekly": "weekly",
}


def _canonicalize_route(raw: str) -> str:
    key = raw.strip().lower()
    return _ROUTE_ALIASES.get(key, raw.strip())


def _canonicalize_freq(raw: str) -> str:
    key = raw.strip().lower()
    return _FREQ_ALIASES.get(key, raw.strip())


def extract_regimens(
    texts: Sequence[str],
    *,
    backend: BackendName = "gliner2",
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    providers: Optional[Sequence[str]] = None,
    num_threads: Optional[int] = None,
    document_id: Optional[str] = None,
    backend_obj: Optional[ExtractorBackend] = None,
) -> RegimenTable:
    """Extract dosing regimen records from protocol text(s)."""
    be = backend_obj or get_backend(
        backend,
        model_id=model_id,
        device=device,
        providers=providers,
        num_threads=num_threads,
    )
    env = be.environment()

    records: List[RegimenRecord] = []

    for i, text in enumerate(texts):
        try:
            ents = be.extract_entities(text, _REGIMEN_SCHEMA)
        except Exception as e:
            raise ExtractionFailed(f"Backend '{backend}' failed on regimen extraction: {e}") from e

        spans_by_label: Dict[str, List[EntitySpan]] = {}
        for ent in ents:
            spans_by_label.setdefault(ent.label, []).append(ent)

        subj_spans = spans_by_label.get("subject_id", [])
        dose_spans = spans_by_label.get("dose_amount", [])
        unit_spans = spans_by_label.get("dose_unit", [])
        route_spans = spans_by_label.get("route", [])
        freq_spans = spans_by_label.get("frequency", [])
        start_spans = spans_by_label.get("start_time", [])
        dur_spans = spans_by_label.get("duration", [])
        occ_spans = spans_by_label.get("occasion", [])

        n_doses = max(len(dose_spans), 1)
        all_spans = [
            ExtractedSpan(label=e.label, text=e.text, start=e.start, end=e.end, score=e.score)
            for e in ents
        ]

        for idx in range(n_doses):
            subj = subj_spans[idx].text if idx < len(subj_spans) else f"{i:04d}"
            dose_val = parse_numeric(dose_spans[idx].text) if idx < len(dose_spans) else None
            if dose_val is None:
                continue  # Skip records without a dose

            units = unit_spans[idx].text.strip() if idx < len(unit_spans) else "mg"
            route = _canonicalize_route(route_spans[idx].text) if idx < len(route_spans) else ""
            freq = _canonicalize_freq(freq_spans[idx].text) if idx < len(freq_spans) else ""
            start = parse_time(start_spans[idx].text) if idx < len(start_spans) else None
            dur = parse_time(dur_spans[idx].text) if idx < len(dur_spans) else None
            occ_val = parse_numeric(occ_spans[idx].text) if idx < len(occ_spans) else None
            occ_id = int(occ_val) if occ_val is not None else None

            records.append(RegimenRecord(
                subject_id=subj,
                dose=dose_val,
                route=route,
                start_time=start,
                duration=dur,
                amount_units=units,
                frequency=freq,
                occasion_id=occ_id,
                spans=all_spans,
                document_id=document_id,
                text_hash=sha256_text(text),
            ))

    return RegimenTable(records=records, backend_env=env)
