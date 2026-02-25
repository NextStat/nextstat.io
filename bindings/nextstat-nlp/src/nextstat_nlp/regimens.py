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
    "duration": "Course duration of dosing (e.g., 14 days, 6 weeks).",
    "infusion_duration": "IV infusion duration (e.g., over 2 hours).",
    "occasion": "Dosing occasion or cycle number.",
}

_ROUTE_ALIASES: Dict[str, str] = {
    "iv": "IV", "intravenous": "IV",
    "oral": "oral", "orally": "oral", "po": "oral", "by mouth": "oral",
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

_KNOWN_UNITS = {"mg", "mg/kg", "mcg", "g", "iu"}


def _canonicalize_route(raw: str) -> str:
    key = raw.strip().lower()
    return _ROUTE_ALIASES.get(key, raw.strip())


def _canonicalize_freq(raw: str) -> str:
    key = raw.strip().lower()
    return _FREQ_ALIASES.get(key, raw.strip())


def _canonicalize_units(raw: str) -> str:
    u = raw.strip().lower()
    u = u.replace("μg", "mcg").replace("ug", "mcg")
    # Strip punctuation commonly produced by NER (e.g., "mg.", "mg/day").
    u = re.sub(r"[^a-z0-9/]+", "", u)
    if u in _KNOWN_UNITS:
        return u
    return ""


def _guess_route_from_text(text: str) -> str:
    t = text.lower()
    # Prefer longer keys first (e.g., "intravenous" before "iv").
    for k in sorted(_ROUTE_ALIASES.keys(), key=len, reverse=True):
        if " " in k:
            if k in t:
                return _ROUTE_ALIASES[k]
            continue
        if re.search(rf"\b{re.escape(k)}\b", t):
            return _ROUTE_ALIASES[k]
    # Fallback: common patterns without strict word boundaries.
    if "intraven" in t:
        return "IV"
    if "infusion" in t:
        return "IV"
    return ""


def _guess_freq_from_text(text: str) -> str:
    t = text.lower()
    for k in sorted(_FREQ_ALIASES.keys(), key=len, reverse=True):
        if " " in k:
            if k in t:
                return _FREQ_ALIASES[k]
            continue
        if re.search(rf"\b{re.escape(k)}\b", t):
            return _FREQ_ALIASES[k]
    # Common qNh pattern.
    m = re.search(r"\bq\s*(\d{1,2})\s*h\b", t)
    if m:
        return f"Q{int(m.group(1))}H"
    return ""


def _guess_infusion_duration_from_text(text: str) -> Optional[float]:
    # Try to pull the common infusion phrasing "over 2 hours".
    m = re.search(
        r"\bover\s+([0-9]+(?:\.[0-9]+)?)\s*(hours?|days?|weeks?|months?|years?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        # parse_time expects the numeric + unit chunk.
        return parse_time(f"{m.group(1)} {m.group(2)}")
    return None


def _guess_duration_from_text(text: str) -> Optional[float]:
    # Backward-compatible alias (tests/tools may import this).
    return _guess_infusion_duration_from_text(text)


def _freq_interval_days(freq: str) -> Optional[float]:
    f = (freq or "").strip().upper()
    if not f:
        return None
    if f in ("ONCE",):
        return None
    if f in ("QD", "Q24H"):
        return 1.0
    if f in ("BID", "Q12H"):
        return 0.5
    if f in ("TID", "Q8H"):
        return 1.0 / 3.0
    if f in ("QID", "Q6H"):
        return 0.25
    if f in ("WEEKLY", "QW"):
        return 7.0
    # QNh variants
    m = re.match(r"^Q(\d{1,2})H$", f)
    if m:
        h = int(m.group(1))
        if h > 0:
            return h / 24.0
    return None


def to_nextstat_regimens(
    records: Sequence[RegimenRecord],
    *,
    expand_frequency: bool = False,
    default_course_days: Optional[float] = None,
    max_events_per_subject: int = 512,
) -> List[Dict[str, Any]]:
    """Convert `RegimenRecord`s into nextstat `_core` `regimens` dictionaries.

    Notes:
    - `duration` is interpreted as *course duration* (days). It is NOT infusion duration.
    - `infusion_duration` is mapped to nextstat event `duration` (days) for IV routes.
    - If `expand_frequency=True`, repeated events are generated only if a course duration
      is known (record.duration or default_course_days).
    """
    by_subj: Dict[str, List[RegimenRecord]] = {}
    for r in records:
        by_subj.setdefault(r.subject_id, []).append(r)

    out: List[Dict[str, Any]] = []
    for subj, rs in by_subj.items():
        events: List[Dict[str, Any]] = []
        meta: List[Dict[str, Any]] = []

        for r in rs:
            t0 = float(r.start_time or 0.0)
            route = (r.route or "oral").strip() or "oral"
            amount = float(r.dose or 0.0)
            if amount <= 0:
                continue

            infusion_dur = float(r.infusion_duration or 0.0) if route.upper() == "IV" else 0.0

            if expand_frequency:
                course = r.duration if r.duration is not None else default_course_days
                interval = _freq_interval_days(r.frequency)
                if course is not None and interval is not None and interval > 0:
                    t_end = t0 + float(course)
                    t = t0
                    k = 0
                    while t <= t_end + 1e-12:
                        events.append({"time": float(t), "amount": amount, "route": route, "duration": infusion_dur})
                        k += 1
                        if k >= max_events_per_subject:
                            break
                        t += interval
                else:
                    events.append({"time": t0, "amount": amount, "route": route, "duration": infusion_dur})
            else:
                events.append({"time": t0, "amount": amount, "route": route, "duration": infusion_dur})

            meta.append({
                "frequency": r.frequency,
                "amount_units": r.amount_units,
                "course_duration_days": r.duration,
                "infusion_duration_days": r.infusion_duration,
                "occasion_id": r.occasion_id,
                "document_id": r.document_id,
            })

        if events:
            out.append({"subject_id": subj, "events": events, "meta": meta})

    return out


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
        inf_dur_spans = spans_by_label.get("infusion_duration", [])
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

            units_raw = unit_spans[idx].text.strip() if idx < len(unit_spans) else "mg"
            units = _canonicalize_units(units_raw) or "mg"

            route_raw = route_spans[idx].text if idx < len(route_spans) else ""
            route = _canonicalize_route(route_raw)
            if not route or route == route_raw.strip():
                route = _guess_route_from_text(text) or route

            freq_raw = freq_spans[idx].text if idx < len(freq_spans) else ""
            freq = _canonicalize_freq(freq_raw)
            if not freq or freq == freq_raw.strip():
                freq = _guess_freq_from_text(text) or freq

            start = parse_time(start_spans[idx].text) if idx < len(start_spans) else None
            course_dur = parse_time(dur_spans[idx].text) if idx < len(dur_spans) else None
            inf_dur = parse_time(inf_dur_spans[idx].text) if idx < len(inf_dur_spans) else None
            if inf_dur is None:
                inf_dur = _guess_infusion_duration_from_text(text)
            occ_val = parse_numeric(occ_spans[idx].text) if idx < len(occ_spans) else None
            occ_id = int(occ_val) if occ_val is not None else None

            records.append(RegimenRecord(
                subject_id=subj,
                dose=dose_val,
                route=route,
                start_time=start,
                duration=course_dur,
                infusion_duration=inf_dur,
                amount_units=units,
                frequency=freq,
                occasion_id=occ_id,
                spans=all_spans,
                document_id=document_id,
                text_hash=sha256_text(text),
            ))

    return RegimenTable(records=records, backend_env=env)
