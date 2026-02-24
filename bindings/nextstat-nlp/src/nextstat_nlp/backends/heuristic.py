"""Regex-based heuristic backend — zero ML dependencies.

Useful for testing the full pipeline without downloading model weights.
Not a substitute for GLiNER2 on real clinical text.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Union

from .base import EntitySpan

# Each pattern: (compiled regex, group index for text, label)
_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    # subject_id: "Subject 001", "Patient 42", "Pt 7"
    (re.compile(r"\b(?:Subject|Patient|Pt)\s+(\w+)", re.IGNORECASE), 1, "subject_id"),
    # age: "Age 63", "aged 58", "63 years old", "58-year-old"
    (re.compile(r"\bAge[d]?\s+(\d+)", re.IGNORECASE), 1, "age"),
    (re.compile(r"\b(\d+)\s*[-–]?\s*years?\s*old\b", re.IGNORECASE), 1, "age"),
    # dose: "20 mg", "5.5 mg/kg", "100 mcg"
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg(?:/kg)?|mcg|g|IU)\b", re.IGNORECASE), 0, "dose"),
    # dose_amount (alias for regimen schema): extract just the number
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mg(?:/kg)?|mcg|g|IU)\b", re.IGNORECASE), 1, "dose_amount"),
    # dose_unit (alias for regimen schema): extract just the unit
    (re.compile(r"\b\d+(?:\.\d+)?\s*(mg(?:/kg)?|mcg|g|IU)\b", re.IGNORECASE), 1, "dose_unit"),
    # time: "day 84", "84 days", "week 12", "12 weeks", "6 months", "month 6", "2 years"
    (re.compile(r"\b(?:day)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:days?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(?:week)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:weeks?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(?:month)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:months?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(?:year)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "time"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:years?)\b", re.IGNORECASE), 0, "time"),
    # event: "progressed", "died", "death", "relapsed"
    (re.compile(r"\b(progressed|progression|died|death|deceased|relapsed|relapse|recurred|recurrence)\b", re.IGNORECASE), 0, "event"),
    # censored: "censored", "withdrew", "lost to follow-up", "alive", "stable"
    (re.compile(r"\b(censored|withdrew|withdrawal|discontinued|completed|alive|stable)\b", re.IGNORECASE), 0, "event"),
    (re.compile(r"(lost\s+to\s+follow[- ]?up|lfu)\b", re.IGNORECASE), 0, "event"),
    # sex/gender
    (re.compile(r"\b(male|female|M|F)\b", re.IGNORECASE), 0, "sex"),
    # stage
    (re.compile(r"\bstage\s+(I{1,3}V?|[1-4][A-Ca-c]?)\b", re.IGNORECASE), 0, "stage"),
    # ECOG PS
    (re.compile(r"\bECOG\s*(?:PS)?\s*(\d)\b", re.IGNORECASE), 0, "ecog_ps"),
    # weight
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*kg\b", re.IGNORECASE), 0, "weight"),
    # route
    (re.compile(r"\b(IV|intravenous|oral|PO|SC|subcutaneous|IM|intramuscular|topical|inhaled)\b", re.IGNORECASE), 0, "route"),
    # frequency
    (re.compile(r"\b(QD|BID|TID|QID|Q12H|Q8H|Q6H|once daily|twice daily|daily|weekly)\b", re.IGNORECASE), 0, "frequency"),
    # PK parameter names (for priors)
    (re.compile(r"\b(CL|clearance|V[12d]?|volume|ka|ke|Q|EC50|Emax|Hill|E0)\b(?:\s*[=:]?\s*(\d+(?:\.\d+)?))?", re.IGNORECASE), 0, "parameter_name"),
    # distribution type (for priors)
    (re.compile(r"\b(normal|lognormal|log-normal|half[_-]normal|uniform|gamma|exponential)\b", re.IGNORECASE), 0, "distribution"),
    # mean/sd (for priors) — "mean 3.5", "SD 1.2", "mean=3.5"
    (re.compile(r"\bmean\s*[=:]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "mean"),
    (re.compile(r"\b(?:SD|std|s\.d\.)\s*[=:]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "sd"),
    # units (for priors) — "L/h", "L", "1/h"
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(L/h|L|1/h|h|mg|mg/kg)\b"), 0, "units"),
    # occasion
    (re.compile(r"\b(?:occasion|cycle)\s*(\d+)\b", re.IGNORECASE), 0, "occasion"),
    # start_time
    (re.compile(r"\bstart(?:ing)?\s+(?:at\s+)?(?:day|week|month)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE), 0, "start_time"),
    # duration (for regimens)
    (re.compile(r"\bfor\s+(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)\b", re.IGNORECASE), 0, "duration"),
    # citation (for priors)
    (re.compile(r"\(([A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4})\)", 0), 1, "citation"),
]


class HeuristicBackend:
    """Regex-based backend for CI-safe testing without model downloads."""

    name = "heuristic"

    def extract_entities(self, text: str, schema: Union[Sequence[str], Dict[str, str]]) -> List[EntitySpan]:
        # Determine which labels are requested
        if isinstance(schema, dict):
            requested = set(schema.keys())
        else:
            requested = set(schema)

        out: List[EntitySpan] = []
        seen: set[tuple[int, int, str]] = set()

        for pattern, group_idx, label in _PATTERNS:
            if label not in requested:
                continue
            for m in pattern.finditer(text):
                span_text = m.group(group_idx).strip() if group_idx > 0 else m.group(0).strip()
                start = m.start(group_idx) if group_idx > 0 else m.start()
                end = m.end(group_idx) if group_idx > 0 else m.end()
                key = (start, end, label)
                if key in seen:
                    continue
                seen.add(key)
                out.append(EntitySpan(
                    label=label,
                    text=span_text,
                    start=start,
                    end=end,
                    score=1.0,
                ))

        return out

    def environment(self) -> Dict[str, Any]:
        return {"backend": "heuristic"}
