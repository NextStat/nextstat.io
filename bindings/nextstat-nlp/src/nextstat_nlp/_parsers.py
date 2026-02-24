"""Deterministic text parsing for clinical/statistical NLP.

All functions are pure, regex-based, and have zero ML dependencies.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_UNIT_TO_DAYS = {
    "day": 1.0,
    "days": 1.0,
    "d": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "wk": 7.0,
    "wks": 7.0,
    "w": 7.0,
    "month": 30.4375,
    "months": 30.4375,
    "mo": 30.4375,
    "mos": 30.4375,
    "year": 365.25,
    "years": 365.25,
    "yr": 365.25,
    "yrs": 365.25,
    "y": 365.25,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
    "hr": 1.0 / 24.0,
    "hrs": 1.0 / 24.0,
    "h": 1.0 / 24.0,
}

# "84 days", "12.5 weeks", "6 months", "2 years", "48 hours"
_PAT_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + "|".join(sorted(_UNIT_TO_DAYS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# "day 84", "week 12", "month 6", "year 2"
_PAT_UNIT_NUM = re.compile(
    r"\b(day|week|month|year|wk|mo|yr|hr|hour)\s+(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def parse_time(text: str) -> Optional[float]:
    """Extract time-to-event from text and normalise to days.

    Supports: "84 days", "day 84", "12 weeks", "week 12",
    "6 months", "month 6", "2 years", "year 2", decimals, hours.
    """
    # Try "unit NUMBER" first (more specific)
    m = _PAT_UNIT_NUM.search(text)
    if m:
        unit = m.group(1).lower()
        value = float(m.group(2))
        # map singular shorthand to full key
        factor = _UNIT_TO_DAYS.get(unit, None)
        if factor is not None:
            return round(value * factor, 6)

    # Then "NUMBER unit"
    m = _PAT_NUM_UNIT.search(text)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        factor = _UNIT_TO_DAYS.get(unit, None)
        if factor is not None:
            return round(value * factor, 6)

    return None


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

_EVENT_KEYWORDS = frozenset({
    "progressed", "progression", "died", "death", "deceased",
    "relapsed", "relapse", "recurred", "recurrence", "event",
})

_CENSORED_KEYWORDS = frozenset({
    "censored", "withdrew", "withdrawal", "lost", "lfu",
    "alive", "stable", "discontinued", "completed",
})

# Pre-compile patterns for whole-word matching
_EVENT_PAT = re.compile(
    r"\b(" + "|".join(sorted(_EVENT_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_CENSORED_PAT = re.compile(
    r"\b(" + "|".join(sorted(_CENSORED_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# Also match multi-word phrases
_CENSORED_PHRASE_PAT = re.compile(
    r"lost\s+to\s+follow[- ]?up",
    re.IGNORECASE,
)


def parse_event(text: str) -> Optional[int]:
    """Determine event (1) vs censored (0) from text keywords."""
    has_event = bool(_EVENT_PAT.search(text))
    has_censored = bool(_CENSORED_PAT.search(text)) or bool(_CENSORED_PHRASE_PAT.search(text))

    if has_event and not has_censored:
        return 1
    if has_censored and not has_event:
        return 0
    if has_event and has_censored:
        # Ambiguous — prefer censored (conservative in survival analysis)
        return 0
    return None


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_NUMERIC_PAT = re.compile(r"(\d+(?:\.\d+)?)")


def parse_numeric(text: str, label: str = "") -> Optional[float]:
    """Extract first numeric value from a span of text.

    >>> parse_numeric("63 years", "age")
    63.0
    >>> parse_numeric("20 mg", "dose")
    20.0
    >>> parse_numeric("5.5 mg/kg", "dose")
    5.5
    """
    m = _NUMERIC_PAT.search(text)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("mg", "g"): 0.001,
    ("g", "mg"): 1000.0,
    ("mcg", "mg"): 0.001,
    ("mg", "mcg"): 1000.0,
    ("kg", "g"): 1000.0,
    ("g", "kg"): 0.001,
    ("weeks", "days"): 7.0,
    ("months", "days"): 30.4375,
    ("years", "days"): 365.25,
    ("hours", "days"): 1.0 / 24.0,
    ("days", "weeks"): 1.0 / 7.0,
    ("days", "months"): 1.0 / 30.4375,
    ("days", "years"): 1.0 / 365.25,
}


def normalize_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert *value* from *from_unit* to *to_unit*.

    Raises ``ValueError`` if the conversion is unknown.
    """
    if from_unit == to_unit:
        return value
    key = (from_unit.lower(), to_unit.lower())
    factor = _UNIT_CONVERSIONS.get(key)
    if factor is None:
        raise ValueError(f"Unknown unit conversion: {from_unit} -> {to_unit}")
    return value * factor
