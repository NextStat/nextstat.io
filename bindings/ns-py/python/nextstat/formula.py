"""Minimal formula parsing + tabular design matrix building (Phase 11).

Goals:
- Dependency-light (no pandas/patsy required).
- Deterministic output: stable column ordering and names.
- Minimal supported grammar:
  - `y ~ 1 + x1 + x2`
  - `-1` or `0` to drop intercept

Categoricals:
- Explicit opt-in via `categorical=[...]`.
- One-hot encoding for string-like columns (deterministic category order).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple


class FormulaError(ValueError):
    """Raised for invalid or unsupported formulas."""


def _is_sequence(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (bytes, str))


def _as_1d_list(name: str, x: Any) -> list[Any]:
    if not _is_sequence(x):
        raise TypeError(f"{name} must be a 1D sequence")
    return list(x)


def _as_float_list(name: str, x: Any) -> list[float]:
    xs = _as_1d_list(name, x)
    out: list[float] = []
    for v in xs:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"{name} must contain only finite values")
        out.append(fv)
    return out


def _tokenize_rhs(rhs: str) -> list[str]:
    # Tokenize by whitespace and +/-, keeping the operators.
    tokens: list[str] = []
    cur: list[str] = []
    for ch in rhs:
        if ch in "+-":
            if cur:
                tok = "".join(cur).strip()
                if tok:
                    tokens.append(tok)
                cur = []
            tokens.append(ch)
        else:
            cur.append(ch)
    if cur:
        tok = "".join(cur).strip()
        if tok:
            tokens.append(tok)
    return [t for t in tokens if t]


def parse_formula(formula: str) -> Tuple[str, list[str], bool]:
    """Parse a minimal formula and return (y_name, terms, include_intercept)."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")

    s = formula.strip()
    if "~" not in s:
        raise FormulaError("formula must contain '~' (e.g. 'y ~ 1 + x')")

    lhs, rhs = s.split("~", 1)
    y_name = lhs.strip()
    if not y_name:
        raise FormulaError("formula LHS must be a response column name")

    rhs = rhs.strip()
    if not rhs:
        raise FormulaError("formula RHS must be non-empty")

    # Reject unsupported operators early (keep minimal + explicit).
    for bad in (":", "*", "/", "(", ")", "^"):
        if bad in rhs:
            raise FormulaError(f"unsupported operator {bad!r} in formula RHS")

    tokens = _tokenize_rhs(rhs)
    if not tokens:
        raise FormulaError("formula RHS must be non-empty")

    include_intercept = True
    terms: list[str] = []

    op = "+"  # implicit leading +
    for tok in tokens:
        if tok in ("+", "-"):
            op = tok
            continue

        t = tok.strip()
        if not t:
            continue

        if t in ("1",):
            if op == "-":
                # Common shorthand: `... - 1` drops the intercept.
                include_intercept = False
                continue
            include_intercept = True
            continue

        if t in ("0", "-1"):
            if op == "-":
                # `y ~ x - 0` doesn't make semantic sense; keep strict.
                raise FormulaError("use '+ 0' or '+ -1' to drop intercept")
            include_intercept = False
            continue

        if op == "-":
            raise FormulaError(f"term removal is not supported in this minimal grammar: '- {t}'")

        # Minimal identifier validation: allow common column-name chars.
        for ch in t:
            ok = ch.isalnum() or ch in ("_", ".", "[", "]")
            if not ok:
                raise FormulaError(f"invalid term name {t!r} in formula RHS")

        terms.append(t)

    # If RHS only set intercept tokens (e.g. y ~ 1), keep it valid with 0 terms.
    return y_name, terms, include_intercept


def design_matrices(
    formula: str,
    data: Mapping[str, Sequence[Any]],
    *,
    categorical: Optional[Sequence[str]] = None,
) -> Tuple[list[float], list[list[float]], list[str]]:
    """Build deterministic (y, X, column_names) from a minimal formula and tabular data.

    Parameters
    - data: dict-of-columns; all columns must have the same length.
    - categorical: explicit list of column names to one-hot encode.
    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of column_name -> sequence")

    y_name, terms, include_intercept = parse_formula(formula)

    if y_name not in data:
        raise KeyError(f"response column {y_name!r} not found in data")

    y = _as_float_list(y_name, data[y_name])
    n = len(y)
    if n == 0:
        raise ValueError("data must have at least 1 row")

    cat = set(str(c) for c in (categorical or ()))

    # Build columns (as lists), then transpose to rows.
    col_names: list[str] = []
    cols: list[list[float]] = []

    if include_intercept:
        col_names.append("Intercept")
        cols.append([1.0] * n)

    for term in terms:
        if term not in data:
            raise KeyError(f"term column {term!r} not found in data")

        raw = _as_1d_list(term, data[term])
        if len(raw) != n:
            raise ValueError(f"column {term!r} has length {len(raw)}, expected {n}")

        if term in cat:
            cats = [("None" if v is None else str(v)) for v in raw]
            levels = sorted(set(cats))
            if include_intercept:
                # Drop the first level to avoid collinearity with the intercept.
                levels_to_encode = levels[1:]
                prefix = f"{term}[T."
                suffix = "]"
            else:
                levels_to_encode = levels
                prefix = f"{term}["
                suffix = "]"

            for lvl in levels_to_encode:
                col_names.append(f"{prefix}{lvl}{suffix}")
                cols.append([1.0 if v == lvl else 0.0 for v in cats])
        else:
            col_names.append(term)
            cols.append(_as_float_list(term, raw))

    # Transpose to X rows.
    if not cols:
        # This can happen only if include_intercept=false and RHS is empty.
        raise FormulaError("design matrix would be empty; add terms or intercept")

    x: list[list[float]] = []
    for i in range(n):
        x.append([col[i] for col in cols])

    return y, x, col_names


__all__ = ["FormulaError", "parse_formula", "design_matrices"]
