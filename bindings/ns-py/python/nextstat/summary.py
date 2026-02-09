"""Fit summary helpers (Phase 11.3).

This module provides a dependency-light, deterministic summary surface:
- coef / std_err
- Wald z-statistic + two-sided p-value (normal approximation)
- confidence intervals

Notes
- The default p-values/CI use asymptotic normality (Wald). This is a baseline API
  designed to be usable without SciPy/pandas.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _two_sided_p_from_z(z: float) -> float:
    # p = 2 * (1 - Phi(|z|))
    a = abs(float(z))
    p = 2.0 * (1.0 - _normal_cdf(a))
    # numeric guardrails
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def wald_summary(
    coef: Sequence[float],
    std_err: Sequence[float],
    *,
    names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    """Compute a Wald (normal) summary table for coefficients."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    c = [float(v) for v in coef]
    s = [float(v) for v in std_err]
    if len(c) != len(s):
        raise ValueError("coef and std_err must have the same length")

    if names is None:
        nms = [f"x{i}" for i in range(len(c))]
    else:
        nms = [str(v) for v in names]
        if len(nms) != len(c):
            raise ValueError("names must have the same length as coef")

    # For alpha=0.05, zcrit ~ 1.959964. Compute via inverse-CDF approximation is overkill;
    # keep a small lookup for common cases and fall back to 1.96.
    if abs(float(alpha) - 0.05) < 1e-12:
        zcrit = 1.959963984540054
    elif abs(float(alpha) - 0.10) < 1e-12:
        zcrit = 1.6448536269514722
    else:
        # Normal quantile approximation is not available in stdlib; baseline uses 1.96.
        zcrit = 1.959963984540054

    stat: list[float] = []
    p_value: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for b, se in zip(c, s):
        if se == 0.0:
            z = float("inf") if b != 0.0 else 0.0
        else:
            z = b / se
        stat.append(z)
        p_value.append(_two_sided_p_from_z(z))
        ci_low.append(b - zcrit * se)
        ci_high.append(b + zcrit * se)

    return {
        "names": nms,
        "coef": c,
        "std_err": s,
        "stat_name": "z",
        "stat": stat,
        "p_value": p_value,
        "alpha": float(alpha),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def fit_summary(
    fit: Any,
    *,
    names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> Mapping[str, Any]:
    """Build a Wald summary for supported NextStat fit objects."""
    # High-level GLM surfaces expose dataclasses with `.coef` / `.standard_errors`.
    if hasattr(fit, "coef") and hasattr(fit, "standard_errors"):
        coef = list(getattr(fit, "coef"))
        se = list(getattr(fit, "standard_errors"))

        # Negative binomial: include dispersion parameter in the summary.
        if hasattr(fit, "log_alpha") and hasattr(fit, "log_alpha_se"):
            coef = [float(v) for v in coef] + [float(getattr(fit, "log_alpha"))]
            se = [float(v) for v in se] + [float(getattr(fit, "log_alpha_se"))]
            if names is not None:
                names = list(names) + ["log_alpha"]
            else:
                names = [f"x{i}" for i in range(len(coef) - 1)] + ["log_alpha"]

        return wald_summary(coef, se, names=names, alpha=alpha)

    # Core FitResult exposes `.parameters` / `.uncertainties`.
    if hasattr(fit, "parameters") and hasattr(fit, "uncertainties"):
        return wald_summary(
            list(getattr(fit, "parameters")),
            list(getattr(fit, "uncertainties")),
            names=names,
            alpha=alpha,
        )

    raise TypeError("unsupported fit type for summary: expected GLM fit dataclass or FitResult")


def summary_to_str(summary: Mapping[str, Any], *, digits: int = 4) -> str:
    """Render a summary mapping as a simple aligned text table."""
    names = list(summary.get("names") or [])
    coef = list(summary.get("coef") or [])
    se = list(summary.get("std_err") or [])
    stat_name = str(summary.get("stat_name") or "z")
    stat = list(summary.get("stat") or [])
    pval = list(summary.get("p_value") or [])
    lo = list(summary.get("ci_low") or [])
    hi = list(summary.get("ci_high") or [])

    n = min(len(names), len(coef), len(se), len(stat), len(pval), len(lo), len(hi))
    if n == 0:
        return "empty summary"

    def fmt(x: float) -> str:
        return f"{float(x):.{int(digits)}g}"

    rows: list[list[str]] = []
    rows.append(["name", "coef", "std_err", stat_name, "p_value", "ci_low", "ci_high"])
    for i in range(n):
        rows.append(
            [
                str(names[i]),
                fmt(coef[i]),
                fmt(se[i]),
                fmt(stat[i]),
                fmt(pval[i]),
                fmt(lo[i]),
                fmt(hi[i]),
            ]
        )

    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    out_lines: list[str] = []
    for ridx, r in enumerate(rows):
        line = "  ".join(v.ljust(widths[i]) for i, v in enumerate(r))
        out_lines.append(line.rstrip())
        if ridx == 0:
            out_lines.append("  ".join("-" * w for w in widths).rstrip())
    return "\n".join(out_lines)


__all__ = ["wald_summary", "fit_summary", "summary_to_str"]

