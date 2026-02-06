"""TREx replacement baseline comparator (numbers-first).

This module is intentionally dependency-free (no numpy), deterministic, and suitable
for pytest golden numeric tests.

Design goals
- Compare by *meaning* (parameter name alignment), not by index.
- Deterministic ordering of diffs (stable sort).
- Clear diff locations (path + optional index + optional name).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class Tol:
    """Scalar tolerance: pass if abs(a-b) <= atol + rtol*abs(b)."""

    atol: float
    rtol: float


@dataclass(frozen=True)
class Diff:
    path: str
    ref: Any
    cand: Any
    abs_diff: float | None
    rel_diff: float | None
    note: str | None = None


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    diffs: list[Diff]

    def worst(self, n: int = 20) -> list[Diff]:
        n = int(n)
        if n <= 0:
            return []
        return self.diffs[:n]


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _rel_diff(a: float, b: float) -> float | None:
    den = abs(b)
    if den == 0.0:
        return None
    return abs(a - b) / den


def _same_inf(a: float, b: float) -> bool:
    # Both infinite and same sign.
    return (a == float("inf") and b == float("inf")) or (a == float("-inf") and b == float("-inf"))


def compare_scalar(*, path: str, ref: Any, cand: Any, tol: Tol) -> Diff | None:
    """Return a Diff if mismatch else None."""

    a = _as_float(ref)
    b = _as_float(cand)
    if a is None or b is None:
        if ref != cand:
            return Diff(path=path, ref=ref, cand=cand, abs_diff=None, rel_diff=None, note="non-numeric mismatch")
        return None

    if isnan(a) or isnan(b):
        if isnan(a) and isnan(b):
            return Diff(path=path, ref=ref, cand=cand, abs_diff=None, rel_diff=None, note="both NaN (treated as mismatch)")
        return Diff(path=path, ref=ref, cand=cand, abs_diff=None, rel_diff=None, note="NaN mismatch")

    if not isfinite(a) or not isfinite(b):
        if _same_inf(a, b):
            return None
        return Diff(path=path, ref=ref, cand=cand, abs_diff=None, rel_diff=None, note="inf mismatch")

    abs_d = abs(a - b)
    allowed = float(tol.atol) + float(tol.rtol) * abs(b)
    if abs_d <= allowed:
        return None
    return Diff(path=path, ref=a, cand=b, abs_diff=abs_d, rel_diff=_rel_diff(a, b))


def _sort_key(d: Diff) -> tuple[float, str]:
    # Deterministic ordering: primary = abs_diff descending; None treated as +inf (worst),
    # tie-breaker = path lexicographic.
    primary = float("inf") if d.abs_diff is None else float(d.abs_diff)
    return (-primary, d.path)


def _sorted_diffs(diffs: Iterable[Diff]) -> list[Diff]:
    return sorted(list(diffs), key=_sort_key)


def align_named_params(
    *,
    ref_params: Sequence[Mapping[str, Any]],
    cand_params: Sequence[Mapping[str, Any]],
    base_path: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Diff]]:
    """Align parameter lists by `name` and return (aligned_ref, aligned_cand, diffs)."""

    ref_by_name: MutableMapping[str, Mapping[str, Any]] = {}
    for p in ref_params:
        name = str(p.get("name", ""))
        if not name:
            continue
        # If duplicates exist, keep first; duplicates should be flagged elsewhere.
        ref_by_name.setdefault(name, p)

    cand_by_name: MutableMapping[str, Mapping[str, Any]] = {}
    for p in cand_params:
        name = str(p.get("name", ""))
        if not name:
            continue
        cand_by_name.setdefault(name, p)

    diffs: list[Diff] = []
    aligned_ref: list[Mapping[str, Any]] = []
    aligned_cand: list[Mapping[str, Any]] = []

    for name in sorted(ref_by_name.keys()):
        if name not in cand_by_name:
            diffs.append(
                Diff(
                    path=f"{base_path}.parameters[{name}]",
                    ref=ref_by_name[name],
                    cand=None,
                    abs_diff=None,
                    rel_diff=None,
                    note="missing parameter in candidate",
                )
            )
            continue
        aligned_ref.append(ref_by_name[name])
        aligned_cand.append(cand_by_name[name])

    for name in sorted(cand_by_name.keys()):
        if name not in ref_by_name:
            diffs.append(
                Diff(
                    path=f"{base_path}.parameters[{name}]",
                    ref=None,
                    cand=cand_by_name[name],
                    abs_diff=None,
                    rel_diff=None,
                    note="extra parameter in candidate",
                )
            )

    return aligned_ref, aligned_cand, diffs


def compare_param_list(
    *,
    path: str,
    ref_params: Sequence[Mapping[str, Any]],
    cand_params: Sequence[Mapping[str, Any]],
    tol_value: Tol,
    tol_unc: Tol,
) -> list[Diff]:
    ref_aligned, cand_aligned, diffs = align_named_params(
        ref_params=ref_params, cand_params=cand_params, base_path=path
    )

    for rp, cp in zip(ref_aligned, cand_aligned):
        name = str(rp.get("name", ""))
        diffs.extend(
            d
            for d in [
                compare_scalar(path=f"{path}.parameters[{name}].value", ref=rp.get("value"), cand=cp.get("value"), tol=tol_value),
                compare_scalar(
                    path=f"{path}.parameters[{name}].uncertainty",
                    ref=rp.get("uncertainty"),
                    cand=cp.get("uncertainty"),
                    tol=tol_unc,
                ),
            ]
            if d is not None
        )
    return diffs


def compare_numeric_vector(
    *,
    path: str,
    ref: Sequence[Any],
    cand: Sequence[Any],
    tol: Tol,
) -> list[Diff]:
    diffs: list[Diff] = []
    if len(ref) != len(cand):
        diffs.append(
            Diff(
                path=path,
                ref={"len": len(ref)},
                cand={"len": len(cand)},
                abs_diff=None,
                rel_diff=None,
                note="length mismatch",
            )
        )
        return diffs
    for i, (a, b) in enumerate(zip(ref, cand)):
        d = compare_scalar(path=f"{path}[{i}]", ref=a, cand=b, tol=tol)
        if d is not None:
            diffs.append(d)
    return diffs


def compare_baseline_v0(
    *,
    ref: Mapping[str, Any],
    cand: Mapping[str, Any],
    tol_twice_nll: Tol,
    tol_expected_data: Tol,
    tol_param_value: Tol,
    tol_param_unc: Tol,
) -> CompareResult:
    """Minimal v0 baseline compare: fit + expected_data surfaces.

    This intentionally starts small and grows as we add report artifacts.
    """

    diffs: list[Diff] = []

    # Fit surface
    diffs.extend(
        d
        for d in [
            compare_scalar(
                path="fit.twice_nll",
                ref=(ref.get("fit") or {}).get("twice_nll"),
                cand=(cand.get("fit") or {}).get("twice_nll"),
                tol=tol_twice_nll,
            )
        ]
        if d is not None
    )

    ref_params = (ref.get("fit") or {}).get("parameters") or []
    cand_params = (cand.get("fit") or {}).get("parameters") or []
    if not isinstance(ref_params, (list, tuple)) or not isinstance(cand_params, (list, tuple)):
        diffs.append(
            Diff(
                path="fit.parameters",
                ref=type(ref_params).__name__,
                cand=type(cand_params).__name__,
                abs_diff=None,
                rel_diff=None,
                note="parameters must be a list",
            )
        )
    else:
        diffs.extend(
            compare_param_list(
                path="fit",
                ref_params=ref_params,
                cand_params=cand_params,
                tol_value=tol_param_value,
                tol_unc=tol_param_unc,
            )
        )

    # Expected data surface
    ref_ed = ref.get("expected_data") or {}
    cand_ed = cand.get("expected_data") or {}
    for key in ["pyhf_main", "pyhf_with_aux"]:
        ra = ref_ed.get(key)
        ca = cand_ed.get(key)
        if not isinstance(ra, (list, tuple)) or not isinstance(ca, (list, tuple)):
            diffs.append(
                Diff(
                    path=f"expected_data.{key}",
                    ref=type(ra).__name__,
                    cand=type(ca).__name__,
                    abs_diff=None,
                    rel_diff=None,
                    note="expected_data arrays must be lists",
                )
            )
            continue
        diffs.extend(compare_numeric_vector(path=f"expected_data.{key}", ref=ra, cand=ca, tol=tol_expected_data))

    diffs = _sorted_diffs(diffs)
    return CompareResult(ok=(len(diffs) == 0), diffs=diffs)


def format_diff(d: Diff) -> str:
    if d.note:
        return f"{d.path}: {d.note} (ref={d.ref!r} cand={d.cand!r})"
    if d.abs_diff is None:
        return f"{d.path}: mismatch (ref={d.ref!r} cand={d.cand!r})"
    if d.rel_diff is None:
        return f"{d.path}: abs_diff={d.abs_diff:.6g} (ref={d.ref!r} cand={d.cand!r})"
    return f"{d.path}: abs_diff={d.abs_diff:.6g} rel_diff={d.rel_diff:.6g} (ref={d.ref!r} cand={d.cand!r})"


def format_report(res: CompareResult, *, top_n: int = 20) -> str:
    lines = []
    lines.append("OK" if res.ok else "FAIL")
    if res.diffs:
        lines.append(f"diffs={len(res.diffs)}")
        for d in res.worst(top_n):
            lines.append(f"- {format_diff(d)}")
    return "\n".join(lines)

