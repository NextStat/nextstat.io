"""Missing data policies (Phase 9.C.2).

Goal: make missing-data handling explicit and reproducible.

This module is intentionally dependency-light (no pandas/numpy requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, List, Literal, Optional, Sequence, Tuple


MissingPolicy = Literal["drop_rows", "impute_mean"]


def _tolist(x: Any) -> Any:
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    return x


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _as_2d_any(x: Any) -> List[List[Any]]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError("X must be a 2D sequence (or numpy array).")
    out: List[List[Any]] = []
    for row in x:
        row = _tolist(row)
        if not isinstance(row, Sequence) or isinstance(row, (bytes, str)):
            raise TypeError("X must be a 2D sequence (or numpy array).")
        out.append(list(row))
    return out


def _as_1d_any(y: Any) -> List[Any]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    return list(y)


def _safe_float(v: Any) -> float:
    if _is_missing(v):
        raise ValueError("missing value cannot be converted to float")
    try:
        return float(v)
    except Exception as e:
        raise TypeError(f"value {v!r} is not numeric") from e


@dataclass(frozen=True)
class MissingResult:
    x: List[List[float]]
    y: Optional[List[Any]]
    kept_row_mask: List[bool]
    x_missing_mask: List[List[bool]]
    y_missing_mask: Optional[List[bool]]

    @property
    def n_dropped(self) -> int:
        return int(sum(1 for k in self.kept_row_mask if not k))

    @property
    def n_kept(self) -> int:
        return int(sum(1 for k in self.kept_row_mask if k))


def apply_policy(
    x: Any,
    y: Any = None,
    *,
    policy: MissingPolicy = "drop_rows",
) -> MissingResult:
    """Apply an explicit missing-data policy to (X, y).

    Policies:
    - `drop_rows`: drop any row with missing in X or y.
    - `impute_mean`: impute missing X with per-column mean; drop rows with missing y.

    Missing values are `None` or `float('nan')`.
    """

    xa = _as_2d_any(x)
    ya = None if y is None else _as_1d_any(y)

    if not xa:
        return MissingResult(x=[], y=([] if ya is not None else None), kept_row_mask=[], x_missing_mask=[], y_missing_mask=([] if ya is not None else None))

    n = len(xa)
    p = len(xa[0])
    if any(len(row) != p for row in xa):
        raise ValueError("X must be rectangular (all rows same length)")
    if ya is not None and len(ya) != n:
        raise ValueError("y length must match number of rows in X")

    x_missing_mask: List[List[bool]] = [[_is_missing(v) for v in row] for row in xa]
    y_missing_mask: Optional[List[bool]] = None
    if ya is not None:
        y_missing_mask = [_is_missing(v) for v in ya]

    if policy == "drop_rows":
        kept = []
        out_x: List[List[float]] = []
        out_y: Optional[List[Any]] = [] if ya is not None else None
        for i in range(n):
            row_has_missing = any(x_missing_mask[i])
            y_is_missing = bool(y_missing_mask[i]) if y_missing_mask is not None else False
            keep = (not row_has_missing) and (not y_is_missing)
            kept.append(bool(keep))
            if not keep:
                continue
            out_x.append([_safe_float(v) for v in xa[i]])
            if out_y is not None:
                out_y.append(ya[i])
        return MissingResult(
            x=out_x,
            y=out_y,
            kept_row_mask=kept,
            x_missing_mask=x_missing_mask,
            y_missing_mask=y_missing_mask,
        )

    if policy == "impute_mean":
        # Column means for X (ignore missing).
        sums = [0.0] * p
        counts = [0] * p
        for i in range(n):
            for j in range(p):
                v = xa[i][j]
                if _is_missing(v):
                    continue
                fv = _safe_float(v)
                if not math.isfinite(fv):
                    raise ValueError("X contains non-finite value")
                sums[j] += fv
                counts[j] += 1
        means = []
        for j in range(p):
            if counts[j] == 0:
                raise ValueError(f"cannot impute column {j}: all values missing")
            means.append(sums[j] / float(counts[j]))

        kept = []
        out_x: List[List[float]] = []
        out_y: Optional[List[Any]] = [] if ya is not None else None

        for i in range(n):
            y_is_missing = bool(y_missing_mask[i]) if y_missing_mask is not None else False
            keep = not y_is_missing
            kept.append(bool(keep))
            if not keep:
                continue
            row = []
            for j in range(p):
                v = xa[i][j]
                if _is_missing(v):
                    row.append(float(means[j]))
                else:
                    row.append(_safe_float(v))
            out_x.append(row)
            if out_y is not None:
                out_y.append(ya[i])

        return MissingResult(
            x=out_x,
            y=out_y,
            kept_row_mask=kept,
            x_missing_mask=x_missing_mask,
            y_missing_mask=y_missing_mask,
        )

    raise ValueError(f"Unsupported policy: {policy!r}")


__all__ = ["MissingPolicy", "MissingResult", "apply_policy"]

