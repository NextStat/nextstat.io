"""Poisson regression (log link) with optional offset/exposure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

from ._linalg import add_intercept, as_2d_float_list, mat_vec_mul


@dataclass(frozen=True)
class PoissonFit:
    coef: List[float]
    standard_errors: List[float]
    nll: float
    converged: bool
    include_intercept: bool
    offset: Optional[List[float]]

    def predict_mean(
        self,
        x: Sequence[Sequence[float]],
        *,
        offset: Optional[Sequence[float]] = None,
        exposure: Optional[Sequence[float]] = None,
    ) -> List[float]:
        if offset is not None and exposure is not None:
            raise ValueError("Specify only one of offset= or exposure=")

        x2 = as_2d_float_list(x)
        xd = add_intercept(x2) if self.include_intercept else x2
        eta = mat_vec_mul(xd, self.coef)

        off: Optional[List[float]] = None
        if exposure is not None:
            off = []
            for v in exposure:
                ev = float(v)
                if not (ev > 0.0) or not math.isfinite(ev):
                    raise ValueError("exposure must be finite and > 0")
                off.append(math.log(ev))
        elif offset is not None:
            off = [float(v) for v in offset]
        else:
            off = self.offset

        if off is not None:
            if len(off) != len(eta):
                raise ValueError("offset/exposure must have length n")
            eta = [e + o for e, o in zip(eta, off)]

        return [math.exp(v) for v in eta]

    def predict_rate(
        self,
        x: Sequence[Sequence[float]],
        *,
        offset: Optional[Sequence[float]] = None,
        exposure: Optional[Sequence[float]] = None,
    ) -> List[float]:
        return self.predict_mean(x, offset=offset, exposure=exposure)


def fit(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    include_intercept: bool = True,
    offset: Optional[Sequence[float]] = None,
    exposure: Optional[Sequence[float]] = None,
) -> PoissonFit:
    import nextstat

    if offset is not None and exposure is not None:
        raise ValueError("Specify only one of offset= or exposure=")

    x2 = as_2d_float_list(x)
    y2: List[int] = []
    for v in y:
        iv = int(v)
        if iv < 0:
            raise ValueError("y must be non-negative for Poisson regression")
        y2.append(iv)

    off2: Optional[List[float]] = None
    if exposure is not None:
        off2 = []
        for v in exposure:
            ev = float(v)
            if not (ev > 0.0) or not math.isfinite(ev):
                raise ValueError("exposure must be finite and > 0")
            off2.append(math.log(ev))
    elif offset is not None:
        off2 = [float(v) for v in offset]

    if off2 is not None and len(off2) != len(y2):
        raise ValueError("offset/exposure must have length n")

    model = nextstat._core.PoissonRegressionModel(
        x2,
        y2,
        include_intercept=include_intercept,
        offset=off2,
    )
    r = nextstat.fit(model)

    return PoissonFit(
        coef=list(r.parameters),
        standard_errors=list(r.uncertainties),
        nll=float(r.nll),
        converged=bool(r.converged),
        include_intercept=include_intercept,
        offset=off2,
    )


__all__ = ["PoissonFit", "fit"]
