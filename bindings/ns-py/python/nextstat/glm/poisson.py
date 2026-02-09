"""Poisson regression (log link) with optional offset/exposure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
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

    if l2 is None or float(l2) <= 0.0:
        model = nextstat._core.PoissonRegressionModel(
            x2,
            y2,
            include_intercept=include_intercept,
            offset=off2,
        )
        r = nextstat.fit(model)
    else:
        lam = float(l2)
        sigma = 1.0 / math.sqrt(lam)
        model = nextstat._core.ComposedGlmModel.poisson_regression(
            x2,
            y2,
            include_intercept=include_intercept,
            offset=off2,
            coef_prior_mu=0.0,
            coef_prior_sigma=sigma,
            penalize_intercept=penalize_intercept,
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

def from_formula(
    formula: str,
    data: Any,
    *,
    categorical: Optional[Sequence[str]] = None,
    offset: Optional[Union[str, Sequence[float]]] = None,
    exposure: Optional[Union[str, Sequence[float]]] = None,
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> Tuple[PoissonFit, List[str]]:
    """Fit Poisson regression from a minimal formula and tabular data.

    `offset`/`exposure` may be a sequence, or a column name in `data`.
    Returns `(fit, column_names)` where `column_names` matches the coefficient order.
    """
    import nextstat

    y_name, terms, _include_intercept = nextstat.formula.parse_formula(formula)
    extra_cols: list[str] = []
    if isinstance(offset, str):
        extra_cols.append(offset)
    if isinstance(exposure, str):
        extra_cols.append(exposure)

    cols = nextstat.formula.to_columnar(data, [y_name] + terms + extra_cols)
    y_float, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    y = [int(v) for v in y_float]
    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        feature_names = list(names_full[1:])
        include_intercept = True
    else:
        x = x_full
        feature_names = list(names_full)
        include_intercept = False

    off_seq: Optional[Sequence[float]] = None
    exp_seq: Optional[Sequence[float]] = None
    if isinstance(offset, str):
        off_seq = [float(v) for v in cols[offset]]
    else:
        off_seq = offset
    if isinstance(exposure, str):
        exp_seq = [float(v) for v in cols[exposure]]
    else:
        exp_seq = exposure

    r = fit(
        x,
        y,
        include_intercept=include_intercept,
        offset=off_seq,
        exposure=exp_seq,
        l2=l2,
        penalize_intercept=penalize_intercept,
    )
    colnames = (["Intercept"] if include_intercept else []) + feature_names
    return r, colnames


__all__ = ["PoissonFit", "fit", "from_formula"]
