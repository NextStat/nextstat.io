"""Phase 9: Cox PH parity vs statsmodels (optional).

This is a *reference parity* test:
- It is skipped unless `statsmodels` (and its deps) are installed.
- It validates that NextStat Cox partial likelihood matches an independent reference
  implementation (statsmodels PHReg) for both Breslow and Efron ties.
"""

from __future__ import annotations

import pytest

import nextstat


statsmodels = pytest.importorskip("statsmodels")
np = pytest.importorskip("numpy")


def _fit_statsmodels(*, times, events01, x, ties: str):
    from statsmodels.duration.hazard_regression import PHReg

    t = np.asarray(times, dtype=float)
    status = np.asarray(events01, dtype=int)
    X = np.asarray(x, dtype=float)
    m = PHReg(t, X, status=status, ties=str(ties))
    res = m.fit(disp=False)
    return res


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_ph_fit_and_llf_parity_vs_statsmodels(ties: str) -> None:
    # Includes ties and censoring.
    times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
    events = [True, True, False, True, False, False]
    events01 = [1 if bool(v) else 0 for v in events]
    x = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [0.5, 0.5],
    ]

    ns_m = nextstat.CoxPhModel(times, events, x, ties=ties)
    ns_fit = nextstat.fit(ns_m)
    assert bool(ns_fit.converged)

    sm_fit = _fit_statsmodels(times=times, events01=events01, x=x, ties=ties)

    ns_beta = [float(v) for v in ns_fit.bestfit]
    sm_beta = [float(v) for v in sm_fit.params]

    # Parameter agreement: should be tight for this small problem.
    assert ns_beta == pytest.approx(sm_beta, rel=0.0, abs=2e-6)

    # Likelihood agreement.
    # NextStat reports NLL; statsmodels reports log-likelihood.
    ns_llf = -float(ns_fit.nll)
    sm_llf = float(sm_fit.llf)
    assert ns_llf == pytest.approx(sm_llf, rel=0.0, abs=2e-6)

