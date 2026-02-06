import math

import pytest

import nextstat


def test_exponential_survival_contract_and_fit() -> None:
    times = [0.5, 1.2, 0.7, 2.0, 0.9]
    events = [True, False, True, False, True]
    m = nextstat.ExponentialSurvivalModel(times, events)

    assert m.n_params() == 1
    assert m.parameter_names() == ["log_rate"]

    init = m.suggested_init()
    nll = m.nll(init)
    g = m.grad_nll(init)
    assert math.isfinite(nll)
    assert len(g) == 1
    assert all(math.isfinite(float(x)) for x in g)

    res = nextstat.fit(m)
    assert len(res.bestfit) == 1
    assert math.isfinite(res.nll)


def test_weibull_survival_rejects_non_positive_times() -> None:
    with pytest.raises(ValueError):
        nextstat.WeibullSurvivalModel([0.0, 1.0], [True, False])


def test_lognormal_survival_rejects_non_positive_times() -> None:
    with pytest.raises(ValueError):
        nextstat.LogNormalAftModel([0.0, 1.0], [True, False])


def test_survival_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        nextstat.ExponentialSurvivalModel([1.0, 2.0], [True])

