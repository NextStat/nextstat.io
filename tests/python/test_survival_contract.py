import math

import pytest

import nextstat


def _exp_clamped(x: float) -> float:
    # Mirror ns_prob::math::exp_clamped (~exp with clamp at ~700 to avoid inf).
    if x > 700.0:
        x = 700.0
    if x < -700.0:
        x = -700.0
    return math.exp(x)


def _cox_nll_reference(times, events, x, beta, *, ties: str) -> float:
    # Reference implementation matching crates/ns-inference/src/survival.rs:
    # sort by time descending, accumulate risk set, group exact ties.
    order = sorted(range(len(times)), key=lambda i: float(times[i]), reverse=True)
    times_s = [float(times[i]) for i in order]
    events_s = [bool(events[i]) for i in order]
    x_s = [[float(v) for v in x[i]] for i in order]

    # group_starts: indices where time changes
    group_starts = [0]
    for i in range(1, len(times_s)):
        if times_s[i] != times_s[i - 1]:
            group_starts.append(i)

    eta = [sum(float(xij) * float(bj) for xij, bj in zip(xi, beta)) for xi in x_s]
    w = [_exp_clamped(e) for e in eta]

    ll = 0.0
    risk0 = 0.0
    for g, start in enumerate(group_starts):
        end = group_starts[g + 1] if g + 1 < len(group_starts) else len(times_s)
        for i in range(start, end):
            risk0 += w[i]

        m = 0
        sum_eta_events = 0.0
        d0 = 0.0
        for i in range(start, end):
            if events_s[i]:
                m += 1
                sum_eta_events += eta[i]
                d0 += w[i]
        if m == 0:
            continue

        ll += sum_eta_events
        if ties == "breslow":
            ll -= float(m) * math.log(max(risk0, 1e-300))
        elif ties == "efron":
            mf = float(m)
            for r in range(m):
                frac = float(r) / mf
                denom = max(risk0 - frac * d0, 1e-300)
                ll -= math.log(denom)
        else:
            raise ValueError("ties must be 'breslow' or 'efron'")

    return -ll


def _finite_diff_grad(f, beta, eps: float = 1e-6):
    g = []
    for i in range(len(beta)):
        b_hi = list(beta)
        b_lo = list(beta)
        b_hi[i] += eps
        b_lo[i] -= eps
        g.append((float(f(b_hi)) - float(f(b_lo))) / (2.0 * eps))
    return g


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


def test_cox_ph_contract_and_fit_smoke() -> None:
    times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
    events = [True, True, False, True, False, False]
    x = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [0.5, 0.5],
    ]

    m = nextstat.CoxPhModel(times, events, x, ties="efron")
    assert m.n_params() == 2
    assert m.parameter_names() == ["beta1", "beta2"]

    init = m.suggested_init()
    nll = m.nll(init)
    g = m.grad_nll(init)
    assert math.isfinite(nll)
    assert len(g) == 2
    assert all(math.isfinite(float(v)) for v in g)

    res = nextstat.fit(m)
    assert len(res.bestfit) == 2
    assert math.isfinite(res.nll)

    m_breslow = nextstat.CoxPhModel(times, events, x, ties="breslow")
    assert math.isfinite(m_breslow.nll(m_breslow.suggested_init()))

    # ties policy validation
    with pytest.raises(ValueError):
        nextstat.CoxPhModel(times, events, x, ties="invalid")


def test_cox_ph_matches_reference_nll_and_grad_breslow_and_efron() -> None:
    # Includes ties and censoring.
    times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
    events = [True, True, False, True, False, False]
    x = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [0.5, 0.5],
    ]
    beta = [0.1, -0.2]

    for ties in ("breslow", "efron"):
        m = nextstat.CoxPhModel(times, events, x, ties=ties)
        got_nll = float(m.nll(beta))
        ref_nll = float(_cox_nll_reference(times, events, x, beta, ties=ties))
        assert got_nll == pytest.approx(ref_nll, rel=1e-12, abs=1e-12)

        got_g = [float(v) for v in m.grad_nll(beta)]
        ref_g = _finite_diff_grad(lambda b: _cox_nll_reference(times, events, x, b, ties=ties), beta)
        assert got_g[0] == pytest.approx(ref_g[0], rel=5e-5, abs=5e-5)
        assert got_g[1] == pytest.approx(ref_g[1], rel=5e-5, abs=5e-5)


def test_cox_ph_is_invariant_to_row_permutation() -> None:
    times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
    events = [True, True, False, True, False, False]
    x = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [0.5, 0.5],
    ]
    beta = [0.1, -0.2]
    perm = [3, 0, 5, 2, 1, 4]
    times_p = [times[i] for i in perm]
    events_p = [events[i] for i in perm]
    x_p = [x[i] for i in perm]

    for ties in ("breslow", "efron"):
        m1 = nextstat.CoxPhModel(times, events, x, ties=ties)
        m2 = nextstat.CoxPhModel(times_p, events_p, x_p, ties=ties)
        assert float(m1.nll(beta)) == pytest.approx(float(m2.nll(beta)), rel=0.0, abs=0.0)
