from __future__ import annotations

import math


def scalar_filter_1d(y: list[float], *, f: float, q: float, h: float, r: float, m0: float, p0: float):
    m_pred = m0
    p_pred = p0
    ln_2pi = math.log(2.0 * math.pi)

    m_filt: list[float] = []
    p_filt: list[float] = []
    ll = 0.0

    for yt in y:
        v = yt - h * m_pred
        s = h * h * p_pred + r
        k = (p_pred * h) / s
        m = m_pred + k * v
        p = (1.0 - k * h) * p_pred * (1.0 - k * h) + k * r * k

        ll += -0.5 * (ln_2pi + math.log(s) + (v * v) / s)

        m_filt.append(m)
        p_filt.append(p)

        m_pred = f * m
        p_pred = f * f * p + q

    return m_filt, p_filt, ll


def test_kalman_filter_matches_scalar_reference():
    import nextstat

    f = 1.0
    q = 0.1
    h = 1.0
    r = 0.2
    m0 = 0.0
    p0 = 1.0

    y = [0.9, 1.2, 0.8, 1.1]
    m_ref, p_ref, ll_ref = scalar_filter_1d(y, f=f, q=q, h=h, r=r, m0=m0, p0=p0)

    model = nextstat.KalmanModel([[f]], [[q]], [[h]], [[r]], [m0], [[p0]])
    out = nextstat.timeseries.kalman_filter(model, [[v] for v in y])

    assert "log_likelihood" in out
    assert len(out["filtered_means"]) == len(y)
    assert len(out["filtered_covs"]) == len(y)

    for t in range(len(y)):
        m = float(out["filtered_means"][t][0])
        p = float(out["filtered_covs"][t][0][0])
        assert abs(m - m_ref[t]) <= 1e-12
        assert abs(p - p_ref[t]) <= 1e-12

    ll = float(out["log_likelihood"])
    assert abs(ll - ll_ref) <= 1e-12


def test_kalman_smooth_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    y = [[0.9], [1.2], [0.8], [1.1]]
    out = nextstat.timeseries.kalman_smooth(model, y)

    assert len(out["filtered_means"]) == len(y)
    assert len(out["filtered_covs"]) == len(y)
    assert len(out["smoothed_means"]) == len(y)
    assert len(out["smoothed_covs"]) == len(y)

    for t in range(len(y)):
        assert math.isfinite(float(out["smoothed_means"][t][0]))
        assert math.isfinite(float(out["smoothed_covs"][t][0][0]))


def test_kalman_em_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.5]], [[1.0]], [[0.5]], [0.0], [[1.0]])
    ys = [[0.9], [1.2], [0.8], [1.1], [0.95], [1.05]]
    out = nextstat.timeseries.kalman_em(model, ys, max_iter=5, tol=1e-9)

    assert "loglik_trace" in out
    assert len(out["loglik_trace"]) >= 2
    assert float(out["q"][0][0]) > 0.0
    assert float(out["r"][0][0]) > 0.0


def test_kalman_forecast_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    ys = [[0.9], [1.2], [0.8], [1.1]]
    out = nextstat.timeseries.kalman_forecast(model, ys, steps=3)

    assert len(out["state_means"]) == 3
    assert len(out["state_covs"]) == 3
    assert len(out["obs_means"]) == 3
    assert len(out["obs_covs"]) == 3


def test_kalman_simulate_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    out = nextstat.timeseries.kalman_simulate(model, t_max=5, seed=123)
    assert len(out["xs"]) == 5
    assert len(out["ys"]) == 5


def test_kalman_filter_allows_missing_none():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    ys = [[0.9], [None], [0.8], [1.1]]
    out = nextstat.timeseries.kalman_filter(model, ys)
    assert float(out["log_likelihood"]) == float(out["log_likelihood"])
