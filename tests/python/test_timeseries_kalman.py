from __future__ import annotations

import math


def scalar_filter_1d(
    y: list[float | None], *, f: float, q: float, h: float, r: float, m0: float, p0: float
):
    m_pred = m0
    p_pred = p0
    ln_2pi = math.log(2.0 * math.pi)

    m_filt: list[float] = []
    p_filt: list[float] = []
    ll = 0.0

    for yt in y:
        if yt is None:
            # Missing observations: skip update + no likelihood contribution.
            m_filt.append(m_pred)
            p_filt.append(p_pred)
            m_pred = f * m_pred
            p_pred = f * f * p_pred + q
            continue

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


def test_kalman_em_allows_missing_none():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.5]], [[1.0]], [[0.5]], [0.0], [[1.0]])
    ys = [[0.9], [None], [0.8], [None], [0.95], [1.05]]
    out = nextstat.timeseries.kalman_em(model, ys, max_iter=5, tol=1e-9)

    assert "loglik_trace" in out
    assert len(out["loglik_trace"]) >= 2
    assert float(out["q"][0][0]) > 0.0
    assert float(out["r"][0][0]) > 0.0


def test_kalman_fit_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.5]], [[1.0]], [[0.5]], [0.0], [[1.0]])
    ys = [[0.9], [1.2], [0.8], [1.1], [0.95], [1.05]]
    out = nextstat.timeseries.kalman_fit(model, ys, max_iter=5, tol=1e-9, forecast_steps=2)

    assert "model" in out
    assert "em" in out
    assert "smooth" in out
    assert "forecast" in out

    assert len(out["em"]["loglik_trace"]) >= 2
    assert len(out["smooth"]["smoothed_means"]) == len(ys)
    assert len(out["forecast"]["obs_means"]) == 2


def test_kalman_forecast_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    ys = [[0.9], [1.2], [0.8], [1.1]]
    out = nextstat.timeseries.kalman_forecast(model, ys, steps=3)

    assert len(out["state_means"]) == 3
    assert len(out["state_covs"]) == 3
    assert len(out["obs_means"]) == 3
    assert len(out["obs_covs"]) == 3


def test_kalman_forecast_intervals_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    ys = [[0.9], [1.2], [0.8], [1.1]]
    out = nextstat.timeseries.kalman_forecast(model, ys, steps=3, alpha=0.05)

    assert len(out["obs_lower"]) == 3
    assert len(out["obs_upper"]) == 3
    assert float(out["alpha"]) == 0.05
    assert math.isfinite(float(out["z"]))


def test_kalman_simulate_shapes_smoke():
    import nextstat

    model = nextstat.KalmanModel([[1.0]], [[0.1]], [[1.0]], [[0.2]], [0.0], [[1.0]])
    out = nextstat.timeseries.kalman_simulate(model, t_max=5, seed=123)
    assert len(out["xs"]) == 5
    assert len(out["ys"]) == 5


def test_kalman_filter_allows_missing_none():
    import nextstat

    f = 1.0
    q = 0.1
    h = 1.0
    r = 0.2
    m0 = 0.0
    p0 = 1.0

    y = [0.9, None, 0.8, 1.1]
    m_ref, p_ref, ll_ref = scalar_filter_1d(y, f=f, q=q, h=h, r=r, m0=m0, p0=p0)

    model = nextstat.KalmanModel([[f]], [[q]], [[h]], [[r]], [m0], [[p0]])
    ys = [[v] for v in y]
    out = nextstat.timeseries.kalman_filter(model, ys)

    assert abs(float(out["log_likelihood"]) - ll_ref) <= 1e-12
    for t in range(len(y)):
        assert abs(float(out["filtered_means"][t][0]) - m_ref[t]) <= 1e-12
        assert abs(float(out["filtered_covs"][t][0][0]) - p_ref[t]) <= 1e-12


def test_local_level_builder_smoke():
    import nextstat

    m = nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)
    assert int(m.n_state()) == 1
    assert int(m.n_obs()) == 1


def test_local_linear_trend_builder_smoke():
    import nextstat

    m = nextstat.timeseries.local_linear_trend_model(q_level=0.1, q_slope=0.05, r=0.2)
    assert int(m.n_state()) == 2
    assert int(m.n_obs()) == 1


def test_ar1_builder_smoke():
    import nextstat

    m = nextstat.timeseries.ar1_model(phi=0.9, q=0.1, r=0.2)
    assert int(m.n_state()) == 1
    assert int(m.n_obs()) == 1


def test_ar1_builder_filter_matches_scalar_reference():
    import nextstat

    phi = 0.9
    q = 0.1
    r = 0.2
    m0 = 0.0
    p0 = 1.0

    y = [0.9, 1.2, 0.8, 1.1]
    m_ref, p_ref, ll_ref = scalar_filter_1d(y, f=phi, q=q, h=1.0, r=r, m0=m0, p0=p0)

    model = nextstat.timeseries.ar1_model(phi=phi, q=q, r=r, m0=m0, p0=p0)
    out = nextstat.timeseries.kalman_filter(model, [[v] for v in y])

    assert abs(float(out["log_likelihood"]) - ll_ref) <= 1e-12
    for t in range(len(y)):
        assert abs(float(out["filtered_means"][t][0]) - m_ref[t]) <= 1e-12
        assert abs(float(out["filtered_covs"][t][0][0]) - p_ref[t]) <= 1e-12


def test_kalman_filter_partial_missing_multivariate_decoupled_matches_scalar_refs():
    import nextstat

    # 2D fully decoupled model.
    model = nextstat.KalmanModel(
        [[1.0, 0.0], [0.0, 1.0]],  # F
        [[0.1, 0.0], [0.0, 0.2]],  # Q
        [[1.0, 0.0], [0.0, 1.0]],  # H
        [[0.3, 0.0], [0.0, 0.4]],  # R
        [0.0, 0.0],  # m0
        [[1.0, 0.0], [0.0, 1.0]],  # p0
    )

    y0 = [0.9, 1.0, 0.8, None]
    y1 = [1.1, None, 0.95, 1.05]
    ys = [[y0[t], y1[t]] for t in range(len(y0))]

    out = nextstat.timeseries.kalman_filter(model, ys)

    m0_ref, p0_ref, ll0_ref = scalar_filter_1d(y0, f=1.0, q=0.1, h=1.0, r=0.3, m0=0.0, p0=1.0)
    m1_ref, p1_ref, ll1_ref = scalar_filter_1d(y1, f=1.0, q=0.2, h=1.0, r=0.4, m0=0.0, p0=1.0)

    assert abs(float(out["log_likelihood"]) - (ll0_ref + ll1_ref)) <= 1e-12

    for t in range(len(y0)):
        assert abs(float(out["filtered_means"][t][0]) - m0_ref[t]) <= 1e-12
        assert abs(float(out["filtered_covs"][t][0][0]) - p0_ref[t]) <= 1e-12
        assert abs(float(out["filtered_means"][t][1]) - m1_ref[t]) <= 1e-12
        assert abs(float(out["filtered_covs"][t][1][1]) - p1_ref[t]) <= 1e-12

        assert abs(float(out["filtered_covs"][t][0][1])) <= 1e-12
        assert abs(float(out["filtered_covs"][t][1][0])) <= 1e-12


def test_local_level_seasonal_builder_smoke():
    import nextstat

    m = nextstat.timeseries.local_level_seasonal_model(period=4, q_level=0.1, q_season=0.2, r=0.3)
    assert int(m.n_state()) == 4
    assert int(m.n_obs()) == 1


def test_local_linear_trend_seasonal_builder_smoke():
    import nextstat

    m = nextstat.timeseries.local_linear_trend_seasonal_model(
        period=4, q_level=0.1, q_slope=0.05, q_season=0.2, r=0.3
    )
    assert int(m.n_state()) == 5
    assert int(m.n_obs()) == 1
