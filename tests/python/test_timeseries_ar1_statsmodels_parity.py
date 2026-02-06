"""Phase 8: AR(1) Kalman parity vs statsmodels (optional).

This is a *reference parity* test:
- It is skipped unless `statsmodels` is installed.
- It validates log-likelihood and filtered/smoothed state moments for a simple 1D AR(1)
  state-space model with observation noise.

Model:
  x_t = phi * x_{t-1} + w_t,  w_t ~ N(0, q)
  y_t = 1.0 * x_t     + v_t,  v_t ~ N(0, r)
  x_0 ~ N(m0, p0)
"""

from __future__ import annotations

import math

import pytest

import nextstat


statsmodels = pytest.importorskip("statsmodels")


def _statsmodels_ar1_filter_smooth(
    *,
    ys: list[float],
    phi: float,
    q: float,
    r: float,
    m0: float,
    p0: float,
) -> dict:
    import numpy as np
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

    y = np.asarray(ys, dtype=float)
    if y.ndim != 1:
        raise AssertionError("ys must be 1D")

    # Note: In statsmodels 0.14+, passing `initialization="known"` requires providing the
    # initial state in the constructor kwargs. To stay version-tolerant, construct first
    # and then call `initialize_known(...)`.
    kf = KalmanFilter(k_endog=1, k_states=1)
    # Some statsmodels versions require bind() before setting matrices; others don't.
    try:
        kf.bind(y[:, None])
    except Exception:
        kf.bind(y)

    kf.design = np.asarray([[1.0]], dtype=float)  # Z
    kf.transition = np.asarray([[float(phi)]], dtype=float)  # T
    kf.selection = np.asarray([[1.0]], dtype=float)  # R
    kf.state_cov = np.asarray([[float(q)]], dtype=float)  # Q
    kf.obs_cov = np.asarray([[float(r)]], dtype=float)  # H
    kf.state_intercept = np.asarray([[0.0]], dtype=float)
    kf.obs_intercept = np.asarray([[0.0]], dtype=float)

    # Known initialization: x0 ~ N(m0, p0).
    # API differs slightly across versions.
    if hasattr(kf, "initialize_known"):
        kf.initialize_known(np.asarray([float(m0)]), np.asarray([[float(p0)]]))
    else:  # pragma: no cover
        # Fallback: older API via initialization object.
        init = getattr(kf, "initialization", None)
        if init is None:
            raise AssertionError("statsmodels KalmanFilter missing initialize_known and initialization")
        init.initialization_type = "known"
        init.constant = np.asarray([float(m0)])
        init.stationary_cov = np.asarray([[float(p0)]])

    fr = kf.filter()

    # Filtered moments.
    fs = getattr(fr, "filtered_state", None)
    fP = getattr(fr, "filtered_state_cov", None)
    if fs is None or fP is None:
        raise AssertionError("statsmodels filter result missing filtered_state/filtered_state_cov")

    n = int(y.shape[0])
    filtered_means = [[float(fs[0, t])] for t in range(n)]
    filtered_covs = [[[float(fP[0, 0, t])]] for t in range(n)]

    # Predicted moments (one-step-ahead). Some versions include n+1, some n.
    ps = getattr(fr, "predicted_state", None)
    pP = getattr(fr, "predicted_state_cov", None)
    predicted_means = None
    predicted_covs = None
    if ps is not None and pP is not None:
        cols = ps.shape[1]
        use = n if cols == n else min(n, cols)
        predicted_means = [[float(ps[0, t])] for t in range(use)]
        predicted_covs = [[[float(pP[0, 0, t])]] for t in range(use)]

    # Smoother (statsmodels exposes smoothing via KalmanSmoother, not KalmanFilter).
    ks = KalmanSmoother(k_endog=1, k_states=1)
    ks.bind(y[:, None])
    ks.design = np.asarray([[1.0]], dtype=float)
    ks.transition = np.asarray([[float(phi)]], dtype=float)
    ks.selection = np.asarray([[1.0]], dtype=float)
    ks.state_cov = np.asarray([[float(q)]], dtype=float)
    ks.obs_cov = np.asarray([[float(r)]], dtype=float)
    ks.state_intercept = np.asarray([[0.0]], dtype=float)
    ks.obs_intercept = np.asarray([[0.0]], dtype=float)
    if hasattr(ks, "initialize_known"):
        ks.initialize_known(np.asarray([float(m0)]), np.asarray([[float(p0)]]))
    sr = ks.smooth()

    ss = getattr(sr, "smoothed_state", None)
    sP = getattr(sr, "smoothed_state_cov", None)
    if ss is None or sP is None:
        raise AssertionError("statsmodels smoother result missing smoothed_state/smoothed_state_cov")
    smoothed_means = [[float(ss[0, t])] for t in range(n)]
    smoothed_covs = [[[float(sP[0, 0, t])]] for t in range(n)]

    # Log-likelihood.
    ll = None
    for key in ("llf", "loglike"):
        v = getattr(fr, key, None)
        if v is None:
            continue
        try:
            ll = float(v() if callable(v) else v)
            break
        except Exception:
            ll = None
    if ll is None:
        llobs = getattr(fr, "llobs", None)
        if llobs is None:
            raise AssertionError("statsmodels filter result missing llf/loglike/llobs")
        ll = float(np.sum(llobs))

    return {
        "log_likelihood": float(ll),
        "predicted_means": predicted_means,
        "predicted_covs": predicted_covs,
        "filtered_means": filtered_means,
        "filtered_covs": filtered_covs,
        "smoothed_means": smoothed_means,
        "smoothed_covs": smoothed_covs,
    }


def _deterministic_ar1_observations(*, n: int, phi: float) -> list[float]:
    # Deterministic "pseudo-data" to avoid RNG / seeding issues in reference comparisons.
    x = 0.1
    ys: list[float] = []
    for t in range(int(n)):
        if t > 0:
            x = float(phi) * x + 0.05 * math.sin(float(t))
        y = x + 0.1 * math.cos(float(t))
        ys.append(float(y))
    return ys


def _flatten_1d_means(xs) -> list[float]:
    # nextstat returns [[m_t]] for 1D, statsmodels reference does the same.
    return [float(v[0]) for v in xs]


def _flatten_1x1_covs(ps) -> list[float]:
    # nextstat returns [[[p_t]]] for 1D.
    return [float(v[0][0]) for v in ps]


def test_ar1_kalman_filter_parity_vs_statsmodels():
    phi = 0.8
    q = 0.05
    r = 0.2
    m0 = 0.0
    p0 = 1.0
    ys = _deterministic_ar1_observations(n=50, phi=phi)

    model = nextstat.timeseries.ar1_model(phi=phi, q=q, r=r, m0=m0, p0=p0)
    ns = nextstat.timeseries.kalman_filter(model, [[y] for y in ys])
    sm = _statsmodels_ar1_filter_smooth(ys=ys, phi=phi, q=q, r=r, m0=m0, p0=p0)

    assert float(ns["log_likelihood"]) == pytest.approx(float(sm["log_likelihood"]), rel=1e-6, abs=1e-6)

    assert _flatten_1d_means(ns["filtered_means"]) == pytest.approx(
        _flatten_1d_means(sm["filtered_means"]), rel=0.0, abs=1e-6
    )
    assert _flatten_1x1_covs(ns["filtered_covs"]) == pytest.approx(
        _flatten_1x1_covs(sm["filtered_covs"]), rel=0.0, abs=1e-6
    )

    if sm["predicted_means"] is not None and sm["predicted_covs"] is not None:
        # Some statsmodels versions expose (n+1) predictions; compare only overlapping prefix.
        n_pred = min(len(ns["predicted_means"]), len(sm["predicted_means"]))
        assert _flatten_1d_means(ns["predicted_means"][:n_pred]) == pytest.approx(
            _flatten_1d_means(sm["predicted_means"][:n_pred]), rel=0.0, abs=1e-6
        )
        assert _flatten_1x1_covs(ns["predicted_covs"][:n_pred]) == pytest.approx(
            _flatten_1x1_covs(sm["predicted_covs"][:n_pred]), rel=0.0, abs=1e-6
        )


def test_ar1_kalman_smoother_parity_vs_statsmodels():
    phi = 0.6
    q = 0.02
    r = 0.15
    m0 = 0.0
    p0 = 1.0
    ys = _deterministic_ar1_observations(n=60, phi=phi)

    model = nextstat.timeseries.ar1_model(phi=phi, q=q, r=r, m0=m0, p0=p0)
    ns = nextstat.timeseries.kalman_smooth(model, [[y] for y in ys])
    sm = _statsmodels_ar1_filter_smooth(ys=ys, phi=phi, q=q, r=r, m0=m0, p0=p0)

    assert float(ns["log_likelihood"]) == pytest.approx(float(sm["log_likelihood"]), rel=1e-6, abs=1e-6)

    # Smoothing parity is optional if the reference API doesn't provide smoothed outputs.
    if sm["smoothed_means"] is None or sm["smoothed_covs"] is None:
        pytest.skip("statsmodels smoothing API not available for this version/config")

    assert _flatten_1d_means(ns["smoothed_means"]) == pytest.approx(
        _flatten_1d_means(sm["smoothed_means"]), rel=0.0, abs=1e-6
    )
    assert _flatten_1x1_covs(ns["smoothed_covs"]) == pytest.approx(
        _flatten_1x1_covs(sm["smoothed_covs"]), rel=0.0, abs=1e-6
    )
