from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _load_ar1_fixture():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "kalman_ar1.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    params = data["ar1"]
    ys = [float(row[0]) for row in data["ys"]]
    return ys, params


def _load_local_level_fixture():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "kalman_local_level.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    params = data["local_level"]
    ys = [float(row[0]) for row in data["ys"]]
    return ys, params


def _load_local_linear_trend_fixture():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "kalman_local_linear_trend.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    params = data["local_linear_trend"]
    ys = [float(row[0]) for row in data["ys"]]
    return ys, params


def _statsmodels_kalman_filter(
    ys: list[float],
    *,
    transition: np.ndarray,
    state_cov: np.ndarray,
    design: np.ndarray,
    obs_cov: np.ndarray,
    init_state: np.ndarray,
    init_cov: np.ndarray,
):
    # Optional parity check: only runs when statsmodels is installed.
    pytest.importorskip("statsmodels")

    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    endog = np.asarray(ys, dtype=float).reshape(-1, 1)
    k_states = int(transition.shape[0])
    kf = KalmanFilter(k_endog=1, k_states=k_states)
    kf.bind(endog)

    def _set(key: str, value: np.ndarray):
        # statsmodels has historically supported both attribute and dict-style access.
        try:
            kf[key] = value
            return
        except Exception:
            pass
        try:
            setattr(kf, key, value)
            return
        except Exception:
            pass
        if hasattr(kf, "ssm"):
            kf.ssm[key] = value
            return
        raise AssertionError(f"Unable to set state-space matrix {key!r} on statsmodels KalmanFilter")

    _set("transition", np.asarray(transition, dtype=float))
    _set("state_cov", np.asarray(state_cov, dtype=float))
    _set("design", np.asarray(design, dtype=float))
    _set("obs_cov", np.asarray(obs_cov, dtype=float))

    if hasattr(kf, "initialize_known"):
        kf.initialize_known(np.asarray(init_state, dtype=float), np.asarray(init_cov, dtype=float))
    else:
        pytest.skip("statsmodels KalmanFilter does not support known initialization in this version")

    res = kf.filter()

    if hasattr(res, "llf"):
        ll = float(res.llf)
    elif hasattr(res, "loglike"):
        ll = float(res.loglike())
    else:
        raise AssertionError("statsmodels FilterResults missing llf/loglike")

    if not hasattr(res, "filtered_state") or not hasattr(res, "filtered_state_cov"):
        raise AssertionError("statsmodels FilterResults missing filtered_state / filtered_state_cov")

    return ll, np.asarray(res.filtered_state, dtype=float), np.asarray(res.filtered_state_cov, dtype=float)


def test_ar1_builder_filter_matches_statsmodels_optional():
    import nextstat

    ys, params = _load_ar1_fixture()
    phi = float(params["phi"])
    q = float(params["q"])
    r = float(params["r"])
    m0 = float(params["m0"])
    p0 = float(params["p0"])

    model = nextstat.timeseries.ar1_model(phi=phi, q=q, r=r, m0=m0, p0=p0)
    out = nextstat.timeseries.kalman_filter(model, [[v] for v in ys])
    ll_ns = float(out["log_likelihood"])

    ll_sm, m_sm, p_sm = _statsmodels_kalman_filter(
        ys,
        transition=np.asarray([[phi]], dtype=float),
        state_cov=np.asarray([[q]], dtype=float),
        design=np.asarray([[1.0]], dtype=float),
        obs_cov=np.asarray([[r]], dtype=float),
        init_state=np.asarray([m0], dtype=float),
        init_cov=np.asarray([[p0]], dtype=float),
    )

    assert abs(ll_ns - ll_sm) <= 1e-8

    # filtered_state: (k_states, nobs) -> compare to our per-timestep (nobs, k_states).
    assert m_sm.shape == (1, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_means"][t][0]) - float(m_sm[0, t])) <= 1e-8

    # filtered_state_cov: (k_states, k_states, nobs)
    assert p_sm.shape == (1, 1, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_covs"][t][0][0]) - float(p_sm[0, 0, t])) <= 1e-8


def test_local_level_builder_filter_matches_statsmodels_optional():
    import nextstat

    ys, params = _load_local_level_fixture()
    q = float(params["q"])
    r = float(params["r"])
    m0 = float(params.get("m0", 0.0))
    p0 = float(params.get("p0", 1.0))

    model = nextstat.timeseries.local_level_model(q=q, r=r, m0=m0, p0=p0)
    out = nextstat.timeseries.kalman_filter(model, [[v] for v in ys])
    ll_ns = float(out["log_likelihood"])

    ll_sm, m_sm, p_sm = _statsmodels_kalman_filter(
        ys,
        transition=np.asarray([[1.0]], dtype=float),
        state_cov=np.asarray([[q]], dtype=float),
        design=np.asarray([[1.0]], dtype=float),
        obs_cov=np.asarray([[r]], dtype=float),
        init_state=np.asarray([m0], dtype=float),
        init_cov=np.asarray([[p0]], dtype=float),
    )

    assert abs(ll_ns - ll_sm) <= 1e-8
    assert m_sm.shape == (1, len(ys))
    assert p_sm.shape == (1, 1, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_means"][t][0]) - float(m_sm[0, t])) <= 1e-8
        assert abs(float(out["filtered_covs"][t][0][0]) - float(p_sm[0, 0, t])) <= 1e-8


def test_local_linear_trend_builder_filter_matches_statsmodels_optional():
    import nextstat

    ys, params = _load_local_linear_trend_fixture()
    q_level = float(params["q_level"])
    q_slope = float(params["q_slope"])
    r = float(params["r"])

    # Fixture uses defaults in CLI/Rust builder: level0=0, slope0=0, p0_level=1, p0_slope=1.
    model = nextstat.timeseries.local_linear_trend_model(q_level=q_level, q_slope=q_slope, r=r)
    out = nextstat.timeseries.kalman_filter(model, [[v] for v in ys])
    ll_ns = float(out["log_likelihood"])

    ll_sm, m_sm, p_sm = _statsmodels_kalman_filter(
        ys,
        transition=np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=float),
        state_cov=np.asarray([[q_level, 0.0], [0.0, q_slope]], dtype=float),
        design=np.asarray([[1.0, 0.0]], dtype=float),
        obs_cov=np.asarray([[r]], dtype=float),
        init_state=np.asarray([0.0, 0.0], dtype=float),
        init_cov=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )

    assert abs(ll_ns - ll_sm) <= 1e-8
    assert m_sm.shape == (2, len(ys))
    assert p_sm.shape == (2, 2, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_means"][t][0]) - float(m_sm[0, t])) <= 1e-8
        assert abs(float(out["filtered_means"][t][1]) - float(m_sm[1, t])) <= 1e-8
        assert abs(float(out["filtered_covs"][t][0][0]) - float(p_sm[0, 0, t])) <= 1e-8
        assert abs(float(out["filtered_covs"][t][1][1]) - float(p_sm[1, 1, t])) <= 1e-8
