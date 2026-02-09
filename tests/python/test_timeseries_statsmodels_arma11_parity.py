from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _load_arma11_fixture():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "kalman_arma11.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    params = data["arma11"]
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
    pytest.importorskip("statsmodels")

    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    endog = np.asarray(ys, dtype=float).reshape(-1, 1)
    k_states = int(transition.shape[0])
    kf = KalmanFilter(k_endog=1, k_states=k_states)
    kf.bind(endog)

    def _set(key: str, value: np.ndarray):
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
    _set("selection", np.eye(k_states, dtype=float))
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


def test_arma11_builder_smoke():
    import nextstat

    ys, params = _load_arma11_fixture()
    model = nextstat.timeseries.arma11_model(
        phi=float(params["phi"]),
        theta=float(params["theta"]),
        sigma2=float(params["sigma2"]),
        r=float(params["r"]),
        m0_x=float(params["m0_x"]),
        m0_eps=float(params["m0_eps"]),
        p0_x=float(params["p0_x"]),
        p0_eps=float(params["p0_eps"]),
    )

    out = nextstat.timeseries.kalman_filter(model, [[v] for v in ys])
    assert "log_likelihood" in out
    assert "filtered_means" in out
    assert "filtered_covs" in out

    ll = float(out["log_likelihood"])
    assert ll == ll  # NaN guard
    assert len(out["filtered_means"]) == len(ys)
    assert len(out["filtered_covs"]) == len(ys)
    assert len(out["filtered_means"][0]) == 2
    assert len(out["filtered_covs"][0]) == 2
    assert len(out["filtered_covs"][0][0]) == 2


def test_arma11_builder_filter_matches_statsmodels_optional(ns_timing):
    import nextstat

    ys, params = _load_arma11_fixture()
    phi = float(params["phi"])
    theta = float(params["theta"])
    sigma2 = float(params["sigma2"])
    r = float(params["r"])
    m0_x = float(params["m0_x"])
    m0_eps = float(params["m0_eps"])
    p0_x = float(params["p0_x"])
    p0_eps = float(params["p0_eps"])

    model = nextstat.timeseries.arma11_model(
        phi=phi,
        theta=theta,
        sigma2=sigma2,
        r=r,
        m0_x=m0_x,
        m0_eps=m0_eps,
        p0_x=p0_x,
        p0_eps=p0_eps,
    )
    with ns_timing.time("nextstat"):
        out = nextstat.timeseries.kalman_filter(model, [[v] for v in ys])
        ll_ns = float(out["log_likelihood"])

    with ns_timing.time("statsmodels"):
        ll_sm, m_sm, p_sm = _statsmodels_kalman_filter(
            ys,
            transition=np.asarray([[phi, theta], [0.0, 0.0]], dtype=float),
            state_cov=np.asarray([[sigma2, sigma2], [sigma2, sigma2]], dtype=float),
            design=np.asarray([[1.0, 0.0]], dtype=float),
            obs_cov=np.asarray([[r]], dtype=float),
            init_state=np.asarray([m0_x, m0_eps], dtype=float),
            init_cov=np.asarray([[p0_x, 0.0], [0.0, p0_eps]], dtype=float),
        )

    assert abs(ll_ns - ll_sm) <= 1e-8

    assert m_sm.shape == (2, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_means"][t][0]) - float(m_sm[0, t])) <= 1e-8
        assert abs(float(out["filtered_means"][t][1]) - float(m_sm[1, t])) <= 1e-8

    assert p_sm.shape == (2, 2, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_covs"][t][0][0]) - float(p_sm[0, 0, t])) <= 1e-8
        assert abs(float(out["filtered_covs"][t][1][1]) - float(p_sm[1, 1, t])) <= 1e-8
