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


def _statsmodels_kalman_ar1(ys: list[float], *, phi: float, q: float, r: float, m0: float, p0: float):
    # Optional parity check: only runs when statsmodels is installed.
    pytest.importorskip("statsmodels")

    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    endog = np.asarray(ys, dtype=float).reshape(-1, 1)
    kf = KalmanFilter(k_endog=1, k_states=1)
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

    _set("transition", np.asarray([[phi]], dtype=float))
    _set("state_cov", np.asarray([[q]], dtype=float))
    _set("design", np.asarray([[1.0]], dtype=float))
    _set("obs_cov", np.asarray([[r]], dtype=float))

    init_state = np.asarray([m0], dtype=float)
    init_cov = np.asarray([[p0]], dtype=float)
    if hasattr(kf, "initialize_known"):
        kf.initialize_known(init_state, init_cov)
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

    ll_sm, m_sm, p_sm = _statsmodels_kalman_ar1(ys, phi=phi, q=q, r=r, m0=m0, p0=p0)

    assert abs(ll_ns - ll_sm) <= 1e-8

    # filtered_state: (k_states, nobs) -> compare to our per-timestep (nobs, k_states).
    assert m_sm.shape == (1, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_means"][t][0]) - float(m_sm[0, t])) <= 1e-8

    # filtered_state_cov: (k_states, k_states, nobs)
    assert p_sm.shape == (1, 1, len(ys))
    for t in range(len(ys)):
        assert abs(float(out["filtered_covs"][t][0][0]) - float(p_sm[0, 0, t])) <= 1e-8

