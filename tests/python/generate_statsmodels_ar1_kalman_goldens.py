"""Generate statsmodels-derived goldens for AR(1) Kalman filter/smoother parity.

Why:
- `tests/python/test_timeseries_ar1_statsmodels_parity.py` is a reference test and is skipped
  unless `statsmodels` is installed.
- These goldens allow running a deterministic regression test without requiring statsmodels at
  runtime (same pattern as other "golden" harnesses).

Usage:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
    tests/python/generate_statsmodels_ar1_kalman_goldens.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _deterministic_ar1_observations(*, n: int, phi: float) -> list[float]:
    x = 0.1
    ys: list[float] = []
    for t in range(int(n)):
        if t > 0:
            x = float(phi) * x + 0.05 * math.sin(float(t))
        y = x + 0.1 * math.cos(float(t))
        ys.append(float(y))
    return ys


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

    kf = KalmanFilter(k_endog=1, k_states=1)
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

    if hasattr(kf, "initialize_known"):
        kf.initialize_known(np.asarray([float(m0)]), np.asarray([[float(p0)]]))
    else:  # pragma: no cover
        init = getattr(kf, "initialization", None)
        if init is None:
            raise AssertionError(
                "statsmodels KalmanFilter missing initialize_known and initialization"
            )
        init.initialization_type = "known"
        init.constant = np.asarray([float(m0)])
        init.stationary_cov = np.asarray([[float(p0)]])

    fr = kf.filter()

    fs = getattr(fr, "filtered_state", None)
    fP = getattr(fr, "filtered_state_cov", None)
    if fs is None or fP is None:
        raise AssertionError("statsmodels filter result missing filtered_state/filtered_state_cov")

    n = int(y.shape[0])
    filtered_means = [[float(fs[0, t])] for t in range(n)]
    filtered_covs = [[[float(fP[0, 0, t])]] for t in range(n)]

    ps = getattr(fr, "predicted_state", None)
    pP = getattr(fr, "predicted_state_cov", None)
    predicted_means = None
    predicted_covs = None
    if ps is not None and pP is not None:
        cols = int(ps.shape[1])
        # Some versions expose n+1 predictions; normalize to exactly n for stable goldens.
        use = n if cols >= n else cols
        predicted_means = [[float(ps[0, t])] for t in range(use)]
        predicted_covs = [[[float(pP[0, 0, t])]] for t in range(use)]

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


def main() -> None:
    # Two fixed cases matching the reference parity tests.
    cases = []
    cases.append(
        {
            "name": "filter_phi0p8_q0p05_r0p2_n50",
            "phi": 0.8,
            "q": 0.05,
            "r": 0.2,
            "m0": 0.0,
            "p0": 1.0,
            "n": 50,
        }
    )
    cases.append(
        {
            "name": "smooth_phi0p6_q0p02_r0p15_n60",
            "phi": 0.6,
            "q": 0.02,
            "r": 0.15,
            "m0": 0.0,
            "p0": 1.0,
            "n": 60,
        }
    )

    out_cases = []
    for c in cases:
        ys = _deterministic_ar1_observations(n=int(c["n"]), phi=float(c["phi"]))
        ref = _statsmodels_ar1_filter_smooth(
            ys=ys,
            phi=float(c["phi"]),
            q=float(c["q"]),
            r=float(c["r"]),
            m0=float(c["m0"]),
            p0=float(c["p0"]),
        )
        out_cases.append({**c, "ys": ys, "ref": ref})

    out = {
        "version": 1,
        "source": "statsmodels",
        "cases": out_cases,
    }

    path = Path("tests/fixtures/statsmodels_ar1_kalman_goldens.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

