from __future__ import annotations

import math
import random

import pytest

import nextstat


def _randn(rng: random.Random) -> float:
    # Box-Muller
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _make_iv_data(n: int, *, seed: int, pi: float, rho_uv: float) -> dict[str, list[float]]:
    rng = random.Random(seed)
    x: list[float] = []
    z: list[float] = []
    v: list[float] = []
    u: list[float] = []

    for _ in range(n):
        xi = _randn(rng)
        zi = _randn(rng)
        vi = _randn(rng)
        ei = _randn(rng)
        ui = rho_uv * vi + math.sqrt(max(0.0, 1.0 - rho_uv * rho_uv)) * ei
        x.append(xi)
        z.append(zi)
        v.append(vi)
        u.append(ui)

    d = [pi * zi + 0.5 * xi + vi for zi, xi, vi in zip(z, x, v)]
    y = [2.0 * di + 1.0 * xi + ui for di, xi, ui in zip(d, x, u)]
    return {"y": y, "d": d, "x": x, "z": z}


def test_iv_2sls_from_formula_recovers_coef():
    data = _make_iv_data(400, seed=0, pi=1.0, rho_uv=0.6)
    fit = nextstat.econometrics.iv_2sls_from_formula(
        "y ~ 1 + x",
        data,
        endog="d",
        instruments=["z"],
        cov="hc1",
    )
    # Coefficient order: endog first, then exog from formula.
    assert fit.column_names[0] == "d"
    assert abs(float(fit.coef[0]) - 2.0) < 0.15
    assert math.isfinite(float(fit.standard_errors[0]))
    assert math.isfinite(float(fit.diagnostics.first_stage_f[0]))


def test_iv_2sls_weak_instrument_has_low_first_stage_f():
    data = _make_iv_data(400, seed=1, pi=0.05, rho_uv=0.6)
    fit = nextstat.econometrics.iv_2sls_from_formula(
        "y ~ 1 + x",
        data,
        endog="d",
        instruments=["z"],
        cov="hc1",
    )
    # Classic rule-of-thumb: first-stage F < 10 indicates weak instruments.
    assert float(fit.diagnostics.first_stage_f[0]) < 10.0


def test_iv_2sls_cluster_smoke():
    data = _make_iv_data(200, seed=2, pi=1.0, rho_uv=0.6)
    # Two clusters.
    cluster = ["a" if i < 100 else "b" for i in range(200)]
    fit = nextstat.econometrics.iv_2sls_from_formula(
        "y ~ 1 + x",
        data,
        endog="d",
        instruments=["z"],
        cov="cluster",
        cluster=cluster,
    )
    assert math.isfinite(float(fit.standard_errors[0]))


def test_iv_2sls_underidentified_raises():
    data = _make_iv_data(50, seed=3, pi=1.0, rho_uv=0.0)
    with pytest.raises(ValueError, match="underidentified"):
        nextstat.econometrics.iv_2sls_from_formula(
            "y ~ 1 + x",
            data,
            endog=["d", "x"],  # 2 endogenous, only 1 instrument
            instruments=["z"],
        )

