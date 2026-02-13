import math

import nextstat


def test_garch11_fit_smoke():
    ys = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02]
    out = nextstat.timeseries.garch11_fit(ys, max_iter=200, tol=1e-6)
    assert "params" in out
    assert "conditional_variance" in out
    assert "log_likelihood" in out
    assert len(out["conditional_variance"]) == len(ys)
    p = out["params"]
    assert p["omega"] > 0.0
    assert p["alpha"] >= 0.0
    assert p["beta"] >= 0.0
    assert p["alpha"] + p["beta"] < 1.0
    assert math.isfinite(float(out["log_likelihood"]))


def test_sv_logchi2_fit_smoke():
    ys = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02]
    out = nextstat.timeseries.sv_logchi2_fit(ys, max_iter=200, tol=1e-6)
    assert "params" in out
    assert "smoothed_h" in out
    assert "smoothed_sigma" in out
    assert len(out["smoothed_h"]) == len(ys)
    assert len(out["smoothed_sigma"]) == len(ys)
    p = out["params"]
    assert abs(p["phi"]) < 1.0
    assert p["sigma"] > 0.0
    assert math.isfinite(float(out["log_likelihood"]))

