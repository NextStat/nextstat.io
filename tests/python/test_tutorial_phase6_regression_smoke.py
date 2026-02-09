from __future__ import annotations

import nextstat


def test_tutorial_phase6_regression_smoke() -> None:
    # Linear (formula)
    x = [[float(i)] for i in range(1, 21)]
    y = [1.0 + 2.0 * xi[0] + 0.1 * ((i % 5) - 2) for i, xi in enumerate(x)]
    data = [{"y": yi, "x": xi[0]} for xi, yi in zip(x, y)]

    fit, names = nextstat.glm.linear.from_formula("y ~ 1 + x", data)
    assert names == ["Intercept", "x"]
    s = nextstat.summary.fit_summary(fit, names=names)
    assert len(s["coef"]) == 2

    # Logistic (regularized)
    xl = [[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]
    yl = [0, 0, 0, 0, 1, 1, 1]
    data_l = [{"y": yi, "x": xi[0]} for xi, yi in zip(xl, yl)]
    fit_l, names_l = nextstat.glm.logistic.from_formula("y ~ 1 + x", data_l, l2=1.0)
    assert names_l == ["Intercept", "x"]
    assert len(fit_l.coef) == 2

    # Poisson (exposure)
    xp = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    yp = [1, 2, 2, 5, 11]
    exposure = [1.0, 1.0, 2.0, 2.0, 4.0]
    data_p = [{"y": yi, "x": xi[0], "exposure": ei} for xi, yi, ei in zip(xp, yp, exposure)]
    fit_p, names_p = nextstat.glm.poisson.from_formula("y ~ 1 + x", data_p, exposure="exposure")
    assert names_p == ["Intercept", "x"]
    assert len(fit_p.coef) == 2

