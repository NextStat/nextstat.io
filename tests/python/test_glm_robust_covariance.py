from __future__ import annotations

import math

import pytest

import nextstat


def _is_finite_matrix(a):
    return all(math.isfinite(float(v)) for row in a for v in row)


def test_ols_hc_covariance_shapes_and_symmetry():
    x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [0.0, 1.0, 2.0, 4.0, 8.0]  # heteroskedastic-ish

    fit = nextstat.glm.linear.fit(x, y, include_intercept=True)
    cov = nextstat.glm.robust.ols_hc_covariance(x, y, fit.coef, include_intercept=True, kind="hc1")

    assert len(cov) == 2
    assert len(cov[0]) == 2
    assert _is_finite_matrix(cov)
    assert abs(float(cov[0][1]) - float(cov[1][0])) <= 1e-12


def test_ols_cluster_covariance_smoke_changes_with_clusters():
    x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    fit = nextstat.glm.linear.fit(x, y, include_intercept=True)

    cov_hc = nextstat.glm.robust.ols_hc_covariance(x, y, fit.coef, include_intercept=True, kind="hc1")
    cov_cl = nextstat.glm.robust.ols_cluster_covariance(x, y, fit.coef, cluster=["a", "a", "a", "b", "b", "b"])

    # Both symmetric, but should not be identical in general.
    assert abs(float(cov_cl[0][1]) - float(cov_cl[1][0])) <= 1e-12
    assert abs(float(cov_hc[0][0]) - float(cov_cl[0][0])) > 0.0


def test_glm_sandwich_covariance_logistic_smoke():
    x = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    y = [0, 0, 0, 1, 1]
    fit = nextstat.glm.logistic.fit(x, y, include_intercept=True, l2=1.0)
    cov = nextstat.glm.robust.logistic_sandwich_covariance(x, y, fit.coef, include_intercept=True)
    assert len(cov) == 2
    assert len(cov[0]) == 2
    assert _is_finite_matrix(cov)


def test_glm_sandwich_covariance_poisson_smoke_with_offset():
    x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1, 1, 2, 3, 6]
    offset = [0.0, 0.1, -0.2, 0.0, 0.3]
    fit = nextstat.glm.poisson.fit(x, y, include_intercept=True, offset=offset, l2=1.0)
    cov = nextstat.glm.robust.poisson_sandwich_covariance(
        x, y, fit.coef, include_intercept=True, offset=offset
    )
    assert len(cov) == 2
    assert len(cov[0]) == 2
    assert _is_finite_matrix(cov)


def test_glm_sandwich_covariance_negbin_smoke():
    x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [1, 2, 1, 3, 2, 6]
    fit = nextstat.glm.negbin.fit(x, y, include_intercept=True)
    cov = nextstat.glm.robust.negbin_sandwich_covariance(
        x, y, fit.coef, alpha=float(fit.alpha), include_intercept=True
    )
    assert len(cov) == 2
    assert len(cov[0]) == 2
    assert _is_finite_matrix(cov)


def test_standard_errors_from_cov():
    cov = [[4.0, 0.0], [0.0, 9.0]]
    se = nextstat.glm.robust.standard_errors(cov)
    assert se == [2.0, 3.0]

