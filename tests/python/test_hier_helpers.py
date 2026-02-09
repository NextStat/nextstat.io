from __future__ import annotations


def test_hier_linear_random_slope_helper_smoke():
    import nextstat

    m = nextstat.hier.linear_random_slope(
        x=[[0.0], [1.0], [2.0], [3.0]],
        y=[0.0, 1.0, 2.0, 3.0],
        include_intercept=False,
        group_idx=[0, 0, 1, 1],
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
        random_slope_feature_idx=0,
        random_intercept_non_centered=True,
        random_slope_non_centered=True,
    )
    names = m.parameter_names()
    assert any(n.startswith("u_beta1_") or n.startswith("z_u_beta1_") for n in names)


def test_hier_logistic_correlated_intercept_slope_helper_smoke():
    import nextstat

    m = nextstat.hier.logistic_correlated_intercept_slope(
        x=[[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
        y=[0, 1, 0, 1, 0, 1],
        include_intercept=False,
        group_idx=[0, 0, 1, 1, 0, 1],
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
        correlated_feature_idx=0,
        lkj_eta=2.0,
    )
    names = m.parameter_names()
    assert "tau_alpha" in names
    assert "rho_alpha_u_beta1" in names

