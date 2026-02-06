from __future__ import annotations


def test_glm_spec_roundtrip_linear_no_groups():
    from nextstat.data import GlmSpec

    spec = GlmSpec.linear_regression(
        x=[[1.0, 0.5], [1.0, -0.3], [1.0, 1.2]],
        y=[0.1, -0.2, 0.25],
        include_intercept=False,
    )
    spec2 = GlmSpec.from_json(spec.to_json())
    assert spec2.kind == "linear"

    m = spec2.build()
    assert m.n_params() == 2
    nll = m.nll(m.suggested_init())
    assert nll == nll and nll >= 0.0


def test_glm_spec_build_logistic_with_groups_smoke():
    import nextstat
    from nextstat.data import GlmSpec

    spec = GlmSpec.logistic_regression(
        x=[[1.0, 0.5], [1.0, -0.3], [1.0, 1.2], [1.0, 0.1]],
        y=[0, 1, 1, 0],
        include_intercept=False,
        group_idx=[0, 1, 0, 1],
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=10.0,
    )

    m = spec.build()
    names = m.parameter_names()
    assert "mu_alpha" in names
    assert "sigma_alpha" in names

    raw = nextstat._core.sample(m, n_chains=1, n_warmup=5, n_samples=10, seed=3)
    assert "posterior" in raw
    assert "diagnostics" in raw

