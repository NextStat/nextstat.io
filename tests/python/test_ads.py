import pytest


def test_beta_binomial_counts_surface_smoke():
    import nextstat

    model = nextstat.ads.BetaBinomialModel.fit_from_counts([12, 19, 10, 24], [900, 1200, 800, 1100])

    assert nextstat.BetaBinomialModel is not None
    assert 0.0 < float(model.mean()) < 1.0
    assert 0.0 < float(model.overdispersion()) < 1.0
    assert float(model.alpha) > 0.0
    assert float(model.beta) > 0.0

    posterior = model.posterior(5, 20)
    assert float(posterior.alpha) > float(model.alpha)
    assert float(posterior.beta) > float(model.beta)


def test_delay_correction_surface_smoke():
    import nextstat

    model = nextstat.ads.DelayCorrectionModel.fit_from_lag_buckets(
        [(1.0, 296), (2.0, 193), (4.0, 252), (7.0, 173), (14.0, 80)]
    )

    assert nextstat.DelayCorrectionModel is not None
    assert float(model.lambda_) > 0.0
    assert float(model.observed_fraction(0.0)) == 0.0
    corrected, uncertainty = model.correct(42.0, 3.0)
    assert float(corrected) > 42.0
    assert float(uncertainty) > 0.0

    with pytest.raises(ValueError):
        model.correct(42.0, 0.0)


def test_ads_response_curve_helpers_smoke():
    import nextstat

    low = nextstat.ads.hill(10.0, 50.0, 1.2)
    high = nextstat.ads.hill(100.0, 50.0, 1.2)
    transformed = nextstat.ads.adstock_geometric([100.0, 0.0, 0.0], 0.5)

    assert high > low
    assert transformed == [100.0, 50.0, 25.0]


def test_cuped_adjust_surface_smoke():
    import nextstat

    result = nextstat.ads.cuped_adjust(
        [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
        [9.5, 11.0, 10.0, 12.0, 8.5, 13.0],
        [11.0, 13.0, 12.0, 14.0, 10.0, 15.0],
        [10.5, 12.0, 11.0, 13.0, 9.5, 14.0],
        covariate_name="pre_clicks",
    )

    assert result["method"] == "cuped"
    assert result["num_covariates"] == 1
    assert result["selected_covariates"] == ["pre_clicks"]
    assert 0.0 <= float(result["r_squared"]) < 1.0
    assert 0.0 < float(result["variance_reduction_factor"]) <= 1.0
    assert float(result["effective_sample_multiplier"]) >= 1.0
    assert result["solver"] in {"svd", "ridge"}
    assert result["pre_treatment_only"] is True


def test_cure_adjust_surface_smoke_and_guardrails():
    import nextstat

    control_covariates = [
        [100.0, 200.0],
        [120.0, 240.0],
        [110.0, 220.0],
        [130.0, 260.0],
        [90.0, 180.0],
        [140.0, 280.0],
    ]
    variant_covariates = [
        [102.0, 204.0],
        [122.0, 244.0],
        [112.0, 224.0],
        [132.0, 264.0],
        [92.0, 184.0],
        [142.0, 284.0],
    ]

    result = nextstat.ads.cure_adjust(
        [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
        control_covariates,
        [11.0, 13.0, 12.0, 14.0, 10.0, 15.0],
        variant_covariates,
        covariate_names=["pre_clicks", "pre_impressions"],
    )

    assert result["method"] == "cure"
    assert result["num_covariates"] == 2
    assert result["selected_covariates"] == ["pre_clicks", "pre_impressions"]
    assert len(result["theta"]) == 2
    assert result["solver"] == "ridge"
    assert result["ridge_lambda"] is not None
    assert float(result["effective_sample_multiplier"]) >= 1.0
    assert result["pre_treatment_only"] is True

    with pytest.raises(ValueError, match="pre-treatment"):
        nextstat.ads.cure_adjust(
            [10.0, 12.0, 11.0, 13.0, 9.0, 14.0],
            control_covariates,
            [11.0, 13.0, 12.0, 14.0, 10.0, 15.0],
            variant_covariates,
            pre_treatment_only=False,
        )
