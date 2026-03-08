import json
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "variance_reduction"
FIXTURE_NAMES = [
    "cuped_binary",
    "cure_revenue",
    "cure_ratio_style",
    "cure_low_conversion",
    "cure_multi_channel",
    "cure_collinear_ridge",
]


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _assert_close(label: str, got: float, want: float, tol: float = 1e-9) -> None:
    diff = abs(float(got) - float(want))
    assert diff <= tol, f"{label}: got {got}, want {want}, diff {diff}, tol {tol}"


def _assert_optional_close(label: str, got, want, tol: float = 1e-9) -> None:
    if got is None and want is None:
        return
    assert got is not None and want is not None, f"{label}: got {got!r}, want {want!r}"
    _assert_close(label, float(got), float(want), tol=tol)


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_ads_variance_reduction_public_surface_matches_committed_fixtures(fixture_name: str):
    import nextstat

    fixture = _load_fixture(fixture_name)
    expected = fixture["expected"]

    if fixture["method"] == "cuped":
        result = nextstat.ads.cuped_adjust(
            fixture["control_outcomes"],
            [row[0] for row in fixture["control_covariates"]],
            fixture["variant_outcomes"],
            [row[0] for row in fixture["variant_covariates"]],
            covariate_name=fixture["selected_covariates"][0],
            covariate_provenance=fixture["covariate_provenance"][0],
            pre_treatment_only=fixture["pre_treatment_only"],
        )
    else:
        result = nextstat.ads.cure_adjust(
            fixture["control_outcomes"],
            fixture["control_covariates"],
            fixture["variant_outcomes"],
            fixture["variant_covariates"],
            covariate_names=fixture["selected_covariates"],
            covariate_provenance=fixture["covariate_provenance"],
            pre_treatment_only=fixture["pre_treatment_only"],
        )

    assert result["method"] == expected["method"]
    assert result["selected_covariates"] == fixture["selected_covariates"]
    assert result["covariate_provenance"] == fixture["covariate_provenance"]
    assert result["provenance_validated"] is True
    assert result["pre_treatment_only"] is True
    assert result["solver"] == expected["solver"]
    assert result["regression_rank"] == expected["regression_rank"]

    _assert_close("adjusted_mean_control", result["adjusted_mean_control"], expected["adjusted_mean_control"])
    _assert_close("adjusted_mean_variant", result["adjusted_mean_variant"], expected["adjusted_mean_variant"])
    _assert_close("effect", result["effect"], expected["effect"])
    _assert_close("r_squared", result["r_squared"], expected["r_squared"])
    _assert_close(
        "variance_reduction_factor",
        result["variance_reduction_factor"],
        expected["variance_reduction_factor"],
    )
    _assert_close(
        "effective_sample_multiplier",
        result["effective_sample_multiplier"],
        expected["effective_sample_multiplier"],
    )
    _assert_optional_close("condition_number", result["condition_number"], expected["condition_number"])
    _assert_optional_close("ridge_lambda", result["ridge_lambda"], expected["ridge_lambda"], tol=1e-12)

    if fixture["method"] == "cuped":
        _assert_close("theta[0]", result["theta"], expected["theta"][0])
        _assert_close("rho", result["rho"], expected["rho"])
    else:
        assert len(result["theta"]) == len(expected["theta"])
        for idx, (got, want) in enumerate(zip(result["theta"], expected["theta"])):
            _assert_close(f"theta[{idx}]", got, want)


def test_ads_variance_reduction_rejects_leaky_provenance():
    import nextstat

    with pytest.raises(ValueError, match="post-treatment|leakage-prone"):
        nextstat.ads.cuped_adjust(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [1.0, 2.0, 3.0],
            covariate_provenance={
                "name": "post_clicks",
                "timing": "post_treatment",
                "source_dataset": "experiment_daily_exports",
            },
        )
