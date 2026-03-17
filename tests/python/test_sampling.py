"""Tests for Python NUTS/HMC sampling surface (Sprint 3.2).

Requires: maturin develop (or pip install -e .) to build the _core extension.
"""

import json
import os
from pathlib import Path

import pytest

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_model():
    ws = load_fixture("simple_workspace.json")
    return nextstat.HistFactoryModel.from_workspace(json.dumps(ws))


def _make_fast_model():
    # Fast, non-HEP model for most API/shape/config tests (keeps non-slow suite quick).
    return nextstat.GaussianMeanModel([1.0, 2.0, 3.0, 4.0] * 5, sigma=1.0)


def _make_linear_regression_model():
    x = [[float(i), ((i * 7) % 5) * 0.25 - 0.5] for i in range(1, 25)]
    y = [0.4 + 1.2 * row[0] - 0.7 * row[1] for row in x]
    return nextstat.LinearRegressionModel(x, y, include_intercept=True)


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------

class TestSampleAPIShape:
    """Return value has correct top-level keys and nested dimensions."""

    def test_returns_dict_with_expected_keys(self):
        model = _make_fast_model()
        result = nextstat.sample(model, n_chains=1, n_warmup=20, n_samples=10, seed=1)
        assert isinstance(result, dict)
        expected_keys = {
            "posterior", "sample_stats", "diagnostics",
            "param_names", "n_chains", "n_warmup", "n_samples",
        }
        assert expected_keys <= set(result.keys())

    def test_posterior_shape(self):
        model = _make_fast_model()
        n_chains, n_samples = 2, 10
        result = nextstat.sample(
            model, n_chains=n_chains, n_warmup=20, n_samples=n_samples, seed=1,
        )
        posterior = result["posterior"]
        assert isinstance(posterior, dict)
        assert len(posterior) == model.n_params()
        for name in result["param_names"]:
            chains = posterior[name]
            assert len(chains) == n_chains
            for chain in chains:
                assert len(chain) == n_samples

    def test_sample_stats_shape(self):
        model = _make_fast_model()
        n_chains, n_samples = 1, 10
        result = nextstat.sample(
            model, n_chains=n_chains, n_warmup=20, n_samples=n_samples, seed=1,
        )
        stats = result["sample_stats"]
        assert "diverging" in stats
        assert "tree_depth" in stats
        assert "accept_prob" in stats
        assert "energy" in stats
        assert "step_size" in stats
        assert "n_leapfrog_warmup_total" in stats
        assert len(stats["diverging"]) == n_chains
        assert len(stats["diverging"][0]) == n_samples
        assert len(stats["energy"][0]) == n_samples
        assert len(stats["step_size"]) == n_chains
        assert len(stats["n_leapfrog_warmup_total"]) == n_chains
        assert stats["n_leapfrog_warmup_total"][0] > 0

    def test_diagnostics_keys(self):
        model = _make_fast_model()
        result = nextstat.sample(model, n_chains=2, n_warmup=20, n_samples=10, seed=1)
        diag = result["diagnostics"]
        assert "r_hat" in diag
        assert "ess_bulk" in diag
        assert "ess_tail" in diag
        assert "divergence_rate" in diag
        assert "max_treedepth_rate" in diag
        assert "ebfmi" in diag
        assert "quality" in diag
        q = diag["quality"]
        assert isinstance(q, dict)
        assert {"status", "enabled", "warnings", "failures"} <= set(q.keys())
        assert q["status"] in {"ok", "warn", "fail"}
        assert isinstance(q["warnings"], list)
        assert isinstance(q["failures"], list)
        # r_hat should be a dict keyed by param name
        for name in result["param_names"]:
            assert name in diag["r_hat"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestSampleReproducibility:
    """Same seed produces identical draws."""

    def test_same_seed_same_draws(self):
        model = _make_fast_model()
        kwargs = dict(n_chains=1, n_warmup=20, n_samples=10, seed=42)
        r1 = nextstat.sample(model, **kwargs)
        r2 = nextstat.sample(model, **kwargs)
        for name in r1["param_names"]:
            assert r1["posterior"][name] == r2["posterior"][name], (
                f"Draws for {name} should be identical with same seed"
            )


class TestWalnutsSurface:
    """Fast product-surface checks for the WALNUTS method."""

    def test_walnuts_returns_dict_with_expected_keys(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model, method="walnuts", n_chains=1, n_warmup=20, n_samples=10, seed=11
        )
        assert isinstance(result, dict)
        expected_keys = {
            "posterior", "sample_stats", "diagnostics",
            "param_names", "n_chains", "n_warmup", "n_samples",
        }
        assert expected_keys <= set(result.keys())

    def test_walnuts_same_seed_same_draws(self):
        model = _make_fast_model()
        kwargs = dict(
            method="walnuts",
            n_chains=1,
            n_warmup=20,
            n_samples=10,
            seed=42,
            max_step_halvings=2,
            min_micro_steps=2,
        )
        r1 = nextstat.sample(model, **kwargs)
        r2 = nextstat.sample(model, **kwargs)
        for name in r1["param_names"]:
            assert r1["posterior"][name] == r2["posterior"][name]

    def test_walnuts_custom_max_treedepth(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model,
            method="walnuts",
            n_chains=1,
            n_warmup=20,
            n_samples=10,
            seed=7,
            max_treedepth=5,
        )
        for depth in result["sample_stats"]["tree_depth"][0]:
            assert depth <= 5

    def test_walnuts_dense_metric_reports_dense_surface(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model,
            method="walnuts",
            n_chains=1,
            n_warmup=20,
            n_samples=10,
            seed=1,
            metric="dense",
        )
        stats = result["sample_stats"]
        assert stats["metric_type"] == "dense"
        assert "inv_mass_matrix" in stats
        assert len(stats["inv_mass_matrix"]) == model.n_params() ** 2

    def test_walnuts_auto_metric_reports_dense_for_small_dim(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model,
            method="walnuts",
            n_chains=1,
            n_warmup=20,
            n_samples=10,
            seed=2,
            metric="auto",
        )
        assert result["sample_stats"]["metric_type"] == "dense"

    def test_walnuts_min_micro_steps_zero_raises(self):
        model = _make_fast_model()
        with pytest.raises(ValueError, match="min_micro_steps must be >= 1"):
            nextstat.sample(
                model,
                method="walnuts",
                n_chains=1,
                n_warmup=10,
                n_samples=10,
                seed=1,
                min_micro_steps=0,
            )

    def test_walnuts_specific_config_smoke(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model,
            method="walnuts",
            n_chains=1,
            n_warmup=20,
            n_samples=10,
            seed=9,
            target_tree_depth=3.0,
            max_step_halvings=2,
            min_micro_steps=2,
            max_energy_error=1.0,
            init_jitter_rel=0.1,
        )
        assert isinstance(result, dict)
        assert result["n_samples"] == 10

    def test_walnuts_cuda_supported_linear_smoke(self):
        if not getattr(nextstat, "has_cuda", lambda: False)():
            pytest.skip("CUDA runtime not available")

        model = _make_linear_regression_model()
        result = nextstat.sample(
            model,
            method="walnuts",
            device="cuda",
            n_chains=1,
            n_warmup=12,
            n_samples=6,
            seed=17,
            metric="diagonal",
        )
        assert isinstance(result, dict)
        assert result["n_chains"] == 1
        assert result["sample_stats"]["metric_type"] == "diagonal"

    def test_walnuts_cuda_rejects_dense_metric(self):
        if not getattr(nextstat, "has_cuda", lambda: False)():
            pytest.skip("CUDA runtime not available")

        model = _make_linear_regression_model()
        with pytest.raises(ValueError, match="supports only metric='diagonal'"):
            nextstat.sample(
                model,
                method="walnuts",
                device="cuda",
                n_chains=1,
                n_warmup=10,
                n_samples=5,
                seed=19,
                metric="dense",
            )

    def test_walnuts_cuda_rejects_unsupported_model_family(self):
        if not getattr(nextstat, "has_cuda", lambda: False)():
            pytest.skip("CUDA runtime not available")

        model = _make_fast_model()
        with pytest.raises(ValueError, match="stable surface currently supports only"):
            nextstat.sample(
                model,
                method="walnuts",
                device="cuda",
                n_chains=1,
                n_warmup=10,
                n_samples=5,
                seed=23,
            )


# ---------------------------------------------------------------------------
# Non-slow CI gates
# ---------------------------------------------------------------------------

class TestSampleNonSlowGates:
    """Non-flaky, non-slow sanity checks for CI."""

    def test_basic_diagnostics_are_finite(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model,
            n_chains=2,
            n_warmup=30,
            n_samples=20,
            seed=123,
            init_jitter_rel=0.10,
        )
        diag = result["diagnostics"]

        assert 0.0 <= diag["divergence_rate"] <= 1.0
        assert 0.0 <= diag["max_treedepth_rate"] <= 1.0

        # R-hat/ESS can be imperfect for short runs, but must be finite.
        for v in diag["r_hat"].values():
            assert v == v and v > 0.0  # not NaN
        for v in diag["ess_bulk"].values():
            assert v == v and v > 0.0
        for v in diag["ess_tail"].values():
            assert v == v and v > 0.0

        # E-BFMI is per-chain; should be finite for real energy traces.
        for bfmi in diag["ebfmi"]:
            assert bfmi == bfmi and bfmi > 0.0


# ---------------------------------------------------------------------------
# Input validation (fast, deterministic)
# ---------------------------------------------------------------------------

class TestSampleInputValidation:
    def test_target_accept_out_of_range_raises(self):
        model = _make_fast_model()
        with pytest.raises(ValueError, match="target_accept must be finite and in"):
            nextstat.sample(model, n_chains=1, n_warmup=10, n_samples=10, seed=1, target_accept=1.0)

    def test_max_treedepth_zero_raises(self):
        model = _make_fast_model()
        with pytest.raises(ValueError, match="max_treedepth must be"):
            nextstat.sample(model, n_chains=1, n_warmup=10, n_samples=10, seed=1, max_treedepth=0)

    def test_negative_init_jitter_raises(self):
        model = _make_fast_model()
        with pytest.raises(ValueError, match="init_jitter must be"):
            nextstat.sample(model, n_chains=1, n_warmup=10, n_samples=10, seed=1, init_jitter=-1.0)

    def test_mutually_exclusive_init_modes_raises(self):
        model = _make_fast_model()
        with pytest.raises(ValueError, match="mutually exclusive"):
            nextstat.sample(
                model,
                n_chains=1,
                n_warmup=10,
                n_samples=10,
                seed=1,
                init_jitter=0.1,
                init_jitter_rel=0.1,
            )


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

class TestSampleQualityGates:
    """Basic quality: R-hat, divergence rate, POI mean."""

    pytestmark = pytest.mark.slow

    @pytest.fixture()
    def result(self):
        if os.environ.get("NS_RUN_SLOW") != "1":
            pytest.skip("Set NS_RUN_SLOW=1 to run slow sampling quality gates.")
        # Keep this quality gate stable and fast: use the small GaussianMeanModel.
        # HistFactory models can require significantly longer warmup/sampling to reach
        # tight R-hat thresholds.
        model = _make_fast_model()
        return nextstat.sample(
            model, n_chains=2, n_warmup=200, n_samples=200, seed=42,
        )

    def test_r_hat_below_threshold(self, result):
        for name, rhat in result["diagnostics"]["r_hat"].items():
            assert rhat < 1.1, f"R-hat for {name} too high: {rhat}"

    def test_divergence_rate_low(self, result):
        assert result["diagnostics"]["divergence_rate"] < 0.05, (
            f"Divergence rate too high: {result['diagnostics']['divergence_rate']}"
        )

    def test_poi_mean_reasonable(self, result):
        poi_name = result["param_names"][0]
        all_draws = [
            x for chain in result["posterior"][poi_name] for x in chain
        ]
        mean = sum(all_draws) / len(all_draws)
        assert 0.0 < mean < 5.0, f"POI mean out of range: {mean}"


# ---------------------------------------------------------------------------
# Data override
# ---------------------------------------------------------------------------

class TestSampleWithOverriddenData:
    """The data= keyword overrides main observations."""

    def test_data_override_changes_posterior(self):
        model = _make_model()
        kwargs = dict(n_chains=1, n_warmup=30, n_samples=20, seed=7)
        r_default = nextstat.sample(model, **kwargs)
        r_override = nextstat.sample(model, data=[200.0, 300.0], **kwargs)
        poi = r_default["param_names"][0]
        mean_default = sum(
            x for c in r_default["posterior"][poi] for x in c
        ) / float(kwargs["n_chains"] * kwargs["n_samples"])
        mean_override = sum(
            x for c in r_override["posterior"][poi] for x in c
        ) / float(kwargs["n_chains"] * kwargs["n_samples"])
        # With much higher counts the POI mean should shift
        assert abs(mean_default - mean_override) > 0.01, (
            f"data= override had no effect: {mean_default} vs {mean_override}"
        )


# ---------------------------------------------------------------------------
# Config options
# ---------------------------------------------------------------------------

class TestSampleConfig:
    """max_treedepth, target_accept kwargs are respected."""

    def test_custom_max_treedepth(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model, n_chains=1, n_warmup=20, n_samples=10,
            seed=1, max_treedepth=5,
        )
        for depth in result["sample_stats"]["tree_depth"][0]:
            assert depth <= 5, f"Tree depth {depth} exceeds max_treedepth=5"

    def test_custom_target_accept(self):
        model = _make_fast_model()
        n_samples = 20
        result = nextstat.sample(
            model, n_chains=1, n_warmup=30, n_samples=n_samples,
            seed=1, target_accept=0.95,
        )
        # With high target_accept, step size should be smaller → higher mean accept
        mean_accept = sum(result["sample_stats"]["accept_prob"][0]) / float(n_samples)
        assert mean_accept > 0.5, f"Mean accept prob too low: {mean_accept}"

    def test_init_jitter_rel_smoke(self):
        model = _make_fast_model()
        result = nextstat.sample(
            model, n_chains=1, n_warmup=20, n_samples=10, seed=1, init_jitter_rel=0.10
        )
        assert isinstance(result, dict)

    def test_init_jitter_mutual_exclusive(self):
        model = _make_fast_model()
        with pytest.raises(ValueError):
            nextstat.sample(
                model,
                n_chains=1,
                n_warmup=10,
                n_samples=10,
                seed=1,
                init_jitter=0.1,
                init_jitter_rel=0.10,
            )

    def test_init_overdispersed_produces_distinct_chains(self):
        model = _make_fast_model()
        r = nextstat.sample(
            model,
            n_chains=2,
            n_warmup=20,
            n_samples=10,
            seed=123,
            init_overdispersed_rel=0.50,
        )
        poi = r["param_names"][0]
        c0 = r["posterior"][poi][0]
        c1 = r["posterior"][poi][1]
        assert c0 != c1, "Overdispersed init should not produce identical chains"
