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


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------

class TestSampleAPIShape:
    """Return value has correct top-level keys and nested dimensions."""

    def test_returns_dict_with_expected_keys(self):
        model = _make_model()
        result = nextstat.sample(model, n_chains=2, n_warmup=50, n_samples=30, seed=1)
        assert isinstance(result, dict)
        expected_keys = {
            "posterior", "sample_stats", "diagnostics",
            "param_names", "n_chains", "n_warmup", "n_samples",
        }
        assert expected_keys <= set(result.keys())

    def test_posterior_shape(self):
        model = _make_model()
        n_chains, n_samples = 2, 30
        result = nextstat.sample(
            model, n_chains=n_chains, n_warmup=50, n_samples=n_samples, seed=1,
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
        model = _make_model()
        n_chains, n_samples = 2, 30
        result = nextstat.sample(
            model, n_chains=n_chains, n_warmup=50, n_samples=n_samples, seed=1,
        )
        stats = result["sample_stats"]
        assert "diverging" in stats
        assert "tree_depth" in stats
        assert "accept_prob" in stats
        assert "step_size" in stats
        assert len(stats["diverging"]) == n_chains
        assert len(stats["diverging"][0]) == n_samples
        assert len(stats["step_size"]) == n_chains

    def test_diagnostics_keys(self):
        model = _make_model()
        result = nextstat.sample(model, n_chains=2, n_warmup=50, n_samples=30, seed=1)
        diag = result["diagnostics"]
        assert "r_hat" in diag
        assert "ess_bulk" in diag
        assert "ess_tail" in diag
        assert "divergence_rate" in diag
        assert "max_treedepth_rate" in diag
        # r_hat should be a dict keyed by param name
        for name in result["param_names"]:
            assert name in diag["r_hat"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestSampleReproducibility:
    """Same seed produces identical draws."""

    def test_same_seed_same_draws(self):
        model = _make_model()
        kwargs = dict(n_chains=2, n_warmup=50, n_samples=20, seed=42)
        r1 = nextstat.sample(model, **kwargs)
        r2 = nextstat.sample(model, **kwargs)
        for name in r1["param_names"]:
            assert r1["posterior"][name] == r2["posterior"][name], (
                f"Draws for {name} should be identical with same seed"
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
        model = _make_model()
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
        kwargs = dict(n_chains=2, n_warmup=100, n_samples=50, seed=7)
        r_default = nextstat.sample(model, **kwargs)
        r_override = nextstat.sample(model, data=[200.0, 300.0], **kwargs)
        poi = r_default["param_names"][0]
        mean_default = sum(
            x for c in r_default["posterior"][poi] for x in c
        ) / (2 * 50)
        mean_override = sum(
            x for c in r_override["posterior"][poi] for x in c
        ) / (2 * 50)
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
        model = _make_model()
        result = nextstat.sample(
            model, n_chains=1, n_warmup=50, n_samples=30,
            seed=1, max_treedepth=5,
        )
        for depth in result["sample_stats"]["tree_depth"][0]:
            assert depth <= 5, f"Tree depth {depth} exceeds max_treedepth=5"

    def test_custom_target_accept(self):
        model = _make_model()
        result = nextstat.sample(
            model, n_chains=1, n_warmup=100, n_samples=50,
            seed=1, target_accept=0.95,
        )
        # With high target_accept, step size should be smaller → higher mean accept
        mean_accept = sum(result["sample_stats"]["accept_prob"][0]) / 50
        assert mean_accept > 0.5, f"Mean accept prob too low: {mean_accept}"
