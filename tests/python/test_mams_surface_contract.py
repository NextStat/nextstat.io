"""Stable-surface contract tests for Python MAMS exposure."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))

import nextstat


def _make_model():
    return nextstat.GaussianMeanModel([1.0, 2.0, 3.0, 4.0] * 5, sigma=1.0)


def test_mams_sample_stats_and_quality_contract():
    result = nextstat.sample(
        _make_model(),
        method="mams",
        n_chains=2,
        n_warmup=40,
        n_samples=20,
        seed=1,
    )

    stats = result["sample_stats"]
    assert stats["metric_type"] == "diagonal"
    assert len(stats["n_leapfrog_warmup_total"]) == 2
    assert all(total > 0 for total in stats["n_leapfrog_warmup_total"])
    assert len(stats["accept_prob"]) == 2
    assert all(len(chain) == 20 for chain in stats["accept_prob"])
    for chain in stats["accept_prob"]:
        assert all(0.0 <= value <= 1.0 for value in chain)

    quality = result["diagnostics"]["quality"]
    assert quality["status"] in {"ok", "warn", "fail"}
    assert isinstance(quality["warnings"], list)
    assert isinstance(quality["failures"], list)


def test_mams_public_signature_defaults_are_stabilized():
    sig = nextstat.sample_mams.__text_signature__
    assert "n_warmup=3500" in sig
    assert "target_accept=0.985" in sig
    assert "max_leapfrog=1024" in sig
    assert "eps_jitter=0.0" in sig


def test_unified_sample_docstring_matches_mams_stable_defaults():
    doc = nextstat.sample.__doc__ or ""
    assert "500 for NUTS/WALNUTS, 3500 for MAMS" in doc
    assert "0.8 (NUTS/WALNUTS), 0.985 (MAMS), 0.9 (LAPS)" in doc
    assert "eps_jitter (float): Step size jitter scale. Default: 0.0 on the stable CPU surface." in doc


def test_mams_default_surface_handles_funnel_geometry():
    result = nextstat.sample(
        nextstat.FunnelModel(),
        method="mams",
        n_chains=4,
        n_samples=2000,
        seed=42,
    )

    max_rhat = max(float(v) for v in result["diagnostics"]["r_hat"].values())
    min_ess_bulk = min(float(v) for v in result["diagnostics"]["ess_bulk"].values())
    assert max_rhat < 1.01
    assert min_ess_bulk > 500.0


def test_mams_surface_reports_divergences_when_transitions_blow_up():
    result = nextstat.sample(
        nextstat.FunnelModel(),
        method="mams",
        n_chains=1,
        n_warmup=0,
        n_samples=40,
        init_step_size=8.0,
        init_l=12.0,
        max_leapfrog=512,
        diagonal_precond=False,
        eps_jitter=0.0,
        seed=1,
    )

    diverging = result["sample_stats"]["diverging"][0]
    assert any(diverging)
    assert result["diagnostics"]["divergence_rate"] > 0.0


@pytest.mark.parametrize("metric", ["dense", "auto"])
def test_mams_rejects_non_diagonal_metrics(metric):
    with pytest.raises(ValueError, match="metric='diagonal'"):
        nextstat.sample(
            _make_model(),
            method="mams",
            metric=metric,
            n_chains=2,
            n_warmup=20,
            n_samples=10,
            seed=1,
        )
