"""Smoke test for generic sampling on a non-HEP model.

This validates that `nextstat.sample()` accepts models beyond HistFactoryModel.
"""

import nextstat


def test_sample_accepts_gaussian_mean_model():
    model = nextstat.GaussianMeanModel([1.0, 2.0, 3.0, 4.0], sigma=1.0)
    result = nextstat.sample(model, n_chains=2, n_warmup=20, n_samples=20, seed=7)

    assert result["param_names"] == ["mu"]
    posterior = result["posterior"]["mu"]
    assert len(posterior) == 2
    assert len(posterior[0]) == 20

    draws = [x for chain in posterior for x in chain]
    mean = sum(draws) / len(draws)
    # The true posterior mean equals sample mean; keep a loose bound for short chains.
    assert 1.5 < mean < 3.5
