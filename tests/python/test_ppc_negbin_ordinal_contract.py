"""Contract tests for PPC utilities (negbin + ordinal)."""

from __future__ import annotations

import nextstat


def test_ppc_negbin_from_sample_smoke():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 1, 3, 2]

    # Negative binomial regression model (log link).
    model = nextstat._core.NegativeBinomialRegressionModel(x, y, include_intercept=True, offset=None)
    raw = nextstat.sample(
        model,
        n_chains=1,
        n_warmup=30,
        n_samples=20,
        seed=321,
        init_jitter_rel=0.1,
    )

    spec = nextstat.ppc.NegBinomSpec(x=[list(r) for r in x], y=[int(v) for v in y], include_intercept=True)
    out = nextstat.ppc.ppc_negbin_from_sample(spec, raw, n_draws=5, seed=0)
    assert isinstance(out.observed, dict)
    assert isinstance(out.replicated, list)
    assert len(out.replicated) == 5
    assert "mean" in out.observed
    assert "mean" in out.replicated[0]


def test_ppc_ordered_logit_from_sample_smoke():
    x = [[-1.0], [0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 2, 2]
    n_levels = 3

    model = nextstat._core.OrderedLogitModel(x, y, n_levels=n_levels)
    raw = nextstat.sample(
        model,
        n_chains=1,
        n_warmup=30,
        n_samples=20,
        seed=322,
        init_jitter_rel=0.1,
    )

    spec = nextstat.ppc.OrderedSpec(x=[list(r) for r in x], y=[int(v) for v in y], n_levels=n_levels, link="logit")
    out = nextstat.ppc.ppc_ordered_from_sample(spec, raw, n_draws=5, seed=0)
    assert isinstance(out.observed, dict)
    assert isinstance(out.replicated, list)
    assert len(out.replicated) == 5
    assert "mean" in out.observed
    assert "mean" in out.replicated[0]


def test_ppc_ordered_probit_from_sample_smoke():
    x = [[-1.0], [0.0], [1.0], [2.0], [3.0]]
    y = [0, 1, 1, 2, 2]
    n_levels = 3

    model = nextstat._core.OrderedProbitModel(x, y, n_levels=n_levels)
    raw = nextstat.sample(
        model,
        n_chains=1,
        n_warmup=30,
        n_samples=20,
        seed=323,
        init_jitter_rel=0.1,
    )

    spec = nextstat.ppc.OrderedSpec(
        x=[list(r) for r in x],
        y=[int(v) for v in y],
        n_levels=n_levels,
        link="probit",
    )
    out = nextstat.ppc.ppc_ordered_from_sample(spec, raw, n_draws=5, seed=0)
    assert isinstance(out.observed, dict)
    assert isinstance(out.replicated, list)
    assert len(out.replicated) == 5

