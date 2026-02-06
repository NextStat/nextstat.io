"""Contract tests for Bayesian Python helpers (no optional deps required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def test_bayes_sample_returns_raw_dict_when_requested():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    raw = nextstat.bayes.sample(
        model,
        n_chains=1,
        n_warmup=20,
        n_samples=10,
        seed=1,
        init_jitter_rel=0.10,
        return_idata=False,
    )
    assert isinstance(raw, dict)
    assert {"posterior", "sample_stats", "diagnostics"} <= set(raw.keys())


def test_bayes_sample_accepts_init_overdispersed_rel_kwarg():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    raw = nextstat.bayes.sample(
        model,
        n_chains=1,
        n_warmup=20,
        n_samples=10,
        seed=2,
        init_overdispersed_rel=0.10,
        return_idata=False,
    )
    assert isinstance(raw, dict)


def test_bayes_sample_multichain_uses_distinct_seeds_by_default():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    raw = nextstat.bayes.sample(
        model,
        n_chains=2,
        n_warmup=30,
        n_samples=15,
        seed=3,
        init_jitter=0.0,
        init_jitter_rel=None,
        init_overdispersed_rel=None,
        return_idata=False,
    )
    posterior = raw["posterior"]
    assert isinstance(posterior, dict)
    assert posterior, "posterior must be non-empty"
    first_param = next(iter(posterior.keys()))
    chains = posterior[first_param]
    assert isinstance(chains, list) and len(chains) == 2
    assert chains[0] != chains[1], "distinct chains should not be identical when jitter=0"


def test_to_inferencedata_requires_arviz():
    # The CI environment doesn't install `arviz`; this should fail loudly and clearly.
    with pytest.raises(ImportError):
        nextstat.bayes.to_inferencedata({"posterior": {}, "sample_stats": {}, "diagnostics": {}})
