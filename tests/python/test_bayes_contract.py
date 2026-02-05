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


def test_to_inferencedata_requires_arviz():
    # The CI environment doesn't install `arviz`; this should fail loudly and clearly.
    with pytest.raises(ImportError):
        nextstat.bayes.to_inferencedata({"posterior": {}, "sample_stats": {}, "diagnostics": {}})

