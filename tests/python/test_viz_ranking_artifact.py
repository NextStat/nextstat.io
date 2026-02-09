from __future__ import annotations

import json

import nextstat

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())

def test_ranking_artifact_contract():
    # Use a fixture with constrained nuisance parameters so ranking is non-empty.
    ws = load_fixture("complex_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    artifact = nextstat.viz.ranking_artifact(model)
    arrays = nextstat.viz.ranking_arrays(artifact)

    assert {"names", "delta_mu_up", "delta_mu_down", "pull", "constraint"} <= set(arrays.keys())

    n = len(arrays["names"])
    assert n > 0
    assert len(arrays["delta_mu_up"]) == n
    assert len(arrays["delta_mu_down"]) == n
    assert len(arrays["pull"]) == n
    assert len(arrays["constraint"]) == n
