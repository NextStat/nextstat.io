"""Contract tests for viz helpers (no matplotlib required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def test_viz_artifacts_contracts():
    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    cls = nextstat.viz.cls_curve(model, [0.0, 0.5, 1.0, 1.5, 2.0], alpha=0.05)
    assert {"alpha", "nsigma_order", "obs_limit", "exp_limits", "points"} <= set(cls.keys())
    assert len(cls["points"]) == 5
    assert {"mu", "cls", "expected"} <= set(cls["points"][0].keys())

    prof = nextstat.viz.profile_curve(model, [0.0, 1.0, 2.0])
    assert {"poi_index", "mu_hat", "nll_hat", "points"} <= set(prof.keys())
    assert len(prof["points"]) == 3


def test_plot_helpers_require_matplotlib():
    with pytest.raises(ImportError):
        nextstat.viz.plot_cls_curve({"alpha": 0.05, "points": []})
    with pytest.raises(ImportError):
        nextstat.viz.plot_profile_curve({"points": []})

