"""Contract tests for viz helpers (no matplotlib required)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.mark.slow
def test_viz_artifacts_contracts():
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow viz contract tests.")

    ws = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))

    cls = nextstat.viz.cls_curve(model, [0.0, 1.0, 2.0], alpha=0.05)
    assert {"alpha", "nsigma_order", "obs_limit", "exp_limits", "points"} <= set(cls.keys())
    assert len(cls["points"]) == 3
    assert {"mu", "cls", "expected"} <= set(cls["points"][0].keys())
    # Expanded series form (arrays aligned with points).
    assert {"mu_values", "cls_obs", "cls_exp"} <= set(cls.keys())
    assert len(cls["mu_values"]) == 3
    assert len(cls["cls_obs"]) == 3
    assert len(cls["cls_exp"]) == 3
    assert len(cls["cls_exp"][0]) == 3

    prof = nextstat.viz.profile_curve(model, [0.0, 2.0])
    assert {"poi_index", "mu_hat", "nll_hat", "points"} <= set(prof.keys())
    assert len(prof["points"]) == 2
    assert {"mu_values", "q_mu_values", "twice_delta_nll"} <= set(prof.keys())
    assert len(prof["mu_values"]) == 2
    assert len(prof["q_mu_values"]) == 2
    assert len(prof["twice_delta_nll"]) == 2


def test_plot_helpers_require_matplotlib():
    with pytest.raises(ImportError):
        nextstat.viz.plot_cls_curve({"alpha": 0.05, "points": []})
    with pytest.raises(ImportError):
        nextstat.viz.plot_profile_curve({"points": []})
