"""Contract tests for viz helpers (no matplotlib required)."""

from __future__ import annotations

import importlib.util
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
    # `cls_exp` is sigma-major: shape (len(nsigma_order), len(mu_values)).
    assert len(cls["cls_exp"]) == len(cls["nsigma_order"])
    assert len(cls["cls_exp"][0]) == 3

    prof = nextstat.viz.profile_curve(model, [0.0, 2.0])
    assert {"poi_index", "mu_hat", "nll_hat", "points"} <= set(prof.keys())
    assert len(prof["points"]) == 2
    assert {"mu_values", "q_mu_values", "twice_delta_nll"} <= set(prof.keys())
    assert len(prof["mu_values"]) == 2
    assert len(prof["q_mu_values"]) == 2
    assert len(prof["twice_delta_nll"]) == 2

    rank = nextstat.viz.ranking_artifact(model, top_n=5)
    assert {"entries", "n_total", "n_returned"} <= set(rank.keys())
    assert isinstance(rank["entries"], list)
    assert int(rank["n_total"]) >= int(rank["n_returned"])
    assert int(rank["n_returned"]) == len(rank["entries"])
    assert int(rank["n_total"]) > 0
    assert len(rank["entries"]) > 0
    assert {"name", "delta_mu_up", "delta_mu_down", "pull", "constraint"} <= set(rank["entries"][0].keys())


def test_plot_helpers_require_matplotlib():
    has_mpl = importlib.util.find_spec("matplotlib") is not None
    if not has_mpl:
        with pytest.raises(ImportError):
            nextstat.viz.plot_cls_curve({"alpha": 0.05, "points": []})
        with pytest.raises(ImportError):
            nextstat.viz.plot_brazil_limits({"alpha": 0.05, "points": []})
        with pytest.raises(ImportError):
            nextstat.viz.plot_profile_curve({"points": []})
        with pytest.raises(ImportError):
            nextstat.viz.plot_pulls({"entries": []})
        with pytest.raises(ImportError):
            nextstat.viz.plot_ranking({"entries": []})
        with pytest.raises(ImportError):
            nextstat.viz.plot_corr_matrix({"parameter_names": [], "corr": []})
        return

    ax = nextstat.viz.plot_cls_curve({"alpha": 0.05, "mu_values": [0.0, 1.0], "cls_obs": [0.9, 0.1]})
    assert ax is not None

    ax = nextstat.viz.plot_brazil_limits(
        {"alpha": 0.05, "obs_limit": 1.0, "exp_limits": [1.4, 1.2, 1.0, 0.8, 0.6], "nsigma_order": [2, 1, 0, -1, -2]}
    )
    assert ax is not None

    ax = nextstat.viz.plot_profile_curve({"mu_values": [0.0, 1.0], "q_mu_values": [0.0, 1.0], "mu_hat": 0.3})
    assert ax is not None

    ax = nextstat.viz.plot_pulls(
        {
            "entries": [
                {"name": "np1", "pull": 0.2, "constraint": 0.8},
                {"name": "np2", "pull": -0.3, "constraint": 1.1},
            ]
        }
    )
    assert ax is not None

    ax_pull, ax_impact = nextstat.viz.plot_ranking(
        {
            "entries": [
                {"name": "np1", "delta_mu_up": 0.1, "delta_mu_down": -0.1, "pull": 0.0, "constraint": 1.0}
            ]
        }
    )
    assert ax_pull is not None and ax_impact is not None

    ax = nextstat.viz.plot_corr_matrix({"parameter_names": ["a", "b"], "corr": [[1.0, 0.0], [0.0, 1.0]]})
    assert ax is not None
