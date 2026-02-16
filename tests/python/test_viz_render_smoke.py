from __future__ import annotations

import importlib.util

import pytest

from nextstat import viz_render


def _has_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


@pytest.mark.parametrize("kind,artifact", [
    (
        "pulls",
        {
            "schema_version": "trex_report_pulls_v0",
            "entries": [
                {"name": "np1", "pull": 0.2, "constraint": 0.9},
                {"name": "np2", "pull": -0.4, "constraint": 1.1},
            ],
        },
    ),
    (
        "corr",
        {
            "schema_version": "trex_report_corr_v0",
            "parameter_names": ["a", "b"],
            "corr": [[1.0, 0.3], [0.3, 1.0]],
        },
    ),
    (
        "ranking",
        {
            "names": ["np1", "np2"],
            "delta_mu_up": [0.1, 0.05],
            "delta_mu_down": [-0.08, -0.04],
            "pull": [0.2, -0.1],
            "constraint": [0.9, 1.0],
        },
    ),
])
def test_render_artifact_smoke(kind, artifact, tmp_path):
    if not _has_matplotlib():
        pytest.skip("matplotlib is not installed")

    out = tmp_path / f"{kind}.png"
    viz_render.render_artifact(kind=kind, artifact=artifact, output=out, dpi=120)
    assert out.exists()
    assert out.stat().st_size > 0
