from __future__ import annotations

import pytest

from nextstat.analysis.preprocess import NegativeBinsHygieneStep, PreprocessPipeline, apply_negative_bins_policy


def test_negative_bins_policy_error_raises() -> None:
    with pytest.raises(ValueError, match="negative bins"):
        apply_negative_bins_policy([1.0, -0.1], policy="error")


def test_negative_bins_policy_keep_warn_no_changes() -> None:
    r = apply_negative_bins_policy([1.0, -0.1], policy="keep_warn")
    assert r.bins == [1.0, -0.1]
    assert r.changed is False
    assert r.n_negative == 1
    assert r.warnings


def test_negative_bins_policy_clamp_renorm_preserves_integral_when_possible() -> None:
    r = apply_negative_bins_policy([-1.0, 3.0], policy="clamp_renorm", renorm=True)
    assert r.changed is True
    assert sum(r.bins) == pytest.approx(2.0)
    assert r.bins == pytest.approx([0.0, 2.0])


def test_negative_bins_policy_clamp_renorm_skips_when_integral_nonpositive() -> None:
    r = apply_negative_bins_policy([-1.0, 0.0], policy="clamp_renorm", renorm=True)
    assert r.changed is True
    assert r.bins == pytest.approx([0.0, 0.0])
    assert r.scale is None
    assert r.warnings


def test_negative_bins_hygiene_step_records_provenance() -> None:
    ws = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [1.0, -0.5],
                        "modifiers": [],
                    }
                ],
            }
        ],
        "observations": [{"name": "SR", "data": [1.0, 1.0]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }

    pipe = PreprocessPipeline([NegativeBinsHygieneStep(policy="clamp_renorm", record_unchanged=True)])
    res = pipe.run(ws)
    assert res.workspace["channels"][0]["samples"][0]["data"][1] >= 0.0
    recs = res.provenance_dict()["steps"][0]["records"]
    assert recs and recs[0]["modifier"] == "__nominal__"

