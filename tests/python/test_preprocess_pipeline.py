from __future__ import annotations

import copy

import pytest

from nextstat.analysis.preprocess import PreprocessPipeline, SymmetrizeHistoSysStep


def _ws_with_one_histosys(*, nominal, hi, lo, channel="SR", sample="bkg", modifier="shape") -> dict:
    return {
        "version": "1.0.0",
        "channels": [
            {
                "name": channel,
                "samples": [
                    {
                        "name": sample,
                        "data": list(nominal),
                        "modifiers": [
                            {
                                "name": modifier,
                                "type": "histosys",
                                "data": {"hi_data": list(hi), "lo_data": list(lo)},
                            }
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": channel, "data": list(nominal)}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }


def test_pipeline_symmetrize_histosys_absmean_updates_workspace_and_provenance() -> None:
    ws = _ws_with_one_histosys(nominal=[10.0, 20.0], hi=[12.0, 18.0], lo=[9.0, 19.0])
    pipe = PreprocessPipeline([SymmetrizeHistoSysStep(method="absmean")])

    res = pipe.run(ws)
    out = res.workspace

    mod = out["channels"][0]["samples"][0]["modifiers"][0]
    assert mod["data"]["hi_data"] == [11.5, 21.5]
    assert mod["data"]["lo_data"] == [8.5, 18.5]

    prov = res.provenance_dict()
    assert prov["version"] == "v0"
    assert len(prov["steps"]) == 1
    recs = prov["steps"][0]["records"]
    assert len(recs) == 1
    assert recs[0]["kind"] == "histosys.symmetrize"
    assert recs[0]["changed"] is True


def test_pipeline_provenance_record_order_is_deterministic() -> None:
    ws = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "B",
                "samples": [
                    {
                        "name": "s2",
                        "data": [1.0],
                        "modifiers": [
                            {"name": "m2", "type": "histosys", "data": {"hi_data": [1.1], "lo_data": [0.9]}},
                            {"name": "a", "type": "normsys", "data": {"hi": 1.0, "lo": 1.0}},
                        ],
                    }
                ],
            },
            {
                "name": "A",
                "samples": [
                    {
                        "name": "s1",
                        "data": [1.0],
                        "modifiers": [
                            {"name": "m1", "type": "histosys", "data": {"hi_data": [1.2], "lo_data": [0.8]}},
                        ],
                    },
                    {
                        "name": "s0",
                        "data": [1.0],
                        "modifiers": [
                            {"name": "m0", "type": "histosys", "data": {"hi_data": [1.3], "lo_data": [0.7]}},
                        ],
                    },
                ],
            },
        ],
        "observations": [{"name": "A", "data": [1.0]}, {"name": "B", "data": [1.0]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }

    pipe = PreprocessPipeline([SymmetrizeHistoSysStep(method="twosided", record_unchanged=True)])
    res = pipe.run(ws)
    recs = res.provenance_dict()["steps"][0]["records"]
    keys = [(r["channel"], r["sample"], r["modifier"]) for r in recs]
    assert keys == [("A", "s0", "m0"), ("A", "s1", "m1"), ("B", "s2", "m2")]


def test_pipeline_is_idempotent_when_already_symmetric() -> None:
    ws = _ws_with_one_histosys(nominal=[100.0, 105.0], hi=[110.0, 115.0], lo=[90.0, 95.0])
    pipe = PreprocessPipeline([SymmetrizeHistoSysStep(method="absmean", record_unchanged=True)])

    r1 = pipe.run(copy.deepcopy(ws))
    r2 = pipe.run(copy.deepcopy(r1.workspace))
    assert r1.workspace == r2.workspace
    assert r1.provenance.output_sha256 == r2.provenance.output_sha256


def test_pipeline_negative_policy_clamp() -> None:
    ws = _ws_with_one_histosys(nominal=[0.1], hi=[0.2], lo=[-0.1])
    pipe = PreprocessPipeline([SymmetrizeHistoSysStep(method="absmean", negative_policy="clamp")])

    out = pipe.run(ws).workspace
    mod = out["channels"][0]["samples"][0]["modifiers"][0]
    assert mod["data"]["hi_data"] == pytest.approx([0.25])
    assert mod["data"]["lo_data"] == pytest.approx([0.0])

