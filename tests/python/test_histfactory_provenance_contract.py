from __future__ import annotations

from pathlib import Path


def test_histfactory_model_exposes_channel_sample_breakdown():
    import nextstat

    xml_path = Path(__file__).resolve().parents[1] / "fixtures" / "histfactory" / "combination.xml"
    model = nextstat.from_histfactory_xml(xml_path)

    obs = model.observed_main_by_channel()
    assert isinstance(obs, list) and obs
    assert "channel_name" in obs[0]
    assert "y" in obs[0]

    params = model.suggested_init()
    exp = model.expected_main_by_channel_sample(params)
    assert isinstance(exp, list) and exp
    row = exp[0]
    assert {"channel_name", "samples", "total"} <= set(row.keys())
    assert isinstance(row["samples"], list) and row["samples"]
    assert {"sample_name", "y"} <= set(row["samples"][0].keys())
