from __future__ import annotations

from pathlib import Path


def test_histfactory_bin_edges_by_channel_smoke():
    import nextstat

    xml_path = Path(__file__).resolve().parents[1] / "fixtures" / "histfactory" / "combination.xml"
    edges = nextstat.histfactory_bin_edges_by_channel(xml_path)
    assert isinstance(edges, dict)
    assert edges, "expected at least 1 channel"

    first = next(iter(edges.values()))
    assert isinstance(first, list)
    assert len(first) >= 2
    assert all(float(first[i + 1]) > float(first[i]) for i in range(len(first) - 1))
