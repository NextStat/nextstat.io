"""TRExFitter `.config` parser contract tests (dependency-light).

These tests do not require the Rust extension module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nextstat.trex_config.parser import TrexConfigParseError, parse_trex_config, parse_trex_config_file


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "trex_config" / "minimal_tutorial.config"


def test_parse_fixture_blocks_and_values():
    doc = parse_trex_config_file(FIXTURE)

    regions = [b for b in doc.blocks if b.kind.lower() == "region"]
    assert len(regions) == 1
    r = regions[0]
    assert r.name == "SR"

    assert r.last("Variable") is not None
    assert r.last("Variable").value.raw == "mbb"

    binning = r.last("Binning")
    assert binning is not None
    assert binning.value.items == ["0", "50", "100", "150", "200", "300"]

    sel = r.last("Selection")
    assert sel is not None
    assert "njet" in sel.value.raw
    assert "#" not in sel.value.raw

    samples = [b for b in doc.blocks if b.kind.lower() == "sample"]
    assert len(samples) == 1
    s = samples[0]
    assert s.name == "signal"
    assert s.last("File").value.raw.endswith("tests/fixtures/simple_tree.root")

    regs = s.last("Regions")
    assert regs is not None
    assert regs.value.items == ["SR", "CR"]

    systs = [b for b in doc.blocks if b.kind.lower() == "systematic"]
    assert len(systs) == 1
    sys = systs[0]
    assert sys.name == "jes"
    assert sys.last("Type").value.raw == "weight"


def test_parse_unclosed_quote_reports_line():
    bad = 'Region: "SR\nVariable: mbb\n'
    with pytest.raises(TrexConfigParseError) as ei:
        parse_trex_config(bad, path="bad.config")
    msg = str(ei.value)
    assert "bad.config:1" in msg
    assert "unclosed quote" in msg or "unterminated" in msg

