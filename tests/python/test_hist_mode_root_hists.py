from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextstat.analysis.hist_mode import read_root_histogram, read_root_histograms


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _load_expected() -> dict[str, dict]:
    return json.loads((FIXTURES_DIR / "simple_histos_expected.json").read_text(encoding="utf-8"))


def test_hist_mode_reads_simple_root_hists_drop_policy():
    root_path = FIXTURES_DIR / "simple_histos.root"
    expected = _load_expected()

    out = read_root_histograms(root_path, expected.keys(), flow_policy="drop")
    assert set(out.keys()) == set(expected.keys())

    for hist_path, exp in expected.items():
        got = out[hist_path]
        assert got["flow_policy"] == "drop"
        assert float(got["underflow"]) == pytest.approx(0.0, abs=1e-12)
        assert float(got["overflow"]) == pytest.approx(0.0, abs=1e-12)

        assert got["bin_edges"] == pytest.approx(exp["bin_edges"], abs=1e-12)
        assert got["bin_content"] == pytest.approx(exp["bin_content"], abs=1e-12)

        sw2 = got.get("sumw2")
        if sw2 is not None:
            assert len(sw2) == len(exp["bin_content"])


def test_hist_mode_fold_policy_is_explicit_and_stable_for_zero_flows():
    root_path = FIXTURES_DIR / "simple_histos.root"
    expected = _load_expected()

    got_drop = read_root_histogram(root_path, "hist1", flow_policy="drop")
    got_fold = read_root_histogram(root_path, "hist1", flow_policy="fold")

    assert got_drop["flow_policy"] == "drop"
    assert got_fold["flow_policy"] == "fold"

    assert got_drop["bin_edges"] == pytest.approx(expected["hist1"]["bin_edges"], abs=1e-12)
    assert got_fold["bin_edges"] == pytest.approx(expected["hist1"]["bin_edges"], abs=1e-12)

    # For this fixture, under/overflow are zero, so folding is a no-op on main bins.
    assert got_fold["bin_content"] == pytest.approx(got_drop["bin_content"], abs=1e-12)
