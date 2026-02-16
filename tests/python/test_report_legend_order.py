from __future__ import annotations

from nextstat.report import _legend_sort_key


def test_legend_order_puts_mc_unc_after_stack_samples() -> None:
    sample_labels = {"signal", "ttbar", "zjets", "wjets"}
    labels = [
        "MC unc. (total)",
        "zjets",
        "Data",
        "MC unc. (stat)",
        "Total postfit",
        "wjets",
        "Total prefit",
        "signal",
    ]
    got = sorted(labels, key=lambda s: _legend_sort_key(s, sample_labels))
    assert got == [
        "Data",
        "Total postfit",
        "Total prefit",
        "signal",
        "wjets",
        "zjets",
        "MC unc. (stat)",
        "MC unc. (total)",
    ]


def test_legend_order_keeps_unknown_labels_between_stack_and_uncertainty() -> None:
    sample_labels = {"background"}
    labels = [
        "MC unc. (total)",
        "background",
        "Aux marker",
    ]
    got = sorted(labels, key=lambda s: _legend_sort_key(s, sample_labels))
    assert got == ["background", "Aux marker", "MC unc. (total)"]
