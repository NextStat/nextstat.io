"""HIST mode: read binned ROOT histograms and map to analysis-ready arrays.

This is a small, dependency-free surface intended for TREx-like workflows
(`ReadFrom: HIST`), where production inputs are ROOT TH1 histograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

import nextstat._core as _core

FlowPolicy = Literal["drop", "fold"]


def _to_path_str(p: str | Path) -> str:
    return str(p) if isinstance(p, Path) else str(p)


def read_root_histogram(
    root_path: str | Path,
    hist_path: str,
    *,
    flow_policy: FlowPolicy = "drop",
) -> dict[str, Any]:
    """Read one histogram from a ROOT file.

    Returns a dict with:
    - `bin_edges`: list[float] (len = n_bins + 1)
    - `bin_content`: list[float] (len = n_bins)
    - `sumw2`: list[float] | None (len = n_bins)
    - `underflow`, `overflow`: float
    - `underflow_sumw2`, `overflow_sumw2`: float | None
    - `flow_policy`: "drop" | "fold"

    Flow policy:
    - "drop": keep `bin_content` as-is (exclude flows).
    - "fold": add underflow to the first bin, overflow to the last bin (same for sumw2).
    """
    if flow_policy not in ("drop", "fold"):
        raise ValueError(f"invalid flow_policy: {flow_policy!r}")

    raw: dict[str, Any] = _core.read_root_histogram(_to_path_str(root_path), str(hist_path))
    raw["flow_policy"] = flow_policy

    # Ensure `sumw2` is always available for downstream TREx-like workflows.
    #
    # Policy:
    # - if ROOT provides sumw2: keep it (`sumw2_policy="root"`)
    # - else: assume unweighted Poisson semantics and set sumw2 := bin_content
    #         (`sumw2_policy="poisson_fallback"`)
    if raw.get("sumw2") is None:
        raw["sumw2_policy"] = "poisson_fallback"
        raw["sumw2"] = list(raw.get("bin_content") or [])
    else:
        raw["sumw2_policy"] = "root"

    if flow_policy == "drop":
        return raw

    # fold
    content = list(raw.get("bin_content") or [])
    if not content:
        return raw

    underflow = float(raw.get("underflow") or 0.0)
    overflow = float(raw.get("overflow") or 0.0)
    content[0] += underflow
    content[-1] += overflow
    raw["bin_content"] = content

    sw2_list = list(raw.get("sumw2") or [])
    if sw2_list:
        uf2 = raw.get("underflow_sumw2")
        of2 = raw.get("overflow_sumw2")
        sw2_list[0] += float(uf2) if uf2 is not None else 0.0
        sw2_list[-1] += float(of2) if of2 is not None else 0.0
        raw["sumw2"] = sw2_list

    return raw


def read_root_histograms(
    root_path: str | Path,
    hist_paths: Iterable[str],
    *,
    flow_policy: FlowPolicy = "drop",
) -> dict[str, dict[str, Any]]:
    """Read many histograms from the same ROOT file."""
    out: dict[str, dict[str, Any]] = {}
    for hp in hist_paths:
        out[str(hp)] = read_root_histogram(root_path, str(hp), flow_policy=flow_policy)
    return out
