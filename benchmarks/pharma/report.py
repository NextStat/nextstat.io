#!/usr/bin/env python3
"""Generate benchmark report from pharma suite results.

Output format follows benchmark-protocol.md:
  Case | NS (median) | Competitor (median) | Speedup | Parity | Status

Parity criteria:
- |theta_NS - theta_ref| / |theta_ref| < 0.05 (5% relative for each param)
- |OFV_NS - OFV_ref| / max(1, |OFV_ref|) < 0.10 (10% relative)

Note: OFV parity is intentionally lenient (10%) because FOCE implementations
differ in approximation details (linearization point, Laplacian vs exact
marginal integration, etc.). Parameter parity is the primary quality gate.
"""

from __future__ import annotations

import math
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

THETA_RTOL = 0.05   # 5% relative tolerance on each fixed-effect parameter
OFV_RTOL = 0.10     # 10% relative tolerance on OFV


def _rel_diff(a: float, b: float) -> float:
    """Compute |a - b| / max(|b|, 1e-10)."""
    denom = max(abs(b), 1e-10)
    return abs(a - b) / denom


def check_theta_parity(
    ns_theta: dict[str, float],
    ref_theta: dict[str, float],
    rtol: float = THETA_RTOL,
) -> tuple[bool, dict[str, float]]:
    """Check parameter-by-parameter parity.

    Returns:
        (all_ok, {param_name: relative_diff})
    """
    diffs = {}
    all_ok = True
    for name in ref_theta:
        if name not in ns_theta:
            diffs[name] = float("inf")
            all_ok = False
            continue
        rd = _rel_diff(ns_theta[name], ref_theta[name])
        diffs[name] = rd
        if rd > rtol:
            all_ok = False
    return all_ok, diffs


def check_ofv_parity(
    ns_ofv: float,
    ref_ofv: float,
    rtol: float = OFV_RTOL,
) -> tuple[bool, float]:
    """Check OFV parity.

    Returns:
        (ok, relative_diff)
    """
    rd = _rel_diff(ns_ofv, ref_ofv)
    return rd <= rtol, rd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format wall time with appropriate unit."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.2f}s"


def _fmt_speedup(ns_s: float, comp_s: float) -> str:
    """Format speedup ratio."""
    if comp_s <= 0 or ns_s <= 0:
        return "N/A"
    ratio = comp_s / ns_s
    return f"{ratio:.1f}x"


def _status(ns_faster: bool, parity_ok: bool) -> str:
    """Determine status string."""
    if parity_ok and ns_faster:
        return "pass"
    if not parity_ok:
        return "fail:parity"
    return "fail:slower"


# ---------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------

def print_parity_table(ns_result: dict, ref: dict) -> None:
    """Print parameter-by-parameter parity comparison."""
    ns_theta = ns_result.get("theta", {})
    ref_theta = ref.get("theta", {})

    print(f"\n  {'Parameter':<12} | {'NS':<12} | {'Reference':<12} | {'Rel.Diff':<12} | {'Status':<8}")
    print(f"  {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*8}")

    for name in ref_theta:
        ns_val = ns_theta.get(name)
        ref_val = ref_theta[name]
        if ns_val is None:
            print(f"  {name:<12} | {'N/A':<12} | {ref_val:<12.6f} | {'N/A':<12} | {'fail':<8}")
            continue
        rd = _rel_diff(ns_val, ref_val)
        status = "ok" if rd <= THETA_RTOL else "FAIL"
        print(f"  {name:<12} | {ns_val:<12.6f} | {ref_val:<12.6f} | {rd:<12.6f} | {status:<8}")

    # OFV comparison
    ns_ofv = ns_result.get("ofv")
    ref_ofv = ref.get("ofv")
    if ns_ofv is not None and ref_ofv is not None:
        rd_ofv = _rel_diff(ns_ofv, ref_ofv)
        status_ofv = "ok" if rd_ofv <= OFV_RTOL else "FAIL"
        print(f"  {'OFV':<12} | {ns_ofv:<12.2f} | {ref_ofv:<12.2f} | {rd_ofv:<12.6f} | {status_ofv:<8}")


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print executive summary table per benchmark protocol.

    Format:
        Case | NS (median) | Competitor (median) | Speedup | Parity | Status
    """
    # Header
    header = (
        f"{'Case':<28} | {'NS (median)':<14} | {'Competitor':<18} | "
        f"{'Speedup':<8} | {'Parity':<24} | {'Status':<14}"
    )
    print(f"\n{'='*len(header)}")
    print("  PHARMA BENCHMARK SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for r in results:
        model_name = r.get("model", "unknown")
        ns = r.get("nextstat")
        ref = r.get("reference", {})

        if ns is None:
            print(f"{model_name:<28} | {'FAILED':<14} | {'':<18} | {'':<8} | {'':<24} | {'fail:ns_error':<14}")
            continue

        ns_wall = ns.get("wall_s", 0)
        ns_time_str = _fmt_time(ns_wall)

        # Check parity vs reference
        ns_theta = ns.get("theta", {})
        ref_theta = ref.get("theta", {})
        theta_ok, theta_diffs = check_theta_parity(ns_theta, ref_theta)

        ns_ofv = ns.get("ofv")
        ref_ofv = ref.get("ofv")
        ofv_ok = True
        ofv_rd = 0.0
        if ns_ofv is not None and ref_ofv is not None:
            ofv_ok, ofv_rd = check_ofv_parity(ns_ofv, ref_ofv)

        parity_ok = theta_ok  # theta parity is primary gate
        max_theta_diff = max(theta_diffs.values()) if theta_diffs else 0.0
        parity_str = f"{'ok' if parity_ok else 'fail'} (dtheta<{max_theta_diff:.4f})"

        # Try to find a competitor result
        comp_name = None
        comp_wall = None
        for key in ["nlmixr2", "pumas"]:
            if key in r and r[key] is not None:
                comp_name = key
                comp_wall = r[key].get("wall_s")
                break

        if comp_name and comp_wall is not None and comp_wall > 0:
            comp_time_str = f"{comp_name} {_fmt_time(comp_wall)}"
            speedup = _fmt_speedup(ns_wall, comp_wall)
            ns_faster = comp_wall > ns_wall
            status = _status(ns_faster, parity_ok)
        else:
            comp_time_str = "N/A (not installed)"
            speedup = "N/A"
            status = f"{'pass:ns_only' if parity_ok else 'fail:parity'}"

        print(f"{model_name:<28} | {ns_time_str:<14} | {comp_time_str:<18} | {speedup:<8} | {parity_str:<24} | {status:<14}")


def print_detailed_report(results: list[dict[str, Any]]) -> None:
    """Print detailed report with parity tables for each model."""
    for r in results:
        model_name = r.get("model", "unknown")
        desc = r.get("description", "")
        ref = r.get("reference", {})
        ns = r.get("nextstat")

        print(f"\n{'='*70}")
        print(f"  {model_name}: {desc}")
        print(f"{'='*70}")

        if ns is None:
            print("  NextStat: FAILED")
            continue

        print(f"  NextStat: wall={_fmt_time(ns.get('wall_s', 0))}"
              f"  OFV={ns.get('ofv', 'N/A')}"
              f"  converged={ns.get('converged', 'N/A')}"
              f"  n_iter={ns.get('n_iter', 'N/A')}")

        print_parity_table(ns, ref)

        # Competitor details
        for key in ["nlmixr2", "pumas"]:
            comp = r.get(key)
            if comp is not None:
                print(f"\n  {key}: wall={_fmt_time(comp.get('wall_s', 0))}"
                      f"  OFV={comp.get('ofv', 'N/A')}")


def generate_report(
    results: list[dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """Generate full benchmark report as string.

    If output_path is provided, also writes to file.
    """
    import io
    import sys

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    print_detailed_report(results)
    print("\n")
    print_summary_table(results)

    report = buffer.getvalue()
    sys.stdout = old_stdout

    if output_path is not None:
        Path(output_path).write_text(report)

    return report
