#!/usr/bin/env python3
"""
Validate NextStat NLL against pyhf reference implementation.

This script computes NLL values using pyhf for our test fixtures,
and compares them against NextStat Rust implementation.

Run with: python tests/validate_pyhf_nll.py
Requires: pip install -e bindings/ns-py[validation]
"""

import argparse
import json
from pathlib import Path
import time
from collections import defaultdict
from contextlib import contextmanager
import pyhf
import nextstat


class _Timing:
    def __init__(self) -> None:
        self.totals_s: dict[str, float] = defaultdict(float)

    @contextmanager
    def time(self, label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals_s[str(label)] += time.perf_counter() - t0

    def print_summary(self) -> None:
        if not self.totals_s:
            return
        print("\n" + "-" * 70)
        print("Timing breakdown (seconds)")
        print("-" * 70)
        for k, v in sorted(self.totals_s.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{k:<28} {v:>12.6f}")
        print("-" * 70)
        total = sum(self.totals_s.values())
        print(f"{'total':<28} {total:>12.6f}")

    def sum_prefix(self, prefix: str) -> float:
        return sum(v for k, v in self.totals_s.items() if str(k).startswith(prefix))

    def print_pyhf_vs_nextstat(self) -> None:
        pyhf_total = self.sum_prefix("pyhf:")
        ns_total = self.sum_prefix("nextstat:")
        if pyhf_total <= 0.0 and ns_total <= 0.0:
            return
        speedup = pyhf_total / max(ns_total, 1e-12)
        print("\nTiming summary (reference vs NextStat)")
        print(f"  reference (pyhf): {pyhf_total:.6f} s")
        print(f"  NextStat:         {ns_total:.6f} s")
        print(f"  speedup:          {speedup:.2f}x")


def load_fixture(name: str) -> dict:
    """Load a test fixture JSON file."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / name) as f:
        return json.load(f)


def compute_pyhf_nll(workspace: dict, measurement_name: str, params: list[float], *, timing: _Timing) -> float:
    """
    Compute NLL using pyhf for given workspace and parameters.

    Args:
        workspace: pyhf workspace dict
        measurement_name: name of measurement to use
        params: parameter values (ordered as pyhf expects)

    Returns:
        NLL value (negative log-likelihood)
    """
    # Create pyhf model
    with timing.time("pyhf:workspace+model"):
        ws = pyhf.Workspace(workspace)
        model = ws.model(
            measurement_name=measurement_name,
            modifier_settings={
                'normsys': {'interpcode': 'code4'},
                'histosys': {'interpcode': 'code4p'},
            }
        )

    # Get observations from workspace
    with timing.time("pyhf:data"):
        observations = ws.data(model)

    # Compute NLL (pyhf returns -2*log(L), we want -log(L))
    with timing.time("pyhf:twice_nll"):
        twice_nll = pyhf.infer.mle.twice_nll(params, observations, model)

    # Extract scalar value if numpy array
    if hasattr(twice_nll, 'item'):
        twice_nll = twice_nll.item()

    nll = twice_nll / 2.0

    return nll


def compute_nextstat_nll(workspace: dict, pyhf_model, pyhf_params: list[float], *, timing: _Timing) -> float:
    """
    Compute NLL using NextStat for given workspace and pyhf parameters.

    This maps `pyhf` parameter order → NextStat parameter order by name.
    """
    with timing.time("nextstat:from_workspace"):
        model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_names = model.parameter_names()
        ns_init = model.suggested_init()
        ns_index = {name: i for i, name in enumerate(ns_names)}

    ns_params = list(ns_init)
    for name, value in zip(pyhf_model.config.par_names, pyhf_params):
        if name not in ns_index:
            raise RuntimeError(f"NextStat model missing parameter '{name}'")
        ns_params[ns_index[name]] = float(value)

    with timing.time("nextstat:nll"):
        return float(model.nll(ns_params))


def validate_simple_workspace():
    """Validate simple_workspace.json fixture."""
    timing = _Timing()
    print("=" * 70)
    print("Validating: simple_workspace.json")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")

    # Build pyhf model and take suggested init (ensures correct order)
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name="GaussExample",
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    params_nominal = model.config.suggested_init()

    nll_nominal = compute_pyhf_nll(workspace, "GaussExample", params_nominal, timing=timing)
    nll_ns_nominal = compute_nextstat_nll(workspace, model, params_nominal, timing=timing)
    print(f"\nNLL at nominal (mu=1.0, gammas=1.0): {nll_nominal:.10f}")
    print(f"NextStat NLL at nominal:               {nll_ns_nominal:.10f}")
    print(f"Diff:                                  {abs(nll_ns_nominal - nll_nominal):.3e}")

    # Varied POI
    poi_idx = model.config.poi_index
    params_mu_0 = params_nominal.copy()
    params_mu_0[poi_idx] = 0.0
    params_mu_2 = params_nominal.copy()
    params_mu_2[poi_idx] = 2.0

    nll_mu_0 = compute_pyhf_nll(workspace, "GaussExample", params_mu_0, timing=timing)
    nll_mu_2 = compute_pyhf_nll(workspace, "GaussExample", params_mu_2, timing=timing)
    nll_ns_mu_0 = compute_nextstat_nll(workspace, model, params_mu_0, timing=timing)
    nll_ns_mu_2 = compute_nextstat_nll(workspace, model, params_mu_2, timing=timing)

    print(f"NLL at mu=0.0: {nll_mu_0:.10f}")
    print(f"NLL at mu=2.0: {nll_mu_2:.10f}")
    print(f"NextStat NLL at mu=0.0: {nll_ns_mu_0:.10f}")
    print(f"NextStat NLL at mu=2.0: {nll_ns_mu_2:.10f}")

    print("\n✓ Expected behavior:")
    print(f"  - NLL values should be finite")
    print(f"  - NLL changes with POI: {nll_mu_0 != nll_nominal != nll_mu_2}")
    timing.print_summary()
    timing.print_pyhf_vs_nextstat()
    return {
        "fixture": "simple_workspace.json",
        "nll": {
            "pyhf_nominal": float(nll_nominal),
            "nextstat_nominal": float(nll_ns_nominal),
            "abs_diff_nominal": float(abs(nll_ns_nominal - nll_nominal)),
            "pyhf_mu0": float(nll_mu_0),
            "pyhf_mu2": float(nll_mu_2),
            "nextstat_mu0": float(nll_ns_mu_0),
            "nextstat_mu2": float(nll_ns_mu_2),
        },
        "timing_s": {k: float(v) for k, v in timing.totals_s.items()},
        "timing_summary_s": {
            "reference_total": float(timing.sum_prefix("pyhf:")),
            "nextstat_total": float(timing.sum_prefix("nextstat:")),
        },
    }


def validate_complex_workspace():
    """Validate complex_workspace.json fixture."""
    timing = _Timing()
    print("\n" + "=" * 70)
    print("Validating: complex_workspace.json")
    print("=" * 70)

    workspace = load_fixture("complex_workspace.json")

    # Get model to determine parameter count
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name="measurement",
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        }
    )

    print(f"\nModel parameters: {model.config.par_names}")
    print(f"Number of parameters: {model.config.npars}")

    # At nominal (all parameters at suggested init values)
    params_nominal = model.config.suggested_init()
    nll_nominal = compute_pyhf_nll(workspace, "measurement", params_nominal, timing=timing)
    nll_ns_nominal = compute_nextstat_nll(workspace, model, params_nominal, timing=timing)

    print(f"\nNLL at nominal: {nll_nominal:.10f}")
    print(f"NextStat NLL at nominal: {nll_ns_nominal:.10f}")
    print(f"Diff: {abs(nll_ns_nominal - nll_nominal):.3e}")
    print(f"Nominal parameters: {params_nominal}")

    # Test with POI varied
    params_mu_0 = params_nominal.copy()
    params_mu_0[model.config.poi_index] = 0.0

    params_mu_2 = params_nominal.copy()
    params_mu_2[model.config.poi_index] = 2.0

    nll_mu_0 = compute_pyhf_nll(workspace, "measurement", params_mu_0, timing=timing)
    nll_mu_2 = compute_pyhf_nll(workspace, "measurement", params_mu_2, timing=timing)
    nll_ns_mu_0 = compute_nextstat_nll(workspace, model, params_mu_0, timing=timing)
    nll_ns_mu_2 = compute_nextstat_nll(workspace, model, params_mu_2, timing=timing)

    print(f"NLL at mu=0.0: {nll_mu_0:.10f}")
    print(f"NLL at mu=2.0: {nll_mu_2:.10f}")
    print(f"NextStat NLL at mu=0.0: {nll_ns_mu_0:.10f}")
    print(f"NextStat NLL at mu=2.0: {nll_ns_mu_2:.10f}")
    timing.print_summary()
    timing.print_pyhf_vs_nextstat()
    return {
        "fixture": "complex_workspace.json",
        "nll": {
            "pyhf_nominal": float(nll_nominal),
            "nextstat_nominal": float(nll_ns_nominal),
            "abs_diff_nominal": float(abs(nll_ns_nominal - nll_nominal)),
            "pyhf_mu0": float(nll_mu_0),
            "pyhf_mu2": float(nll_mu_2),
            "nextstat_mu0": float(nll_ns_mu_0),
            "nextstat_mu2": float(nll_ns_mu_2),
        },
        "timing_s": {k: float(v) for k, v in timing.totals_s.items()},
        "timing_summary_s": {
            "reference_total": float(timing.sum_prefix("pyhf:")),
            "nextstat_total": float(timing.sum_prefix("nextstat:")),
        },
    }


def main():
    """Run all validation tests."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (machine-readable summary, includes timings).",
    )
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("pyhf NLL Validation - Reference Values")
    print("=" * 70)
    print("\nThese values should match NextStat Rust implementation")
    print()

    try:
        t0 = time.perf_counter()
        cases = [
            validate_simple_workspace(),
            validate_complex_workspace(),
        ]
        wall_s = time.perf_counter() - t0

        print("\n" + "=" * 70)
        print("✓ Validation complete!")
        print("=" * 70)

        if args.out is not None:
            ref_total = float(sum(c["timing_summary_s"]["reference_total"] for c in cases))
            ns_total = float(sum(c["timing_summary_s"]["nextstat_total"] for c in cases))
            report = {
                "schema_version": "validate_pyhf_nll_v1",
                "status": "ok",
                "timing_summary_s": {
                    "reference_total": ref_total,
                    "nextstat_total": ns_total,
                    "speedup_reference_over_nextstat": float(ref_total / max(ns_total, 1e-12)),
                    "wall_total": float(wall_s),
                },
                "cases": cases,
            }
            if args.out.parent and not args.out.parent.exists():
                args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote: {args.out}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
