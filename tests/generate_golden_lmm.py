#!/usr/bin/env python3
"""
Generate deterministic "golden" fixtures for LMM marginal likelihood models.

These fixtures are intended to be used by pure-Python sanity tests at runtime
(`tests/python/test_golden_lmm_fixtures.py`). The generator itself uses the
NextStat Python extension to compute MLE parameters at the optimum.

Usage:
  PYTHONPATH=bindings/ns-py/python python3 tests/generate_golden_lmm.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[0]
FIXTURES_DIR = ROOT / "fixtures" / "lmm"


def _fit_fixture(kind, x, y, group_idx, n_groups, include_intercept=True, random_slope_feature_idx=None):
    try:
        import nextstat as ns  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Unable to import nextstat. Run with:\n"
            "  PYTHONPATH=bindings/ns-py/python python3 tests/generate_golden_lmm.py"
        ) from e

    model = ns.LmmMarginalModel(
        x,
        y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        random_slope_feature_idx=random_slope_feature_idx,
    )
    mle = ns.MaximumLikelihoodEstimator()
    r = mle.fit(model)
    if not r.converged:
        raise SystemExit(f"fit did not converge for fixture {kind!r}")

    return {
        "kind": "lmm_marginal",
        "random_effects": kind,
        "include_intercept": include_intercept,
        "random_slope_feature_idx": random_slope_feature_idx,
        "n": len(y),
        "p": len(x[0]),
        "n_groups": n_groups,
        "x": x,
        "y": y,
        "group_idx": group_idx,
        "parameter_names": model.parameter_names(),
        "params_hat": r.parameters,
        "nll_at_hat": r.nll,
        "source": {
            "tool": "nextstat",
            "version": ns.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic small dataset (no RNG) for fixtures.
    n_groups = 3
    x1 = [-1.0, -0.2, 0.7, 1.4, -0.8, 0.0, 0.9, 1.6, -1.2, -0.1, 0.5, 1.1]
    group_idx = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    x = [[v] for v in x1]

    # Random intercept only.
    beta0 = 0.8
    beta1 = -0.4
    alpha = [0.5, -0.2, 0.1]
    eps = [0.05, -0.02, 0.01, 0.03, -0.04, 0.02, -0.01, 0.00, 0.06, -0.03, 0.02, -0.01]
    y = [beta0 + beta1 * x1[i] + alpha[group_idx[i]] + eps[i] for i in range(len(x1))]
    f0 = _fit_fixture("intercept", x, y, group_idx, n_groups, include_intercept=True)
    (FIXTURES_DIR / "lmm_intercept_small.json").write_text(
        json.dumps(f0, indent=2, sort_keys=True) + "\n"
    )

    # Random intercept + random slope (independent).
    beta0 = 1.2
    beta1 = 0.6
    alpha = [0.4, -0.3, 0.2]
    u = [0.15, -0.10, 0.05]
    eps = [0.02, -0.01, 0.03, 0.00, -0.02, 0.01, -0.03, 0.02, 0.01, -0.02, 0.00, 0.03]
    y2 = [
        beta0 + beta1 * x1[i] + alpha[group_idx[i]] + u[group_idx[i]] * x1[i] + eps[i]
        for i in range(len(x1))
    ]
    f1 = _fit_fixture(
        "intercept_slope",
        x,
        y2,
        group_idx,
        n_groups,
        include_intercept=True,
        random_slope_feature_idx=0,
    )
    (FIXTURES_DIR / "lmm_intercept_slope_small.json").write_text(
        json.dumps(f1, indent=2, sort_keys=True) + "\n"
    )

    print("Wrote:")
    print(" -", FIXTURES_DIR / "lmm_intercept_small.json")
    print(" -", FIXTURES_DIR / "lmm_intercept_slope_small.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

