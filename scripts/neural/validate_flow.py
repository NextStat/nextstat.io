#!/usr/bin/env python3
"""Validate a trained normalizing flow exported for NextStat.

Runs three checks on the ONNX models described by a ``flow_manifest.json``:

1. **Normalization** — Gauss-Legendre quadrature verifies ∫ p(x) dx ≈ 1.
2. **Probability Integral Transform (PIT)** — CDF of samples should be Uniform(0,1).
   Measured via Kolmogorov-Smirnov test.
3. **Closure** — Train/test NLL gap should be small (overfitting detector).

Usage
-----
python scripts/neural/validate_flow.py \\
    --manifest models/signal_flow/flow_manifest.json \\
    --data samples.npy \\
    [--context-data context.npy] \\
    [--n-quadrature 128] \\
    [--n-pit 10000] \\
    [--pit-alpha 0.01]

Requirements
------------
pip install numpy onnxruntime scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        m = json.load(f)
    if m.get("schema_version") != "nextstat_flow_v0":
        print(f"ERROR: unsupported schema_version: {m.get('schema_version')}", file=sys.stderr)
        sys.exit(1)
    return m


def make_session(manifest_dir: Path, model_file: str):
    import onnxruntime as ort

    return ort.InferenceSession(str(manifest_dir / model_file))


def log_prob(session, x: np.ndarray) -> np.ndarray:
    """Evaluate log p(x) via ONNX session. x is [batch, input_dim]."""
    return session.run(None, {"input": x.astype(np.float32)})[0]


# ---------------------------------------------------------------------------
# Check 1: Normalization via Gauss-Legendre quadrature
# ---------------------------------------------------------------------------
def gauss_legendre_nodes_weights(n: int):
    """Compute GL nodes and weights on [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def check_normalization(
    session,
    manifest: dict,
    n_quad: int,
    context_values: np.ndarray | None,
) -> tuple[float, bool]:
    """Returns (integral, passed)."""
    features = manifest["features"]
    support = np.array(manifest["support"])  # [features, 2]
    context_features = manifest.get("context_features", 0)

    nodes_1d, weights_1d = gauss_legendre_nodes_weights(n_quad)

    if features == 1:
        lo, hi = support[0]
        half = (hi - lo) / 2.0
        mid = (lo + hi) / 2.0
        x = (mid + half * nodes_1d).reshape(-1, 1)
        w = weights_1d * half

        if context_features > 0 and context_values is not None:
            ctx = np.tile(context_values.reshape(1, -1), (len(x), 1))
            inp = np.hstack([x, ctx])
        else:
            inp = x

        lp = log_prob(session, inp)
        integral = np.sum(np.exp(lp) * w)
    elif features == 2:
        n_per_dim = min(n_quad, 64)
        nodes_1d_2, weights_1d_2 = gauss_legendre_nodes_weights(n_per_dim)

        lo0, hi0 = support[0]
        lo1, hi1 = support[1]
        h0 = (hi0 - lo0) / 2.0
        m0 = (lo0 + hi0) / 2.0
        h1 = (hi1 - lo1) / 2.0
        m1 = (lo1 + hi1) / 2.0

        x0 = m0 + h0 * nodes_1d_2
        x1 = m1 + h1 * nodes_1d_2
        w0 = weights_1d_2 * h0
        w1 = weights_1d_2 * h1

        xx0, xx1 = np.meshgrid(x0, x1, indexing="ij")
        ww0, ww1 = np.meshgrid(w0, w1, indexing="ij")
        grid = np.column_stack([xx0.ravel(), xx1.ravel()])
        wgrid = (ww0 * ww1).ravel()

        if context_features > 0 and context_values is not None:
            ctx = np.tile(context_values.reshape(1, -1), (len(grid), 1))
            inp = np.hstack([grid, ctx])
        else:
            inp = grid

        lp = log_prob(session, inp)
        integral = np.sum(np.exp(lp) * wgrid)
    else:
        # Monte Carlo for >2D
        n_mc = 100_000
        samples = np.zeros((n_mc, features), dtype=np.float32)
        volume = 1.0
        for d in range(features):
            lo, hi = support[d]
            samples[:, d] = np.random.uniform(lo, hi, size=n_mc).astype(np.float32)
            volume *= hi - lo

        if context_features > 0 and context_values is not None:
            ctx = np.tile(context_values.reshape(1, -1), (n_mc, 1))
            inp = np.hstack([samples, ctx])
        else:
            inp = samples

        lp = log_prob(session, inp)
        integral = float(np.mean(np.exp(lp)) * volume)

    tol = 0.01
    passed = abs(integral - 1.0) < tol
    return float(integral), passed


# ---------------------------------------------------------------------------
# Check 2: PIT — Probability Integral Transform
# ---------------------------------------------------------------------------
def check_pit_1d(
    session_lp,
    session_sample,
    manifest: dict,
    n_samples: int,
    alpha: float,
    context_values: np.ndarray | None,
) -> tuple[float, float, bool]:
    """1-D PIT check. Returns (ks_stat, p_value, passed)."""
    from scipy import stats

    features = manifest["features"]
    context_features = manifest.get("context_features", 0)
    support = np.array(manifest["support"])

    if features != 1:
        return float("nan"), float("nan"), True

    # Generate samples from the flow
    input_dim = features + context_features
    z = np.random.randn(n_samples, input_dim).astype(np.float32)
    if context_features > 0 and context_values is not None:
        z[:, features:] = context_values.reshape(1, -1)

    samples = session_sample.run(None, {"input": z})[0]  # [n, features]
    x_samples = samples[:, 0]

    # Compute CDF via numerical integration at each sample point
    lo, hi = support[0]
    x_sorted = np.sort(x_samples)
    x_sorted = x_sorted[(x_sorted >= lo) & (x_sorted <= hi)]

    n_quad = 256
    nodes, weights = gauss_legendre_nodes_weights(n_quad)
    half = (hi - lo) / 2.0
    mid = (lo + hi) / 2.0
    quad_x = mid + half * nodes
    quad_w = weights * half

    if context_features > 0 and context_values is not None:
        ctx = np.tile(context_values.reshape(1, -1), (n_quad, 1))
        inp = np.hstack([quad_x.reshape(-1, 1), ctx])
    else:
        inp = quad_x.reshape(-1, 1)

    lp = log_prob(session_lp, inp)
    pdf_vals = np.exp(lp)

    # CDF at each quad point (cumulative sum approach)
    cdf_at_sorted = np.zeros(len(x_sorted))
    for i, xi in enumerate(x_sorted):
        mask = quad_x <= xi
        cdf_at_sorted[i] = np.sum(pdf_vals[mask] * quad_w[mask])

    cdf_at_sorted = np.clip(cdf_at_sorted, 0, 1)

    ks_stat, p_value = stats.kstest(cdf_at_sorted, "uniform")
    passed = p_value > alpha
    return float(ks_stat), float(p_value), passed


# ---------------------------------------------------------------------------
# Check 3: Closure — train/test NLL gap
# ---------------------------------------------------------------------------
def check_closure(
    session_lp,
    data: np.ndarray,
    context_data: np.ndarray | None,
    manifest: dict,
    test_frac: float = 0.2,
) -> tuple[float, float, float, bool]:
    """Returns (train_nll, test_nll, gap, passed)."""
    n = data.shape[0]
    context_features = manifest.get("context_features", 0)
    perm = np.random.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    def mean_nll(idx):
        x = data[idx]
        if context_features > 0 and context_data is not None:
            c = context_data[idx]
            inp = np.hstack([x, c]).astype(np.float32)
        else:
            inp = x.astype(np.float32)
        lp = log_prob(session_lp, inp)
        return -float(np.mean(lp))

    train_nll = mean_nll(train_idx)
    test_nll = mean_nll(test_idx)
    gap = test_nll - train_nll
    max_gap = 0.5
    passed = gap < max_gap
    return train_nll, test_nll, gap, passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a NextStat normalizing flow")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to flow_manifest.json")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Training data (.npy) for closure check",
    )
    parser.add_argument(
        "--context-data",
        type=Path,
        default=None,
        help="Context data (.npy) for conditional flows",
    )
    parser.add_argument("--n-quadrature", type=int, default=128, help="Quadrature nodes (default: 128)")
    parser.add_argument("--n-pit", type=int, default=5000, help="PIT samples (default: 5000)")
    parser.add_argument("--pit-alpha", type=float, default=0.01, help="PIT KS test alpha (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)

    manifest = load_manifest(args.manifest)
    manifest_dir = args.manifest.parent
    features = manifest["features"]
    context_features = manifest.get("context_features", 0)

    print(f"Flow: {features}D, {context_features} context features")
    print(f"Observables: {manifest['observable_names']}")
    print(f"Support: {manifest['support']}")

    session_lp = make_session(manifest_dir, manifest["models"]["log_prob"])
    session_sample = None
    if manifest["models"].get("sample"):
        session_sample = make_session(manifest_dir, manifest["models"]["sample"])

    context_values = None
    if context_features > 0 and args.context_data is not None:
        context_values = np.load(args.context_data).astype(np.float32)
        if context_values.ndim == 1:
            context_values = context_values[:1]
        else:
            context_values = context_values[0]

    all_passed = True
    print()

    # --- Check 1: Normalization ---
    print("=" * 50)
    print("CHECK 1: Normalization (quadrature)")
    integral, norm_ok = check_normalization(session_lp, manifest, args.n_quadrature, context_values)
    status = "PASS" if norm_ok else "FAIL"
    print(f"  ∫ p(x) dx = {integral:.6f}  (expected ≈ 1.0)")
    print(f"  [{status}]")
    if not norm_ok:
        all_passed = False
    print()

    # --- Check 2: PIT ---
    if session_sample is not None and features == 1:
        print("=" * 50)
        print("CHECK 2: PIT (Kolmogorov-Smirnov)")
        ks_stat, p_value, pit_ok = check_pit_1d(
            session_lp, session_sample, manifest, args.n_pit, args.pit_alpha, context_values
        )
        status = "PASS" if pit_ok else "FAIL"
        print(f"  KS statistic = {ks_stat:.4f}")
        print(f"  p-value      = {p_value:.4f}  (alpha = {args.pit_alpha})")
        print(f"  [{status}]")
        if not pit_ok:
            all_passed = False
        print()
    else:
        print("=" * 50)
        print("CHECK 2: PIT — SKIPPED (no sample model or >1D)")
        print()

    # --- Check 3: Closure ---
    if args.data is not None:
        print("=" * 50)
        print("CHECK 3: Closure (train/test NLL gap)")
        data = np.load(args.data).astype(np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        context_data = None
        if args.context_data is not None:
            context_data = np.load(args.context_data).astype(np.float32)
            if context_data.ndim == 1:
                context_data = context_data.reshape(-1, 1)

        train_nll, test_nll, gap, closure_ok = check_closure(
            session_lp, data, context_data, manifest
        )
        status = "PASS" if closure_ok else "FAIL"
        print(f"  Train NLL = {train_nll:.4f}")
        print(f"  Test  NLL = {test_nll:.4f}")
        print(f"  Gap       = {gap:.4f}  (threshold: 0.5)")
        print(f"  [{status}]")
        if not closure_ok:
            all_passed = False
        print()
    else:
        print("=" * 50)
        print("CHECK 3: Closure — SKIPPED (no --data provided)")
        print()

    # --- Summary ---
    print("=" * 50)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review flow training before using in a fit")
        sys.exit(1)


if __name__ == "__main__":
    main()
