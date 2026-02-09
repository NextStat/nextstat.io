"""Sanity checks for LMM (marginal likelihood) golden fixtures.

These tests validate fixture internal consistency without relying on the Rust
extension module at runtime.

Notes:
- The fixture convention matches `crates/ns-inference/src/lmm.rs`.
- NLL is up to an additive constant (no 0.5 * n * log(2*pi)).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "lmm"


def _row_dot(row, beta):
    return sum(x * b for x, b in zip(row, beta))


def _lmm_nll_and_fixed_grad(x, y, group_idx, n_groups, params, include_intercept, random_slope_feature_idx):
    n = len(y)
    p = len(x[0])
    assert len(group_idx) == n

    beta_offset = 1 if include_intercept else 0
    nb = p + beta_offset
    beta = params[:nb]
    log_sigma_y = float(params[nb])
    log_tau_alpha = float(params[nb + 1])
    log_tau_u = float(params[nb + 2]) if random_slope_feature_idx is not None else None

    sigma_y = math.exp(log_sigma_y)
    tau_alpha = math.exp(log_tau_alpha)
    tau_u = math.exp(log_tau_u) if log_tau_u is not None else None

    inv_a = 1.0 / (sigma_y * sigma_y)
    log_a = math.log(sigma_y * sigma_y)

    # Group indices
    groups = [[] for _ in range(n_groups)]
    for i, g in enumerate(group_idx):
        groups[int(g)].append(i)

    nll = 0.0
    grad = [0.0] * nb

    for g in range(n_groups):
        idxs = groups[g]
        m = len(idxs)
        if m == 0:
            continue

        # Residuals and sufficient stats
        sum_r2 = 0.0
        t0 = 0.0
        t1 = 0.0
        s01 = 0.0
        s11 = 0.0
        r_cache = []

        for i in idxs:
            row = x[i]
            eta = beta[0] + _row_dot(row, beta[1:]) if include_intercept else _row_dot(row, beta)
            r = y[i] - eta
            r_cache.append((i, r))
            sum_r2 += r * r
            t0 += r

            if random_slope_feature_idx is not None:
                xk = row[int(random_slope_feature_idx)]
                t1 += xk * r
                s01 += xk
                s11 += xk * xk

        if random_slope_feature_idx is None:
            inv_tau2 = 1.0 / (tau_alpha * tau_alpha)
            m11 = inv_tau2 + inv_a * float(m)
            log_det_m = math.log(m11)
            u0 = (inv_a * t0) / m11
            quad_correction = (inv_a * t0) * u0

            log_det_v = float(m) * log_a + math.log(tau_alpha * tau_alpha) + log_det_m
            quad = inv_a * sum_r2 - quad_correction
            nll += 0.5 * (log_det_v + quad)

            # Fixed-effects gradient via v = inv_a*r - inv_a*(Z u)
            for i, r in r_cache:
                row = x[i]
                z_dot_u = u0
                v = inv_a * r - inv_a * z_dot_u
                if include_intercept:
                    grad[0] += -v
                    grad[1] += -v * row[0]
                else:
                    grad[0] += -v * row[0]
        else:
            assert tau_u is not None
            inv_tau_a2 = 1.0 / (tau_alpha * tau_alpha)
            inv_tau_u2 = 1.0 / (tau_u * tau_u)
            s00 = float(m)
            m00 = inv_tau_a2 + inv_a * s00
            m01 = inv_a * s01
            m11 = inv_tau_u2 + inv_a * s11
            det = m00 * m11 - m01 * m01
            if not (det > 0.0) or not math.isfinite(det):
                raise AssertionError("invalid determinant")
            log_det_m = math.log(det)
            b0 = inv_a * t0
            b1 = inv_a * t1
            u0 = (m11 * b0 - m01 * b1) / det
            u1 = (-m01 * b0 + m00 * b1) / det
            quad_correction = b0 * u0 + b1 * u1

            log_det_v = (
                float(m) * log_a
                + math.log(tau_alpha * tau_alpha)
                + math.log(tau_u * tau_u)
                + log_det_m
            )
            quad = inv_a * sum_r2 - quad_correction
            nll += 0.5 * (log_det_v + quad)

            for i, r in r_cache:
                row = x[i]
                xk = row[int(random_slope_feature_idx)]
                z_dot_u = u0 + xk * u1
                v = inv_a * r - inv_a * z_dot_u
                if include_intercept:
                    grad[0] += -v
                    grad[1] += -v * row[0]
                else:
                    grad[0] += -v * row[0]

    return nll, grad


def test_golden_lmm_fixtures_exist():
    assert FIXTURES_DIR.is_dir()
    names = {p.name for p in FIXTURES_DIR.glob("*.json")}
    assert {"lmm_intercept_small.json", "lmm_intercept_slope_small.json"} <= names


def test_golden_lmm_fixtures_nll_matches_recompute():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        assert data["kind"] == "lmm_marginal"
        x = data["x"]
        y = data["y"]
        group_idx = data["group_idx"]
        n_groups = int(data["n_groups"])
        include_intercept = bool(data["include_intercept"])
        re = data["random_effects"]
        k = data.get("random_slope_feature_idx")
        if re == "intercept":
            k = None
        nll, _ = _lmm_nll_and_fixed_grad(
            x=x,
            y=y,
            group_idx=group_idx,
            n_groups=n_groups,
            params=data["params_hat"],
            include_intercept=include_intercept,
            random_slope_feature_idx=k,
        )
        assert abs(nll - float(data["nll_at_hat"])) < 1e-8, f"{path.name}: nll mismatch"


def test_golden_lmm_fixtures_have_near_zero_fixed_effects_gradients():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        x = data["x"]
        y = data["y"]
        group_idx = data["group_idx"]
        n_groups = int(data["n_groups"])
        include_intercept = bool(data["include_intercept"])
        re = data["random_effects"]
        k = data.get("random_slope_feature_idx")
        if re == "intercept":
            k = None
        _, g = _lmm_nll_and_fixed_grad(
            x=x,
            y=y,
            group_idx=group_idx,
            n_groups=n_groups,
            params=data["params_hat"],
            include_intercept=include_intercept,
            random_slope_feature_idx=k,
        )
        g_inf = max(abs(v) for v in g) if g else 0.0
        # L-BFGS termination tolerances + float noise: keep this loose but meaningful.
        assert g_inf < 1e-5, f"{path.name}: fixed-effects gradient too large: {g_inf}"


def test_golden_lmm_external_reference_matches_params_hat_when_present():
    """If a fixture includes external_reference, require it to agree with params_hat.

    This enables committing precomputed lme4/Stan numbers while keeping test-time
    dependencies to the Python stdlib only.
    """

    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        ext = data.get("external_reference")
        if not isinstance(ext, dict):
            continue
        est = ext.get("estimates")
        if not isinstance(est, dict):
            raise AssertionError(f"{path.name}: external_reference.estimates missing")

        names = data["parameter_names"]
        params = list(map(float, data["params_hat"]))
        if len(names) != len(params):
            raise AssertionError(f"{path.name}: parameter_names/params_hat length mismatch")

        # Build an external vector in the same order. Missing tau_u is allowed for intercept-only.
        ext_vec = []
        for nm in names:
            if nm not in est:
                raise AssertionError(f"{path.name}: external estimate missing {nm!r}")
            ext_vec.append(float(est[nm]))

        # Tolerance: external tools may differ slightly on small problems.
        # This is a golden/parity check, not a strict numerical identity.
        tol = 5e-3
        for nm, a, b in zip(names, params, ext_vec):
            if abs(a - b) > tol:
                raise AssertionError(f"{path.name}: {nm} mismatch: {a} vs {b} (tol={tol})")
