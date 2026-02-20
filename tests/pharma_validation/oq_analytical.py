"""OQ (Operational Qualification) test cases -- analytical correctness.

Implements OQ test cases from the IQ/OQ/PQ protocol (NS-VAL-001 v2.0.0,
Sections 3.1--3.4).  Tests cover:
  - Section 3.1: PK analytical C(t), NLL, gradient, constructor correctness
  - Section 3.2: Error models (additive, proportional, combined)
  - Section 3.3: LLOQ handling (ignore, replace_half, censored)
  - Section 3.4: MLE estimation (suggested_init, suggested_bounds, gradient)
"""
from __future__ import annotations

import math
import time
from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_result(
    test_id: str,
    section: str,
    title: str,
    ok: bool | None,
    observed: dict[str, Any],
    acceptance: str,
    deviation: str | None,
    wall_s: float,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "category": "OQ",
        "section": section,
        "title": title,
        "ok": ok,
        "observed": observed,
        "acceptance": acceptance,
        "deviation": deviation,
        "wall_s": round(wall_s, 6),
    }


def _rel_err(a: float, b: float) -> float:
    """Relative error between *a* and *b*."""
    return abs(a - b) / max(abs(a), 1e-30)


# ===================================================================
# Section 3.1 -- PK Analytical Correctness (8 tests)
# ===================================================================

def _oq_pk_001() -> dict[str, Any]:
    """1-cpt oral predict -- analytical C(t) correctness."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        F = 1.0
        CL = 2.5
        V = 35.0
        Ka = 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 24.0]

        # Pure-Python analytical formula
        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        expected = [c_analytical(t) for t in times]

        # Use dummy observations (value does not matter for predict)
        dummy_y = [1.0] * len(times)
        model = OneCompartmentOralPkModel(
            times, dummy_y, dose=dose, bioavailability=F, sigma=0.05,
        )
        predicted = model.predict([CL, V, Ka])

        max_rel = 0.0
        for e, p in zip(expected, predicted):
            max_rel = max(max_rel, _rel_err(e, p))

        ok = max_rel < 1e-6
        observed = {"max_rel_err": max_rel, "n_points": len(times)}
        deviation = None if ok else f"max_rel_err={max_rel:.3e} >= 1e-6"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-001",
        section="3.1",
        title="1-cpt oral predict -- analytical C(t) correctness",
        ok=ok,
        observed=observed,
        acceptance="Max relative error < 1e-6 vs analytical formula",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_002() -> dict[str, Any]:
    """1-cpt oral NLL finite -- NLL at true params is finite."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL = 2.5
        V = 35.0
        Ka = 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 24.0]

        # Generate synthetic observations from known params
        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_exact = [c_analytical(t) for t in times]
        # Add small perturbation so residuals are non-zero
        y_obs = [c * 1.01 for c in y_exact]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.05,
        )
        nll_val = model.nll([CL, V, Ka])

        ok = math.isfinite(nll_val)
        observed = {"nll": nll_val, "finite": math.isfinite(nll_val)}
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-002",
        section="3.1",
        title="1-cpt oral NLL finite",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_003() -> dict[str, Any]:
    """1-cpt oral gradient -- grad_nll returns correct length, all finite."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL = 2.5
        V = 35.0
        Ka = 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.01 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.05,
        )
        grad = model.grad_nll([CL, V, Ka])

        correct_length = len(grad) == 3  # CL, V, Ka
        all_finite = all(math.isfinite(g) for g in grad)
        ok = correct_length and all_finite
        observed = {
            "grad_length": len(grad),
            "expected_length": 3,
            "all_finite": all_finite,
            "grad_values": grad,
        }
        parts = []
        if not correct_length:
            parts.append(f"length={len(grad)} != 3")
        if not all_finite:
            parts.append("not all finite")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-003",
        section="3.1",
        title="1-cpt oral gradient -- correct length (3) and all finite",
        ok=ok,
        observed=observed,
        acceptance="grad_nll returns 3 finite values (CL, V, Ka)",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_004() -> dict[str, Any]:
    """2-cpt IV predict -- analytical C(t) correctness."""
    t0 = time.monotonic()
    try:
        from nextstat._core import TwoCompartmentIvPkModel  # type: ignore[import-untyped]

        dose = 1000.0
        CL = 10.0
        V1 = 20.0
        V2 = 30.0
        Q = 5.0
        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        # Bi-exponential analytical formula
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2
        a = k10 + k12 + k21
        alpha = 0.5 * (a + math.sqrt(a * a - 4.0 * k10 * k21))
        beta = 0.5 * (a - math.sqrt(a * a - 4.0 * k10 * k21))
        A_coeff = (dose / V1) * (alpha - k21) / (alpha - beta)
        B_coeff = (dose / V1) * (k21 - beta) / (alpha - beta)

        expected = [A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t)
                    for t in times]

        dummy_y = [1.0] * len(times)
        model = TwoCompartmentIvPkModel(
            times, dummy_y, dose=dose, sigma=0.05,
        )
        predicted = model.predict([CL, V1, V2, Q])

        max_rel = 0.0
        for e, p in zip(expected, predicted):
            max_rel = max(max_rel, _rel_err(e, p))

        ok = max_rel < 1e-4
        observed = {"max_rel_err": max_rel, "n_points": len(times)}
        deviation = None if ok else f"max_rel_err={max_rel:.3e} >= 1e-4"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-004",
        section="3.1",
        title="2-cpt IV predict -- analytical C(t) correctness",
        ok=ok,
        observed=observed,
        acceptance="Max relative error < 1e-4 vs bi-exponential formula",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_005() -> dict[str, Any]:
    """2-cpt oral predict -- analytical C(t) correctness."""
    t0 = time.monotonic()
    try:
        from nextstat._core import TwoCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 500.0
        F = 1.0
        CL = 2.0
        V1 = 20.0
        V2 = 30.0
        Q = 3.0
        Ka = 1.5
        times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 12.0, 24.0]

        # Micro-constants
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2
        s = k10 + k12 + k21
        disc = math.sqrt(s * s - 4.0 * k10 * k21)
        alpha = 0.5 * (s + disc)
        beta = 0.5 * (s - disc)

        # Three-exponential solution for 2-cpt oral
        pref = Ka * F * dose / V1

        def c_analytical(t: float) -> float:
            denom_a = (Ka - alpha) * (beta - alpha)
            denom_b = (Ka - beta) * (alpha - beta)
            denom_c = (alpha - Ka) * (beta - Ka)
            ta = (k21 - alpha) / denom_a * math.exp(-alpha * t)
            tb = (k21 - beta) / denom_b * math.exp(-beta * t)
            tc = (k21 - Ka) / denom_c * math.exp(-Ka * t)
            return pref * (ta + tb + tc)

        expected = [c_analytical(t) for t in times]

        dummy_y = [max(c, 0.001) for c in expected]
        model = TwoCompartmentOralPkModel(
            times, dummy_y, dose=dose, bioavailability=F, sigma=0.05,
        )
        predicted = model.predict([CL, V1, V2, Q, Ka])

        max_rel = 0.0
        for e, p in zip(expected, predicted):
            if abs(e) < 1e-30:
                continue
            max_rel = max(max_rel, _rel_err(e, p))

        ok = max_rel < 1e-4
        observed = {"max_rel_err": max_rel, "n_points": len(times)}
        deviation = None if ok else f"max_rel_err={max_rel:.3e} >= 1e-4"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-005",
        section="3.1",
        title="2-cpt oral predict -- analytical C(t) correctness",
        ok=ok,
        observed=observed,
        acceptance="Max relative error < 1e-4 vs three-exponential formula",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_006() -> dict[str, Any]:
    """3-cpt IV constructor -- n_params()=6, parameter_names() length=6."""
    t0 = time.monotonic()
    try:
        from nextstat._core import ThreeCompartmentIvPkModel  # type: ignore[import-untyped]

        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        dummy_y = [1.0] * len(times)
        model = ThreeCompartmentIvPkModel(
            times, dummy_y, dose=1000.0, sigma=0.05,
        )

        n_params = model.n_params()
        param_names = model.parameter_names()
        n_names = len(param_names)

        ok = n_params == 6 and n_names == 6
        observed = {
            "n_params": n_params,
            "parameter_names": param_names,
            "n_names": n_names,
        }
        parts = []
        if n_params != 6:
            parts.append(f"n_params={n_params} != 6")
        if n_names != 6:
            parts.append(f"len(parameter_names)={n_names} != 6")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-006",
        section="3.1",
        title="3-cpt IV constructor -- n_params=6, parameter_names length=6",
        ok=ok,
        observed=observed,
        acceptance="n_params()=6 and len(parameter_names())=6",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_007() -> dict[str, Any]:
    """3-cpt oral constructor -- n_params()=7, parameter_names() length=7."""
    t0 = time.monotonic()
    try:
        from nextstat._core import ThreeCompartmentOralPkModel  # type: ignore[import-untyped]

        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        dummy_y = [1.0] * len(times)
        model = ThreeCompartmentOralPkModel(
            times, dummy_y, dose=1000.0, sigma=0.05,
        )

        n_params = model.n_params()
        param_names = model.parameter_names()
        n_names = len(param_names)

        ok = n_params == 7 and n_names == 7
        observed = {
            "n_params": n_params,
            "parameter_names": param_names,
            "n_names": n_names,
        }
        parts = []
        if n_params != 7:
            parts.append(f"n_params={n_params} != 7")
        if n_names != 7:
            parts.append(f"len(parameter_names)={n_names} != 7")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-007",
        section="3.1",
        title="3-cpt oral constructor -- n_params=7, parameter_names length=7",
        ok=ok,
        observed=observed,
        acceptance="n_params()=7 and len(parameter_names())=7",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_pk_008() -> dict[str, Any]:
    """Parameter names consistency -- each PK model returns expected names."""
    t0 = time.monotonic()
    try:
        from nextstat._core import (  # type: ignore[import-untyped]
            OneCompartmentOralPkModel,
            TwoCompartmentIvPkModel,
            TwoCompartmentOralPkModel,
            ThreeCompartmentIvPkModel,
            ThreeCompartmentOralPkModel,
        )

        times = [1.0, 2.0, 4.0, 8.0]
        dummy_y = [1.0] * len(times)

        # Build each model and collect parameter_names
        models_info: list[dict[str, Any]] = []

        m1 = OneCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.05)
        names_1 = m1.parameter_names()
        np_1 = m1.n_params()
        models_info.append({
            "class": "OneCompartmentOralPkModel",
            "n_params": np_1,
            "parameter_names": names_1,
            "names_match_nparams": len(names_1) == np_1,
        })

        m2 = TwoCompartmentIvPkModel(times, dummy_y, dose=100.0, sigma=0.05)
        names_2 = m2.parameter_names()
        np_2 = m2.n_params()
        models_info.append({
            "class": "TwoCompartmentIvPkModel",
            "n_params": np_2,
            "parameter_names": names_2,
            "names_match_nparams": len(names_2) == np_2,
        })

        m3 = TwoCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.05)
        names_3 = m3.parameter_names()
        np_3 = m3.n_params()
        models_info.append({
            "class": "TwoCompartmentOralPkModel",
            "n_params": np_3,
            "parameter_names": names_3,
            "names_match_nparams": len(names_3) == np_3,
        })

        m4 = ThreeCompartmentIvPkModel(times, dummy_y, dose=100.0, sigma=0.05)
        names_4 = m4.parameter_names()
        np_4 = m4.n_params()
        models_info.append({
            "class": "ThreeCompartmentIvPkModel",
            "n_params": np_4,
            "parameter_names": names_4,
            "names_match_nparams": len(names_4) == np_4,
        })

        m5 = ThreeCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.05)
        names_5 = m5.parameter_names()
        np_5 = m5.n_params()
        models_info.append({
            "class": "ThreeCompartmentOralPkModel",
            "n_params": np_5,
            "parameter_names": names_5,
            "names_match_nparams": len(names_5) == np_5,
        })

        all_match = all(m["names_match_nparams"] for m in models_info)
        ok = all_match
        observed = {"models": models_info, "all_match": all_match}
        mismatches = [m["class"] for m in models_info if not m["names_match_nparams"]]
        deviation = (
            f"Mismatch in: {', '.join(mismatches)}" if mismatches else None
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-PK-008",
        section="3.1",
        title="Parameter names consistency across all PK model classes",
        ok=ok,
        observed=observed,
        acceptance="len(parameter_names()) == n_params() for all 5 models",
        deviation=deviation,
        wall_s=wall_s,
    )


# ===================================================================
# Section 3.2 -- Error Models (5 tests)
# ===================================================================

def _oq_err_001() -> dict[str, Any]:
    """Additive error -- NLL finite with default additive error model."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        sigma = 0.5
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.02 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
        )
        nll_val = model.nll([CL, V, Ka])

        ok = math.isfinite(nll_val)
        observed = {"nll": nll_val, "sigma": sigma, "finite": math.isfinite(nll_val)}
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-ERR-001",
        section="3.2",
        title="Additive error -- NLL finite with sigma=0.5",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_err_002() -> dict[str, Any]:
    """Proportional error -- NLL finite with error_model='proportional'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import TwoCompartmentIvPkModel  # type: ignore[import-untyped]

        dose = 1000.0
        CL, V1, V2, Q = 10.0, 20.0, 30.0, 5.0
        sigma = 0.1
        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        # Generate synthetic data from known params
        m_gen = TwoCompartmentIvPkModel(
            times, [1.0] * len(times), dose=dose, sigma=0.05,
        )
        y_exact = m_gen.predict([CL, V1, V2, Q])
        y_obs = [c * 1.03 for c in y_exact]

        model = TwoCompartmentIvPkModel(
            times, y_obs, dose=dose,
            error_model="proportional", sigma=sigma,
        )
        nll_val = model.nll([CL, V1, V2, Q])

        ok = math.isfinite(nll_val)
        observed = {
            "nll": nll_val,
            "error_model": "proportional",
            "sigma": sigma,
            "finite": math.isfinite(nll_val),
        }
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-ERR-002",
        section="3.2",
        title="Proportional error -- NLL finite with sigma=0.1",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_err_003() -> dict[str, Any]:
    """Combined error -- NLL finite with error_model='combined'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import TwoCompartmentIvPkModel  # type: ignore[import-untyped]

        dose = 1000.0
        CL, V1, V2, Q = 10.0, 20.0, 30.0, 5.0
        sigma = 0.1
        sigma_add = 0.5
        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        m_gen = TwoCompartmentIvPkModel(
            times, [1.0] * len(times), dose=dose, sigma=0.05,
        )
        y_exact = m_gen.predict([CL, V1, V2, Q])
        y_obs = [c * 1.03 for c in y_exact]

        model = TwoCompartmentIvPkModel(
            times, y_obs, dose=dose,
            error_model="combined", sigma=sigma, sigma_add=sigma_add,
        )
        nll_val = model.nll([CL, V1, V2, Q])

        ok = math.isfinite(nll_val)
        observed = {
            "nll": nll_val,
            "error_model": "combined",
            "sigma": sigma,
            "sigma_add": sigma_add,
            "finite": math.isfinite(nll_val),
        }
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-ERR-003",
        section="3.2",
        title="Combined error -- NLL finite with sigma=0.1, sigma_add=0.5",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_err_004() -> dict[str, Any]:
    """Error model NLL ordering -- additive != proportional for same data."""
    t0 = time.monotonic()
    try:
        from nextstat._core import TwoCompartmentIvPkModel  # type: ignore[import-untyped]

        dose = 1000.0
        CL, V1, V2, Q = 10.0, 20.0, 30.0, 5.0
        sigma = 0.1
        times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        m_gen = TwoCompartmentIvPkModel(
            times, [1.0] * len(times), dose=dose, sigma=0.05,
        )
        y_exact = m_gen.predict([CL, V1, V2, Q])
        y_obs = [c * 1.05 for c in y_exact]

        m_add = TwoCompartmentIvPkModel(
            times, y_obs, dose=dose,
            error_model="additive", sigma=sigma,
        )
        m_prop = TwoCompartmentIvPkModel(
            times, y_obs, dose=dose,
            error_model="proportional", sigma=sigma,
        )

        nll_add = m_add.nll([CL, V1, V2, Q])
        nll_prop = m_prop.nll([CL, V1, V2, Q])

        differ = abs(nll_add - nll_prop) > 1e-10
        both_finite = math.isfinite(nll_add) and math.isfinite(nll_prop)
        ok = differ and both_finite
        observed = {
            "nll_additive": nll_add,
            "nll_proportional": nll_prop,
            "abs_diff": abs(nll_add - nll_prop),
            "differ": differ,
            "both_finite": both_finite,
        }
        deviation = None if ok else "Additive and proportional NLLs are equal or not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-ERR-004",
        section="3.2",
        title="Error model NLL ordering -- additive != proportional",
        ok=ok,
        observed=observed,
        acceptance="Additive and proportional NLLs differ for same data",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_err_005() -> dict[str, Any]:
    """Sigma sensitivity -- NLL should change when sigma changes."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.05 for c in [c_analytical(t) for t in times]]

        sigma_a = 0.3
        sigma_b = 0.6
        m_a = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma_a,
        )
        m_b = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma_b,
        )

        nll_a = m_a.nll([CL, V, Ka])
        nll_b = m_b.nll([CL, V, Ka])

        differ = abs(nll_a - nll_b) > 1e-10
        both_finite = math.isfinite(nll_a) and math.isfinite(nll_b)
        ok = differ and both_finite
        observed = {
            "nll_sigma_03": nll_a,
            "nll_sigma_06": nll_b,
            "abs_diff": abs(nll_a - nll_b),
            "differ": differ,
            "both_finite": both_finite,
        }
        deviation = None if ok else "NLLs with different sigmas are equal or not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-ERR-005",
        section="3.2",
        title="Sigma sensitivity -- NLL changes when sigma changes",
        ok=ok,
        observed=observed,
        acceptance="NLL values differ for sigma=0.3 vs sigma=0.6",
        deviation=deviation,
        wall_s=wall_s,
    )


# ===================================================================
# Section 3.3 -- LLOQ Handling (4 tests)
# ===================================================================

def _oq_lloq_001() -> dict[str, Any]:
    """LLOQ ignore -- NLL finite with some y < lloq, policy='ignore'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        sigma = 0.3
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_exact = [c_analytical(t) for t in times]
        # Set last two points below LLOQ
        y_obs = list(y_exact)
        y_obs[-1] = 0.05
        y_obs[-2] = 0.08
        lloq = 0.1

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
            lloq=lloq, lloq_policy="ignore",
        )
        nll_val = model.nll([CL, V, Ka])

        ok = math.isfinite(nll_val)
        observed = {
            "nll": nll_val,
            "lloq": lloq,
            "lloq_policy": "ignore",
            "finite": math.isfinite(nll_val),
            "n_below_lloq": sum(1 for y in y_obs if y < lloq),
        }
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-LLOQ-001",
        section="3.3",
        title="LLOQ ignore -- NLL finite with BLQ observations",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite with lloq_policy='ignore'",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_lloq_002() -> dict[str, Any]:
    """LLOQ replace_half -- NLL finite with policy='replace_half'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        sigma = 0.3
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_exact = [c_analytical(t) for t in times]
        y_obs = list(y_exact)
        y_obs[-1] = 0.05
        y_obs[-2] = 0.08
        lloq = 0.1

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
            lloq=lloq, lloq_policy="replace_half",
        )
        nll_val = model.nll([CL, V, Ka])

        ok = math.isfinite(nll_val)
        observed = {
            "nll": nll_val,
            "lloq": lloq,
            "lloq_policy": "replace_half",
            "finite": math.isfinite(nll_val),
            "n_below_lloq": sum(1 for y in y_obs if y < lloq),
        }
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-LLOQ-002",
        section="3.3",
        title="LLOQ replace_half -- NLL finite with BLQ observations",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite with lloq_policy='replace_half'",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_lloq_003() -> dict[str, Any]:
    """LLOQ censored -- NLL finite with policy='censored'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        sigma = 0.3
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_exact = [c_analytical(t) for t in times]
        y_obs = list(y_exact)
        y_obs[-1] = 0.05
        y_obs[-2] = 0.08
        lloq = 0.1

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
            lloq=lloq, lloq_policy="censored",
        )
        nll_val = model.nll([CL, V, Ka])

        ok = math.isfinite(nll_val)
        observed = {
            "nll": nll_val,
            "lloq": lloq,
            "lloq_policy": "censored",
            "finite": math.isfinite(nll_val),
            "n_below_lloq": sum(1 for y in y_obs if y < lloq),
        }
        deviation = None if ok else f"NLL={nll_val} not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-LLOQ-003",
        section="3.3",
        title="LLOQ censored -- NLL finite with BLQ observations",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite with lloq_policy='censored'",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_lloq_004() -> dict[str, Any]:
    """LLOQ affects NLL -- NLL with LLOQ handling differs from NLL without."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        sigma = 0.3
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_exact = [c_analytical(t) for t in times]
        y_obs = list(y_exact)
        # Force last two points below LLOQ
        y_obs[-1] = 0.05
        y_obs[-2] = 0.08
        lloq = 0.1

        # Model without LLOQ handling
        m_none = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
        )
        nll_none = m_none.nll([CL, V, Ka])

        # Model with LLOQ censored
        m_cens = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=sigma,
            lloq=lloq, lloq_policy="censored",
        )
        nll_cens = m_cens.nll([CL, V, Ka])

        differ = abs(nll_none - nll_cens) > 1e-10
        both_finite = math.isfinite(nll_none) and math.isfinite(nll_cens)
        ok = differ and both_finite
        observed = {
            "nll_no_lloq": nll_none,
            "nll_censored": nll_cens,
            "abs_diff": abs(nll_none - nll_cens),
            "differ": differ,
            "both_finite": both_finite,
        }
        deviation = None if ok else "NLL with LLOQ censored == NLL without LLOQ, or not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-LLOQ-004",
        section="3.3",
        title="LLOQ affects NLL -- censored NLL differs from no-LLOQ NLL",
        ok=ok,
        observed=observed,
        acceptance="NLL with LLOQ handling differs from NLL without",
        deviation=deviation,
        wall_s=wall_s,
    )


# ===================================================================
# Section 3.4 -- MLE Individual Estimation (5 tests)
# ===================================================================

def _oq_mle_001() -> dict[str, Any]:
    """suggested_init exists -- returns list of correct length for 1-cpt oral."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.01 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.3,
        )
        init = model.suggested_init()
        n_params = model.n_params()

        correct_length = len(init) == n_params
        is_list = isinstance(init, (list, tuple))
        ok = correct_length and is_list
        observed = {
            "suggested_init": init,
            "length": len(init),
            "n_params": n_params,
            "correct_length": correct_length,
            "is_list": is_list,
        }
        parts = []
        if not is_list:
            parts.append("not a list/tuple")
        if not correct_length:
            parts.append(f"length={len(init)} != n_params={n_params}")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-MLE-001",
        section="3.4",
        title="suggested_init exists and has correct length (1-cpt oral)",
        ok=ok,
        observed=observed,
        acceptance="suggested_init() returns list of length n_params()",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_mle_002() -> dict[str, Any]:
    """suggested_bounds exists -- returns list of (low, high) tuples."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.01 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.3,
        )
        bounds = model.suggested_bounds()
        n_params = model.n_params()

        correct_length = len(bounds) == n_params
        all_tuples = all(
            isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds
        )
        all_ordered = all(b[0] < b[1] for b in bounds) if all_tuples else False

        ok = correct_length and all_tuples and all_ordered
        observed = {
            "suggested_bounds": bounds,
            "length": len(bounds),
            "n_params": n_params,
            "correct_length": correct_length,
            "all_tuples_len2": all_tuples,
            "all_low_lt_high": all_ordered,
        }
        parts = []
        if not correct_length:
            parts.append(f"length={len(bounds)} != n_params={n_params}")
        if not all_tuples:
            parts.append("not all entries are (low, high) pairs")
        if not all_ordered:
            parts.append("not all low < high")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-MLE-002",
        section="3.4",
        title="suggested_bounds exists and returns (low, high) tuples",
        ok=ok,
        observed=observed,
        acceptance="suggested_bounds() returns n_params tuples with low < high",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_mle_003() -> dict[str, Any]:
    """NLL at suggested_init -- NLL is finite at the suggested initial values."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.01 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.3,
        )
        init = model.suggested_init()
        nll_val = model.nll(init)

        ok = math.isfinite(nll_val)
        observed = {
            "suggested_init": init,
            "nll_at_init": nll_val,
            "finite": math.isfinite(nll_val),
        }
        deviation = None if ok else f"NLL={nll_val} at suggested_init is not finite"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-MLE-003",
        section="3.4",
        title="NLL at suggested_init is finite",
        ok=ok,
        observed=observed,
        acceptance="NLL is finite at suggested_init()",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_mle_004() -> dict[str, Any]:
    """Gradient finite at init -- grad_nll at suggested_init all finite, correct length."""
    t0 = time.monotonic()
    try:
        from nextstat._core import OneCompartmentOralPkModel  # type: ignore[import-untyped]

        dose = 320.0
        CL, V, Ka = 2.5, 35.0, 1.2
        Ke = CL / V
        times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

        def c_analytical(t: float) -> float:
            return (dose * Ka / (V * (Ka - Ke))) * (
                math.exp(-Ke * t) - math.exp(-Ka * t)
            )

        y_obs = [c * 1.01 for c in [c_analytical(t) for t in times]]

        model = OneCompartmentOralPkModel(
            times, y_obs, dose=dose, bioavailability=1.0, sigma=0.3,
        )
        init = model.suggested_init()
        grad = model.grad_nll(init)
        n_params = model.n_params()

        correct_length = len(grad) == n_params
        all_finite = all(math.isfinite(g) for g in grad)
        ok = correct_length and all_finite
        observed = {
            "suggested_init": init,
            "grad_at_init": grad,
            "grad_length": len(grad),
            "n_params": n_params,
            "correct_length": correct_length,
            "all_finite": all_finite,
        }
        parts = []
        if not correct_length:
            parts.append(f"grad length={len(grad)} != n_params={n_params}")
        if not all_finite:
            parts.append("not all gradient components finite")
        deviation = "; ".join(parts) if parts else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-MLE-004",
        section="3.4",
        title="Gradient finite at suggested_init, correct length",
        ok=ok,
        observed=observed,
        acceptance="grad_nll at suggested_init: all finite, length == n_params",
        deviation=deviation,
        wall_s=wall_s,
    )


def _oq_mle_005() -> dict[str, Any]:
    """Gradient dimension matches n_params -- for each PK model."""
    t0 = time.monotonic()
    try:
        from nextstat._core import (  # type: ignore[import-untyped]
            OneCompartmentOralPkModel,
            TwoCompartmentIvPkModel,
            TwoCompartmentOralPkModel,
            ThreeCompartmentIvPkModel,
            ThreeCompartmentOralPkModel,
        )

        times = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        dummy_y = [5.0] * len(times)

        results: list[dict[str, Any]] = []

        # 1-cpt oral: 3 params
        m1 = OneCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.5)
        init1 = m1.suggested_init()
        g1 = m1.grad_nll(init1)
        np1 = m1.n_params()
        results.append({
            "class": "OneCompartmentOralPkModel",
            "n_params": np1,
            "grad_length": len(g1),
            "match": len(g1) == np1,
        })

        # 2-cpt IV: 4 params
        m2 = TwoCompartmentIvPkModel(times, dummy_y, dose=100.0, sigma=0.5)
        init2 = m2.suggested_init()
        g2 = m2.grad_nll(init2)
        np2 = m2.n_params()
        results.append({
            "class": "TwoCompartmentIvPkModel",
            "n_params": np2,
            "grad_length": len(g2),
            "match": len(g2) == np2,
        })

        # 2-cpt oral: 5 params
        m3 = TwoCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.5)
        init3 = m3.suggested_init()
        g3 = m3.grad_nll(init3)
        np3 = m3.n_params()
        results.append({
            "class": "TwoCompartmentOralPkModel",
            "n_params": np3,
            "grad_length": len(g3),
            "match": len(g3) == np3,
        })

        # 3-cpt IV: 6 params
        m4 = ThreeCompartmentIvPkModel(times, dummy_y, dose=100.0, sigma=0.5)
        init4 = m4.suggested_init()
        g4 = m4.grad_nll(init4)
        np4 = m4.n_params()
        results.append({
            "class": "ThreeCompartmentIvPkModel",
            "n_params": np4,
            "grad_length": len(g4),
            "match": len(g4) == np4,
        })

        # 3-cpt oral: 7 params
        m5 = ThreeCompartmentOralPkModel(times, dummy_y, dose=100.0, sigma=0.5)
        init5 = m5.suggested_init()
        g5 = m5.grad_nll(init5)
        np5 = m5.n_params()
        results.append({
            "class": "ThreeCompartmentOralPkModel",
            "n_params": np5,
            "grad_length": len(g5),
            "match": len(g5) == np5,
        })

        all_match = all(r["match"] for r in results)
        ok = all_match
        observed = {"models": results, "all_match": all_match}
        mismatches = [r["class"] for r in results if not r["match"]]
        deviation = (
            f"Mismatch in: {', '.join(mismatches)}" if mismatches else None
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="OQ-MLE-005",
        section="3.4",
        title="Gradient dimension matches n_params for each PK model",
        ok=ok,
        observed=observed,
        acceptance="len(grad_nll()) == n_params() for all 5 PK model classes",
        deviation=deviation,
        wall_s=wall_s,
    )


# ===================================================================
# Public entry point
# ===================================================================

def run_oq_analytical_tests() -> list[dict[str, Any]]:
    """Run OQ analytical tests: PK, Error, LLOQ, MLE. Returns list of results."""
    return [
        # Section 3.1: PK Analytical Correctness
        _oq_pk_001(),
        _oq_pk_002(),
        _oq_pk_003(),
        _oq_pk_004(),
        _oq_pk_005(),
        _oq_pk_006(),
        _oq_pk_007(),
        _oq_pk_008(),
        # Section 3.2: Error Models
        _oq_err_001(),
        _oq_err_002(),
        _oq_err_003(),
        _oq_err_004(),
        _oq_err_005(),
        # Section 3.3: LLOQ Handling
        _oq_lloq_001(),
        _oq_lloq_002(),
        _oq_lloq_003(),
        _oq_lloq_004(),
        # Section 3.4: MLE Individual Estimation
        _oq_mle_001(),
        _oq_mle_002(),
        _oq_mle_003(),
        _oq_mle_004(),
        _oq_mle_005(),
    ]
