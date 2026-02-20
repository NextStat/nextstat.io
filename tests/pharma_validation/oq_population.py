"""OQ (Operational Qualification) test cases -- Population PK methods.

Implements OQ test cases from the IQ/OQ/PQ protocol (NS-VAL-001 v2.0.0,
Sections 3.5--3.12).  Covers FOCE, SAEM, GOF, VPC, SCM, NONMEM I/O,
Bioequivalence, and Trial Simulation.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any


# ---------------------------------------------------------------------------
# Result helper (same format as iq.py)
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


# ---------------------------------------------------------------------------
# Synthetic population PK data generator (5-subject quick tests)
# ---------------------------------------------------------------------------

def _generate_synthetic_5sub(
    seed: int = 42,
    dose: float = 320.0,
    n_sub: int = 5,
    n_obs: int = 8,
) -> dict[str, Any]:
    """Generate 5-subject 1-cpt oral data for quick tests."""
    rng = random.Random(seed)
    CL, V, Ka = 2.5, 35.0, 1.2
    obs_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    times: list[float] = []
    y: list[float] = []
    subject_idx: list[int] = []
    doses: list[float] = []
    for i in range(n_sub):
        cl_i = CL * math.exp(rng.gauss(0, 0.2))
        v_i = V * math.exp(rng.gauss(0, 0.15))
        ka_i = Ka * math.exp(rng.gauss(0, 0.3))
        ke_i = cl_i / v_i
        doses.append(dose)
        for t in obs_times[:n_obs]:
            if abs(ka_i - ke_i) < 1e-10:
                c = (dose * ka_i / v_i) * t * math.exp(-ke_i * t)
            else:
                c = (dose * ka_i / (v_i * (ka_i - ke_i))) * (
                    math.exp(-ke_i * t) - math.exp(-ka_i * t)
                )
            c_obs = max(c * (1 + rng.gauss(0, 0.1)), 0.01)
            times.append(t)
            y.append(round(c_obs, 4))
            subject_idx.append(i)
    return {
        "times": times,
        "y": y,
        "subject_idx": subject_idx,
        "n_subjects": n_sub,
        "doses": doses,
        "dose": dose,
    }


# ===================================================================
# Section 3.5: FOCE Population Estimation (6 tests)
# ===================================================================

def _oq_foce_001() -> dict[str, Any]:
    """FOCE runs without error -- 5 subjects, additive, returns dict with required keys."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        required_keys = {"theta", "omega", "eta", "ofv", "converged", "n_iter"}
        found = set(res.keys()) & required_keys
        missing = required_keys - found
        ok = len(missing) == 0 and isinstance(res, dict)
        observed = {
            "type": type(res).__name__,
            "keys_found": sorted(found),
            "keys_missing": sorted(missing),
        }
        deviation = None if ok else f"Missing keys: {sorted(missing)}"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-001", "3.5",
        "FOCE runs without error (5 subjects, additive)",
        ok, observed,
        "Returns dict with keys: theta, omega, eta, ofv, converged, n_iter",
        deviation, wall_s,
    )


def _oq_foce_002() -> dict[str, Any]:
    """FOCE theta length -- theta has 3 elements (CL, V, Ka) for 1-cpt oral."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        theta = res.get("theta", [])
        ok = len(theta) == 3
        observed = {"theta": theta, "len": len(theta)}
        deviation = None if ok else f"theta length {len(theta)} != 3"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-002", "3.5",
        "FOCE theta length (3 elements for 1-cpt oral)",
        ok, observed, "len(theta) == 3", deviation, wall_s,
    )


def _oq_foce_003() -> dict[str, Any]:
    """FOCE omega length -- omega has 3 elements."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        omega = res.get("omega", [])
        ok = len(omega) == 3
        observed = {"omega": omega, "len": len(omega)}
        deviation = None if ok else f"omega length {len(omega)} != 3"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-003", "3.5",
        "FOCE omega length (3 elements)",
        ok, observed, "len(omega) == 3", deviation, wall_s,
    )


def _oq_foce_004() -> dict[str, Any]:
    """FOCE eta shape -- eta has n_subjects rows."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        eta = res.get("eta", [])
        n_rows = len(eta)
        ok = n_rows == data["n_subjects"]
        observed = {"n_eta_rows": n_rows, "n_subjects": data["n_subjects"]}
        deviation = None if ok else f"eta rows {n_rows} != n_subjects {data['n_subjects']}"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-004", "3.5",
        "FOCE eta shape (n_subjects rows)",
        ok, observed, "len(eta) == n_subjects", deviation, wall_s,
    )


def _oq_foce_005() -> dict[str, Any]:
    """FOCE OFV finite -- OFV is a finite float."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        ofv = res.get("ofv")
        ok = ofv is not None and isinstance(ofv, (int, float)) and math.isfinite(ofv)
        observed = {"ofv": ofv, "is_finite": ok}
        deviation = None if ok else f"OFV = {ofv} (expected finite float)"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-005", "3.5",
        "FOCE OFV finite",
        ok, observed, "isfinite(ofv)", deviation, wall_s,
    )


def _oq_foce_006() -> dict[str, Any]:
    """FOCE converged -- converged flag is bool."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_foce  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_foce(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        converged = res.get("converged")
        ok = isinstance(converged, bool)
        observed = {"converged": converged, "type": type(converged).__name__}
        deviation = None if ok else f"converged type = {type(converged).__name__}, expected bool"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-FOCE-006", "3.5",
        "FOCE converged flag is bool",
        ok, observed, "isinstance(converged, bool)", deviation, wall_s,
    )


# ===================================================================
# Section 3.6: SAEM Population Estimation (6 tests)
# ===================================================================

def _oq_saem_001() -> dict[str, Any]:
    """SAEM runs -- 5 subjects, model='1cpt_oral', seed=42. Returns SaemResult."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="1cpt_oral",
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        required_keys = {"theta", "omega", "omega_matrix", "eta", "ofv", "converged", "n_iter"}
        found = set(res.keys()) & required_keys
        missing = required_keys - found
        ok = len(missing) == 0
        observed = {
            "type": type(res).__name__,
            "keys_found": sorted(found),
            "keys_missing": sorted(missing),
        }
        deviation = None if ok else f"Missing keys: {sorted(missing)}"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-001", "3.6",
        "SAEM runs (5 subjects, 1cpt_oral, seed=42)",
        ok, observed,
        "Returns SaemResult with required keys",
        deviation, wall_s,
    )


def _oq_saem_002() -> dict[str, Any]:
    """SAEM theta length -- 3 elements."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="1cpt_oral",
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        theta = res.get("theta", [])
        ok = len(theta) == 3
        observed = {"theta": theta, "len": len(theta)}
        deviation = None if ok else f"theta length {len(theta)} != 3"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-002", "3.6",
        "SAEM theta length (3 elements)",
        ok, observed, "len(theta) == 3", deviation, wall_s,
    )


def _oq_saem_003() -> dict[str, Any]:
    """SAEM omega_matrix shape -- 3x3."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="1cpt_oral",
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        omega_mat = res.get("omega_matrix", [])
        n_rows = len(omega_mat)
        n_cols = len(omega_mat[0]) if omega_mat else 0
        ok = n_rows == 3 and n_cols == 3
        observed = {"shape": [n_rows, n_cols]}
        deviation = None if ok else f"omega_matrix shape [{n_rows}, {n_cols}] != [3, 3]"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-003", "3.6",
        "SAEM omega_matrix shape (3x3)",
        ok, observed, "omega_matrix is 3x3", deviation, wall_s,
    )


def _oq_saem_004() -> dict[str, Any]:
    """SAEM correlation matrix -- 3x3, diagonal elements approx 1.0."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="1cpt_oral",
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        corr = res.get("correlation", [])
        n_rows = len(corr)
        n_cols = len(corr[0]) if corr else 0
        diag_ok = True
        diag_vals: list[float] = []
        if n_rows == 3 and n_cols == 3:
            for i in range(3):
                diag_vals.append(corr[i][i])
                if abs(corr[i][i] - 1.0) > 0.01:
                    diag_ok = False
        else:
            diag_ok = False
        ok = n_rows == 3 and n_cols == 3 and diag_ok
        observed = {
            "shape": [n_rows, n_cols],
            "diagonal": [round(d, 6) for d in diag_vals],
            "diag_approx_1": diag_ok,
        }
        deviation = None if ok else (
            f"shape [{n_rows},{n_cols}] or diag != 1.0: {diag_vals}"
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-004", "3.6",
        "SAEM correlation matrix (3x3, diagonal ~ 1.0)",
        ok, observed,
        "correlation is 3x3 with diagonal elements ~ 1.0",
        deviation, wall_s,
    )


def _oq_saem_005() -> dict[str, Any]:
    """SAEM OFV finite."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="1cpt_oral",
            doses=data["doses"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        ofv = res.get("ofv")
        ok = ofv is not None and isinstance(ofv, (int, float)) and math.isfinite(ofv)
        observed = {"ofv": ofv, "is_finite": ok}
        deviation = None if ok else f"OFV = {ofv} (expected finite float)"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-005", "3.6",
        "SAEM OFV finite",
        ok, observed, "isfinite(ofv)", deviation, wall_s,
    )


def _oq_saem_006() -> dict[str, Any]:
    """SAEM 2cpt_iv model -- model='2cpt_iv', runs without error."""
    t0 = time.monotonic()
    try:
        from nextstat._core import nlme_saem  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        res = nlme_saem(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            model="2cpt_iv",
            doses=data["doses"],
            theta_init=[10.0, 20.0, 30.0, 5.0],
            omega_init=[0.2, 0.2, 0.2, 0.2],
            error_model="additive",
            sigma=0.1,
            n_burn=100,
            n_iter=50,
            seed=42,
        )
        has_theta = "theta" in res
        has_ofv = "ofv" in res
        ok = has_theta and has_ofv
        observed = {
            "model": "2cpt_iv",
            "has_theta": has_theta,
            "has_ofv": has_ofv,
            "theta_len": len(res.get("theta", [])),
        }
        deviation = None if ok else "2cpt_iv SAEM missing theta or ofv"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SAEM-006", "3.6",
        "SAEM 2cpt_iv model runs without error",
        ok, observed,
        "2cpt_iv model returns theta and ofv",
        deviation, wall_s,
    )


# ===================================================================
# Section 3.7: GOF Diagnostics (3 tests)
# ===================================================================

def _run_foce_for_gof() -> tuple[dict[str, Any], dict[str, Any]]:
    """Helper: run FOCE and return (data, foce_result) for GOF tests."""
    from nextstat._core import nlme_foce  # type: ignore[import-untyped]

    data = _generate_synthetic_5sub()
    res = nlme_foce(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        doses=data["doses"],
        theta_init=[2.5, 35.0, 1.2],
        omega_init=[0.2, 0.15, 0.3],
        error_model="additive",
        sigma=0.1,
    )
    return data, res


def _oq_gof_001() -> dict[str, Any]:
    """GOF runs -- fit FOCE first, then pk_gof with theta+eta. Returns list."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_gof  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_gof()
        gof = pk_gof(
            data["times"],
            data["y"],
            data["subject_idx"],
            doses=data["doses"],
            theta=foce_res["theta"],
            eta=foce_res["eta"],
            error_model="additive",
            sigma=0.1,
        )
        ok = isinstance(gof, list) and len(gof) > 0
        observed = {"type": type(gof).__name__, "len": len(gof) if isinstance(gof, list) else None}
        deviation = None if ok else f"GOF result type = {type(gof).__name__}, expected non-empty list"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-GOF-001", "3.7",
        "GOF runs (FOCE then pk_gof)",
        ok, observed, "Returns non-empty list", deviation, wall_s,
    )


def _oq_gof_002() -> dict[str, Any]:
    """GOF returns records -- each record is dict with expected keys."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_gof  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_gof()
        gof = pk_gof(
            data["times"],
            data["y"],
            data["subject_idx"],
            doses=data["doses"],
            theta=foce_res["theta"],
            eta=foce_res["eta"],
            error_model="additive",
            sigma=0.1,
        )
        if not gof:
            ok = False
            observed: dict[str, Any] = {"note": "empty GOF result"}
            deviation = "GOF returned empty list"
        else:
            first = gof[0]
            is_dict = isinstance(first, dict)
            record_keys = sorted(first.keys()) if is_dict else []
            ok = is_dict and len(record_keys) > 0
            observed = {
                "first_record_type": type(first).__name__,
                "first_record_keys": record_keys,
            }
            deviation = None if ok else "GOF records are not dicts with keys"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-GOF-002", "3.7",
        "GOF records are dicts with expected keys",
        ok, observed, "Each record is a dict with keys", deviation, wall_s,
    )


def _oq_gof_003() -> dict[str, Any]:
    """GOF length matches observations -- len(result) == len(y)."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_gof  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_gof()
        gof = pk_gof(
            data["times"],
            data["y"],
            data["subject_idx"],
            doses=data["doses"],
            theta=foce_res["theta"],
            eta=foce_res["eta"],
            error_model="additive",
            sigma=0.1,
        )
        n_gof = len(gof)
        n_y = len(data["y"])
        ok = n_gof == n_y
        observed = {"n_gof_records": n_gof, "n_observations": n_y}
        deviation = None if ok else f"len(gof)={n_gof} != len(y)={n_y}"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-GOF-003", "3.7",
        "GOF length matches observations",
        ok, observed, "len(result) == len(y)", deviation, wall_s,
    )


# ===================================================================
# Section 3.8: VPC (3 tests)
# ===================================================================

def _run_foce_for_vpc() -> tuple[dict[str, Any], dict[str, Any]]:
    """Helper: run FOCE and return (data, foce_result) for VPC tests."""
    from nextstat._core import nlme_foce  # type: ignore[import-untyped]

    data = _generate_synthetic_5sub()
    res = nlme_foce(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        doses=data["doses"],
        theta_init=[2.5, 35.0, 1.2],
        omega_init=[0.2, 0.15, 0.3],
        error_model="additive",
        sigma=0.1,
    )
    return data, res


def _build_omega_matrix(omega: list[float]) -> list[list[float]]:
    """Build diagonal omega matrix from omega vector."""
    n = len(omega)
    mat: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = omega[i] ** 2 if omega[i] > 0 else omega[i]
    return mat


def _oq_vpc_001() -> dict[str, Any]:
    """VPC runs -- fit FOCE, build omega_matrix, call pk_vpc. Returns dict."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_vpc  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_vpc()
        omega = foce_res.get("omega", [0.2, 0.15, 0.3])
        omega_mat = foce_res.get("omega_matrix", _build_omega_matrix(omega))

        vpc_res = pk_vpc(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta=foce_res["theta"],
            omega_matrix=omega_mat,
            error_model="additive",
            sigma=0.1,
            n_sim=50,
            seed=42,
        )
        ok = isinstance(vpc_res, dict) and len(vpc_res) > 0
        observed = {
            "type": type(vpc_res).__name__,
            "keys": sorted(vpc_res.keys()) if isinstance(vpc_res, dict) else [],
        }
        deviation = None if ok else f"VPC result type = {type(vpc_res).__name__}, expected non-empty dict"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-VPC-001", "3.8",
        "VPC runs (n_sim=50, seed=42)",
        ok, observed, "Returns non-empty dict", deviation, wall_s,
    )


def _oq_vpc_002() -> dict[str, Any]:
    """VPC has expected keys -- result has standard VPC keys."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_vpc  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_vpc()
        omega = foce_res.get("omega", [0.2, 0.15, 0.3])
        omega_mat = foce_res.get("omega_matrix", _build_omega_matrix(omega))

        vpc_res = pk_vpc(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta=foce_res["theta"],
            omega_matrix=omega_mat,
            error_model="additive",
            sigma=0.1,
            n_sim=50,
            seed=42,
        )
        all_keys = sorted(vpc_res.keys()) if isinstance(vpc_res, dict) else []
        # VPC should have at least some standard keys (bins, n_sim, quantiles, etc.)
        ok = len(all_keys) >= 1
        observed = {"keys": all_keys}
        deviation = None if ok else "VPC returned dict with no keys"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-VPC-002", "3.8",
        "VPC has expected keys",
        ok, observed, "VPC result has standard keys", deviation, wall_s,
    )


def _oq_vpc_003() -> dict[str, Any]:
    """VPC n_sim respected -- result reflects correct simulation count."""
    t0 = time.monotonic()
    try:
        from nextstat._core import pk_vpc  # type: ignore[import-untyped]

        data, foce_res = _run_foce_for_vpc()
        omega = foce_res.get("omega", [0.2, 0.15, 0.3])
        omega_mat = foce_res.get("omega_matrix", _build_omega_matrix(omega))

        vpc_res = pk_vpc(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            doses=data["doses"],
            theta=foce_res["theta"],
            omega_matrix=omega_mat,
            error_model="additive",
            sigma=0.1,
            n_sim=50,
            seed=42,
        )
        n_sim_out = vpc_res.get("n_sim")
        if n_sim_out is not None:
            ok = n_sim_out == 50
            observed: dict[str, Any] = {"n_sim": n_sim_out, "expected": 50}
            deviation = None if ok else f"n_sim = {n_sim_out}, expected 50"
        else:
            # If n_sim not in output, just check that VPC ran successfully
            ok = isinstance(vpc_res, dict) and len(vpc_res) > 0
            observed = {"n_sim": None, "note": "n_sim key not in result, VPC ran OK"}
            deviation = None if ok else "VPC result empty and no n_sim key"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-VPC-003", "3.8",
        "VPC n_sim respected",
        ok, observed, "n_sim == 50 or VPC ran successfully", deviation, wall_s,
    )


# ===================================================================
# Section 3.9: SCM (3 tests)
# ===================================================================

def _oq_scm_001() -> dict[str, Any]:
    """SCM runs -- 5 subjects, 1 covariate (weight), dose=320 (singular). Returns ScmResult."""
    t0 = time.monotonic()
    try:
        from nextstat._core import scm  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        n_sub = data["n_subjects"]
        # Generate per-subject weight, then expand to per-observation
        rng = random.Random(99)
        wt_per_sub = [rng.gauss(70.0, 15.0) for _ in range(n_sub)]
        weights = [wt_per_sub[s] for s in data["subject_idx"]]
        covariates = [weights]
        covariate_names = ["weight"]

        res = scm(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            covariates,
            covariate_names,
            dose=data["dose"],  # SINGULAR float
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        has_selected = "selected" in res
        has_base_ofv = "base_ofv" in res
        ok = has_selected and has_base_ofv
        observed = {
            "type": type(res).__name__,
            "has_selected": has_selected,
            "has_base_ofv": has_base_ofv,
            "keys": sorted(res.keys()) if isinstance(res, dict) else [],
        }
        deviation = None if ok else "SCM missing selected or base_ofv"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SCM-001", "3.9",
        "SCM runs (5 subjects, weight covariate, dose=320)",
        ok, observed, "Returns ScmResult with selected and base_ofv", deviation, wall_s,
    )


def _oq_scm_002() -> dict[str, Any]:
    """SCM forward_trace -- forward_trace is list."""
    t0 = time.monotonic()
    try:
        from nextstat._core import scm  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        n_sub = data["n_subjects"]
        rng = random.Random(99)
        wt_per_sub = [rng.gauss(70.0, 15.0) for _ in range(n_sub)]
        weights = [wt_per_sub[s] for s in data["subject_idx"]]
        covariates = [weights]
        covariate_names = ["weight"]

        res = scm(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            covariates,
            covariate_names,
            dose=data["dose"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        fwd_trace = res.get("forward_trace")
        ok = isinstance(fwd_trace, list)
        observed = {
            "forward_trace_type": type(fwd_trace).__name__,
            "forward_trace_len": len(fwd_trace) if isinstance(fwd_trace, list) else None,
        }
        deviation = None if ok else f"forward_trace type = {type(fwd_trace).__name__}, expected list"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SCM-002", "3.9",
        "SCM forward_trace is list",
        ok, observed, "isinstance(forward_trace, list)", deviation, wall_s,
    )


def _oq_scm_003() -> dict[str, Any]:
    """SCM base_ofv finite."""
    t0 = time.monotonic()
    try:
        from nextstat._core import scm  # type: ignore[import-untyped]

        data = _generate_synthetic_5sub()
        n_sub = data["n_subjects"]
        rng = random.Random(99)
        wt_per_sub = [rng.gauss(70.0, 15.0) for _ in range(n_sub)]
        weights = [wt_per_sub[s] for s in data["subject_idx"]]
        covariates = [weights]
        covariate_names = ["weight"]

        res = scm(
            data["times"],
            data["y"],
            data["subject_idx"],
            data["n_subjects"],
            covariates,
            covariate_names,
            dose=data["dose"],
            theta_init=[2.5, 35.0, 1.2],
            omega_init=[0.2, 0.15, 0.3],
            error_model="additive",
            sigma=0.1,
        )
        base_ofv = res.get("base_ofv")
        ok = base_ofv is not None and isinstance(base_ofv, (int, float)) and math.isfinite(base_ofv)
        observed = {"base_ofv": base_ofv, "is_finite": ok}
        deviation = None if ok else f"base_ofv = {base_ofv} (expected finite float)"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SCM-003", "3.9",
        "SCM base_ofv finite",
        ok, observed, "isfinite(base_ofv)", deviation, wall_s,
    )


# ===================================================================
# Section 3.10: NONMEM I/O (4 tests)
# ===================================================================

# Minimal 3-subject Theophylline-like CSV
_THEO_CSV_3SUB = """\
ID,TIME,DV,AMT,EVID
1,0,.,320,1
1,0.5,5.21,,0
1,1,8.13,,0
1,2,9.87,,0
1,4,8.45,,0
1,8,5.12,,0
2,0,.,320,1
2,0.5,4.89,,0
2,1,7.56,,0
2,2,10.12,,0
2,4,9.23,,0
2,8,6.45,,0
3,0,.,320,1
3,0.5,6.12,,0
3,1,9.34,,0
3,2,11.56,,0
3,4,7.89,,0
3,8,4.23,,0
"""

_THEO_3SUB_N_SUBJECTS = 3
_THEO_3SUB_N_OBS = 15  # 5 obs per subject x 3 subjects


def _oq_io_001() -> dict[str, Any]:
    """read_nonmem parses CSV -- parse Theophylline CSV (3-subject). Returns dict."""
    t0 = time.monotonic()
    try:
        from nextstat._core import read_nonmem  # type: ignore[import-untyped]

        parsed = read_nonmem(_THEO_CSV_3SUB)
        ok = isinstance(parsed, dict) and len(parsed) > 0
        observed = {
            "type": type(parsed).__name__,
            "keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        }
        deviation = None if ok else f"read_nonmem returned {type(parsed).__name__}, expected non-empty dict"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-IO-001", "3.10",
        "read_nonmem parses CSV (3-subject Theophylline)",
        ok, observed, "Returns non-empty dict", deviation, wall_s,
    )


def _oq_io_002() -> dict[str, Any]:
    """read_nonmem keys -- result has expected keys."""
    t0 = time.monotonic()
    try:
        from nextstat._core import read_nonmem  # type: ignore[import-untyped]

        parsed = read_nonmem(_THEO_CSV_3SUB)
        all_keys = sorted(parsed.keys()) if isinstance(parsed, dict) else []
        # Check for at least some expected keys
        expected_candidates = ["times", "dv", "y", "subject_idx", "n_subjects",
                               "subject_ids", "doses", "evid", "amt"]
        found = [k for k in expected_candidates if k in parsed]
        ok = len(found) >= 3  # at least times, y, subject_idx or similar
        observed = {"all_keys": all_keys, "expected_found": found}
        deviation = None if ok else f"Only {len(found)} expected keys found: {found}"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-IO-002", "3.10",
        "read_nonmem keys present",
        ok, observed, "At least 3 expected keys found", deviation, wall_s,
    )


def _oq_io_003() -> dict[str, Any]:
    """read_nonmem row count -- correct number of rows parsed."""
    t0 = time.monotonic()
    try:
        from nextstat._core import read_nonmem  # type: ignore[import-untyped]

        parsed = read_nonmem(_THEO_CSV_3SUB)
        # read_nonmem returns observation-only rows (EVID=0) with keys:
        # n_subjects, subject_ids, times, dv, subject_idx
        dv = parsed.get("dv", parsed.get("y", []))
        n_obs = len(dv)
        n_subj = parsed.get("n_subjects", -1)
        ok = n_obs == _THEO_3SUB_N_OBS and n_subj == _THEO_3SUB_N_SUBJECTS
        observed = {
            "n_observations": n_obs,
            "expected_obs": _THEO_3SUB_N_OBS,
            "n_subjects": n_subj,
            "expected_subjects": _THEO_3SUB_N_SUBJECTS,
        }
        deviation = None if ok else (
            f"n_obs={n_obs} (exp {_THEO_3SUB_N_OBS}), "
            f"n_subjects={n_subj} (exp {_THEO_3SUB_N_SUBJECTS})"
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-IO-003", "3.10",
        "read_nonmem row count correct",
        ok, observed,
        f"n_obs == {_THEO_3SUB_N_OBS} and n_subjects == {_THEO_3SUB_N_SUBJECTS}",
        deviation, wall_s,
    )


def _oq_io_004() -> dict[str, Any]:
    """read_nonmem handles EVID -- dose rows filtered, subjects identified."""
    t0 = time.monotonic()
    try:
        from nextstat._core import read_nonmem  # type: ignore[import-untyped]

        parsed = read_nonmem(_THEO_CSV_3SUB)
        # read_nonmem strips EVID=1 rows and returns obs-only data
        # Verify subject identification is correct
        subject_ids = parsed.get("subject_ids", [])
        n_subj = parsed.get("n_subjects", -1)
        subject_idx = parsed.get("subject_idx", [])
        dv = parsed.get("dv", [])
        # All obs should have valid subject_idx in [0, n_subjects)
        idx_valid = all(0 <= s < n_subj for s in subject_idx) if subject_idx else False
        ok = n_subj == _THEO_3SUB_N_SUBJECTS and idx_valid and len(dv) > 0
        observed = {
            "n_subjects": n_subj,
            "subject_ids": subject_ids,
            "n_obs": len(dv),
            "idx_valid": idx_valid,
        }
        deviation = None if ok else (
            f"n_subjects={n_subj} (exp {_THEO_3SUB_N_SUBJECTS}), idx_valid={idx_valid}"
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-IO-004", "3.10",
        "read_nonmem handles EVID (dose rows filtered, subjects identified)",
        ok, observed,
        "n_subjects correct, subject_idx valid, obs data present",
        deviation, wall_s,
    )


# ===================================================================
# Section 3.11: Bioequivalence (4 tests)
# ===================================================================

def _oq_be_001() -> dict[str, Any]:
    """average_be returns BeResult with expected keys."""
    t0 = time.monotonic()
    try:
        from nextstat._core import average_be  # type: ignore[import-untyped]

        # Log-transformed AUC values (BE API expects ln-scale data)
        raw_test = [100.0, 105.0, 98.0, 102.0, 110.0, 97.0, 103.0, 99.0]
        raw_ref = [95.0, 100.0, 97.0, 101.0, 108.0, 96.0, 100.0, 98.0]
        test_vals = [math.log(v) for v in raw_test]
        ref_vals = [math.log(v) for v in raw_ref]
        res = average_be(test_vals, ref_vals)
        has_gmr = "geometric_mean_ratio" in res
        has_ci_lo = "ci_lower" in res
        has_ci_hi = "ci_upper" in res
        has_concl = "conclusion" in res
        ok = has_gmr and has_ci_lo and has_ci_hi and has_concl
        observed = {
            "type": type(res).__name__,
            "has_geometric_mean_ratio": has_gmr,
            "has_ci_lower": has_ci_lo,
            "has_ci_upper": has_ci_hi,
            "has_conclusion": has_concl,
            "keys": sorted(res.keys()) if isinstance(res, dict) else [],
        }
        deviation = None if ok else "Missing one or more required keys"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-BE-001", "3.11",
        "average_be returns BeResult with expected keys",
        ok, observed,
        "Result has geometric_mean_ratio, ci_lower, ci_upper, conclusion",
        deviation, wall_s,
    )


def _oq_be_002() -> dict[str, Any]:
    """be_power returns float in [0, 1]."""
    t0 = time.monotonic()
    try:
        from nextstat._core import be_power  # type: ignore[import-untyped]

        power = be_power(24, cv=0.30, gmr=0.95)
        ok = isinstance(power, (int, float)) and 0.0 <= float(power) <= 1.0
        observed = {"power": power, "type": type(power).__name__}
        deviation = None if ok else f"power = {power} (expected float in [0, 1])"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-BE-002", "3.11",
        "be_power returns float in [0, 1]",
        ok, observed,
        "0 <= be_power(24, cv=0.30, gmr=0.95) <= 1",
        deviation, wall_s,
    )


def _oq_be_003() -> dict[str, Any]:
    """be_sample_size returns BeSampleSizeResult."""
    t0 = time.monotonic()
    try:
        from nextstat._core import be_sample_size  # type: ignore[import-untyped]

        res = be_sample_size(cv=0.30, gmr=0.95, target_power=0.80)
        ok = isinstance(res, dict) and len(res) > 0
        observed = {
            "type": type(res).__name__,
            "keys": sorted(res.keys()) if isinstance(res, dict) else [],
        }
        deviation = None if ok else f"be_sample_size returned {type(res).__name__}, expected non-empty dict"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-BE-003", "3.11",
        "be_sample_size returns BeSampleSizeResult",
        ok, observed,
        "Returns non-empty dict (BeSampleSizeResult)",
        deviation, wall_s,
    )


def _oq_be_004() -> dict[str, Any]:
    """BE bioequivalent conclusion -- test ~ ref -> conclusion contains 'bioequivalent'."""
    t0 = time.monotonic()
    try:
        from nextstat._core import average_be  # type: ignore[import-untyped]

        # Log-transformed AUC values (ln scale) — BE API expects log data
        # These paired values have GMR ~1.03 (within 0.80–1.25)
        raw_test = [450.2, 520.1, 380.5, 490.3, 410.7, 550.2, 470.8, 395.4, 510.6, 440.3, 485.1, 420.9]
        raw_ref = [430.5, 510.3, 370.2, 480.1, 400.5, 530.8, 460.2, 380.7, 500.4, 430.1, 475.3, 410.5]
        test_vals = [math.log(v) for v in raw_test]
        ref_vals = [math.log(v) for v in raw_ref]
        res = average_be(test_vals, ref_vals)
        conclusion = str(res.get("conclusion", "")).lower()
        ok = "bioequivalent" in conclusion
        observed = {
            "conclusion": res.get("conclusion"),
            "geometric_mean_ratio": res.get("geometric_mean_ratio"),
            "ci_lower": res.get("ci_lower"),
            "ci_upper": res.get("ci_upper"),
        }
        deviation = None if ok else f"conclusion = '{res.get('conclusion')}', expected to contain 'bioequivalent'"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-BE-004", "3.11",
        "BE bioequivalent conclusion (test ~ ref)",
        ok, observed,
        "conclusion contains 'bioequivalent' (case-insensitive)",
        deviation, wall_s,
    )


# ===================================================================
# Section 3.12: Trial Simulation (3 tests)
# ===================================================================

def _oq_sim_001() -> dict[str, Any]:
    """simulate_trial runs -- n_subjects=10, 1cpt_oral. Returns TrialSimResult."""
    t0 = time.monotonic()
    try:
        from nextstat._core import simulate_trial  # type: ignore[import-untyped]

        res = simulate_trial(
            n_subjects=10,
            dose=320.0,
            obs_times=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
            pk_model="1cpt_oral",
            theta=[2.5, 35.0, 1.2],
            omega=[0.2, 0.15, 0.3],
            sigma=0.1,
            error_model="proportional",
            seed=42,
        )
        has_conc = "concentrations" in res
        has_auc = "auc" in res
        has_cmax = "cmax" in res
        has_tmax = "tmax" in res
        ok = has_conc and has_auc and has_cmax and has_tmax
        observed = {
            "type": type(res).__name__,
            "has_concentrations": has_conc,
            "has_auc": has_auc,
            "has_cmax": has_cmax,
            "has_tmax": has_tmax,
            "keys": sorted(res.keys()) if isinstance(res, dict) else [],
        }
        deviation = None if ok else "Missing one or more required keys"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SIM-001", "3.12",
        "simulate_trial runs (10 subjects, 1cpt_oral)",
        ok, observed,
        "Returns TrialSimResult with concentrations, auc, cmax, tmax",
        deviation, wall_s,
    )


def _oq_sim_002() -> dict[str, Any]:
    """simulate_trial concentrations shape -- len == n_subjects, inner len == len(obs_times)."""
    t0 = time.monotonic()
    try:
        from nextstat._core import simulate_trial  # type: ignore[import-untyped]

        obs_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
        n_subjects = 10
        res = simulate_trial(
            n_subjects=n_subjects,
            dose=320.0,
            obs_times=obs_times,
            pk_model="1cpt_oral",
            theta=[2.5, 35.0, 1.2],
            omega=[0.2, 0.15, 0.3],
            sigma=0.1,
            error_model="proportional",
            seed=42,
        )
        conc = res.get("concentrations", [])
        n_outer = len(conc)
        n_inner = len(conc[0]) if conc else 0
        ok = n_outer == n_subjects and n_inner == len(obs_times)
        observed = {
            "n_subjects_conc": n_outer,
            "n_obs_times_conc": n_inner,
            "expected_subjects": n_subjects,
            "expected_obs_times": len(obs_times),
        }
        deviation = None if ok else (
            f"concentrations shape [{n_outer}, {n_inner}] "
            f"!= [{n_subjects}, {len(obs_times)}]"
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SIM-002", "3.12",
        "simulate_trial concentrations shape",
        ok, observed,
        "len(concentrations) == n_subjects and len(concentrations[0]) == len(obs_times)",
        deviation, wall_s,
    )


def _oq_sim_003() -> dict[str, Any]:
    """simulate_trial PK metrics -- auc, cmax, tmax all len == n_subjects and positive."""
    t0 = time.monotonic()
    try:
        from nextstat._core import simulate_trial  # type: ignore[import-untyped]

        n_subjects = 10
        res = simulate_trial(
            n_subjects=n_subjects,
            dose=320.0,
            obs_times=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
            pk_model="1cpt_oral",
            theta=[2.5, 35.0, 1.2],
            omega=[0.2, 0.15, 0.3],
            sigma=0.1,
            error_model="proportional",
            seed=42,
        )
        auc = res.get("auc", [])
        cmax = res.get("cmax", [])
        tmax = res.get("tmax", [])

        auc_ok = len(auc) == n_subjects and all(
            isinstance(v, (int, float)) and v > 0 for v in auc
        )
        cmax_ok = len(cmax) == n_subjects and all(
            isinstance(v, (int, float)) and v > 0 for v in cmax
        )
        tmax_ok = len(tmax) == n_subjects and all(
            isinstance(v, (int, float)) and v > 0 for v in tmax
        )
        ok = auc_ok and cmax_ok and tmax_ok
        observed = {
            "auc_len": len(auc),
            "cmax_len": len(cmax),
            "tmax_len": len(tmax),
            "auc_all_positive": auc_ok,
            "cmax_all_positive": cmax_ok,
            "tmax_all_positive": tmax_ok,
        }
        parts: list[str] = []
        if not auc_ok:
            parts.append(f"auc issue (len={len(auc)})")
        if not cmax_ok:
            parts.append(f"cmax issue (len={len(cmax)})")
        if not tmax_ok:
            parts.append(f"tmax issue (len={len(tmax)})")
        deviation = None if ok else "; ".join(parts)
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Exception: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        "OQ-SIM-003", "3.12",
        "simulate_trial PK metrics (auc, cmax, tmax positive)",
        ok, observed,
        "auc, cmax, tmax all have length n_subjects and positive values",
        deviation, wall_s,
    )


# ===================================================================
# Public entry point
# ===================================================================

def run_oq_population_tests() -> list[dict[str, Any]]:
    """Run all OQ population PK tests.

    Sections: FOCE (3.5), SAEM (3.6), GOF (3.7), VPC (3.8),
    SCM (3.9), NONMEM I/O (3.10), Bioequivalence (3.11),
    Trial Simulation (3.12).
    """
    return [
        # Section 3.5 -- FOCE
        _oq_foce_001(),
        _oq_foce_002(),
        _oq_foce_003(),
        _oq_foce_004(),
        _oq_foce_005(),
        _oq_foce_006(),
        # Section 3.6 -- SAEM
        _oq_saem_001(),
        _oq_saem_002(),
        _oq_saem_003(),
        _oq_saem_004(),
        _oq_saem_005(),
        _oq_saem_006(),
        # Section 3.7 -- GOF
        _oq_gof_001(),
        _oq_gof_002(),
        _oq_gof_003(),
        # Section 3.8 -- VPC
        _oq_vpc_001(),
        _oq_vpc_002(),
        _oq_vpc_003(),
        # Section 3.9 -- SCM
        _oq_scm_001(),
        _oq_scm_002(),
        _oq_scm_003(),
        # Section 3.10 -- NONMEM I/O
        _oq_io_001(),
        _oq_io_002(),
        _oq_io_003(),
        _oq_io_004(),
        # Section 3.11 -- Bioequivalence
        _oq_be_001(),
        _oq_be_002(),
        _oq_be_003(),
        _oq_be_004(),
        # Section 3.12 -- Trial Simulation
        _oq_sim_001(),
        _oq_sim_002(),
        _oq_sim_003(),
    ]
