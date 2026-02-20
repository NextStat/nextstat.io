"""Unit tests for nextstat.scm() — Stepwise Covariate Modeling.

These tests require the built nextstat wheel. They verify:
  - Basic API contract (return keys match ScmResult TypedDict)
  - Forward selection of a significant covariate
  - Correct return types
  - Rejection of noise covariates
"""

from __future__ import annotations

import math
import random

import pytest


def _generate_pk_data(
    n_subjects: int,
    seed: int,
    wt_exponent: float = 0.75,
    include_age: bool = False,
):
    """Generate synthetic 1-cpt oral PK data with optional weight effect on CL.

    Returns dict with times, y, subject_idx, and covariate vectors.
    """
    rng = random.Random(seed)

    cl_pop = 1.2
    v_pop = 15.0
    ka_pop = 2.0
    dose = 100.0
    bioav = 1.0
    sigma = 0.05
    omega_sd = 0.15
    wt_center = 70.0

    sample_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

    weights = [40.0 + 60.0 * i / max(n_subjects - 1, 1) for i in range(n_subjects)]

    times = []
    y = []
    subject_idx = []
    wt_obs = []
    age_obs = []

    for sid in range(n_subjects):
        wt = weights[sid]
        age = 25.0 + 40.0 * rng.random()  # random age, no true effect

        eta_cl = rng.gauss(0, omega_sd)
        eta_v = rng.gauss(0, omega_sd)
        eta_ka = rng.gauss(0, omega_sd)

        wt_effect = (wt / wt_center) ** wt_exponent if abs(wt_exponent) > 1e-12 else 1.0
        cl_i = cl_pop * wt_effect * math.exp(eta_cl)
        v_i = v_pop * math.exp(eta_v)
        ka_i = ka_pop * math.exp(eta_ka)

        ke = cl_i / v_i
        if abs(ka_i - ke) < 1e-10:
            ka_i += 0.01

        for t in sample_times:
            c = (bioav * dose * ka_i) / (v_i * (ka_i - ke)) * (
                math.exp(-ke * t) - math.exp(-ka_i * t)
            )
            obs = max(0.0, c + rng.gauss(0, sigma))
            times.append(t)
            y.append(obs)
            subject_idx.append(sid)
            wt_obs.append(wt)
            age_obs.append(age)

    result = {
        "times": times,
        "y": y,
        "subject_idx": subject_idx,
        "n_subjects": n_subjects,
        "dose": dose,
        "bioav": bioav,
        "sigma": sigma,
        "wt_obs": wt_obs,
    }
    if include_age:
        result["age_obs"] = age_obs
    return result


def test_scm_basic():
    """Run SCM on synthetic data and verify result keys match ScmResult TypedDict."""
    import nextstat

    data = _generate_pk_data(n_subjects=20, seed=42, wt_exponent=0.75)

    result = nextstat.scm(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        [data["wt_obs"]],
        ["WT"],
        dose=data["dose"],
        bioavailability=data["bioav"],
        error_model="additive",
        sigma=data["sigma"],
        theta_init=[1.0, 10.0, 1.5],
        omega_init=[0.30, 0.30, 0.30],
    )

    # Verify all ScmResult keys are present.
    expected_keys = {
        "selected",
        "forward_trace",
        "backward_trace",
        "base_ofv",
        "final_ofv",
        "n_forward_steps",
        "n_backward_steps",
        "theta",
        "omega",
    }
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}, "
        f"Extra keys: {set(result.keys()) - expected_keys}"
    )

    # Basic sanity checks.
    assert math.isfinite(result["base_ofv"])
    assert math.isfinite(result["final_ofv"])
    assert isinstance(result["n_forward_steps"], int)
    assert isinstance(result["n_backward_steps"], int)
    assert isinstance(result["theta"], list)
    assert isinstance(result["omega"], list)
    assert len(result["theta"]) >= 3  # at least [CL, V, Ka]


def test_scm_selects_significant_covariate():
    """Verify that a strong weight effect on CL is detected by SCM."""
    import nextstat

    # 30 subjects, strong WT effect (exponent = 0.75).
    data = _generate_pk_data(n_subjects=30, seed=77, wt_exponent=0.75)

    result = nextstat.scm(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        [data["wt_obs"]],
        ["WT"],
        dose=data["dose"],
        bioavailability=data["bioav"],
        error_model="additive",
        sigma=data["sigma"],
        theta_init=[1.0, 10.0, 1.5],
        omega_init=[0.30, 0.30, 0.30],
        relationships=["power"],
        forward_alpha=0.05,
        backward_alpha=0.01,
    )

    # WT_on_CL should be in the selected set.
    selected_names = [s["name"] for s in result["selected"]]
    assert "WT_on_CL" in selected_names, (
        f"WT_on_CL not selected; selected = {selected_names}"
    )

    # The selected step should have a significant p-value.
    wt_step = [s for s in result["selected"] if s["name"] == "WT_on_CL"][0]
    assert wt_step["p_value"] < 0.05, f"p-value too high: {wt_step['p_value']}"
    assert wt_step["delta_ofv"] < 0.0, f"delta_OFV should be negative: {wt_step['delta_ofv']}"

    # Final OFV should improve over base.
    assert result["final_ofv"] <= result["base_ofv"], (
        f"Final OFV {result['final_ofv']} > base {result['base_ofv']}"
    )


def test_scm_rejects_noise_covariates():
    """Verify that SCM does not select a noise covariate."""
    import nextstat

    # Generate data WITHOUT covariate effect.
    data = _generate_pk_data(n_subjects=15, seed=123, wt_exponent=0.0, include_age=True)

    result = nextstat.scm(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        [data["wt_obs"], data["age_obs"]],
        ["WT", "AGE"],
        dose=data["dose"],
        bioavailability=data["bioav"],
        error_model="additive",
        sigma=data["sigma"],
        theta_init=[1.0, 10.0, 1.5],
        omega_init=[0.30, 0.30, 0.30],
        relationships=["power", "exponential"],
        forward_alpha=0.05,
        backward_alpha=0.01,
    )

    # No covariates should be selected.
    assert len(result["selected"]) == 0, (
        f"Expected 0 selected covariates for noise data, "
        f"got {[s['name'] for s in result['selected']]}"
    )


def test_scm_result_types():
    """Verify all return types are correct per the ScmResult / ScmStepResult TypedDicts."""
    import nextstat

    data = _generate_pk_data(n_subjects=20, seed=42, wt_exponent=0.75)

    result = nextstat.scm(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        [data["wt_obs"]],
        ["WT"],
        dose=data["dose"],
        bioavailability=data["bioav"],
        error_model="additive",
        sigma=data["sigma"],
        theta_init=[1.0, 10.0, 1.5],
        omega_init=[0.30, 0.30, 0.30],
    )

    # Top-level types.
    assert isinstance(result["selected"], list)
    assert isinstance(result["forward_trace"], list)
    assert isinstance(result["backward_trace"], list)
    assert isinstance(result["base_ofv"], float)
    assert isinstance(result["final_ofv"], float)
    assert isinstance(result["n_forward_steps"], int)
    assert isinstance(result["n_backward_steps"], int)
    assert isinstance(result["theta"], list)
    assert isinstance(result["omega"], list)

    # Theta values should all be float.
    for v in result["theta"]:
        assert isinstance(v, float), f"theta element is {type(v)}, expected float"

    # Omega should be a list of lists (matrix).
    for row in result["omega"]:
        assert isinstance(row, list), f"omega row is {type(row)}, expected list"
        for v in row:
            assert isinstance(v, float), f"omega element is {type(v)}, expected float"

    # Step types (check all traces).
    step_keys = {"name", "param_index", "relationship", "delta_ofv", "p_value", "coefficient", "included"}
    for trace_name in ("selected", "forward_trace", "backward_trace"):
        for step in result[trace_name]:
            assert isinstance(step, dict), f"step in {trace_name} is {type(step)}"
            assert set(step.keys()) == step_keys, (
                f"Step keys mismatch in {trace_name}: "
                f"missing={step_keys - set(step.keys())}, "
                f"extra={set(step.keys()) - step_keys}"
            )
            assert isinstance(step["name"], str)
            assert isinstance(step["param_index"], int)
            assert step["relationship"] in ("power", "proportional", "exponential")
            assert isinstance(step["delta_ofv"], float)
            assert isinstance(step["p_value"], float)
            assert isinstance(step["coefficient"], float)
            assert isinstance(step["included"], bool)
