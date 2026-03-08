from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _numeric_tolerance_for_path(path: str, *, default_rtol: float, default_atol: float) -> tuple[float, float]:
    # Cross-platform BLAS/LAPACK differences can perturb pharma optimizer
    # outputs even when the fit itself is correct.  Keep relaxations tightly
    # scoped to known-sensitive fields.
    if "nextstat_pharma_fit" in path:
        if ".result.eta[" in path:
            return (0.0, 5e-3)
        if ".result.omega[" in path or ".result.omega_matrix[" in path:
            return (5e-3, 1e-4)
        if ".result.sigma" in path and ".result.sigma_init" not in path:
            return (5e-3, 1e-4)
    return (default_rtol, default_atol)


def assert_json_close(a: Any, b: Any, *, rtol: float, atol: float, path: str = "$") -> None:
    if _is_number(a) and _is_number(b):
        rtol, atol = _numeric_tolerance_for_path(path, default_rtol=rtol, default_atol=atol)
        af = float(a)
        bf = float(b)
        diff = abs(af - bf)
        if diff <= atol:
            return
        denom = max(abs(af), abs(bf), 1.0)
        if diff / denom <= rtol:
            return
        raise AssertionError(f"{path}: {af} != {bf} (diff={diff}, rtol={rtol}, atol={atol})")

    if type(a) is not type(b):
        raise AssertionError(f"{path}: type mismatch {type(a)} != {type(b)}")

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"{path}: keys mismatch {set(a.keys())} != {set(b.keys())}")
        for k in a.keys():
            assert_json_close(a[k], b[k], rtol=rtol, atol=atol, path=f"{path}.{k}")
        return

    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length mismatch {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            assert_json_close(ai, bi, rtol=rtol, atol=atol, path=f"{path}[{i}]")
        return

    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def assert_pharma_saem_acceptance_envelope(
    result: dict[str, Any],
    *,
    theta_max_magnitude: float = 1e6,
) -> None:
    """Validate SAEM result via scientific acceptance envelope.

    SAEM is stochastic — exact cross-platform snapshot parity is NOT a valid
    release criterion.  Cross-platform release gates must use this envelope
    instead of ``assert_json_close`` on raw omega/eta/sigma values.

    Acceptance criteria
    -------------------
    1. ``converged`` is True
    2. OFV is finite
    3. All theta entries are finite and |theta| < *theta_max_magnitude*
    4. All omega (SD) entries are finite and positive
    5. omega_matrix diagonal entries are finite and positive (PSD proxy)
    """
    assert result.get("converged") is True, (
        f"SAEM must converge, got converged={result.get('converged')}"
    )

    ofv = result.get("ofv")
    assert ofv is not None and math.isfinite(ofv), f"OFV must be finite, got {ofv}"

    theta = result.get("theta", [])
    assert len(theta) >= 1, "theta must be non-empty"
    for i, t in enumerate(theta):
        assert math.isfinite(t), f"theta[{i}] must be finite, got {t}"
        assert abs(t) < theta_max_magnitude, (
            f"theta[{i}] magnitude unreasonable: {t}"
        )

    omega = result.get("omega", [])
    for i, o in enumerate(omega):
        assert math.isfinite(o), f"omega[{i}] (SD) must be finite, got {o}"
        assert o > 0, f"omega[{i}] (SD) must be positive, got {o}"

    omega_matrix = result.get("omega_matrix")
    if omega_matrix:
        for i, row in enumerate(omega_matrix):
            diag = row[i]
            assert math.isfinite(diag), f"omega_matrix[{i}][{i}] must be finite"
            assert diag > 0, f"omega_matrix[{i}][{i}] must be positive, got {diag}"


def normalize_tool_envelope(x: dict[str, Any]) -> dict[str, Any]:
    # Contract comparisons should focus on semantically stable fields rather
    # than build metadata or optimizer bookkeeping.
    out = json.loads(json.dumps(x))
    meta = out.get("meta", {})
    if isinstance(meta, dict):
        meta.pop("nextstat_version", None)
        meta.pop("threads_applied", None)
        meta.pop("device", None)

    def _drop_unstable_fields(v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            v.pop("n_iter", None)
            v.pop("wall_time_s", None)
            v.pop("elapsed_s", None)
            v.pop("scenarios_per_sec", None)
            v.pop("mu_values", None)
            return {k: _drop_unstable_fields(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_drop_unstable_fields(vv) for vv in v]
        return v

    return _drop_unstable_fields(out)


def build_canonical_semantic_tool_chain(workspace_json: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("nextstat_workspace_audit", {"workspace_json": workspace_json}),
        ("nextstat_fit", {"workspace_json": workspace_json}),
        ("nextstat_discovery_asymptotic", {"workspace_json": workspace_json}),
        ("nextstat_hypotest", {"workspace_json": workspace_json, "mu": 1.0}),
        ("nextstat_upper_limit", {"workspace_json": workspace_json, "expected": True}),
        (
            "nextstat_scan",
            {
                "workspace_json": workspace_json,
                "start": 0.0,
                "stop": 2.0,
                "points": 5,
            },
        ),
        ("nextstat_ranking", {"workspace_json": workspace_json, "top_n": 5}),
    ]


def build_canonical_glm_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_glm_fit",
        {
            "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "y": [1.1, 2.9, 5.2, 6.8, 9.1],
            "family": "linear",
            "include_intercept": True,
            "l2": 0.5,
        },
    )


def build_canonical_bayesian_sample_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_bayesian_sample",
        {
            "model_type": "linear_regression",
            "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "y": [1.0, 2.1, 2.9, 4.2, 5.1],
            "n_chains": 2,
            "n_warmup": 20,
            "n_samples": 20,
            "seed": 42,
            "target_accept": 0.8,
        },
    )


def build_canonical_survival_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_survival_fit",
        {
            "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
            "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
            "event": [1, 1, 0, 1, 0, 1],
            "model": "cox_ph",
        },
    )


def build_canonical_kaplan_meier_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_kaplan_meier",
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 1, 0, 1, 0, 1],
            "group": [0, 0, 1, 1, 0, 1],
        },
    )


def build_canonical_log_rank_test_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_log_rank_test",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
        },
    )


def build_canonical_root_histogram_tool_case() -> tuple[str, dict[str, Any]]:
    fixture = (_repo_root() / "tests" / "fixtures" / "simple_histos.root").read_bytes()
    return (
        "nextstat_read_root_histogram",
        {
            "root_bytes_base64": base64.b64encode(fixture).decode("ascii"),
            "filename_hint": "simple_histos.root",
            "hist_path": "hist1",
        },
    )


def build_canonical_meta_analysis_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_meta_analysis",
        {
            "effects": [0.2, 0.5, -0.1, 0.3],
            "standard_errors": [0.1, 0.2, 0.15, 0.12],
            "method": "random",
        },
    )


def build_canonical_panel_fe_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_panel_fe",
        {
            "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
            "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
            "entity": [0, 0, 1, 1, 2, 2],
            "time": [0, 1, 0, 1, 0, 1],
            "cluster": "two_way",
        },
    )


def build_canonical_did_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_did",
        {
            "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
            "treat": [0, 0, 0, 0, 1, 1, 1, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "entity": [0, 0, 1, 1, 2, 2, 3, 3],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "cluster": "two_way",
        },
    )


def build_canonical_iv_2sls_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_iv_2sls",
        {
            "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
            "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
            "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
            "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
            "cov": "hc1",
        },
    )


def build_canonical_aipw_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_aipw",
        {
            "x": [[0.0], [1.0], [0.0], [1.0], [0.2], [1.2], [0.1], [1.1]],
            "y": [1.0, 2.0, 1.2, 2.2, 1.1, 2.4, 1.3, 2.5],
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
            "estimand": "ate",
        },
    )


def build_canonical_event_study_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_event_study",
        {
            "y": [1.0, 1.1, 1.2, 1.4, 1.0, 1.0, 1.1, 1.2, 2.0, 2.3, 2.7, 3.0],
            "entity": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "treat_time": [None, None, None, None, 2, 2, 2, 2, 1, 1, 1, 1],
            "n_leads": 2,
            "n_lags": 2,
            "cluster": "entity",
        },
    )


def build_canonical_garch_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_garch_fit",
        {
            "returns": [0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.012, -0.008],
            "model": "garch",
        },
    )


def build_canonical_kalman_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_kalman",
        {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.2]],
            "x0": [0.0],
            "P0": [[1.0]],
            "y": [[1.0], [1.2], [0.9], [1.1]],
            "operation": "filter",
        },
    )


def build_canonical_kalman_simulate_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_kalman",
        {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.2]],
            "x0": [0.0],
            "P0": [[1.0]],
            "operation": "simulate",
            "t_max": 4,
            "seed": 42,
            "init": "mean",
        },
    )


def build_canonical_kalman_em_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_kalman",
        {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.2]],
            "x0": [0.0],
            "P0": [[1.0]],
            "y": [[0.1], [0.2], [0.0]],
            "operation": "em",
            "max_iter": 3,
            "tol": 1e-6,
            "estimate_q": True,
            "estimate_r": True,
            "estimate_f": False,
            "estimate_h": False,
            "min_diag": 1e-8,
        },
    )


def build_canonical_churn_generate_data_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_generate_data",
        {
            "n_customers": 12,
            "n_cohorts": 3,
            "max_time": 18.0,
            "treatment_fraction": 0.25,
            "seed": 42,
        },
    )


def build_canonical_churn_risk_model_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_risk_model",
        {
            "times": [
                18.0,
                3.302151210546639,
                4.329991875135302,
                5.475084445095048,
                18.0,
                16.76730374185681,
                12.884032820472772,
                8.271505063122506,
                3.632766793868914,
                18.0,
                12.32888547551182,
                18.0,
                18.0,
                1.4580664577117772,
                7.416636560585132,
                4.245245279862313,
            ],
            "events": [
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                True,
            ],
            "covariates": [
                [1.0, 0.0, 1.3117235495872972, 2.0],
                [0.0, 0.0, -0.9022021759009674, 0.0],
                [1.0, 0.0, 0.528865100865726, 2.0],
                [1.0, 0.0, -1.454760464783373, 3.0],
                [0.0, 1.0, -1.5385144547016696, 1.0],
                [1.0, 0.0, 1.5124745460816813, 1.0],
                [1.0, 0.0, -0.5771217015082334, 1.0],
                [0.0, 0.0, 1.189054183234974, 0.0],
                [0.0, 0.0, -0.4320950454679947, 0.0],
                [1.0, 0.0, 1.147124602981333, 0.0],
                [0.0, 0.0, -0.3221810058112955, 2.0],
                [0.0, 1.0, -0.5197845574589398, 3.0],
                [0.0, 1.0, 0.937777824579468, 1.0],
                [0.0, 0.0, -1.6269177382249136, 1.0],
                [1.0, 0.0, 0.8667568109417092, 0.0],
                [0.0, 1.0, 0.17866086448304247, 1.0],
            ],
            "names": [
                "plan_basic",
                "plan_premium",
                "usage_score",
                "support_tickets",
            ],
            "conf_level": 0.95,
        },
    )


def build_canonical_churn_retention_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_retention",
        {
            "times": [30.0, 60.0, 90.0, 120.0],
            "events": [True, False, True, False],
            "groups": [0, 0, 1, 1],
            "conf_level": 0.95,
        },
    )


def build_canonical_churn_compare_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_compare",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "conf_level": 0.95,
            "correction": "benjamini_hochberg",
            "alpha": 0.05,
        },
    )


def build_canonical_churn_uplift_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_uplift",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "treated": [0, 0, 1, 1, 0, 1, 0, 1],
            "covariates": [
                [1.0, 0.2],
                [0.5, -0.1],
                [1.2, 0.4],
                [0.7, 0.0],
                [0.3, -0.2],
                [1.1, 0.1],
                [0.4, -0.3],
                [0.9, 0.3],
            ],
            "horizon": 60.0,
        },
    )


def build_canonical_churn_diagnostics_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_diagnostics",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "treated": [0, 0, 1, 1, 0, 1, 0, 1],
            "covariates": [
                [1.0, 0.2],
                [0.5, -0.1],
                [1.2, 0.4],
                [0.7, 0.0],
                [0.3, -0.2],
                [1.1, 0.1],
                [0.4, -0.3],
                [0.9, 0.3],
            ],
            "covariate_names": ["x1", "x2"],
            "trim": 0.01,
        },
    )


def build_canonical_churn_cohort_matrix_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_cohort_matrix",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "period_boundaries": [30.0, 60.0, 90.0],
        },
    )


def build_canonical_churn_bootstrap_hr_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_bootstrap_hr",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "covariates": [
                [1.0, 0.2],
                [0.5, -0.1],
                [1.2, 0.4],
                [0.7, 0.0],
                [0.3, -0.2],
                [1.1, 0.1],
                [0.4, -0.3],
                [0.9, 0.3],
            ],
            "names": ["x1", "x2"],
            "n_bootstrap": 8,
            "seed": 42,
            "conf_level": 0.95,
            "ci_method": "percentile",
            "n_jackknife": 4,
        },
    )


def build_canonical_churn_ingest_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_ingest",
        {
            "times": [30.0, 45.0, 90.0, 12.0],
            "events": [True, False, True, False],
            "groups": [0, 0, 1, 1],
            "treated": [0, 1, 0, 1],
            "covariates": [
                [0.2, 1.0],
                [0.3, 0.0],
                [0.8, 1.0],
                [0.5, 0.0],
            ],
            "covariate_names": ["usage_score", "plan_pro"],
            "observation_end": 60.0,
        },
    )


def build_canonical_churn_uplift_survival_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_churn_uplift_survival",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "treated": [0, 0, 1, 1, 0, 1, 0, 1],
            "covariates": [
                [1.0, 0.2],
                [0.5, -0.1],
                [1.2, 0.4],
                [0.7, 0.0],
                [0.3, -0.2],
                [1.1, 0.1],
                [0.4, -0.3],
                [0.9, 0.3],
            ],
            "horizon": 60.0,
            "eval_horizons": [30.0, 60.0, 90.0],
            "trim": 0.01,
        },
    )


def build_canonical_chain_ladder_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_chain_ladder",
        {
            "triangle": [
                [100.0, 150.0, 180.0],
                [110.0, 160.0, None],
                [120.0, None, None],
            ],
            "method": "mack",
            "conf_level": 0.95,
        },
    )


def build_canonical_pharma_fit_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_pharma_fit",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0, 100.0],
            "theta_init": [1.0, 5.0, 0.8],
            "omega_init": [0.2, 0.2, 0.2],
            "method": "focei",
            "model": "1cpt_oral",
            "sigma": 0.1,
            "error_model": "proportional",
        },
    )


def build_canonical_pharma_vpc_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_pharma_vpc",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0, 100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "omega_matrix": [
                [0.04, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.04],
            ],
            "sigma": 0.1,
            "error_model": "proportional",
            "n_sim": 16,
            "seed": 42,
        },
    )


def build_canonical_pk_gof_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_pk_gof",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "doses": [100.0, 100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "eta": [
                [0.05, -0.02, 0.01],
                [-0.04, 0.03, -0.02],
            ],
            "sigma": 0.1,
            "error_model": "proportional",
        },
    )


def build_canonical_pk_npde_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_pk_npde",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0, 100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "omega_matrix": [
                [0.04, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.04],
            ],
            "sigma": 0.1,
            "error_model": "proportional",
            "n_sim": 16,
            "seed": 42,
        },
    )


def build_canonical_trial_simulate_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_trial_simulate",
        {
            "n_subjects": 3,
            "dose": 100.0,
            "obs_times": [0.5, 1.0, 2.0, 4.0],
            "pk_model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "omega": [0.2, 0.2, 0.2],
            "sigma": 0.1,
            "error_model": "proportional",
            "bioavailability": 1.0,
            "seed": 42,
        },
    )


def build_canonical_bioequivalence_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_bioequivalence",
        {
            "operation": "test",
            "test_values": [4.58, 4.61, 4.55, 4.62],
            "ref_values": [4.60, 4.63, 4.57, 4.64],
        },
    )


def build_canonical_ads_cuped_adjust_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_ads_cuped_adjust",
        {
            "control_outcomes": [10.0, 12.0, 9.0, 11.0, 13.0],
            "control_covariates": [8.0, 10.0, 7.0, 9.0, 11.0],
            "variant_outcomes": [11.0, 13.0, 10.0, 12.0, 14.0],
            "variant_covariates": [8.5, 10.5, 7.5, 9.5, 11.5],
            "covariate_name": "pre_clicks",
            "covariate_provenance": {
                "name": "pre_clicks",
                "timing": "pre_treatment",
                "source_dataset": "ads_preperiod_daily",
            },
            "pre_treatment_only": True,
        },
    )


def build_canonical_ads_cure_adjust_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_ads_cure_adjust",
        {
            "control_outcomes": [100.0, 110.0, 95.0, 105.0, 115.0, 120.0],
            "control_covariates": [
                [80.0, 1000.0],
                [88.0, 1100.0],
                [75.0, 950.0],
                [84.0, 1025.0],
                [92.0, 1150.0],
                [96.0, 1180.0],
            ],
            "variant_outcomes": [104.0, 113.0, 99.0, 109.0, 118.0, 124.0],
            "variant_covariates": [
                [81.0, 1008.0],
                [89.0, 1110.0],
                [76.0, 960.0],
                [85.0, 1035.0],
                [93.0, 1165.0],
                [97.0, 1192.0],
            ],
            "covariate_names": ["pre_clicks", "pre_impressions"],
            "covariate_provenance": [
                {
                    "name": "pre_clicks",
                    "timing": "pre_treatment",
                    "source_dataset": "ads_preperiod_daily",
                },
                {
                    "name": "pre_impressions",
                    "timing": "pre_treatment",
                    "source_dataset": "ads_preperiod_daily",
                },
            ],
            "pre_treatment_only": True,
        },
    )


def build_canonical_dose_response_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_dose_response",
        {
            "model": "emax",
            "e0": 5.0,
            "emax": 50.0,
            "ec50": 2.0,
            "conc": [0.0, 1.0, 2.0, 4.0, 8.0],
        },
    )


def build_canonical_competing_risks_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_competing_risks",
        {
            "operation": "cif",
            "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "events": [1, 2, 0, 1, 2, 1],
            "target_cause": 1,
            "conf_level": 0.95,
        },
    )


def build_canonical_fault_tree_mc_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_fault_tree_mc",
        {
            "spec": {
                "components": [
                    {"type": "bernoulli", "p": 0.01},
                    {"type": "bernoulli", "p": 0.02},
                ],
                "nodes": [
                    {"type": "component", "index": 0},
                    {"type": "component", "index": 1},
                    {"type": "or", "children": [0, 1]},
                ],
                "top_event": 2,
            },
            "n_scenarios": 10000,
            "seed": 42,
            "device": "cpu",
        },
    )


def build_canonical_fault_tree_ce_is_tool_case() -> tuple[str, dict[str, Any]]:
    return (
        "nextstat_fault_tree_ce_is",
        {
            "spec": {
                "components": [
                    {"type": "bernoulli", "p": 0.01},
                    {"type": "bernoulli", "p": 0.02},
                ],
                "nodes": [
                    {"type": "component", "index": 0},
                    {"type": "component", "index": 1},
                    {"type": "and", "children": [0, 1]},
                ],
                "top_event": 2,
            },
            "n_per_level": 1000,
            "elite_fraction": 0.02,
            "max_levels": 6,
            "q_max": 0.9,
            "seed": 42,
        },
    )
