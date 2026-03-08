import json
from pathlib import Path

import pytest
from tests._tool_contract_helpers import build_canonical_root_histogram_tool_case


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_fixture_text(rel: str) -> str:
    p = _repo_root() / rel
    return p.read_text()


def test_tools_execute_tool_envelope_smoke():
    nextstat = pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    r = execute_tool("nextstat_fit", {"workspace_json": ws, "execution": {"deterministic": True}})
    assert r.get("schema_version") == "nextstat.tool_result.v1"
    assert r.get("ok") is True, r
    assert isinstance(r.get("result"), dict)
    assert r.get("error") is None
    assert r.get("meta", {}).get("tool_name") == "nextstat_fit"
    assert r.get("meta", {}).get("nextstat_version") == getattr(nextstat, "__version__", None)


def test_read_root_histogram_supports_in_memory_bytes_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    name, args = build_canonical_root_histogram_tool_case()
    out = execute_tool(name, args)
    assert out["schema_version"] == "nextstat.tool_result.v1"
    assert out["ok"] is True, out
    assert out["meta"]["tool_name"] == name
    assert out["result"]["name"] == "hist1"
    assert out["result"]["bin_content"] == [10.0, 20.0, 30.0]
    assert out["result"]["underflow"] == 0.0
    assert out["result"]["overflow"] == 0.0


def test_toolkit_descriptor_local_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import get_tool_names, get_toolkit, get_toolkit_descriptor

    descriptor = get_toolkit_descriptor()
    assert descriptor["schema_version"] == "nextstat.tool_schema.v1"
    assert descriptor["transport"] == "local"

    tools = descriptor["tools"]
    capabilities = descriptor["capabilities"]
    guidance = descriptor["guidance"]
    assert isinstance(tools, list) and tools
    assert isinstance(capabilities, list) and capabilities
    assert isinstance(guidance, dict)
    assert isinstance(guidance["hints"], list) and guidance["hints"]
    assert isinstance(guidance["recipes"], list) and guidance["recipes"]
    assert tools == get_toolkit()
    assert [tool["function"]["name"] for tool in tools] == get_tool_names()

    capability_names = {entry["name"] for entry in capabilities}
    tool_names = {tool["function"]["name"] for tool in tools}
    assert tool_names.issubset(capability_names)

    root_hist = next(
        entry for entry in capabilities if entry["name"] == "nextstat_read_root_histogram"
    )
    assert root_hist["local_available"] is True
    assert root_hist["server_available"] is True
    assert root_hist["server_policy"]["availability"] == "exposed"
    assert root_hist["server_policy"]["reason_code"] == "server_safe_subset"

    referenced_names = {tool_name for recipe in guidance["recipes"] for tool_name in recipe["tools"]}
    assert tool_names.issubset(referenced_names)
    assert all(recipe["transport"] == "local" for recipe in guidance["recipes"])

    glm = next(entry for entry in capabilities if entry["name"] == "nextstat_glm_fit")
    assert glm["local_available"] is True
    assert glm["server_available"] is True
    assert glm["server_policy"]["availability"] == "exposed"
    assert glm["server_policy"]["reason_code"] == "server_safe_subset"

    bayesian_sample = next(
        entry for entry in capabilities if entry["name"] == "nextstat_bayesian_sample"
    )
    assert bayesian_sample["local_available"] is True
    assert bayesian_sample["server_available"] is True
    assert bayesian_sample["server_policy"]["availability"] == "exposed"
    assert bayesian_sample["server_policy"]["reason_code"] == "server_safe_subset"

    survival = next(entry for entry in capabilities if entry["name"] == "nextstat_survival_fit")
    assert survival["local_available"] is True
    assert survival["server_available"] is True
    assert survival["server_policy"]["availability"] == "exposed"
    assert survival["server_policy"]["reason_code"] == "server_safe_subset"

    km = next(entry for entry in capabilities if entry["name"] == "nextstat_kaplan_meier")
    assert km["local_available"] is True
    assert km["server_available"] is True
    assert km["server_policy"]["availability"] == "exposed"
    assert km["server_policy"]["reason_code"] == "server_safe_subset"

    log_rank = next(entry for entry in capabilities if entry["name"] == "nextstat_log_rank_test")
    assert log_rank["local_available"] is True
    assert log_rank["server_available"] is True
    assert log_rank["server_policy"]["availability"] == "exposed"
    assert log_rank["server_policy"]["reason_code"] == "server_safe_subset"

    meta = next(entry for entry in capabilities if entry["name"] == "nextstat_meta_analysis")
    assert meta["local_available"] is True
    assert meta["server_available"] is True
    assert meta["server_policy"]["availability"] == "exposed"
    assert meta["server_policy"]["reason_code"] == "server_safe_subset"

    panel_fe = next(entry for entry in capabilities if entry["name"] == "nextstat_panel_fe")
    assert panel_fe["local_available"] is True
    assert panel_fe["server_available"] is True
    assert panel_fe["server_policy"]["availability"] == "exposed"
    assert panel_fe["server_policy"]["reason_code"] == "server_safe_subset"

    did = next(entry for entry in capabilities if entry["name"] == "nextstat_did")
    assert did["local_available"] is True
    assert did["server_available"] is True
    assert did["server_policy"]["availability"] == "exposed"
    assert did["server_policy"]["reason_code"] == "server_safe_subset"

    iv_2sls = next(entry for entry in capabilities if entry["name"] == "nextstat_iv_2sls")
    assert iv_2sls["local_available"] is True
    assert iv_2sls["server_available"] is True
    assert iv_2sls["server_policy"]["availability"] == "exposed"
    assert iv_2sls["server_policy"]["reason_code"] == "server_safe_subset"

    aipw = next(entry for entry in capabilities if entry["name"] == "nextstat_aipw")
    assert aipw["local_available"] is True
    assert aipw["server_available"] is True
    assert aipw["server_policy"]["availability"] == "exposed"
    assert aipw["server_policy"]["reason_code"] == "server_safe_subset"

    event_study = next(entry for entry in capabilities if entry["name"] == "nextstat_event_study")
    assert event_study["local_available"] is True
    assert event_study["server_available"] is True
    assert event_study["server_policy"]["availability"] == "exposed"
    assert event_study["server_policy"]["reason_code"] == "server_safe_subset"

    garch = next(entry for entry in capabilities if entry["name"] == "nextstat_garch_fit")
    assert garch["local_available"] is True
    assert garch["server_available"] is True
    assert garch["server_policy"]["availability"] == "exposed"
    assert garch["server_policy"]["reason_code"] == "server_safe_subset"

    ads_cuped = next(
        entry for entry in capabilities if entry["name"] == "nextstat_ads_cuped_adjust"
    )
    assert ads_cuped["local_available"] is True
    assert ads_cuped["server_available"] is True
    assert ads_cuped["server_policy"]["availability"] == "exposed"
    assert ads_cuped["server_policy"]["reason_code"] == "server_safe_subset"

    ads_cure = next(
        entry for entry in capabilities if entry["name"] == "nextstat_ads_cure_adjust"
    )
    assert ads_cure["local_available"] is True
    assert ads_cure["server_available"] is True
    assert ads_cure["server_policy"]["availability"] == "exposed"
    assert ads_cure["server_policy"]["reason_code"] == "server_safe_subset"

    kalman = next(entry for entry in capabilities if entry["name"] == "nextstat_kalman")
    assert kalman["local_available"] is True
    assert kalman["server_available"] is True
    assert kalman["server_policy"]["availability"] == "exposed"
    assert kalman["server_policy"]["reason_code"] == "server_safe_subset"

    churn_generate_data = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_generate_data"
    )
    assert churn_generate_data["local_available"] is True
    assert churn_generate_data["server_available"] is True
    assert churn_generate_data["server_policy"]["availability"] == "exposed"
    assert churn_generate_data["server_policy"]["reason_code"] == "server_safe_subset"

    churn_risk_model = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_risk_model"
    )
    assert churn_risk_model["local_available"] is True
    assert churn_risk_model["server_available"] is True
    assert churn_risk_model["server_policy"]["availability"] == "exposed"
    assert churn_risk_model["server_policy"]["reason_code"] == "server_safe_subset"

    churn_retention = next(entry for entry in capabilities if entry["name"] == "nextstat_churn_retention")
    assert churn_retention["local_available"] is True
    assert churn_retention["server_available"] is True
    assert churn_retention["server_policy"]["availability"] == "exposed"
    assert churn_retention["server_policy"]["reason_code"] == "server_safe_subset"

    churn_diagnostics = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_diagnostics"
    )
    assert churn_diagnostics["local_available"] is True
    assert churn_diagnostics["server_available"] is True
    assert churn_diagnostics["server_policy"]["availability"] == "exposed"
    assert churn_diagnostics["server_policy"]["reason_code"] == "server_safe_subset"

    churn_cohort_matrix = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_cohort_matrix"
    )
    assert churn_cohort_matrix["local_available"] is True
    assert churn_cohort_matrix["server_available"] is True
    assert churn_cohort_matrix["server_policy"]["availability"] == "exposed"
    assert churn_cohort_matrix["server_policy"]["reason_code"] == "server_safe_subset"

    churn_bootstrap_hr = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_bootstrap_hr"
    )
    assert churn_bootstrap_hr["local_available"] is True
    assert churn_bootstrap_hr["server_available"] is True
    assert churn_bootstrap_hr["server_policy"]["availability"] == "exposed"
    assert churn_bootstrap_hr["server_policy"]["reason_code"] == "server_safe_subset"

    churn_ingest = next(entry for entry in capabilities if entry["name"] == "nextstat_churn_ingest")
    assert churn_ingest["local_available"] is True
    assert churn_ingest["server_available"] is True
    assert churn_ingest["server_policy"]["availability"] == "exposed"
    assert churn_ingest["server_policy"]["reason_code"] == "server_safe_subset"

    churn_compare = next(entry for entry in capabilities if entry["name"] == "nextstat_churn_compare")
    assert churn_compare["local_available"] is True
    assert churn_compare["server_available"] is True
    assert churn_compare["server_policy"]["availability"] == "exposed"
    assert churn_compare["server_policy"]["reason_code"] == "server_safe_subset"

    churn_uplift = next(entry for entry in capabilities if entry["name"] == "nextstat_churn_uplift")
    assert churn_uplift["local_available"] is True
    assert churn_uplift["server_available"] is True
    assert churn_uplift["server_policy"]["availability"] == "exposed"
    assert churn_uplift["server_policy"]["reason_code"] == "server_safe_subset"

    churn_uplift_survival = next(
        entry for entry in capabilities if entry["name"] == "nextstat_churn_uplift_survival"
    )
    assert churn_uplift_survival["local_available"] is True
    assert churn_uplift_survival["server_available"] is True
    assert churn_uplift_survival["server_policy"]["availability"] == "exposed"
    assert churn_uplift_survival["server_policy"]["reason_code"] == "server_safe_subset"

    pharma_fit = next(entry for entry in capabilities if entry["name"] == "nextstat_pharma_fit")
    assert pharma_fit["local_available"] is True
    assert pharma_fit["server_available"] is True
    assert pharma_fit["server_policy"]["availability"] == "exposed"
    assert pharma_fit["server_policy"]["reason_code"] == "server_safe_subset"

    pharma_vpc = next(entry for entry in capabilities if entry["name"] == "nextstat_pharma_vpc")
    assert pharma_vpc["local_available"] is True
    assert pharma_vpc["server_available"] is True
    assert pharma_vpc["server_policy"]["availability"] == "exposed"
    assert pharma_vpc["server_policy"]["reason_code"] == "server_safe_subset"

    pk_gof = next(entry for entry in capabilities if entry["name"] == "nextstat_pk_gof")
    assert pk_gof["local_available"] is True
    assert pk_gof["server_available"] is True
    assert pk_gof["server_policy"]["availability"] == "exposed"
    assert pk_gof["server_policy"]["reason_code"] == "server_safe_subset"

    pk_npde = next(entry for entry in capabilities if entry["name"] == "nextstat_pk_npde")
    assert pk_npde["local_available"] is True
    assert pk_npde["server_available"] is True
    assert pk_npde["server_policy"]["availability"] == "exposed"
    assert pk_npde["server_policy"]["reason_code"] == "server_safe_subset"

    ads_cuped = next(entry for entry in capabilities if entry["name"] == "nextstat_ads_cuped_adjust")
    assert ads_cuped["local_available"] is True
    assert ads_cuped["server_available"] is True
    assert ads_cuped["server_policy"]["availability"] == "exposed"
    assert ads_cuped["server_policy"]["reason_code"] == "server_safe_subset"

    ads_cure = next(entry for entry in capabilities if entry["name"] == "nextstat_ads_cure_adjust")
    assert ads_cure["local_available"] is True
    assert ads_cure["server_available"] is True
    assert ads_cure["server_policy"]["availability"] == "exposed"
    assert ads_cure["server_policy"]["reason_code"] == "server_safe_subset"

    trial_simulate = next(
        entry for entry in capabilities if entry["name"] == "nextstat_trial_simulate"
    )
    assert trial_simulate["local_available"] is True
    assert trial_simulate["server_available"] is True
    assert trial_simulate["server_policy"]["availability"] == "exposed"
    assert trial_simulate["server_policy"]["reason_code"] == "server_safe_subset"

    chain_ladder = next(entry for entry in capabilities if entry["name"] == "nextstat_chain_ladder")
    assert chain_ladder["local_available"] is True
    assert chain_ladder["server_available"] is True
    assert chain_ladder["server_policy"]["availability"] == "exposed"
    assert chain_ladder["server_policy"]["reason_code"] == "server_safe_subset"

    bioequivalence = next(
        entry for entry in capabilities if entry["name"] == "nextstat_bioequivalence"
    )
    assert bioequivalence["local_available"] is True
    assert bioequivalence["server_available"] is True
    assert bioequivalence["server_policy"]["availability"] == "exposed"
    assert bioequivalence["server_policy"]["reason_code"] == "server_safe_subset"

    dose_response = next(
        entry for entry in capabilities if entry["name"] == "nextstat_dose_response"
    )
    assert dose_response["local_available"] is True
    assert dose_response["server_available"] is True
    assert dose_response["server_policy"]["availability"] == "exposed"
    assert dose_response["server_policy"]["reason_code"] == "server_safe_subset"

    competing_risks = next(
        entry for entry in capabilities if entry["name"] == "nextstat_competing_risks"
    )
    assert competing_risks["local_available"] is True
    assert competing_risks["server_available"] is True
    assert competing_risks["server_policy"]["availability"] == "exposed"
    assert competing_risks["server_policy"]["reason_code"] == "server_safe_subset"

    fault_tree_mc = next(
        entry for entry in capabilities if entry["name"] == "nextstat_fault_tree_mc"
    )
    assert fault_tree_mc["local_available"] is True
    assert fault_tree_mc["server_available"] is True
    assert fault_tree_mc["server_policy"]["availability"] == "exposed"
    assert fault_tree_mc["server_policy"]["reason_code"] == "server_safe_subset"

    fault_tree_ce_is = next(
        entry for entry in capabilities if entry["name"] == "nextstat_fault_tree_ce_is"
    )
    assert fault_tree_ce_is["local_available"] is True
    assert fault_tree_ce_is["server_available"] is True
    assert fault_tree_ce_is["server_policy"]["availability"] == "exposed"
    assert fault_tree_ce_is["server_policy"]["reason_code"] == "server_safe_subset"

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        schema = json.loads(
            (
                _repo_root()
                / "docs"
                / "schemas"
                / "tools"
                / "nextstat_tool_schema_v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.validate(instance=descriptor, schema=schema)

    example = json.loads(
        (
            _repo_root()
            / "docs"
            / "specs"
            / "nextstat_tool_schema_local_v1.example.json"
        ).read_text(encoding="utf-8")
    )
    assert descriptor == example


def test_tools_hypotest_and_upper_limit_shapes_smoke():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    ht = execute_tool(
        "nextstat_hypotest",
        {"workspace_json": ws, "mu": 1.0, "execution": {"deterministic": True}},
    )
    assert ht.get("ok") is True, ht
    payload = ht["result"]
    assert set(payload.keys()) >= {"mu", "cls", "clsb", "clb"}
    assert 0.0 <= float(payload["cls"]) <= 1.0

    ul_obs = execute_tool(
        "nextstat_upper_limit",
        {"workspace_json": ws, "expected": False, "execution": {"deterministic": True}},
    )
    assert ul_obs.get("ok") is True, ul_obs
    assert "obs_limit" in ul_obs["result"]

    ul_exp = execute_tool(
        "nextstat_upper_limit",
        {"workspace_json": ws, "expected": True, "execution": {"deterministic": True}},
    )
    assert ul_exp.get("ok") is True, ul_exp
    assert "exp_limits" in ul_exp["result"]
    assert isinstance(ul_exp["result"]["exp_limits"], list)


def test_tools_discovery_asymptotic_smoke():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    r = execute_tool(
        "nextstat_discovery_asymptotic",
        {"workspace_json": ws, "execution": {"deterministic": True}},
    )
    assert r.get("ok") is True, r
    payload = r["result"]
    assert set(payload.keys()) >= {"q0", "z0", "p0", "nll_hat", "nll_mu0"}
    assert 0.0 <= float(payload["p0"]) <= 1.0


def test_pharma_fit_tool_combined_error_model_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_pharma_fit",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0, 100.0],
            "theta_init": [1.0, 5.0, 0.8],
            "omega_init": [0.2, 0.2, 0.2],
            "method": "foce",
            "model": "1cpt_oral",
            "sigma": 0.1,
            "sigma_add": 0.05,
            "error_model": "combined",
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert "theta" in out["result"]
    assert "omega" in out["result"]
    assert "sigma" in out["result"]


def test_pharma_vpc_tool_extended_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_pharma_vpc",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "omega_matrix": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
            "sigma": 0.1,
            "sigma_add": 0.05,
            "error_model": "combined",
            "bioavailability": 1.0,
            "quantiles": [0.1, 0.5, 0.9],
            "n_bins": 4,
            "n_sim": 16,
            "seed": 42,
            "pi_level": 0.9,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert "bins" in out["result"]
    assert "quantiles" in out["result"]
    assert out["result"]["n_sim"] == 16


def test_pk_gof_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_pk_gof",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "doses": [100.0, 100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "eta": [[0.05, -0.02, 0.01], [-0.04, 0.03, -0.02]],
            "sigma": 0.1,
            "error_model": "proportional",
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["model"] == "1cpt_oral"
    assert out["result"]["n_subjects"] == 2
    assert out["result"]["n_records"] == 8
    assert len(out["result"]["records"]) == 8
    assert set(out["result"]["records"][0].keys()) >= {
        "subject",
        "time",
        "dv",
        "pred",
        "ipred",
        "iwres",
        "cwres",
    }


def test_pk_npde_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_pk_npde",
        {
            "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
            "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "n_subjects": 2,
            "doses": [100.0, 100.0],
            "model": "1cpt_oral",
            "theta": [1.0, 5.0, 0.8],
            "omega_matrix": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
            "sigma": 0.1,
            "error_model": "proportional",
            "n_sim": 16,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["model"] == "1cpt_oral"
    assert out["result"]["n_subjects"] == 2
    assert out["result"]["n_records"] == 8
    assert out["result"]["n_sim"] == 16
    assert out["result"]["seed"] == 42
    assert isinstance(out["result"]["records"], list) and len(out["result"]["records"]) == 8
    assert set(out["result"]["records"][0].keys()) >= {
        "subject",
        "time",
        "dv",
        "percentile",
        "npde",
    }


def test_ads_cuped_adjust_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
    assert out.get("ok") is True, out
    assert out["result"]["method"] == "cuped"
    assert out["result"]["provenance_validated"] is True
    assert out["result"]["selected_covariates"] == ["pre_clicks"]
    assert "rho" in out["result"]


def test_ads_cure_adjust_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
    assert out.get("ok") is True, out
    assert out["result"]["method"] == "cure"
    assert out["result"]["provenance_validated"] is True
    assert out["result"]["selected_covariates"] == ["pre_clicks", "pre_impressions"]
    assert isinstance(out["result"]["theta"], list) and len(out["result"]["theta"]) == 2


def test_kalman_tool_supports_em_and_simulate_contracts():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    simulate = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert simulate.get("ok") is True, simulate
    assert len(simulate["result"]["xs"]) == 4
    assert len(simulate["result"]["ys"]) == 4

    forecast = execute_tool(
        "nextstat_kalman",
        {
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.2]],
            "x0": [0.0],
            "P0": [[1.0]],
            "y": [[1.0], [1.2], [0.9], [1.1]],
            "operation": "forecast",
            "n_ahead": 3,
            "alpha": 0.1,
            "execution": {"deterministic": True},
        },
    )
    assert forecast.get("ok") is True, forecast
    assert set(forecast["result"].keys()) >= {"state_means", "obs_means", "alpha", "z", "obs_lower", "obs_upper"}

    em = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert em.get("ok") is True, em
    assert isinstance(em["result"]["converged"], bool)
    assert em["result"]["n_iter"] == 3
    assert isinstance(em["result"]["loglik_trace"], list) and len(em["result"]["loglik_trace"]) >= 1
    assert set(em["result"].keys()) >= {"f", "h", "q", "r"}


def test_churn_generate_data_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_churn_generate_data",
        {
            "n_customers": 12,
            "n_cohorts": 3,
            "max_time": 18.0,
            "treatment_fraction": 0.25,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "n",
        "n_events",
        "times",
        "events",
        "groups",
        "treated",
        "covariates",
        "covariate_names",
        "plan",
        "region",
        "cohort",
        "usage_score",
    }
    assert out["result"]["n"] == 12
    assert out["result"]["n_events"] == 8
    assert out["result"]["covariate_names"] == [
        "plan_basic",
        "plan_premium",
        "usage_score",
        "support_tickets",
    ]
    assert len(out["result"]["times"]) == 12
    assert len(out["result"]["covariates"]) == 12
    assert len(out["result"]["plan"]) == 12
    assert len(out["result"]["region"]) == 12
    assert len(out["result"]["cohort"]) == 12


def test_log_rank_test_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_log_rank_test",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "n",
        "chi_squared",
        "df",
        "p_value",
        "group_ids",
        "observed",
        "expected",
    }
    assert out["result"]["n"] == 8
    assert out["result"]["df"] == 1
    assert out["result"]["group_ids"] == [0, 1]
    assert len(out["result"]["observed"]) == 2
    assert len(out["result"]["expected"]) == 2


def test_churn_risk_model_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "n",
        "n_events",
        "nll",
        "names",
        "coefficients",
        "se",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
    }
    assert out["result"]["n"] == 16
    assert out["result"]["n_events"] == 11
    assert out["result"]["names"] == [
        "plan_basic",
        "plan_premium",
        "usage_score",
        "support_tickets",
    ]
    assert len(out["result"]["coefficients"]) == 4
    assert len(out["result"]["hazard_ratios"]) == 4
    assert len(out["result"]["hr_ci_lower"]) == 4
    assert len(out["result"]["hr_ci_upper"]) == 4


def test_churn_compare_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_churn_compare",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "conf_level": 0.95,
            "correction": "bh",
            "alpha": 0.05,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["correction_method"] == "benjamini_hochberg"
    assert out["result"]["n"] == 8
    assert out["result"]["n_events"] == 4
    assert isinstance(out["result"]["segments"], list) and len(out["result"]["segments"]) == 2
    assert isinstance(out["result"]["pairwise"], list) and len(out["result"]["pairwise"]) == 1
    assert set(out["result"]["pairwise"][0].keys()) >= {
        "group_a",
        "group_b",
        "chi_squared",
        "p_value",
        "p_adjusted",
        "hazard_ratio_proxy",
        "median_diff",
        "significant",
    }


def test_churn_diagnostics_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "n",
        "n_events",
        "overall_censoring_frac",
        "trust_gate_passed",
        "censoring_by_segment",
        "covariate_balance",
        "propensity_overlap",
        "warnings",
    }
    assert out["result"]["n"] == 8
    assert out["result"]["n_events"] == 4
    assert isinstance(out["result"]["censoring_by_segment"], list) and len(out["result"]["censoring_by_segment"]) == 2
    assert isinstance(out["result"]["covariate_balance"], list) and len(out["result"]["covariate_balance"]) == 2
    assert isinstance(out["result"]["propensity_overlap"], dict)
    assert isinstance(out["result"]["warnings"], list) and out["result"]["warnings"]


def test_churn_cohort_matrix_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_churn_cohort_matrix",
        {
            "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
            "events": [True, False, True, False, False, True, False, True],
            "groups": [0, 0, 0, 0, 1, 1, 1, 1],
            "period_boundaries": [30.0, 60.0, 90.0],
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "period_boundaries",
        "cohorts",
        "overall",
    }
    assert out["result"]["period_boundaries"] == [30.0, 60.0, 90.0]
    assert isinstance(out["result"]["cohorts"], list) and len(out["result"]["cohorts"]) == 2
    assert out["result"]["overall"]["cohort"] == -1
    assert out["result"]["overall"]["n_total"] == 8
    assert isinstance(out["result"]["overall"]["periods"], list) and len(out["result"]["overall"]["periods"]) == 3


def test_churn_bootstrap_hr_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "names",
        "hr_point",
        "hr_ci_lower",
        "hr_ci_upper",
        "n_bootstrap",
        "n_jackknife_requested",
        "n_jackknife_attempted",
        "n_converged",
        "elapsed_s",
        "ci_method_requested",
        "ci_method_effective",
        "ci_diagnostics",
    }
    assert out["result"]["names"] == ["x1", "x2"]
    assert out["result"]["n_bootstrap"] == 8
    assert out["result"]["ci_method_requested"] == "percentile"
    assert out["result"]["n_converged"] == 8
    assert isinstance(out["result"]["ci_method_effective"], list) and len(out["result"]["ci_method_effective"]) == 2
    assert isinstance(out["result"]["ci_diagnostics"], list) and len(out["result"]["ci_diagnostics"]) == 2


def test_churn_ingest_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "n",
        "n_events",
        "times",
        "events",
        "groups",
        "treated",
        "covariates",
        "covariate_names",
        "n_dropped",
        "warnings",
    }
    assert out["result"]["n"] == 4
    assert out["result"]["n_events"] == 1
    assert out["result"]["n_dropped"] == 0
    assert out["result"]["times"] == [30.0, 45.0, 60.0, 12.0]
    assert out["result"]["events"] == [True, False, False, False]
    assert out["result"]["treated"] == [0, 1, 0, 1]
    assert out["result"]["covariate_names"] == ["usage_score", "plan_pro"]
    assert isinstance(out["result"]["warnings"], list)


def test_churn_uplift_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "ate",
        "se",
        "ci_lower",
        "ci_upper",
        "n_treated",
        "n_control",
        "gamma_critical",
        "horizon",
    }
    assert out["result"]["n_treated"] == 4
    assert out["result"]["n_control"] == 4
    assert out["result"]["horizon"] == 60.0


def test_churn_uplift_survival_tool_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert set(out["result"].keys()) >= {
        "rmst_treated",
        "rmst_control",
        "delta_rmst",
        "horizon",
        "ipw_applied",
        "arms",
        "survival_diffs",
        "overlap",
    }
    assert out["result"]["horizon"] == 60.0
    assert isinstance(out["result"]["arms"], list) and len(out["result"]["arms"]) == 2
    assert isinstance(out["result"]["survival_diffs"], list) and len(out["result"]["survival_diffs"]) == 3
    assert set(out["result"]["overlap"].keys()) >= {
        "n_total",
        "n_after_trim",
        "n_trimmed",
        "mean_propensity",
        "min_propensity",
        "max_propensity",
        "ess_treated",
        "ess_control",
    }


def test_trial_simulate_tool_extended_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
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
            "bioavailability": 0.85,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert "concentrations" in out["result"]
    assert "individual_params" in out["result"]
    assert "auc" in out["result"]
    assert "cmax" in out["result"]
    assert "tmax" in out["result"]
    assert "ctrough" in out["result"]


def test_bayesian_sample_tool_bounded_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_bayesian_sample",
        {
            "model_type": "linear_regression",
            "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "y": [1.0, 2.1, 2.9, 4.2, 5.1],
            "n_chains": 2,
            "n_warmup": 20,
            "n_samples": 20,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["model_type"] == "linear_regression"
    assert out["result"]["n_chains"] == 2
    assert out["result"]["n_warmup"] == 20
    assert out["result"]["n_samples"] == 20
    assert isinstance(out["result"]["param_names"], list) and out["result"]["param_names"]
    assert "diagnostics" in out["result"]
    assert "posterior_summary" in out["result"]
    assert "quality" in out["result"]["diagnostics"]
    assert out["result"]["diagnostics"]["quality"]["status"] in {"ok", "warn", "fail"}
    assert "intercept" in out["result"]["posterior_summary"]


def test_bayesian_sample_tool_survival_contract_uses_correct_constructor_order():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    out = execute_tool(
        "nextstat_bayesian_sample",
        {
            "model_type": "cox_ph",
            "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
            "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
            "event": [1, 1, 0, 1, 0, 1],
            "n_chains": 2,
            "n_warmup": 20,
            "n_samples": 20,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["model_type"] == "cox_ph"
    assert isinstance(out["result"]["param_names"], list) and out["result"]["param_names"]
    assert "diagnostics" in out["result"]
    assert "posterior_summary" in out["result"]


def test_bayesian_sample_tool_histfactory_bounded_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    out = execute_tool(
        "nextstat_bayesian_sample",
        {
            "model_type": "histfactory",
            "workspace_json": ws,
            "n_chains": 2,
            "n_warmup": 10,
            "n_samples": 10,
            "seed": 42,
            "execution": {"deterministic": True},
        },
    )
    assert out.get("ok") is True, out
    assert out["result"]["model_type"] == "histfactory"
    assert out["result"]["n_chains"] == 2
    assert out["result"]["n_warmup"] == 10
    assert out["result"]["n_samples"] == 10
    assert isinstance(out["result"]["param_names"], list) and out["result"]["param_names"]
    assert "diagnostics" in out["result"]
    assert "posterior_summary" in out["result"]
