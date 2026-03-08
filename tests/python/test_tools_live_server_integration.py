from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from tests._bootstrap_renderer_assertions import (
    LIVE_PROVIDER_PAYLOAD_RENDERER_ASSERTIONS,
    LIVE_PROVIDER_TOOL_NAME_EXTRACTORS,
    assert_live_provider_payload_renderer,
    tool_names_for_live_provider_payload,
)
from tests._bootstrap_profile_manifest_helpers import (
    bootstrap_provider_example_path,
    get_runnable_example_profiles as _runnable_example_profiles,
    get_runnable_template_registry as _runnable_template_registry,
    repo_root as _repo_root,
)
from tests._tool_contract_helpers import (
    assert_json_close,
    build_canonical_ads_cuped_adjust_tool_case,
    build_canonical_ads_cure_adjust_tool_case,
    build_canonical_aipw_tool_case,
    build_canonical_bayesian_sample_tool_case,
    build_canonical_bioequivalence_tool_case,
    build_canonical_churn_bootstrap_hr_tool_case,
    build_canonical_churn_generate_data_tool_case,
    build_canonical_churn_risk_model_tool_case,
    build_canonical_chain_ladder_tool_case,
    build_canonical_churn_cohort_matrix_tool_case,
    build_canonical_churn_compare_tool_case,
    build_canonical_churn_diagnostics_tool_case,
    build_canonical_churn_ingest_tool_case,
    build_canonical_churn_retention_tool_case,
    build_canonical_churn_uplift_tool_case,
    build_canonical_churn_uplift_survival_tool_case,
    build_canonical_competing_risks_tool_case,
    build_canonical_did_tool_case,
    build_canonical_dose_response_tool_case,
    build_canonical_event_study_tool_case,
    build_canonical_fault_tree_ce_is_tool_case,
    build_canonical_fault_tree_mc_tool_case,
    build_canonical_garch_tool_case,
    build_canonical_glm_tool_case,
    build_canonical_iv_2sls_tool_case,
    build_canonical_kalman_em_tool_case,
    build_canonical_kalman_simulate_tool_case,
    build_canonical_kalman_tool_case,
    build_canonical_kaplan_meier_tool_case,
    build_canonical_log_rank_test_tool_case,
    build_canonical_meta_analysis_tool_case,
    build_canonical_panel_fe_tool_case,
    build_canonical_pharma_fit_tool_case,
    build_canonical_pk_gof_tool_case,
    build_canonical_pk_npde_tool_case,
    build_canonical_pharma_vpc_tool_case,
    build_canonical_root_histogram_tool_case,
    build_canonical_semantic_tool_chain,
    build_canonical_survival_tool_case,
    build_canonical_trial_simulate_tool_case,
    normalize_tool_envelope,
)


pytestmark = [pytest.mark.slow, pytest.mark.live_server]

LIVE_METRICS: dict[str, float] = {}


def _workspace_text() -> str:
    return (_repo_root() / "tests" / "fixtures" / "simple_workspace.json").read_text(
        encoding="utf-8"
    )


def _simplified_workspace_text() -> str:
    return (_repo_root() / "tests" / "fixtures" / "sl_covariance_three_bin.json").read_text(
        encoding="utf-8"
    )


def _load_fixture_text(rel: str) -> str:
    return (_repo_root() / rel).read_text(encoding="utf-8")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_live_metric(name: str, value: float) -> None:
    LIVE_METRICS[name] = round(float(value), 6)


def _post_raw_json(
    url: str,
    body: bytes,
    *,
    api_key: str,
    content_type: str = "application/json",
) -> tuple[int, object]:
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = resp.read().decode("utf-8")
            return resp.status, json.loads(data)
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8")
        return exc.code, json.loads(data)


def _assert_server_process_alive(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        logs = proc.stdout.read() if proc.stdout is not None else ""
        raise AssertionError(f"live nextstat-server exited unexpectedly:\n{logs}")


@pytest.fixture(scope="module")
def server_binary() -> Path:
    if os.environ.get("NS_RUN_LIVE_SERVER") != "1":
        pytest.skip("Set NS_RUN_LIVE_SERVER=1 to run live nextstat-server integration tests.")
    pytest.importorskip("nextstat")
    LIVE_METRICS.clear()
    return _build_server_binary()


@pytest.fixture(scope="module", autouse=True)
def emit_live_metrics() -> None:
    yield
    metrics_path = os.environ.get("NEXTSTAT_TOOL_CONTRACT_LIVE_METRICS_PATH")
    if not metrics_path:
        return
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "nextstat.tool_contract_live_metrics.v1",
                "metrics": LIVE_METRICS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def live_server_runtime(server_binary: Path) -> dict[str, object]:
    api_key = "live-secret-key"
    with _live_server(binary=server_binary, api_key=api_key) as (server_url, proc):
        yield {
            "api_key": api_key,
            "server_url": server_url,
            "workspace_json": _workspace_text(),
            "proc": proc,
        }
        _assert_server_process_alive(proc)


@pytest.fixture
def rate_limited_live_server_runtime(server_binary: Path) -> dict[str, object]:
    api_key = "live-secret-key"
    with _live_server(binary=server_binary, api_key=api_key, rate_limit=1) as (server_url, proc):
        yield {
            "api_key": api_key,
            "server_url": server_url,
            "proc": proc,
        }
        _assert_server_process_alive(proc)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cargo_target_dir() -> Path:
    override = os.environ.get("CARGO_TARGET_DIR")
    if override:
        return Path(override).resolve()
    cargo_config = _repo_root() / ".cargo" / "config.toml"
    data = tomllib.loads(cargo_config.read_text(encoding="utf-8"))
    rel = data["build"]["target-dir"]
    return (_repo_root() / rel).resolve()


def _server_binary_path() -> Path:
    name = "nextstat-server.exe" if os.name == "nt" else "nextstat-server"
    return _cargo_target_dir() / "debug" / name


def _build_server_binary() -> Path:
    override = os.environ.get("NEXTSTAT_SERVER_BINARY")
    if override:
        binary = Path(override).resolve()
        assert binary.exists(), f"expected server binary override at {binary}"
        _record_live_metric("server_build_duration_s", 0.0)
        return binary
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for live nextstat-server integration tests")
    started = time.perf_counter()
    subprocess.run(
        ["cargo", "build", "-q", "-p", "ns-server"],
        cwd=_repo_root(),
        check=True,
    )
    _record_live_metric("server_build_duration_s", time.perf_counter() - started)
    binary = _server_binary_path()
    assert binary.exists(), f"expected server binary at {binary}"
    return binary


def _wait_for_health(base_url: str, *, timeout_s: float, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + timeout_s
    last_error = "server not started"
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{base_url}/v1/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
                last_error = f"unexpected health status {resp.status}"
        except Exception as exc:  # pragma: no cover - best-effort polling
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise RuntimeError(last_error)


@contextmanager
def _live_server(
    *,
    binary: Path,
    api_key: str,
    rate_limit: int = 0,
) -> tuple[str, subprocess.Popen[str]]:
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    with tempfile.TemporaryDirectory(prefix="nextstat-server-auth-") as tmpdir:
        key_file = Path(tmpdir) / "api_keys.txt"
        key_file.write_text(f"{api_key}\n", encoding="utf-8")

        proc = subprocess.Popen(
            [
                str(binary),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--api-keys",
                str(key_file),
                "--rate-limit",
                str(rate_limit),
            ],
            cwd=_repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "RUST_LOG": "error"},
        )
        try:
            started = time.perf_counter()
            _wait_for_health(base_url, timeout_s=30.0, proc=proc)
            _record_live_metric("server_startup_duration_s", time.perf_counter() - started)
            yield base_url, proc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)


def _assert_live_server_semantic_parity(
    *,
    workspace_json: str,
    server_url: str,
    api_key: str,
) -> None:
    from nextstat.tools import execute_tool

    execution = {"deterministic": True, "eval_mode": "parity", "threads": 1}

    for name, tool_args in build_canonical_semantic_tool_chain(workspace_json):
        local_args = dict(tool_args)
        local_args["execution"] = dict(execution)
        server_args = dict(tool_args)
        server_args["execution"] = dict(execution)

        local = execute_tool(name, local_args, transport="local")
        server = execute_tool(
            name,
            server_args,
            transport="server",
            server_url=server_url,
            api_key=api_key,
            timeout_s=15.0,
            fallback_to_local=False,
        )

        assert local["ok"] is True, local
        assert server["ok"] is True, server
        assert_json_close(
            normalize_tool_envelope(server),
            normalize_tool_envelope(local),
            rtol=1e-6,
            atol=1e-8,
            path=f"tool:{name}",
        )


def test_python_client_matches_live_auth_enabled_nextstat_server(
    live_server_runtime: dict[str, object],
):
    from nextstat.tools import ServerHTTPError, execute_tool, get_toolkit_descriptor

    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])
    ws = str(live_server_runtime["workspace_json"])

    with pytest.raises(ServerHTTPError):
        get_toolkit_descriptor(transport="server", server_url=server_url, timeout_s=5.0)

    started = time.perf_counter()
    descriptor = get_toolkit_descriptor(
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
    )
    _record_live_metric("tools_schema_get_duration_s", time.perf_counter() - started)
    assert descriptor["schema_version"] == "nextstat.tool_schema.v1"
    assert descriptor["transport"] == "server"
    tool_names = [tool["function"]["name"] for tool in descriptor["tools"]]
    assert "nextstat_fit" in tool_names
    assert "nextstat_read_root_histogram" in tool_names
    assert "nextstat_glm_fit" in tool_names
    assert "nextstat_bayesian_sample" in tool_names
    assert "nextstat_survival_fit" in tool_names
    assert "nextstat_kaplan_meier" in tool_names
    assert "nextstat_log_rank_test" in tool_names
    assert "nextstat_meta_analysis" in tool_names
    assert "nextstat_panel_fe" in tool_names
    assert "nextstat_did" in tool_names
    assert "nextstat_iv_2sls" in tool_names
    assert "nextstat_aipw" in tool_names
    assert "nextstat_event_study" in tool_names
    assert "nextstat_garch_fit" in tool_names
    assert "nextstat_ads_cuped_adjust" in tool_names
    assert "nextstat_ads_cure_adjust" in tool_names
    assert "nextstat_kalman" in tool_names
    assert "nextstat_churn_generate_data" in tool_names
    assert "nextstat_churn_risk_model" in tool_names
    assert "nextstat_churn_retention" in tool_names
    assert "nextstat_churn_diagnostics" in tool_names
    assert "nextstat_churn_cohort_matrix" in tool_names
    assert "nextstat_churn_bootstrap_hr" in tool_names
    assert "nextstat_churn_ingest" in tool_names
    assert "nextstat_churn_compare" in tool_names
    assert "nextstat_churn_uplift" in tool_names
    assert "nextstat_churn_uplift_survival" in tool_names
    assert "nextstat_chain_ladder" in tool_names
    assert "nextstat_pk_gof" in tool_names
    assert "nextstat_pk_npde" in tool_names
    assert "nextstat_bioequivalence" in tool_names
    assert "nextstat_dose_response" in tool_names
    assert "nextstat_competing_risks" in tool_names
    assert "nextstat_fault_tree_mc" in tool_names
    assert "nextstat_fault_tree_ce_is" in tool_names
    assert isinstance(descriptor["guidance"], dict)
    assert isinstance(descriptor["guidance"]["hints"], list) and descriptor["guidance"]["hints"]
    assert isinstance(descriptor["guidance"]["recipes"], list) and descriptor["guidance"]["recipes"]
    assert all(recipe["transport"] == "server" for recipe in descriptor["guidance"]["recipes"])

    unauthorized = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        timeout_s=5.0,
        fallback_to_local=False,
    )
    assert unauthorized["ok"] is False
    assert unauthorized["error"]["type"] == "ServerHTTPError"
    assert "HTTP 401" in unauthorized["error"]["message"]

    started = time.perf_counter()
    audit = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
        fallback_to_local=False,
    )
    _record_live_metric("workspace_audit_duration_s", time.perf_counter() - started)
    assert audit["schema_version"] == "nextstat.tool_result.v1"
    assert audit["ok"] is True
    assert audit["meta"]["tool_name"] == "nextstat_workspace_audit"

    started = time.perf_counter()
    fit = execute_tool(
        "nextstat_fit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=10.0,
        fallback_to_local=False,
    )
    _record_live_metric("fit_duration_s", time.perf_counter() - started)
    assert fit["schema_version"] == "nextstat.tool_result.v1"
    assert fit["ok"] is True
    assert fit["meta"]["tool_name"] == "nextstat_fit"
    assert fit["meta"]["deterministic"] is True
    assert fit["meta"]["eval_mode"] == "parity"
    assert fit["result"]["converged"] is True
    assert fit["result"]["poi_index"] == 0
    assert "mu" in fit["result"]["parameters"]

    env = dict(os.environ)
    env["NEXTSTAT_TOOLS_SERVER_URL"] = server_url
    env["NEXTSTAT_TOOLS_API_KEY"] = api_key

    descriptor_script = subprocess.run(
        [sys.executable, "scripts/validate_tool_schema_descriptor.py", "--transport", "server"],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        descriptor_script.returncode == 0
    ), descriptor_script.stderr or descriptor_script.stdout
    assert "Validated server tool descriptor" in descriptor_script.stdout


def test_live_server_workspace_audit_accepts_simplified_likelihood(
    live_server_runtime: dict[str, object],
):
    from nextstat.tools import execute_tool

    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])
    ws = _load_fixture_text("tests/fixtures/sl_covariance_three_bin.json")

    audit = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=10.0,
        fallback_to_local=False,
    )

    assert audit["schema_version"] == "nextstat.tool_result.v1"
    assert audit["ok"] is True
    assert audit["meta"]["tool_name"] == "nextstat_workspace_audit"
    assert audit["result"]["schema_version"] == "nextstat_simplified_likelihood_audit_v0"
    assert audit["result"]["input_schema_version"] == "nextstat_simplified_likelihood_v0"
    assert audit["result"]["uncertainty_model_kind"] == "covariance"
    assert (
        audit["result"]["diagnostics"]["factorization"]["method"]
        == "symmetric_eigendecomposition"
    )

    fit = execute_tool(
        "nextstat_fit",
        {
            "workspace_json": str(live_server_runtime["workspace_json"]),
            "execution": {"deterministic": True},
        },
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=10.0,
        fallback_to_local=False,
    )
    assert fit["ok"] is True

    env = dict(os.environ)
    env["NEXTSTAT_TOOLS_SERVER_URL"] = server_url
    env["NEXTSTAT_TOOLS_API_KEY"] = api_key
    smoke_script = subprocess.run(
        [
            sys.executable,
            "scripts/tool_call_smoke.py",
            "--transport",
            "server",
            "--timeout-s",
            "10",
            "--no-fallback",
        ],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_script.returncode == 0, smoke_script.stderr or smoke_script.stdout
    assert "== nextstat_fit ==" in smoke_script.stdout
    assert "nextstat.tool_result.v1" in smoke_script.stdout

    with TemporaryDirectory(prefix="nextstat-live-e2e-") as tmpdir:
        started = time.perf_counter()
        e2e_script = subprocess.run(
            [
                sys.executable,
                "scripts/e2e_discovery.py",
                "--workspace",
                "tests/fixtures/simple_workspace.json",
                "--out-dir",
                tmpdir,
                "--transport",
                "server",
                "--timeout-s",
                "10",
                "--scan-points",
                "3",
                "--ranking-top-n",
                "1",
                "--no-fallback",
            ],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        _record_live_metric("e2e_discovery_duration_s", time.perf_counter() - started)
        assert e2e_script.returncode == 0, e2e_script.stderr or e2e_script.stdout
        summary_path = Path(tmpdir, "summary.json")
        calls_path = Path(tmpdir, "calls.json")
        assert summary_path.exists(), "e2e_discovery must write summary.json"
        assert calls_path.exists(), "e2e_discovery must write calls.json"
        summary = _load_json(summary_path)
        calls = _load_json(calls_path)
        expected_names = [
            "nextstat_workspace_audit",
            "nextstat_fit",
            "nextstat_discovery_asymptotic",
            "nextstat_hypotest",
            "nextstat_upper_limit",
            "nextstat_scan",
            "nextstat_ranking",
        ]
        call_names = [call["name"] for call in calls]
        assert summary["schema_version"] == "nextstat.e2e_discovery.v1"
        assert summary["inputs"]["transport"] == "server"
        assert summary["inputs"]["server_url"] is None
        assert summary["key_results"]["poi_value"] == fit["result"]["poi_value"]
        assert summary["calls"] == calls
        assert call_names == expected_names
        assert all(call["response"]["schema_version"] == "nextstat.tool_result.v1" for call in calls)


def test_generated_provider_examples_can_load_live_server_descriptor(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    env = dict(os.environ)
    env["NEXTSTAT_TOOLS_SERVER_URL"] = server_url
    env["NEXTSTAT_TOOLS_API_KEY"] = api_key

    runnable_template_registry = _runnable_template_registry()
    assert {
        entry["renderer"] for entry in runnable_template_registry.values()
    } == set(LIVE_PROVIDER_PAYLOAD_RENDERER_ASSERTIONS)
    assert set(LIVE_PROVIDER_PAYLOAD_RENDERER_ASSERTIONS) == set(LIVE_PROVIDER_TOOL_NAME_EXTRACTORS)
    for profile in _runnable_example_profiles():
        client = str(profile["id"])
        runnable = profile["runnable_example"]
        assert isinstance(runnable, dict)
        template_family = str(runnable["template_family"])
        renderer = str(runnable_template_registry[template_family]["renderer"])
        payload_key = str(runnable["payload_key"])
        example_path = bootstrap_provider_example_path(client, "server")
        proc = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        payload = json.loads(proc.stdout)
        assert_live_provider_payload_renderer(
            renderer,
            payload=payload,
            client=client,
            runnable=runnable,
            payload_key=payload_key,
        )
        tool_names = tool_names_for_live_provider_payload(
            renderer,
            payload=payload,
            payload_key=payload_key,
        )
        assert "nextstat_fit" in tool_names
        assert "nextstat_read_root_histogram" in tool_names


def test_live_server_semantic_parity_matches_local_runtime(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])
    ws = str(live_server_runtime["workspace_json"])
    _assert_live_server_semantic_parity(
        workspace_json=ws,
        server_url=server_url,
        api_key=api_key,
    )

    from nextstat.tools import execute_tool

    name, tool_args = build_canonical_glm_tool_case()
    execution = {"deterministic": True, "eval_mode": "parity", "threads": 1}
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_bayesian_sample_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_root_histogram_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_meta_analysis_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_panel_fe_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_did_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_iv_2sls_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_aipw_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_event_study_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_ads_cuped_adjust_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_ads_cure_adjust_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_garch_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_kalman_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    forecast_args = {
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
    }
    local = execute_tool("nextstat_kalman", {**forecast_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        "nextstat_kalman",
        {**forecast_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path="tool:nextstat_kalman:forecast_alpha",
    )

    name, tool_args = build_canonical_kalman_simulate_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}:simulate",
    )

    name, tool_args = build_canonical_kalman_em_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}:em",
    )

    name, tool_args = build_canonical_churn_generate_data_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_risk_model_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_retention_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_pharma_fit_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_pharma_vpc_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_pk_gof_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_pk_npde_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_trial_simulate_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_compare_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_diagnostics_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_cohort_matrix_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_bootstrap_hr_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_ingest_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_uplift_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_churn_uplift_survival_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_chain_ladder_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_bioequivalence_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_dose_response_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_fault_tree_ce_is_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_competing_risks_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_fault_tree_mc_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_kaplan_meier_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_log_rank_test_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )

    name, tool_args = build_canonical_survival_tool_case()
    local = execute_tool(name, {**tool_args, "execution": dict(execution)}, transport="local")
    server = execute_tool(
        name,
        {**tool_args, "execution": dict(execution)},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=15.0,
        fallback_to_local=False,
    )
    assert local["ok"] is True, local
    assert server["ok"] is True, server
    assert_json_close(
        normalize_tool_envelope(server),
        normalize_tool_envelope(local),
        rtol=1e-6,
        atol=1e-8,
        path=f"tool:{name}",
    )


def test_live_server_semantic_parity_matches_local_runtime_for_simplified_likelihood(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    _assert_live_server_semantic_parity(
        workspace_json=_simplified_workspace_text(),
        server_url=server_url,
        api_key=api_key,
    )


def test_live_server_unknown_tool_returns_error_envelope(
    live_server_runtime: dict[str, object],
):
    from nextstat.tools import execute_tool

    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    out = execute_tool(
        "nextstat_not_a_real_tool",
        {},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
        fallback_to_local=False,
    )

    assert out["schema_version"] == "nextstat.tool_result.v1"
    assert out["ok"] is False
    assert out["result"] is None
    assert out["meta"]["tool_name"] == "nextstat_not_a_real_tool"
    assert out["error"]["type"] == "ToolError"
    assert "Unknown tool" in out["error"]["message"]


def test_live_server_fit_missing_workspace_returns_tool_error(
    live_server_runtime: dict[str, object],
):
    from nextstat.tools import execute_tool

    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    out = execute_tool(
        "nextstat_fit",
        {"execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
        fallback_to_local=False,
    )

    assert out["schema_version"] == "nextstat.tool_result.v1"
    assert out["ok"] is False
    assert out["result"] is None
    assert out["meta"]["tool_name"] == "nextstat_fit"
    assert out["meta"]["deterministic"] is True
    assert out["error"]["type"] == "ToolError"
    assert "either workspace_json or model_id must be provided" in out["error"]["message"]


def test_live_server_tools_schema_rate_limit_returns_http_429(
    rate_limited_live_server_runtime: dict[str, object],
):
    from nextstat.tools import ServerHTTPError, get_toolkit_descriptor

    api_key = str(rate_limited_live_server_runtime["api_key"])
    server_url = str(rate_limited_live_server_runtime["server_url"])

    descriptor = get_toolkit_descriptor(
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
    )
    assert descriptor["schema_version"] == "nextstat.tool_schema.v1"

    with pytest.raises(ServerHTTPError) as exc:
        get_toolkit_descriptor(
            transport="server",
            server_url=server_url,
            api_key=api_key,
            timeout_s=5.0,
        )
    assert exc.value.status_code == 429
    assert "rate limit exceeded" in exc.value.detail


def test_live_server_tools_execute_rate_limit_returns_server_http_error_envelope(
    rate_limited_live_server_runtime: dict[str, object],
):
    from nextstat.tools import execute_tool

    api_key = str(rate_limited_live_server_runtime["api_key"])
    server_url = str(rate_limited_live_server_runtime["server_url"])
    ws = _workspace_text()

    first = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
        fallback_to_local=False,
    )
    assert first["ok"] is True

    second = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=server_url,
        api_key=api_key,
        timeout_s=5.0,
        fallback_to_local=False,
    )
    assert second["schema_version"] == "nextstat.tool_result.v1"
    assert second["ok"] is False
    assert second["result"] is None
    assert second["meta"]["tool_name"] == "nextstat_workspace_audit"
    assert second["error"]["type"] == "ServerHTTPError"
    assert "HTTP 429" in second["error"]["message"]


def test_live_server_tools_execute_rejects_malformed_json_request(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    status, body = _post_raw_json(
        f"{server_url}/v1/tools/execute",
        b'{"name":',
        api_key=api_key,
    )
    assert status == 400
    assert isinstance(body, dict)
    assert "Failed to parse the request body as JSON" in body["error"]


def test_live_server_tools_execute_rejects_missing_name_field(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    status, body = _post_raw_json(
        f"{server_url}/v1/tools/execute",
        json.dumps({"arguments": {}}).encode("utf-8"),
        api_key=api_key,
    )
    assert status == 400
    assert isinstance(body, dict)
    assert "Failed to deserialize the JSON body into the target type" in body["error"]
    assert "missing field" in body["error"]
    assert "name" in body["error"]


def test_live_server_tools_execute_rejects_non_json_content_type_with_http_415(
    live_server_runtime: dict[str, object],
):
    api_key = str(live_server_runtime["api_key"])
    server_url = str(live_server_runtime["server_url"])

    status, body = _post_raw_json(
        f"{server_url}/v1/tools/execute",
        json.dumps({"name": "nextstat_not_a_real_tool", "arguments": {}}).encode("utf-8"),
        api_key=api_key,
        content_type="text/plain",
    )
    assert status == 415
    assert isinstance(body, dict)
    assert "Content-Type" in body["error"]
    assert "application/json" in body["error"]
