import json
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest_helper():
    module_path = (
        _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("nextstat._tool_manifest", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_result_schema_and_example_smoke():
    schema_path = _repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_result_v1.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"

    example_path = _repo_root() / "docs" / "specs" / "nextstat_tool_result_v1.example.json"
    assert example_path.exists(), f"missing example: {example_path}"
    example = json.loads(example_path.read_text())
    assert example.get("schema_version") == "nextstat.tool_result.v1"

    strict_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_result_strict_v1.schema.json"
    )
    assert strict_schema_path.exists(), f"missing schema: {strict_schema_path}"
    strict_schema = json.loads(strict_schema_path.read_text())

    server_strict_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_result_server_strict_v1.schema.json"
    )
    assert server_strict_schema_path.exists(), f"missing schema: {server_strict_schema_path}"
    server_strict_schema = json.loads(server_strict_schema_path.read_text())

    tool_schema_path = _repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_schema_v1.schema.json"
    assert tool_schema_path.exists(), f"missing schema: {tool_schema_path}"
    tool_schema = json.loads(tool_schema_path.read_text())
    assert tool_schema.get("$schema"), "tool schema must declare $schema"
    assert tool_schema.get("$id"), "tool schema must declare $id"
    assert tool_schema.get("type") == "object"
    assert "guidance" in tool_schema.get("properties", {}), "tool schema must declare guidance"

    runner_report_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_runner_report_v1.schema.json"
    )
    assert runner_report_schema_path.exists(), f"missing schema: {runner_report_schema_path}"
    runner_report_schema = json.loads(runner_report_schema_path.read_text())
    assert runner_report_schema.get("$schema"), "runner report schema must declare $schema"
    assert runner_report_schema.get("$id"), "runner report schema must declare $id"
    assert runner_report_schema.get("type") == "object"

    dashboard_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_dashboard_v1.schema.json"
    )
    assert dashboard_schema_path.exists(), f"missing schema: {dashboard_schema_path}"
    dashboard_schema = json.loads(dashboard_schema_path.read_text())
    assert dashboard_schema.get("$schema"), "dashboard schema must declare $schema"
    assert dashboard_schema.get("$id"), "dashboard schema must declare $id"
    assert dashboard_schema.get("type") == "object"

    artifact_manifest_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_artifact_manifest_v1.schema.json"
    )
    assert artifact_manifest_schema_path.exists(), f"missing schema: {artifact_manifest_schema_path}"
    artifact_manifest_schema = json.loads(artifact_manifest_schema_path.read_text())
    assert artifact_manifest_schema.get("$schema"), "artifact manifest schema must declare $schema"
    assert artifact_manifest_schema.get("$id"), "artifact manifest schema must declare $id"
    assert artifact_manifest_schema.get("type") == "object"

    performance_budget_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_performance_budget_v1.schema.json"
    )
    assert performance_budget_schema_path.exists(), f"missing schema: {performance_budget_schema_path}"
    performance_budget_schema = json.loads(performance_budget_schema_path.read_text())
    assert performance_budget_schema.get("$schema"), "performance budget schema must declare $schema"
    assert performance_budget_schema.get("$id"), "performance budget schema must declare $id"
    assert performance_budget_schema.get("type") == "object"

    agent_bootstrap_profile_manifest_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_profile_manifest_v1.schema.json"
    )
    assert (
        agent_bootstrap_profile_manifest_schema_path.exists()
    ), f"missing schema: {agent_bootstrap_profile_manifest_schema_path}"
    agent_bootstrap_profile_manifest_schema = json.loads(
        agent_bootstrap_profile_manifest_schema_path.read_text()
    )
    assert (
        agent_bootstrap_profile_manifest_schema.get("$schema")
    ), "agent bootstrap profile manifest schema must declare $schema"
    assert (
        agent_bootstrap_profile_manifest_schema.get("$id")
    ), "agent bootstrap profile manifest schema must declare $id"
    assert agent_bootstrap_profile_manifest_schema.get("type") == "object"

    agent_bootstrap_pack_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_pack_v1.schema.json"
    )
    assert agent_bootstrap_pack_schema_path.exists(), f"missing schema: {agent_bootstrap_pack_schema_path}"
    agent_bootstrap_pack_schema = json.loads(agent_bootstrap_pack_schema_path.read_text())
    assert agent_bootstrap_pack_schema.get("$schema"), "agent bootstrap pack schema must declare $schema"
    assert agent_bootstrap_pack_schema.get("$id"), "agent bootstrap pack schema must declare $id"
    assert agent_bootstrap_pack_schema.get("type") == "object"

    agent_bootstrap_doc_path = _repo_root() / "docs" / "references" / "agent-bootstrap.md"
    assert agent_bootstrap_doc_path.exists(), f"missing doc: {agent_bootstrap_doc_path}"

    local_tool_example_path = (
        _repo_root() / "docs" / "specs" / "nextstat_tool_schema_local_v1.example.json"
    )
    assert local_tool_example_path.exists(), f"missing example: {local_tool_example_path}"
    local_tool_example = json.loads(local_tool_example_path.read_text())

    server_tool_example_path = (
        _repo_root() / "docs" / "specs" / "nextstat_tool_schema_server_v1.example.json"
    )
    assert server_tool_example_path.exists(), f"missing example: {server_tool_example_path}"
    server_tool_example = json.loads(server_tool_example_path.read_text())

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=example, schema=schema)
        jsonschema.validate(instance=example, schema=strict_schema)
        jsonschema.validate(instance=example, schema=server_strict_schema)
        jsonschema.validate(instance=local_tool_example, schema=tool_schema)
        jsonschema.validate(instance=server_tool_example, schema=tool_schema)

    check = subprocess.run(
        [sys.executable, "scripts/generate_tool_contract_schemas.py", "--check"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout

    descriptor_check = subprocess.run(
        [sys.executable, "scripts/validate_tool_schema_descriptor.py", "--transport", "local"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert descriptor_check.returncode == 0, descriptor_check.stderr or descriptor_check.stdout

    example_check = subprocess.run(
        [sys.executable, "scripts/generate_tool_schema_examples.py", "--check"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert example_check.returncode == 0, example_check.stderr or example_check.stdout

    bootstrap_check = subprocess.run(
        [sys.executable, "-m", "scripts.generate_agent_bootstrap_packs", "--check"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert bootstrap_check.returncode == 0, bootstrap_check.stderr or bootstrap_check.stdout


def test_tool_manifest_smoke():
    manifest_path = (
        _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"
    )
    assert manifest_path.exists(), f"missing tool manifest: {manifest_path}"
    manifest_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_manifest_v1.schema.json"
    )
    assert manifest_schema_path.exists(), f"missing tool manifest schema: {manifest_schema_path}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("schema_version") == "nextstat.tool_manifest.v1"
    policies = manifest.get("policies")
    assert isinstance(policies, dict)
    assert isinstance(policies.get("server"), dict)
    guidance = manifest.get("guidance")
    assert isinstance(guidance, dict)
    assert isinstance(guidance.get("transport_hints"), dict)
    assert isinstance(guidance.get("recipes"), list) and guidance["recipes"]

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        manifest_schema = json.loads(manifest_schema_path.read_text())
        jsonschema.validate(instance=manifest, schema=manifest_schema)

    check = subprocess.run(
        [sys.executable, "scripts/validate_tool_manifest.py"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout

    tools = manifest.get("tools")
    assert isinstance(tools, list) and tools, "tool manifest must contain a non-empty tools list"

    names = [record.get("name") for record in tools]
    assert all(isinstance(name, str) and name for name in names), "every manifest record needs a name"
    assert len(names) == len(set(names)), "tool manifest names must be unique"

    local_names = []
    server_names = []
    golden_names = []
    for record in tools:
        if isinstance(record.get("local"), dict) and isinstance(record["local"].get("tool"), dict):
            local_names.append(record["name"])
        if isinstance(record.get("server"), dict) and isinstance(record["server"].get("tool"), dict):
            server_names.append(record["name"])
        golden_cases = record.get("golden_cases")
        if isinstance(golden_cases, dict) and golden_cases:
            golden_names.append(record["name"])
            for case_name, case in golden_cases.items():
                assert isinstance(case_name, str) and case_name
                assert isinstance(case, dict) and isinstance(case.get("arguments"), dict)

    assert set(server_names).issubset(set(local_names))
    assert golden_names, "manifest should define at least one golden case"

    strict_pairs = {
        record["name"]: record["strict_result_ref"]
        for record in tools
        if isinstance(record.get("strict_result_ref"), str)
    }
    assert strict_pairs, "manifest should declare strict_result_ref for core strict tool payloads"

    strict_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_result_strict_v1.schema.json"
    )
    strict_schema = json.loads(strict_schema_path.read_text())
    strict_enum = strict_schema["properties"]["meta"]["properties"]["tool_name"]["enum"]
    assert strict_enum == list(strict_pairs.keys())

    server_strict_schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_result_server_strict_v1.schema.json"
    )
    server_strict_schema = json.loads(server_strict_schema_path.read_text())
    server_strict_enum = server_strict_schema["properties"]["meta"]["properties"]["tool_name"]["enum"]
    server_expected = [
        record["name"]
        for record in tools
        if isinstance(record.get("strict_result_ref"), str)
        and isinstance(record.get("server"), dict)
        and isinstance(record["server"].get("tool"), dict)
    ]
    assert server_strict_enum == server_expected

    manifest_helper = _load_manifest_helper()
    nextstat_fit_policy = manifest_helper.get_server_policy("nextstat_fit")
    assert nextstat_fit_policy["availability"] == "exposed"
    assert nextstat_fit_policy["reason_code"] == "server_safe_subset"

    root_hist_policy = manifest_helper.get_server_policy("nextstat_read_root_histogram")
    assert root_hist_policy["availability"] == "exposed"
    assert root_hist_policy["reason_code"] == "server_safe_subset"

    local_guidance = manifest_helper.build_tool_guidance("local")
    server_guidance = manifest_helper.build_tool_guidance("server")
    assert isinstance(local_guidance["hints"], list) and local_guidance["hints"]
    assert isinstance(local_guidance["recipes"], list) and local_guidance["recipes"]
    assert isinstance(server_guidance["hints"], list) and server_guidance["hints"]
    assert isinstance(server_guidance["recipes"], list) and server_guidance["recipes"]


def test_tool_docs_match_manifest_names():
    check = subprocess.run(
        [sys.executable, "scripts/generate_tool_reference_docs.py", "--check"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout

    tool_api = (_repo_root() / "docs" / "references" / "tool-api.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED TOOL CAPABILITY MATRIX -->" in tool_api
    assert "<!-- END GENERATED TOOL CAPABILITY MATRIX -->" in tool_api
    assert "<!-- BEGIN GENERATED SERVER TOOL SUBSET -->" in tool_api
    assert "<!-- END GENERATED SERVER TOOL SUBSET -->" in tool_api
    assert "<!-- BEGIN GENERATED LOCAL GUIDANCE RECIPES -->" in tool_api
    assert "<!-- END GENERATED LOCAL GUIDANCE RECIPES -->" in tool_api
    assert "<!-- BEGIN GENERATED SERVER GUIDANCE RECIPES -->" in tool_api
    assert "<!-- END GENERATED SERVER GUIDANCE RECIPES -->" in tool_api

    server_api = (_repo_root() / "docs" / "references" / "server-api.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED SERVER TOOL SUBSET -->" in server_api
    assert "<!-- END GENERATED SERVER TOOL SUBSET -->" in server_api
    assert "<!-- BEGIN GENERATED SERVER GUIDANCE RECIPES -->" in server_api
    assert "<!-- END GENERATED SERVER GUIDANCE RECIPES -->" in server_api


def test_documented_top_level_python_api_functions_have_tool_surface():
    python_api = (_repo_root() / "docs" / "references" / "python-api.md").read_text(encoding="utf-8")
    documented_funcs = {
        match.group(1)
        for match in re.finditer(r"^### `nextstat\.([A-Za-z0-9_]+)\(", python_api, flags=re.MULTILINE)
    }
    assert documented_funcs, "expected documented top-level nextstat.* functions in python-api.md"

    manifest_path = (
        _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tool_names = {entry["name"] for entry in manifest["tools"]}

    missing = sorted(f"nextstat_{func_name}" for func_name in documented_funcs if f"nextstat_{func_name}" not in tool_names)
    assert not missing, (
        "Every documented top-level nextstat.* function must have a canonical tool surface entry. "
        f"Missing: {missing}"
    )
