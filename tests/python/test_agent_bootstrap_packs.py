from __future__ import annotations

import json
import subprocess
import sys

from tests._bootstrap_renderer_assertions import (
    RUNNABLE_PAYLOAD_RENDERER_ASSERTIONS,
    WORKSPACE_OUTPUT_RENDERER_ASSERTIONS,
    assert_runnable_payload_renderer,
    assert_workspace_output_renderer,
)
from tests._bootstrap_profile_manifest_helpers import (
    bootstrap_pack_path,
    bootstrap_provider_example_path,
    bootstrap_reference_doc_path,
    load_agent_bootstrap_profile_manifest_helper,
    load_repo_module,
    repo_root,
)


def test_agent_bootstrap_profile_manifest_and_packs_are_valid():
    manifest_path = repo_root() / "scripts" / "agent_bootstrap_profile_manifest_v1.json"
    manifest_schema_path = (
        repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_profile_manifest_v1.schema.json"
    )
    pack_schema_path = (
        repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_pack_v1.schema.json"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = json.loads(manifest_schema_path.read_text(encoding="utf-8"))
    pack_schema = json.loads(pack_schema_path.read_text(encoding="utf-8"))

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=manifest, schema=manifest_schema)

    helper = load_agent_bootstrap_profile_manifest_helper()
    generator = load_repo_module(
        "generate_agent_bootstrap_packs",
        "scripts/generate_agent_bootstrap_packs.py",
    )
    loaded = helper.load_agent_bootstrap_profile_manifest()
    runnable_profiles = helper.get_runnable_example_profiles(loaded)
    workspace_output_profiles = helper.get_workspace_output_profiles(loaded)
    workspace_output_registry = helper.get_workspace_output_template_registry(loaded)
    runnable_template_registry = helper.get_runnable_template_registry(loaded)
    expected_manifest_schema = helper.build_agent_bootstrap_profile_manifest_schema(loaded)
    coverage_policy = loaded["coverage_policy"]
    expected_profile_ids = set(coverage_policy["required_profile_ids"])
    expected_runnable_profile_ids = set(coverage_policy["required_runnable_profile_ids"])
    workspace_output_owner_profile_ids = set(coverage_policy["workspace_output_owner_profile_ids"])
    required_workspace_output_profile_ids = set(coverage_policy["required_workspace_output_profile_ids"])
    assert loaded["schema_version"] == "nextstat.agent_bootstrap_profile_manifest.v1"
    assert manifest_schema == expected_manifest_schema
    assert {profile["id"] for profile in loaded["profiles"]} == expected_profile_ids
    assert {profile["id"] for profile in runnable_profiles} == expected_runnable_profile_ids
    assert expected_runnable_profile_ids.issubset(expected_profile_ids)
    assert workspace_output_owner_profile_ids.issubset(expected_profile_ids)
    assert required_workspace_output_profile_ids.issubset(workspace_output_owner_profile_ids)
    assert set(pack_schema["properties"]["client"]["enum"]) == expected_profile_ids
    assert {
        entry["renderer"] for entry in workspace_output_registry.values()
    } == set(WORKSPACE_OUTPUT_RENDERER_ASSERTIONS)
    assert {
        entry["renderer"] for entry in runnable_template_registry.values()
    } == set(RUNNABLE_PAYLOAD_RENDERER_ASSERTIONS)
    assert set(generator._workspace_output_renderer_generators()) == set(WORKSPACE_OUTPUT_RENDERER_ASSERTIONS)
    assert set(generator._runnable_renderer_generators()) == set(RUNNABLE_PAYLOAD_RENDERER_ASSERTIONS)
    workspace_output_profile_ids = {profile["id"] for profile in workspace_output_profiles}
    assert workspace_output_profile_ids.issubset(workspace_output_owner_profile_ids)
    assert required_workspace_output_profile_ids.issubset(workspace_output_profile_ids)
    assert workspace_output_profile_ids

    doc = bootstrap_reference_doc_path().read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED IDE WORKSPACE FILES -->" in doc
    assert "<!-- END GENERATED IDE WORKSPACE FILES -->" in doc
    assert "<!-- BEGIN GENERATED PROVIDER EXAMPLES -->" in doc
    assert "<!-- END GENERATED PROVIDER EXAMPLES -->" in doc
    assert "<!-- BEGIN GENERATED AGENT PROFILE MATRIX -->" in doc
    assert "<!-- END GENERATED AGENT PROFILE MATRIX -->" in doc
    assert "<!-- BEGIN GENERATED AGENT PROFILE DETAILS -->" in doc
    assert "<!-- END GENERATED AGENT PROFILE DETAILS -->" in doc

    for client in sorted(expected_profile_ids):
        for transport in ("local", "server"):
            pack_path = bootstrap_pack_path(client, transport)
            assert pack_path.exists(), f"missing generated pack: {pack_path}"
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            if jsonschema is not None:
                jsonschema.validate(instance=pack, schema=pack_schema)
            assert pack["client"] == client
            assert pack["transport"] == transport
            assert pack["discovery_contract"]["descriptor_schema_version"] == "nextstat.tool_schema.v1"
            assert isinstance(pack["instructions"], list) and pack["instructions"]
            assert isinstance(pack["recipes"], list) and pack["recipes"]
            assert all(recipe["transport"] == transport for recipe in pack["recipes"])
            assert isinstance(pack["snippets"]["bootstrap"], str) and pack["snippets"]["bootstrap"]
            assert isinstance(pack["snippets"]["execution_loop"], str) and pack["snippets"]["execution_loop"]

    for profile in workspace_output_profiles:
        for workspace_output in profile["workspace_outputs"]:
            output_path = repo_root() / workspace_output["path"]
            content = output_path.read_text(encoding="utf-8")
            assert "Generated by scripts/generate_agent_bootstrap_packs.py" in content
            renderer = workspace_output_registry[workspace_output["template_family"]]["renderer"]
            assert_workspace_output_renderer(renderer, content=content)

    for profile in runnable_profiles:
        client = profile["id"]
        runnable = profile["runnable_example"]
        template_family = runnable["template_family"]
        renderer = runnable_template_registry[template_family]["renderer"]
        payload_key = runnable["payload_key"]
        for transport in ("local", "server"):
            example_path = bootstrap_provider_example_path(client, transport)
            assert example_path.exists(), f"missing generated provider example: {example_path}"
            script = example_path.read_text(encoding="utf-8")
            assert "Generated by scripts/generate_agent_bootstrap_packs.py" in script
            assert "get_toolkit_descriptor" in script
            assert "if __name__ == \"__main__\":" in script

            help_proc = subprocess.run(
                [sys.executable, str(example_path), "--help"],
                cwd=repo_root(),
                capture_output=True,
                text=True,
            )
            assert help_proc.returncode == 0, help_proc.stderr or help_proc.stdout
            assert "--print-pack" in help_proc.stdout

            if transport == "local":
                run_proc = subprocess.run(
                    [sys.executable, str(example_path)],
                    cwd=repo_root(),
                    capture_output=True,
                    text=True,
                )
                assert run_proc.returncode == 0, run_proc.stderr or run_proc.stdout
                payload = json.loads(run_proc.stdout)
                assert_runnable_payload_renderer(
                    renderer,
                    payload=payload,
                    client=client,
                    transport=transport,
                    runnable=runnable,
                    payload_key=payload_key,
                )
            else:
                run_proc = subprocess.run(
                    [sys.executable, str(example_path), "--print-pack"],
                    cwd=repo_root(),
                    capture_output=True,
                    text=True,
                )
                assert run_proc.returncode == 0, run_proc.stderr or run_proc.stdout
                payload = json.loads(run_proc.stdout)
                assert payload["client"] == client
                assert payload["transport"] == transport


def test_agent_bootstrap_generator_is_in_sync():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.generate_agent_bootstrap_packs", "--check"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    schema_proc = subprocess.run(
        [sys.executable, "scripts/generate_agent_bootstrap_profile_manifest_schema.py", "--check"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert schema_proc.returncode == 0, schema_proc.stderr or schema_proc.stdout


def test_agent_bootstrap_generator_uses_shared_artifact_path_builders():
    source = (repo_root() / "scripts" / "generate_agent_bootstrap_packs.py").read_text(
        encoding="utf-8"
    )
    helper_source = (repo_root() / "scripts" / "bootstrap_artifact_paths.py").read_text(
        encoding="utf-8"
    )
    repo_loader_source = (repo_root() / "scripts" / "repo_module_loader.py").read_text(
        encoding="utf-8"
    )
    test_helper_source = (
        repo_root() / "tests" / "_bootstrap_profile_manifest_helpers.py"
    ).read_text(encoding="utf-8")
    assert "from scripts import agent_bootstrap_profile_manifest as _profile_manifest" in source
    assert "from scripts.repo_module_loader import load_repo_module" in source
    assert "from scripts.bootstrap_artifact_paths import (" in source
    assert "python -m scripts.generate_agent_bootstrap_packs" in source
    assert "python scripts/generate_agent_bootstrap_packs.py" not in source
    assert "from scripts import agent_bootstrap_profile_manifest as _profile_manifest" in test_helper_source
    assert "from scripts.repo_module_loader import load_repo_module" in test_helper_source
    assert "from scripts.bootstrap_artifact_paths import (" in test_helper_source
    assert '"repo_module_loader.py"' not in source
    assert '"repo_module_loader.py"' not in test_helper_source
    assert "def load_repo_module(" in repo_loader_source
    assert "bootstrap_helper_loader" not in source
    assert "bootstrap_helper_loader" not in test_helper_source
    assert "load_bootstrap_helper_module(" not in source
    assert "bootstrap_artifact_paths" in source
    assert "agent_bootstrap_profile_manifest" in source
    assert "bootstrap_pack_output_path(" in test_helper_source
    assert "bootstrap_provider_example_output_path(" in test_helper_source
    assert "bootstrap_reference_doc_output_path(" in test_helper_source
    assert "return _profile_manifest" in source
    assert "return _profile_manifest" in test_helper_source
    assert "def _pack_relative_path(" not in source
    assert "def _provider_example_relative_path(" not in source
    assert "def _reference_doc_relative_path(" not in source
    assert "nextstat_{client}_{transport}_bootstrap_v1.json" not in source
    assert "nextstat_{client}_{transport}_example.py" not in source
    assert helper_source.count("nextstat_{client}_{transport}_bootstrap_v1.json") == 1
    assert helper_source.count("nextstat_{client}_{transport}_example.py") == 1
    assert "docs/specs/agent_bootstrap/nextstat_{profile['id']}_local_bootstrap_v1.json" not in source
    assert "docs/specs/agent_bootstrap/examples/nextstat_{client}_local_example.py" not in source


def test_agent_bootstrap_packs_cover_every_callable_tool_for_each_transport():
    tool_helper = load_repo_module(
        "nextstat._tool_manifest",
        "bindings/ns-py/python/nextstat/_tool_manifest.py",
    )
    tool_manifest = tool_helper.load_tool_manifest()
    tools = tool_manifest["tools"]
    callable_names_by_transport = {
        "local": {
            entry["name"]
            for entry in tools
            if isinstance(entry.get("local"), dict) and entry["local"].get("tool") is not None
        },
        "server": {
            entry["name"]
            for entry in tools
            if isinstance(entry.get("server"), dict) and entry["server"].get("tool") is not None
        },
    }

    profile_helper = load_agent_bootstrap_profile_manifest_helper()
    profile_manifest = profile_helper.load_agent_bootstrap_profile_manifest()

    for transport, callable_names in callable_names_by_transport.items():
        assert callable_names, f"expected non-empty {transport} callable tool set in tool manifest"
        guidance = tool_helper.build_tool_guidance(transport)
        recipe_tool_names = {
            tool_name for recipe in guidance["recipes"] for tool_name in recipe["tools"]
        }
        assert callable_names.issubset(recipe_tool_names), (
            f"Every {transport} callable tool must be covered by at least one {transport} guidance recipe. "
            f"Missing: {sorted(callable_names - recipe_tool_names)}"
        )

        for profile in profile_manifest["profiles"]:
            client = profile["id"]
            pack_path = bootstrap_pack_path(client, transport)
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            pack_recipe_tool_names = {
                tool_name for recipe in pack["recipes"] for tool_name in recipe["tools"]
            }
            assert callable_names.issubset(pack_recipe_tool_names), (
                f"{transport} bootstrap pack for {client} must cover every {transport} callable tool. "
                f"Missing: {sorted(callable_names - pack_recipe_tool_names)}"
            )
