from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_artifact_manifest_module():
    module_path = _repo_root() / "scripts" / "tool_contract_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("tool_contract_artifact_manifest", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tool_contract_dashboard_job_runs_on_upstream_failures():
    workflow = (_repo_root() / ".github" / "workflows" / "python-tests.yml").read_text(encoding="utf-8")

    assert "tool-contract-dashboard:" in workflow
    assert (
        "if: ${{ always() && (needs.tool-contracts.result == 'success' || needs.tool-contracts.result == 'failure')"
        in workflow
    )
    assert (
        "(needs.live-server-contracts.result == 'success' || needs.live-server-contracts.result == 'failure' || needs.live-server-contracts.result == 'skipped') }}"
        in workflow
    )
    assert (
        "if: needs.live-server-contracts.result == 'success' || needs.live-server-contracts.result == 'failure'"
        in workflow
    )
    assert 'cat "${{ steps.tool-contract-artifacts.outputs.dashboard_markdown_path }}" >> "$GITHUB_STEP_SUMMARY"' in workflow
    assert "name: ${{ steps.tool-contract-artifacts.outputs.dashboard_artifact_name }}" in workflow
    assert workflow.count("id: tool-contract-artifacts") == 3


def test_tool_contract_artifact_names_and_paths_stay_in_sync():
    module = _load_artifact_manifest_module()
    outputs = module.to_github_outputs(module.load_tool_contract_artifact_manifest())
    workflow = (_repo_root() / ".github" / "workflows" / "python-tests.yml").read_text(encoding="utf-8")

    assert (
        'python scripts/tool_contract_artifact_manifest.py --format github-output >> "$GITHUB_OUTPUT"'
        in workflow
    )
    assert "--report-json \"${{ steps.tool-contract-artifacts.outputs.fast_report_path }}\"" in workflow
    assert "--report-json \"${{ steps.tool-contract-artifacts.outputs.live_report_path }}\"" in workflow

    assert "name: ${{ steps.tool-contract-artifacts.outputs.fast_artifact_name }}" in workflow
    assert "path: ${{ steps.tool-contract-artifacts.outputs.fast_report_path }}" in workflow
    assert "name: ${{ steps.tool-contract-artifacts.outputs.live_artifact_name }}" in workflow
    assert "path: ${{ steps.tool-contract-artifacts.outputs.live_report_path }}" in workflow

    assert "path: ${{ steps.tool-contract-artifacts.outputs.fast_download_dir }}" in workflow
    assert "path: ${{ steps.tool-contract-artifacts.outputs.live_download_dir }}" in workflow
    assert "${{ steps.tool-contract-artifacts.outputs.fast_downloaded_report_path }}" in workflow
    assert "${{ steps.tool-contract-artifacts.outputs.live_downloaded_report_path }}" in workflow

    assert "name: ${{ steps.tool-contract-artifacts.outputs.dashboard_artifact_name }}" in workflow
    assert "${{ steps.tool-contract-artifacts.outputs.dashboard_json_path }}" in workflow
    assert "${{ steps.tool-contract-artifacts.outputs.dashboard_markdown_path }}" in workflow

    hard_coded_path_values = (
        outputs["fast_report_path"],
        outputs["live_report_path"],
        outputs["fast_download_dir"],
        outputs["live_download_dir"],
        outputs["fast_downloaded_report_path"],
        outputs["live_downloaded_report_path"],
        outputs["dashboard_json_path"],
        outputs["dashboard_markdown_path"],
    )
    for value in hard_coded_path_values:
        assert value not in workflow, f"workflow should consume manifest outputs, not hard-code {value!r}"


def test_tool_contract_workflow_job_metadata_and_dependencies_stay_in_sync():
    workflow = (_repo_root() / ".github" / "workflows" / "python-tests.yml").read_text(encoding="utf-8")

    assert "tool-contracts:" in workflow
    assert "name: Tool Contracts" in workflow
    assert "live-server-contracts:" in workflow
    assert "name: Live Server Contracts (Python + Scripts + Semantic Parity)" in workflow
    assert "tool-contract-dashboard:" in workflow
    assert "name: Tool Contract Dashboard" in workflow
    assert "test:" in workflow

    assert "if: github.event_name == 'push'" in workflow
    assert "needs: [tool-contracts]" in workflow
    assert "needs: [tool-contracts, live-server-contracts]" in workflow

    assert (
        "python scripts/check_tool_contracts.py --mode fast --report-json "
        "\"${{ steps.tool-contract-artifacts.outputs.fast_report_path }}\""
        in workflow
    )
    assert (
        "python scripts/check_tool_contracts.py --mode live --report-json "
        "\"${{ steps.tool-contract-artifacts.outputs.live_report_path }}\""
        in workflow
    )

    assert workflow.count("name: Upload tool-contract fast report") == 1
    assert workflow.count("name: Upload tool-contract live report") == 1
    assert workflow.count("name: Upload tool-contract dashboard") == 1
    assert workflow.count("if: always()") >= 3
