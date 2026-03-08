import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tests._tool_contract_helpers import (
    assert_json_close,
    assert_pharma_saem_acceptance_envelope,
    normalize_tool_envelope,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _activate_repo_python_package() -> None:
    import nextstat

    repo_pkg = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat"
    pkg_path = str(repo_pkg)
    package_paths = getattr(nextstat, "__path__", None)
    if package_paths is None:
        raise RuntimeError("nextstat package has no __path__; cannot overlay repo Python modules")
    if pkg_path in package_paths:
        package_paths.remove(pkg_path)
    package_paths.insert(0, pkg_path)
    sys.modules.pop("nextstat.tools", None)
    sys.modules.pop("nextstat._tool_manifest", None)


def _assert_no_unstable_perf_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        assert "wall_time_s" not in value, f"{path} must not persist wall_time_s in golden files"
        assert "elapsed_s" not in value, f"{path} must not persist elapsed_s in golden files"
        assert "scenarios_per_sec" not in value, (
            f"{path} must not persist scenarios_per_sec in golden files"
        )
        for key, nested in value.items():
            _assert_no_unstable_perf_fields(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            _assert_no_unstable_perf_fields(nested, path=f"{path}[{idx}]")


def test_tool_goldens_manifest_cases():
    pytest.importorskip("nextstat")
    _activate_repo_python_package()
    from nextstat.tools import execute_tool
    from nextstat._tool_manifest import get_tool_records, resolve_golden_case_arguments

    golden_dir = _repo_root() / "tests" / "fixtures" / "tool_goldens"
    records = get_tool_records()
    case_names = sorted(
        {
            case_name
            for record in records
            for case_name in (record.get("golden_cases", {}) or {}).keys()
            if isinstance(case_name, str)
        }
    )
    assert case_names, "manifest must define at least one golden case"
    expected_files = {f"{case_name}.v1.json" for case_name in case_names}
    actual_files = {path.name for path in golden_dir.glob("*.v1.json")}
    assert actual_files == expected_files, "golden directory must exactly match manifest case names"

    # Keep strict defaults here; the helper owns narrowly scoped cross-platform
    # relaxations for known optimizer-sensitive fields.
    rtol = 1e-6
    atol = 1e-8

    for case_name in case_names:
        golden_path = golden_dir / f"{case_name}.v1.json"
        assert golden_path.exists(), f"missing golden file: {golden_path} (run scripts/generate_tool_goldens.py)"

        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        assert golden.get("schema_version") == "nextstat.tool_goldens.v1"
        assert golden.get("case_name") == case_name

        tools: dict[str, Any] = golden.get("tools", {})
        assert isinstance(tools, dict) and tools, "golden tools map must be non-empty"
        _assert_no_unstable_perf_fields(golden)

        manifest_names = []
        for record in records:
            name = record.get("name")
            if not isinstance(name, str):
                continue
            cases = record.get("golden_cases")
            if not isinstance(cases, dict) or case_name not in cases:
                continue
            manifest_names.append(name)
            golden_env = tools[name]
            args = resolve_golden_case_arguments(name, case_name, _repo_root())
            got = execute_tool(name, args)
            assert got.get("ok") is True, got

            assert_json_close(
                normalize_tool_envelope(got),
                normalize_tool_envelope(golden_env),
                rtol=rtol,
                atol=atol,
                path=f"case:{case_name}.tool:{name}",
            )

        assert set(manifest_names) == set(tools.keys())


def test_tool_goldens_generator_check_mode():
    pytest.importorskip("nextstat")
    check = subprocess.run(
        [sys.executable, "scripts/generate_tool_goldens.py", "--check"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_tool_contract_tolerance_scope_for_pharma_fit_eta():
    assert_json_close(
        0.012952322987712603,
        0.009768206806171575,
        rtol=1e-6,
        atol=1e-8,
        path="tool:nextstat_pharma_fit.result.eta[0][0]",
    )

    with pytest.raises(AssertionError):
        assert_json_close(
            0.012952322987712603,
            0.009768206806171575,
            rtol=1e-6,
            atol=1e-8,
            path="tool:nextstat_pharma_fit.result.theta[0]",
        )


def test_tool_contract_tolerance_scope_for_pharma_fit_omega():
    """omega/omega_matrix/sigma get cross-platform BLAS tolerance (rtol=5e-3)."""
    # omega SD 0.232 vs 0.233 → diff 0.001, rtol = 0.001/0.233 ≈ 0.004 < 5e-3 → OK
    assert_json_close(
        0.23205333134381162,
        0.23305333134381162,
        rtol=1e-6,
        atol=1e-8,
        path="tool:nextstat_pharma_fit.result.omega[0]",
    )
    # omega_matrix diagonal
    assert_json_close(
        0.05384874858776082,
        0.05394874858776082,
        rtol=1e-6,
        atol=1e-8,
        path="tool:nextstat_pharma_fit.result.omega_matrix[0][0]",
    )
    # sigma (not sigma_init)
    assert_json_close(
        0.4957107292952698,
        0.4967107292952698,
        rtol=1e-6,
        atol=1e-8,
        path="tool:nextstat_pharma_fit.result.sigma",
    )
    # sigma_init must stay tight (it's an input, not computed)
    with pytest.raises(AssertionError):
        assert_json_close(
            0.1,
            0.101,
            rtol=1e-6,
            atol=1e-8,
            path="tool:nextstat_pharma_fit.result.sigma_init",
        )


def test_saem_acceptance_envelope_passes_on_valid_result():
    """SAEM acceptance envelope validates scientific criteria, not exact values."""
    result = {
        "converged": True,
        "ofv": 123.456,
        "theta": [0.13, 8.0, 1.5],
        "omega": [0.23, 0.17, 0.20],
        "omega_matrix": [
            [0.053, 0.0, 0.0],
            [0.0, 0.029, 0.0],
            [0.0, 0.0, 0.040],
        ],
    }
    assert_pharma_saem_acceptance_envelope(result)


def test_saem_acceptance_envelope_rejects_non_converged():
    with pytest.raises(AssertionError, match="converge"):
        assert_pharma_saem_acceptance_envelope({"converged": False, "ofv": 1.0, "theta": [1.0], "omega": [0.1]})


def test_saem_acceptance_envelope_rejects_infinite_ofv():
    with pytest.raises(AssertionError, match="finite"):
        assert_pharma_saem_acceptance_envelope({"converged": True, "ofv": float("inf"), "theta": [1.0], "omega": [0.1]})


def test_saem_acceptance_envelope_rejects_negative_omega():
    with pytest.raises(AssertionError, match="positive"):
        assert_pharma_saem_acceptance_envelope({"converged": True, "ofv": 1.0, "theta": [1.0], "omega": [-0.1]})
