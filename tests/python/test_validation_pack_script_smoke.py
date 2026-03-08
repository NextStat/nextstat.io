import os
import subprocess
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_validation_pack_script_bash_syntax_ok() -> None:
    # Keep this lightweight and platform-agnostic: just ensure the script parses.
    subprocess.run(["bash", "-n", "validation-pack/render_validation_pack.sh"], check=True)


def test_validation_pack_script_help_mentions_json_only() -> None:
    out = subprocess.run(
        ["bash", "validation-pack/render_validation_pack.sh", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    assert "--json-only" in out
    assert "--m15-config" in out
    assert "--bayesian-design-report" in out
    # Signing flags are part of the validation-pack contract (manifest distribution workflow).
    assert "--sign-gpg" in out
    assert "--sign-openssl-key" in out


def test_validation_pack_release_paths_do_not_shadow_installed_nextstat() -> None:
    script = (_repo_root() / "validation-pack" / "render_validation_pack.sh").read_text(
        encoding="utf-8"
    )

    assert 'PYTHONPATH="$repo_root/bindings/ns-py/python${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$bayesian_design_report_out"' not in script
    assert 'PYTHONPATH="$repo_root/bindings/ns-py/python${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$bayesian_design_appendix_json"' not in script
    assert 'PYTHONPATH="$repo_root/bindings/ns-py/python${PYTHONPATH:+:$PYTHONPATH}" "$py" -m nextstat.m15_report render' not in script
    assert "Do not prepend `bindings/ns-py/python` here" in script


def test_validation_report_reference_mentions_bayesian_design_optional_artifacts() -> None:
    reference = (_repo_root() / "docs" / "references" / "validation-report.md").read_text(
        encoding="utf-8"
    )

    assert "--bayesian-design-report" in reference
    assert "--m15-config" in reference
    assert "m15_profile_diff_report.json" in reference
    assert "m15_report.md" in reference
    assert "m15_report.pdf" in reference
    assert "m15_report.docx" in reference
    assert "python -m nextstat.m15_report render" in reference
    assert "PYTHONPATH=bindings/ns-py/python \\\n  python -m nextstat.m15_report render" not in reference
    assert "bayesian_design_report.json" in reference
    assert "bayesian_design_regulatory_appendix.json" in reference
    assert "bayesian_design_regulatory_appendix.md" in reference
    assert "bayesian_design_regulatory_appendix.pdf" in reference
    assert "docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md" in reference
    assert "docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md" in reference
    assert "canonical pharma release evidence" in reference
    assert "--skip-pharma-validation" in reference
    assert "Cross-platform SAEM validation should use scientific acceptance criteria" in reference
    assert ".internal/" not in reference


def test_pq_ref_011_uses_shared_saem_acceptance_helper() -> None:
    pq_source = (_repo_root() / "tests" / "pharma_validation" / "pq.py").read_text(
        encoding="utf-8"
    )

    assert "from tests._tool_contract_helpers import assert_pharma_saem_acceptance_envelope" in pq_source
    assert "assert_pharma_saem_acceptance_envelope(saem)" in pq_source
    assert (
        "converged + finite OFV + theta bounded + omega positive/finite + omega_matrix diagonal positive/finite"
        in pq_source
    )


def test_pharma_validation_runner_bootstraps_repo_root_for_shared_test_helpers() -> None:
    runner_source = (_repo_root() / "tests" / "pharma_validation" / "runner.py").read_text(
        encoding="utf-8"
    )

    assert "def _ensure_repo_root_on_syspath()" in runner_source
    assert 'Path(__file__).resolve().parents[1]' in runner_source
    assert 'Path(__file__).resolve().parents[2]' in runner_source
    assert "sys.path.insert(0, tests_root_str)" in runner_source
    assert "sys.path.insert(0, repo_root_str)" in runner_source


def test_pharma_validation_runner_emits_strict_json_for_release_validation(tmp_path: Path) -> None:
    out = tmp_path / "pharma_validation.json"

    subprocess.run(
        [
            str(_repo_root() / ".venv" / "bin" / "python"),
            "tests/pharma_validation/runner.py",
            "--deterministic",
            "--out",
            str(out),
        ],
        check=True,
        cwd=_repo_root(),
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    json.dumps(payload, allow_nan=False)
