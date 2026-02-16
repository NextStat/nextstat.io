import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_bca_ci_gates_script_pass(tmp_path: Path) -> None:
    hep_summary = {
        "percentile": {"median_wall_s": 10.0},
        "bca": {
            "runs": 10,
            "median_wall_s": 11.0,
            "fallback_count": 0,
            "effective_bca_count": 10,
        },
    }
    churn_summary = {
        "percentile": {"median_wall_s": 20.0},
        "bca": {
            "median_wall_s": 30.0,
            "fallback_total": 1,
            "effective_bca_total": 39,
        },
    }

    hep_path = tmp_path / "hep_summary.json"
    churn_path = tmp_path / "churn_summary.json"
    out_json = tmp_path / "gate_report.json"
    out_md = tmp_path / "gate_report.md"
    _write_json(hep_path, hep_summary)
    _write_json(churn_path, churn_summary)

    cmd = [
        "python3",
        str(_repo_root() / "scripts" / "benchmarks" / "check_bca_ci_gates.py"),
        "--hep-summary",
        str(hep_path),
        "--churn-summary",
        str(churn_path),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.check_call(cmd)

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall_pass"] is True
    assert report["failures"] == []
    assert out_md.exists()


def test_bca_ci_gates_script_fail(tmp_path: Path) -> None:
    hep_summary = {
        "percentile": {"median_wall_s": 10.0},
        "bca": {
            "runs": 10,
            "median_wall_s": 15.0,
            "fallback_count": 0,
            "effective_bca_count": 7,
        },
    }
    churn_summary = {
        "percentile": {"median_wall_s": 20.0},
        "bca": {
            "median_wall_s": 40.0,
            "fallback_total": 2,
            "effective_bca_total": 38,
        },
    }

    hep_path = tmp_path / "hep_summary.json"
    churn_path = tmp_path / "churn_summary.json"
    out_json = tmp_path / "gate_report.json"
    _write_json(hep_path, hep_summary)
    _write_json(churn_path, churn_summary)

    cmd = [
        "python3",
        str(_repo_root() / "scripts" / "benchmarks" / "check_bca_ci_gates.py"),
        "--hep-summary",
        str(hep_path),
        "--churn-summary",
        str(churn_path),
        "--out-json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert "BCa CI gates: FAIL" in proc.stdout

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["failures"]
