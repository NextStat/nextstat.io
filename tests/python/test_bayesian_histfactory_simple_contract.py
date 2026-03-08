import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import nextstat


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "run.py"
DATASET_PATH = (
    REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "datasets" / "simple_workspace.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("nextstat_bayesian_run_histfactory", RUN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manual_histfactory_simple_nll(*, mu: float, nuisances: list[float], signal: list[float], background: list[float], observed: list[int], tau: list[float]) -> float:
    nll = 0.0
    for n, s, b, theta, tau_i in zip(observed, signal, background, nuisances, tau):
        expected = mu * s + b * theta
        nll += expected - n * math.log(expected) + math.lgamma(n + 1.0)
        constraint_mean = tau_i * theta
        nll += constraint_mean - tau_i * math.log(constraint_mean) + math.lgamma(tau_i + 1.0)
    return nll


def test_extract_histfactory_simple_workspace_components_matches_expected_fixture() -> None:
    module = _load_module()
    workspace = json.loads(DATASET_PATH.read_text())

    extracted = module._extract_histfactory_simple_workspace_components(workspace)
    assert extracted["mu_name"] == "mu"
    assert extracted["mu_bounds"] == [0.0, 10.0]
    assert extracted["shapesys_name"] == "uncorr_bkguncrt"
    assert extracted["signal"] == [5.0, 10.0]
    assert extracted["background"] == [50.0, 60.0]
    assert extracted["observed"] == [53, 65]
    assert extracted["sigma_abs"] == [5.0, 6.0]
    assert extracted["tau"] == [100.0, 100.0]


def test_histfactory_simple_manual_nll_matches_nextstat_model() -> None:
    workspace = json.loads(DATASET_PATH.read_text())
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    params = [0.5, 1.0, 1.0]
    manual = _manual_histfactory_simple_nll(
        mu=params[0],
        nuisances=params[1:],
        signal=[5.0, 10.0],
        background=[50.0, 60.0],
        observed=[53, 65],
        tau=[100.0, 100.0],
    )
    observed = model.nll(params)
    assert abs(observed - manual) < 1e-9


def test_histfactory_simple_rejects_non_identifier_shapesys_name() -> None:
    module = _load_module()
    workspace = json.loads(DATASET_PATH.read_text())
    workspace["channels"][0]["samples"][1]["modifiers"][0]["name"] = "uncorr-bkguncrt"

    try:
        module._extract_histfactory_simple_workspace_components(workspace)
    except ValueError as exc:
        assert "valid identifier" in str(exc)
    else:
        raise AssertionError("expected invalid shapesys identifier to be rejected")


def test_numpyro_histfactory_simple_emits_not_supported(tmp_path: Path) -> None:
    out_path = tmp_path / "histfactory_numpyro.json"
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--case",
        "histfactory_simple_8p",
        "--model",
        "histfactory_simple",
        "--backend",
        "numpyro",
        "--out",
        str(out_path),
        "--n-chains",
        "2",
        "--warmup",
        "10",
        "--samples",
        "20",
        "--seed",
        "42",
        "--dataset-seed",
        "12345",
    ]
    completed = subprocess.run(cmd, check=False)
    assert completed.returncode == 0

    artifact = json.loads(out_path.read_text())
    assert artifact["status"] == "warn"
    assert artifact["reason"] == "backend_not_supported_for_model:numpyro:histfactory_simple"
