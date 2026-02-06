"""Phase 3.1: Frequentist limits parity vs pyhf (asymptotics, qtilde)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import nextstat
import nextstat.infer as ns_infer

pyhf = pytest.importorskip("pyhf")


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def pyhf_model_and_data(workspace: dict, measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def test_hypotest_cls_simple_parity(ns_timing):
    workspace = load_fixture("simple_workspace.json")
    with ns_timing.time("pyhf"):
        model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    for mu in [0.0, 0.5, 1.0, 2.0]:
        with ns_timing.time("pyhf"):
            pyhf_cls = float(
                pyhf.infer.hypotest(
                    mu,
                    data,
                    model,
                    test_stat="qtilde",
                    calctype="asymptotics",
                )
            )
        with ns_timing.time("nextstat"):
            ns_cls = float(ns_infer.hypotest(mu, ns_model))
        assert abs(ns_cls - pyhf_cls) < 5e-6


def test_profile_scan_qmu_simple_parity(ns_timing):
    workspace = load_fixture("simple_workspace.json")
    with ns_timing.time("pyhf"):
        model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    mu_values = [0.0, 0.5, 1.0, 2.0]
    with ns_timing.time("nextstat"):
        scan = ns_infer.profile_scan(ns_model, mu_values)
    points_by_mu = {float(p["mu"]): p for p in scan["points"]}

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    for mu in mu_values:
        with ns_timing.time("pyhf"):
            qmu_pyhf = float(
                pyhf.infer.test_statistics.qmu_tilde(
                    mu,
                    data,
                    model,
                    init_pars,
                    par_bounds,
                    fixed_params,
                )
            )
        qmu_ns = float(points_by_mu[mu]["q_mu"])
        assert abs(qmu_ns - qmu_pyhf) < 5e-6


@pytest.mark.slow
def test_upper_limits_linear_scan_simple_parity(ns_timing):
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow upper-limit scan parity tests.")

    workspace = load_fixture("simple_workspace.json")
    with ns_timing.time("pyhf"):
        model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    # Scan-based UL is expensive (many hypotest evaluations). Keep it opt-in and configurable.
    n_scan = int(os.environ.get("NS_UL_SCAN_POINTS", "21"))
    scan_stop = float(os.environ.get("NS_UL_SCAN_STOP", "5.0"))
    scan = np.linspace(0.0, scan_stop, n_scan)
    with ns_timing.time("pyhf"):
        pyhf_obs, pyhf_exp = pyhf.infer.intervals.upper_limits.upper_limit(
            data,
            model,
            scan=scan,
            level=0.05,
            test_stat="qtilde",
            calctype="asymptotics",
        )

    with ns_timing.time("nextstat"):
        ns_obs, ns_exp = ns_infer.upper_limits(ns_model, scan.tolist(), alpha=0.05)

    assert abs(float(ns_obs) - float(pyhf_obs)) < 5e-4
    assert len(ns_exp) == 5
    for a, b in zip(ns_exp, pyhf_exp):
        # Upper limits are obtained by interpolation; tiny CLs numerical differences can
        # amplify into small differences in mu at the crossing. Additionally, NextStat and pyhf
        # use different optimizers/minimizers, so Asimov fits can differ at the 1e-6 level.
        assert abs(float(a) - float(b)) < 5e-4


def test_upper_limits_rootfind_simple_parity(ns_timing):
    workspace = load_fixture("simple_workspace.json")
    with ns_timing.time("pyhf"):
        model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    with ns_timing.time("nextstat"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    with ns_timing.time("pyhf"):
        pyhf_obs, pyhf_exp = pyhf.infer.intervals.upper_limits.upper_limit(
            data,
            model,
            scan=None,
            level=0.05,
            rtol=1e-4,
            test_stat="qtilde",
            calctype="asymptotics",
        )

    with ns_timing.time("nextstat"):
        ns_obs, ns_exp = ns_infer.upper_limits_root(
            ns_model, alpha=0.05, lo=0.0, hi=5.0, rtol=1e-4, max_iter=80
        )

    assert abs(float(ns_obs) - float(pyhf_obs)) < 5e-4
    assert len(ns_exp) == 5
    for a, b in zip(ns_exp, pyhf_exp):
        assert abs(float(a) - float(b)) < 5e-4
