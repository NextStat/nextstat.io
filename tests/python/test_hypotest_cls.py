"""Phase 3.1: Frequentist limits parity vs pyhf (asymptotics, qtilde)."""

import json
from pathlib import Path

import numpy as np
import pyhf

import nextstat
import nextstat.infer as ns_infer


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


def test_hypotest_cls_simple_parity():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    for mu in [0.0, 0.5, 1.0, 2.0]:
        pyhf_cls = float(
            pyhf.infer.hypotest(
                mu,
                data,
                model,
                test_stat="qtilde",
                calctype="asymptotics",
            )
        )
        ns_cls = float(ns_infer.hypotest(mu, ns_model))
        assert abs(ns_cls - pyhf_cls) < 5e-6


def test_profile_scan_qmu_simple_parity():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    mu_values = [0.0, 0.5, 1.0, 2.0]
    scan = ns_infer.profile_scan(ns_model, mu_values)
    points_by_mu = {float(p["mu"]): p for p in scan["points"]}

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    for mu in mu_values:
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


def test_upper_limits_linear_scan_simple_parity():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    scan = np.linspace(0.0, 5.0, 201)
    pyhf_obs, pyhf_exp = pyhf.infer.intervals.upper_limits.upper_limit(
        data,
        model,
        scan=scan,
        level=0.05,
        test_stat="qtilde",
        calctype="asymptotics",
    )

    ns_obs, ns_exp = ns_infer.upper_limits(ns_model, scan.tolist(), alpha=0.05)

    assert abs(float(ns_obs) - float(pyhf_obs)) < 5e-6
    assert len(ns_exp) == 5
    for a, b in zip(ns_exp, pyhf_exp):
        # Upper limits are obtained by interpolation; tiny CLs numerical differences can
        # amplify into small differences in mu at the crossing. Additionally, NextStat and pyhf
        # use different optimizers/minimizers, so Asimov fits can differ at the 1e-6 level.
        assert abs(float(a) - float(b)) < 5e-5
