"""Deterministic parity tests vs pyhf (Phase 1 contract)."""

import json
from pathlib import Path

import numpy as np
import pytest

import nextstat

from _tolerances import (
    TWICE_NLL_ATOL,
    TWICE_NLL_RTOL,
    GRADIENT_ATOL,
    GRADIENT_RTOL,
    EXPECTED_DATA_PER_BIN_ATOL,
)


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


def pyhf_twice_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item())


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise AssertionError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    return out


def test_simple_nll_parity_nominal_and_poi():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    pyhf_params = model.config.suggested_init()
    pyhf_val = pyhf_twice_nll(model, data, pyhf_params)

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )
    ns_val = 2.0 * float(ns_model.nll(ns_params))

    assert ns_val == pytest.approx(pyhf_val, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)

    # POI variations
    poi_idx = model.config.poi_index
    for poi in [0.0, 2.0]:
        pyhf_params_var = list(pyhf_params)
        pyhf_params_var[poi_idx] = poi
        pyhf_val_var = pyhf_twice_nll(model, data, pyhf_params_var)

        ns_params_var = map_params_by_name(
            model.config.par_names,
            pyhf_params_var,
            ns_model.parameter_names(),
            ns_model.suggested_init(),
        )
        ns_val_var = 2.0 * float(ns_model.nll(ns_params_var))
        assert ns_val_var == pytest.approx(pyhf_val_var, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)


def test_complex_nll_parity_nominal_and_poi():
    workspace = load_fixture("complex_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="measurement")

    pyhf_params = model.config.suggested_init()
    pyhf_val = pyhf_twice_nll(model, data, pyhf_params)

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_names = ns_model.parameter_names()
    assert set(ns_names) == set(model.config.par_names)

    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_params,
        ns_names,
        ns_model.suggested_init(),
    )
    ns_val = 2.0 * float(ns_model.nll(ns_params))
    assert ns_val == pytest.approx(pyhf_val, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)

    poi_idx = model.config.poi_index
    for poi in [0.0, 2.0]:
        pyhf_params_var = list(pyhf_params)
        pyhf_params_var[poi_idx] = poi
        pyhf_val_var = pyhf_twice_nll(model, data, pyhf_params_var)

        ns_params_var = map_params_by_name(
            model.config.par_names,
            pyhf_params_var,
            ns_names,
            ns_model.suggested_init(),
        )
        ns_val_var = 2.0 * float(ns_model.nll(ns_params_var))
        assert ns_val_var == pytest.approx(pyhf_val_var, rel=TWICE_NLL_RTOL, abs=TWICE_NLL_ATOL)


def test_simple_mle_parity_bestfit_uncertainties(ns_timing):
    workspace = load_fixture("simple_workspace.json")
    with ns_timing.time("pyhf:model+data"):
        model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    with ns_timing.time("pyhf:fit"):
        pyhf_bestfit = np.asarray(pyhf.infer.mle.fit(data, model), dtype=float)
        pyhf_bestfit_nll = pyhf_twice_nll(model, data, pyhf_bestfit) / 2.0

    # Numerical Hessian for uncertainties (NLL, not twice_nll)
    def nll_func(x: np.ndarray) -> float:
        return pyhf_twice_nll(model, data, x) / 2.0

    with ns_timing.time("pyhf:uncertainties_hessian"):
        n = len(pyhf_bestfit)
        h_step = 1e-4
        damping = 1e-9
        f0 = nll_func(pyhf_bestfit)
        hess = np.zeros((n, n), dtype=float)

        for i in range(n):
            hi = h_step * max(abs(pyhf_bestfit[i]), 1.0)
            xp = pyhf_bestfit.copy()
            xm = pyhf_bestfit.copy()
            xp[i] += hi
            xm[i] -= hi
            fp = nll_func(xp)
            fm = nll_func(xm)
            hess[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

            for j in range(i + 1, n):
                hj = h_step * max(abs(pyhf_bestfit[j]), 1.0)
                xpp = pyhf_bestfit.copy()
                xpm = pyhf_bestfit.copy()
                xmp = pyhf_bestfit.copy()
                xmm = pyhf_bestfit.copy()
                xpp[i] += hi
                xpp[j] += hj
                xpm[i] += hi
                xpm[j] -= hj
                xmp[i] -= hi
                xmp[j] += hj
                xmm[i] -= hi
                xmm[j] -= hj
                fij = (nll_func(xpp) - nll_func(xpm) - nll_func(xmp) + nll_func(xmm)) / (
                    4.0 * hi * hj
                )
                hess[i, j] = fij
                hess[j, i] = fij

        hess = hess + np.eye(n) * damping
        cov = np.linalg.inv(hess)
        pyhf_unc = np.sqrt(np.maximum(np.diag(cov), 0.0))

    with ns_timing.time("nextstat:model"):
        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        mle = nextstat.MaximumLikelihoodEstimator()
    with ns_timing.time("nextstat:fit"):
        ns_res = mle.fit(ns_model)

    ns_names = ns_model.parameter_names()
    ns_bestfit_by_name = dict(zip(ns_names, ns_res.parameters))
    pyhf_bestfit_by_name = dict(zip(model.config.par_names, pyhf_bestfit.tolist()))

    for name in model.config.par_names:
        assert abs(ns_bestfit_by_name[name] - pyhf_bestfit_by_name[name]) < 2e-4

    ns_nll = float(ns_res.nll)
    assert abs(ns_nll - pyhf_bestfit_nll) < 1e-6

    ns_unc_by_name = dict(zip(ns_names, ns_res.uncertainties))
    pyhf_unc_by_name = dict(zip(model.config.par_names, pyhf_unc.tolist()))
    for name in model.config.par_names:
        assert abs(ns_unc_by_name[name] - pyhf_unc_by_name[name]) < 5e-4


def _pyhf_nll_gradient_fd(model, data, params, h=1e-5):
    """Central finite-difference gradient of NLL (= twice_nll / 2)."""
    params = np.asarray(params, dtype=float)
    n = len(params)
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        hi = h * max(abs(params[i]), 1.0)
        xp = params.copy()
        xm = params.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = pyhf_twice_nll(model, data, xp.tolist()) / 2.0
        fm = pyhf_twice_nll(model, data, xm.tolist()) / 2.0
        grad[i] = (fp - fm) / (2.0 * hi)
    return grad


def _gradient_parity_at_point(pyhf_model, pyhf_data, ns_model, pyhf_params, label):
    """Compare pyhf FD gradient vs NextStat AD gradient at a parameter point."""
    ns_params = map_params_by_name(
        pyhf_model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )

    pyhf_grad = _pyhf_nll_gradient_fd(pyhf_model, pyhf_data, pyhf_params)
    ns_grad = np.array(ns_model.grad_nll(ns_params), dtype=float)

    # Reorder NS gradient to pyhf parameter ordering for comparison
    ns_names = ns_model.parameter_names()
    ns_name_to_idx = {name: i for i, name in enumerate(ns_names)}
    ns_grad_reordered = np.array(
        [ns_grad[ns_name_to_idx[name]] for name in pyhf_model.config.par_names],
        dtype=float,
    )

    for i, name in enumerate(pyhf_model.config.par_names):
        diff = abs(ns_grad_reordered[i] - pyhf_grad[i])
        scale = max(abs(pyhf_grad[i]), 1.0)
        assert diff < GRADIENT_ATOL + GRADIENT_RTOL * scale, (
            f"{label} param '{name}' [#{i}]: "
            f"NS_grad={ns_grad_reordered[i]:.8e}, pyhf_FD={pyhf_grad[i]:.8e}, "
            f"diff={diff:.2e}, tol={GRADIENT_ATOL + GRADIENT_RTOL * scale:.2e}"
        )


def test_gradient_parity_simple():
    """NextStat AD gradient vs pyhf central FD gradient — simple workspace."""
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")
    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    # Test at suggested_init
    init = model.config.suggested_init()
    _gradient_parity_at_point(model, data, ns_model, init, "simple@init")

    # Test at a perturbed point (shift each param by small amount)
    rng = np.random.default_rng(42)
    perturbed = [p + rng.normal(0, 0.1) for p in init]
    # Clamp to bounds
    bounds = model.config.suggested_bounds()
    perturbed = [
        np.clip(p, lo + 1e-6, hi - 1e-6) for p, (lo, hi) in zip(perturbed, bounds)
    ]
    _gradient_parity_at_point(model, data, ns_model, perturbed, "simple@perturbed")


def test_gradient_parity_complex():
    """NextStat AD gradient vs pyhf central FD gradient — complex workspace."""
    workspace = load_fixture("complex_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="measurement")
    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    init = model.config.suggested_init()
    _gradient_parity_at_point(model, data, ns_model, init, "complex@init")

    rng = np.random.default_rng(123)
    perturbed = [p + rng.normal(0, 0.05) for p in init]
    bounds = model.config.suggested_bounds()
    perturbed = [
        np.clip(p, lo + 1e-6, hi - 1e-6) for p, (lo, hi) in zip(perturbed, bounds)
    ]
    _gradient_parity_at_point(model, data, ns_model, perturbed, "complex@perturbed")


def test_expected_data_per_bin_parity_simple():
    """Per-bin expected data must match pyhf at near-machine precision."""
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")
    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))

    pyhf_params = model.config.suggested_init()
    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )

    pyhf_expected = np.array(model.expected_data(pyhf_params), dtype=float)
    ns_expected = np.array(ns_model.expected_data(ns_params), dtype=float)

    # pyhf expected_data includes main + aux; compare lengths
    n_main = sum(model.config.channel_nbins.values())
    # Compare main data bins
    pyhf_main = pyhf_expected[:n_main]
    ns_main = ns_expected[:n_main]

    for i in range(n_main):
        diff = abs(ns_main[i] - pyhf_main[i])
        assert diff < EXPECTED_DATA_PER_BIN_ATOL, (
            f"Bin {i}: NS={ns_main[i]:.15e}, pyhf={pyhf_main[i]:.15e}, diff={diff:.2e}"
        )
