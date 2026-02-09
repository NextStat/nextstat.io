"""Validate that Fast mode NLL does not diverge from Parity mode beyond tolerance.

For each workspace at suggested_init and random parameter points, we verify that
the absolute difference between Fast and Parity NLL is within acceptable bounds.
"""

import json
import random
from pathlib import Path

import pytest

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# Fast vs Parity should be near-identical on small models.
# On large models (1000+ bins), summation order can differ at ~1e-13 per term.
NLL_FAST_VS_PARITY_ATOL = 1e-10


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def sample_params(rng, init, bounds):
    out = []
    for x0, (lo, hi) in zip(init, bounds):
        lo_f, hi_f = float(lo), float(hi)
        if not (lo_f < hi_f):
            out.append(float(x0))
            continue
        span = hi_f - lo_f
        center = min(max(float(x0), lo_f), hi_f)
        half = 0.25 * span
        a = max(lo_f, center - half)
        b = min(hi_f, center + half)
        if not (a < b):
            a, b = lo_f, hi_f
        out.append(rng.uniform(a, b))
    return out


@pytest.fixture(autouse=True)
def restore_fast_mode():
    yield
    nextstat.set_eval_mode("fast")


# All workspaces that are valid pyhf HistFactory JSON.
_WORKSPACES = []
for _name in [
    "simple_workspace.json",
    "complex_workspace.json",
    "workspace_tHu.json",
    "tttt-prod_workspace.json",
    "tchannel_workspace.json",
]:
    if (FIXTURES_DIR / _name).exists():
        try:
            _ws = json.loads((FIXTURES_DIR / _name).read_text())
            if "channels" in _ws:
                _WORKSPACES.append(_name)
        except (json.JSONDecodeError, KeyError):
            pass


@pytest.mark.parametrize("fixture", _WORKSPACES)
def test_fast_vs_parity_nll_tolerance(fixture):
    """Fast and Parity NLL must agree within 1e-10 at init + 10 random points."""
    ws = load_fixture(fixture)
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    init = model.suggested_init()
    bounds = model.suggested_bounds()

    rng = random.Random(2025)
    param_sets = [init]
    for _ in range(10):
        param_sets.append(sample_params(rng, init, bounds))

    for k, params in enumerate(param_sets):
        nextstat.set_eval_mode("fast")
        nll_fast = model.nll(params)

        nextstat.set_eval_mode("parity")
        nll_parity = model.nll(params)

        diff = abs(nll_fast - nll_parity)
        assert diff < NLL_FAST_VS_PARITY_ATOL, (
            f"{fixture} point {k}: fast={nll_fast:.15e}, parity={nll_parity:.15e}, "
            f"diff={diff:.2e}, tol={NLL_FAST_VS_PARITY_ATOL:.0e}"
        )


@pytest.mark.parametrize("fixture", _WORKSPACES)
def test_fast_vs_parity_grad_tolerance(fixture):
    """Fast and Parity gradient must agree within 1e-10 at init."""
    ws = load_fixture(fixture)
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    params = model.suggested_init()

    nextstat.set_eval_mode("fast")
    grad_fast = model.grad_nll(params)

    nextstat.set_eval_mode("parity")
    grad_parity = model.grad_nll(params)

    for i, (gf, gp) in enumerate(zip(grad_fast, grad_parity)):
        diff = abs(gf - gp)
        assert diff < 1e-10, (
            f"{fixture} param {i}: fast_grad={gf:.10e}, parity_grad={gp:.10e}, diff={diff:.2e}"
        )
