"""Tests for set_eval_mode / get_eval_mode Python API."""

import json
from pathlib import Path

import pytest

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def restore_fast_mode():
    """Ensure fast mode is restored after each test."""
    yield
    nextstat.set_eval_mode("fast")


def test_default_mode_is_fast():
    assert nextstat.get_eval_mode() == "fast"


def test_set_parity_mode():
    nextstat.set_eval_mode("parity")
    assert nextstat.get_eval_mode() == "parity"


def test_set_fast_mode():
    nextstat.set_eval_mode("parity")
    nextstat.set_eval_mode("fast")
    assert nextstat.get_eval_mode() == "fast"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown eval mode"):
        nextstat.set_eval_mode("turbo")


def test_nll_parity_vs_fast_within_tolerance():
    """NLL in parity mode vs fast mode should differ by < 1e-12."""
    workspace = json.loads((FIXTURES_DIR / "simple_workspace.json").read_text())
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    params = model.suggested_init()

    nextstat.set_eval_mode("fast")
    nll_fast = model.nll(params)

    nextstat.set_eval_mode("parity")
    nll_parity = model.nll(params)

    diff = abs(nll_fast - nll_parity)
    assert diff < 1e-12, f"Parity vs Fast NLL diff={diff:.2e}"


def test_grad_parity_vs_fast_within_tolerance():
    """Gradient in parity mode vs fast mode should be very close."""
    workspace = json.loads((FIXTURES_DIR / "simple_workspace.json").read_text())
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    params = model.suggested_init()

    nextstat.set_eval_mode("fast")
    grad_fast = model.grad_nll(params)

    nextstat.set_eval_mode("parity")
    grad_parity = model.grad_nll(params)

    for i, (gf, gp) in enumerate(zip(grad_fast, grad_parity)):
        diff = abs(gf - gp)
        assert diff < 1e-12, (
            f"Param {i}: fast_grad={gf}, parity_grad={gp}, diff={diff:.2e}"
        )
