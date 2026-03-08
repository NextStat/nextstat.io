"""Simulation-Based Calibration (SBC) for NUTS (Sprint 5.4.2).

These tests are intentionally slow and opt-in.

Run with:
  NS_RUN_SLOW=1 NS_RUN_SBC=1 NS_SBC_RUNS=30 NS_SBC_WARMUP=300 NS_SBC_SAMPLES=300 \
    PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q -m slow tests/python/test_sbc_nuts.py
"""

import pytest

from _sbc_sampler_suite import (
    run_moments_golden_gaussian_strict,
    run_quality_gate_glm_strict,
    run_sbc_linear_regression_1d_mean_only,
    run_sbc_linear_regression_2d,
    run_sbc_random_intercept_bernoulli_smoke,
    run_sbc_random_intercept_gaussian_smoke,
)

pytestmark = [pytest.mark.slow, pytest.mark.sbc]


def test_sbc_linear_regression_1d_mean_only():
    run_sbc_linear_regression_1d_mean_only("nuts")


def test_sbc_linear_regression_2d():
    run_sbc_linear_regression_2d("nuts")


def test_sbc_random_intercept_gaussian_smoke():
    run_sbc_random_intercept_gaussian_smoke("nuts")


def test_sbc_random_intercept_bernoulli_smoke():
    run_sbc_random_intercept_bernoulli_smoke("nuts")


def test_nuts_quality_gate_glm_strict():
    run_quality_gate_glm_strict("nuts")


def test_nuts_moments_golden_gaussian_strict():
    run_moments_golden_gaussian_strict("nuts")
