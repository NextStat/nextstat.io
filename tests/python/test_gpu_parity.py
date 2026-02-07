"""GPU vs CPU parity tests.

Tests that GPU-accelerated paths produce results within tolerance of CPU paths.
Both CUDA (f64) and Metal (f32) tolerances are tested.
Skipped if the corresponding GPU feature is not compiled in.
"""

import json
import math

import pytest

import nextstat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_WORKSPACE = {
    "channels": [
        {
            "name": "singlechannel",
            "samples": [
                {
                    "name": "signal",
                    "data": [12.0, 11.0],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                    ],
                },
                {
                    "name": "background",
                    "data": [50.0, 52.0],
                    "modifiers": [
                        {
                            "name": "uncorr_bkguncrt",
                            "type": "shapesys",
                            "data": [3.0, 7.0],
                        }
                    ],
                },
            ],
        }
    ],
    "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
    "measurements": [
        {
            "name": "Measurement",
            "config": {
                "poi": "mu",
                "parameters": [],
            },
        }
    ],
    "version": "1.0.0",
}


@pytest.fixture
def simple_model():
    """Single-channel model with NormFactor signal + ShapeSys background."""
    ws_json = json.dumps(SIMPLE_WORKSPACE)
    return nextstat.from_pyhf(ws_json)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cpu_fit(model):
    """CPU MLE fit."""
    mle = nextstat.MaximumLikelihoodEstimator()
    return mle.fit(model)


def close_enough(a, b, atol):
    """Check if two values are within absolute tolerance."""
    return abs(a - b) < atol


# ---------------------------------------------------------------------------
# CUDA parity tests (require --features cuda)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not nextstat.has_cuda(), reason="CUDA not available")
class TestCudaParity:
    """CUDA GPU vs CPU parity."""

    def test_fit_gpu_vs_cpu(self, simple_model):
        """GPU fit NLL and params match CPU."""
        from _tolerances import GPU_FIT_NLL_ATOL, GPU_PARAM_ATOL

        cpu_result = cpu_fit(simple_model)

        mle = nextstat.MaximumLikelihoodEstimator()
        gpu_result = mle.fit_gpu(simple_model)

        assert close_enough(gpu_result.nll, cpu_result.nll, GPU_FIT_NLL_ATOL), (
            f"GPU NLL {gpu_result.nll} vs CPU NLL {cpu_result.nll}, "
            f"diff={abs(gpu_result.nll - cpu_result.nll):.2e}"
        )

        for i, (gp, cp) in enumerate(
            zip(gpu_result.parameters, cpu_result.parameters)
        ):
            assert close_enough(gp, cp, GPU_PARAM_ATOL), (
                f"Param[{i}]: GPU={gp} vs CPU={cp}, diff={abs(gp - cp):.2e}"
            )

    def test_profile_scan_gpu_vs_cpu(self, simple_model):
        """GPU profile scan matches CPU scan points."""
        from _tolerances import GPU_FIT_NLL_ATOL

        cpu_scan = nextstat.profile_scan(simple_model, n_points=5)
        gpu_scan = nextstat.profile_scan(simple_model, n_points=5, use_gpu=True)

        assert len(cpu_scan) == len(gpu_scan)
        for cp, gp in zip(cpu_scan, gpu_scan):
            assert close_enough(gp[0], cp[0], 1e-6), f"mu: {gp[0]} vs {cp[0]}"
            assert close_enough(gp[1], cp[1], GPU_FIT_NLL_ATOL), (
                f"NLL at mu={gp[0]}: GPU={gp[1]}, CPU={cp[1]}"
            )

    def test_ranking_gpu_vs_cpu(self, simple_model):
        """GPU ranking matches CPU ranking."""
        from _tolerances import GPU_PARAM_ATOL

        cpu_ranking = nextstat.ranking(simple_model)
        gpu_ranking = nextstat.ranking_gpu(simple_model)

        assert len(cpu_ranking) == len(gpu_ranking)
        for cr, gr in zip(cpu_ranking, gpu_ranking):
            assert cr["name"] == gr["name"]
            assert close_enough(
                gr["impact_up"], cr["impact_up"], GPU_PARAM_ATOL
            ), f"{cr['name']} impact_up: GPU={gr['impact_up']} vs CPU={cr['impact_up']}"
            assert close_enough(
                gr["impact_down"], cr["impact_down"], GPU_PARAM_ATOL
            ), f"{cr['name']} impact_down: GPU={gr['impact_down']} vs CPU={cr['impact_down']}"

    def test_fit_toys_batch_gpu_deterministic(self, simple_model):
        """Same seed → same result for GPU batch toy fit."""
        from _tolerances import GPU_NLL_ATOL

        result1 = nextstat.fit_toys_batch(
            simple_model, n_toys=10, seed=42, use_gpu=True
        )
        result2 = nextstat.fit_toys_batch(
            simple_model, n_toys=10, seed=42, use_gpu=True
        )

        for nll1, nll2 in zip(result1["nll"], result2["nll"]):
            assert close_enough(nll1, nll2, GPU_NLL_ATOL), (
                f"Non-deterministic: {nll1} vs {nll2}"
            )


# ---------------------------------------------------------------------------
# Metal parity tests (require --features metal)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not nextstat.has_metal(), reason="Metal not available")
class TestMetalParity:
    """Metal GPU vs CPU parity (f32 computation)."""

    def test_fit_toys_batch_metal_vs_cpu(self, simple_model):
        """Metal batch toy NLL within f32 tolerance of CPU."""
        from _tolerances import METAL_NLL_ATOL

        cpu_result = nextstat.fit_toys_batch(
            simple_model, n_toys=5, seed=42
        )
        metal_result = nextstat.fit_toys_batch(
            simple_model, n_toys=5, seed=42, use_metal=True
        )

        for nll_cpu, nll_metal in zip(
            cpu_result["nll"], metal_result["nll"]
        ):
            assert close_enough(nll_metal, nll_cpu, METAL_NLL_ATOL), (
                f"Metal NLL {nll_metal} vs CPU NLL {nll_cpu}, "
                f"diff={abs(nll_metal - nll_cpu):.2e}"
            )
