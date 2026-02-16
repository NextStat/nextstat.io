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
    return nextstat.fit(model, device="cpu")


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
        gpu_result = nextstat.fit(simple_model, device="cuda")

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

        mu_values = [0.0, 0.25, 0.50, 0.75, 1.0]
        cpu_scan = nextstat.profile_scan(simple_model, mu_values, device="cpu")
        gpu_scan = nextstat.profile_scan(simple_model, mu_values, device="cuda")

        assert int(cpu_scan["poi_index"]) == int(gpu_scan["poi_index"])
        assert len(cpu_scan["points"]) == len(gpu_scan["points"])
        for cp, gp in zip(cpu_scan["points"], gpu_scan["points"]):
            assert close_enough(gp["mu"], cp["mu"], 1e-12), f"mu: {gp['mu']} vs {cp['mu']}"
            assert close_enough(gp["nll_mu"], cp["nll_mu"], GPU_FIT_NLL_ATOL), (
                f"NLL at mu={gp['mu']}: GPU={gp['nll_mu']}, CPU={cp['nll_mu']}"
            )

    def test_fit_toys_gpu_deterministic(self, simple_model):
        """Same seed → same result for GPU batch toy fit."""
        from _tolerances import GPU_NLL_ATOL

        # Use a fixed init point for determinism.
        init = list(simple_model.suggested_init())
        result1 = nextstat.fit_toys(
            simple_model, init, n_toys=10, seed=42, device="cuda"
        )
        result2 = nextstat.fit_toys(
            simple_model, init, n_toys=10, seed=42, device="cuda"
        )

        for r1, r2 in zip(result1, result2):
            assert close_enough(r1.nll, r2.nll, GPU_NLL_ATOL), (
                f"Non-deterministic: {r1.nll} vs {r2.nll}"
            )


# ---------------------------------------------------------------------------
# Metal parity tests (require --features metal)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not nextstat.has_metal(), reason="Metal not available")
class TestMetalParity:
    """Metal GPU vs CPU parity (f32 computation)."""

    def test_fit_toys_metal_vs_cpu(self, simple_model):
        """Metal batch toy NLL within f32 tolerance of CPU."""
        from _tolerances import METAL_NLL_ATOL

        init = list(simple_model.suggested_init())
        cpu_result = nextstat.fit_toys(
            simple_model, init, n_toys=5, seed=42
        )
        metal_result = nextstat.fit_toys(
            simple_model, init, n_toys=5, seed=42, device="metal"
        )

        for rc, rm in zip(cpu_result, metal_result):
            assert close_enough(rm.nll, rc.nll, METAL_NLL_ATOL), (
                f"Metal NLL {rm.nll} vs CPU NLL {rc.nll}, "
                f"diff={abs(rm.nll - rc.nll):.2e}"
            )
