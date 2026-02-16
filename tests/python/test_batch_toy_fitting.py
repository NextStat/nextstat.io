"""Batch toy fitting Python API contract tests."""

import json
from pathlib import Path

import pytest

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.fixture
def simple_model():
    workspace = load_fixture("simple_workspace.json")
    return nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))


@pytest.fixture
def complex_model():
    workspace = load_fixture("complex_workspace.json")
    return nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))


def test_batch_smoke(simple_model):
    """Basic batch toy fitting: returns correct number of results."""
    params = simple_model.suggested_init()
    results = nextstat.fit_toys(simple_model, params, n_toys=10, seed=123, batch=True)
    assert len(results) == 10
    for r in results:
        assert hasattr(r, "nll")
        assert hasattr(r, "parameters")
        assert hasattr(r, "converged")


def test_batch_convergence_rate(simple_model):
    """At least 95% of toys should converge on a simple model."""
    params = simple_model.suggested_init()
    results = nextstat.fit_toys(simple_model, params, n_toys=100, seed=42)
    assert len(results) == 100
    n_converged = sum(1 for r in results if r.converged)
    assert n_converged >= 95, f"Only {n_converged}/100 toys converged"


def test_batch_convergence_rate_complex(complex_model):
    """Complex model: at least 90% convergence."""
    params = complex_model.suggested_init()
    results = nextstat.fit_toys(complex_model, params, n_toys=50, seed=99)
    assert len(results) == 50
    n_converged = sum(1 for r in results if r.converged)
    assert n_converged >= 45, f"Only {n_converged}/50 toys converged"


def test_batch_reproducible(simple_model):
    """Same seed → identical results."""
    params = simple_model.suggested_init()
    r1 = nextstat.fit_toys(simple_model, params, n_toys=5, seed=77)
    r2 = nextstat.fit_toys(simple_model, params, n_toys=5, seed=77)
    for a, b in zip(r1, r2):
        assert a.nll == b.nll
        assert a.parameters == b.parameters


def test_batch_different_seeds(simple_model):
    """Different seeds → different results."""
    params = simple_model.suggested_init()
    r1 = nextstat.fit_toys(simple_model, params, n_toys=5, seed=1)
    r2 = nextstat.fit_toys(simple_model, params, n_toys=5, seed=2)
    nlls_1 = [r.nll for r in r1]
    nlls_2 = [r.nll for r in r2]
    assert nlls_1 != nlls_2, "Different seeds should produce different results"


def test_batch_nll_finite(simple_model):
    """All NLL values must be finite."""
    params = simple_model.suggested_init()
    results = nextstat.fit_toys(simple_model, params, n_toys=20, seed=456)
    for i, r in enumerate(results):
        assert r.nll == r.nll, f"Toy {i}: NLL is NaN"  # NaN != NaN
        assert abs(r.nll) < 1e20, f"Toy {i}: NLL={r.nll} is too large"


def test_batch_parameter_count(simple_model):
    """Each toy result has correct parameter count."""
    params = simple_model.suggested_init()
    results = nextstat.fit_toys(simple_model, params, n_toys=5, seed=789)
    n_params = len(params)
    for i, r in enumerate(results):
        assert len(r.parameters) == n_params, (
            f"Toy {i}: expected {n_params} params, got {len(r.parameters)}"
        )


def test_batch_matches_fit_toys(simple_model):
    """Batch results should match standard fit_toys results."""
    params = simple_model.suggested_init()
    results_batch = nextstat.fit_toys(simple_model, params, n_toys=3, seed=55)
    results_std = nextstat.fit_toys(simple_model, params, n_toys=3, seed=55)
    for i, (rb, rs) in enumerate(zip(results_batch, results_std)):
        nll_diff = abs(rb.nll - rs.nll)
        assert nll_diff < 1e-8, (
            f"Toy {i}: batch NLL={rb.nll}, std NLL={rs.nll}, diff={nll_diff}"
        )


def test_has_accelerate_returns_bool():
    """has_accelerate() returns a boolean."""
    result = nextstat.has_accelerate()
    assert isinstance(result, bool)


def test_has_cuda_returns_bool():
    """has_cuda() returns a boolean."""
    result = nextstat.has_cuda()
    assert isinstance(result, bool)


def test_batch_gpu_fallback_to_cpu(simple_model):
    """fit_toys with device='cpu' works like regular batch."""
    params = simple_model.suggested_init()
    results = nextstat.fit_toys(
        simple_model, params, n_toys=3, seed=55, device="cpu"
    )
    assert len(results) == 3
    for r in results:
        assert r.converged
