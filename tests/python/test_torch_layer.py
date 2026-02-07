"""Integration tests for the CUDA zero-copy differentiable PyTorch layer.

These tests require:
  - nextstat built with --features cuda
  - PyTorch with CUDA support
  - An NVIDIA GPU

Tests are skipped automatically if requirements are not met.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip entire module if CUDA or PyTorch unavailable
try:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

try:
    import nextstat
    from nextstat.torch import NextStatNLL, create_session, nll_loss

    if nextstat.DifferentiableSession is None:
        pytest.skip("DifferentiableSession not available (build without CUDA?)", allow_module_level=True)
except ImportError:
    pytest.skip("nextstat not installed", allow_module_level=True)


WS_PATH = Path("tests/fixtures/simple_workspace.json")


def _load_workspace():
    ws_json = WS_PATH.read_text(encoding="utf-8")
    model = nextstat.HistFactoryModel.from_workspace(ws_json)
    return model


def _load_signal_nominal(ws_path: Path, channel: str, sample: str) -> list[float]:
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    for ch in ws["channels"]:
        if ch["name"] != channel:
            continue
        for s in ch["samples"]:
            if s["name"] == sample:
                return [float(x) for x in s["data"]]
    raise KeyError(f"Missing {channel}/{sample}")


class TestNextStatNLL:
    """Tests for the CUDA zero-copy differentiable NLL layer."""

    def test_nll_loss_grad_flows(self):
        """signal.requires_grad=True → .backward() → signal.grad is not None."""
        model = _load_workspace()
        session = create_session(model, signal_sample_name="signal")

        nominal = _load_signal_nominal(WS_PATH, "singlechannel", "signal")
        signal = torch.tensor(nominal, dtype=torch.float64, device="cuda", requires_grad=True)

        loss = nll_loss(signal, session)
        loss.backward()

        assert signal.grad is not None
        assert signal.grad.shape == signal.shape
        assert torch.all(torch.isfinite(signal.grad))

    def test_nll_loss_matches_cpu(self):
        """NLL from torch layer matches model.nll() at same parameters."""
        model = _load_workspace()
        session = create_session(model, signal_sample_name="signal")

        nominal = _load_signal_nominal(WS_PATH, "singlechannel", "signal")
        signal = torch.tensor(nominal, dtype=torch.float64, device="cuda")

        init_params = session.parameter_init()
        params_tensor = torch.tensor(init_params, dtype=torch.float64)

        # GPU NLL
        gpu_nll = nll_loss(signal, session, params=params_tensor).item()

        # CPU NLL (at same params, with same signal = nominal)
        cpu_nll = model.nll(init_params)

        # Should be very close (signal == nominal → same model)
        assert abs(gpu_nll - cpu_nll) < 1e-6, f"GPU NLL {gpu_nll} vs CPU NLL {cpu_nll}"

    def test_grad_numerical_vs_analytical(self):
        """Analytical gradient matches finite differences."""
        model = _load_workspace()
        session = create_session(model, signal_sample_name="signal")

        nominal = _load_signal_nominal(WS_PATH, "singlechannel", "signal")
        signal = torch.tensor(nominal, dtype=torch.float64, device="cuda", requires_grad=True)

        # Analytical gradient
        loss = nll_loss(signal, session)
        loss.backward()
        analytical_grad = signal.grad.clone()

        # Finite differences
        eps = 1e-5
        fd_grad = torch.zeros_like(signal)
        for i in range(len(nominal)):
            signal_plus = signal.data.clone()
            signal_plus[i] += eps
            signal_minus = signal.data.clone()
            signal_minus[i] -= eps

            # Need fresh session calls (no grad tracking)
            nll_plus = nll_loss(signal_plus.detach(), session).item()
            nll_minus = nll_loss(signal_minus.detach(), session).item()
            fd_grad[i] = (nll_plus - nll_minus) / (2 * eps)

        # Compare
        max_err = (analytical_grad - fd_grad).abs().max().item()
        assert max_err < 1e-4, f"Max gradient error: {max_err}"

    def test_training_loop_smoke(self):
        """10 steps of Adam on simple model: NLL should generally decrease."""
        model = _load_workspace()
        session = create_session(model, signal_sample_name="signal")

        nominal = _load_signal_nominal(WS_PATH, "singlechannel", "signal")
        # Start with slightly perturbed signal
        signal = torch.tensor(
            [x * 1.5 for x in nominal], dtype=torch.float64, device="cuda", requires_grad=True
        )

        optimizer = torch.optim.Adam([signal], lr=0.1)
        losses = []

        for _ in range(10):
            optimizer.zero_grad()
            loss = nll_loss(signal, session)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # NLL should decrease overall (first < last is ok due to noise,
        # but general trend should be downward)
        assert losses[-1] < losses[0], f"NLL did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_session_metadata(self):
        """DifferentiableSession exposes correct metadata."""
        model = _load_workspace()
        session = create_session(model, signal_sample_name="signal")

        assert session.signal_n_bins() > 0
        assert session.n_params() > 0
        assert len(session.parameter_init()) == session.n_params()

    def test_invalid_signal_sample_name(self):
        """Creating session with nonexistent sample name raises."""
        model = _load_workspace()
        with pytest.raises((ValueError, RuntimeError)):
            create_session(model, signal_sample_name="nonexistent_sample")
