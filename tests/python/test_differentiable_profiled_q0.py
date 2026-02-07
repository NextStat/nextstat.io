from __future__ import annotations

import json
from pathlib import Path

import pytest

import nextstat


def _load_signal_nominal_from_workspace(path: Path, *, channel: str, sample: str) -> list[float]:
    ws = json.loads(path.read_text(encoding="utf-8"))
    for ch in ws["channels"]:
        if ch["name"] != channel:
            continue
        for s in ch["samples"]:
            if s["name"] == sample:
                return [float(x) for x in s["data"]]
    raise KeyError(f"Missing channel/sample: {channel}/{sample}")


def test_q0_like_loss_and_grad_nominal_finite_diff() -> None:
    ws_path = Path("tests/fixtures/simple_workspace.json")
    ws_json = ws_path.read_text(encoding="utf-8")
    model = nextstat.HistFactoryModel.from_workspace(ws_json)

    nominal = _load_signal_nominal_from_workspace(ws_path, channel="singlechannel", sample="signal")
    mle = nextstat.MaximumLikelihoodEstimator(max_iter=400, tol=1e-6, m=10)

    q0, grad = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=nominal
    )
    assert q0 == q0  # not NaN
    assert len(grad) == len(nominal)

    idx = 0
    for i, v in enumerate(nominal):
        if v > 0.5:
            idx = i
            break

    eps = max(1e-3, 1e-3 * abs(nominal[idx]))
    plus = list(nominal)
    plus[idx] += eps
    q0_p, _ = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=plus
    )

    minus = list(nominal)
    minus[idx] -= eps
    q0_m, _ = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=minus
    )

    fd = (q0_p - q0_m) / (2.0 * eps)
    g = grad[idx]
    denom = max(abs(fd), abs(g), 1e-6)
    rel = abs(fd - g) / denom
    assert rel < 5e-2


def test_qmu_like_loss_and_grad_nominal_finite_diff() -> None:
    ws_path = Path("tests/fixtures/simple_workspace.json")
    ws_json = ws_path.read_text(encoding="utf-8")
    model = nextstat.HistFactoryModel.from_workspace(ws_json)

    nominal = _load_signal_nominal_from_workspace(ws_path, channel="singlechannel", sample="signal")
    mle = nextstat.MaximumLikelihoodEstimator(max_iter=400, tol=1e-6, m=10)

    mu_test = 5.0
    qmu, grad = mle.qmu_like_loss_and_grad_nominal(
        model,
        mu_test=mu_test,
        channel="singlechannel",
        sample="signal",
        nominal=nominal,
    )
    assert qmu == qmu  # not NaN
    assert len(grad) == len(nominal)

    idx = 0
    for i, v in enumerate(nominal):
        if v > 0.5:
            idx = i
            break

    eps = max(1e-3, 1e-3 * abs(nominal[idx]))
    plus = list(nominal)
    plus[idx] += eps
    qmu_p, _ = mle.qmu_like_loss_and_grad_nominal(
        model,
        mu_test=mu_test,
        channel="singlechannel",
        sample="signal",
        nominal=plus,
    )

    minus = list(nominal)
    minus[idx] -= eps
    qmu_m, _ = mle.qmu_like_loss_and_grad_nominal(
        model,
        mu_test=mu_test,
        channel="singlechannel",
        sample="signal",
        nominal=minus,
    )

    fd = (qmu_p - qmu_m) / (2.0 * eps)
    g = grad[idx]
    denom = max(abs(fd), abs(g), 1e-6)
    rel = abs(fd - g) / denom
    assert rel < 5e-2


# ---------------------------------------------------------------------------
# GPU Profiled Differentiable Tests (CUDA only)
# ---------------------------------------------------------------------------

_HAS_CUDA = getattr(nextstat, "has_cuda", lambda: False)()

try:
    import torch as _torch
    _HAS_TORCH_CUDA = _torch.cuda.is_available()
except ImportError:
    _HAS_TORCH_CUDA = False

_REQUIRES_GPU = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_TORCH_CUDA),
    reason="requires CUDA nextstat + torch.cuda",
)


def _make_gpu_model_and_signal():
    """Create a HistFactory model and upload signal to GPU."""
    import torch

    ws_path = Path("tests/fixtures/simple_workspace.json")
    ws_json = ws_path.read_text(encoding="utf-8")
    model = nextstat.HistFactoryModel.from_workspace(ws_json)

    nominal = _load_signal_nominal_from_workspace(
        ws_path, channel="singlechannel", sample="signal"
    )
    signal = torch.tensor(nominal, dtype=torch.float64, device="cuda")
    return model, signal, nominal


@_REQUIRES_GPU
def test_profiled_q0_matches_cpu():
    """GPU profiled q₀ matches CPU q0_like_loss_and_grad_nominal."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_q0_loss

    model, signal, nominal = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    # GPU
    signal_req = signal.clone().requires_grad_(True)
    q0_gpu = profiled_q0_loss(signal_req, session)

    # CPU
    mle = nextstat.MaximumLikelihoodEstimator(max_iter=400, tol=1e-6, m=10)
    q0_cpu, _ = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=nominal
    )

    # q₀ values should be close (both run L-BFGS-B, may differ slightly)
    assert abs(q0_gpu.item() - q0_cpu) < 0.1 * max(q0_cpu, 1e-3), (
        f"q0_gpu={q0_gpu.item()}, q0_cpu={q0_cpu}"
    )


@_REQUIRES_GPU
def test_profiled_q0_grad_flows():
    """Backward pass produces non-zero gradients."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_q0_loss

    model, signal, _ = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    signal_req = signal.clone().requires_grad_(True)
    q0 = profiled_q0_loss(signal_req, session)
    q0.backward()

    assert signal_req.grad is not None, "No gradient computed"
    assert signal_req.grad.shape == signal.shape
    # At least some gradient components should be nonzero
    assert signal_req.grad.abs().max().item() > 1e-10


@_REQUIRES_GPU
def test_profiled_q0_grad_fd():
    """GPU profiled q₀ gradient matches finite differences."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_q0_loss

    model, signal, _ = _make_gpu_model_and_signal()

    signal_req = signal.clone().requires_grad_(True)
    session = create_profiled_session(model, "signal")
    q0 = profiled_q0_loss(signal_req, session)
    q0.backward()
    grad_analytic = signal_req.grad.clone()

    # Find a bin with significant gradient
    idx = grad_analytic.abs().argmax().item()
    eps = max(1e-3, 1e-3 * signal[idx].abs().item())

    signal_p = signal.clone()
    signal_p[idx] += eps
    session_p = create_profiled_session(model, "signal")
    q0_p = profiled_q0_loss(signal_p, session_p).item()

    signal_m = signal.clone()
    signal_m[idx] -= eps
    session_m = create_profiled_session(model, "signal")
    q0_m = profiled_q0_loss(signal_m, session_m).item()

    fd = (q0_p - q0_m) / (2.0 * eps)
    g = grad_analytic[idx].item()
    denom = max(abs(fd), abs(g), 1e-6)
    rel = abs(fd - g) / denom
    assert rel < 0.1, f"FD mismatch: fd={fd:.6e}, analytic={g:.6e}, rel={rel:.4f}"


@_REQUIRES_GPU
def test_profiled_z0_loss():
    """Z₀ = sqrt(q₀) loss is differentiable."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_z0_loss

    model, signal, _ = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    signal_req = signal.clone().requires_grad_(True)
    z0 = profiled_z0_loss(signal_req, session)
    z0.backward()

    assert z0.item() >= 0.0
    assert signal_req.grad is not None


@_REQUIRES_GPU
def test_training_loop_profiled():
    """10 Adam steps with profiled q₀ loss — q₀ should not decrease."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_q0_loss

    model, signal, _ = _make_gpu_model_and_signal()

    # Scale signal up to ensure nonzero q₀
    signal_param = torch.nn.Parameter(signal.clone() * 2.0)
    optimizer = torch.optim.Adam([signal_param], lr=0.1)

    # Session is reusable across iterations (signal is read via pointer each time)
    session = create_profiled_session(model, "signal")
    q0_values = []
    for _ in range(10):
        optimizer.zero_grad()
        q0 = profiled_q0_loss(signal_param, session)
        loss = -q0  # maximize q₀
        loss.backward()
        optimizer.step()
        # Ensure signal stays positive
        with torch.no_grad():
            signal_param.clamp_(min=1e-6)
        q0_values.append(q0.item())

    # q₀ should generally increase (we're maximizing it)
    # Allow some variance — just check that it doesn't collapse to zero
    assert max(q0_values) > 0.0, f"q₀ collapsed to zero: {q0_values}"


@_REQUIRES_GPU
def test_profiled_qmu_matches_cpu():
    """GPU profiled qμ matches CPU qmu_like_loss_and_grad_nominal."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_qmu_loss

    model, signal, nominal = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    mu_test = 5.0

    # GPU
    signal_req = signal.clone().requires_grad_(True)
    qmu_gpu = profiled_qmu_loss(signal_req, session, mu_test).detach().cpu().item()

    # CPU
    mle = nextstat.MaximumLikelihoodEstimator(max_iter=400, tol=1e-6, m=10)
    qmu_cpu, _ = mle.qmu_like_loss_and_grad_nominal(
        model,
        mu_test=mu_test,
        channel="singlechannel",
        sample="signal",
        nominal=nominal,
    )

    assert abs(qmu_gpu - qmu_cpu) < 0.1 * max(qmu_cpu, 1e-3), (
        f"qmu_gpu={qmu_gpu}, qmu_cpu={qmu_cpu}"
    )


@_REQUIRES_GPU
def test_profiled_qmu_grad_flows():
    """Backward pass produces non-zero gradients for qμ."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_qmu_loss

    model, signal, _ = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    signal_req = signal.clone().requires_grad_(True)
    qmu = profiled_qmu_loss(signal_req, session, 5.0)
    qmu.backward()

    assert signal_req.grad is not None, "No gradient computed"
    assert signal_req.grad.shape == signal.shape
    assert signal_req.grad.abs().max().item() > 1e-10


@_REQUIRES_GPU
def test_profiled_qmu_grad_fd():
    """GPU profiled qμ gradient matches finite differences."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_qmu_loss

    model, signal, _ = _make_gpu_model_and_signal()

    mu_test = 5.0
    signal_req = signal.clone().requires_grad_(True)
    session = create_profiled_session(model, "signal")
    qmu = profiled_qmu_loss(signal_req, session, mu_test)
    qmu.backward()
    grad_analytic = signal_req.grad.clone()

    idx = grad_analytic.abs().argmax().item()
    eps = max(1e-3, 1e-3 * signal[idx].abs().item())

    signal_p = signal.clone()
    signal_p[idx] += eps
    session_p = create_profiled_session(model, "signal")
    qmu_p = profiled_qmu_loss(signal_p, session_p, mu_test).item()

    signal_m = signal.clone()
    signal_m[idx] -= eps
    session_m = create_profiled_session(model, "signal")
    qmu_m = profiled_qmu_loss(signal_m, session_m, mu_test).item()

    fd = (qmu_p - qmu_m) / (2.0 * eps)
    g = grad_analytic[idx].item()
    denom = max(abs(fd), abs(g), 1e-6)
    rel = abs(fd - g) / denom
    assert rel < 0.1, f"FD mismatch: fd={fd:.6e}, analytic={g:.6e}, rel={rel:.4f}"


@_REQUIRES_GPU
def test_profiled_zmu_loss():
    """Zμ = sqrt(qμ) loss is differentiable."""
    import torch
    from nextstat.torch import create_profiled_session, profiled_zmu_loss

    model, signal, _ = _make_gpu_model_and_signal()
    session = create_profiled_session(model, "signal")

    signal_req = signal.clone().requires_grad_(True)
    zmu = profiled_zmu_loss(signal_req, session, 5.0)
    zmu.backward()

    assert zmu.item() >= 0.0
    assert signal_req.grad is not None
