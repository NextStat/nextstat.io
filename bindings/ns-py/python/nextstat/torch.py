"""PyTorch helpers (optional dependency).

This module provides autograd wrappers that call into the native NextStat
extension. It is intentionally small and only imported when needed.
"""

from __future__ import annotations

from typing import Any, Optional


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for `nextstat.torch` helpers. Install torch and retry."
        ) from e


def _to_cpu_f64_1d(x: Any):
    torch = _require_torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape={tuple(x.shape)}")
    if x.device.type != "cpu":
        raise ValueError("Only CPU tensors are supported for now (move tensor to CPU).")
    return x.detach().to(dtype=torch.float64).contiguous()


def _as_buffer_1d_f64(x_cpu_f64):
    # Prefer a buffer-protocol object for fast extraction in PyO3.
    try:
        import numpy as np  # type: ignore

        if hasattr(x_cpu_f64, "numpy"):
            arr = x_cpu_f64.numpy()
            if isinstance(arr, np.ndarray):
                return arr
    except Exception:
        pass
    return x_cpu_f64.tolist()


class NextStatQ0:
    """Autograd wrapper for discovery-style `q0` (profiled) w.r.t. nominal yields."""

    @staticmethod
    def apply(
        nominal,
        *,
        mle,
        model,
        channel: str,
        sample: str,
    ):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                x_cpu = _to_cpu_f64_1d(x)
                q0, grad = mle.q0_like_loss_and_grad_nominal(
                    model, channel=channel, sample=sample, nominal=_as_buffer_1d_f64(x_cpu)
                )
                g = torch.tensor(grad, dtype=x.dtype, device=x.device)
                ctx.save_for_backward(g)
                return torch.tensor(q0, dtype=x.dtype, device=x.device)

            @staticmethod
            def backward(ctx, grad_output):
                (g,) = ctx.saved_tensors
                return grad_output * g

        return _Fn.apply(nominal)


class NextStatZ0:
    """Autograd wrapper for discovery significance `Z0 = sqrt(q0)` (profiled)."""

    @staticmethod
    def apply(
        nominal,
        *,
        mle,
        model,
        channel: str,
        sample: str,
        eps: float = 1e-12,
    ):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                x_cpu = _to_cpu_f64_1d(x)
                q0, grad_q0 = mle.q0_like_loss_and_grad_nominal(
                    model, channel=channel, sample=sample, nominal=_as_buffer_1d_f64(x_cpu)
                )
                z = (q0 + float(eps)) ** 0.5
                scale = 0.5 / z
                g = torch.tensor([scale * v for v in grad_q0], dtype=x.dtype, device=x.device)
                ctx.save_for_backward(g)
                return torch.tensor(z, dtype=x.dtype, device=x.device)

            @staticmethod
            def backward(ctx, grad_output):
                (g,) = ctx.saved_tensors
                return grad_output * g

        return _Fn.apply(nominal)


def discovery_z0(
    nominal,
    *,
    mle,
    model,
    channel: str,
    sample: str,
    eps: float = 1e-12,
):
    """Convenience: returns `Z0` as a differentiable scalar tensor."""
    return NextStatZ0.apply(nominal, mle=mle, model=model, channel=channel, sample=sample, eps=eps)


# ---------------------------------------------------------------------------
# CUDA Zero-Copy Differentiable Layer
# ---------------------------------------------------------------------------


def _get_diff_session_class():
    """Import DifferentiableSession from native extension (CUDA only)."""
    try:
        from nextstat._core import DifferentiableSession  # type: ignore

        return DifferentiableSession
    except ImportError:
        return None


class NextStatNLL:
    """CUDA zero-copy differentiable NLL for PyTorch training loops.

    Enables end-to-end gradient-based optimization of neural network
    classifiers directly on physics significance, with full systematic
    uncertainty handling.

    The signal histogram lives on GPU (PyTorch CUDA tensor). NextStat's
    CUDA kernel reads it directly (zero-copy) and writes the gradient
    back into a PyTorch tensor — no Host↔Device transfers.

    References:
        - neos (gradhep): https://github.com/gradhep/neos
        - arXiv:2203.05570: End-to-End-Optimised Summary Statistics
        - arXiv:2508.17802: Differentiating a HEP Analysis Pipeline
    """

    @staticmethod
    def apply(signal_histogram, session, params_tensor):
        """Compute differentiable NLL.

        Args:
            signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
            session: DifferentiableSession (from create_session())
            params_tensor: torch.Tensor — nuisance parameters (detached)

        Returns:
            NLL scalar (torch.Tensor with grad_fn on same device)
        """
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, signal):
                assert signal.is_cuda, "signal must be on CUDA"
                assert signal.dtype == torch.float64, "signal must be float64"
                assert signal.is_contiguous(), "signal must be contiguous"

                # Pre-allocate gradient tensor on same device (zeroed — kernel uses atomicAdd)
                grad_signal = torch.zeros_like(signal)

                # Sync PyTorch's CUDA stream before NextStat kernel launch
                torch.cuda.synchronize()

                # Convert params via numpy (faster than .tolist() for large vectors)
                params_list = params_tensor.detach().cpu().numpy().tolist()

                # Zero-copy: pass raw device pointers to Rust → CUDA kernel
                nll_val = session.nll_grad_signal(
                    params_list,
                    signal.data_ptr(),
                    grad_signal.data_ptr(),
                )

                # Sync after kernel: NextStat launches on its own CUDA stream,
                # so we must wait for it before PyTorch reads grad_signal.
                torch.cuda.synchronize()

                ctx.save_for_backward(grad_signal)
                return signal.new_tensor(nll_val)

            @staticmethod
            def backward(ctx, grad_output):
                (grad_signal,) = ctx.saved_tensors
                return grad_output * grad_signal

        return _Fn.apply(signal_histogram)


def create_session(model, signal_sample_name: str = "signal"):
    """Create a differentiable GPU session for a HistFactory model.

    Args:
        model: nextstat.HistFactoryModel
        signal_sample_name: name of the signal sample in the workspace

    Returns:
        DifferentiableSession ready for use in training loop

    Raises:
        ImportError: if CUDA support is not compiled in
        ValueError: if signal sample not found
    """
    DiffSession = _get_diff_session_class()
    if DiffSession is None:
        raise ImportError(
            "DifferentiableSession requires CUDA support. "
            "Build nextstat with --features cuda."
        )
    return DiffSession(model, signal_sample_name)


def nll_loss(signal_histogram, session, params=None):
    """Compute differentiable NLL loss.

    This is the main entry point for using NextStat as a differentiable
    loss function in PyTorch training loops.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
            Output of neural network (soft histogram of predicted signal)
        session: DifferentiableSession from create_session()
        params: optional nuisance parameter tensor (default: model init)

    Returns:
        NLL scalar (torch.Tensor with grad_fn, differentiable w.r.t. signal)

    Example::

        import nextstat
        from nextstat.torch import create_session, nll_loss

        model = nextstat.from_pyhf(workspace_json)
        session = create_session(model, signal_sample_name="signal")

        # Training loop
        for batch in dataloader:
            signal_hist = nn(batch).double().cuda()  # NN → histogram
            loss = nll_loss(signal_hist, session)
            loss.backward()
            optimizer.step()
    """
    torch = _require_torch()
    if params is None:
        params = torch.tensor(session.parameter_init(), dtype=torch.float64)
    return NextStatNLL.apply(signal_histogram, session, params)


# ---------------------------------------------------------------------------
# CUDA Profiled Significance (Phase 2)
# ---------------------------------------------------------------------------


def _get_profiled_session_class():
    """Import ProfiledDifferentiableSession from native extension (CUDA only)."""
    try:
        from nextstat._core import ProfiledDifferentiableSession  # type: ignore

        return ProfiledDifferentiableSession
    except ImportError:
        return None


class ProfiledQ0Loss:
    """GPU-accelerated profiled q₀ with envelope theorem gradient.

    Computes q₀ = 2·(NLL(μ=0,θ̂₀) − NLL(μ̂,θ̂)) and ∂q₀/∂signal
    using two GPU L-BFGS-B fits per forward pass. The gradient uses
    the envelope theorem (exact at convergence).

    Example::

        from nextstat.torch import create_profiled_session, profiled_q0_loss

        session = create_profiled_session(model, "signal")

        for batch in dataloader:
            signal_hist = nn(batch).double().cuda()
            loss = -profiled_q0_loss(signal_hist, session)  # maximize q₀
            loss.backward()
            optimizer.step()
    """

    @staticmethod
    def apply(signal_histogram, session):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, signal):
                assert signal.is_cuda, "signal must be on CUDA"
                assert signal.dtype == torch.float64, "signal must be float64"
                assert signal.is_contiguous(), "signal must be contiguous"

                torch.cuda.synchronize()

                q0, grad_list = session.profiled_q0_and_grad(signal.data_ptr())

                torch.cuda.synchronize()

                grad_signal = torch.tensor(
                    grad_list, dtype=torch.float64, device=signal.device
                )
                ctx.save_for_backward(grad_signal)
                return signal.new_tensor(q0)

            @staticmethod
            def backward(ctx, grad_output):
                (grad_signal,) = ctx.saved_tensors
                return grad_output * grad_signal

        return _Fn.apply(signal_histogram)


class ProfiledQmuLoss:
    """GPU-accelerated profiled qμ with envelope theorem gradient."""

    @staticmethod
    def apply(signal_histogram, session, mu_test):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, signal):
                assert signal.is_cuda, "signal must be on CUDA"
                assert signal.dtype == torch.float64, "signal must be float64"
                assert signal.is_contiguous(), "signal must be contiguous"

                torch.cuda.synchronize()

                qmu, grad_list = session.profiled_qmu_and_grad(mu_test, signal.data_ptr())

                torch.cuda.synchronize()

                grad_signal = torch.tensor(
                    grad_list, dtype=torch.float64, device=signal.device
                )
                ctx.save_for_backward(grad_signal)
                return signal.new_tensor(qmu)

            @staticmethod
            def backward(ctx, grad_output):
                (grad_signal,) = ctx.saved_tensors
                return grad_output * grad_signal

        return _Fn.apply(signal_histogram)


def create_profiled_session(model, signal_sample_name: str = "signal"):
    """Create a GPU session for profiled significance training.

    Args:
        model: nextstat.HistFactoryModel
        signal_sample_name: name of the signal sample in the workspace

    Returns:
        ProfiledDifferentiableSession ready for use in training loop

    Raises:
        ImportError: if CUDA support is not compiled in
        ValueError: if signal sample not found or no POI defined
    """
    cls = _get_profiled_session_class()
    if cls is None:
        raise ImportError(
            "ProfiledDifferentiableSession requires CUDA support. "
            "Build nextstat with --features cuda."
        )
    return cls(model, signal_sample_name)


def profiled_q0_loss(signal_histogram, session):
    """Compute differentiable profiled q₀ loss.

    Main entry point for training NNs to maximize discovery significance.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
        session: ProfiledDifferentiableSession from create_profiled_session()

    Returns:
        q₀ scalar (torch.Tensor with grad_fn, differentiable w.r.t. signal)
    """
    return ProfiledQ0Loss.apply(signal_histogram, session)


def profiled_z0_loss(signal_histogram, session, eps=1e-12):
    """Compute differentiable Z₀ = sqrt(q₀) loss.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
        session: ProfiledDifferentiableSession from create_profiled_session()
        eps: small constant for numerical stability

    Returns:
        Z₀ scalar (torch.Tensor with grad_fn)
    """
    q0 = profiled_q0_loss(signal_histogram, session)
    return (q0 + eps).sqrt()


def profiled_qmu_loss(signal_histogram, session, mu_test):
    """Compute differentiable profiled qμ loss.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
        session: ProfiledDifferentiableSession from create_profiled_session()
        mu_test: float — signal strength hypothesis

    Returns:
        qμ scalar (torch.Tensor with grad_fn)
    """
    return ProfiledQmuLoss.apply(signal_histogram, session, mu_test)


def profiled_zmu_loss(signal_histogram, session, mu_test, eps=1e-12):
    """Compute differentiable Zμ = sqrt(qμ) loss.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins] on CUDA, float64
        session: ProfiledDifferentiableSession from create_profiled_session()
        mu_test: float — signal strength hypothesis
        eps: small constant for numerical stability

    Returns:
        Zμ scalar (torch.Tensor with grad_fn)
    """
    qmu = profiled_qmu_loss(signal_histogram, session, mu_test)
    return (qmu + eps).sqrt()
