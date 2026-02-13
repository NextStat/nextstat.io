"""PyTorch helpers (optional dependency).

This module provides autograd wrappers that call into the native NextStat
extension. It is intentionally small and only imported when needed.

It includes:
- Phase 1 (CUDA): a **zero-copy** differentiable HistFactory NLL layer (`nll_loss`)
  whose kernel reads the signal histogram directly from a PyTorch CUDA tensor and
  writes `∂NLL/∂signal` directly back to a PyTorch CUDA tensor.
- Phase 2 (CUDA/Metal): profiled objectives (`q₀`, `qμ`, `Z₀`, `Zμ`) with an
  **envelope-theorem** gradient, avoiding backpropagation through optimizer
  iterations.
- Legacy CPU-only wrappers (`NextStatQ0`/`NextStatZ0`) for nominal-yield vectors.

Notes / limitations:
- CUDA paths require contiguous CUDA `float64` tensors for the signal histogram.
- Profiled CUDA paths return the signal-gradient as a small host vector which is
  then materialized as a CUDA tensor (tiny transfer; the fits dominate runtime).
- Metal profiled paths run on GPU in `f32` and upload the signal via CPU (no
  raw-pointer interop with MPS tensors).
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
                if not signal.is_cuda:
                    raise ValueError("signal must be on CUDA")
                if signal.dtype != torch.float64:
                    raise ValueError("signal must be float64")
                if not signal.is_contiguous():
                    raise ValueError("signal must be contiguous")

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


def _get_metal_profiled_session_class():
    """Import MetalProfiledDifferentiableSession from native extension (Metal only)."""
    try:
        from nextstat._core import MetalProfiledDifferentiableSession  # type: ignore

        return MetalProfiledDifferentiableSession
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
                if not signal.is_cuda:
                    raise ValueError("signal must be on CUDA")
                if signal.dtype != torch.float64:
                    raise ValueError("signal must be float64")
                if not signal.is_contiguous():
                    raise ValueError("signal must be contiguous")

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
                if not signal.is_cuda:
                    raise ValueError("signal must be on CUDA")
                if signal.dtype != torch.float64:
                    raise ValueError("signal must be float64")
                if not signal.is_contiguous():
                    raise ValueError("signal must be contiguous")

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


def create_profiled_session(model, signal_sample_name: str = "signal", device: str = "auto"):
    """Create a GPU session for profiled significance training.

    Args:
        model: nextstat.HistFactoryModel
        signal_sample_name: name of the signal sample in the workspace
        device: "cuda", "metal", or "auto" (default — prefers CUDA, falls back to Metal)

    Returns:
        ProfiledDifferentiableSession (CUDA) or MetalProfiledDifferentiableSession (Metal)

    Raises:
        ImportError: if no GPU support is compiled in
        ValueError: if signal sample not found or no POI defined
    """
    cuda_cls = _get_profiled_session_class()
    metal_cls = _get_metal_profiled_session_class()

    if device == "auto":
        if cuda_cls is not None:
            return cuda_cls(model, signal_sample_name)
        if metal_cls is not None:
            return metal_cls(model, signal_sample_name)
        raise ImportError(
            "ProfiledDifferentiableSession requires CUDA or Metal support. "
            "Build nextstat with --features cuda or --features metal."
        )
    elif device == "cuda":
        if cuda_cls is None:
            raise ImportError(
                "CUDA profiled session not available. Build nextstat with --features cuda."
            )
        return cuda_cls(model, signal_sample_name)
    elif device == "metal":
        if metal_cls is None:
            raise ImportError(
                "Metal profiled session not available. Build nextstat with --features metal."
            )
        return metal_cls(model, signal_sample_name)
    else:
        raise ValueError(f"Unknown device: {device!r}. Use 'cuda', 'metal', or 'auto'.")


def _is_metal_session(session):
    """Check if session is a MetalProfiledDifferentiableSession."""
    return type(session).__name__ == "MetalProfiledDifferentiableSession"


def profiled_q0_loss(signal_histogram, session):
    """Compute differentiable profiled q₀ loss.

    Main entry point for training NNs to maximize discovery significance.
    Auto-detects CUDA or Metal session.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins], float64
            (CUDA for CUDA session, any device for Metal session)
        session: ProfiledDifferentiableSession from create_profiled_session()

    Returns:
        q₀ scalar (torch.Tensor with grad_fn, differentiable w.r.t. signal)
    """
    if _is_metal_session(session):
        return MetalProfiledQ0Loss.apply(signal_histogram, session)
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

    Auto-detects CUDA or Metal session.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins], float64
        session: ProfiledDifferentiableSession from create_profiled_session()
        mu_test: float — signal strength hypothesis

    Returns:
        qμ scalar (torch.Tensor with grad_fn)
    """
    if _is_metal_session(session):
        return MetalProfiledQmuLoss.apply(signal_histogram, session, mu_test)
    return ProfiledQmuLoss.apply(signal_histogram, session, mu_test)


def batch_profiled_qmu_loss(signal_histogram, session, mu_values):
    """Compute profiled qμ for multiple mu_test values in one call.

    Session GPU state is reused across all mu values (model uploaded once).
    Each mu_test requires 2 L-BFGS-B fits.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins], float64
        session: CUDA or Metal profiled session from create_profiled_session()
        mu_values: list[float] — signal strength hypotheses

    Returns:
        list[Tensor] — [qmu_tensor, ...] per mu value (each with grad_fn)
    """
    return [profiled_qmu_loss(signal_histogram, session, mu) for mu in mu_values]


def profiled_zmu_loss(signal_histogram, session, mu_test, eps=1e-12):
    """Compute differentiable Zμ = sqrt(qμ) loss.

    Args:
        signal_histogram: torch.Tensor [n_signal_bins], float64
        session: ProfiledDifferentiableSession from create_profiled_session()
        mu_test: float — signal strength hypothesis
        eps: small constant for numerical stability

    Returns:
        Zμ scalar (torch.Tensor with grad_fn)
    """
    qmu = profiled_qmu_loss(signal_histogram, session, mu_test)
    return (qmu + eps).sqrt()


# ---------------------------------------------------------------------------
# Metal Profiled Significance (f32 GPU, CPU signal upload)
# ---------------------------------------------------------------------------


class MetalProfiledQ0Loss:
    """Metal GPU profiled q₀ with envelope theorem gradient.

    Unlike CUDA version, signal is uploaded via CPU (no zero-copy).
    All GPU computation in f32; results returned as f64.
    """

    @staticmethod
    def apply(signal_histogram, session):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, signal):
                assert signal.dtype == torch.float64, "signal must be float64"
                signal_cpu = signal.detach().cpu().contiguous()
                session.upload_signal(signal_cpu.numpy().tolist())
                q0, grad_list = session.profiled_q0_and_grad()
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


class MetalProfiledQmuLoss:
    """Metal GPU profiled qμ with envelope theorem gradient."""

    @staticmethod
    def apply(signal_histogram, session, mu_test):
        torch = _require_torch()

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, signal):
                assert signal.dtype == torch.float64, "signal must be float64"
                signal_cpu = signal.detach().cpu().contiguous()
                session.upload_signal(signal_cpu.numpy().tolist())
                qmu, grad_list = session.profiled_qmu_and_grad(mu_test)
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


# ---------------------------------------------------------------------------
# SignificanceLoss — ML-friendly class API (wraps profiled_z0_loss)
# ---------------------------------------------------------------------------


class SignificanceLoss:
    r"""Differentiable discovery significance loss for PyTorch training loops.

    Wraps the profiled :math:`Z_0 = \sqrt{q_0}` computation as a stateful
    loss object with familiar ``__init__`` / ``__call__`` semantics.

    The forward pass runs two GPU L-BFGS-B fits (free + constrained) and
    returns :math:`-Z_0` (negated so that ``loss.backward()`` *maximises*
    significance via standard gradient descent).

    Args:
        model: ``nextstat.HistFactoryModel`` — the statistical model.
        signal_sample_name: name of the signal sample in the workspace
            (default ``"signal"``).
        device: ``"cuda"``, ``"metal"``, or ``"auto"`` (default).
        negate: if ``True`` (default), returns :math:`-Z_0` so that
            minimising the loss maximises significance. Set to ``False``
            to get raw :math:`Z_0`.
        eps: small constant added inside :math:`\sqrt{\cdot}` for
            numerical stability (default ``1e-12``).

    Example::

        from nextstat.torch import SignificanceLoss

        loss_fn = SignificanceLoss(model, "signal")

        for batch in dataloader:
            signal_hist = nn(batch).double().cuda()
            loss = loss_fn(signal_hist)   # -Z₀
            loss.backward()
            optimizer.step()

    ML translation:
        - **Nuisance parameters** → latent variables / systematics
        - **Profile likelihood** → loss with profiled (marginalised) latents
        - **Z₀ (significance)** → metric (higher = better signal/background
          separation); negated here so SGD minimises it

    References:
        - neos (gradhep): https://github.com/gradhep/neos
        - arXiv:2203.05570 — End-to-End-Optimised Summary Statistics
        - arXiv:2508.17802 — Differentiating a HEP Analysis Pipeline
    """

    def __init__(
        self,
        model,
        signal_sample_name: str = "signal",
        *,
        device: str = "auto",
        negate: bool = True,
        eps: float = 1e-12,
    ):
        self._session = create_profiled_session(
            model, signal_sample_name, device=device
        )
        self._negate = negate
        self._eps = eps

    def __call__(self, signal_histogram):
        """Compute :math:`-Z_0` (or :math:`Z_0` if ``negate=False``).

        Args:
            signal_histogram: ``torch.Tensor`` ``[n_signal_bins]``, float64.
                On CUDA for CUDA sessions, any device for Metal sessions.

        Returns:
            Scalar ``torch.Tensor`` with ``grad_fn``.
        """
        z0 = profiled_z0_loss(signal_histogram, self._session, eps=self._eps)
        return -z0 if self._negate else z0

    def q0(self, signal_histogram):
        """Raw :math:`q_0` (not negated, not square-rooted)."""
        return profiled_q0_loss(signal_histogram, self._session)

    def z0(self, signal_histogram):
        r"""Raw :math:`Z_0 = \sqrt{q_0 + \epsilon}`."""
        return profiled_z0_loss(signal_histogram, self._session, eps=self._eps)

    @property
    def n_bins(self) -> int:
        """Number of signal bins expected by the model."""
        return self._session.signal_n_bins()

    @property
    def n_params(self) -> int:
        """Number of model (nuisance) parameters."""
        return self._session.n_params()

    @property
    def session(self):
        """Underlying GPU session (for advanced use)."""
        return self._session


def batch_profiled_q0_loss(signal_histograms, session):
    """Compute profiled q₀ for a batch of signal histograms.

    Each histogram is evaluated independently (no parameter sharing across
    batch elements). Useful for ensemble training or Bayesian NN ensembles.

    Args:
        signal_histograms: ``torch.Tensor`` ``[batch, n_signal_bins]``, float64
        session: profiled session from ``create_profiled_session()``

    Returns:
        ``list[Tensor]`` — one q₀ scalar per batch element (each with ``grad_fn``)
    """
    torch = _require_torch()
    if signal_histograms.ndim == 1:
        return [profiled_q0_loss(signal_histograms, session)]
    if signal_histograms.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D tensor, got shape={tuple(signal_histograms.shape)}"
        )
    return [
        profiled_q0_loss(signal_histograms[i], session)
        for i in range(signal_histograms.shape[0])
    ]


# ---------------------------------------------------------------------------
# SoftHistogram — differentiable binning layer (KDE / sigmoid)
# ---------------------------------------------------------------------------


class SoftHistogram:
    r"""Differentiable histogram via kernel density estimation.

    Converts continuous NN outputs into a soft histogram that can be passed
    to :class:`SignificanceLoss`. Uses Gaussian KDE (default) or sigmoid
    binning to produce smooth, differentiable bin counts.

    This is the "missing link" between a neural network classifier output
    and the statistical inference layer:

    .. code-block:: text

        NN(x) → scores → SoftHistogram → bin counts → SignificanceLoss

    Args:
        bin_edges: 1D tensor of ``n_bins + 1`` bin edges (ascending).
        bandwidth: KDE bandwidth (sigma). Smaller = sharper bins but
            noisier gradients. ``"auto"`` uses Scott's rule on the bin
            width. Default ``"auto"``.
        mode: ``"kde"`` (Gaussian kernel, default) or ``"sigmoid"``
            (sigmoid approximation to hard cuts — faster, less smooth).

    Example::

        soft_hist = SoftHistogram(
            bin_edges=torch.linspace(0.0, 1.0, 11),  # 10 bins
            bandwidth=0.05,
        )
        scores = classifier(batch)          # [N]
        histogram = soft_hist(scores)        # [10], differentiable
        loss = loss_fn(histogram.double())   # → SignificanceLoss
    """

    def __init__(
        self,
        bin_edges,
        bandwidth="auto",
        mode: str = "kde",
    ):
        torch = _require_torch()
        if not isinstance(bin_edges, torch.Tensor):
            bin_edges = torch.as_tensor(bin_edges, dtype=torch.float64)
        if bin_edges.ndim != 1 or bin_edges.shape[0] < 2:
            raise ValueError("bin_edges must be a 1D tensor with at least 2 elements")
        self._edges = bin_edges
        self._centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self._n_bins = len(self._centers)
        widths = bin_edges[1:] - bin_edges[:-1]

        if bandwidth == "auto":
            self._bw = widths.mean().item() * 0.5
        else:
            self._bw = float(bandwidth)

        if mode not in ("kde", "sigmoid"):
            raise ValueError(f"mode must be 'kde' or 'sigmoid', got {mode!r}")
        self._mode = mode

    def __call__(self, scores, weights=None):
        """Bin continuous scores into a soft histogram.

        Args:
            scores: ``torch.Tensor`` ``[N]`` — continuous classifier outputs.
            weights: optional ``torch.Tensor`` ``[N]`` — per-event weights
                (default: uniform weight 1).

        Returns:
            ``torch.Tensor`` ``[n_bins]`` — differentiable bin counts.
        """
        torch = _require_torch()
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1D, got shape={tuple(scores.shape)}")

        edges = self._edges.to(device=scores.device, dtype=scores.dtype)
        centers = self._centers.to(device=scores.device, dtype=scores.dtype)

        if self._mode == "kde":
            return self._kde_hist(scores, centers, weights, torch)
        else:
            return self._sigmoid_hist(scores, edges, weights, torch)

    def _kde_hist(self, scores, centers, weights, torch):
        # scores: [N], centers: [B]
        # diff[i,j] = (scores[i] - centers[j]) / bandwidth
        diff = (scores.unsqueeze(1) - centers.unsqueeze(0)) / self._bw  # [N, B]
        kernels = torch.exp(-0.5 * diff * diff)  # Gaussian kernel
        # Normalise each event across bins so total contribution = 1
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-30)
        if weights is not None:
            kernels = kernels * weights.unsqueeze(1)
        return kernels.sum(dim=0)  # [B]

    def _sigmoid_hist(self, scores, edges, weights, torch):
        # Sigmoid approximation: P(x in bin j) ≈ σ((x-lo)/τ) - σ((x-hi)/τ)
        tau = self._bw
        lo = edges[:-1].unsqueeze(0)  # [1, B]
        hi = edges[1:].unsqueeze(0)   # [1, B]
        x = scores.unsqueeze(1)       # [N, 1]
        probs = torch.sigmoid((x - lo) / tau) - torch.sigmoid((x - hi) / tau)  # [N, B]
        if weights is not None:
            probs = probs * weights.unsqueeze(1)
        return probs.sum(dim=0)  # [B]

    @property
    def n_bins(self) -> int:
        return self._n_bins

    @property
    def bin_edges(self):
        return self._edges

    @property
    def bin_centers(self):
        return self._centers


# ---------------------------------------------------------------------------
# Jacobian export — full ∂L/∂signal as numpy/torch
# ---------------------------------------------------------------------------


def signal_jacobian(signal_histogram, session):
    """Compute the full Jacobian ∂q₀/∂signal as a 1D tensor.

    This extracts the gradient vector without going through ``autograd``,
    useful for external optimisers (SciPy L-BFGS, Optuna surrogate models)
    or for identifying low-impact bins (fast pruning).

    Args:
        signal_histogram: ``torch.Tensor`` ``[n_signal_bins]``, float64.
            On CUDA for CUDA sessions, any device for Metal sessions.
        session: profiled session from ``create_profiled_session()``.

    Returns:
        ``torch.Tensor`` ``[n_signal_bins]`` — gradient ∂q₀/∂signal on
        the same device as ``signal_histogram``.

    Example — fast pruning via Jacobian::

        grad = signal_jacobian(signal_hist, session)
        important_bins = grad.abs() > threshold
        # Bins where |∂q₀/∂s_i| ≈ 0 can be dropped
    """
    torch = _require_torch()
    if _is_metal_session(session):
        signal_cpu = signal_histogram.detach().cpu().contiguous()
        session.upload_signal(signal_cpu.numpy().tolist())
        _q0, grad_list = session.profiled_q0_and_grad()
    else:
        assert signal_histogram.is_cuda, "signal must be on CUDA for CUDA session"
        assert signal_histogram.dtype == torch.float64
        assert signal_histogram.is_contiguous()
        torch.cuda.synchronize()
        _q0, grad_list = session.profiled_q0_and_grad(signal_histogram.data_ptr())
        torch.cuda.synchronize()

    return torch.tensor(
        grad_list, dtype=torch.float64, device=signal_histogram.device
    )


def signal_jacobian_numpy(signal_histogram, session):
    """Like :func:`signal_jacobian` but returns a NumPy array.

    Convenience for users who pipe gradients into SciPy or Optuna.
    """
    import numpy as np  # type: ignore

    grad_tensor = signal_jacobian(signal_histogram, session)
    return grad_tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Array API bridge — accept any __dlpack__-compatible array
# ---------------------------------------------------------------------------


def as_tensor(x):
    """Convert any array-API-compatible object to a ``torch.Tensor``.

    Supports:
    - ``torch.Tensor`` (passthrough)
    - NumPy ``ndarray`` (zero-copy on CPU)
    - Any object with ``__dlpack__`` (JAX, CuPy, Arrow, etc.)
    - Any object with ``__cuda_array_interface__`` (CuPy, Numba)
    - Python lists / tuples

    This is the recommended entry point for users coming from JAX or CuPy
    who want to feed arrays into NextStat's PyTorch layer.

    Args:
        x: array-like object.

    Returns:
        ``torch.Tensor``

    Example::

        import jax.numpy as jnp
        from nextstat.torch import as_tensor, SignificanceLoss

        jax_hist = jnp.array([10.0, 20.0, 30.0])
        torch_hist = as_tensor(jax_hist).double()
        loss = loss_fn(torch_hist)
    """
    torch = _require_torch()

    if isinstance(x, torch.Tensor):
        return x

    # DLPack protocol (JAX, CuPy, Arrow, NumPy 2.x, etc.)
    if hasattr(x, "__dlpack__"):
        return torch.from_dlpack(x)

    # CUDA Array Interface (CuPy, Numba CUDA) — fallback for objects
    # without __dlpack__ (rare; most modern CuPy/Numba support DLPack)
    if hasattr(x, "__cuda_array_interface__"):
        return torch.as_tensor(x)

    # NumPy ndarray (zero-copy)
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except ImportError:
        pass

    # Fallback: list, tuple, scalar
    return torch.as_tensor(x)

