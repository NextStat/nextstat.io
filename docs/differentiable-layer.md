# Differentiable HistFactory Layer for PyTorch

NextStat provides a differentiable NLL layer that integrates directly into PyTorch training loops. This enables **end-to-end optimization of neural network classifiers on physics significance** with full systematic uncertainty handling.

## Architecture

```
PyTorch Training Loop (GPU)
    |
    +-- Neural Network --> signal_histogram (torch.Tensor, CUDA)
    |
    +-- NextStatNLL.forward(signal_histogram)
    |   +-- tensor.data_ptr() --> raw CUDA device pointer (u64)
    |   +-- Rust (PyO3): passes pointer to CUDA kernel
    |   +-- Kernel: reads signal bins from PyTorch memory (ZERO-COPY)
    |   +-- Kernel: writes dNLL/d(signal) into PyTorch grad tensor (ZERO-COPY)
    |   +-- Return: NLL scalar
    |
    +-- NextStatNLL.backward(grad_output)
        +-- grad_signal = cached from forward (already on CUDA)
        +-- return grad_output * grad_signal --> flows back to NN
```

**Key insight**: No Host-to-Device or Device-to-Host copies in the training loop. The CUDA kernel reads signal bins directly from PyTorch's GPU memory and writes gradients directly back.

## Quick Start

```python
import torch
import nextstat
from nextstat.torch import create_session, nll_loss

# Load workspace
model = nextstat.from_pyhf(workspace_json)

# Create GPU session (one-time cost)
session = create_session(model, signal_sample_name="signal")

# Training loop
optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)
for batch in dataloader:
    optimizer.zero_grad()

    # Neural network outputs signal histogram (on CUDA, float64)
    signal_hist = nn(batch).double().cuda()

    # Differentiable NLL with full systematics
    loss = nll_loss(signal_hist, session)

    # Gradient flows through NextStat back to NN
    loss.backward()
    optimizer.step()
```

## API Reference

### `nextstat.torch.create_session(model, signal_sample_name="signal")`

Create a differentiable GPU session for a HistFactory model.

**Args:**
- `model` (`nextstat.HistFactoryModel`): Model from `from_pyhf()` or `from_histfactory_xml()`
- `signal_sample_name` (`str`): Name of the signal sample in the workspace

**Returns:** `DifferentiableSession` — opaque handle holding GPU state

**Raises:**
- `ImportError`: if CUDA support not compiled in
- `ValueError`: if signal sample name not found

### `nextstat.torch.nll_loss(signal_histogram, session, params=None)`

Compute differentiable NLL loss.

**Args:**
- `signal_histogram` (`torch.Tensor`): Shape `[n_signal_bins]`, CUDA, float64
- `session` (`DifferentiableSession`): From `create_session()`
- `params` (`torch.Tensor`, optional): Nuisance parameters. Default: model init values.

**Returns:** NLL scalar (`torch.Tensor` with `grad_fn`)

### `nextstat.torch.NextStatNLL`

Low-level `torch.autograd.Function`. Use `nll_loss()` for the convenience API.

```python
nll = NextStatNLL.apply(signal_histogram, session, params_tensor)
```

### `DifferentiableSession` (native)

Available as `nextstat._core.DifferentiableSession` when built with CUDA.

- `DifferentiableSession(model, signal_sample_name)` — constructor
- `.nll_grad_signal(params, signal_ptr, grad_signal_ptr)` — raw kernel call
- `.signal_n_bins()` — number of signal bins
- `.n_params()` — number of model parameters
- `.parameter_init()` — default parameter values

## Phase 1 Gradient Formula

For fixed nuisance parameters, the gradient of NLL w.r.t. signal bin `i` is:

```
expected_i = (signal_i + delta_i) * factor_i + sum_{other samples}
dNLL/d(signal_i) = (1 - obs_i / expected_i) * factor_i
```

Where `factor_i` is the product of all multiplicative modifiers (NormFactor, NormSys, ShapeSys, etc.) applied to the signal sample at bin `i`.

## Competitive Landscape (February 2026)

| Project | Status | Limitations |
|---------|--------|-------------|
| [pyhf #882](https://github.com/scikit-hep/pyhf/issues/882) | Open, not implemented | Pure Python, SLSQP not differentiable |
| [neos](https://github.com/gradhep/neos) ([arXiv:2203.05570](https://arxiv.org/abs/2203.05570)) | PoC 2022, no updates | Slow, no GPU batch, simple models only |
| [gradhep/relaxed](https://github.com/gradhep/relaxed) | Utility library | Not a pipeline, only soft histogram ops |
| [arXiv:2508.17802](https://arxiv.org/abs/2508.17802) | Scikit-HEP+JAX, 2025 | JAX-only (no PyTorch), JIT tracer leaks |
| [cabinetry](https://iris-hep.org/projects/cabinetry.html) | AD planned | Not implemented |
| **NextStat** | **Production** | **First PyTorch-native, CUDA zero-copy** |

### NextStat advantages

1. **Native PyTorch integration** — `torch.autograd.Function`, works with any optimizer
2. **CUDA zero-copy** — no data movement between PyTorch and NextStat GPU memory
3. **Full HistFactory** — all modifier types (NormSys, HistoSys, ShapeSys, StatError, etc.)
4. **37-880x faster** than pyhf on profile scans (CPU), even faster on GPU
5. **Analytical gradients** — no finite differences, no AD overhead for signal gradient

### Related work

- [Differentiable Histogram (Hard-Binning)](https://arxiv.org/abs/2012.06311)
- [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606)

## Architecture Decisions

### Why a separate CUDA kernel?

The existing `batch_nll_grad.cu` kernel is optimized for batch toy fitting (1 block = 1 toy). The differentiable kernel has different requirements:

1. Single model evaluation (1 block), not batch
2. External signal pointer (reads from PyTorch memory)
3. Signal gradient output (writes to PyTorch memory)
4. Different argument signature

Keeping them separate avoids branching in the hot path and simplifies maintenance.

### Why zero-copy?

Traditional approach: PyTorch GPU → CPU → Rust → GPU → CPU → PyTorch GPU. This adds 4 PCIe transfers per forward pass.

Zero-copy approach: PyTorch GPU ← CUDA kernel → PyTorch GPU. The kernel reads signal bins and writes gradients directly in GPU memory. The only H→D transfer is the small nuisance parameter vector (~250 doubles = 2KB).

### Phase 2 (future): Implicit differentiation

For profiled significance (where nuisance parameters are optimized), the gradient requires implicit differentiation:

```
dq/ds = dq/ds|_{theta fixed} - (d2NLL/ds dtheta)^T (d2NLL/dtheta^2)^{-1} dq/dtheta|_{s fixed}
```

This requires the cross-Hessian d2NLL/ds/dtheta, which can be computed via finite differences of the GPU gradient w.r.t. signal bins.
