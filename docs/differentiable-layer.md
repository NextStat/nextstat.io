---
title: "Differentiable HistFactory Layer for PyTorch"
description: "Canonical API and correctness contract for NextStat’s differentiable HistFactory likelihood and profiled discovery significance objectives in PyTorch (CUDA zero-copy, Metal fallback)."
status: shipped
last_updated: 2026-02-08
keywords:
  - differentiable inference
  - HistFactory
  - discovery significance
  - Z0
  - q0
  - PyTorch autograd
  - CUDA zero-copy
  - envelope theorem
  - NextStat
---

# Differentiable HistFactory Layer for PyTorch

NextStat provides a differentiable NLL layer that integrates directly into PyTorch training loops. This enables **end-to-end optimization of neural network classifiers on physics significance** with full systematic uncertainty handling.

If you want to train on the actual discovery metric (rather than a surrogate like BCE/MSE), the intended end-to-end pipeline is:

> NN scores → differentiable histogram → profiled likelihood → q₀ → Z₀

Concretely, this is exposed as `SoftHistogram` + `SignificanceLoss` in `nextstat.torch`.

## What is differentiable (and what is not)

- The layer is **differentiable w.r.t. the signal histogram** `s` (a 1D tensor of bin yields produced by your model + `SoftHistogram`).
- For the fixed-parameter objective `nll_loss(...)`, nuisance parameters are treated as constants: the backward pass returns `∂NLL/∂s`.
- For profiled objectives (`q₀`, `qμ`, `Z₀`, `Zμ`), NextStat uses the **envelope theorem** to compute an exact gradient w.r.t. `s` *at the fitted optima*, without backpropagating through optimizer iterations.
- This does **not** (yet) differentiate derived quantities like CLs bands or `μ_up(α)`; those generally require implicit differentiation through root-finding (see “Phase 2 (future)” below).

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

**Key insight (CUDA, Phase 1 NLL)**: No device↔host copies of the *signal histogram* (or its *signal-gradient*) in the training loop. The CUDA kernel reads signal bins directly from PyTorch GPU memory and writes gradients directly back. (There are still small transfers for nuisance parameters and returning the scalar NLL.)

For the profiled objectives (q₀/qμ/Z₀/Zμ), the signal histogram is still read zero-copy via a raw device pointer, but the final ∂/∂signal vector is currently returned to Python as a small host `Vec<f64>` and then materialized as a CUDA tensor (tiny D↔H transfer: O(100–1000) floats).

## Quick Start (end-to-end Z₀ training)

```python
import torch
import nextstat
from nextstat.torch import SoftHistogram, SignificanceLoss

model = nextstat.from_pyhf(workspace_json)

# One-time GPU session init (CUDA preferred, Metal fallback if enabled)
loss_fn = SignificanceLoss(model, signal_sample_name="signal", device="auto")  # returns -Z0 by default

soft_hist = SoftHistogram(bin_edges=torch.linspace(0.0, 1.0, 11), bandwidth="auto", mode="kde")

optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)
for batch in dataloader:
    optimizer.zero_grad()

    scores = nn(batch)                      # [N]
    signal_hist = soft_hist(scores).double()  # [B]
    if torch.cuda.is_available():
        signal_hist = signal_hist.cuda()    # CUDA path expects CUDA float64

    loss = loss_fn(signal_hist)             # -Z0
    loss.backward()
    optimizer.step()
```

## Quick Start (fixed-parameter NLL)

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

## Profiled significance (q₀ / qμ) on GPU (CUDA / Metal)

NextStat also provides a GPU-accelerated **profiled** layer (CUDA or Metal) for the common
profile-likelihood test statistics:

- Discovery: `q₀ = 2·(NLL(μ=0, θ̂₀) − NLL(μ̂, θ̂))`
- Upper limits: `qμ = 2·(NLL(μ_test, θ̂_μ) − NLL(μ̂, θ̂))` with one-sided clipping

These require two profile fits per forward pass; they are more expensive than
`nll_loss` but directly optimize a physics test statistic.

```python
import torch
from nextstat.torch import (
    create_profiled_session,
    profiled_q0_loss,
    profiled_z0_loss,
    profiled_qmu_loss,
)

session = create_profiled_session(model, "signal")
signal = nn(batch).double().cuda().requires_grad_(True)

q0 = profiled_q0_loss(signal, session)     # maximize discovery power
z0 = profiled_z0_loss(signal, session)     # sqrt(q0)
qmu = profiled_qmu_loss(signal, session, mu_test=5.0)  # upper-limit statistic
```

### Gradient formula (envelope theorem)

For profiled objectives of the form `NLL(θ̂(s), s)`, the envelope theorem gives:

```
 d/ds NLL(θ̂(s), s) = ∂/∂s NLL(θ, s) |_{θ = θ̂(s)}
```

So for `q₀` the gradient w.r.t. the signal histogram is:

```
∂q₀/∂s = 2 · ( ∂NLL/∂s |_{θ̂₀} − ∂NLL/∂s |_{θ̂} )
```

**One-sided discovery note**: NextStat implements the standard one-sided discovery convention: if the fitted signal strength satisfies `μ̂ < 0` (or the statistic clamps to `q₀ = 0`), the returned `q₀` and its gradient are zero.

## Phase 1 Gradient Formula

For fixed nuisance parameters, the gradient of NLL w.r.t. signal bin `i` is:

```
expected_i = (signal_i + delta_i) * factor_i + sum_{other samples}
dNLL/d(signal_i) = (1 - obs_i / expected_i) * factor_i
```

Where `factor_i` is the product of all multiplicative modifiers (NormFactor, NormSys, ShapeSys, etc.) applied to the signal sample at bin `i`.

## Practical notes (important for correctness)

### Signal gradient buffer must start at zero (CUDA)

The CUDA kernel accumulates into the gradient buffer with `atomicAdd`. That means the gradient output tensor **must be zeroed** before calling into NextStat. The Python wrapper does this via `torch.zeros_like(signal)`.

### CUDA stream synchronization

PyTorch and NextStat may use different CUDA streams. The current `nextstat.torch` wrappers call `torch.cuda.synchronize()` before and after the native call to ensure that:

- the NextStat kernel sees the fully-written `signal_histogram`, and
- PyTorch sees the fully-written `grad_signal`.

### Multi-channel signal layout

If the signal sample appears in multiple channels, NextStat treats the external signal buffer as a concatenation of the per-channel segments:

`signal = [ch0_bins..., ch1_bins..., ...]`

`session.signal_n_bins()` returns the **total** number of signal bins across all entries.

In PyTorch, the expected shape is therefore a single 1D tensor. For multi-channel setups, build it by concatenating per-channel histograms in the same order as the model’s signal entries:

```python
signal = torch.cat([signal_sr, signal_vr], dim=0).double().cuda()
loss = loss_fn(signal)
```

## Metal (Apple Silicon) support

`create_profiled_session(..., device="auto")` prefers CUDA when available and can fall back to a Metal backend (if NextStat is built with `--features metal`).

Key differences vs CUDA:

- GPU computation is **f32** (Apple GPU precision constraints); inputs/outputs are converted at the API boundary.
- The signal histogram is uploaded from CPU (no raw pointer interop with MPS tensors), so there is no CUDA-style zero-copy path.
- L-BFGS-B tolerance is relaxed (default 1e-3) to match f32 behavior.

## Validation and evidence

This layer is validated with finite-difference checks and integration tests:

- CUDA zero-copy NLL + signal gradient (`∂NLL/∂s`): `tests/python/test_torch_layer.py`
- Profiled `q₀`/`qμ` envelope gradients: `tests/python/test_differentiable_profiled_q0.py`
- Benchmark fixtures report max FD error (example): `docs/benchmarks.md`

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

For **derived profiled metrics** (e.g. interpolated upper limits `μ_up(α)` or full CLs bands),
you generally need implicit differentiation through the solver/root-finding layer.
The simple envelope-theorem gradient is sufficient for `q₀` and `qμ`, but not for
all downstream quantities.

```
dq/ds = dq/ds|_{theta fixed} - (d2NLL/ds dtheta)^T (d2NLL/dtheta^2)^{-1} dq/dtheta|_{s fixed}
```

This requires the cross-Hessian d2NLL/ds/dtheta, which can be computed via finite differences of the GPU gradient w.r.t. signal bins.

## Further reading

- Canonical implementation reference: `bindings/ns-py/python/nextstat/torch.py`
- Scientific deep-dive (problem → solution → derivations): [How NextStat Makes the HistFactory Pipeline Differentiable in PyTorch](/blog/differentiable-layer)
