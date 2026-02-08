<!--
  Blog draft (technical).
  Keep in sync with implementation in:
    - crates/ns-compute/kernels/differentiable_nll_grad.cu
    - crates/ns-compute/src/differentiable.rs
    - crates/ns-inference/src/differentiable.rs
    - bindings/ns-py/python/nextstat/torch.py
-->

# How NextStat Makes the HistFactory Pipeline Differentiable in PyTorch

**Last updated:** 2026-02-08  
**Status:** Blog draft (technical)

## Abstract

High‑energy physics (HEP) analyses rarely care about classifier accuracy per se. What ultimately matters is an inference metric such as *discovery significance* $Z_0$, computed from a profile likelihood ratio. The standard pipeline

> NN scores → histogram → HistFactory likelihood → profiled fit → $q_0$ → $Z_0$

is traditionally **non‑differentiable**: histogram binning is discrete, and the profiled fit introduces an $\operatorname{argmin}$ that breaks straightforward backpropagation. NextStat makes this pipeline differentiable end‑to‑end by combining:

1. a differentiable histogram layer in Python (`SoftHistogram`),
2. a fused GPU kernel that evaluates HistFactory NLL and **analytical gradients**,
3. a Rust session that keeps the model resident on GPU, and
4. an envelope‑theorem gradient that avoids differentiating through the optimizer.

The result is a PyTorch‑native loss: you can train a network directly to **maximize $Z_0$** (or minimize $-Z_0$) with standard optimizers.

---

## 1. The problem: the metric lives *after* inference

In many HEP workflows, a neural network produces a one‑dimensional score per event, and that score is binned into a histogram. The histogram then becomes the sufficient statistic for the downstream binned likelihood model (HistFactory/pyhf style).

The discovery test statistic is typically defined as:

$$
q_0 \;=\; 2\left[\mathrm{NLL}(\hat\theta_0,\mu=0) - \mathrm{NLL}(\hat\theta,\hat\mu)\right]_+,
\qquad
Z_0 \;=\; \sqrt{q_0},
$$

where:

- $\mu$ is the parameter of interest (POI, “signal strength”),
- $\theta$ are nuisance parameters (systematics, MC stat, constraints),
- hats denote maximum likelihood estimates (MLEs),
- $[\cdot]_+$ indicates one‑sided discovery clipping.

This is the objective physics cares about. But the *computational graph* is long and historically breaks differentiability in two places:

1. **Histogramming**: hard bin edges make gradients zero almost everywhere.
2. **Profiling**: $\hat\theta(s)=\arg\min_\theta \mathrm{NLL}(\theta; s)$ is an optimization subproblem inside the forward pass.

NextStat’s differentiable layer addresses both.

---

## 2. Architecture at a glance (four layers)

Below is the end‑to‑end stack that turns a network output into a differentiable $Z_0$ objective:

```
Python (PyTorch)                              Rust (NextStat)                         GPU
──────────────────────────────────────────     ───────────────────────────────────     ───────────────────────────────
SoftHistogram(scores)                          ProfiledDifferentiableSession            differentiable_nll_grad.cu/msl
  scores → soft bin counts                  →   2× profiled L‑BFGS‑B fits (CPU)      →  Fused Poisson NLL + gradients
                                               (NLL+∂NLL/∂params evaluated on GPU)      reads signal from external ptr

SignificanceLoss(model)                        Envelope theorem gradient                 writes ∂NLL/∂signal
  calls profiled_z0_loss()                  →   ∂q0/∂s = 2(∂NLL/∂s|0 − ∂NLL/∂s|free)     (zero‑copy for Phase 1 NLL)

torch.autograd.Function                        PyO3 bridge                              (CUDA) no signal memcpy
  backward: chain rule                      →   u64 device pointers in/out              (Metal) CPU upload (f32)
```

Implementation anchors:

- **CUDA kernel:** `crates/ns-compute/kernels/differentiable_nll_grad.cu`
- **Rust CUDA session:** `crates/ns-compute/src/differentiable.rs`
- **Profiled objective + envelope gradient:** `crates/ns-inference/src/differentiable.rs`
- **PyTorch autograd wrappers, SoftHistogram, SignificanceLoss:** `bindings/ns-py/python/nextstat/torch.py`

### 2.1 What actually happens when you call `loss.backward()`

It helps to make the cross‑language control flow explicit.

**Phase 1 (fixed‑parameter NLL, zero‑copy signal + gradient):**

1. Python: `nextstat.torch.nll_loss(...)` → `NextStatNLL.apply(...)`
2. PyTorch autograd: `_Fn.forward(...)` allocates `grad_signal` and calls the native session
3. PyO3: `nextstat._core.DifferentiableSession.nll_grad_signal(...)` (exposed in `bindings/ns-py/src/lib.rs`)
4. Rust inference: `ns_inference::DifferentiableSession::nll_grad_signal(...)`
5. Rust compute: `ns_compute::DifferentiableAccelerator::nll_grad_wrt_signal(...)`
6. CUDA: launch `differentiable_nll_grad` from `crates/ns-compute/kernels/differentiable_nll_grad.cu`

**Phase 2 (profiled $q_0$ / $q_\mu$, envelope gradient):**

1. Python: `nextstat.torch.profiled_q0_loss(...)` → `ProfiledQ0Loss.apply(...)`
2. PyO3: `nextstat._core.ProfiledDifferentiableSession.profiled_q0_and_grad(...)`
3. Rust inference: `ProfiledDifferentiableSession::profiled_q0_and_grad(...)`
4. Two profiled fits (CPU `LbfgsState`), where each iteration calls:
   `DifferentiableAccelerator::nll_and_grad_params(...)` → one fused GPU kernel launch
5. Two signal‑gradient evaluations at the optima:
   `DifferentiableAccelerator::nll_grad_all(...)` → fused GPU kernel + download of small `grad_signal`
6. Python builds a CUDA tensor from the returned gradient and autograd applies the chain rule.

---

## 3. Layer 1 — SoftHistogram: making binning differentiable

HistFactory consumes **binned yields**, but a neural network emits **continuous scores**. A hard histogram

$$
h_j = \sum_{k=1}^N \mathbf{1}\{e_j \le x_k < e_{j+1}\}
$$

is not differentiable w.r.t. $x_k$.

NextStat’s `SoftHistogram` replaces the indicator function with a smooth approximation. Two modes are implemented:

### 3.1 Gaussian KDE mode (default)

Each event contributes softly to all bins via a Gaussian kernel centered at the bin center:

$$
w_{k j} \propto \exp\left(-\frac{1}{2}\left(\frac{x_k - c_j}{\sigma}\right)^2\right),
\qquad
h_j = \sum_k \tilde w_{k j},
$$

with per‑event normalization $\sum_j \tilde w_{k j}=1$ to keep the total contribution stable.

### 3.2 Sigmoid‑edge mode (faster)

Approximate bin membership as a difference of two sigmoids:

$$
\Pr(x \in [e_j,e_{j+1})) \approx \sigma\!\left(\frac{x-e_j}{\tau}\right)-\sigma\!\left(\frac{x-e_{j+1}}{\tau}\right).
$$

This is cheaper and still differentiable, but typically less smooth than KDE.

Both modes produce a differentiable tensor of bin counts, which becomes the “signal yield vector” $s$ in the inference layer.

---

## 4. Layer 2 — the fused GPU kernel: NLL + analytical gradients

The core performance and differentiability primitive is a fused kernel that evaluates:

1. the (constrained) Poisson negative log‑likelihood, and
2. analytical gradients w.r.t.
   - nuisance parameters $\theta$ (for L‑BFGS‑B), and
   - signal yields $s$ (for backpropagation to the network).

### 4.1 HistFactory expected yields and NLL

For each main bin $i$:

$$
\nu_i(\theta; s) = \sum_{\text{samples }a}\bigl(\text{nom}_{a,i} + \Delta_{a,i}(\theta)\bigr)\cdot F_{a,i}(\theta),
$$

where:

- $F_{a,i}$ is the product of multiplicative modifiers (NormFactor/Lumi/NormSys/ShapeSys/ShapeFactor/StatError),
- $\Delta_{a,i}$ is an additive term (HistoSys),
- constraints (aux Poisson, Gaussian) add extra NLL terms and gradients.

The kernel evaluates the Poisson NLL in the numerically stable form:

$$
\mathrm{NLL}_i = \nu_i + \mathbb{1}[n_i>0]\left(\ln(n_i!) - n_i\ln \nu_i\right),
$$

with a small clamp $\nu_i \leftarrow \max(\nu_i, 10^{-10})$ to avoid $\log 0$.

### 4.2 The key gradient identity

Let $w_i = \partial \mathrm{NLL}_i / \partial \nu_i = 1 - n_i/\nu_i$. For the signal yield $s_i$ (nominal of the signal sample in bin $i$),

$$
\frac{\partial \mathrm{NLL}}{\partial s_i}
= \frac{\partial \mathrm{NLL}}{\partial \nu_i}\cdot\frac{\partial \nu_i}{\partial s_i}
= \left(1 - \frac{n_i}{\nu_i}\right)\cdot F_{\text{signal},i}.
$$

This is exactly what the kernel writes into the signal gradient buffer.

### 4.3 Zero‑copy for PyTorch CUDA tensors (Phase 1)

For the fixed‑parameter differentiable NLL layer, PyTorch owns the signal tensor on GPU. NextStat never copies it:

- Python passes `signal.data_ptr()` (a raw CUDA device address) to Rust as a `u64`.
- Rust passes that pointer into the kernel as `g_external_signal`.
- The kernel reads `g_external_signal[...]` directly.
- The kernel writes `∂NLL/∂signal` via `atomicAdd` into an *external* gradient buffer whose pointer comes from `grad_signal.data_ptr()`.

The important nuance is *what* is “zero‑copy”: the large, per‑bin signal vector stays entirely on device. (There are still small H↔D transfers for the nuisance parameter vector and the returned scalar NLL, which are tiny compared to moving the histogram.)

Concretely, the CUDA kernel’s inner loop computes the scalar weight $w_i$ and performs an atomic add into the *external* gradient buffer:

```c
double w = 1.0 - obs / expected_bin;
atomicAdd(&g_grad_signal_out[sig_local_bin], w * signal_factor);
```

### 4.4 Systematic modifiers covered

The fused kernel implements all 7 HistFactory modifier types used in NextStat’s GPU contract:

- `NormFactor`, `Lumi` (global multiplicative)
- `NormSys` (code‑4 polynomial/exponential interpolation)
- `HistoSys` (per‑bin additive delta, code‑4p)
- `ShapeSys`, `ShapeFactor`, `StatError` (per‑bin multiplicative, parameter‑indexed)

and includes gradients for:

- auxiliary Poisson constraints (Barlow–Beeston‑style $\gamma$ parameters),
- Gaussian constraints for constrained nuisance parameters.

---

## 5. Layer 3 — Rust GPU sessions: keep the model on device

The CUDA kernel is only useful if we can call it repeatedly (hundreds of times per forward pass for profiled objectives) without paying setup costs.

NextStat does this by serializing a HistFactory model once into flat GPU buffers:

- nominal sample yields,
- sample descriptors (bin ranges, offsets),
- modifier descriptors + offset tables,
- modifier data (e.g. NormSys coefficients, HistoSys deltas),
- constraint tables.

### 5.1 `DifferentiableAccelerator` (CUDA)

`crates/ns-compute/src/differentiable.rs` implements `DifferentiableAccelerator`:

- loads the PTX for `differentiable_nll_grad` (compiled at build time),
- uploads static model buffers once,
- uploads observed data once (including precomputed $\ln(n!)$),
- allocates reusable device buffers for parameters and gradients,
- launches the kernel with a single block and shared memory sized as:
  `params[n_params] + scratch[block_size]`.

The build script compiles the differentiable kernel *without* `--use_fast_math` to avoid introducing gradient noise that can harm neural network training stability (`crates/ns-compute/build.rs`).

### 5.2 Multi‑channel signal support

In real analyses, the “signal sample” often appears in multiple channels (SR/CR/VR). NextStat represents this as a list of `(sample_idx, first_bin, n_bins)` entries.

The external signal tensor is laid out as a concatenation of the per‑channel segments:

$$
s = [s^{(0)}\;||\;s^{(1)}\;||\;\cdots].
$$

The kernel uses these entries to map a main‑bin index $i$ to the correct offset inside `g_external_signal`, and writes gradients into the matching location in the external gradient buffer.

---

## 6. Layer 4 — profiled significance + envelope theorem (no “differentiate through argmin”)

Profiling introduces a nested optimization problem. Naively, one might unroll the optimizer iterations and backpropagate through them. That is expensive, memory‑heavy, and yields gradients that depend on the chosen optimizer and stopping criteria.

NextStat takes the classical alternative: **envelope theorem**.

### 6.1 Discovery statistic as difference of two profiled optima

NextStat’s `ProfiledDifferentiableSession` computes:

- the free (unconditional) optimum $(\hat\mu,\hat\theta)$, and
- the constrained optimum at $\mu=0$, denoted $\hat\theta_0$.

Both are found with **bounded L‑BFGS‑B on CPU**, where each function/gradient evaluation is a single GPU kernel launch returning:

- NLL value, and
- $\partial\mathrm{NLL}/\partial\theta$ (needed for the optimizer).

Warm‑starting the constrained fit from the free fit makes this practical.

### 6.2 Envelope gradient for $q_0$

For an optimum value function

$$
V(s) = \min_\theta f(\theta, s),
$$

the envelope theorem states (under standard regularity/KKT conditions):

$$
\frac{dV}{ds} = \left.\frac{\partial f}{\partial s}\right|_{\theta=\hat\theta(s)}.
$$

Intuitively: at the optimum, $\partial f / \partial \theta = 0$, so the implicit dependence $\theta(s)$ does not contribute to the total derivative.

Applying this to both terms in $q_0$ yields a remarkably simple exact gradient:

$$
\frac{\partial q_0}{\partial s}
=
2\left(
\left.\frac{\partial \mathrm{NLL}}{\partial s}\right|_{\mu=0,\;\theta=\hat\theta_0}
\;-\;
\left.\frac{\partial \mathrm{NLL}}{\partial s}\right|_{\mu=\hat\mu,\;\theta=\hat\theta}
\right).
$$

In code (`crates/ns-inference/src/differentiable.rs`), this is implemented by:

1. running two profile fits (`profile_fit`),
2. evaluating `∂NLL/∂signal` at each optimum via the same fused kernel, and
3. taking the difference.

This gradient is **exact at convergence**. If the optimizer fails to converge, NextStat returns an error rather than silently producing an incorrect gradient.

> Note: today the profiled layer returns the signal gradient to Python as a small `Vec<f64>` and then constructs a PyTorch tensor. This introduces tiny device↔host transfers (hundreds of floats), while still avoiding any device↔host movement of the full signal histogram itself.

---

## 7. PyTorch autograd: wiring gradients into backprop

On the Python side, NextStat exposes these objectives as `torch.autograd.Function` wrappers.

### 7.1 Fixed‑parameter differentiable NLL (Phase 1)

The forward pass:

1. allocates a zeroed `grad_signal` tensor on CUDA (required because the kernel uses `atomicAdd`),
2. synchronizes CUDA streams,
3. calls the native session with raw device pointers,
4. synchronizes again, and
5. stores `grad_signal` for the backward pass.

Backward is just the chain rule:

$$
\frac{\partial L}{\partial s} = \frac{\partial L}{\partial \mathrm{NLL}}\cdot\frac{\partial \mathrm{NLL}}{\partial s}.
$$

### 7.2 Profiled $q_0$, $Z_0$, and the `SignificanceLoss` convenience API

The blog‑friendly API is:

```python
from nextstat.torch import SoftHistogram, SignificanceLoss

loss_fn = SignificanceLoss(model, signal_sample_name="signal")  # loads GPU session once
soft_hist = SoftHistogram(bin_edges, bandwidth="auto", mode="kde")

scores = classifier(batch)                 # [N]
signal_hist = soft_hist(scores).double()   # [B] (float64 expected by CUDA)

loss = loss_fn(signal_hist)               # returns -Z0 by default
loss.backward()                           # gradients flow to NN weights
optimizer.step()
```

`SignificanceLoss` simply wraps `profiled_z0_loss` and negates it so you can maximize $Z_0$ with standard gradient descent.

### 7.3 Why the explicit `torch.cuda.synchronize()` calls?

PyTorch and NextStat launch kernels on different CUDA streams. Without a barrier, the NextStat kernel might read a signal tensor that PyTorch has not finished writing, or PyTorch might read a gradient tensor that NextStat has not finished writing.

The current implementation uses full `torch.cuda.synchronize()` before and after the native call for correctness and simplicity. A future optimization is to share or import the active PyTorch stream to replace global synchronizations with stream‑local event waits.

---

## 8. Metal backend (Apple Silicon): same algorithm, different constraints

Apple GPUs do not provide hardware `f64` arithmetic in the same way NVIDIA GPUs do, so NextStat’s Metal backend:

- runs the fused kernel in **f32** (`crates/ns-compute/kernels/differentiable_nll_grad.metal`),
- converts inputs/outputs at the API boundary (PyTorch tensors remain f64),
- relaxes L‑BFGS‑B convergence tolerance to **1e‑3** (vs 1e‑5 on CUDA),
- uploads the signal histogram via CPU (no raw pointer interop with MPS tensors).

Empirically, NLL parity vs the CPU f64 reference mode is at the ~1e‑6 relative level for typical workspaces (see `CHANGELOG.md`).

---

## 9. Validation and performance notes

Two things must be true for a differentiable inference layer to be trustworthy:

1. **Numerical correctness** (parity of NLL and fits), and
2. **Gradient correctness** (analytical gradients match finite differences).

NextStat validates gradients with finite‑difference checks; for the differentiable NLL layer, the maximum absolute error vs finite differences is reported as **2.07e‑9** on the benchmark fixture (`docs/benchmarks.md`).

In terms of runtime, the differentiable layer is designed so that the expensive computation stays on the GPU, while CPU work is limited to orchestrating the profiled fit (L‑BFGS‑B state updates).

---

## 10. Limitations and future work

- **Derived inference quantities** (e.g. $\mu_\text{up}$, CLs bands) generally require differentiating through a root‑finder and/or implicit differentiation through the profiled solution. The envelope theorem is sufficient for $q_0$ and $q_\mu$, but not for all downstream statistics.
- **CUDA stream interop** can reduce synchronization overhead.
- **More aggressive kernel fusion** is possible (e.g. compute both profiled gradients inside Rust and write directly into a PyTorch grad buffer), though today’s host transfer is already tiny.

---

## Conclusion

NextStat turns a historically non‑differentiable HEP inference pipeline into a PyTorch‑friendly loss function by:

- replacing hard binning with a differentiable histogram (`SoftHistogram`),
- implementing a fused GPU kernel for HistFactory NLL + analytical gradients,
- keeping serialized model state resident on device via Rust sessions, and
- using the envelope theorem to obtain exact gradients for profiled test statistics without differentiating through the optimizer.

This enables training neural networks *directly on physics significance*—optimizing what the analysis ultimately cares about, not a surrogate proxy.
