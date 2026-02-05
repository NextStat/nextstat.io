# NextStat: High-Performance Statistical Inference for Binned Likelihood Models

**Version:** 0.2.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

We present NextStat, an open-source statistical inference toolkit implemented in Rust with Python bindings. NextStat provides a high-performance engine for binned likelihood analyses based on the HistFactory probability model, supporting maximum likelihood estimation (MLE), profile likelihood scans, asymptotic CLs hypothesis tests, and Bayesian posterior sampling via the No-U-Turn Sampler (NUTS). The toolkit ingests the pyhf JSON workspace format and maintains strict numerical parity with pyhf in a deterministic reference mode, while exploiting SIMD vectorization and thread-level parallelism for performance-critical workflows. We describe the software architecture, inference algorithms, automatic differentiation system, validation methodology, and performance characteristics.

---

## 1. Introduction

Binned likelihood models are a workhorse of statistical inference in the physical sciences. In high-energy physics (HEP), the HistFactory specification [1] provides a widely used template for such models, combining Poisson counting statistics with systematic uncertainty modifiers. Binned likelihoods also arise in astrophysics, nuclear physics, and other domains where histogrammed data is compared against parameterized predictions.

The pyhf project [2] provides a pure-Python implementation of HistFactory likelihoods and defines a JSON schema for declarative model specification. This format has been adopted for publishing full statistical likelihoods on public repositories [3], enabling independent reinterpretation studies. Other tools in this space include the original RooFit/RooStats framework [4, 5] in C++/ROOT, and stanhf [6] which transpiles HistFactory models into the Stan probabilistic programming language [7].

While these tools serve the community well, certain workflows remain computationally demanding:

- **Toy ensembles** for bias, pull, and coverage studies require $O(10^3$–$10^5)$ repeated fits.
- **Profile likelihood scans** evaluate the likelihood at $O(10^2)$ grid points, each requiring a constrained minimization.
- **Upper limit searches** combine profile scans with root-finding over a signal strength parameter.
- **Bayesian posterior sampling** via MCMC requires $O(10^4$–$10^6)$ gradient evaluations.

NextStat addresses these by implementing inference algorithms in Rust, leveraging compile-time optimization, SIMD vectorization, and thread-level parallelism. The toolkit exposes three interfaces: a Rust library, a Python package (via PyO3), and a command-line interface (CLI). All three share a common numerical core and produce identical results.

The companion model specification [8] documents the mathematical details. This paper focuses on the software itself: architecture (Section 2), quickstart usage (Section 3), inference algorithms (Section 4), automatic differentiation (Section 5), performance (Section 6), validation (Section 7), and outlook (Section 8).

## 2. Software Architecture

### 2.1 Crate Structure

NextStat is organized as a Rust workspace (edition 2024, MSRV 1.93) with eight crates following a layered architecture:

```
User Layer        ns-cli          ns-py (PyO3)       ns-viz
                     \               |                /
Inference Layer       \      ns-inference            /
                       \        /        \          /
Model Layer         ns-translate         ns-ad
                         |                 |
Compute Layer        ns-compute          /
                         \              /
Core Layer                ns-core
```

| Crate | Lines | Purpose |
|-------|-------|---------|
| `ns-core` | 197 | Shared traits (`LogDensityModel`, `PreparedNll`), types, error handling |
| `ns-compute` | 524 | SIMD-accelerated Poisson NLL kernels; CPU/Metal/CUDA dispatch |
| `ns-ad` | 998 | Forward-mode (Dual) and reverse-mode (Tape) automatic differentiation |
| `ns-translate` | 1,000+ | pyhf JSON parsing, `HistFactoryModel` construction, modifier algebra |
| `ns-inference` | 2,958 | L-BFGS-B optimization, profile likelihood, CLs, NUTS/HMC sampling |
| `ns-viz` | — | Plot-friendly JSON artifacts (CLs curves, profile scans) |
| `ns-cli` | 513 | Command-line interface via `clap` |
| `ns-py` | 600+ | Python bindings via PyO3 and maturin |

Key external dependencies: `argmin` (L-BFGS-B optimizer), `ndarray`/`nalgebra` (linear algebra), `wide` (4-wide SIMD), `rayon` (data parallelism), `statrs` (distributions), `rand`/`rand_distr` (RNG).

### 2.2 Core Traits

The `LogDensityModel` trait abstracts over concrete model implementations, enabling inference algorithms to work with any model type:

```rust
pub trait LogDensityModel: Send + Sync {
    type Prepared<'a>: PreparedNll + 'a;

    fn dim(&self) -> usize;
    fn parameter_names(&self) -> Vec<String>;
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
    fn parameter_init(&self) -> Vec<f64>;
    fn nll(&self, params: &[f64]) -> Result<f64>;
    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>>;
    fn prepared(&self) -> Self::Prepared<'_>;
}
```

The `prepared()` method returns a cached evaluator (`PreparedNll`) that amortizes setup cost over thousands of evaluations — critical for profile scans, toy ensembles, and MCMC chains.

### 2.3 PreparedModel

The `PreparedModel` struct caches precomputed quantities for the HistFactory likelihood:

| Cached quantity | Purpose |
|----------------|---------|
| Flattened observed data | Single contiguous array across all channels |
| $\ln\Gamma(n+1)$ terms | Log-factorials for Poisson NLL |
| Observation mask | Sparse bin optimization ($n = 0$ bins) |
| Constraint constant | $\sum_j [\ln\sigma_j + \frac{1}{2}\ln 2\pi]$ for Gaussian constraints |
| Code 4 coefficients | 6th-order polynomial interpolation for NormSys |
| Barlow–Beeston $\tau$ values | Effective sample sizes for ShapeSys |

### 2.4 Design Principles

**Trait-based abstraction.** Inference algorithms depend on `LogDensityModel`, not on `HistFactoryModel` directly. Adding GPU backends or alternative model formats does not require changes to inference code.

**Deterministic reference path.** A single-threaded CPU path with stable summation order produces bit-reproducible results. All parity tests against pyhf execute in this mode.

**Separation of concerns.** Model construction (`ns-translate`) is cleanly separated from inference (`ns-inference`), which in turn is separated from AD (`ns-ad`) and compute kernels (`ns-compute`).

**Generic scalar type.** The NLL function `nll_generic<T: Scalar>` works with `f64` (plain evaluation), `Dual` (forward-mode AD), and `Tape` variables (reverse-mode AD) without code duplication.

## 3. Quickstart

### 3.1 Installation

**Python** (via PyPI):
```bash
pip install nextstat
```

**Rust** (add to `Cargo.toml`):
```toml
[dependencies]
ns-inference = { git = "https://github.com/nextstat/nextstat" }
ns-translate = { git = "https://github.com/nextstat/nextstat" }
```

**CLI** (from source):
```bash
git clone https://github.com/nextstat/nextstat
cd nextstat
cargo install --path crates/ns-cli
```

### 3.2 Basic Workflow

Given a JSON workspace file (e.g., downloaded from a public repository or created manually):

```bash
# MLE fit — best-fit parameters and uncertainties
nextstat fit --input workspace.json

# Asymptotic CLs hypothesis test at mu=1
nextstat hypotest --input workspace.json --mu 1.0 --expected-set

# Upper limit (95% CL)
nextstat upper-limit --input workspace.json --alpha 0.05 \
    --scan-start 0.0 --scan-stop 5.0 --scan-points 201

# Profile likelihood scan
nextstat scan --input workspace.json --start 0.0 --stop 2.0 --points 21

# Plot-ready CLs curve (Brazil band)
nextstat viz cls --input workspace.json --alpha 0.05 \
    --scan-start 0.0 --scan-stop 5.0 --scan-points 201
```

All commands accept `--threads N` to control parallelism (N=1 for deterministic mode).

### 3.3 Python API

```python
import json, nextstat
from pathlib import Path

# Load workspace
ws = json.loads(Path("workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(ws))

# Inspect model
print(f"Parameters: {model.parameter_names()}")
print(f"Dimensions: {model.n_params()}")
print(f"POI index:  {model.poi_index()}")

# MLE fit
fit = nextstat.fit(model)
print(f"Best-fit:       {fit.bestfit}")
print(f"Uncertainties:  {fit.uncertainties}")
print(f"NLL:            {fit.nll}")
print(f"Converged:      {fit.converged}")

# Hypothesis test
result = nextstat.infer.hypotest(1.0, model)
print(f"CLs(mu=1): {result['cls']}")

# Upper limit
limit = nextstat.infer.upper_limit(model, alpha=0.05)
print(f"mu_up: {limit['mu_up']}")

# Profile scan
scan = nextstat.infer.profile_scan(model, mu_values=[0.0, 0.5, 1.0, 1.5, 2.0])

# Bayesian posterior sampling (NUTS)
chains = nextstat.sample_nuts(model, n_chains=4,
                               n_warmup=500, n_samples=1000, seed=42)
```

### 3.4 Rust Library

```rust
use ns_translate::pyhf::Workspace;
use ns_inference::mle::MaximumLikelihoodEstimator;

let json = std::fs::read_to_string("workspace.json")?;
let workspace = Workspace::from_json(&json)?;
let model = workspace.to_model()?;

let mle = MaximumLikelihoodEstimator::default();
let result = mle.fit(&model)?;

println!("Best-fit:  {:?}", result.parameters);
println!("NLL:       {}", result.nll);
println!("Converged: {}", result.converged);
```

## 4. Inference Algorithms

### 4.1 Maximum Likelihood Estimation

NextStat performs MLE via the L-BFGS-B algorithm (through the `argmin` crate [9]), which supports box constraints on parameters. The optimizer minimizes the twice negative log-likelihood $2\,\text{nll}(\boldsymbol{\theta})$ subject to parameter bounds.

**Optimizer configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 1000 | Maximum iterations |
| `tol` | $10^{-6}$ | Gradient norm tolerance |
| `m` | 10 | L-BFGS history depth (corrections to approximate inverse Hessian) |

The fit output includes best-fit parameters $\hat{\boldsymbol{\theta}}$, NLL at minimum, Hessian-based covariance $\hat{\Sigma} = H^{-1}$ (via Cholesky or LU decomposition of the numerical Hessian), parameter uncertainties $\hat{\sigma}_i = \sqrt{\hat{\Sigma}_{ii}}$, convergence flag, and evaluation count.

**Gradients** are computed via forward-mode AD (see Section 5). A numerical gradient fallback using central differences with adaptive step size is available.

**Fast path.** The `fit_minimum()` variant skips Hessian computation, returning only the best-fit point and NLL. This is used internally by profile scans and toy fits where uncertainties are not needed.

### 4.2 Profile Likelihood Scan

For a parameter of interest $\mu$, the profile likelihood ratio test statistic is:

$$q_\mu = 2 \left[ \text{nll}(\mu, \hat{\hat{\boldsymbol{\theta}}}) - \text{nll}(\hat{\mu}, \hat{\boldsymbol{\theta}}) \right]$$

where $(\hat{\mu}, \hat{\boldsymbol{\theta}})$ is the global MLE and $\hat{\hat{\boldsymbol{\theta}}}$ denotes the profiled nuisance parameters at fixed $\mu$. The procedure:

1. Perform the unconditional (free) fit to obtain $(\hat{\mu}, \hat{\boldsymbol{\theta}})$ and $\text{nll}_\text{free}$.
2. For each grid point $\mu_i$: fix the POI, minimize over nuisance parameters using `fit_minimum()`.
3. Compute $q_{\mu_i} = \max(0,\ 2(\text{nll}_{\mu_i} - \text{nll}_\text{free}))$.

The non-negativity constraint enforces the physical requirement that the conditional fit cannot be better than the unconditional one.

### 4.3 Asymptotic CLs Hypothesis Test

NextStat implements asymptotic CLs using the $\tilde{q}_\mu$ test statistic following the framework of Cowan et al. [10].

**Asimov dataset.** The expected dataset under the background-only hypothesis ($\mu = 0$) is constructed by performing a conditional fit at $\mu = 0$ and computing expected bin counts at the fitted nuisance parameter values.

**Test statistic.** The transformed test statistic $t_\mu$ is:

$$t_\mu = \begin{cases} \sqrt{q_\mu} - \sqrt{q_{\mu,A}} & \text{if } \sqrt{q_\mu} \leq \sqrt{q_{\mu,A}} \\ \dfrac{q_\mu - q_{\mu,A}}{2\sqrt{q_{\mu,A}}} & \text{otherwise} \end{cases}$$

where $q_{\mu,A}$ is the test statistic evaluated on the Asimov dataset.

**CLs value:**

$$\text{CLs}(\mu) = \frac{\text{CL}_{s+b}}{\text{CL}_b} = \frac{\Phi(-(t_\mu + \sqrt{q_{\mu,A}}))}{\Phi(-t_\mu)}$$

**Expected CLs band.** Under the background-only hypothesis, the expected CLs at $n\sigma$ is:

$$\text{CLs}^\text{exp}(n\sigma) = \frac{\Phi(-(n + \sqrt{q_{\mu,A}}))}{\Phi(-n)}$$

for $n \in \{-2, -1, 0, +1, +2\}$ (the Brazil band).

**Upper limit.** The observed upper limit $\mu_\text{up}$ satisfies $\text{CLs}(\mu_\text{up}) = \alpha$ (typically $\alpha = 0.05$). Two root-finding strategies are available: linear scan with interpolation and bisection with configurable tolerance.

### 4.4 Bayesian Posterior Sampling (NUTS/HMC)

NextStat includes a No-U-Turn Sampler (NUTS) [11] implemented in Rust, following the multinomial variant described in [7].

**Log-posterior:**

$$\log p(\boldsymbol{\theta} \mid \text{data}) = \log\mathcal{L}(\text{data} \mid \boldsymbol{\theta}) + \log\pi(\boldsymbol{\theta}) + \text{const}$$

Constraint terms already included in $\log\mathcal{L}$ are not repeated in the prior $\pi$ to avoid double-counting.

**Unconstrained parameterization.** For bounded parameters, NextStat samples in an unconstrained space $\mathbf{z} \in \mathbb{R}^n$ with bijective transforms and their log-Jacobian corrections:

| Bounds | Transform | Log-Jacobian |
|--------|-----------|--------------|
| $(-\infty, +\infty)$ | $\theta = z$ (identity) | $0$ |
| $(a, +\infty)$ | $\theta = a + e^z$ | $z$ |
| $(a, b)$ | $\theta = a + (b-a)\sigma(z)$ | $\log(b-a) + z - 2\log(1+e^z)$ |

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function.

**Leapfrog integrator.** Symplectic integration of Hamilton's equations with half-steps:

$$\mathbf{p}_{1/2} = \mathbf{p}_0 - \frac{\varepsilon}{2}\nabla_\mathbf{z} U(\mathbf{z}_0), \quad
\mathbf{z}_1 = \mathbf{z}_0 + \varepsilon\, M^{-1}\mathbf{p}_{1/2}, \quad
\mathbf{p}_1 = \mathbf{p}_{1/2} - \frac{\varepsilon}{2}\nabla_\mathbf{z} U(\mathbf{z}_1)$$

where $U = -\log p$ is the potential energy and $M$ is the mass matrix.

**Tree building.** Multinomial NUTS builds a balanced binary tree by recursive doubling. Expansion stops when a U-turn is detected ($\langle \Delta\mathbf{z}, \mathbf{p}\rangle < 0$) or the maximum tree depth (default 10) is reached. Proposals are selected via multinomial weighting. Divergent transitions ($|\Delta H| > 1000$) are flagged.

**Adaptation.** During warmup, step size is adapted via dual averaging (target acceptance rate 0.8), and a diagonal mass matrix is estimated via Welford online variance. A windowed adaptation schedule divides warmup into slow, fast, and final-tuning phases.

**NUTS configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_treedepth` | 10 | Maximum tree depth |
| `target_accept` | 0.8 | Target acceptance probability |
| `init_jitter` | 0.0 | Jitter in unconstrained space |
| `num_warmup` | 500 | Warmup iterations |
| `num_samples` | 1000 | Post-warmup samples |

**Multi-chain sampling** runs chains in parallel via Rayon with deterministic seeding: `chain_seed = base_seed + chain_id`.

**Diagnostics:**

| Diagnostic | Threshold | Description |
|------------|-----------|-------------|
| Split $\hat{R}$ | $< 1.01$ | Potential scale reduction factor |
| ESS bulk | $\geq 100$ per chain | Effective sample size (bulk) |
| ESS tail | $\geq 100$ per chain | Effective sample size (tail) |
| E-BFMI | $> 0.3$ | Energy Bayesian fraction of missing information |
| Divergence rate | $< 1\%$ | Fraction of divergent transitions |

## 5. Automatic Differentiation

The `ns-ad` crate provides two AD modes unified under a `Scalar` trait.

### 5.1 Forward Mode (Dual Numbers)

The `Dual` struct carries a value and its derivative simultaneously:

```rust
pub struct Dual {
    pub val: f64,
    pub dot: f64,
}
```

All elementary operations (`+`, `−`, `×`, `÷`, `ln`, `exp`, `sqrt`, `pow`, `abs`, `max`, `clamp`) implement the chain rule. To compute $\partial\,\text{nll}/\partial\theta_i$, one evaluates the NLL with $\theta_i$ seeded as `Dual { val: θ_i, dot: 1.0 }` and all other parameters seeded with `dot: 0.0`. The full gradient requires $d$ evaluations for $d$ parameters.

**Cost:** $O(d)$ function evaluations. Memory: $O(1)$ additional per evaluation (no tape).

**Best for:** Models with $d \lesssim 30$ parameters, where the overhead of tape recording is not justified.

### 5.2 Reverse Mode (Tape)

The `Tape` struct records a computation DAG in the forward pass:

```rust
pub struct Tape {
    nodes: Vec<Node>,
    adjoints: Vec<f64>,
}
pub struct Var(usize);  // Handle to a tape node
```

A single backward sweep computes $\nabla_{\boldsymbol{\theta}}\,\text{nll}$ by propagating adjoints from the output to all input variables.

**Cost:** 1 forward pass + 1 backward pass, independent of $d$. Memory: $O(N)$ where $N$ is the number of operations recorded.

**Best for:** Models with $d > 100$ parameters.

### 5.3 Generic Scalar Trait

The `Scalar` trait unifies `f64`, `Dual`, and `Tape::Var`:

```rust
pub trait Scalar: Copy + Add + Sub + Mul + Div + Neg {
    fn from_f64(v: f64) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn max(self, other: Self) -> Self;
    // ...
}
```

The NLL function `nll_generic<T: Scalar>` is written once and works for plain evaluation, forward AD, and reverse AD. This design eliminates code duplication between the evaluation and differentiation codepaths.

## 6. Performance

### 6.1 SIMD Vectorization

The Poisson NLL inner loop in `ns-compute` processes four bins simultaneously using `f64x4` SIMD operations (via the `wide` crate [12]):

```
nll_i = exp_i - obs_i · ln(exp_i) + ln(obs_i!)
```

The kernel uses a branchless mask to skip $\ln$ computation when all four lanes have $\text{obs} = 0$. This is important for sparse histograms common in tail regions.

**Precision.** The `wide::f64x4::ln()` function exhibits approximately 1000 ULP error relative to scalar `f64::ln()`. NextStat instead extracts each lane to scalar, computes `f64::ln()`, and repacks into the SIMD register. This preserves bit-exact agreement with the scalar reference path while still benefiting from SIMD for all arithmetic operations.

**Dispatch.** The SIMD backend auto-detects AVX2 (x86_64) or NEON (aarch64) at compile time and falls back to scalar when neither is available.

### 6.2 Thread-Level Parallelism

Rayon [13] provides work-stealing parallelism for:

| Workflow | Parallelization |
|----------|----------------|
| Toy ensembles | Each toy (generate + fit) is independent |
| Multi-chain NUTS | Each chain runs independently |
| Profile scans | Each grid point (conditional fit) is independent |

Thread count is controlled via `--threads` (CLI) or `rayon::ThreadPoolBuilder`.

### 6.3 Compiler Optimizations

The release profile enables aggressive optimization:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

Benchmarks use `lto = "fat"` for maximum cross-crate inlining.

### 6.4 Benchmark Infrastructure

Performance is measured using Criterion [14] (Rust micro-benchmarking framework) with the following suites:

| Suite | Measures |
|-------|----------|
| `simd_benchmark` | Poisson NLL: scalar vs SIMD at 4, 100, 1K, 10K bins |
| `poisson_nll_full` | NLL throughput with sparse/dense histograms |
| `mle_benchmark` | End-to-end fit time (simple + complex workspaces) |
| `ad_benchmark` | Forward vs reverse AD cost scaling with parameter count |

## 7. Validation

### 7.1 Parity Contract with pyhf

NextStat's deterministic CPU path is validated against pyhf (NumPy backend) on a suite of test workspaces. The parity contract specifies:

| Quantity | Relative tol. | Absolute tol. |
|----------|---------------|---------------|
| `twice_nll(θ)` | $10^{-6}$ | $10^{-8}$ |
| Best-fit parameters | — | $2 \times 10^{-4}$ |
| Uncertainties | — | $5 \times 10^{-4}$ |

Parameters are compared **by name** (not by index) to catch ordering mismatches between implementations.

### 7.2 Test Fixtures

| Fixture | Channels | Bins | Samples | Modifiers | Parameters |
|---------|----------|------|---------|-----------|------------|
| `simple_workspace.json` | 1 | 2 | 2 | 3 | 3 |
| `complex_workspace.json` | 2 | 8 | 5 | 12 | 18 |

Parity tests are evaluated at nominal parameters, random perturbations, and parameter boundaries.

### 7.3 Test Statistics and CLs

Golden-point tests compare $q_\mu$, $\tilde{q}_\mu$, observed/expected CLs values, and upper limits at selected signal strength points.

### 7.4 Statistical Quality (Toy Ensembles)

Toy ensembles ($N_\text{toys} = 200$ for smoke tests, $\geq 5000$ for certification) verify that no systematic bias is introduced:

| Metric | Tolerance |
|--------|-----------|
| $|\Delta\,\text{mean(pull)}|$ | $\leq 0.05$ |
| $|\Delta\,\text{std(pull)}|$ | $\leq 0.05$ |
| $|\Delta\,\text{coverage}_{1\sigma}|$ | $\leq 0.03$ |

### 7.5 Bayesian Diagnostics

NUTS sampler quality is validated on toy distributions (Normal, MVN):

- MAP with flat prior agrees with MLE within frequentist tolerances
- $\hat{R} < 1.01$, divergence rate $< 1\%$, adequate ESS

### 7.6 Running Tests

```bash
# Rust unit and integration tests (excludes ns-py)
cargo test -p ns-core -p ns-ad -p ns-compute -p ns-translate -p ns-inference

# Release-mode tests (required for toy ensembles)
cargo test -p ns-inference --release -- test_fit_toys

# Python parity tests
pytest tests/python/
```

## 8. Summary and Outlook

NextStat provides a high-performance, multi-frontend inference engine for binned likelihood models. Key characteristics:

- **Numerical parity** with pyhf in deterministic mode, validated across likelihood evaluation, MLE, CLs, and toy ensemble statistics.
- **Performance** via SIMD vectorization (f64x4) and Rayon-based parallelism.
- **Unified inference** — both frequentist (MLE, profile, CLs) and Bayesian (NUTS/HMC) in one toolkit.
- **Three access modes** (Rust library, Python package, CLI) sharing a common numerical core.
- **Composable AD** — forward and reverse mode via a generic `Scalar` trait, enabling gradient-based optimization and sampling without code duplication.

**Future work** includes GPU backends (Metal, CUDA), expanded input format support, full mass matrix adaptation for NUTS, rank-normalized MCMC diagnostics, and published benchmark comparisons.

NextStat is open-source under AGPL-3.0-or-later with optional commercial licensing. Source code: [https://github.com/nextstat/nextstat](https://github.com/nextstat/nextstat). Documentation: [https://nextstat.io](https://nextstat.io).

## References

[1] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[2] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[3] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[4] R. Brun, F. Rademakers, "ROOT — An object oriented data analysis framework," Nucl. Instrum. Meth. A389, 81–86, 1997.

[5] W. Verkerke, D. Kirkby, "The RooFit toolkit for data modeling," arXiv:physics/0306116, 2003.

[6] A. Fowlie, "stanhf: HistFactory models in the probabilistic programming language Stan," arXiv:2503.22188, 2025.

[7] B. Carpenter et al., "Stan: A Probabilistic Programming Language," J. Stat. Softw. 76(1), 2017.

[8] NextStat Contributors, "NextStat Model Specification: HistFactory Probability Densities and Modifiers," docs/papers/model-specification.md, 2026.

[9] S. Knopp, "argmin: A pure Rust optimization framework," https://argmin-rs.org, 2024.

[10] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011; Erratum ibid. C73, 2501, 2013.

[11] M. D. Hoffman, A. Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo," JMLR 15, 1593–1623, 2014.

[12] Lokathor, "wide: SIMD-compatible data types," https://crates.io/crates/wide, 2024.

[13] J. Stone, N. Matsakis, "Rayon: A data parallelism library for Rust," https://crates.io/crates/rayon, 2024.

[14] Brook Heisler, "Criterion.rs: Statistics-driven micro-benchmarking," https://crates.io/crates/criterion, 2024.
