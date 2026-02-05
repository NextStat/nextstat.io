# NextStat: High-Performance HistFactory Inference in Rust

**Version:** 0.1.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

We present NextStat, an open-source statistical inference toolkit for High Energy Physics (HEP) implemented in Rust with Python bindings. NextStat provides a high-performance engine for HistFactory-style binned likelihood analyses, supporting maximum likelihood estimation, profile likelihood scans, asymptotic CLs hypothesis tests, and Bayesian posterior sampling via NUTS/HMC. The toolkit ingests pyhf JSON workspaces and maintains strict numerical parity with pyhf in a deterministic CPU reference mode, while exploiting SIMD vectorization and thread-level parallelism for performance-critical workflows. We describe the motivation, software architecture, inference algorithms, validation methodology, and performance characteristics of the toolkit.

---

## 1. Introduction

Modern particle physics analyses at the LHC rely on parameterized binned likelihood models to extract physics results from data. The HistFactory framework [1] provides a widely-used specification for such models, combining Poisson counting statistics with systematic uncertainty modifiers. The pyhf project [2] implements HistFactory likelihoods in pure Python, enabling software-independent reproducibility of ATLAS and CMS results [3].

While pyhf provides an accessible and well-validated reference implementation, many analysis workflows are computationally demanding:

- **Toy ensembles** for bias, pull, and coverage studies require $O(10^3$–$10^5)$ repeated fits.
- **Profile likelihood scans** evaluate the likelihood at $O(10^2)$ grid points, each requiring a constrained minimization.
- **Upper limit searches** combine profile scans with root-finding over a signal strength grid.
- **Bayesian posterior sampling** via MCMC methods requires $O(10^4$–$10^6)$ gradient evaluations.

NextStat addresses these computational challenges by implementing the core inference algorithms in Rust, leveraging:

- **Zero-cost abstractions** and compile-time optimization for tight inner loops.
- **SIMD vectorization** (via the `wide` crate) for batch Poisson NLL evaluation.
- **Thread-level parallelism** (via Rayon) for independent chain sampling and toy generation.
- **Automatic differentiation** (forward and reverse mode) for gradient-based optimization and sampling.

The toolkit exposes three user interfaces: a Rust library, a Python package (via PyO3/maturin), and a command-line interface (CLI). All three share a common core and produce identical numerical results.

This paper is structured as follows. Section 2 provides background on HistFactory and pyhf. Section 3 describes the software architecture. Section 4 details the inference algorithms. Section 5 demonstrates usage through examples. Section 6 presents the validation methodology and results. Section 7 discusses performance. Section 8 concludes with an outlook.

## 2. Background

### 2.1 HistFactory

The HistFactory framework [1] specifies parameterized probability models for binned analyses. A model consists of one or more *channels* (histogram regions), each containing multiple *samples* (signal and background contributions). The expected event count in each bin is a function of *parameters of interest* (POI, typically a signal strength $\mu$) and *nuisance parameters* ($\boldsymbol{\theta}$) encoding systematic uncertainties.

The full likelihood factorizes into Poisson terms for the observed bin counts and constraint terms for the nuisance parameters:

$$\mathcal{L}(\mu, \boldsymbol{\theta}) = \prod_{c \in \text{channels}} \prod_{b \in \text{bins}} \text{Pois}(n_{cb} \mid \lambda_{cb}(\mu, \boldsymbol{\theta})) \times \prod_{j} C_j(\theta_j)$$

where $C_j$ represents Gaussian or Poisson auxiliary constraints on nuisance parameters (see the companion model specification [4] for details).

### 2.2 pyhf and the JSON Workspace Format

The pyhf project [2] provides a pure-Python implementation of HistFactory likelihoods, independent of the ROOT framework [5]. Crucially, pyhf defines a JSON schema for *workspaces* that declaratively describe the full model: channels, samples, modifiers, and their associated data.

The ATLAS collaboration has adopted pyhf JSON as a format for publishing full statistical likelihoods [3], enabling independent reinterpretation of search results. NextStat uses pyhf JSON as its primary input format, ensuring compatibility with this growing ecosystem of published likelihoods.

### 2.3 Related Work

Several tools address similar needs in the HEP statistics ecosystem:

- **RooFit/RooStats** [6]: The original C++/ROOT implementation of HistFactory. Tightly integrated with ROOT; not easily used outside that ecosystem.
- **pyhf** [2]: Pure-Python reference implementation. Supports multiple backends (NumPy, JAX, PyTorch). Widely adopted for published likelihoods.
- **stanhf** [7]: Converts HistFactory models to the Stan probabilistic programming language, enabling Bayesian inference with Stan's NUTS sampler.
- **cabiern** [8]: Rust-based gradient computation for HistFactory models.

NextStat occupies a complementary niche: a self-contained Rust engine that covers both frequentist and Bayesian workflows, while maintaining strict parity with pyhf for validation.

## 3. Software Architecture

### 3.1 Crate Structure

NextStat is organized as a Rust workspace with seven crates, following a layered architecture with dependency inversion:

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

| Crate | Purpose |
|-------|---------|
| `ns-core` | Shared types, traits (`ComputeBackend`, `Model`), error types |
| `ns-compute` | CPU backend with SIMD kernels; optional Metal/CUDA backends |
| `ns-ad` | Forward-mode (Dual) and reverse-mode (Tape) automatic differentiation |
| `ns-translate` | pyhf JSON parsing and `HistFactoryModel` construction |
| `ns-inference` | MLE, profile likelihood, CLs hypothesis tests, NUTS/HMC sampling |
| `ns-viz` | Plot-friendly JSON artifacts (CLs curves, profile scans) |
| `ns-cli` | Command-line interface |
| `ns-py` | Python bindings via PyO3 and maturin |

### 3.2 Design Principles

**Trait-based abstraction.** Inference algorithms depend on abstract traits (`ComputeBackend` for NLL/gradient evaluation, `Model` for parameter metadata), not concrete implementations. This allows backends to evolve independently—adding GPU support does not require changing inference code.

**Deterministic reference path.** A single-threaded CPU path with stable summation order serves as the ground truth for validation. All parity tests with pyhf execute against this path.

**Separation of translation and inference.** Model construction from pyhf JSON (`ns-translate`) is cleanly separated from inference algorithms (`ns-inference`). This enables future support for additional model formats (HistFactory XML, custom specifications) without modifying inference code.

### 3.3 PreparedModel

The `PreparedModel` struct in `ns-translate` caches precomputed quantities for efficient repeated evaluation:

- Flattened observed data across all channels
- Precomputed $\ln\Gamma(n+1)$ terms (log-factorials)
- Observation mask for sparse bins ($n = 0$)
- Constant offset from Gaussian constraints ($\sum_j [\ln\sigma_j + \frac{1}{2}\ln 2\pi]$)
- Code4 interpolation coefficients for NormSys modifiers
- Barlow-Beeston $\tau$ values for ShapeSys modifiers

This caching strategy amortizes setup cost over thousands of NLL evaluations, which is critical for profile scans, toy ensembles, and MCMC sampling.

## 4. Inference Algorithms

### 4.1 Maximum Likelihood Estimation

NextStat performs MLE using the L-BFGS-B algorithm (via the `argmin` crate), which supports box constraints on parameters. The optimizer minimizes the negative log-likelihood $\text{nll}(\boldsymbol{\theta}) = -\log\mathcal{L}(\boldsymbol{\theta})$ subject to parameter bounds.

The fit output includes:

- **Best-fit parameters** $\hat{\boldsymbol{\theta}}$
- **NLL at minimum** $\text{nll}(\hat{\boldsymbol{\theta}})$
- **Hessian-based covariance** $\hat{\Sigma} = H^{-1}$ where $H_{ij} = \partial^2 \text{nll} / \partial\theta_i \partial\theta_j \big|_{\hat{\boldsymbol{\theta}}}$
- **Parameter uncertainties** $\hat{\sigma}_i = \sqrt{\hat{\Sigma}_{ii}}$

A fast path (`fit_minimum`) skips the Hessian computation for use cases that only need the best-fit point (profile scans, toy fits).

### 4.2 Profile Likelihood Scan

For a parameter of interest $\mu$, the profile likelihood ratio is:

$$q_\mu = 2 \left[ \text{nll}(\mu, \hat{\hat{\boldsymbol{\theta}}}) - \text{nll}(\hat{\mu}, \hat{\boldsymbol{\theta}}) \right]$$

where $(\hat{\mu}, \hat{\boldsymbol{\theta}})$ is the global MLE and $\hat{\hat{\boldsymbol{\theta}}}$ denotes the profiled nuisance parameters at fixed $\mu$. NextStat evaluates $q_\mu$ over a user-specified grid with the following procedure:

1. Perform the unconditional (free) fit to obtain $(\hat{\mu}, \hat{\boldsymbol{\theta}})$ and $\text{nll}_\text{free}$.
2. For each grid point $\mu_i$: fix the POI and minimize over nuisance parameters.
3. Compute $q_{\mu_i} = \max(0, 2(\text{nll}_{\mu_i} - \text{nll}_\text{free}))$.

The clipping $q_\mu \geq 0$ enforces the physical boundary: the conditional fit cannot be better than the unconditional one.

### 4.3 Asymptotic CLs Hypothesis Test

NextStat implements asymptotic CLs calculations following Cowan et al. [9] with the $\tilde{q}_\mu$ test statistic, consistent with pyhf's `calctype="asymptotics"` and `test_stat="qtilde"`.

**Asimov dataset construction.** The expected dataset under the background-only hypothesis ($\mu = 0$) is constructed by:

1. Performing a conditional fit at $\mu = 0$.
2. Computing expected bin counts at the fitted nuisance parameter values.
3. Replacing observed data and constraint auxdata with these expected values.

**Test statistic.** The transformed test statistic $t_\mu$ is computed from the observed $q_\mu$ and the Asimov $q_{\mu,A}$:

$$t_\mu = \begin{cases} \sqrt{q_\mu} - \sqrt{q_{\mu,A}} & \text{if } \sqrt{q_\mu} \leq \sqrt{q_{\mu,A}} \\ \frac{q_\mu - q_{\mu,A}}{2\sqrt{q_{\mu,A}}} & \text{if } \sqrt{q_\mu} > \sqrt{q_{\mu,A}} \end{cases}$$

**CLs.** The CLs value at signal strength $\mu$ is:

$$\text{CLs}(\mu) = \frac{\text{CL}_{s+b}}{\text{CL}_b} = \frac{\Phi(-(t_\mu + \sqrt{q_{\mu,A}}))}{\Phi(-t_\mu)}$$

where $\Phi$ is the standard normal CDF.

**Expected CLs band.** Expected CLs values under the background-only hypothesis are computed by shifting the test statistic by $n\sigma$ ($n \in \{-2, -1, 0, +1, +2\}$):

$$\text{CLs}^\text{exp}(n\sigma) = \frac{\Phi(-(n + \sqrt{q_{\mu,A}}))}{\Phi(-n)}$$

**Upper limit.** The observed upper limit $\mu_\text{up}$ satisfies $\text{CLs}(\mu_\text{up}) = \alpha$ (typically $\alpha = 0.05$). NextStat finds this via linear scan with interpolation or bisection root-finding.

### 4.4 Bayesian Posterior Sampling (NUTS/HMC)

NextStat implements a No-U-Turn Sampler (NUTS) [10] in Rust for Bayesian posterior inference. The implementation follows Stan's variant [11]:

**Posterior.** The log-posterior density is:

$$\log p(\boldsymbol{\theta} \mid \text{data}) = \log\mathcal{L}(\text{data} \mid \boldsymbol{\theta}) + \log\pi(\boldsymbol{\theta}) + \text{const}$$

To avoid double-counting, constraints already included in $\log\mathcal{L}$ are not repeated in the prior $\pi$.

**Unconstrained parameterization.** For bounded parameters, NextStat samples in an unconstrained space $\mathbf{z} \in \mathbb{R}^n$ with bijective transforms:

| Bounds | Transform | Jacobian |
|--------|-----------|----------|
| $(-\infty, +\infty)$ | $\theta = z$ (identity) | $0$ |
| $(a, +\infty)$ | $\theta = a + e^z$ | $z$ |
| $(a, b)$ | $\theta = a + (b-a)\sigma(z)$ | $\log(b-a) + z - 2\log(1+e^z)$ |

The log-determinant of the Jacobian is added to the log-posterior.

**Leapfrog integrator.** Symplectic integration of Hamilton's equations with potential $U = -\log p$ and kinetic energy $K = \frac{1}{2}\mathbf{p}^T M^{-1} \mathbf{p}$.

**Tree building.** Multinomial NUTS builds a balanced binary tree by doubling, selecting proposals via multinomial weighting. Tree expansion stops when a U-turn is detected or when the maximum tree depth is reached. Slice variable semantics enforce $\log u = \log U - H_0$ where $H_0$ is the initial Hamiltonian. Divergent transitions ($|\Delta H| > 1000$) are flagged.

**Adaptation.** During warmup:

- Step size is adapted via dual averaging (target acceptance rate 0.8).
- Diagonal mass matrix is estimated via Welford online variance.

**Diagnostics.** Per-chain and cross-chain diagnostics include:

- Split $\hat{R}$ (potential scale reduction factor)
- Effective sample size (ESS), bulk and tail
- Divergence count and rate
- Tree depth saturation rate

### 4.5 Automatic Differentiation

NextStat provides two AD modes in the `ns-ad` crate:

- **Forward mode** (Dual numbers): efficient when the number of parameters is small ($d \lesssim 20$). Each function evaluation simultaneously computes the value and one directional derivative.
- **Reverse mode** (Tape): efficient for large parameter spaces. Records operations on a computational tape and computes the full gradient in a single backward pass.

The `Scalar` trait unifies `f64`, `Dual`, and `Tape` types, allowing the NLL function `nll_generic<T: Scalar>` to be used for both evaluation and differentiation without code duplication.

## 5. Usage Examples

### 5.1 Command-Line Interface

```bash
# MLE fit
nextstat fit --input workspace.json

# Asymptotic CLs hypothesis test
nextstat hypotest --input workspace.json --mu 1.0 --expected-set

# Upper limit (scan + interpolation)
nextstat upper-limit --input workspace.json --alpha 0.05 \
    --scan-start 0.0 --scan-stop 5.0 --scan-points 201

# Profile likelihood scan
nextstat scan --input workspace.json --start 0.0 --stop 2.0 --points 21

# Plot-friendly CLs curve artifact
nextstat viz cls --input workspace.json --alpha 0.05 \
    --scan-start 0.0 --scan-stop 5.0 --scan-points 201
```

### 5.2 Python API

```python
import json
from pathlib import Path
import nextstat
from nextstat import infer

# Load pyhf workspace
ws = json.loads(Path("workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(ws))

# MLE fit
fit = nextstat.fit(model)
print(f"bestfit: {fit.bestfit}")
print(f"uncertainties: {fit.uncertainties}")

# CLs hypothesis test
result = infer.hypotest(1.0, model)
print(f"CLs(mu=1): {result['cls']}")

# Upper limit
limit = infer.upper_limit(model, alpha=0.05)
print(f"mu_up: {limit['mu_up']}")

# NUTS posterior sampling
config = nextstat.NutsConfig(num_warmup=500, num_samples=1000, seed=42)
sampler = nextstat.Sampler(model, config)
chains = sampler.sample(num_chains=4)
```

### 5.3 Rust Library

```rust
use ns_translate::pyhf::Workspace;
use ns_inference::mle::MaximumLikelihoodEstimator;

let json = std::fs::read_to_string("workspace.json")?;
let workspace = Workspace::from_json(&json)?;
let model = workspace.to_model()?;

let mle = MaximumLikelihoodEstimator::default();
let result = mle.fit(&model)?;

println!("bestfit: {:?}", result.parameters);
println!("nll: {}", result.nll);
```

## 6. Validation

### 6.1 Parity with pyhf (Deterministic Mode)

NextStat's deterministic CPU path is validated against pyhf (NumPy backend) on a suite of test workspaces. The parity contract specifies:

| Quantity | Tolerance |
|----------|-----------|
| `twice_nll` | $\text{rtol} = 10^{-6}$, $\text{atol} = 10^{-8}$ |
| Best-fit parameters | $\text{atol} = 2 \times 10^{-4}$ |
| Uncertainties | $\text{atol} = 5 \times 10^{-4}$ |

Tests compare values parameter-by-parameter by name, not by index, to catch ordering mismatches.

### 6.2 Test Statistics and CLs

Golden-point tests compare $q_\mu$, $\tilde{q}_\mu$, CLs values, and upper limits at selected signal strength points against pyhf output. These tests cover:

- Zero-signal and non-zero-signal hypotheses
- Observed and expected (Brazil band) CLs values
- Upper limits via scan+interpolation and bisection

### 6.3 Bias, Pull, and Coverage

Toy ensembles ($N_\text{toys} = 200$) verify statistical quality:

- **Bias**: $\text{bias}(\hat{\theta}) = E[\hat{\theta}] - \theta_\text{true}$
- **Pull mean/std**: $(E[\text{pull}], \text{std}(\text{pull})) \approx (0, 1)$
- **Coverage**: fraction of intervals containing $\theta_\text{true}$

Tolerances for NextStat vs pyhf comparison:

| Metric | Tolerance |
|--------|-----------|
| $|\Delta\text{mean(pull)}|$ | $\leq 0.05$ |
| $|\Delta\text{std(pull)}|$ | $\leq 0.05$ |
| $|\Delta\text{coverage}_{1\sigma}|$ | $\leq 0.03$ |

### 6.4 Bayesian Diagnostics

NUTS sampler quality is monitored via:

- $\hat{R} < 1.01$ on toy problems (Normal, MVN, banana)
- Divergence rate $< 1\%$
- ESS per parameter baseline for regression tracking

## 7. Performance

### 7.1 SIMD Vectorization

The Poisson NLL inner loop processes four bins simultaneously using `f64x4` SIMD lanes (via the `wide` crate). Arithmetic operations (addition, subtraction, multiplication, reduction) use SIMD natively. The natural logarithm is evaluated lane-by-lane with scalar `f64::ln()` to maintain bit-exact precision (`wide::f64x4::ln()` exhibits $\sim$1000 ULP error).

### 7.2 Thread-Level Parallelism

Rayon provides data parallelism for:

- Independent MCMC chain sampling
- Toy ensemble generation and fitting
- Profile scan grid evaluation (each $\mu$ point fitted independently)

### 7.3 Benchmark Strategy

Performance is measured with Criterion (Rust) and timed comparisons against pyhf. Key metrics:

- NLL evaluation throughput (evaluations/second)
- End-to-end MLE fit time
- Profile scan wall time for $N$ grid points
- NUTS effective samples per second

Benchmarks pin CPU frequency scaling and report medians over multiple runs.

## 8. Summary and Outlook

NextStat provides a high-performance, multi-frontend inference engine for HistFactory-style binned likelihoods. Key contributions include:

- **Numerical parity** with pyhf in deterministic mode, validated by a comprehensive test suite.
- **Performance** via SIMD vectorization and thread-level parallelism.
- **Unified frequentist and Bayesian** inference in a single toolkit.
- **Three access modes** (Rust, Python, CLI) sharing a common numerical core.

Future development includes GPU backends (Metal, CUDA), expanded model format support (HistFactory XML), rank-normalized MCMC diagnostics, and comprehensive benchmark publications.

NextStat is open-source under AGPL-3.0-or-later with optional commercial licensing. The source code is available at [https://github.com/nextstat](https://github.com/nextstat).

## References

[1] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[2] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[3] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[4] NextStat Contributors, "NextStat Model Specification: HistFactory Probability Densities and Modifiers," docs/papers/model-specification.md, 2026.

[5] R. Brun, F. Rademakers, "ROOT — An object oriented data analysis framework," Nucl. Instrum. Meth. A389, 81–86, 1997.

[6] W. Verkerke, D. Kirkby, "The RooFit toolkit for data modeling," arXiv:physics/0306116, 2003.

[7] A. Fowlie, "stanhf: HistFactory models in the probabilistic programming language Stan," arXiv:2503.22188, 2025.

[8] L. Heinrich, "cabiern: Rust-based HistFactory gradient computation," GitHub, 2023.

[9] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011.

[10] M. D. Hoffman, A. Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo," JMLR 15, 1593–1623, 2014.

[11] B. Carpenter et al., "Stan: A Probabilistic Programming Language," J. Stat. Softw. 76(1), 2017.
