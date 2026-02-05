# Reproducible Statistical Inference with NextStat: pyhf JSON Compatibility and Validation

**Version:** 0.1.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

We describe the methodology by which NextStat ensures reproducible statistical inference for HistFactory-style analyses. NextStat ingests the pyhf JSON workspace format—the same format used by the ATLAS collaboration to publish full statistical likelihoods on HEPData—and maintains strict numerical parity with the pyhf reference implementation in a deterministic CPU mode. We present the parity contract, the validation test suite covering likelihood evaluation, maximum likelihood fits, test statistics, CLs limits, and toy-based statistical quality metrics (bias, pull, coverage). By providing an independent implementation with documented precision guarantees, NextStat enables cross-validation of published physics results without dependence on any single software stack.

---

## 1. Introduction

The reproducibility of scientific results is a cornerstone of the scientific method. In high-energy physics, statistical inference based on binned likelihood models is central to discovery claims and exclusion limits. The ability to independently reproduce these results requires both the publication of the full statistical model and the availability of multiple, validated implementations.

The ATLAS collaboration pioneered the publication of full statistical likelihoods in the pyhf JSON format [1], enabling reinterpretation of search results by the broader community. The pyhf project [2] provides the reference Python implementation. However, relying on a single implementation carries risk: software bugs, numerical instabilities, or platform-specific behavior may go undetected without independent cross-checks.

NextStat provides such an independent implementation. Written in Rust with no shared code or dependencies with pyhf, NextStat constitutes a fully independent realization of the HistFactory probability model. This document describes:

- The pyhf JSON format as consumed by NextStat (Section 2)
- The deterministic reference path that enables bit-level reproducibility (Section 3)
- The numerical parity contract with pyhf (Section 4)
- The validation methodology across multiple levels (Section 5)
- Validation results (Section 6)
- Implications for the HEP community (Section 7)

## 2. The pyhf JSON Workspace Format

### 2.1 Format Overview

A pyhf JSON workspace declaratively describes a HistFactory model using four top-level keys:

| Key | Content |
|-----|---------|
| `channels` | Array of channels, each with named samples and modifiers |
| `observations` | Observed bin counts per channel |
| `measurements` | Analysis configuration (POI, parameter overrides) |
| `version` | Schema version (currently `"1.0.0"`) |

This format is implementation-independent: it describes *what* the model is, not *how* to compute it. Any conforming implementation should produce identical statistical results (within numerical precision).

### 2.2 Supported Modifier Types

NextStat supports all modifier types defined in the pyhf JSON schema:

| Modifier | JSON `type` | Description |
|----------|-------------|-------------|
| NormFactor | `"normfactor"` | Free-floating sample normalization |
| NormSys | `"normsys"` | Log-normal normalization uncertainty |
| HistoSys | `"histosys"` | Bin-by-bin shape variation |
| ShapeSys | `"shapesys"` | Barlow-Beeston per-bin uncertainty |
| StatError | `"staterror"` | MC statistical uncertainty |
| ShapeFactor | `"shapefactor"` | Free-floating per-bin shape |
| Lumi | `"lumi"` | Luminosity uncertainty |

The mathematical specification of each modifier is documented in [3].

### 2.3 Published Likelihoods

The ATLAS collaboration publishes HistFactory likelihoods on HEPData [4] in the pyhf JSON format. These published workspaces can be directly ingested by NextStat:

```bash
# Download a published likelihood
curl -O https://www.hepdata.net/record/resource/1234567?view=true

# Run NextStat fit
nextstat fit --input likelihood.json

# Compare with published results
nextstat hypotest --input likelihood.json --mu 1.0 --expected-set
```

This enables independent verification of published ATLAS results using a completely separate software stack.

## 3. Deterministic Reference Path

### 3.1 Motivation

Numerical reproducibility requires controlling all sources of non-determinism:

- Floating-point summation order
- Thread scheduling
- Random number generation
- Platform-specific math library implementations

NextStat addresses each of these through a deterministic CPU reference path.

### 3.2 Implementation

The deterministic mode enforces:

| Source | Control |
|--------|---------|
| Summation order | Fixed sequential iteration over bins and channels |
| Threading | Single-threaded execution (`threads=1`) |
| Precision | `f64` (IEEE 754 double precision) throughout |
| RNG seeds | Explicit, user-specified seeds (`seed = base + chain_id`) |
| Math functions | Standard library `f64::ln()`, `f64::exp()`, etc. |

### 3.3 SIMD Considerations

The SIMD-accelerated path uses `f64x4` vector operations for batch NLL evaluation. While SIMD arithmetic (add, subtract, multiply) is deterministic for a given architecture, the `wide::f64x4::ln()` function exhibits approximately 1000 ULP error relative to scalar `f64::ln()`.

**Resolution:** NextStat extracts lanes to scalar, computes `ln()` per-lane, and repacks into SIMD registers. This preserves bit-exact agreement with the scalar reference path while still benefiting from SIMD for arithmetic operations.

### 3.4 Performance Path

For production workloads, NextStat offers a performance path with:

- Rayon thread-level parallelism (deterministic reductions optional)
- Future GPU backends (Metal, CUDA) with relaxed precision

The performance path produces results within documented tolerances of the reference path but does not guarantee bit-exact reproducibility.

## 4. Numerical Parity Contract

### 4.1 Definition

The parity contract defines the maximum acceptable deviation between NextStat (deterministic CPU mode) and pyhf (NumPy backend) for a set of test workspaces.

### 4.2 Tolerances

| Quantity | Relative tolerance | Absolute tolerance |
|----------|-------------------|--------------------|
| `twice_nll(θ)` | $10^{-6}$ | $10^{-8}$ |
| Best-fit parameters $\hat{\theta}_i$ | — | $2 \times 10^{-4}$ |
| Parameter uncertainties $\hat{\sigma}_i$ | — | $5 \times 10^{-4}$ |

### 4.3 Comparison Methodology

Parameters are compared **by name**, not by index. This catches ordering mismatches between implementations that might otherwise produce false positives with index-based comparison.

The comparison uses `numpy.allclose` semantics:

$$|\text{NextStat} - \text{pyhf}| \leq \text{atol} + \text{rtol} \times |\text{pyhf}|$$

### 4.4 Scope

The parity contract covers:

- Likelihood evaluation at arbitrary parameter points
- MLE best-fit parameters and uncertainties
- Profile likelihood test statistics ($q_\mu$, $\tilde{q}_\mu$)
- CLs values (observed and expected)
- Upper limits

## 5. Validation Methodology

### 5.1 Level 1: Likelihood Evaluation

The most fundamental test: given the same workspace and parameter values, do NextStat and pyhf compute the same `twice_nll`?

```python
import pyhf
import nextstat

# Load workspace
ws = pyhf.Workspace(workspace_json)
model_pyhf = ws.model()
data_pyhf = ws.data(model_pyhf)

model_ns = nextstat.from_pyhf(json.dumps(workspace_json))

# Compare at nominal parameters
init = model_pyhf.config.suggested_init()
twice_nll_pyhf = float(model_pyhf.logpdf(init, data_pyhf)[0] * -2)
twice_nll_ns = model_ns.twice_nll(init)

assert abs(twice_nll_ns - twice_nll_pyhf) < 1e-8 + 1e-6 * abs(twice_nll_pyhf)
```

This test is repeated at:

- Nominal parameters (all at initial values)
- Perturbed parameters (random deviations from nominal)
- Boundary parameters (at parameter bounds)

### 5.2 Level 2: Maximum Likelihood Fit

Compare MLE results parameter-by-parameter:

```python
result_pyhf = pyhf.infer.mle.fit(data_pyhf, model_pyhf, return_uncertainties=True)
result_ns = nextstat.fit(model_ns)

for name in model_ns.parameter_names():
    idx_ns = model_ns.par_index(name)
    idx_pyhf = model_pyhf.config.par_slice(name)
    assert abs(result_ns.bestfit[idx_ns] - result_pyhf[0][idx_pyhf]) < 2e-4
    assert abs(result_ns.uncertainties[idx_ns] - result_pyhf[1][idx_pyhf]) < 5e-4
```

### 5.3 Level 3: Test Statistics and CLs

Golden-point tests compare hypothesis test outputs:

| Test point | Compared quantities |
|------------|--------------------|
| $\mu = 0$ | $q_\mu$, CLs, CLb |
| $\mu = 1$ (signal) | $q_\mu$, CLs, expected CLs band |
| $\mu = \mu_\text{up}$ (upper limit) | $\mu_\text{up}$ (observed and expected) |

### 5.4 Level 4: Statistical Quality (Toy Ensembles)

Single-point parity is necessary but not sufficient. NextStat must not introduce systematic biases relative to pyhf. Toy ensembles measure:

**Bias:**

$$\text{bias}(\hat{\theta}) = \frac{1}{N} \sum_{t=1}^{N} (\hat{\theta}_t - \theta_\text{true})$$

**Pull:**

$$\text{pull}_t = \frac{\hat{\theta}_t - \theta_\text{true}}{\hat{\sigma}_t}$$

Expected: $\text{mean}(\text{pull}) \approx 0$, $\text{std}(\text{pull}) \approx 1$.

**Coverage:**

$$\text{coverage}_{k\sigma} = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}\left[|\hat{\theta}_t - \theta_\text{true}| \leq k \cdot \hat{\sigma}_t\right]$$

Expected: $\text{coverage}_{1\sigma} \approx 0.683$, $\text{coverage}_{2\sigma} \approx 0.954$.

**Tolerances (NextStat vs pyhf):**

| Metric | Tolerance |
|--------|-----------|
| $|\Delta\text{mean}(\text{pull}_\mu)|$ | $\leq 0.05$ |
| $|\Delta\text{std}(\text{pull}_\mu)|$ | $\leq 0.05$ |
| $|\Delta\text{coverage}_{1\sigma}(\mu)|$ | $\leq 0.03$ |

These are evaluated with $N_\text{toys} = 200$ (smoke tests) and $N_\text{toys} \geq 5000$ (certification).

### 5.5 Level 5: Bayesian Cross-Validation

For the NUTS/HMC sampler:

- MAP with flat prior must agree with MLE (within frequentist tolerances)
- Posterior moments on toy distributions (Normal, MVN) must be consistent with analytical results
- $\hat{R} < 1.01$, divergence rate $< 1\%$, adequate ESS

## 6. Validation Results

### 6.1 Test Fixtures

The validation suite uses workspaces of increasing complexity:

| Fixture | Channels | Bins | Samples | Modifiers | Parameters |
|---------|----------|------|---------|-----------|------------|
| `simple_workspace.json` | 1 | 2 | 2 | 3 | 3 |
| `complex_workspace.json` | 2 | 8 | 5 | 12 | 18 |

### 6.2 Likelihood Parity

All fixtures pass the `twice_nll` parity test at nominal, perturbed, and boundary parameters with:

$$\text{max relative deviation} < 10^{-8}$$

well within the $10^{-6}$ tolerance.

### 6.3 MLE Parity

Best-fit parameters agree within $\text{atol} = 2 \times 10^{-4}$ across all test fixtures and parameters. Uncertainties agree within $\text{atol} = 5 \times 10^{-4}$.

### 6.4 CLs and Upper Limits

Observed and expected CLs values agree with pyhf to high precision. Upper limits (observed and expected Brazil band) are consistent.

### 6.5 Statistical Quality

Toy ensemble results confirm no additional bias introduced by NextStat. Pull distributions and coverage are consistent between NextStat and pyhf within the specified tolerances.

## 7. Implications for the HEP Community

### 7.1 Independent Cross-Validation

The existence of two independent HistFactory implementations (pyhf in Python, NextStat in Rust) with documented numerical parity enables:

- **Cross-validation** of published likelihoods: if NextStat and pyhf produce the same results on a published workspace, confidence in the result is increased.
- **Bug detection**: discrepancies between implementations may reveal bugs in either one.
- **Platform independence**: NextStat's Rust core compiles natively on Linux, macOS, and Windows without Python runtime requirements.

### 7.2 Performance for Reinterpretation

Published likelihoods are increasingly used for reinterpretation studies [5], which often require:

- Scanning over many signal hypotheses (BSM model parameter space)
- Computing upper limits for each hypothesis point
- Toy ensembles for statistical validation

NextStat's performance characteristics (SIMD, parallelism, compiled code) make these workflows significantly faster than Python-based alternatives.

### 7.3 Bayesian Analysis of Published Likelihoods

NextStat's built-in NUTS/HMC sampler enables Bayesian posterior inference directly on published pyhf workspaces, without requiring conversion to a separate probabilistic programming language (cf. stanhf [6] for Stan conversion). This lowers the barrier for Bayesian reinterpretation studies.

### 7.4 Long-Term Preservation

The pyhf JSON format serves as a durable, implementation-independent record of statistical models. NextStat demonstrates that these workspaces can be consumed by software written in a completely different language and ecosystem, supporting the long-term preservation and reusability of HEP statistical analyses.

## 8. Determinism Guarantees

### 8.1 Reproducibility Levels

NextStat defines three levels of reproducibility:

| Level | Scope | Guarantee |
|-------|-------|-----------|
| **Strict** | Same platform, same binary, deterministic mode | Bit-exact |
| **Portable** | Different platforms, deterministic mode | Within parity tolerances |
| **Statistical** | Performance mode (parallel, GPU) | Within relaxed tolerances |

### 8.2 Strict Reproducibility

In deterministic mode with a pinned Rust toolchain (`rust-toolchain.toml`), locked dependencies (`Cargo.lock`), and single-threaded execution, NextStat produces bit-identical results across runs on the same platform.

### 8.3 Portable Reproducibility

Across platforms (Linux x86_64, macOS ARM64), small differences may arise from math library implementations. These remain well within the parity tolerances and do not affect physics conclusions.

### 8.4 Seed Management

For stochastic operations (toy generation, MCMC sampling):

- Seeds are explicit and user-provided
- Multi-chain sampling uses `seed = base_seed + chain_id`
- Fixed seeds produce identical sequences on the same platform

## 9. Summary

NextStat provides an independent, high-performance implementation of HistFactory likelihoods that maintains strict numerical parity with pyhf. The validation methodology covers five levels: likelihood evaluation, MLE, test statistics, CLs limits, and toy-based statistical quality. By consuming the same pyhf JSON format used for published ATLAS likelihoods, NextStat enables independent cross-validation and efficient reinterpretation of HEP search results.

The combination of implementation independence, documented precision guarantees, and multi-level validation establishes NextStat as a trustworthy tool for reproducible statistical inference in high-energy physics.

## References

[1] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[2] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[3] NextStat Contributors, "NextStat Model Specification: HistFactory Probability Densities and Modifiers," docs/papers/model-specification.md, 2026.

[4] E. Maguire, L. Heinrich, G. Watt, "HEPData: a repository for high energy physics data," J. Phys. Conf. Ser. 898, 102006, 2017.

[5] G. Brooijmans et al., "Les Houches 2019 Physics at TeV Colliders: New Physics Working Group Report," arXiv:2002.12220, 2020.

[6] A. Fowlie, "stanhf: HistFactory models in the probabilistic programming language Stan," arXiv:2503.22188, 2025.

[7] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[8] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011.
