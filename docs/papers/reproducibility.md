# Reproducible Statistical Inference with NextStat: Numerical Parity and Multi-Level Validation

**Version:** 0.2.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

We describe the methodology by which NextStat ensures reproducible statistical inference for binned likelihood analyses. NextStat ingests the pyhf JSON workspace format [1] and maintains strict numerical parity with the pyhf reference implementation in a deterministic CPU mode. We present the deterministic execution model, the numerical parity contract, and a five-level validation test suite covering likelihood evaluation, maximum likelihood fits, test statistics, CLs limits, toy-based statistical quality metrics (bias, pull, coverage), and Bayesian sampler diagnostics. The validation suite comprises 9 Python test modules, 3 Rust integration test files, and 5 test fixtures, with tolerances ranging from $10^{-10}$ (NLL parity) to $5 \times 10^{-4}$ (iterative root-finding). By providing an independent implementation with documented precision guarantees, NextStat enables cross-validation of published results and reduces dependence on any single software stack.

---

## 1. Introduction

The reproducibility of statistical results requires both the publication of the full statistical model and the availability of multiple, validated implementations. The pyhf project [1] provides a pure-Python reference implementation of the HistFactory probability model [2] and defines a JSON schema for declarative model specification. This format has been adopted for publishing full statistical likelihoods on public data repositories [3, 4].

Relying on a single implementation carries inherent risk: software bugs, numerical instabilities, or platform-specific behavior may go undetected without independent cross-checks. NextStat provides such an independent check. Written in Rust with no shared code or dependencies with pyhf, NextStat constitutes a fully independent realization of the HistFactory probability model. The mathematical details are documented in the companion model specification [5]; this paper focuses on the validation methodology.

We describe:

- The pyhf JSON format as consumed by NextStat (Section 2)
- The deterministic reference path that enables bit-level reproducibility (Section 3)
- The numerical parity contract (Section 4)
- The five-level validation methodology (Section 5)
- The concrete test suite implementation (Section 6)
- Determinism guarantees and seed management (Section 7)
- Reproducibility levels (Section 8)

## 2. The pyhf JSON Workspace Format

### 2.1 Format Overview

A pyhf JSON workspace declaratively describes a HistFactory model using four top-level keys:

| Key | Content |
|-----|---------|
| `channels` | Array of channels, each with named samples and modifiers |
| `observations` | Observed bin counts per channel |
| `measurements` | Analysis configuration (POI, parameter overrides) |
| `version` | Schema version (currently `"1.0.0"`) |

This format is implementation-independent: it describes *what* the model is, not *how* to compute it. Any conforming implementation should produce identical statistical results within numerical precision.

### 2.2 Supported Modifier Types

NextStat supports all seven modifier types defined in the pyhf JSON schema:

| Modifier | JSON `type` | Parameters | Constraint |
|----------|-------------|------------|------------|
| NormFactor | `normfactor` | 1 (free) | None |
| NormSys | `normsys` | 1 | Gaussian |
| HistoSys | `histosys` | 1 | Gaussian |
| ShapeSys | `shapesys` | $N_\text{bins}$ | Poisson ($\gamma\tau$) |
| StatError | `staterror` | $N_\text{bins}$ | Gaussian |
| ShapeFactor | `shapefactor` | $N_\text{bins}$ | None |
| Lumi | `lumi` | 1 | Gaussian |

### 2.3 Workspace Ingestion

NextStat parses the JSON workspace and constructs an internal `HistFactoryModel` with parameter ordering derived from the modifier structure. Parameter names and ordering must match pyhf exactly; this is verified in the test suite (Section 6).

```python
import nextstat, json

ws = json.loads(open("workspace.json").read())
model = nextstat.from_pyhf(json.dumps(ws))

# Verify parameter layout
print(model.parameter_names())   # e.g. ["mu", "uncorr_bkguncrt[0]", "uncorr_bkguncrt[1]"]
print(model.n_params())          # e.g. 3
print(model.poi_index())         # e.g. 0
print(model.suggested_init())    # e.g. [1.0, 1.0, 1.0]
print(model.suggested_bounds())  # e.g. [(0.0, 10.0), (1e-10, 10.0), (1e-10, 10.0)]
```

## 3. Deterministic Reference Path

### 3.1 Sources of Non-Determinism

Numerical reproducibility requires controlling:

- Floating-point summation order (associativity of IEEE 754 addition)
- Thread scheduling and work partitioning
- Random number generation
- Platform-specific math library implementations
- SIMD approximation errors

### 3.2 Deterministic Mode

NextStat's deterministic mode enforces:

| Source | Control mechanism |
|--------|------------------|
| Summation order | Fixed sequential iteration over bins and channels |
| Threading | Single-threaded execution (`--threads 1`) |
| Precision | `f64` (IEEE 754 double precision) throughout |
| RNG seeds | Explicit, user-specified (`seed = base + chain_id`) |
| Math functions | Scalar `f64::ln()`, `f64::exp()` from Rust standard library |

### 3.3 SIMD Precision

The SIMD-accelerated path uses `f64x4` vector operations for batch NLL evaluation. While SIMD arithmetic (add, subtract, multiply) is deterministic per-architecture, the `wide::f64x4::ln()` function exhibits approximately 1000 ULP error relative to scalar `f64::ln()`.

**Resolution.** NextStat extracts lanes to scalar, computes `ln()` per-lane, and repacks into SIMD registers. This preserves bit-exact agreement with the scalar reference path while retaining SIMD benefits for arithmetic operations. The branchless observation mask further avoids `ln()` calls entirely when all four lanes have $\text{obs} = 0$ (sparse bins).

### 3.4 Performance Mode

For production workloads, a performance mode offers:

- Rayon thread-level parallelism for toy ensembles, MCMC chains, and profile scans
- Future GPU backends (Metal, CUDA) with relaxed precision

Results remain within documented tolerances of the reference path but do not guarantee bit-exact reproducibility.

## 4. Numerical Parity Contract

### 4.1 Definition

The parity contract defines the maximum acceptable deviation between NextStat (deterministic CPU mode) and pyhf (NumPy backend) for a set of test workspaces.

### 4.2 Tolerances

| Quantity | Relative tol. | Absolute tol. |
|----------|---------------|---------------|
| `twice_nll(θ)` | — | $10^{-10}$ |
| Best-fit parameters $\hat{\theta}_i$ | — | $2 \times 10^{-4}$ |
| Parameter uncertainties $\hat{\sigma}_i$ | — | $5 \times 10^{-4}$ |
| CLs values | — | $5 \times 10^{-6}$ |
| $q_\mu$ test statistic | — | $5 \times 10^{-6}$ |
| Upper limits (scan + interpolation) | — | $5 \times 10^{-5}$ |
| Upper limits (root-finding) | — | $5 \times 10^{-4}$ |

The NLL tolerance ($10^{-10}$) is tighter than the contract minimum ($10^{-8}$) because the deterministic mode achieves near-exact agreement. Upper limit tolerances are broader because interpolation and iterative root-finding amplify small NLL differences.

### 4.3 Comparison Methodology

Parameters are compared **by name**, not by index. This catches ordering mismatches between implementations that might otherwise produce false-positive agreement.

The comparison uses `numpy.allclose` semantics:

$$|\text{NextStat} - \text{pyhf}| \leq \text{atol} + \text{rtol} \times |\text{pyhf}|$$

### 4.4 Scope

The parity contract covers:

- Likelihood evaluation at arbitrary parameter points (nominal, perturbed, boundary)
- MLE best-fit parameters and uncertainties
- Profile likelihood test statistic $q_\mu$
- CLs values (observed and expected Brazil band)
- Upper limits (observed and expected, via scan and root-finding)
- Toy-based pull distribution statistics (mean, std, coverage)

## 5. Validation Methodology: Five Levels

### 5.1 Level 1 — Likelihood Evaluation

The most fundamental test: given the same workspace and parameter values, do NextStat and pyhf compute the same `twice_nll`?

```python
import pyhf, nextstat, json

# Load workspace in both implementations
ws_json = json.loads(open("workspace.json").read())
model_pyhf = pyhf.Workspace(ws_json).model()
data_pyhf = pyhf.Workspace(ws_json).data(model_pyhf)
model_ns = nextstat.from_pyhf(json.dumps(ws_json))

# Compare at nominal parameters
init = list(model_pyhf.config.suggested_init())
twice_nll_pyhf = float(model_pyhf.logpdf(init, data_pyhf)[0] * -2)
twice_nll_ns = model_ns.nll(init) * 2

assert abs(twice_nll_ns - twice_nll_pyhf) < 1e-10
```

This test is repeated at:
- **Nominal** parameters (all at initial values)
- **Perturbed** parameters (shifted by $\pm 0.123$ from nominal)
- **Varied POI** values ($\mu \in \{0.0, 0.5, 1.0, 2.0\}$)

### 5.2 Level 2 — Maximum Likelihood Fit

Compare MLE results parameter-by-parameter (matched by name):

```python
# pyhf fit
result_pyhf = pyhf.infer.mle.fit(data_pyhf, model_pyhf, return_uncertainties=True)

# NextStat fit
result_ns = nextstat.fit(model_ns)

# Compare best-fit values
for i, name in enumerate(model_ns.parameter_names()):
    assert abs(result_ns.bestfit[i] - result_pyhf[0][i]) < 2e-4
    assert abs(result_ns.uncertainties[i] - result_pyhf[1][i]) < 5e-4

# Compare NLL at minimum
assert abs(result_ns.nll * 2 - float(model_pyhf.logpdf(result_pyhf[0].tolist(), data_pyhf)[0] * -2)) < 1e-6
```

### 5.3 Level 3 — Test Statistics and CLs

Golden-point tests compare hypothesis test outputs at multiple signal strength values:

```python
from nextstat import infer

for mu_test in [0.0, 0.5, 1.0, 2.0]:
    # pyhf
    cls_pyhf = float(pyhf.infer.hypotest(
        mu_test, data_pyhf, model_pyhf,
        test_stat="qtilde", calctype="asymptotics"
    ))
    # NextStat
    cls_ns = infer.hypotest(mu_test, model_ns)

    assert abs(cls_ns - cls_pyhf) < 5e-6
```

**Upper limits** are compared via two strategies:

| Strategy | NextStat | pyhf | Tolerance |
|----------|----------|------|-----------|
| Linear scan + interpolation | `upper_limit(mode="scan", points=201)` | `pyhf.infer.intervals.upper_limits.upper_limit` | $5 \times 10^{-5}$ |
| Bisection root-finding | `upper_limit(mode="rootfind", rtol=1e-4)` | — | $5 \times 10^{-4}$ vs scan |

Both observed and expected limits (5-element Brazil band) are validated.

### 5.4 Level 4 — Statistical Quality (Toy Ensembles)

Single-point parity is necessary but not sufficient. NextStat must not introduce systematic biases. Toy ensembles measure:

**Bias:**

$$\text{bias}(\hat{\theta}) = \frac{1}{N} \sum_{t=1}^{N} (\hat{\theta}_t - \theta_\text{true})$$

**Pull:**

$$\text{pull}_t = \frac{\hat{\theta}_t - \theta_\text{true}}{\hat{\sigma}_t}$$

Expected: $\text{mean}(\text{pull}) \approx 0$, $\text{std}(\text{pull}) \approx 1$.

**Coverage:**

$$\text{coverage}_{k\sigma} = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}\left[|\hat{\theta}_t - \theta_\text{true}| \leq k \cdot \hat{\sigma}_t\right]$$

Expected: $\text{coverage}_{1\sigma} \approx 0.683$.

**Tolerances (NextStat regression bounds):**

| Metric | Tolerance |
|--------|-----------|
| $|\Delta\,\text{mean}(\text{pull}_\mu)|$ | $\leq 0.05$ |
| $|\Delta\,\text{std}(\text{pull}_\mu)|$ | $\leq 0.05$ |
| $|\Delta\,\text{coverage}_{1\sigma}(\mu)|$ | $\leq 0.03$ |

Evaluated with $N_\text{toys} = 200$ (CI smoke tests) and $N_\text{toys} \geq 5000$ (certification).

**Upper limit coverage** is additionally validated: toy datasets are generated from Poisson at a known truth ($\mu_\text{true} = 1.0$), observed upper limits are computed for each toy, and the fraction of toys where $\mu_\text{up} \geq \mu_\text{true}$ is checked against the nominal $1 - \alpha = 95\%$ coverage.

### 5.5 Level 5 — Bayesian Sampler Diagnostics

For the NUTS/HMC sampler:

| Diagnostic | Threshold | Source |
|------------|-----------|--------|
| Split $\hat{R}$ | $< 1.1$ (test) / $< 1.01$ (production) | Cross-chain convergence |
| Divergence rate | $< 5\%$ (test) / $< 1\%$ (production) | Leapfrog integration quality |
| POI posterior mean | Finite, in $(0, 5)$ | Sanity check |
| ESS bulk / tail | $\geq 100$ per chain | Mixing quality |
| E-BFMI | $> 0.3$ | Energy transition efficiency |

Additionally:
- **Reproducibility**: same seed $\Rightarrow$ identical draws (verified per-element)
- **MAP consistency**: MAP with flat prior agrees with MLE within frequentist tolerances
- **Data override**: posterior changes when observed data is replaced (sensitivity check)

## 6. Test Suite Implementation

### 6.1 Test Fixtures

The validation suite uses five workspace files:

| Fixture | Purpose | Channels | Bins | Params |
|---------|---------|----------|------|--------|
| `simple_workspace.json` | Primary parity fixture | 1 | 2 | 3 |
| `complex_workspace.json` | Multi-channel, multi-modifier | 2 (SR + CR) | 4+4 | 18 |
| `bad_observations_length_mismatch.json` | Error handling | — | — | — |
| `bad_sample_length_mismatch.json` | Error handling | — | — | — |
| `bad_histosys_template_length_mismatch.json` | Error handling | — | — | — |

**Simple workspace.** One channel (`singlechannel`) with 2 bins. Signal sample `(5.0, 10.0)` with `normfactor` modifier `mu`. Background `(50.0, 60.0)` with `shapesys` modifier. Observations: `(53.0, 65.0)`.

**Complex workspace.** Signal region (`SR`) with signal + background, plus control region (`CR`) with background only. Modifiers include `normfactor`, `lumi` (2% uncertainty), `normsys`, `histosys`, `staterror`, and `shapefactor`. Tests interactions between all modifier types simultaneously.

**Generated workspaces.** The test module `test_pyhf_generated_workspaces.py` programmatically generates additional fixtures (shapefactor patterns, histosys+normsys+staterror combinations) to expand coverage beyond hand-crafted files.

### 6.2 Python Test Modules

| Module | Tests | Level | Speed |
|--------|-------|-------|-------|
| `test_pyhf_validation.py` | NLL & MLE parity | 1–2 | Fast |
| `test_hypotest_cls.py` | CLs, $q_\mu$, upper limits | 3 | Fast |
| `test_pyhf_generated_workspaces.py` | NLL parity on synthetic models | 1 | Fast |
| `test_bindings_api.py` | Python API contracts | — | Fast |
| `test_sampling.py` | NUTS shape, reproducibility, diagnostics | 5 | Mixed |
| `test_bayes_contract.py` | Bayesian helper contracts | — | Fast |
| `test_bias_pulls.py` | Toy pull distributions | 4 | Slow |
| `test_coverage_regression.py` | Upper limit coverage | 4 | Slow |
| `test_viz_contract.py` | Visualization artifacts | — | Fast |

### 6.3 Rust Test Suites

| File | Tests | Purpose |
|------|-------|---------|
| `cli_fit.rs` | 6 | CLI fit output contract, error handling |
| `cli_limits.rs` | 2 | Upper limit and hypotest CLI contracts |
| `mle.rs` (internal) | 4 | Toy fits: convergence, reproducibility, pull distribution |

### 6.4 Running the Suite

```bash
# Level 1–3: Fast parity tests (< 30 seconds)
pytest tests/python/test_pyhf_validation.py tests/python/test_hypotest_cls.py \
       tests/python/test_pyhf_generated_workspaces.py -v

# Level 5: NUTS diagnostics (~ 2 minutes)
pytest tests/python/test_sampling.py -v

# Level 4: Toy pull distributions (opt-in, ~ 5 minutes)
NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 pytest tests/python/test_bias_pulls.py -v -m slow

# Level 4: Coverage regression (opt-in, ~ 10 minutes)
NS_RUN_SLOW=1 NS_TOYS=20 NS_SEED=0 NS_SCAN_POINTS=81 \
    pytest tests/python/test_coverage_regression.py -v -m slow

# Rust unit tests (excludes ns-py)
cargo test -p ns-core -p ns-ad -p ns-compute -p ns-translate -p ns-inference

# Rust toy pull distribution (requires release mode)
cargo test -p ns-inference --release test_fit_toys_pull_distribution -- --ignored
```

### 6.5 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NS_RUN_SLOW` | `0` | Enable slow (Level 4) tests |
| `NS_TOYS` | `200` | Number of toys for pull/coverage tests |
| `NS_SEED` | `0` | Base RNG seed for reproducibility |
| `NS_FIXTURES` | `simple,complex` | Comma-separated fixture list |
| `NS_SCAN_POINTS` | `81` | Grid points for upper limit scan |

## 7. Determinism Guarantees

### 7.1 Seed Management

For stochastic operations (toy generation, MCMC sampling):

- Seeds are explicit and user-provided (never auto-generated)
- Multi-chain NUTS uses `chain_seed = base_seed + chain_id`
- The test `test_same_seed_same_draws()` verifies bit-exact reproducibility: two runs with the same seed produce element-wise identical posterior samples

### 7.2 Toolchain Pinning

For strict reproducibility across builds:

- `rust-toolchain.toml` pins the compiler version
- `Cargo.lock` locks all dependency versions
- MSRV (Minimum Supported Rust Version) is 1.93

### 7.3 Init Jitter

NUTS initialization supports a jitter parameter (`init_jitter`) for exploring initialization sensitivity. Default is `0.0` (no jitter), ensuring deterministic initialization from `suggested_init()`. When jitter is nonzero, the test suite verifies that `init_jitter` and `init_jitter_rel` are mutually exclusive.

## 8. Reproducibility Levels

NextStat defines three levels of reproducibility:

| Level | Scope | Guarantee | Mechanism |
|-------|-------|-----------|-----------|
| **Strict** | Same platform, same binary, `--threads 1` | Bit-exact | Deterministic mode |
| **Portable** | Different platforms, `--threads 1` | Within parity tolerances | Scalar `f64` math |
| **Statistical** | Multi-threaded, GPU | Within relaxed tolerances | Performance mode |

### 8.1 Strict Reproducibility

In deterministic mode with pinned toolchain, locked dependencies, and single-threaded execution, NextStat produces bit-identical results across runs on the same platform. This is verified by the `test_fit_toys_reproducible()` and `test_same_seed_same_draws()` tests.

### 8.2 Portable Reproducibility

Across platforms (Linux x86_64, macOS ARM64), small differences may arise from math library implementations (`libm` vs platform-specific `ln`/`exp`). These remain well within the parity tolerances and do not affect inference conclusions.

### 8.3 Statistical Reproducibility

In performance mode (multi-threaded), floating-point reduction order may vary between runs. Results remain statistically equivalent but are not bit-exact. Toy ensembles with performance mode must use sufficiently large $N_\text{toys}$ to absorb this variation.

## 9. Summary

NextStat provides an independent implementation of HistFactory likelihoods that maintains strict numerical parity with pyhf. The validation methodology covers five levels:

| Level | What | Tolerance |
|-------|------|-----------|
| 1 | Likelihood evaluation | $10^{-10}$ |
| 2 | MLE parameters and uncertainties | $2$–$5 \times 10^{-4}$ |
| 3 | CLs, $q_\mu$, upper limits | $5 \times 10^{-6}$–$5 \times 10^{-4}$ |
| 4 | Toy pull/coverage statistics | $0.03$–$0.05$ |
| 5 | NUTS diagnostics ($\hat{R}$, ESS, divergences) | Pass/fail thresholds |

The test suite is designed for practical CI integration: fast parity tests run in under 30 seconds, while slow certification tests (toy ensembles, coverage) are gated behind environment variables for release validation.

The combination of implementation independence (no shared code with pyhf), documented precision guarantees, and multi-level validation establishes NextStat as a tool for reproducible statistical inference.

## References

[1] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[2] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[3] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[4] E. Maguire, L. Heinrich, G. Watt, "HEPData: a repository for high energy physics data," J. Phys. Conf. Ser. 898, 102006, 2017.

[5] NextStat Contributors, "NextStat Model Specification: HistFactory Probability Densities and Modifiers," docs/papers/model-specification.md, 2026.

[6] NextStat Contributors, "NextStat: High-Performance Statistical Inference for Binned Likelihood Models," docs/papers/nextstat-software.md, 2026.

[7] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011; Erratum ibid. C73, 2501, 2013.

[8] M. D. Hoffman, A. Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo," JMLR 15, 1593–1623, 2014.
