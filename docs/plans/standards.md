# NextStat Project Standards (Source of Truth)

This document defines the cross-phase standards for NextStat: canonical definitions (`twice_nll`), tolerances, precision policy, and determinism requirements.

If any phase plan conflicts with this document, this document wins.

## 0) Source of Truth

- `docs/plans/README.md`: navigation, timeline, high-level goals
- `docs/plans/2026-02-05-nextstat-implementation-plan.md`: master plan (cross-phase structure)
- `docs/plans/phase-*.md`: executable phase plans (epics/tasks/code/commands)
- `docs/references/*`: external research and notes (if referenced by plans, it must be available)

## 1) Terms and Math (Canonical)

### 1.1 `logpdf`, `nll`, `twice_nll`

- `logpdf(theta)`: log-likelihood including constraint terms, consistent with `pyhf.Model.logpdf`
- `nll(theta) = -logpdf(theta)`
- `twice_nll(theta) = 2 * nll(theta) = -2 * logpdf(theta)`

Important: `twice_nll` is used for parity against pyhf and for reporting. Hot-path optimizations may omit constants, but the parity contract must reflect this explicitly (e.g., test deltas, not absolute values).

### 1.2 Poisson term (binned)

For one bin:

- `log P(n | lambda) = n * ln(lambda) - lambda - ln Gamma(n + 1)`
- `poisson_nll(n, lambda) = lambda - n * ln(lambda) + ln Gamma(n + 1)`
- `poisson_twice_nll = 2 * poisson_nll`

`Gamma` is the gamma function (generalized factorial: `Gamma(n+1) = n!` for integer `n`).

Contract (Phase 1, deterministic f64): `twice_nll` must match `float(model.logpdf(pars, data)[0] * -2)` from pyhf (numpy backend) for the fixture suite.

### 1.3 Gaussian constraint term

For a constraint `x ~ Normal(mu, sigma)`:

- `log N(x | mu, sigma) = -0.5 * z^2 - ln(sigma) - 0.5 * ln(2*pi)`, where `z = (x-mu)/sigma`
- `gaussian_nll = 0.5 * z^2 + ln(sigma) + 0.5 * ln(2*pi)`
- `gaussian_twice_nll = z^2 + 2 ln(sigma) + ln(2*pi)`

If constants are intentionally omitted for performance, this must be a separate mode/function and tested as `Delta twice_nll` relative to a reference point (e.g., best-fit), not as an absolute value.

## 2) Precision Policy

### 2.1 Types

- CPU reference/validation path: `f64` everywhere, stable reductions
- GPU path: mixed precision is allowed (e.g., `f32` compute + `f64` accumulation), but parity tests must be evaluated via the CPU reference path

### 2.2 Accuracy targets

- Deterministic CPU parity (Phase 1 contract):
  - `twice_nll`: `rtol=1e-6`, `atol=1e-8` (strict in deterministic mode only)
  - `bestfit` (fixtures, compare by parameter names): `atol=2e-4`
  - `uncertainties` (fixtures, compare by parameter names): `atol=5e-4`
- Performance modes (Rayon / GPU):
  - softer tolerances are allowed (must be specified per phase)
  - prefer comparing `Delta twice_nll` relative to best-fit

All tolerances must be defined in one place in test code (constants), and plans should reference those constants instead of duplicating numbers.

## 3) Determinism (Required for Parity)

### 3.1 Requirement

Parity vs pyhf must hold in:

- `backend=cpu`
- `threads=1` or a deterministic parallel reduce
- fixed summation order
- fixed RNG seeds

### 3.2 Rayon and reductions

If Rayon is used, there must be a deterministic mode which either:

- forces sequential execution, or
- uses a fixed reduction order (pairwise/tree reduction with deterministic chunking)

## 4) Testing and Coverage

### 4.1 Test matrix

- Unit tests (Rust)
- Property tests (bounded runtime; fix seeds when investigating flakes)
- Python validation against pyhf (deterministic CPU)

### 4.2 Coverage

Preferred Rust coverage tool: `cargo llvm-cov`.

## 5) "Executable Plan" Standards

- Code blocks in plans must be compilable/valid (or explicitly marked as pseudocode).
- The phrase "Write failing test" is reserved for real test harnesses.
  - For expected-to-fail shell checks, use "Sanity check (expected to fail)".
- TODOs/placeholders in plans must either:
  - be moved into a backlog section, or
  - have explicit acceptance criteria and a link to a task/epic.

### 5.1 Accepted engineering decisions (Phase 1-2)

These are intentional short-term deviations from the ideal architecture to keep the early system simple and shippable:

1. `ns-inference` depends directly on `ns-translate::HistFactoryModel`.
   - Rationale: currently there is only one model type (HistFactory). Introduce an abstract model trait when a second model type appears.

2. `FitResult.n_evaluations` equals `n_iter` (argmin limitation).
   - A true `n_fev` requires explicit cost/grad call counters. Keep as-is until Phase 3 polish, then either rename the field or add `n_fev`.

## 6) Bias / Coverage (Frequentist QA)

NextStat's baseline mode is frequentist (MLE, profile likelihood, asymptotics). This does not guarantee unbiased estimates under all conditions (small counts, parameter bounds, non-linearities).

### 6.1 Terms (canonical)

- Bias: `bias(theta_hat) = E[theta_hat] - theta_true`
- Pull (when an uncertainty estimate exists): `pull = (theta_hat - theta_true) / sigma_hat`
  - expected in a "good" regime: `mean(pull) ~= 0`, `std(pull) ~= 1`
- Coverage: fraction of toy experiments where an interval (e.g., `theta_hat +/- 1 sigma_hat`) contains `theta_true`

### 6.2 Policy

1. Phase 1 contract is parity vs pyhf, not "magical bias removal".
   - If pyhf has systematic bias/undercoverage for specific models, NextStat must not "fix" it silently.

2. We must measure bias/coverage on toy ensembles and ensure NextStat does not introduce additional bias relative to pyhf.

3. Any bias correction (if introduced) must be:
   - explicit (off by default),
   - toy-validated,
   - documented for edge cases (bounds, constraints).

### 6.3 Quality gates (by phase)

- Phase 1 (smoke, opt-in): toy pull/coverage regression vs pyhf on `tests/fixtures/*`.
  - Compare `mean/std(pull)` and coverage for the POI (usually `mu`) between NextStat and pyhf.
  - Fix seeds to avoid flakes.
  - Keep it lightweight (e.g., `N_TOYS=200`).
  - Suggested tolerances (difference vs pyhf):
    - `|Delta mean(pull_mu)| <= 0.05`
    - `|Delta std(pull_mu)| <= 0.05`
    - `|Delta coverage_1sigma(mu)| <= 0.03`

- Phase 3 (certification): expanded bias/coverage report (nightly/manual).
  - For fixture packs "small/medium/large" run `N_TOYS >= 5000` (offline or scheduled).
  - Artifact: JSON report + plots (publish as CI artifacts).

## 7) Bayesian / Posterior Contract (Optional)

NextStat aims to support both frequentist and Bayesian workflows, but the Bayesian layer must not break the frequentist parity baseline.

### 7.1 Terms (canonical)

- Likelihood: `L(data | theta)`; `log L` may include auxiliary/constraint terms (as in HistFactory/pyhf).
- Prior: `p(theta)`; provided explicitly (primarily for POI and unconstrained nuisance parameters).
- Posterior: `p(theta | data) proportional_to L(data | theta) * p(theta)`.
  - `log posterior = log L + log prior + const`

### 7.2 Policy decisions

1. Constraints are not treated as an extra prior by default.
   - Phase 1/2 contract: `model.logpdf` / `model.nll` already includes constraints (pyhf parity).
   - In Bayesian mode we add only an additional prior to avoid double-counting.

2. Phase 3 goal: HMC/NUTS in Rust core (`ns-inference`).
   - Minimum: correct `posterior_logpdf(params)` and `posterior_grad(params)` (via Phase 2B AD).
   - Python sampling is acceptable for cross-checks/diagnostics, not as the primary runtime.

3. MAP sanity contract:
   - With a flat prior (or no additional prior), MAP should match MLE within frequentist tolerances.

### 7.3 Algorithmic baseline (MVP)

To make HMC/NUTS correct and robust with bounded parameters:

- Unconstrained parameterization:
  - Sample `z in R^n`, transform to `theta = transform(z)` accounting for bounds.
  - Add `log|det J|` (Jacobian) to the log posterior.
  - Bijectors:
    - `(-inf, inf)`: identity
    - `(0, inf)`: `exp(z)`
    - `(a, inf)`: `a + exp(z)`
    - `(a, b)`: `a + (b-a) * sigmoid(z)`

- Integrator: leapfrog (symplectic) + track energy error in diagnostics.
- NUTS variant: Stan-style multinomial NUTS with "no U-turn" stop criterion.
- Warmup/adaptation (Phase 3 baseline):
  - Dual averaging for step size (`target_accept` default 0.8).
  - Mass matrix: diagonal (online variance), windowed adaptation (simplified Stan-style).
- Determinism: fixed seeds and independent chains (`seed = base + chain_id`).

### 7.4 Quality gates

- Phase 3 (optional):
  - A unified Posterior API (Rust + Python surface) that:
    - does not double-count constraints,
    - accepts seed/chain id for reproducibility,
    - runs NUTS/HMC in Rust core and returns draws + diagnostics.
  - Minimal "golden" tests:
    - 1D Normal: mean ~ 0, var ~ 1
    - MVN (dim=4): mean/cov within statistical error
    - Banana/funnel stress tests: no exploding energy; track divergences and treedepth
  - Reference quality targets:
    - R-hat < 1.01 on toy problems
    - divergence rate < 1% (or a documented explanation)
    - acceptable ESS per parameter and per second (baseline for regressions)

