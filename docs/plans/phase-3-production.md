# Phase III: Production Ready (P1)

> **Execution note (humans + AI agents):** execute tasks sequentially. Canonical definitions / tolerances / determinism: `docs/plans/standards.md`.

**Goal:** bring NextStat to a production-ready state: stable APIs, release discipline, documentation, visualization, expanded validation, and pilot readiness.

**Duration:** Months 9-15

**Dependencies:** Phases 0-2 (especially Phase 2A CPU performance and Phase 2B AD/gradients).

**Tech Stack:** Rust, PyO3/maturin, tracing, Plotly/Matplotlib (Python), Sphinx/MkDocs, GitHub Actions releases.

---

## Contents

- [Acceptance Criteria (Phase)](#acceptance-criteria-phase)
- [Sprint 3.1: Frequentist limits (P0)](#sprint-31-frequentist-limits-p0-weeks-37-40)
- [Sprint 3.2: Bayesian interface (P1, optional)](#sprint-32-bayesian-interface-p1-optional-weeks-41-44)
- [Sprint 3.3: Production hardening](#sprint-33-production-hardening-weeks-45-48)
- [Sprint 3.4: Visualization](#sprint-34-visualization-weeks-49-52)
- [Sprint 3.5: Documentation](#sprint-35-documentation-weeks-53-60)
- [Sprint 3.6: Validation Suite v1](#sprint-36-validation-suite-v1-weeks-61-68)
- [Phase completion criteria](#phase-completion-criteria)

---

## Acceptance Criteria (Phase)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Release discipline | semver + changelog | tag `v0.x.y` + `CHANGELOG.md` |
| Wheels | Linux/macOS wheels | `pip install nextstat` without compilation |
| Deterministic parity | stable in CI | `pytest tests/python/test_pyhf_validation.py` |
| Bias / coverage regression | matches pyhf on toys | `NS_RUN_SLOW=1 pytest -m slow tests/python/test_bias_pulls.py` + report artifacts |
| Limits (CLs) | matches pyhf (asymptotics) | `pytest -k hypotest tests/python` (golden points) |
| Profile likelihood | `q(mu)` matches pyhf | `pytest -k profile tests/python` (golden points) |
| Bayesian sampling (NUTS/HMC) | Rust core sampler + diagnostics | `cargo test -p ns-inference` (sampler golden tests) |
| Viz outputs | pulls/ranking/corr | golden image/json tests |
| Docs | "hello fit -> ranking" | docs build green + tutorials |

> Bayesian sampling is implemented as **NUTS/HMC in the Rust core** (`ns-inference`). It remains P1/optional for Phase 3 completion, but the contract and golden tests should be locked early (see `docs/plans/standards.md` section 7).

---

## Sprint 3.1: Frequentist limits (P0) (Weeks 37-40)

Sprint goal: complete the canonical HEP workflow for **upper limits**:

- profile likelihood scan (fix POI, profile nuisance parameters)
- asymptotic **CLs** (pyhf-compatible, `test_stat="qtilde"`)
- convenient surfaces (CLI/Python) + parity tests

### Epic 3.1.1: Profile likelihood scan (q_mu / qtilde_mu)

#### Task 3.1.1.1: Fast minimization path (skip Hessian)

**Priority:** P0  
**Goal:** support many repeated minimizations (scan/limits) quickly, without computing covariance/Hessian.

**Acceptance Criteria:**
- [ ] Provide an API/internal helper: minimize NLL without Hessian
- [ ] Do not break `FitResult` and existing `nextstat.fit(...)`
- [ ] Determinism: same inputs yield the same minimum

#### Task 3.1.1.2: Scan API + q(mu)

**Acceptance Criteria:**
- [ ] Unconditional fit: `(mu_hat, nll_hat)`
- [ ] Conditional fit: `nll(mu)` with POI fixed
- [ ] Return a scan result structure: `mu, q, nll, converged, n_iter`
- [ ] Golden parity: `q(mu)` matches pyhf on `simple_workspace.json` at multiple points

### Epic 3.1.2: Asymptotic CLs limits (pyhf hypotest compatibility)

#### Task 3.1.2.1: Hypotest (asymptotics, qtilde)

**Acceptance Criteria:**
- [ ] Implement `hypotest(mu_test)` with `test_stat="qtilde"`, `calctype="asymptotics"`
- [ ] Return `CLs`, `CLs+b`, `CLb` and internal `qtilde_obs`/`qtilde_A`
- [ ] Golden parity: values match pyhf for `simple_workspace.json` (mu=0, 0.5, 1, 2)

#### Task 3.1.2.2: Upper limit finder (CLs=alpha)

**Acceptance Criteria:**
- [ ] `upper_limit(alpha=0.05)` (bracket + root find) for fixed model/data
- [ ] Robust on reasonable workspaces (simple/complex fixtures)

### Epic 3.1.3: Surfaces + docs

#### Task 3.1.3.1: CLI + Python surface

**Acceptance Criteria:**
- [ ] CLI: `nextstat hypotest ...` and `nextstat upper-limit ...` with JSON output
- [ ] Python: `nextstat.infer.hypotest(...)`, `nextstat.infer.upper_limit(...)`, `nextstat.infer.scan(...)`

---

## Sprint 3.2: Bayesian interface (P1, optional) (Weeks 41-44)

### Epic 3.2.1: Posterior contract + transforms

**Goal:** lock the posterior logpdf/grad interface without committing to "writing NUTS immediately".

> Source of truth: `docs/plans/standards.md` section 7 (Bayesian / Posterior contract).

#### Task 3.2.1.1: `Posterior` API (logpdf + grad) (required for NUTS)

**Priority:** P1  
**Effort:** 6-10 hours  
**Dependencies:** Phase 2B (analytic gradients available in the CPU reference path)

**Files:**
- Create: `crates/ns-inference/src/posterior.rs`
- Create: `crates/ns-inference/src/transforms.rs` (bijectors + log|J| + grad chain-rule)
- Modify: `crates/ns-core/src/traits.rs` (add a trait for logpdf/grad or an adapter over `ComputeBackend`)

**Acceptance Criteria:**
- [ ] Posterior defines prior terms for POI/NP (minimum: flat for POI, Normal for constrained NPs)
- [ ] `posterior_logpdf = model.logpdf + prior_terms` (canonical)
- [ ] `posterior_grad` uses `Model::gradient()` when available
- [ ] With flat prior, MAP matches MLE (sanity contract in `docs/plans/standards.md`)
- [ ] Bounded parameters are correctly mapped to `z in R^n` (see standards section 7.3), including `log|J|`

**Implementation sketch (pseudocode; trait names may differ):**

```rust
// crates/ns-inference/src/posterior.rs
use ns_core::types::Float;
use ns_core::Model;

pub struct Posterior<'a, M: Model> {
    model: &'a M,
}

impl<'a, M: Model> Posterior<'a, M> {
    pub fn new(model: &'a M) -> Self {
        Self { model }
    }

    pub fn logpdf(&self, x: &[Float]) -> ns_core::Result<Float> {
        let lp = self.model.logpdf(x)?;
        Ok(lp + self.prior_logpdf(x))
    }

    pub fn grad(&self, x: &[Float]) -> ns_core::Result<Vec<Float>> {
        // Phase 2B provides `Model::gradient()`.
        let mut g = self.model.gradient(x)?;
        let pg = self.prior_grad(x);
        for (gi, pgi) in g.iter_mut().zip(pg.iter()) {
            *gi += *pgi;
        }
        Ok(g)
    }

    fn prior_logpdf(&self, _x: &[Float]) -> Float {
        // Minimal default: flat priors for POI, no extra priors for NP (constraints are already in model.logpdf).
        0.0
    }

    fn prior_grad(&self, x: &[Float]) -> Vec<Float> {
        vec![0.0; x.len()]
    }
}
```

#### Task 3.2.1.2: Transform correctness tests (golden)

**Priority:** P1  
**Effort:** 4-8 hours  
**Dependencies:** Task 3.2.1.1

**Files:**
- Create: `crates/ns-inference/src/transforms_tests.rs`

**Acceptance Criteria:**
- [ ] `inv(transform(z)) ~= z` (rtol 1e-10) for all bijectors
- [ ] `grad_z(log|J|)` matches finite differences (rtol 1e-7)
- [ ] Chain-rule gradient `d/dz log p(theta(z))` matches finite differences on simple functions

---

### Epic 3.2.2: HMC kernel (leapfrog)

#### Task 3.2.2.1: Leapfrog integrator + accept/reject (static HMC)

**Priority:** P1  
**Effort:** 10-16 hours  
**Dependencies:** Epic 3.2.1 (posterior logpdf+grad in unconstrained space)

**Files:**
- Create: `crates/ns-inference/src/hmc.rs`
- Create: `crates/ns-inference/src/hmc_tests.rs`

**Acceptance Criteria:**
- [ ] For 1D Normal, static HMC yields mean ~= 0, var ~= 1 (N=5k, burn-in=1k)
- [ ] Deterministic behavior with fixed seed
- [ ] Diagnostics: acceptance rate and energy error available in results

---

### Epic 3.2.3: NUTS + adaptation (core)

#### Task 3.2.3.1: Dual averaging step-size adaptation

**Priority:** P1  
**Effort:** 6-10 hours  
**Dependencies:** Epic 3.2.2

**Files:**
- Create: `crates/ns-inference/src/adapt.rs`

**Acceptance Criteria:**
- [ ] `target_accept` is reached during warmup (+/- 0.05) on an MVN toy
- [ ] Step size is fixed after warmup and reproducible

#### Task 3.2.3.2: NUTS tree building (Stan-style multinomial)

**Priority:** P1  
**Effort:** 14-24 hours  
**Dependencies:** Task 3.2.3.1

**Files:**
- Create: `crates/ns-inference/src/nuts.rs`
- Create: `crates/ns-inference/src/nuts_tests.rs`

**Acceptance Criteria:**
- [ ] MVN(dim=4): mean/cov within statistical error vs analytic
- [ ] `max_treedepth` guard + diagnostics (treedepth saturations)
- [ ] Divergences are counted and reported

#### Task 3.2.3.3: Diagonal mass matrix adaptation (warmup windows)

**Priority:** P1  
**Effort:** 8-14 hours  
**Dependencies:** Task 3.2.3.2

**Acceptance Criteria:**
- [ ] For MVN with unbalanced scales, ESS/sec improves significantly vs unit mass
- [ ] Mass matrix is fixed after warmup; final diag mass is saved in draws

---

### Epic 3.2.4: Multi-chain runner + diagnostics

#### Task 3.2.4.1: Multi-chain + R-hat/ESS (baseline)

**Priority:** P1  
**Effort:** 8-12 hours  
**Dependencies:** Epic 3.2.3

**Files:**
- Create: `crates/ns-inference/src/diagnostics.rs`
- Create: `crates/ns-inference/src/chain.rs` (draws + stats struct)

**Acceptance Criteria:**
- [ ] 4-chain runner (Rayon optional) is deterministic given seeds
- [ ] R-hat/ESS computed for test distributions; `R-hat < 1.01` on simple tasks

#### Task 3.2.4.2: Python surface for draws (optional)

**Priority:** P2  
**Effort:** 4-8 hours  
**Dependencies:** Task 3.2.4.1

**Goal:** export draws in a numpy/arrow-friendly format + analyze in ArviZ.

---

## Sprint 3.3: Production hardening (Weeks 45-48)

### Epic 3.3.1: Release + observability

#### Task 3.3.1.1: Release pipeline (maturin + GitHub Actions)

**Priority:** P0  
**Effort:** 6-10 hours  
**Dependencies:** Phase 1.4 (Python bindings) + Phase 0 CI

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `CHANGELOG.md`
- Modify: `pyproject.toml` (versioning policy)

**Acceptance Criteria:**
- [ ] `release.yml` builds wheels for Linux/macOS and attaches them to the GitHub Release
- [ ] Manual release (workflow_dispatch) + tag-based release
- [ ] `CHANGELOG.md` is updated before a release

#### Task 3.3.1.2: Structured logging (`tracing`) end-to-end

**Priority:** P0  
**Effort:** 4-6 hours  
**Dependencies:** Phase 1 CLI

**Files:**
- Modify: `crates/ns-cli/src/main.rs`
- Modify: `crates/ns-core/src/error.rs` (error contexts)

**Acceptance Criteria:**
- [ ] CLI supports `--log-level` and prints structured logs
- [ ] Errors include context (which file, which backend, which POI)

---

## Sprint 3.4: Visualization (Weeks 49-52)

### Epic 3.4.1: Plots for HEP workflows

#### Task 3.4.1.1: Pull plot + ranking plot (Python)

**Priority:** P0  
**Effort:** 8-12 hours  
**Dependencies:** Phase 1.5 validation outputs + Phase 2C ranking optimization

**Files:**
- Create: `bindings/ns-py/python/nextstat/viz/pulls.py`
- Create: `bindings/ns-py/python/nextstat/viz/ranking.py`
- Test: `tests/python/test_viz_outputs.py`

**Acceptance Criteria:**
- [ ] `plot_pulls(result)` and `plot_ranking(entries)` return a Plotly figure
- [ ] Golden tests: JSON snapshot (no pixel-flakes)

#### Task 3.4.1.2: Correlation matrix plot

**Priority:** P1  
**Effort:** 6-8 hours  
**Dependencies:** correlation extraction available (Phase 2B/3.3)

---

## Sprint 3.5: Documentation (Weeks 53-60)

### Epic 3.5.1: Docs site + tutorials

#### Task 3.5.1.1: Build docs (Sphinx/MkDocs) + API reference

**Priority:** P0  
**Effort:** 6-10 hours  
**Dependencies:** stable public API

**Acceptance Criteria:**
- [ ] Docs build in CI
- [ ] Tutorial: `load → fit → ranking → scan`

---

## Sprint 3.6: Validation Suite v1 (Weeks 61-68)

### Epic 3.6.1: Expand fixtures + regression tests

#### Task 3.6.1.1: Fixture pack “small/medium/large”

**Priority:** P0  
**Effort:** 8-12 hours  
**Dependencies:** Phase 1 parity suite

**Acceptance Criteria:**
- [ ] >= 10 workspaces covering modifier combinations
- [ ] Stored references: bestfit/uncertainties/Δtwice_nll
- [ ] CI matrix runs deterministic parity

---

### Epic 3.6.2: Bias & Coverage certification (toys)

Goal: control **bias and coverage** on toy ensembles and ensure NextStat does not diverge from pyhf not only at single workspaces but also statistically.

> Important: this is a **slow** workload. Do not add it to the PR matrix. Run via nightly/manual workflow + publish reports as CI artifacts.

#### Task 3.6.2.1: Pull/bias regression vs pyhf (toy ensembles)

**Priority:** P1  
**Effort:** 6-10 hours  
**Dependencies:** Phase 1 parity suite + stable `nextstat.fit(..., data=...)` API

**Files:**
- Use/extend: `tests/python/test_bias_pulls.py` (Phase 1 adds a smoke version; opt-in via `NS_RUN_SLOW=1`)
- Create: `reports/bias_coverage/README.md`
- (optional) Create: `.github/workflows/toys-validation.yml` (nightly/manual)

**Acceptance Criteria:**
- [ ] For the fixture pack (minimum: `simple`, `complex`), compute pulls for `mu` and key constrained NPs.
- [ ] Summary metrics NextStat vs pyhf are within tolerances (see `docs/plans/standards.md` section 6).
- [ ] Generate a JSON report (mean/std/coverage) and save it as an artifact.

#### Task 3.6.2.2: Coverage checks for 1σ / 2σ intervals (baseline)

**Priority:** P1  
**Effort:** 4-8 hours  
**Dependencies:** Task 3.6.2.1

**Notes / decisions:**
- Baseline intervals in Phase 3: `mu_hat +/- 1 sigma_hat` and `mu_hat +/- 2 sigma_hat` (Hessian-based), fast and stable.
- Later (Phase 3/4) add profile-likelihood intervals and coverage for them (slower but more canonical).

**Acceptance Criteria:**
- [ ] Coverage(68%) and Coverage(95%) compared to pyhf on the toy ensemble.
- [ ] Coverage difference NextStat vs pyhf <= 1-3% (tunable; depends on `N_TOYS`).

---

## Phase completion criteria

Phase 3 is complete when:

1. [ ] Release pipeline ships wheels and changelog (P0)
2. [ ] Documentation and tutorials build in CI (P0)
3. [ ] Pull/ranking/correlation visualizations are available from Python (P0/P1)
4. [ ] Validation suite v1 is stable (determinism + regressions) (P0)
5. [ ] (P1) Bias/coverage regression vs pyhf is documented and reproducible (toys report)
