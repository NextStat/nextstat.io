# pyhf Parity Contract: Tolerance Tiers

> Source of truth for numerical tolerances between NextStat and pyhf (NumPy backend).
> Referenced by `tests/python/_tolerances.py`.

---

## Overview

NextStat uses pyhf's NumPy backend (f64, deterministic) as the **canonical oracle** for all
numerical parity testing. Two evaluation modes govern the trade-off between precision and speed:

| Mode | Summation | Backend | Threads | Use Case |
|------|-----------|---------|---------|----------|
| **Parity** | Kahan compensated | SIMD (Accelerate disabled) | 1 (forced) | Validation vs pyhf |
| **Fast** | Naive | SIMD / Accelerate / CUDA | Rayon (auto) | Production inference |

### Activating Parity Mode

**CLI:**
```bash
nextstat fit --parity model.json
```

**Python:**
```python
import nextstat
nextstat.set_eval_mode("parity")
```

**Rust:**
```rust
ns_compute::set_eval_mode(ns_compute::EvalMode::Parity);
```

When Parity mode is activated:
1. `EvalMode::Parity` is set process-wide (atomic flag)
2. Apple Accelerate is automatically disabled
3. Thread count is forced to 1 (sequential Rayon)
4. Kahan compensated summation is used for Poisson NLL

---

## Tolerance Tiers

### Tier 1: Per-Bin Expected Data (Pure Arithmetic)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `abs(ns_bin - pyhf_bin)` | **1e-12** | 1e-10 | Identical f64 arithmetic; no reduction noise |

**What:** `model.expected_data(params, include_auxdata=False)` compared bin-by-bin.

**Why near-exact:** Expected data computation is a sequence of multiplications and additions
with no reduction over bins. The same f64 operations in the same order produce identical results.

**Test:** `test_expected_data_per_bin_golden` in `test_expected_data_parity.py`

**Python constant:** `EXPECTED_DATA_PER_BIN_ATOL = 1e-12`

---

### Tier 2: Expected Data Vector (Accumulated)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `max(abs(ns - pyhf))` (all bins) | **1e-8** | 1e-8 | pytest.approx on full vector |

**What:** Full expected data vector (main + auxdata) compared with `pytest.approx`.

**Test:** `test_expected_data_matches_pyhf_at_suggested_init` in `test_expected_data_parity.py`

**Python constant:** `EXPECTED_DATA_ATOL = 1e-8`

---

### Tier 3: NLL Value (Scalar Reduction)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `abs(2*ns_nll - pyhf_twice_nll)` | **1e-8** (atol) | 1e-6 (rtol) | Sum over O(1000) bins accumulates rounding |

**What:** Negative log-likelihood evaluated at identical parameter values.

**Why differs from Tier 1:** NLL requires summing `λ - n·ln(λ) + ln(n!)` over all bins.
Summation order and compensated vs naive accumulation affect the result at ~1e-10 level
for typical models (~1000 bins). Kahan summation in Parity mode reduces this to ~1e-14.

**Test:** `test_simple_nll_parity_nominal_and_poi` in `test_pyhf_validation.py`

**Python constants:**
```python
TWICE_NLL_RTOL = 1e-6
TWICE_NLL_ATOL = 1e-8
```

---

### Tier 4: Gradient (AD vs Finite-Difference)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `abs(ns_grad_i - pyhf_fd_i)` | **1e-6** (atol) + **1e-4** (rtol) | Same | FD noise dominates, not mode |

**What:** NextStat reverse-mode AD gradient vs pyhf central finite-difference gradient (h=1e-5).

**Why looser:** AD produces exact derivatives (to machine precision). Finite-difference
has O(h^2) truncation error plus O(ε/h) cancellation error. With h=1e-5, the FD error
floor is ~1e-6 for well-conditioned functions. The tolerance reflects FD noise, not NextStat error.

**Test:** `test_gradient_parity_simple`, `test_gradient_parity_complex` in `test_pyhf_validation.py`

**Python constants:**
```python
GRADIENT_ATOL = 1e-6
GRADIENT_RTOL = 1e-4
```

---

### Tier 5: Best-Fit Parameters (Optimizer Surface)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `abs(ns_hat_i - pyhf_hat_i)` | **2e-4** | 2e-4 | Flat NLL surface near minimum |

**What:** MLE best-fit parameter values, compared by parameter name.

**Why looser:** Near the NLL minimum, the surface is flat (by definition). Small differences
in gradient accuracy or stopping criteria can shift the minimum by O(1e-4) in parameter space
while changing NLL by O(1e-10). On large models (>100 params), NextStat's L-BFGS-B may find
a *lower* NLL than pyhf's SLSQP (see `docs/references/optimizer-convergence.md`).

**Test:** `test_simple_mle_parity_bestfit_uncertainties` in `test_pyhf_validation.py`

**Python constant:** `PARAM_VALUE_ATOL = 2e-4`

---

### Tier 6: Parameter Uncertainties (Hessian Sensitivity)

| Metric | Parity Tolerance | Fast Tolerance | Rationale |
|--------|-----------------|----------------|-----------|
| `abs(ns_unc_i - pyhf_unc_i)` | **5e-4** | 5e-4 | Hessian ∝ ∂²NLL/∂θ² is noisy |

**What:** Parameter uncertainties (sqrt of diagonal of inverse Hessian).

**Why loosest:** Uncertainties depend on the inverse Hessian at the minimum. The Hessian
is a second derivative, amplifying numerical noise. Additionally, if the minimum differs
slightly (Tier 5), the Hessian is evaluated at a different point, compounding the difference.

**Test:** `test_simple_mle_parity_bestfit_uncertainties` in `test_pyhf_validation.py`

**Python constant:** `PARAM_UNCERTAINTY_ATOL = 5e-4`

---

### Tier 7: Toy Ensemble (Statistical)

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| `abs(Δmean(pull_mu))` | **0.05** | Statistical noise from finite N_toys |
| `abs(Δstd(pull_mu))` | **0.05** | Same |
| `abs(Δcoverage_1σ)` | **0.03** | Binomial noise |
| `abs(Δcoverage)` | **0.05** | General coverage |

**What:** Toy-experiment ensemble properties: pull distribution mean/width, coverage.

**Why statistical:** These are Monte Carlo estimates from N_toys pseudo-experiments.
The tolerance reflects expected statistical fluctuation at N_toys=200, not systematic error.

**Python constants:**
```python
PULL_MEAN_DELTA_MAX = 0.05
PULL_STD_DELTA_MAX = 0.05
COVERAGE_1SIGMA_DELTA_MAX = 0.03
COVERAGE_DELTA_MAX = 0.05
```

---

## Tolerance Hierarchy (Visual Summary)

```
Tighter ◄────────────────────────────────────────────── Looser

1e-12     1e-10     1e-8      1e-6      1e-4      1e-2
  │         │         │         │         │         │
  ├─ Tier 1: per-bin expected_data (1e-12)
  │         │         │         │         │
  │         ├─ Tier 2: expected_data vector (1e-8)
  │         │         │         │         │
  │         │         ├─ Tier 3: NLL value (atol 1e-8, rtol 1e-6)
  │         │         │         │         │
  │         │         │         ├─ Tier 4: gradient (atol 1e-6, rtol 1e-4)
  │         │         │         │         │
  │         │         │         │         ├─ Tier 5: best-fit params (2e-4)
  │         │         │         │         ├─ Tier 6: uncertainties (5e-4)
  │         │         │         │         │         │
  │         │         │         │         │         ├─ Tier 7: toy stats (0.03–0.05)
```

---

## Architecture: "Spec vs Speed"

The parity system is built on a clean separation of concerns:

```
┌──────────────────────────────────────────────────────┐
│  pyhf NumPy = SPEC (oracle for CI/regressions)       │
│  expected_data(params), nll(params), grad_nll(params) │
│  in fixed points — NOT "how the optimizer walked"     │
└───────────────────┬──────────────────────────────────┘
                    │ tolerance contract (7 tiers above)
┌───────────────────┴──────────────────────────────────┐
│  PreparedModel = COMPILED MODEL                       │
│  workspace JSON → flat layout:                        │
│    observed_flat, ln_factorials, obs_mask,             │
│    constraint_const, modifier CSR indices              │
│  params → nll: SIMD f64x4, sparse dispatch            │
│  (expected_data_f64_fast allocates scratch vectors)    │
├──────────────────┬───────────────────────────────────┤
│  Parity mode     │  Fast mode                         │
│  Kahan, 1 thread │  Accelerate/CUDA, Rayon            │
│  CI-gated        │  production                        │
└──────────────────┴───────────────────────────────────┘
```

**Key principle:** Parity mode is narrow but bulletproof. Fast mode is wide and aggressive.
Micro-divergences in the last bits are acceptable in Fast mode, but must remain within the
tolerance contract above.

**Optimizer philosophy:** Default = best-NLL (L-BFGS-B finds deeper minimum than SLSQP
on large models). pyhf-compat = warm-start `init_pars=pyhf_hat` for reproducibility.
These are different layers: parity = identical likelihood function, bestfit = optimizer quality.

### Batch Toy Fitting

Both CPU and GPU batch paths use the same compiled model:

- **CPU:** `fit_toys_batch()` — Rayon `par_iter().map_init()`, one AD Tape per thread
- **GPU:** `fit_toys_batch_gpu()` — lockstep L-BFGS-B, fused NLL+gradient kernel

Batch fitting skips Hessian computation for speed. Seed-based reproducibility:
`toy_seed = base_seed + toy_index`.

### Performance Characteristics

| Operation | vs Baseline |
|-----------|------------|
| PreparedModel NLL eval | ~200K evals/sec (simple), ~500/sec (184-param tHu) |
| Kahan overhead | <5% (confirmed benchmark) |
| Batch toys (1000, CPU Rayon) | ~50x vs serial |
| Batch toys (1000, CUDA) | ~200x vs serial |
| TTree parse + histogram fill | ~8.5x vs uproot+numpy |
| Ranking (16 NPs, autodiff) | ~4x vs pyhf FD |

---

## CI Integration

### Golden Report Generation

Set `NEXTSTAT_GOLDEN_REPORT_DIR` to write per-bin JSON reports during CI:

```bash
NEXTSTAT_GOLDEN_REPORT_DIR=reports/ pytest tests/python/test_expected_data_parity.py -v
```

Reports include: `max|Δ|`, `mean|Δ|`, worst bin index, per-fixture per-parameter-point.

### Recommended CI Test Command

```bash
# Full parity suite (deterministic, single-thread)
pytest tests/python/test_pyhf_validation.py tests/python/test_expected_data_parity.py -v
```

---

## Appendix: ROOT/HistFactory Cross-Validation

NextStat has been validated not only against pyhf but also against ROOT/RooFit using
three canonical HistFactory fixtures (xmlimport, multichannel, coupled_histosys).

**Key result:** NextStat agrees with pyhf at **< 1e-5** on q(mu) across all fixtures.
Where ROOT disagrees (up to 22.5 on coupled HistoSys), ROOT's own fit diagnostic
reports convergence failure (`status = -1`).

| Fixture | NS vs pyhf max |dq(mu)| | NS vs ROOT max |dq(mu)| | ROOT status |
|---------|----------------------------|-----------------------------|-------------|
| xmlimport | 1e-7 | 0.051 | 0 (ok) |
| multichannel | 4e-7 | 3.4e-8 | 0 (ok) |
| coupled_histosys | 5e-6 | 22.5 | -1 (FAILED) |

pyhf is the specification reference. ROOT deviations are informational, not gating.

Full analysis: `docs/references/root-histfactory-comparison.md`

---

## References

- `docs/plans/standards.md` — Project-wide standards (source of truth for math definitions)
- `tests/python/_tolerances.py` — Python constants (canonical values)
- `docs/references/optimizer-convergence.md` — L-BFGS-B vs SLSQP analysis
- `docs/references/root-histfactory-comparison.md` — ROOT/HistFactory 3-way comparison
- `docs/plans/2026-02-07_pyhf-spec-parity-plan.md` — Implementation plan for Parity/Fast modes
