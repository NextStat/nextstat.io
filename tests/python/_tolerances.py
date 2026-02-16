"""Shared tolerances for Python regression tests.

Source of truth: `docs/plans/standards.md` and `docs/pyhf-parity-contract.md`.
"""

# Deterministic CPU parity (Phase 1 contract).
# pyhf returns `twice_nll`; NextStat returns `nll` (half that).
TWICE_NLL_RTOL = 1e-6
TWICE_NLL_ATOL = 1e-8

# Expected data parity (main + auxdata ordering).
EXPECTED_DATA_ATOL = 1e-8

# Fit parameter surfaces (Phase 1 contract; compare by parameter name).
PARAM_VALUE_ATOL = 2e-4
PARAM_UNCERTAINTY_ATOL = 5e-4

# Bias/pulls regression (NextStat vs pyhf deltas)
PULL_MEAN_DELTA_MAX = 0.05
PULL_STD_DELTA_MAX = 0.05
COVERAGE_1SIGMA_DELTA_MAX = 0.03

# Gradient parity (NextStat AD vs pyhf finite-diff).
# AD is more accurate than FD, so tolerance accounts for FD noise.
GRADIENT_ATOL = 1e-6
GRADIENT_RTOL = 1e-4

# Per-bin expected data parity (pure arithmetic, should be near-exact).
EXPECTED_DATA_PER_BIN_ATOL = 1e-12

# Coverage regression (NextStat vs pyhf deltas)
COVERAGE_DELTA_MAX = 0.05

# ---------------------------------------------------------------------------
# GPU parity (CUDA, f64, Code4/Code4p — same interpolation as CPU default)
# ---------------------------------------------------------------------------
GPU_NLL_ATOL = 1e-8       # NLL at same params: |gpu − cpu| < atol
GPU_NLL_RTOL = 1e-6       # NLL at same params: relative
GPU_GRAD_ATOL = 1e-5      # Per-element gradient (slightly relaxed for reduction order)
GPU_PARAM_ATOL = 2e-4     # Best-fit parameter values
GPU_FIT_NLL_ATOL = 1e-6   # NLL at best-fit point

# ---------------------------------------------------------------------------
# Unbinned closure + coverage (HP3)
# ---------------------------------------------------------------------------
# Closure: |fitted - truth| for each free parameter after fitting a large dataset.
UNBINNED_CLOSURE_PARAM_ATOL = 0.15       # absolute (generous: accounts for stat fluctuation)
UNBINNED_CLOSURE_PARAM_RTOL = 0.10       # relative
# Coverage: fraction of toys where truth is within 1σ of best-fit.
# Nominal is 0.683; Hessian-based uncertainties from L-BFGS-B are typically
# conservative (overestimate σ), so over-coverage up to ~95% is expected.
UNBINNED_COVERAGE_1SIGMA_LO = 0.55       # lower bound (N_toys ~ 100-200)
UNBINNED_COVERAGE_1SIGMA_HI = 0.98       # upper bound (conservative σ → over-coverage)

# ---------------------------------------------------------------------------
# Metal parity (f32 compute — Apple Silicon has no hardware f64)
# ---------------------------------------------------------------------------
METAL_NLL_ATOL = 1e-3
METAL_GRAD_ATOL = 1e-2
METAL_PARAM_ATOL = 1e-2
METAL_FIT_NLL_ATOL = 1e-3
# Batch toy fitting compounds f32 rounding across many L-BFGS-B iterations
# per toy, so the per-toy NLL tolerance is wider than a single fit.
METAL_BATCH_TOY_NLL_ATOL = 5e-3
