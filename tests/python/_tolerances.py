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
