"""Shared tolerances for Python regression tests.

Source of truth: `docs/plans/standards.md`.
"""

# Deterministic CPU parity (Phase 1 contract).
# pyhf returns `twice_nll`; NextStat returns `nll` (half that).
TWICE_NLL_RTOL = 1e-6
TWICE_NLL_ATOL = 1e-8

# Expected data parity (main + auxdata ordering).
EXPECTED_DATA_ATOL = 1e-8

# Bias/pulls regression (NextStat vs pyhf deltas)
PULL_MEAN_DELTA_MAX = 0.05
PULL_STD_DELTA_MAX = 0.05
COVERAGE_1SIGMA_DELTA_MAX = 0.03

# Coverage regression (NextStat vs pyhf deltas)
COVERAGE_DELTA_MAX = 0.05
