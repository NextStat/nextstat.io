"""Shared tolerances for Python regression tests.

Source of truth: `docs/plans/standards.md`.
"""

# Bias/pulls regression (NextStat vs pyhf deltas)
PULL_MEAN_DELTA_MAX = 0.05
PULL_STD_DELTA_MAX = 0.05
COVERAGE_1SIGMA_DELTA_MAX = 0.03

# Coverage regression (NextStat vs pyhf deltas)
COVERAGE_DELTA_MAX = 0.05
