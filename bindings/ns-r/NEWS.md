# nextstat 0.9.0

Initial CRAN submission.

## New features

- **HistFactory inference**: `nextstat_fit()`, `nextstat_hypotest()`,
  `nextstat_upper_limit()` — maximum-likelihood fitting, asymptotic CLs
  hypothesis testing, and upper limits (Brazil bands) from pyhf JSON or
  HS3 workspaces.
- **GLM wrappers**: `nextstat_glm_logistic()`, `nextstat_glm_poisson()`,
  `nextstat_glm_negbin()` — logistic, Poisson, and negative binomial
  regression via maximum likelihood.
- **Time series**: `nextstat_kalman()` (Kalman filter + RTS smoother),
  `nextstat_garch()` (GARCH(1,1)), `nextstat_sv()` (stochastic volatility
  via log-chi-squared quasi-likelihood).
- **Core utilities**: `ns_normal_logpdf()`, `ns_ols_fit()`.

## Infrastructure

- Rust backend via extendr (static library linked into the R shared object).
- `configure` script validates Rust toolchain (>= 1.85) at install time.
- testthat test suite (4 test files covering all 11 exported functions).
- Getting-started vignette with runnable examples.
- Self-contained vendored Rust workspace under `src/` for reproducible
  `R CMD build` / `R CMD check` tarball builds.
