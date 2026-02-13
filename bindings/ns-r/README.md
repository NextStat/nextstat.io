# `nextstat` (R bindings)

This directory contains an **experimental** R package that links NextStat’s Rust core into R.

## Local install (dev)

```bash
mkdir -p tmp/r-lib
R CMD INSTALL --library=tmp/r-lib bindings/ns-r
R -q -e 'library(nextstat, lib.loc="tmp/r-lib"); print(ns_normal_logpdf(c(-1,0,1))); print(ns_ols_fit(matrix(c(1,2,3,4),2,2), c(1,2)))'
```

## HistFactory fit (pyhf JSON)

```r
ws <- paste(readLines(system.file("extdata", "simple_workspace.json", package = "nextstat")), collapse = "\n")
fit <- nextstat_fit(ws)
str(fit)
```

## CLs hypotest (asymptotics)

```r
ws <- paste(readLines(system.file("extdata", "simple_workspace.json", package = "nextstat")), collapse = "\n")
ht <- nextstat_hypotest(ws, mu_test = 1.0)
str(ht)
```

## Upper limit (Brazil band)

```r
ws <- paste(readLines(system.file("extdata", "simple_workspace.json", package = "nextstat")), collapse = "\n")
ul <- nextstat_upper_limit(ws, cl = 0.95, mu_range = c(0, 10), points = 61)
str(ul)
```

## GLM wrappers

```r
x <- matrix(rnorm(200), nrow = 100, ncol = 2)
y_bin <- as.numeric(x[, 1] + 0.3 * x[, 2] + rnorm(100) > 0)
logit <- nextstat_glm_logistic(x, y_bin)

y_pois <- rpois(100, lambda = exp(0.2 + 0.1 * x[, 1] - 0.15 * x[, 2]))
pois <- nextstat_glm_poisson(x, y_pois)

y_nb <- rnbinom(100, size = 2, mu = exp(0.2 + 0.1 * x[, 1] - 0.15 * x[, 2]))
nb <- nextstat_glm_negbin(x, y_nb)
```

## Time series wrappers

```r
# Kalman filter + RTS smoother (1D observations)
y <- cumsum(rnorm(200, sd = 0.2)) + rnorm(200, sd = 0.5)
F <- matrix(1, 1, 1); H <- matrix(1, 1, 1)
Q <- matrix(0.1, 1, 1); R <- matrix(0.25, 1, 1)
kf <- nextstat_kalman(y, F, H, Q, R)

# Volatility baselines
rets <- rnorm(1000, sd = 0.01)
g <- nextstat_garch(rets)
sv <- nextstat_sv(rets)
```

## Notes

- The Rust code lives in `bindings/ns-r/src/rust/` and is built during `R CMD INSTALL`.
- The package uses **extendr** (no hand-written `.Call` C wrappers); the Rust staticlib is linked into `nextstat.so` during install.
- Vendored Rust crates for CRAN tarball self-containment are synced from root via `bash scripts/nsr_vendor_sync.sh` (or `make nsr-vendor-sync`).
- For clean local CRAN-style logs, run `make nsr-cran-check-clean` (requires `pandoc`, `checkbashisms`, and R suggests: `testthat`, `knitr`, `rmarkdown`).
- Current API is intentionally small; it’s meant as a starting point.
