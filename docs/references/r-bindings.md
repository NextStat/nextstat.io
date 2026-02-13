---
title: "R Bindings Reference"
status: experimental
last_updated: 2026-02-10
---

# R Bindings Reference

The `nextstat` R package provides a native R interface to the NextStat Rust core via staticlib FFI (`bindings/ns-r/`). Built with [extendr](https://extendr.github.io/).

**Status:** Experimental. The API is intentionally small and may change.

## Installation

```bash
# From the repository root
R CMD INSTALL bindings/ns-r

# Or into a local library
mkdir -p tmp/r-lib
R CMD INSTALL --library=tmp/r-lib bindings/ns-r
```

**Requirements:**
- Rust toolchain (`cargo`) — the Rust staticlib is compiled during `R CMD INSTALL`.
- C compiler toolchain compatible with R.
- R ≥ 4.0.

## Functions

### `ns_normal_logpdf(x, mu = 0, sigma = 1)`

Vectorized log-PDF of the normal distribution.

**Parameters:**
- `x` — numeric vector of evaluation points.
- `mu` — mean (scalar, default 0).
- `sigma` — standard deviation (scalar, default 1, must be > 0).

**Returns:** numeric vector of `log p(x | mu, sigma)`.

```r
library(nextstat)

logp <- ns_normal_logpdf(c(-1, 0, 1))
# [1] -1.418939 -0.918939 -1.418939

logp <- ns_normal_logpdf(c(5, 10, 15), mu = 10, sigma = 3)
```

### `ns_ols_fit(x, y, include_intercept = TRUE)`

Ordinary least squares regression.

**Parameters:**
- `x` — numeric matrix of predictors (n × p).
- `y` — numeric vector of responses (length n).
- `include_intercept` — logical, whether to prepend an intercept column (default `TRUE`).

**Returns:** numeric vector of coefficients. If `include_intercept = TRUE`, the first element is the intercept.

```r
library(nextstat)

x <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
y <- c(1.1, 2.3, 3.0)
beta <- ns_ols_fit(x, y)
# beta[1] = intercept, beta[2:3] = slopes
```

## Architecture

- **Rust code:** `bindings/ns-r/src/rust/src/lib.rs` — extendr-annotated functions.
- **R wrappers:** `bindings/ns-r/R/nextstat.R` — `.Call()` wrappers with input validation.
- **Build:** `R CMD INSTALL` triggers `cargo build --release` for the Rust staticlib, which is linked into `nextstat.so`.
- **No hand-written C:** extendr generates the C glue automatically.

## Extending

To add a new function:

1. Add an `#[extendr]` function in `bindings/ns-r/src/rust/src/lib.rs`.
2. Register it in the `extendr_module!` block.
3. Add an R wrapper in `bindings/ns-r/R/nextstat.R`.
4. Export in `NAMESPACE`.
5. `R CMD INSTALL bindings/ns-r` to rebuild.

## Limitations

- GPU acceleration is not yet exposed to R.
- HistFactory inference is exposed, but currently limited to asymptotic CLs (`nextstat_fit`, `nextstat_hypotest`, `nextstat_upper_limit`); toy-based inference is not yet exposed.
- ONNX-backed neural PDFs require the `neural` Cargo feature, which is not enabled in the default R build.
