#!/usr/bin/env Rscript

# Generate external (lme4) reference estimates for NextStat LMM fixture JSON.
#
# Requirements (R env):
# - lme4
# - jsonlite
#
# Usage:
#   Rscript tests/external/generate_lmm_goldens_lme4.R tests/fixtures/lmm/lmm_intercept_small.json
#
# Output:
# - Writes a JSON object to stdout containing parameter estimates in the
#   NextStat naming convention.

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("expected exactly 1 arg: path to fixture json", call. = FALSE)
}
path <- args[[1]]

fx <- jsonlite::fromJSON(path)

if (!isTRUE(fx$kind == "lmm_marginal")) {
  stop("fixture kind must be 'lmm_marginal'", call. = FALSE)
}

if (!isTRUE(fx$include_intercept)) {
  stop("this script expects include_intercept=true fixtures", call. = FALSE)
}

if (!is.numeric(fx$x) || ncol(fx$x) != 1) {
  stop("this script currently supports p=1 fixtures only", call. = FALSE)
}

df <- data.frame(
  y = as.numeric(fx$y),
  x1 = as.numeric(fx$x[, 1]),
  group = factor(as.integer(fx$group_idx))
)

re <- as.character(fx$random_effects)

if (re == "intercept") {
  # Random intercept only.
  fml <- y ~ 1 + x1 + (1 | group)
} else if (re == "intercept_slope") {
  # Random intercept + random slope with *independent* (diagonal) covariance.
  # lme4 syntax: `||` means no correlation.
  fml <- y ~ 1 + x1 + (1 + x1 || group)
} else {
  stop(paste0("unknown random_effects: ", re), call. = FALSE)
}

fit <- lme4::lmer(fml, data = df, REML = FALSE)

beta <- lme4::fixef(fit)
sigma_y <- sigma(fit)

vc <- lme4::VarCorr(fit)

tau_alpha <- as.numeric(attr(vc[["group"]], "stddev")[[1]])
tau_u <- NA
if (re == "intercept_slope") {
  # With (1 + x1 || group), lme4 creates a second term "group.1" for the slope.
  tau_u <- as.numeric(attr(vc[["group.1"]], "stddev")[[1]])
}

out <- list(
  tool = "lme4",
  tool_version = as.character(utils::packageVersion("lme4")),
  reml = FALSE,
  formula = deparse(fml),
  estimates = list(
    intercept = as.numeric(beta[["(Intercept)"]]),
    beta1 = as.numeric(beta[["x1"]]),
    log_sigma_y = log(sigma_y),
    log_tau_alpha = log(tau_alpha),
    log_tau_u_beta1 = if (re == "intercept_slope") log(tau_u) else NULL
  )
)

cat(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE), "\n")

