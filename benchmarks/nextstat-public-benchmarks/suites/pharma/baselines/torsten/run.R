#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default_value = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default_value)
  if (idx == length(args)) return(default_value)
  return(args[idx + 1])
}

out_path <- get_arg("--out", "")
in_path <- get_arg("--in", "")
repeat_n <- suppressWarnings(as.integer(get_arg("--repeat", "1")))
seed <- suppressWarnings(as.integer(get_arg("--seed", "12345")))
iter_max <- suppressWarnings(as.integer(get_arg("--iter", "500")))

if (is.null(out_path) || out_path == "") {
  stop("missing required --out")
}
if (is.null(in_path) || in_path == "") {
  stop("missing required --in")
}
if (is.na(repeat_n) || repeat_n < 1L) repeat_n <- 1L
if (is.na(seed)) seed <- 12345L
if (is.na(iter_max) || iter_max < 50L) iter_max <- 500L

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to write baseline JSON (install.packages('jsonlite'))")
}

write_json <- function(path, obj) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  json <- jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE)
  writeLines(paste0(json, "\n"), con = path)
}

meta <- list(
  r_version = as.character(getRversion()),
  platform = paste0(R.version$platform, " (", R.version$os, ")")
)

if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "torsten",
    case = "unknown",
    status = "skipped",
    reason = "cmdstanr not installed",
    meta = meta
  ))
  quit(status = 0)
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(cmdstanr)
})

case <- fromJSON(in_path)
spec <- case$dataset$spec
case_id <- if (!is.null(case$case)) as.character(case$case) else "unknown"

skip_with_reason <- function(reason) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "torsten",
    case = case_id,
    status = "skipped",
    reason = reason,
    meta = meta,
    packages = list(
      cmdstanr = as.character(utils::packageVersion("cmdstanr"))
    )
  ))
  quit(status = 0)
}

if (!is.list(spec)) {
  skip_with_reason("input case JSON missing dataset.spec")
}
if (!identical(as.character(spec$kind), "pop_pk_1c_oral")) {
  skip_with_reason(paste0("unsupported dataset kind: ", as.character(spec$kind)))
}
if (!identical(as.character(spec$error_model), "additive")) {
  skip_with_reason(paste0("unsupported error_model: ", as.character(spec$error_model)))
}

cmdstan_path <- tryCatch(cmdstanr::cmdstan_path(), error = function(e) "")
if (!nzchar(cmdstan_path)) {
  skip_with_reason("cmdstanr installed but cmdstan_path is not configured")
}

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_file <- if (length(script_arg) > 0) sub("^--file=", "", script_arg[[1]]) else ""
stan_file <- if (nzchar(script_file)) {
  file.path(dirname(normalizePath(script_file, winslash = "/")), "pop_pk_1cpt_map.stan")
} else {
  ""
}
if (!nzchar(stan_file) || !file.exists(stan_file)) {
  stan_file <- file.path(getwd(), "benchmarks/nextstat-public-benchmarks/suites/pharma/baselines/torsten/pop_pk_1cpt_map.stan")
}
if (!file.exists(stan_file)) {
  stop("could not locate pop_pk_1cpt_map.stan")
}

stan_data <- list(
  N = length(spec$y),
  S = as.integer(spec$n_subjects),
  sid = as.integer(spec$subject_idx) + 1L,
  time = as.numeric(spec$times),
  y = as.numeric(spec$y),
  dose = as.numeric(spec$dose),
  sigma = as.numeric(spec$sigma)
)

cache_dir <- file.path(tempdir(), "nextstat-cmdstan-cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
exe_file <- file.path(cache_dir, "pop_pk_1cpt_map")
model <- cmdstan_model(stan_file, exe_file = exe_file, quiet = TRUE)

init_list <- list(
  tcl = log(0.134),
  tv = log(8.0),
  tka = log(1.0),
  omega = c(0.20, 0.15, 0.25),
  z = matrix(0.0, nrow = as.integer(stan_data$S), ncol = 3)
)

fit_once <- function(run_seed) {
  model$optimize(
    data = stan_data,
    init = list(init_list),
    seed = as.integer(run_seed),
    algorithm = "lbfgs",
    iter = as.integer(iter_max),
    refresh = 0,
    jacobian = TRUE
  )
}

status <- "ok"
reason <- NULL
fit <- NULL
runs <- numeric(repeat_n)

tryCatch({
  invisible(fit_once(seed)) # warmup
  for (i in seq_len(repeat_n)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- fit_once(seed + i)
    runs[i] <- proc.time()[["elapsed"]] - t0
  }
}, error = function(e) {
  status <<- "failed"
  reason <<- paste0(class(e)[1], ": ", conditionMessage(e))
})

if (status != "ok" || is.null(fit) || length(runs) == 0L) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "torsten",
    case = case_id,
    status = "failed",
    reason = if (is.null(reason)) "fit failed" else reason,
    meta = meta,
    packages = list(
      cmdstanr = as.character(utils::packageVersion("cmdstanr"))
    )
  ))
  quit(status = 0)
}

mle <- fit$mle()
theta_log_hat <- c(
  as.numeric(mle[["tcl"]]),
  as.numeric(mle[["tv"]]),
  as.numeric(mle[["tka"]])
)
theta_hat <- exp(theta_log_hat)
omega_hat <- c(
  as.numeric(mle[["omega[1]"]]),
  as.numeric(mle[["omega[2]"]]),
  as.numeric(mle[["omega[3]"]])
)
lp <- if ("lp__" %in% names(mle)) {
  as.numeric(mle[["lp__"]])
} else {
  suppressWarnings(as.numeric(tryCatch(fit$summary("lp__")$estimate[[1]], error = function(e) NA_real_)))
}

true_theta <- as.numeric(spec$true_theta)
true_omega <- as.numeric(spec$true_omega)

recovery <- list(
  CL = list(hat = theta_hat[[1]], true = true_theta[[1]], rel_err = abs(theta_hat[[1]] - true_theta[[1]]) / abs(true_theta[[1]])),
  V = list(hat = theta_hat[[2]], true = true_theta[[2]], rel_err = abs(theta_hat[[2]] - true_theta[[2]]) / abs(true_theta[[2]])),
  Ka = list(hat = theta_hat[[3]], true = true_theta[[3]], rel_err = abs(theta_hat[[3]] - true_theta[[3]]) / abs(true_theta[[3]])),
  w_CL = list(hat = omega_hat[[1]], true = true_omega[[1]], rel_err = abs(omega_hat[[1]] - true_omega[[1]]) / abs(true_omega[[1]])),
  w_V = list(hat = omega_hat[[2]], true = true_omega[[2]], rel_err = abs(omega_hat[[2]] - true_omega[[2]]) / abs(true_omega[[2]])),
  w_Ka = list(hat = omega_hat[[3]], true = true_omega[[3]], rel_err = abs(omega_hat[[3]] - true_omega[[3]]) / abs(true_omega[[3]]))
)

write_json(out_path, list(
  schema_version = "nextstat.pharma_baseline_result.v1",
  baseline = "torsten",
  case = case_id,
  status = "ok",
  timing = list(
    fit_time_s = as.numeric(min(runs)),
    raw = list(repeat_n = as.integer(repeat_n), policy = "min", per_fit_s = as.numeric(runs))
  ),
  meta = c(
    meta,
    list(
      method = "cmdstanr_optimize_lbfgs",
      objective = lp,
      implementation = "stan_analytic_pop_pk_1cpt",
      iter = as.integer(iter_max),
      cmdstan_path = cmdstan_path
    )
  ),
  packages = list(
    cmdstanr = as.character(utils::packageVersion("cmdstanr")),
    cmdstan = tryCatch(paste(cmdstanr::cmdstan_version(), collapse = "."), error = function(e) NA_character_)
  ),
  recovery = recovery
))
