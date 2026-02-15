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
method <- get_arg("--method", "focei")
repeat_n <- suppressWarnings(as.integer(get_arg("--repeat", "1")))

if (is.null(out_path) || out_path == "") stop("missing required --out")
if (is.null(in_path) || in_path == "") stop("missing required --in")
if (is.na(repeat_n) || repeat_n < 1L) repeat_n <- 1L

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to write baseline JSON (install.packages('jsonlite'))")
}

write_json <- function(path, obj) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  json <- jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE)
  writeLines(paste0(json, "\n"), con = path)
}

schema_version <- "nextstat.pharma_baseline_result.v1"
meta_base <- list(
  r_version = as.character(getRversion()),
  platform = paste0(R.version$platform, " (", R.version$os, ")")
)

if (!requireNamespace("nlmixr2", quietly = TRUE)) {
  write_json(out_path, list(
    schema_version = schema_version,
    baseline = "nlmixr2",
    case = "unknown",
    status = "skipped",
    reason = "nlmixr2 not installed",
    meta = meta_base
  ))
  quit(status = 0)
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(nlmixr2)
})

obj <- fromJSON(in_path)
spec <- obj$dataset$spec
case_id <- if (!is.null(obj$case)) as.character(obj$case) else "unknown"

if (!is.list(spec) || !is.numeric(spec$n_subjects) || !is.numeric(spec$times) || !is.numeric(spec$y)) {
  write_json(out_path, list(
    schema_version = schema_version,
    baseline = "nlmixr2",
    case = case_id,
    status = "failed",
    reason = "input case JSON missing expected dataset.spec fields",
    meta = meta_base
  ))
  quit(status = 0)
}

if (!identical(as.character(spec$error_model), "additive")) {
  write_json(out_path, list(
    schema_version = schema_version,
    baseline = "nlmixr2",
    case = case_id,
    status = "skipped",
    reason = paste0("unsupported error_model for baseline runner: ", as.character(spec$error_model)),
    meta = meta_base,
    packages = list(nlmixr2 = as.character(utils::packageVersion("nlmixr2")))
  ))
  quit(status = 0)
}

ids <- as.integer(spec$subject_idx) + 1L
n_sub <- as.integer(spec$n_subjects)
dose <- as.numeric(spec$dose)

obs <- data.frame(
  ID = ids,
  TIME = as.numeric(spec$times),
  DV = as.numeric(spec$y),
  EVID = 0L,
  AMT = 0,
  CMT = 2L
)

doses <- data.frame(
  ID = seq_len(n_sub),
  TIME = 0,
  DV = NA_real_,
  EVID = 1L,
  AMT = dose,
  CMT = 1L
)

dat <- rbind(doses, obs)
dat <- dat[order(dat$ID, dat$TIME, -dat$EVID), ]

onec <- function() {
  ini({
    tcl <- log(0.13)
    tv <- log(8)
    tka <- log(1)
    eta.cl ~ 0.04
    eta.v ~ 0.0225
    eta.ka ~ 0.0625
    add.err <- 0.3
  })
  model({
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    ka <- exp(tka + eta.ka)
    linCmt() ~ add(add.err)
  })
}

if (method == "focei") {
  ctl <- foceiControl(maxOuterIterations = 300, maxInnerIterations = 30, print = 0)
} else if (method == "saem") {
  ctl <- saemControl(nBurn = 200, nEm = 100, print = 0)
} else {
  stop("unknown --method; expected focei|saem")
}

fit_once <- function() {
  nlmixr2(onec, dat, est = method, control = ctl)
}

status <- "ok"
reason <- NULL
fit <- NULL
runs <- numeric(repeat_n)

tryCatch({
  # warmup: compile/codegen once
  invisible(fit_once())

  for (i in seq_len(repeat_n)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- fit_once()
    runs[i] <- proc.time()[["elapsed"]] - t0
  }
}, error = function(e) {
  status <<- "failed"
  reason <<- paste0(class(e)[1], ": ", conditionMessage(e))
})

if (status != "ok" || is.null(fit) || length(runs) == 0L) {
  write_json(out_path, list(
    schema_version = schema_version,
    baseline = "nlmixr2",
    case = case_id,
    status = "failed",
    reason = if (is.null(reason)) "fit failed" else reason,
    meta = meta_base,
    packages = list(
      nlmixr2 = as.character(utils::packageVersion("nlmixr2")),
      rxode2 = if (requireNamespace("rxode2", quietly = TRUE)) as.character(utils::packageVersion("rxode2")) else NA_character_
    )
  ))
  quit(status = 0)
}

fx <- as.list(fixef(fit))
theta_hat <- c(exp(fx$tcl), exp(fx$tv), exp(fx$tka))
omega_hat <- as.numeric(sqrt(diag(as.matrix(fit$omega))))
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
  schema_version = schema_version,
  baseline = "nlmixr2",
  case = case_id,
  status = "ok",
  timing = list(
    fit_time_s = as.numeric(min(runs)),
    raw = list(repeat_n = as.integer(repeat_n), policy = "min", per_fit_s = as.numeric(runs))
  ),
  meta = c(
    meta_base,
    list(method = method, objective = as.numeric(fit$objf), converged = !is.null(fit$objf))
  ),
  packages = list(
    nlmixr2 = as.character(utils::packageVersion("nlmixr2")),
    rxode2 = if (requireNamespace("rxode2", quietly = TRUE)) as.character(utils::packageVersion("rxode2")) else NA_character_
  ),
  recovery = recovery
))
