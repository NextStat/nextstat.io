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

if (is.null(out_path) || out_path == "") {
  stop("missing required --out")
}
if (is.null(in_path) || in_path == "") {
  stop("missing required --in")
}
if (is.na(repeat_n) || repeat_n < 1L) repeat_n <- 1L
if (is.na(seed)) seed <- 12345L

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

case_id <- "unknown"
case_obj <- tryCatch(jsonlite::fromJSON(in_path), error = function(e) NULL)
if (!is.null(case_obj) && !is.null(case_obj$case)) {
  case_id <- as.character(case_obj$case)
}

if (!requireNamespace("saemix", quietly = TRUE)) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "saemix",
    case = case_id,
    status = "skipped",
    reason = "saemix not installed",
    meta = meta
  ))
  quit(status = 0)
}

suppressPackageStartupMessages({
  library(saemix)
  library(jsonlite)
})

spec <- case_obj$dataset$spec
if (is.null(spec) || !is.list(spec)) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "saemix",
    case = case_id,
    status = "failed",
    reason = "input case JSON missing dataset.spec",
    meta = meta,
    packages = list(saemix = as.character(utils::packageVersion("saemix")))
  ))
  quit(status = 0)
}

if (!identical(as.character(spec$kind), "pop_pk_1c_oral")) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "saemix",
    case = case_id,
    status = "skipped",
    reason = paste0("unsupported dataset kind: ", as.character(spec$kind)),
    meta = meta,
    packages = list(saemix = as.character(utils::packageVersion("saemix")))
  ))
  quit(status = 0)
}

if (!identical(as.character(spec$error_model), "additive")) {
  write_json(out_path, list(
    schema_version = "nextstat.pharma_baseline_result.v1",
    baseline = "saemix",
    case = case_id,
    status = "skipped",
    reason = paste0("unsupported error_model for baseline runner: ", as.character(spec$error_model)),
    meta = meta,
    packages = list(saemix = as.character(utils::packageVersion("saemix")))
  ))
  quit(status = 0)
}

ids <- as.integer(spec$subject_idx) + 1L
times <- as.numeric(spec$times)
y <- as.numeric(spec$y)
n_sub <- as.integer(spec$n_subjects)
dose <- as.numeric(spec$dose)
true_theta <- as.numeric(spec$true_theta)
true_omega <- as.numeric(spec$true_omega)

df <- data.frame(
  id = ids,
  time = times,
  y = y,
  dose = rep(dose, length(y))
)

model_1cpt_oral <- function(psi, id, xidep) {
  time <- xidep[, "time"]
  dose_col <- xidep[, "dose"]
  cl <- exp(psi[id, 1])
  v <- exp(psi[id, 2])
  ka <- exp(psi[id, 3])
  ke <- cl / v
  den <- pmax(ka - ke, 1e-12)
  pred <- (dose_col * ka / (v * den)) * (exp(-ke * time) - exp(-ka * time))
  pred
}

psi0 <- matrix(c(log(0.13), log(8.0), log(1.0)), ncol = 3, byrow = TRUE)
sae_data <- saemixData(
  name.data = df,
  name.group = "id",
  name.predictors = c("time", "dose"),
  name.response = "y"
)
sae_model <- saemixModel(
  model = model_1cpt_oral,
  description = "1cpt oral additive",
  psi0 = psi0,
  transform.par = c(0, 0, 0),
  covariance.model = diag(3),
  error.model = "constant",
  name.modpar = c("logCL", "logV", "logKa")
)

fit_once <- function(run_seed) {
  ctl <- saemixControl(
    seed = as.integer(run_seed),
    nbiter.saemix = c(200L, 100L),
    displayProgress = FALSE,
    print = FALSE,
    save = FALSE,
    save.graphs = FALSE,
    nb.chains = 1L
  )
  suppressWarnings(saemix(sae_model, sae_data, ctl))
}

status <- "ok"
reason <- NULL
fit <- NULL
runs <- numeric(repeat_n)

tryCatch({
  invisible(fit_once(seed))
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
    baseline = "saemix",
    case = case_id,
    status = "failed",
    reason = if (is.null(reason)) "fit failed" else reason,
    meta = meta,
    packages = list(saemix = as.character(utils::packageVersion("saemix")))
  ))
  quit(status = 0)
}

fx <- as.numeric(fit@results@fixed.effects)
theta_hat <- c(exp(fx[1]), exp(fx[2]), exp(fx[3]))
omega_mat <- as.matrix(fit@results@omega)
omega_hat <- as.numeric(sqrt(pmax(diag(omega_mat), 0.0)))
ll <- suppressWarnings(as.numeric(fit@results@ll.is))

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
  baseline = "saemix",
  case = case_id,
  status = "ok",
  timing = list(
    fit_time_s = as.numeric(min(runs)),
    raw = list(repeat_n = as.integer(repeat_n), policy = "min", per_fit_s = as.numeric(runs))
  ),
  meta = c(
    meta,
    list(
      method = "saemix",
      objective = if (is.na(ll)) NA_real_ else as.numeric(-2.0 * ll),
      log_likelihood = ll
    )
  ),
  packages = list(saemix = as.character(utils::packageVersion("saemix"))),
  recovery = recovery
))
