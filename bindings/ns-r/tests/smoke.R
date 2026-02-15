args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args) >= 1) args[[1]] else NULL

if (is.null(lib)) {
  library(nextstat)
} else {
  library(nextstat, lib.loc = lib)
}

lp <- ns_normal_logpdf(c(-1, 0, 1), mu = 0, sigma = 1)
stopifnot(length(lp) == 3)
stopifnot(all(is.finite(lp)))
stopifnot(abs(lp[2] - (-0.5 * log(2 * pi))) < 1e-10)

x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
y <- c(1, 2)
b <- ns_ols_fit(x, y, include_intercept = TRUE)
stopifnot(is.numeric(b))
stopifnot(length(b) == 3)

cat("OK\n")

# -------------------------------------------------------------------------
# HistFactory fit from pyhf JSON (R1)
# -------------------------------------------------------------------------
ws_path <- if (is.null(lib)) {
  system.file("extdata", "simple_workspace.json", package = "nextstat")
} else {
  system.file("extdata", "simple_workspace.json", package = "nextstat", lib.loc = lib)
}
stopifnot(nzchar(ws_path))
ws_json <- paste(readLines(ws_path, warn = FALSE), collapse = "\n")
fit <- nextstat_fit(ws_json)
stopifnot(is.list(fit))
stopifnot(isTRUE(fit$converged))
stopifnot(is.numeric(fit$nll))
stopifnot(length(fit$bestfit) == length(fit$parameter_names))
stopifnot(length(fit$uncertainties) == length(fit$parameter_names))
stopifnot(is.finite(fit$mu_hat))
stopifnot(fit$mu_hat >= 0)

cat("OK (fit)\n")

# -------------------------------------------------------------------------
# Asymptotic CLs hypotest (R2)
# -------------------------------------------------------------------------
ht <- nextstat_hypotest(ws_json, mu_test = 1.0)
stopifnot(is.list(ht))
stopifnot(is.finite(ht$cls))
stopifnot(is.finite(ht$clb))
stopifnot(is.finite(ht$clsb))
stopifnot(ht$cls >= 0 && ht$cls <= 1)
stopifnot(ht$clb >= 0 && ht$clb <= 1)
stopifnot(ht$clsb >= 0 && ht$clsb <= 1)

cat("OK (hypotest)\n")

# -------------------------------------------------------------------------
# Upper limit (Brazil band) (R3)
# -------------------------------------------------------------------------
ul <- nextstat_upper_limit(ws_json, cl = 0.95, mu_range = c(0, 10), points = 61)
stopifnot(is.list(ul))
stopifnot(is.finite(ul$observed_limit))
stopifnot(ul$observed_limit >= 0 && ul$observed_limit <= 10)
stopifnot(length(ul$expected_limits) == 5)
stopifnot(all(is.finite(ul$expected_limits)))

cat("OK (upper_limit)\n")

# -------------------------------------------------------------------------
# GLM wrappers (R4)
# -------------------------------------------------------------------------
set.seed(1)
x <- matrix(rnorm(40), nrow = 20, ncol = 2)

# logistic
y_bin <- as.numeric(x[, 1] + 0.3 * x[, 2] + rnorm(20, sd = 0.5) > 0)
lg <- nextstat_glm_logistic(x, y_bin, include_intercept = TRUE)
stopifnot(is.list(lg))
stopifnot(length(lg$coefficients) == length(lg$parameter_names))
stopifnot(length(lg$se) == length(lg$parameter_names))
stopifnot(is.finite(lg$aic))

# poisson
eta <- 0.2 + 0.1 * x[, 1] - 0.15 * x[, 2]
mu <- exp(eta)
y_pois <- rpois(20, lambda = mu)
pg <- nextstat_glm_poisson(x, y_pois, include_intercept = TRUE)
stopifnot(is.list(pg))
stopifnot(length(pg$coefficients) == length(pg$parameter_names))
stopifnot(is.finite(pg$aic))

# negbin (slightly overdispersed)
y_nb <- rnbinom(20, size = 2, mu = mu)
ng <- nextstat_glm_negbin(x, y_nb, include_intercept = TRUE)
stopifnot(is.list(ng))
stopifnot(length(ng$coefficients) == length(ng$parameter_names))
stopifnot(is.finite(ng$aic))

cat("OK (glm)\n")

# -------------------------------------------------------------------------
# Time series wrappers (R5)
# -------------------------------------------------------------------------
# Kalman (local level)
y <- cumsum(rnorm(50, sd = 0.2)) + rnorm(50, sd = 0.5)
F <- matrix(1, 1, 1)
H <- matrix(1, 1, 1)
Q <- matrix(0.1, 1, 1)
R <- matrix(0.25, 1, 1)
kf <- nextstat_kalman(y, F, H, Q, R)
stopifnot(is.list(kf))
stopifnot(is.finite(kf$log_likelihood))
stopifnot(length(kf$filtered_means) == length(y))
stopifnot(length(kf$smoothed_means) == length(y))

# GARCH / SV
rets <- rnorm(200, sd = 0.01)
g <- nextstat_garch(rets)
stopifnot(is.list(g))
stopifnot(is.finite(g$log_likelihood))
stopifnot(length(g$conditional_variance) == length(rets))

sv <- nextstat_sv(rets)
stopifnot(is.list(sv))
stopifnot(is.finite(sv$log_likelihood))
stopifnot(length(sv$smoothed_sigma) == length(rets))

cat("OK (timeseries)\n")

# -------------------------------------------------------------------------
# SCM wrapper (R6)
# -------------------------------------------------------------------------
set.seed(7)
n_subjects <- 8L
dose <- 100
times_per <- c(0.5, 1, 2, 4)
n_per <- length(times_per)
id <- rep(0:(n_subjects - 1L), each = n_per)
times <- rep(times_per, times = n_subjects)

wt <- seq(55, 95, length.out = n_subjects)
cl_pop <- 1.2
v_pop <- 10.0
ka_pop <- 1.6
sigma_add <- 0.05

dv <- numeric(length(times))
for (sid in seq_len(n_subjects)) {
  idx <- ((sid - 1L) * n_per + 1L):(sid * n_per)
  cl_i <- cl_pop * (wt[sid] / 70)^0.75
  kel_i <- cl_i / v_pop
  conc <- (dose / v_pop) * (ka_pop / (ka_pop - kel_i)) *
    (exp(-kel_i * times_per) - exp(-ka_pop * times_per))
  dv[idx] <- pmax(conc + rnorm(n_per, sd = sigma_add), 0)
}

scm <- ns_scm(
  times = times,
  dv = dv,
  id = id,
  n_subjects = n_subjects,
  dose = dose,
  theta = c(1.0, 10.0, 1.5),
  omega = c(0.3, 0.3, 0.3),
  sigma = 0.1,
  covariates = data.frame(WT = wt),
  alpha_forward = 0.05,
  alpha_backward = 0.01
)

stopifnot(is.list(scm))
stopifnot(is.finite(scm$ofv))
stopifnot(is.finite(scm$base_ofv))
stopifnot(is.numeric(scm$theta))
stopifnot(!is.null(scm$selected_names))

cat("OK (scm)\n")
