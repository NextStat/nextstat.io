#' Normal Log-PDF
#'
#' Compute the log-density of the normal distribution for each element of
#' \code{x}.
#'
#' @param x Numeric vector of observations.
#' @param mu Mean of the normal distribution (scalar, default 0).
#' @param sigma Standard deviation (scalar, must be > 0, default 1).
#' @return Numeric vector of log-densities, same length as \code{x}.
#' @examples
#' ns_normal_logpdf(c(-1, 0, 1))
#' ns_normal_logpdf(c(2, 3), mu = 2.5, sigma = 0.5)
#' @export
ns_normal_logpdf <- function(x, mu = 0, sigma = 1) {
  .Call(
    "wrap__ns_normal_logpdf",
    as.numeric(x),
    as.numeric(mu),
    as.numeric(sigma),
    PACKAGE = "nextstat"
  )
}

#' Ordinary Least Squares Fit
#'
#' Fit a linear regression model via OLS.
#'
#' @param x Numeric matrix of predictors (\eqn{n \times p}).
#' @param y Numeric response vector of length \eqn{n}.
#' @param include_intercept Logical; if \code{TRUE} (default), prepend an
#'   intercept column.
#' @return Numeric vector of fitted coefficients (length \eqn{p} or
#'   \eqn{p + 1} if \code{include_intercept = TRUE}).
#' @examples
#' x <- matrix(c(1, 2, 3, 4), nrow = 2)
#' ns_ols_fit(x, c(1, 2))
#' @export
ns_ols_fit <- function(x, y, include_intercept = TRUE) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("nrow(x) must match length(y)")
  }
  .Call(
    "wrap__ns_ols_fit",
    x,
    y,
    as.logical(include_intercept),
    PACKAGE = "nextstat"
  )
}

#' HistFactory Maximum-Likelihood Fit
#'
#' Perform a maximum-likelihood fit of a HistFactory model defined by a pyhf
#' JSON or HS3 workspace string.
#'
#' @param workspace_json Character string containing the full pyhf JSON or
#'   HS3 workspace.
#' @return A named list with components:
#'   \describe{
#'     \item{parameter_names}{Character vector of parameter names.}
#'     \item{poi_index}{1-based index of the parameter of interest, or
#'       \code{NULL}.}
#'     \item{poi_name}{Name of the POI, or \code{NULL}.}
#'     \item{bestfit}{Numeric vector of best-fit parameter values.}
#'     \item{uncertainties}{Numeric vector of Hessian-based uncertainties.}
#'     \item{nll}{Negative log-likelihood at the best fit.}
#'     \item{twice_nll}{\eqn{2 \times} NLL (useful for \eqn{\chi^2}
#'       comparisons).}
#'     \item{converged}{Logical, whether the optimizer converged.}
#'     \item{n_iter, n_fev, n_gev}{Iteration / function / gradient evaluation
#'       counts.}
#'     \item{mu_hat, mu_sigma}{POI best-fit value and uncertainty (or
#'       \code{NULL}).}
#'   }
#' @examples
#' \dontrun{
#' ws <- paste(readLines("workspace.json"), collapse = "\n")
#' fit <- nextstat_fit(ws)
#' fit$bestfit
#' }
#' @export
nextstat_fit <- function(workspace_json) {
  .Call(
    "wrap__nextstat_fit",
    as.character(workspace_json),
    PACKAGE = "nextstat"
  )
}

#' Asymptotic CLs Hypothesis Test
#'
#' Test the signal-strength hypothesis \eqn{\mu = \mu_{\text{test}}} using the
#' asymptotic \eqn{\tilde{q}_\mu} test statistic and CLs procedure.
#'
#' @param workspace_json Character string containing the pyhf/HS3 workspace.
#' @param mu_test Signal strength to test (must be \eqn{\ge 0}).
#' @return A named list with components:
#'   \describe{
#'     \item{mu_test}{The tested signal strength.}
#'     \item{cls, clb, clsb}{CLs, CLb, and CLs+b p-values.}
#'     \item{teststat}{Observed test statistic value.}
#'     \item{q_mu, q_mu_a}{Observed and expected (Asimov) test statistics.}
#'     \item{mu_hat}{Unconditional MLE of \eqn{\mu}.}
#'   }
#' @examples
#' \dontrun{
#' ws <- paste(readLines("workspace.json"), collapse = "\n")
#' ht <- nextstat_hypotest(ws, mu_test = 1.0)
#' ht$cls
#' }
#' @export
nextstat_hypotest <- function(workspace_json, mu_test) {
  .Call(
    "wrap__nextstat_hypotest",
    as.character(workspace_json),
    as.numeric(mu_test),
    PACKAGE = "nextstat"
  )
}

#' Asymptotic Upper Limit (Brazil Band)
#'
#' Compute the observed and expected upper limits on the signal strength
#' \eqn{\mu} using a linear scan of CLs values.
#'
#' @param workspace_json Character string containing the pyhf/HS3 workspace.
#' @param cl Confidence level (default 0.95).
#' @param mu_range Numeric vector of length 2 giving the scan range
#'   \eqn{[\mu_{\min}, \mu_{\max}]}.
#' @param points Number of scan points (default 41, minimum 2).
#' @return A named list with components:
#'   \describe{
#'     \item{cl, alpha}{Confidence level and \eqn{\alpha = 1 - \text{CL}}.}
#'     \item{nsigma_order}{The 5 \eqn{N\sigma} levels for expected bands
#'       (\eqn{-2, -1, 0, +1, +2}).}
#'     \item{scan}{Numeric vector of scanned \eqn{\mu} values.}
#'     \item{observed_cls}{Observed CLs at each scan point.}
#'     \item{expected_cls}{List of 5-element numeric vectors (one per scan
#'       point) with expected CLs at each \eqn{N\sigma} level.}
#'     \item{observed_limit}{Observed upper limit on \eqn{\mu}.}
#'     \item{expected_limits}{Numeric vector of 5 expected limits
#'       (\eqn{-2\sigma \ldots +2\sigma}).}
#'   }
#' @examples
#' \dontrun{
#' ws <- paste(readLines("workspace.json"), collapse = "\n")
#' ul <- nextstat_upper_limit(ws, cl = 0.95, mu_range = c(0, 10), points = 61)
#' ul$observed_limit
#' }
#' @export
nextstat_upper_limit <- function(workspace_json, cl = 0.95, mu_range = c(0, 5), points = 41) {
  mu_range <- as.numeric(mu_range)
  if (length(mu_range) != 2) {
    stop("mu_range must be length-2 numeric vector")
  }
  .Call(
    "wrap__nextstat_upper_limit",
    as.character(workspace_json),
    as.numeric(cl),
    mu_range,
    as.integer(points),
    PACKAGE = "nextstat"
  )
}

#' Logistic Regression (GLM)
#'
#' Fit a logistic regression model via maximum likelihood.
#'
#' @param x Numeric matrix of predictors (\eqn{n \times p}).
#' @param y Binary response vector (0/1) of length \eqn{n}.
#' @param include_intercept Logical; if \code{TRUE} (default), prepend an
#'   intercept.
#' @return A named list with components:
#'   \describe{
#'     \item{parameter_names}{Character vector of coefficient names.}
#'     \item{coefficients}{Numeric vector of fitted coefficients.}
#'     \item{se}{Standard errors.}
#'     \item{nll}{Negative log-likelihood.}
#'     \item{deviance}{\eqn{2 \times} NLL.}
#'     \item{aic}{Akaike information criterion.}
#'     \item{converged}{Logical.}
#'     \item{n_iter, n_fev, n_gev}{Optimizer diagnostics.}
#'   }
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(200), 100, 2)
#' y <- as.numeric(x[, 1] + rnorm(100) > 0)
#' nextstat_glm_logistic(x, y)
#' @export
nextstat_glm_logistic <- function(x, y, include_intercept = TRUE) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("nrow(x) must match length(y)")
  }
  .Call(
    "wrap__nextstat_glm_logistic",
    x,
    y,
    as.logical(include_intercept),
    PACKAGE = "nextstat"
  )
}

#' Poisson Regression (GLM)
#'
#' Fit a Poisson regression model via maximum likelihood.
#'
#' @param x Numeric matrix of predictors (\eqn{n \times p}).
#' @param y Non-negative integer response vector of length \eqn{n}.
#' @param include_intercept Logical; if \code{TRUE} (default), prepend an
#'   intercept.
#' @return A named list with the same structure as
#'   \code{\link{nextstat_glm_logistic}}.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(200), 100, 2)
#' y <- rpois(100, lambda = exp(0.2 + 0.1 * x[, 1]))
#' nextstat_glm_poisson(x, y)
#' @export
nextstat_glm_poisson <- function(x, y, include_intercept = TRUE) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("nrow(x) must match length(y)")
  }
  .Call(
    "wrap__nextstat_glm_poisson",
    x,
    y,
    as.logical(include_intercept),
    PACKAGE = "nextstat"
  )
}

#' Negative Binomial Regression (GLM)
#'
#' Fit a negative binomial regression model via maximum likelihood
#' (jointly estimating the dispersion parameter).
#'
#' @param x Numeric matrix of predictors (\eqn{n \times p}).
#' @param y Non-negative integer response vector of length \eqn{n}.
#' @param include_intercept Logical; if \code{TRUE} (default), prepend an
#'   intercept.
#' @return A named list with the same structure as
#'   \code{\link{nextstat_glm_logistic}}.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(200), 100, 2)
#' y <- rnbinom(100, size = 2, mu = exp(0.2 + 0.1 * x[, 1]))
#' nextstat_glm_negbin(x, y)
#' @export
nextstat_glm_negbin <- function(x, y, include_intercept = TRUE) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("nrow(x) must match length(y)")
  }
  .Call(
    "wrap__nextstat_glm_negbin",
    x,
    y,
    as.logical(include_intercept),
    PACKAGE = "nextstat"
  )
}

#' Kalman Filter and RTS Smoother
#'
#' Run a Kalman filter followed by the Rauch--Tung--Striebel smoother on a
#' univariate time series.
#'
#' @param y Numeric vector of observations (NaN encodes missing values).
#' @param F State transition matrix (\eqn{k \times k}).
#' @param H Observation matrix (\eqn{1 \times k}).
#' @param Q Process noise covariance (\eqn{k \times k}).
#' @param R Observation noise variance (\eqn{1 \times 1} matrix).
#' @return A named list with components:
#'   \describe{
#'     \item{log_likelihood}{Marginal log-likelihood.}
#'     \item{filtered_means}{List of filtered state mean vectors.}
#'     \item{smoothed_means}{List of smoothed state mean vectors.}
#'   }
#' @examples
#' y <- cumsum(rnorm(100, sd = 0.2)) + rnorm(100, sd = 0.5)
#' kf <- nextstat_kalman(y,
#'   F = matrix(1), H = matrix(1),
#'   Q = matrix(0.1), R = matrix(0.25))
#' kf$log_likelihood
#' @export
nextstat_kalman <- function(y, F, H, Q, R) {
  y <- as.numeric(y)
  F <- as.matrix(F); storage.mode(F) <- "double"
  H <- as.matrix(H); storage.mode(H) <- "double"
  Q <- as.matrix(Q); storage.mode(Q) <- "double"
  R <- as.matrix(R); storage.mode(R) <- "double"
  .Call(
    "wrap__nextstat_kalman",
    y, F, H, Q, R,
    PACKAGE = "nextstat"
  )
}

#' GARCH(1,1) Volatility Model
#'
#' Fit a GARCH(1,1) model to a return series via maximum likelihood.
#'
#' @param y Numeric vector of returns.
#' @return A named list with components:
#'   \describe{
#'     \item{mu, omega, alpha, beta}{Estimated GARCH(1,1) parameters.}
#'     \item{log_likelihood}{Maximized log-likelihood.}
#'     \item{conditional_variance}{Numeric vector of fitted conditional
#'       variances.}
#'     \item{converged}{Logical.}
#'     \item{n_iter, n_fev, n_gev}{Optimizer diagnostics.}
#'   }
#' @examples
#' set.seed(1)
#' nextstat_garch(rnorm(500, sd = 0.01))
#' @export
nextstat_garch <- function(y) {
  .Call(
    "wrap__nextstat_garch",
    as.numeric(y),
    PACKAGE = "nextstat"
  )
}

#' Stochastic Volatility (Log-Chi-Squared)
#'
#' Fit a stochastic volatility model using the log-chi-squared
#' quasi-likelihood approximation.
#'
#' @param y Numeric vector of returns.
#' @param log_eps Small positive constant added inside the log transform to
#'   avoid \eqn{\log(0)} (default \code{1e-12}).
#' @return A named list with components:
#'   \describe{
#'     \item{mu, phi, sigma}{Estimated SV parameters.}
#'     \item{log_likelihood}{Quasi log-likelihood.}
#'     \item{smoothed_h}{Smoothed log-volatility series.}
#'     \item{smoothed_sigma}{Smoothed volatility series
#'       (\eqn{\exp(h_t / 2)}).}
#'     \item{converged}{Logical.}
#'     \item{n_iter, n_fev, n_gev}{Optimizer diagnostics.}
#'   }
#' @examples
#' set.seed(1)
#' nextstat_sv(rnorm(500, sd = 0.01))
#' @export
nextstat_sv <- function(y, log_eps = 1e-12) {
  .Call(
    "wrap__nextstat_sv",
    as.numeric(y),
    as.numeric(log_eps),
    PACKAGE = "nextstat"
  )
}
