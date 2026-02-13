\name{nextstat_timeseries}
\alias{nextstat_kalman}
\alias{nextstat_garch}
\alias{nextstat_sv}
\title{Time series wrappers (Kalman, GARCH, stochastic volatility)}
\description{
Convenience wrappers around NextStat's Rust time series baselines.
\code{nextstat_kalman()} runs Kalman filtering and RTS smoothing for 1D observations.
\code{nextstat_garch()} fits a Gaussian GARCH(1,1) model by MLE.
\code{nextstat_sv()} fits an approximate stochastic volatility model (log-chi2 + Kalman MLE).
}
\usage{
nextstat_kalman(y, F, H, Q, R)

nextstat_garch(y)

nextstat_sv(y, log_eps = 1e-12)
}
\arguments{
  \item{y}{Numeric vector of observations (may include \code{NaN} for missing, but not \code{Inf}).}
  \item{F}{State transition matrix.}
  \item{H}{Observation matrix (currently 1 x n-state).}
  \item{Q}{Process noise covariance (n-state x n-state).}
  \item{R}{Observation noise covariance (1 x 1).}
  \item{log_eps}{Small epsilon added inside \code{log(y^2 + eps)} for SV stability.}
}
\value{
\code{nextstat_kalman()} returns a named list with \code{log_likelihood},
\code{filtered_means}, and \code{smoothed_means}. \code{nextstat_garch()} and
\code{nextstat_sv()} return fitted parameters, log-likelihood, and smoothed/variance outputs.
}
\examples{
set.seed(1)
y <- cumsum(rnorm(50, sd = 0.2)) + rnorm(50, sd = 0.5)
F <- matrix(1, 1, 1); H <- matrix(1, 1, 1)
Q <- matrix(0.1, 1, 1); R <- matrix(0.25, 1, 1)
nextstat_kalman(y, F, H, Q, R)

rets <- rnorm(200, sd = 0.01)
nextstat_garch(rets)
nextstat_sv(rets)
}
