\name{nextstat_glm}
\alias{nextstat_glm_logistic}
\alias{nextstat_glm_poisson}
\alias{nextstat_glm_negbin}
\title{Generalized linear model (GLM) wrappers}
\description{
Fit logistic (Bernoulli), Poisson, and negative binomial regression models
using NextStat's Rust optimizer and likelihood implementations.
}
\usage{
nextstat_glm_logistic(x, y, include_intercept = TRUE)

nextstat_glm_poisson(x, y, include_intercept = TRUE)

nextstat_glm_negbin(x, y, include_intercept = TRUE)
}
\arguments{
  \item{x}{Numeric matrix of predictors.}
  \item{y}{Response vector. For logistic, must be 0/1. For Poisson and negbin, must be non-negative integers.}
  \item{include_intercept}{Logical; include an intercept term.}
}
\value{
A named list including \code{parameter_names}, \code{coefficients}, \code{se},
\code{nll}, \code{deviance}, and \code{aic}.
}
\examples{
set.seed(1)
x <- matrix(rnorm(40), nrow = 20, ncol = 2)
y_bin <- as.numeric(x[, 1] + 0.3 * x[, 2] + rnorm(20, sd = 0.5) > 0)
nextstat_glm_logistic(x, y_bin)

eta <- 0.2 + 0.1 * x[, 1] - 0.15 * x[, 2]
y_pois <- rpois(20, lambda = exp(eta))
nextstat_glm_poisson(x, y_pois)

y_nb <- rnbinom(20, size = 2, mu = exp(eta))
nextstat_glm_negbin(x, y_nb)
}
