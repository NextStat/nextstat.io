\name{ns_ols_fit}
\alias{ns_ols_fit}
\title{Ordinary least squares (OLS) fit}
\description{
Fit a linear regression model by ordinary least squares.
}
\usage{
ns_ols_fit(x, y, include_intercept = TRUE)
}
\arguments{
  \item{x}{Numeric matrix with \code{nrow(x)} observations and \code{ncol(x)} features.}
  \item{y}{Numeric response vector of length \code{nrow(x)}.}
  \item{include_intercept}{Logical; include an intercept term.}
}
\value{
Numeric vector of coefficients. If \code{include_intercept = TRUE}, the first
element is the intercept.
}
\examples{
x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
y <- c(1, 2)
ns_ols_fit(x, y)
}
