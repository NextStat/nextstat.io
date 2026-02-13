\name{ns_normal_logpdf}
\alias{ns_normal_logpdf}
\title{Normal log-density (vectorized)}
\description{
Compute the log-density of a normal distribution for a vector of inputs.
}
\usage{
ns_normal_logpdf(x, mu = 0, sigma = 1)
}
\arguments{
  \item{x}{Numeric vector of values.}
  \item{mu}{Mean (scalar).}
  \item{sigma}{Standard deviation (scalar, > 0).}
}
\value{
Numeric vector of log-densities with the same length as \code{x}.
}
\examples{
ns_normal_logpdf(c(-1, 0, 1))
}
