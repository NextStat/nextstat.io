\name{nextstat_histfactory}
\alias{nextstat_fit}
\alias{nextstat_hypotest}
\alias{nextstat_upper_limit}
\title{HistFactory-style inference from pyhf JSON}
\description{
Wrappers around NextStat's Rust HistFactory inference pipeline.
\code{nextstat_fit()} runs an MLE fit, \code{nextstat_hypotest()} computes an
asymptotic CLs (qtilde) hypotest at a fixed POI value, and
\code{nextstat_upper_limit()} computes an asymptotic upper limit with Brazil bands
via a linear scan and interpolation.
}
\usage{
nextstat_fit(workspace_json)

nextstat_hypotest(workspace_json, mu_test)

nextstat_upper_limit(workspace_json, cl = 0.95, mu_range = c(0, 5), points = 41)
}
\arguments{
  \item{workspace_json}{A JSON string containing a pyhf workspace (or HS3 JSON).}
  \item{mu_test}{Tested POI value (typically \code{mu}).}
  \item{cl}{Confidence level for the upper limit (e.g. 0.95).}
  \item{mu_range}{Numeric length-2 vector \code{c(mu_lo, mu_hi)} for the scan range.}
  \item{points}{Number of scan points (>= 2).}
}
\value{
\code{nextstat_fit()} returns a named list with parameter names, best-fit values,
uncertainties and diagnostics. \code{nextstat_hypotest()} returns CLs/CLb/CLs+b
and related test statistic values. \code{nextstat_upper_limit()} returns observed
and expected limits and the scan curves.
}
\examples{
ws <- paste(
  readLines(system.file("extdata", "simple_workspace.json", package = "nextstat")),
  collapse = "\\n"
)
fit <- nextstat_fit(ws)
ht <- nextstat_hypotest(ws, mu_test = 1.0)
ul <- nextstat_upper_limit(ws, cl = 0.95, mu_range = c(0, 10), points = 61)
}
