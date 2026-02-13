if (!requireNamespace("testthat", quietly = TRUE)) {
  cat("SKIP: package 'testthat' is not installed\n")
  quit(status = 0)
}

library(testthat)
library(nextstat)

test_check("nextstat")
