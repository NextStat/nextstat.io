test_that("nextstat_glm_logistic fits binary data", {
  set.seed(1)
  x <- matrix(rnorm(40), nrow = 20, ncol = 2)
  y_bin <- as.numeric(x[, 1] + 0.3 * x[, 2] + rnorm(20, sd = 0.5) > 0)
  lg <- nextstat_glm_logistic(x, y_bin, include_intercept = TRUE)
  expect_type(lg, "list")
  expect_length(lg$coefficients, length(lg$parameter_names))
  expect_length(lg$se, length(lg$parameter_names))
  expect_true(is.finite(lg$aic))
})

test_that("nextstat_glm_poisson fits count data", {
  set.seed(1)
  x <- matrix(rnorm(40), nrow = 20, ncol = 2)
  eta <- 0.2 + 0.1 * x[, 1] - 0.15 * x[, 2]
  y_pois <- rpois(20, lambda = exp(eta))
  pg <- nextstat_glm_poisson(x, y_pois, include_intercept = TRUE)
  expect_type(pg, "list")
  expect_length(pg$coefficients, length(pg$parameter_names))
  expect_true(is.finite(pg$aic))
})

test_that("nextstat_glm_negbin fits overdispersed count data", {
  set.seed(1)
  x <- matrix(rnorm(40), nrow = 20, ncol = 2)
  mu <- exp(0.2 + 0.1 * x[, 1] - 0.15 * x[, 2])
  y_nb <- rnbinom(20, size = 2, mu = mu)
  ng <- nextstat_glm_negbin(x, y_nb, include_intercept = TRUE)
  expect_type(ng, "list")
  expect_length(ng$coefficients, length(ng$parameter_names))
  expect_true(is.finite(ng$aic))
})
