test_that("ns_normal_logpdf returns correct values", {
  lp <- ns_normal_logpdf(c(-1, 0, 1))
  expect_length(lp, 3)
  expect_true(all(is.finite(lp)))
  expect_equal(lp[2], -0.5 * log(2 * pi), tolerance = 1e-10)
})

test_that("ns_ols_fit returns coefficients", {
  x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  y <- c(1, 2)
  b <- ns_ols_fit(x, y, include_intercept = TRUE)
  expect_true(is.numeric(b))
  expect_length(b, 3)
})
