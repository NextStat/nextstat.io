ws_path <- system.file("extdata", "simple_workspace.json", package = "nextstat")
skip_if(!nzchar(ws_path), "simple_workspace.json not found")
ws_json <- paste(readLines(ws_path, warn = FALSE), collapse = "\n")

test_that("nextstat_fit returns valid MLE result", {
  fit <- nextstat_fit(ws_json)
  expect_type(fit, "list")
  expect_true(isTRUE(fit$converged))
  expect_true(is.finite(fit$nll))
  expect_length(fit$bestfit, length(fit$parameter_names))
  expect_length(fit$uncertainties, length(fit$parameter_names))
  expect_true(is.finite(fit$mu_hat))
  expect_gte(fit$mu_hat, 0)
})

test_that("nextstat_hypotest returns valid CLs result", {
  ht <- nextstat_hypotest(ws_json, mu_test = 1.0)
  expect_type(ht, "list")
  expect_true(is.finite(ht$cls))
  expect_true(is.finite(ht$clb))
  expect_true(is.finite(ht$clsb))
  expect_gte(ht$cls, 0)
  expect_lte(ht$cls, 1)
})

test_that("nextstat_upper_limit returns valid Brazil band", {
  ul <- nextstat_upper_limit(ws_json, cl = 0.95, mu_range = c(0, 10), points = 61)
  expect_type(ul, "list")
  expect_true(is.finite(ul$observed_limit))
  expect_gte(ul$observed_limit, 0)
  expect_lte(ul$observed_limit, 10)
  expect_length(ul$expected_limits, 5)
  expect_true(all(is.finite(ul$expected_limits)))
})
