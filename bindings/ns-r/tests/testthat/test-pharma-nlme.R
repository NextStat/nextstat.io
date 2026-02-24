make_multicpt_dataset <- function(n_subjects = 4L, dose = 120) {
  times_per <- c(0.5, 1.0, 2.0, 4.0, 8.0)
  n_per <- length(times_per)
  id <- rep(0:(n_subjects - 1L), each = n_per)
  times <- rep(times_per, times = n_subjects)

  dv <- numeric(length(times))
  for (sid in seq_len(n_subjects)) {
    idx <- ((sid - 1L) * n_per + 1L):(sid * n_per)
    decay <- 0.22 + 0.03 * (sid - 1L)
    scale <- 9.0 + 0.7 * (sid - 1L)
    shape <- 1.0 + 0.05 * sin(times_per + sid)
    dv[idx] <- pmax((dose / scale) * exp(-decay * times_per) * shape, 1e-6)
  }

  list(
    times = as.numeric(times),
    dv = as.numeric(dv),
    id = as.integer(id),
    n_subjects = as.integer(n_subjects),
    dose = as.numeric(dose)
  )
}

test_that("ns_foce multi-cpt FO/ITS/IMP dispatch smoke", {
  dat <- make_multicpt_dataset()

  res_fo <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "2cpt_iv",
    method = "fo",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0, 0.8, 20.0),
    omega_init = c(0.2, 0.2, 0.2, 0.2),
    max_outer_iter = 8L,
    tol = 1e-4
  )
  expect_true(is.list(res_fo))
  expect_length(res_fo$theta, 4L)
  expect_true(is.finite(res_fo$ofv))
  expect_equal(unname(dim(res_fo$eta)), c(dat$n_subjects, 4L))

  res_its <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "2cpt_oral",
    method = "its",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0, 0.8, 20.0, 1.5),
    omega_init = c(0.2, 0.2, 0.2, 0.2, 0.2),
    max_outer_iter = 8L,
    tol = 1e-4,
    its_max_iter = 5L,
    its_max_individual_iter = 20L,
    its_tol = 1e-4,
    its_omega_damping = 0.3
  )
  expect_true(is.list(res_its))
  expect_length(res_its$theta, 5L)
  expect_true(is.finite(res_its$ofv))
  expect_equal(unname(dim(res_its$eta)), c(dat$n_subjects, 5L))

  res_imp <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "3cpt_iv",
    method = "imp",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.1, 14.0, 0.7, 18.0, 0.5, 28.0),
    omega_init = c(0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
    max_outer_iter = 8L,
    tol = 1e-4,
    imp_n_iter = 4L,
    imp_n_samples = 80L,
    imp_proposal_scale = 1.0,
    imp_seed = 42L,
    imp_tol = 1e-4,
    imp_e_only = FALSE
  )
  expect_true(is.list(res_imp))
  expect_length(res_imp$theta, 6L)
  expect_true(is.finite(res_imp$ofv))
  expect_equal(unname(dim(res_imp$eta)), c(dat$n_subjects, 6L))
  expect_true(is.list(res_imp$imp))
  expect_gte(length(res_imp$imp$ofv_trace), 1L)
})

test_that("nlme_foce alias is available and callable", {
  dat <- make_multicpt_dataset()
  res <- nlme_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "2cpt_iv",
    method = "fo",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0, 0.8, 20.0),
    omega_init = c(0.2, 0.2, 0.2, 0.2),
    max_outer_iter = 6L,
    tol = 1e-4
  )
  expect_true(is.list(res))
  expect_length(res$theta, 4L)
})

test_that("ns_foce supports 1cpt_iv with FO/ITS/IMP", {
  dat <- make_multicpt_dataset()

  res_fo <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "1cpt_iv",
    method = "fo",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0),
    omega_init = c(0.2, 0.2),
    max_outer_iter = 6L,
    tol = 1e-4
  )
  expect_true(is.list(res_fo))
  expect_length(res_fo$theta, 2L)
  expect_true(is.finite(res_fo$ofv))

  res_its <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "1cpt_iv",
    method = "its",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0),
    omega_init = c(0.2, 0.2),
    max_outer_iter = 8L,
    tol = 1e-4,
    its_max_iter = 5L,
    its_max_individual_iter = 20L,
    its_tol = 1e-4,
    its_omega_damping = 0.3
  )
  expect_true(is.list(res_its))
  expect_length(res_its$theta, 2L)
  expect_true(is.finite(res_its$ofv))

  res_imp <- ns_foce(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "1cpt_iv",
    method = "imp",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0),
    omega_init = c(0.2, 0.2),
    max_outer_iter = 8L,
    tol = 1e-4,
    imp_n_iter = 4L,
    imp_n_samples = 80L,
    imp_proposal_scale = 1.0,
    imp_seed = 42L,
    imp_tol = 1e-4,
    imp_e_only = FALSE
  )
  expect_true(is.list(res_imp))
  expect_length(res_imp$theta, 2L)
  expect_true(is.finite(res_imp$ofv))
  expect_true(is.list(res_imp$imp))
})

test_that("ns_saem supports 1cpt_iv dispatch", {
  dat <- make_multicpt_dataset()
  res <- ns_saem(
    times = dat$times,
    dv = dat$dv,
    id = dat$id,
    n_subjects = dat$n_subjects,
    dose = dat$dose,
    model = "1cpt_iv",
    error_model = "additive",
    sigma = 0.1,
    theta_init = c(1.2, 15.0),
    omega_init = c(0.2, 0.2),
    n_burn = 40L,
    n_iter = 30L,
    n_chains = 1L,
    seed = 42L,
    tol = 1e-4
  )
  expect_true(is.list(res))
  expect_length(res$theta, 2L)
  expect_true(is.finite(res$ofv))
})
