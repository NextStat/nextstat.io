#' @title FOCE/FOCEI Estimation for Population PK
#'
#' @description Fit population PK models with FOCE-family methods:
#'   \code{"foce"}, \code{"focei"}, \code{"fo"} (first-order),
#'   \code{"its"} (iterative two-stage), and \code{"imp"} (importance sampling).
#'   Supported models are \code{"1cpt_iv"}, \code{"1cpt_oral"}, \code{"2cpt_iv"},
#'   \code{"2cpt_oral"}, \code{"3cpt_iv"}, and \code{"3cpt_oral"}.
#'
#' @param times Numeric vector of observation times.
#' @param dv Numeric vector of observed concentrations (dependent variable).
#' @param id Integer vector of subject identifiers (0-indexed).
#' @param n_subjects Integer, number of unique subjects.
#' @param dose Numeric scalar, dose amount.
#' @param bioav Numeric scalar, bioavailability fraction (default 1.0).
#' @param error_model Character, one of \code{"additive"}, \code{"proportional"},
#'   or \code{"combined"}.
#' @param sigma Numeric, residual error SD (for additive/proportional) or
#'   numeric vector of length 2 (for combined: \code{c(sigma_add, sigma_prop)}).
#' @param theta_init Initial population parameters. Length must match \code{model}:
#'   2 (\code{1cpt_iv}), 3 (\code{1cpt_oral}), 4 (\code{2cpt_iv}), 5 (\code{2cpt_oral}),
#'   6 (\code{3cpt_iv}), or 7 (\code{3cpt_oral}).
#' @param omega_init Initial random effect SDs with same length as \code{theta_init}.
#' @param model Character model selector:
#'   \code{"1cpt_iv"}, \code{"1cpt_oral"}, \code{"2cpt_iv"}, \code{"2cpt_oral"},
#'   \code{"3cpt_iv"}, \code{"3cpt_oral"}.
#' @param method Character estimation method:
#'   \code{"foce"}, \code{"focei"}, \code{"fo"}, \code{"its"}, \code{"imp"}.
#' @param interaction Logical, use FOCEI (default \code{TRUE}) or FOCE.
#' @param max_outer_iter Maximum outer iterations (default 100).
#' @param tol Convergence tolerance (default 1e-4).
#' @param its_max_iter ITS outer iterations.
#' @param its_max_individual_iter ITS per-subject iterations.
#' @param its_tol ITS convergence tolerance.
#' @param its_omega_damping ITS Omega damping.
#' @param imp_n_iter IMP EM iterations.
#' @param imp_n_samples IMP Monte Carlo samples per subject.
#' @param imp_proposal_scale IMP proposal scale.
#' @param imp_seed IMP RNG seed.
#' @param imp_tol IMP convergence tolerance.
#' @param imp_e_only IMP E-only mode.
#'
#' @return A named list with components:
#'   \describe{
#'     \item{theta}{Numeric vector of fitted population parameters.}
#'     \item{omega}{Numeric vector of fitted random effect SDs.}
#'     \item{eta}{Matrix of individual random effects (n_subjects x n_theta).}
#'     \item{ofv}{Objective function value (-2 log L).}
#'     \item{converged}{Logical.}
#'     \item{n_iter}{Number of iterations.}
#'     \item{correlation}{Correlation matrix of random effects.}
#'     \item{imp}{IMP diagnostics list (\code{ofv_trace}, \code{ess_fraction_trace},
#'       \code{max_weight_trace}) when \code{method="imp"}, otherwise \code{NULL}.}
#'   }
#'
#' @examples
#' \dontrun{
#' fit <- ns_foce(
#'   times = dat$TIME, dv = dat$DV, id = dat$ID - 1,
#'   n_subjects = length(unique(dat$ID)),
#'   dose = 100, theta_init = c(0.133, 8, 0.8),
#'   omega_init = c(0.3, 0.25, 0.3), sigma = 0.5
#' )
#' fit2 <- ns_foce(
#'   times = dat$TIME, dv = dat$DV, id = dat$ID - 1,
#'   n_subjects = length(unique(dat$ID)),
#'   dose = 100,
#'   model = "2cpt_iv",
#'   method = "fo",
#'   theta_init = c(1.2, 15, 0.9, 20),
#'   omega_init = rep(0.2, 4),
#'   sigma = 0.1
#' )
#' fit$theta
#' }
#' @export
ns_foce <- function(times, dv, id, n_subjects, dose, bioav = 1.0,
                    error_model = "additive", sigma = 1.0,
                    theta_init, omega_init,
                    model = "1cpt_oral",
                    method = "focei",
                    interaction = TRUE,
                    max_outer_iter = 100L, tol = 1e-4,
                    its_max_iter = 30L,
                    its_max_individual_iter = 60L,
                    its_tol = 1e-4,
                    its_omega_damping = 0.3,
                    imp_n_iter = 15L,
                    imp_n_samples = 300L,
                    imp_proposal_scale = 1.0,
                    imp_seed = 42L,
                    imp_tol = 1e-4,
                    imp_e_only = FALSE) {
  stopifnot(
    is.numeric(times), is.numeric(dv), is.numeric(id),
    length(times) == length(dv), length(times) == length(id),
    is.numeric(theta_init), length(theta_init) >= 2, length(theta_init) <= 7,
    is.numeric(omega_init), length(omega_init) >= 2, length(omega_init) <= 7,
    model %in% c("1cpt_iv", "1cpt_oral", "2cpt_iv", "2cpt_oral", "3cpt_iv", "3cpt_oral"),
    method %in% c("foce", "focei", "fo", "its", "imp"),
    error_model %in% c("additive", "proportional", "combined")
  )
  .Call(
    "wrap__ns_foce",
    as.numeric(times), as.numeric(dv), as.integer(id),
    as.integer(n_subjects), as.numeric(dose), as.numeric(bioav),
    as.character(error_model), as.numeric(sigma),
    as.numeric(theta_init), as.numeric(omega_init),
    as.character(model), as.character(method),
    as.logical(interaction),
    as.integer(max_outer_iter), as.numeric(tol),
    as.integer(its_max_iter), as.integer(its_max_individual_iter),
    as.numeric(its_tol), as.numeric(its_omega_damping),
    as.integer(imp_n_iter), as.integer(imp_n_samples),
    as.numeric(imp_proposal_scale), as.integer(imp_seed),
    as.numeric(imp_tol), as.logical(imp_e_only),
    PACKAGE = "nextstat"
  )
}

#' @title NLME FOCE Interface
#'
#' @description Alias for \code{ns_foce()} provided for cross-language
#'   \code{nlme_*} API parity.
#'
#' @inheritParams ns_foce
#' @return Same as \code{ns_foce()}.
#' @export
nlme_foce <- ns_foce

#' @title SAEM Estimation for Population PK
#'
#' @description Fit population PK models using Stochastic Approximation EM
#'   (SAEM). More robust than FOCE for complex nonlinear models.
#'
#' @inheritParams ns_foce
#' @param n_burn Integer, number of burn-in iterations (default 200).
#' @param n_iter Integer, number of estimation iterations (default 100).
#' @param n_chains Integer, MCMC chains per subject (default 1).
#' @param seed Integer, random seed for reproducibility.
#'
#' @return A named list with components:
#'   \describe{
#'     \item{theta}{Numeric vector of fitted population parameters.}
#'     \item{omega}{Numeric vector of fitted random effect SDs.}
#'     \item{eta}{Matrix of individual random effects (n_subjects x n_theta).}
#'     \item{ofv}{Objective function value (-2 log L).}
#'     \item{converged}{Logical.}
#'     \item{n_iter}{Total iterations (burn-in + estimation).}
#'     \item{acceptance_rates}{Numeric vector of MCMC acceptance rates per subject.}
#'     \item{ofv_trace}{Numeric vector of OFV values across iterations.}
#'   }
#'
#' @examples
#' \dontrun{
#' fit <- ns_saem(
#'   times = dat$TIME, dv = dat$DV, id = dat$ID - 1,
#'   n_subjects = length(unique(dat$ID)),
#'   dose = 100, theta_init = c(0.133, 8, 0.8),
#'   omega_init = c(0.3, 0.25, 0.3), sigma = 0.5, seed = 42
#' )
#' fit$theta
#' }
#' @export
ns_saem <- function(times, dv, id, n_subjects, dose, bioav = 1.0,
                    error_model = "additive", sigma = 1.0,
                    theta_init, omega_init,
                    model = "1cpt_oral",
                    n_burn = 200L, n_iter = 100L, n_chains = 1L,
                    seed = 12345L, tol = 1e-4) {
  stopifnot(
    is.numeric(times), is.numeric(dv), is.numeric(id),
    length(times) == length(dv), length(times) == length(id),
    is.numeric(theta_init), length(theta_init) >= 2, length(theta_init) <= 7,
    is.numeric(omega_init), length(omega_init) >= 2, length(omega_init) <= 7,
    model %in% c("1cpt_iv", "1cpt_oral", "2cpt_iv", "2cpt_oral", "3cpt_iv", "3cpt_oral"),
    error_model %in% c("additive", "proportional", "combined")
  )
  .Call(
    "wrap__ns_saem",
    as.numeric(times), as.numeric(dv), as.integer(id),
    as.integer(n_subjects), as.numeric(dose), as.numeric(bioav),
    as.character(error_model), as.numeric(sigma),
    as.numeric(theta_init), as.numeric(omega_init),
    as.character(model),
    as.integer(n_burn), as.integer(n_iter), as.integer(n_chains),
    as.integer(seed), as.numeric(tol),
    PACKAGE = "nextstat"
  )
}

#' @title NLME SAEM Interface
#'
#' @description Alias for \code{ns_saem()} provided for cross-language
#'   \code{nlme_*} API parity.
#'
#' @inheritParams ns_saem
#' @return Same as \code{ns_saem()}.
#' @export
nlme_saem <- ns_saem

#' @title Emax PD Model Prediction
#'
#' @description Predict pharmacodynamic effect using the Emax model:
#'   \eqn{E(C) = E_0 + E_{max} \cdot C / (EC_{50} + C)}.
#'
#' @param conc Numeric vector of drug concentrations.
#' @param e0 Baseline effect (default 0).
#' @param emax Maximum drug effect.
#' @param ec50 Concentration at 50\% of Emax (must be > 0).
#'
#' @return Numeric vector of predicted effects.
#'
#' @examples
#' conc <- c(0, 1, 5, 10, 50, 100)
#' ns_emax(conc, e0 = 0, emax = 100, ec50 = 10)
#' @export
ns_emax <- function(conc, e0 = 0, emax, ec50) {
  stopifnot(is.numeric(conc), ec50 > 0)
  conc <- pmax(conc, 0)
  e0 + emax * conc / (ec50 + conc)
}

#' @title Sigmoid Emax (Hill) PD Model Prediction
#'
#' @description Predict pharmacodynamic effect using the sigmoid Emax model:
#'   \eqn{E(C) = E_0 + E_{max} \cdot C^\gamma / (EC_{50}^\gamma + C^\gamma)}.
#'
#' @inheritParams ns_emax
#' @param gamma Hill coefficient (steepness parameter, must be > 0).
#'
#' @return Numeric vector of predicted effects.
#'
#' @examples
#' conc <- c(0, 1, 5, 10, 50, 100)
#' ns_sigmoid_emax(conc, e0 = 0, emax = 100, ec50 = 10, gamma = 2)
#' @export
ns_sigmoid_emax <- function(conc, e0 = 0, emax, ec50, gamma = 1) {
  stopifnot(is.numeric(conc), ec50 > 0, gamma > 0)
  conc <- pmax(conc, 0)
  cg <- conc^gamma
  ecg <- ec50^gamma
  e0 + emax * cg / (ecg + cg)
}

#' @title Indirect Response Model Simulation
#'
#' @description Simulate an indirect response (IDR) model given a drug
#'   concentration time profile.
#'
#' @param conc_times Numeric vector of times for concentration profile.
#' @param conc_values Numeric vector of concentrations at each time.
#' @param output_times Numeric vector of times at which to report response.
#' @param type Character, one of \code{"inhibit_production"} (Type I),
#'   \code{"inhibit_loss"} (Type II), \code{"stimulate_production"} (Type III),
#'   \code{"stimulate_loss"} (Type IV).
#' @param kin Zero-order production rate.
#' @param kout First-order loss rate constant.
#' @param max_effect Maximum drug effect (Imax for inhibition in [0,1],
#'   Emax for stimulation > 0).
#' @param c50 Concentration at 50\% max effect (IC50 or EC50).
#' @param r0 Initial response (default: \code{kin/kout}).
#'
#' @return Numeric vector of response values at \code{output_times}.
#'
#' @examples
#' \dontrun{
#' times <- 0:48
#' conc <- 20 * exp(-0.1 * times)
#' resp <- ns_idr(times, conc, 0:48,
#'   type = "stimulate_production",
#'   kin = 1, kout = 0.1, max_effect = 2, c50 = 5)
#' plot(times, resp, type = "l")
#' }
#' @export
ns_idr <- function(conc_times, conc_values, output_times,
                   type = "stimulate_production",
                   kin, kout, max_effect, c50, r0 = NULL) {
  stopifnot(
    is.numeric(conc_times), is.numeric(conc_values),
    length(conc_times) == length(conc_values),
    is.numeric(output_times),
    type %in% c("inhibit_production", "inhibit_loss",
                "stimulate_production", "stimulate_loss"),
    kin > 0, kout > 0, c50 > 0, max_effect > 0
  )
  if (is.null(r0)) r0 <- kin / kout
  .Call(
    "wrap__ns_idr",
    as.numeric(conc_times), as.numeric(conc_values),
    as.numeric(output_times),
    as.character(type),
    as.numeric(kin), as.numeric(kout),
    as.numeric(max_effect), as.numeric(c50),
    as.numeric(r0),
    PACKAGE = "nextstat"
  )
}

#' @title Visual Predictive Check (VPC)
#'
#' @description Run a VPC for a fitted 1-compartment oral PK model.
#'
#' @param times Numeric vector of observation times.
#' @param dv Numeric vector of observed concentrations.
#' @param id Integer vector of subject identifiers (0-indexed).
#' @param n_subjects Integer.
#' @param dose Numeric scalar.
#' @param theta Numeric vector of population parameters \code{c(CL, V, Ka)}.
#' @param omega Numeric vector of random effect SDs.
#' @param sigma Numeric scalar, residual error SD.
#' @param n_rep Integer, number of simulation replicates (default 200).
#' @param n_bins Integer, number of time bins (default 10).
#'
#' @return A named list with VPC results.
#'
#' @examples
#' \dontrun{
#' vpc <- ns_vpc(dat$TIME, dat$DV, dat$ID-1, 32, 100,
#'   theta = fit$theta, omega = fit$omega, sigma = 0.5)
#' }
#' @export
ns_vpc <- function(times, dv, id, n_subjects, dose,
                   theta, omega, sigma,
                   n_rep = 200L, n_bins = 10L) {
  stopifnot(
    is.numeric(times), is.numeric(dv), is.numeric(id),
    length(theta) == 3, length(omega) == 3
  )
  .Call(
    "wrap__ns_vpc",
    as.numeric(times), as.numeric(dv), as.integer(id),
    as.integer(n_subjects), as.numeric(dose),
    as.numeric(theta), as.numeric(omega), as.numeric(sigma),
    as.integer(n_rep), as.integer(n_bins),
    PACKAGE = "nextstat"
  )
}

#' @title Goodness-of-Fit Diagnostics
#'
#' @description Compute PRED, IPRED, IWRES, and CWRES for a fitted
#'   1-compartment oral PK model.
#'
#' @inheritParams ns_vpc
#' @param eta Matrix of individual random effects (n_subjects x 3).
#'
#' @return A data.frame with columns TIME, DV, PRED, IPRED, IWRES, CWRES.
#'
#' @export
ns_gof <- function(times, dv, id, n_subjects, dose,
                   theta, omega, eta, sigma) {
  stopifnot(
    is.numeric(times), is.numeric(dv), is.numeric(id),
    length(theta) == 3, length(omega) == 3
  )
  .Call(
    "wrap__ns_gof",
    as.numeric(times), as.numeric(dv), as.integer(id),
    as.integer(n_subjects), as.numeric(dose),
    as.numeric(theta), as.numeric(omega),
    as.matrix(eta), as.numeric(sigma),
    PACKAGE = "nextstat"
  )
}

#' @title Stepwise Covariate Modeling (SCM)
#'
#' @description Run forward selection + backward elimination of covariate
#'   effects on PK parameters.
#'
#' @param times Numeric vector of observation times.
#' @param dv Numeric vector of observed concentrations.
#' @param id Integer vector of subject identifiers (0-indexed).
#' @param n_subjects Integer.
#' @param dose Numeric scalar.
#' @param theta Numeric vector of population parameters \code{c(CL, V, Ka)}.
#' @param omega Numeric vector of random effect SDs.
#' @param sigma Numeric scalar, residual error SD.
#' @param covariates A data.frame with one row per subject and columns for
#'   each covariate (e.g., WT, AGE, CRCL).
#' @param alpha_forward Forward selection threshold (default 0.05).
#' @param alpha_backward Backward elimination threshold (default 0.01).
#'
#' @return A named list with selected covariate effects and SCM traces.
#'
#' @export
ns_scm <- function(times, dv, id, n_subjects, dose,
                   theta, omega, sigma, covariates,
                   alpha_forward = 0.05, alpha_backward = 0.01) {
  stopifnot(
    is.numeric(times), is.numeric(dv), is.numeric(id),
    length(times) == length(dv), length(times) == length(id),
    is.data.frame(covariates), nrow(covariates) == n_subjects,
    is.numeric(theta), length(theta) == 3,
    is.numeric(omega), length(omega) == 3,
    is.numeric(sigma), length(sigma) == 1, sigma > 0,
    is.numeric(alpha_forward), length(alpha_forward) == 1,
    is.numeric(alpha_backward), length(alpha_backward) == 1
  )

  cov_mat <- as.matrix(covariates)
  if (!is.numeric(cov_mat)) {
    stop("covariates must be numeric", call. = FALSE)
  }
  if (any(!is.finite(cov_mat))) {
    stop("covariates must contain only finite values", call. = FALSE)
  }
  cov_names <- colnames(cov_mat)
  if (is.null(cov_names)) {
    cov_names <- paste0("cov", seq_len(ncol(cov_mat)))
  }

  .Call(
    "wrap__ns_scm",
    as.numeric(times), as.numeric(dv), as.integer(id),
    as.integer(n_subjects), as.numeric(dose),
    as.numeric(theta), as.numeric(omega), as.numeric(sigma),
    cov_mat, as.character(cov_names),
    as.numeric(alpha_forward), as.numeric(alpha_backward),
    PACKAGE = "nextstat"
  )
}
