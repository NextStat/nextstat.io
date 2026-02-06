//! # ns-inference
//!
//! Statistical inference for NextStat.
//!
//! This crate provides:
//! - Maximum Likelihood Estimation (MLE) - Phase 1
//! - NUTS (MCMC) sampling - Phase 3
//! - Profile likelihood - Phase 3
//! - Hypothesis testing - Phase 3
//!
//! ## Architecture
//!
//! This crate depends on `ComputeBackend` trait from ns-core, NOT on
//! concrete backend implementations. This is clean architecture in action.

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Adaptation: step-size dual averaging and diagonal mass matrix (Welford variance).
pub mod adapt;
/// Chain storage and multi-chain parallel runner.
pub mod chain;
/// MCMC diagnostics: split R-hat, bulk/tail ESS.
pub mod diagnostics;
/// HMC leapfrog integrator.
pub mod hmc;
/// Frequentist hypothesis testing (asymptotic CLs).
pub mod hypotest;
/// Laplace approximation utilities (generic).
pub mod laplace;
/// Linear mixed models (marginal likelihood baseline).
pub mod lmm;
/// Maximum-likelihood estimation via L-BFGS-B.
pub mod mle;
/// Ordinary differential equation (ODE) solvers (Phase 13 baseline).
pub mod ode;
/// Pharmacometrics models (Phase 13).
pub mod pk;
/// NUTS tree-building and sampling.
pub mod nuts;
/// Generic numerical optimizer (L-BFGS-B backend).
pub mod optimizer;
/// Ordinal regression models (Phase 9 Pack C).
pub mod ordinal;
/// Posterior API: log-pdf, gradient, transforms.
pub mod posterior;
/// Profile likelihood scans.
pub mod profile_likelihood;
/// Regression models (general statistics).
pub mod regression;
/// Parametric survival models (Phase 9 Pack A).
pub mod survival;
/// Time series and state space models (Phase 8).
pub mod timeseries;
/// Toy data generation (Asimov + Poisson).
pub mod toys;
/// Bijective transforms for unconstrained parameterisation.
pub mod transforms;

#[cfg(test)]
mod universal_tests;

pub use builder::{ComposedGlmModel, ModelBuilder};
pub use chain::{SamplerResult, sample_nuts_multichain};
pub use diagnostics::DiagnosticsResult;
pub use hypotest::{AsymptoticCLsContext, HypotestResult};
pub use laplace::{LaplaceResult, laplace_log_marginal};
pub use lmm::{LmmMarginalModel, RandomEffects as LmmRandomEffects};
pub use mle::{MaximumLikelihoodEstimator, RankingEntry};
pub use ode::{OdeSolution, rk4_linear};
pub use pk::{LloqPolicy, OneCompartmentOralPkModel};
pub use nuts::{NutsConfig, sample_nuts};
pub use optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
pub use ordinal::{OrderedLogitModel, OrderedProbitModel};
pub use posterior::{Posterior, Prior};
pub use profile_likelihood::{ProfileLikelihoodScan, ProfilePoint};
pub use regression::{
    LinearRegressionModel, LogisticRegressionModel, PoissonRegressionModel, ols_fit,
};
pub use survival::{
    CoxPhModel, CoxTies, ExponentialSurvivalModel, LogNormalAftModel, WeibullSurvivalModel,
};
pub use toys::{asimov_main, poisson_main_from_expected, poisson_main_toys};
pub use transforms::ParameterTransform;
/// Model builder (composition) MVP for general statistics.
pub mod builder;
