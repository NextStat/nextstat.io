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
/// Maximum-likelihood estimation via L-BFGS-B.
pub mod mle;
/// NUTS tree-building and sampling.
pub mod nuts;
/// Generic numerical optimizer (L-BFGS-B backend).
pub mod optimizer;
/// Posterior API: log-pdf, gradient, transforms.
pub mod posterior;
/// Profile likelihood scans.
pub mod profile_likelihood;
/// Bijective transforms for unconstrained parameterisation.
pub mod transforms;

pub use chain::{SamplerResult, sample_nuts_multichain};
pub use diagnostics::DiagnosticsResult;
pub use hypotest::{AsymptoticCLsContext, HypotestResult};
pub use mle::{MaximumLikelihoodEstimator, RankingEntry};
pub use nuts::{NutsConfig, sample_nuts};
pub use optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
pub use posterior::{Posterior, Prior};
pub use profile_likelihood::{ProfileLikelihoodScan, ProfilePoint};
pub use transforms::ParameterTransform;
