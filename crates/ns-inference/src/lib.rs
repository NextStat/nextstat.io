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

pub mod hypotest;
pub mod mle;
pub mod optimizer;
pub mod profile_likelihood;
pub mod transforms;

pub use hypotest::{AsymptoticCLsContext, HypotestResult};
pub use mle::{MaximumLikelihoodEstimator, RankingEntry};
pub use optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
pub use profile_likelihood::{ProfileLikelihoodScan, ProfilePoint};
