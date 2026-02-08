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
//! This crate depends on core model/compute abstractions from `ns-core`, and
//! does not depend on concrete backend implementations. GPU execution is
//! accessed through session/accelerator APIs (see `gpu_session`, `gpu_batch`,
//! `metal_batch`, and differentiable modules).

#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::suspicious_operation_groupings)]
#![allow(clippy::identity_op)]
#![allow(clippy::same_item_push)]
#![allow(dead_code)]

/// Adaptation: step-size dual averaging and diagonal mass matrix (Welford variance).
pub mod adapt;
/// Batch toy fitting with optional Accelerate-optimized NLL.
pub mod batch;
/// Chain storage and multi-chain parallel runner.
pub mod chain;
/// MCMC diagnostics: split R-hat, bulk/tail ESS.
pub mod diagnostics;
/// HMC leapfrog integrator.
pub mod hmc;
/// Frequentist hypothesis testing (CLs): asymptotic + toy-based.
pub mod hypotest;
/// Laplace approximation utilities (generic).
pub mod laplace;
/// Shared L-BFGS-B state machine for GPU lockstep optimization.
pub(crate) mod lbfgs;
/// Linear mixed models (marginal likelihood baseline).
pub mod lmm;
/// Maximum-likelihood estimation via L-BFGS-B.
pub mod mle;
/// NUTS tree-building and sampling.
pub mod nuts;
/// Ordinary differential equation (ODE) solvers (Phase 13 baseline).
pub mod ode;
/// Generic numerical optimizer (L-BFGS-B backend).
pub mod optimizer;
/// Ordinal regression models (Phase 9 Pack C).
pub mod ordinal;
/// Pharmacometrics models (Phase 13).
pub mod pk;
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
/// Toy-based frequentist inference (CLs).
pub mod toybased;
/// Toy data generation (Asimov + Poisson).
pub mod toys;

/// Differentiable NLL session for PyTorch zero-copy integration (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod differentiable;
/// GPU-accelerated batch toy fitting (requires `cuda` feature + NVIDIA GPU).
#[cfg(feature = "cuda")]
pub mod gpu_batch;
    /// GPU-accelerated single-model fit path (CUDA/Metal).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub mod gpu_session;
/// Metal GPU-accelerated batch toy fitting (requires `metal` feature + Apple Silicon).
#[cfg(feature = "metal")]
pub mod metal_batch;
/// Metal differentiable NLL session for profiled fitting (requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_differentiable;
/// Bijective transforms for unconstrained parameterisation.
pub mod transforms;

#[cfg(test)]
mod universal_tests;

pub use batch::{fit_toys_batch, is_accelerate_available};
pub use builder::{ComposedGlmModel, ModelBuilder};
pub use chain::{SamplerResult, sample_nuts_multichain};
pub use diagnostics::DiagnosticsResult;
#[cfg(feature = "cuda")]
pub use differentiable::{DifferentiableSession, ProfiledDifferentiableSession};
#[cfg(feature = "cuda")]
pub use gpu_batch::{fit_toys_batch_gpu, fit_toys_from_data_gpu, is_cuda_available};
    #[cfg(feature = "cuda")]
    pub use gpu_session::{CudaGpuSession, cuda_session, is_cuda_single_available};
    #[cfg(feature = "metal")]
    pub use gpu_session::{MetalGpuSession, metal_session, is_metal_single_available};
#[cfg(feature = "cuda")]
pub use mle::ranking_gpu;
#[cfg(feature = "metal")]
pub use mle::ranking_metal;
pub use hypotest::{AsymptoticCLsContext, HypotestResult};
pub use laplace::{LaplaceResult, laplace_log_marginal};
pub use lmm::{LmmMarginalModel, RandomEffects as LmmRandomEffects};
#[cfg(feature = "metal")]
pub use metal_batch::{fit_toys_batch_metal, is_metal_available};
pub use mle::{MaximumLikelihoodEstimator, RankingEntry};
pub use nuts::{NutsConfig, sample_nuts};
pub use ode::{OdeSolution, rk4_linear};
pub use optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
pub use ordinal::{OrderedLogitModel, OrderedProbitModel};
pub use pk::{LloqPolicy, OneCompartmentOralPkModel, OneCompartmentOralPkNlmeModel};
pub use posterior::{Posterior, Prior};
#[cfg(feature = "cuda")]
pub use profile_likelihood::scan_gpu;
pub use profile_likelihood::{ProfileLikelihoodScan, ProfilePoint, scan_histfactory};
pub use regression::{
    LinearRegressionModel, LogisticRegressionModel, PoissonRegressionModel, ols_fit,
};
pub use survival::{
    CoxPhModel, CoxTies, ExponentialSurvivalModel, LogNormalAftModel, WeibullSurvivalModel,
};
pub use toybased::{
    ToyHypotestExpectedSet, ToyHypotestResult, hypotest_qtilde_toys,
    hypotest_qtilde_toys_expected_set,
};
#[cfg(feature = "metal")]
pub use metal_differentiable::MetalProfiledDifferentiableSession;
#[cfg(any(feature = "metal", feature = "cuda"))]
pub use toybased::{hypotest_qtilde_toys_expected_set_gpu, hypotest_qtilde_toys_gpu};
pub use toys::{asimov_main, poisson_main_from_expected, poisson_main_toys};
pub use transforms::ParameterTransform;
/// Model builder (composition) MVP for general statistics.
pub mod builder;
