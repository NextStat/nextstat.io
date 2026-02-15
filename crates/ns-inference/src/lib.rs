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
/// Standardized NLME artifact schema and export functions.
pub mod artifacts;
/// Batch toy fitting with optional Accelerate-optimized NLL.
pub mod batch;
/// Model builder (composition) MVP for general statistics.
pub mod builder;
/// MCMC chain storage + I/O.
pub mod chain;
/// Chain Ladder and Mack stochastic reserving (Phase 9 Cross-Vertical).
pub mod chain_ladder;
/// Subscription / Churn vertical pack (survival + causal workflows).
pub mod churn;
/// Competing risks analysis: CIF, Gray's test, Fine-Gray regression (Phase 9 Cross-Vertical).
pub mod competing_risks;
/// MCMC diagnostics: split R-hat, bulk/tail ESS.
pub mod diagnostics;
/// Dosing regimen abstraction for pharmacometric models (Phase 13).
pub mod dosing;
/// Econometrics & causal inference (Phase 12): panel FE, DiD, IV/2SLS, AIPW.
pub mod econometrics;
/// Eight Schools hierarchical model (non-centered parameterization).
pub mod eight_schools;
/// Extreme Value Theory: GEV (block maxima) and GPD (peaks over threshold).
pub mod evt;
/// Monte Carlo fault-tree engine for aviation reliability analysis.
pub mod fault_tree_mc;
/// FOCE/FOCEI estimation for population PK models (Phase 13).
pub mod foce;
/// HMC leapfrog integrator.
pub mod hmc;
/// Hybrid likelihood: combine two LogDensityModel implementations with shared parameters.
pub mod hybrid;
/// Frequentist hypothesis testing (CLs): asymptotic + toy-based.
pub mod hypotest;
/// Laplace approximation utilities (generic).
pub mod laplace;
/// LAPS: Late-Adjusted Parallel Sampler — GPU MAMS on CUDA.
#[cfg(feature = "cuda")]
pub mod laps;
/// Shared L-BFGS-B state machine for GPU lockstep optimization.
pub(crate) mod lbfgs;
/// Linear mixed models (marginal likelihood baseline).
pub mod lmm;
/// Metropolis-Adjusted Microcanonical Sampler (MAMS).
pub mod mams;
/// Meta-analysis: fixed-effects and random-effects pooling.
pub mod meta_analysis;
/// Maximum-likelihood estimation via L-BFGS-B.
pub mod mle;
/// NONMEM-format dataset reader for pharmacometric analysis (Phase 13).
pub mod nonmem;
/// NUTS tree-building and sampling.
pub mod nuts;
/// Ordinary differential equation (ODE) solvers (Phase 13 baseline).
pub mod ode;
/// Adaptive ODE solvers for nonlinear PK/PD systems.
pub mod ode_adaptive;
/// Generic numerical optimizer (L-BFGS-B backend).
pub mod optimizer;
/// Ordinal regression models (Phase 9 Pack C).
pub mod ordinal;
/// Pharmacodynamic models (Emax, sigmoid Emax, indirect response).
pub mod pd;
/// Pharmacometrics models (Phase 13).
pub mod pk;
/// Posterior API: log-pdf, gradient, transforms.
pub mod posterior;
/// Profile likelihood scans.
pub mod profile_likelihood;
/// Regression models (general statistics).
pub mod regression;
/// SAEM algorithm for NLME estimation.
pub mod saem;
/// Stepwise Covariate Modeling (SCM) for population PK.
pub mod scm;
/// Group sequential testing: O'Brien-Fleming, Pocock, alpha-spending (Phase 9 Cross-Vertical).
pub mod sequential;
/// Parametric survival models (Phase 9 Pack A).
pub mod survival;
/// Time series and state space models (Phase 8).
pub mod timeseries;
/// Toy-based frequentist inference (CLs).
pub mod toybased;
/// Toy data generation (Asimov + Poisson).
pub mod toys;
/// Gamma and Tweedie GLM families (Phase 9 Cross-Vertical).
pub mod tweedie;
/// Visual Predictive Check (VPC) and GOF diagnostics for population PK.
pub mod vpc;

/// Differentiable NLL session for PyTorch zero-copy integration (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod differentiable;

#[cfg(test)]
pub(crate) mod testutil {
    use std::sync::Mutex;

    /// Global lock for tests that mutate process-wide runtime flags (e.g. `ns_compute::set_eval_mode`).
    ///
    /// Rust tests run concurrently by default; without a lock, concurrent mutations can make
    /// determinism tests flaky.
    pub static RUNTIME_MODE_LOCK: Mutex<()> = Mutex::new(());
}
/// GPU-accelerated batch toy fitting (requires `cuda` feature + NVIDIA GPU).
#[cfg(feature = "cuda")]
pub mod gpu_batch;
/// GPU-accelerated flow PDF session: flow eval + GPU NLL reduction (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod gpu_flow_session;
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
/// CPU batch toy fitting for unbinned models: warm-start, retry, Rayon parallelism.
pub mod unbinned_batch_cpu;
/// GPU-accelerated batch toy fitting for unbinned models (CUDA/Metal).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod unbinned_gpu_batch;

#[cfg(test)]
mod universal_tests;

pub use artifacts::{
    DatasetProvenance, FixedEffectsSummary, NlmeArtifact, RandomEffectsSummary,
    ReferenceToolVersion, RunBundle, SCHEMA_VERSION, ScmArtifact,
};
pub use batch::{fit_toys_batch, is_accelerate_available};
pub use builder::{ComposedGlmModel, ModelBuilder};
pub use chain::{SamplerResult, sample_nuts_multichain};
pub use chain_ladder::{
    BootstrapReserveResult, ChainLadderResult, ChainLadderRow, ClaimsTriangle, MackResult, MackRow,
    bootstrap_reserves, chain_ladder as chain_ladder_fit, mack_chain_ladder,
};
pub use churn::{
    ArmSummary, BootstrapHrResult, CensoringSegment, ChurnDataConfig, ChurnDataset,
    ChurnDiagnosticsReport, ChurnIngestError, ChurnIngestResult, ChurnMappingConfig, ChurnRecord,
    ChurnRiskModel, ChurnUpliftResult, CohortMatrixCell, CohortMatrixRow, CohortRetentionMatrix,
    CorrectionMethod, CovariateBalance, DiagnosticWarning, OverlapDiagnostics, PairwiseComparison,
    PropensityOverlap, RetentionAnalysis, SegmentComparisonReport, SegmentSummary,
    SurvivalDiffAtHorizon, SurvivalUpliftReport, bootstrap_hazard_ratios, churn_diagnostics_report,
    churn_risk_model, churn_uplift, cohort_retention_matrix, compute_rmst, generate_churn_dataset,
    ingest_churn_arrays, retention_analysis, segment_comparison_report, survival_uplift_report,
};
pub use competing_risks::{
    CifEstimate, CifStep, FineGrayResult, GrayTestResult, cumulative_incidence, fine_gray_fit,
    gray_test,
};
pub use diagnostics::{DiagnosticsResult, QualityGates, compute_diagnostics, quality_summary};
#[cfg(feature = "cuda")]
pub use differentiable::{DifferentiableSession, ProfiledDifferentiableSession};
pub use dosing::{DoseEvent, DoseRoute, DosingRegimen};
pub use econometrics::{
    AipwResult, DidResult, EventStudyResult, FirstStageResult, FixedEffectsSolver, IvResult,
    PanelFEResult, RosenbaumResult, aipw_ate, cluster_robust_se, did_canonical, event_study,
    iv_2sls, panel_fe_fit, rosenbaum_bounds,
};
pub use eight_schools::EightSchoolsModel;
pub use evt::{GevModel, GpdModel};
#[cfg(feature = "cuda")]
pub use fault_tree_mc::fault_tree_mc_cuda;
#[cfg(feature = "metal")]
pub use fault_tree_mc::fault_tree_mc_metal;
pub use fault_tree_mc::{
    DEFAULT_CHUNK_SIZE, FailureMode, FaultTreeCeIsConfig, FaultTreeCeIsResult, FaultTreeMcResult,
    FaultTreeNode, FaultTreeSpec, Gate, fault_tree_mc_ce_is, fault_tree_mc_cpu,
};
pub use foce::{FoceConfig, FoceEstimator, FoceResult, OmegaMatrix};
#[cfg(feature = "cuda")]
pub use gpu_batch::{fit_toys_batch_gpu, fit_toys_from_data_gpu, is_cuda_available};
#[cfg(feature = "cuda")]
pub use gpu_session::{CudaGpuSession, cuda_session, is_cuda_single_available};
#[cfg(feature = "metal")]
pub use gpu_session::{MetalGpuSession, is_metal_single_available, metal_session};
pub use hybrid::{HybridLikelihood, SharedParameterMap};
pub use hypotest::{AsymptoticCLsContext, HypotestResult};
pub use laplace::{LaplaceResult, laplace_log_marginal};
#[cfg(feature = "cuda")]
pub use laps::{LapsConfig, LapsModel, LapsResult, sample_laps};
pub use lmm::{LmmMarginalModel, RandomEffects as LmmRandomEffects};
pub use mams::{MamsConfig, sample_mams, sample_mams_multichain};
pub use meta_analysis::{
    ForestRow, Heterogeneity, MetaAnalysisResult, StudyEffect, meta_fixed, meta_random,
};
#[cfg(feature = "metal")]
pub use metal_batch::{fit_toys_batch_metal, is_metal_available};
#[cfg(feature = "metal")]
pub use metal_differentiable::MetalProfiledDifferentiableSession;
#[cfg(feature = "cuda")]
pub use mle::ranking_gpu;
#[cfg(feature = "metal")]
pub use mle::ranking_metal;
pub use mle::{MaximumLikelihoodEstimator, RankingEntry};
pub use nonmem::{NonmemDataset, NonmemRecord};
pub use nuts::{InitStrategy, MetricType, NutsConfig, sample_nuts};
pub use ode::{OdeSolution, rk4_linear};
pub use ode_adaptive::{OdeOptions, OdeSystem, esdirk4, rk45, solve_at_times};
pub use optimizer::{
    LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig, OptimizerStrategy,
    ToyFitConfig,
};
pub use ordinal::{OrderedLogitModel, OrderedProbitModel};
pub use pd::{EmaxModel, IndirectResponseModel, IndirectResponseType, PkPdLink, SigmoidEmaxModel};
pub use pk::{
    ErrorModel, LloqPolicy, OneCompartmentOralPkModel, OneCompartmentOralPkNlmeModel,
    TwoCompartmentIvPkModel, TwoCompartmentOralPkModel,
};
pub use posterior::{Posterior, Prior};
#[cfg(feature = "cuda")]
pub use profile_likelihood::scan_gpu;
pub use profile_likelihood::scan_histfactory_diag;
#[cfg(feature = "metal")]
pub use profile_likelihood::scan_metal;
pub use profile_likelihood::{
    ProfileCiResult, ProfileLikelihoodScan, ProfilePoint, TestStatistic, compute_test_statistic,
    profile_ci, profile_ci_all, scan, scan_histfactory,
};
pub use regression::{
    LinearRegressionModel, LogisticRegressionModel, NegativeBinomialRegressionModel,
    PoissonRegressionModel, ols_fit,
};
pub use saem::{SaemConfig, SaemDiagnostics, SaemEstimator};
pub use scm::{
    CovariateCandidate, CovariateRelationship, ScmConfig, ScmEstimator, ScmResult, ScmStep,
};
pub use sequential::{
    BoundaryType, SequentialDesign, SequentialLook, SequentialTestResult, SpendingFunction,
    alpha_spending_design, group_sequential_design, sequential_test,
};
pub use survival::{
    CensoringType, CoxPhModel, CoxTies, ExponentialSurvivalModel, IntervalCensoredExponentialModel,
    IntervalCensoredLogNormalModel, IntervalCensoredWeibullAftModel, IntervalCensoredWeibullModel,
    KaplanMeierEstimate, KaplanMeierStep, LogNormalAftModel, LogRankResult, WeibullSurvivalModel,
    kaplan_meier, log_rank_test,
};
pub use toybased::{
    ToyHypotestExpectedSet, ToyHypotestResult, hypotest_qtilde_toys,
    hypotest_qtilde_toys_expected_set, hypotest_qtilde_toys_expected_set_with_sampler,
    hypotest_qtilde_toys_with_sampler,
};
#[cfg(any(feature = "metal", feature = "cuda"))]
pub use toybased::{hypotest_qtilde_toys_expected_set_gpu, hypotest_qtilde_toys_gpu};
pub use toys::{asimov_main, poisson_main_from_expected, poisson_main_toys};
pub use transforms::ParameterTransform;
pub use tweedie::{GammaRegressionModel, TweedieRegressionModel};
pub use unbinned_batch_cpu::{UnbinnedToyBatchResult, fit_unbinned_toys_batch_cpu};
pub use vpc::{GofRecord, VpcBin, VpcConfig, VpcResult, gof_1cpt_oral, vpc_1cpt_oral};
