//! Python bindings for NextStat
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::identity_op)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::needless_update)]
#![allow(unused_parens)]

use pyo3::IntoPyObjectExt;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// Re-export types from core crates
use ns_ad::tape::Tape as AdTape;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel, PreparedNll};
use ns_core::{Error as NsError, Result as NsResult};
use ns_inference::OptimizerConfig;
use ns_inference::chain::{Chain as RustChain, SamplerResult as RustSamplerResult};
use ns_inference::chain_ladder::{
    ClaimsTriangle, chain_ladder as rust_chain_ladder, mack_chain_ladder as rust_mack_chain_ladder,
};
use ns_inference::diagnostics::{QualityGates, compute_diagnostics, quality_summary};
use ns_inference::eight_schools::EightSchoolsModel as RustEightSchoolsModel;
use ns_inference::evt::{GevModel as RustGevModel, GpdModel as RustGpdModel};
use ns_inference::hybrid::{HybridLikelihood, SharedParameterMap};
use ns_inference::lmm::{
    LmmMarginalModel as RustLmmMarginalModel, RandomEffects as RustLmmRandomEffects,
};
use ns_inference::meta_analysis::{
    StudyEffect as RustStudyEffect, meta_fixed as rust_meta_fixed, meta_random as rust_meta_random,
};
use ns_inference::mle::{MaximumLikelihoodEstimator as RustMLE, RankingEntry};
use ns_inference::nuts::{InitStrategy, NutsConfig, sample_nuts};
use ns_inference::optimizer::OptimizationResult as RustOptimizationResult;
use ns_inference::regression::NegativeBinomialRegressionModel as RustNegativeBinomialRegressionModel;
use ns_inference::timeseries::em::{
    KalmanEmConfig as RustKalmanEmConfig, kalman_em as rust_kalman_em,
};
use ns_inference::timeseries::forecast::{
    kalman_forecast as rust_kalman_forecast,
    kalman_forecast_intervals as rust_kalman_forecast_intervals,
};
use ns_inference::timeseries::kalman::{
    KalmanModel as RustKalmanModel, kalman_filter as rust_kalman_filter,
    rts_smoother as rust_rts_smoother,
};
use ns_inference::timeseries::simulate::{
    kalman_simulate as rust_kalman_simulate,
    kalman_simulate_with_x0 as rust_kalman_simulate_with_x0,
};
use ns_inference::timeseries::volatility::{
    Garch11Config as RustGarch11Config, SvLogChi2Config as RustSvLogChi2Config,
    garch11_fit as rust_garch11_fit, sv_logchi2_fit as rust_sv_logchi2_fit,
};
use ns_inference::transforms::ParameterTransform;
use ns_inference::tweedie::{
    GammaRegressionModel as RustGammaRegressionModel,
    TweedieRegressionModel as RustTweedieRegressionModel,
};
use ns_inference::{
    ComposedGlmModel as RustComposedGlmModel, CoxPhModel as RustCoxPhModel, CoxTies as RustCoxTies,
    ExponentialSurvivalModel as RustExponentialSurvivalModel,
    LinearRegressionModel as RustLinearRegressionModel, LloqPolicy as RustLloqPolicy,
    LogNormalAftModel as RustLogNormalAftModel,
    LogisticRegressionModel as RustLogisticRegressionModel, ModelBuilder as RustModelBuilder,
    OneCompartmentOralPkModel as RustOneCompartmentOralPkModel,
    OneCompartmentOralPkNlmeModel as RustOneCompartmentOralPkNlmeModel,
    OrderedLogitModel as RustOrderedLogitModel, OrderedProbitModel as RustOrderedProbitModel,
    PoissonRegressionModel as RustPoissonRegressionModel,
    WeibullSurvivalModel as RustWeibullSurvivalModel, hypotest::AsymptoticCLsContext as RustCLsCtx,
    ols_fit as rust_ols_fit, profile_likelihood as pl,
};
use ns_root::RootFile;
use ns_translate::histfactory::from_xml as histfactory_from_xml;
use ns_translate::pyhf::{
    ExpectedChannelSampleYields, ExpectedSampleYields, HistFactoryModel as RustModel,
    ObservedChannelData, Workspace as RustWorkspace,
};
use ns_unbinned::UnbinnedModel as RustUnbinnedModel;
use ns_unbinned::spec as unbinned_spec;

type RustHybridModel = HybridLikelihood<RustModel, RustUnbinnedModel>;
use ns_viz::{ClsCurveArtifact, ProfileCurveArtifact};

fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Fast-path: accept any object supporting the Python buffer protocol (e.g. array('d'),
    // memoryview, numpy.ndarray). This avoids per-element Python float extraction for large
    // parameter vectors (e.g. shapesys gamma parameters per bin).
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        if buf.dimensions() > 1 {
            return Err(PyValueError::new_err(
                "params buffer must be 1D (or scalar), got ndim > 1",
            ));
        }
        let n = buf.item_count();
        let mut out = vec![0.0f64; n];
        buf.copy_to_slice(obj.py(), &mut out)?;
        return Ok(out);
    }

    // Fallback: accept any Python sequence of floats.
    obj.extract::<Vec<f64>>()
}

fn sample_nuts_multichain_with_seeds(
    model: &(impl LogDensityModel),
    n_warmup: usize,
    n_samples: usize,
    seeds: &[u64],
    config: NutsConfig,
) -> NsResult<RustSamplerResult> {
    use rayon::prelude::*;
    let chains: Vec<NsResult<RustChain>> = seeds
        .par_iter()
        .map(|&seed| sample_nuts(model, n_warmup, n_samples, seed, config.clone()))
        .collect();
    let chains: Vec<RustChain> = chains.into_iter().collect::<NsResult<Vec<_>>>()?;
    Ok(RustSamplerResult {
        chains,
        param_names: model.parameter_names(),
        n_warmup,
        n_samples,
        diagnostics: None,
    })
}

fn sampler_result_to_py<'py>(
    py: Python<'py>,
    result: &RustSamplerResult,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
) -> PyResult<Py<PyAny>> {
    let diag = compute_diagnostics(result);
    let param_names = &result.param_names;
    let n_params = param_names.len();

    // Build "posterior" dict: {param_name: [[chain0_draws], [chain1_draws], ...]}
    let posterior = PyDict::new(py);
    for (p, name) in param_names.iter().enumerate() {
        let chains_draws: Vec<Vec<f64>> = result.param_draws(p);
        posterior.set_item(name, chains_draws)?;
    }

    // Build "sample_stats" dict
    let sample_stats = PyDict::new(py);
    let diverging: Vec<Vec<bool>> = result.chains.iter().map(|c| c.divergences.clone()).collect();
    let tree_depth: Vec<Vec<usize>> = result.chains.iter().map(|c| c.tree_depths.clone()).collect();
    let accept_prob: Vec<Vec<f64>> = result.chains.iter().map(|c| c.accept_probs.clone()).collect();
    let energy: Vec<Vec<f64>> = result.chains.iter().map(|c| c.energies.clone()).collect();
    let step_sizes: Vec<f64> = result.chains.iter().map(|c| c.step_size).collect();
    let n_leapfrog: Vec<Vec<usize>> = result.chains.iter().map(|c| c.n_leapfrog.clone()).collect();
    sample_stats.set_item("diverging", diverging)?;
    sample_stats.set_item("tree_depth", tree_depth)?;
    sample_stats.set_item("accept_prob", accept_prob)?;
    sample_stats.set_item("energy", energy)?;
    sample_stats.set_item("step_size", step_sizes)?;
    sample_stats.set_item("n_leapfrog", n_leapfrog)?;

    // Build "diagnostics" dict
    let diagnostics_dict = PyDict::new(py);

    let r_hat_dict = PyDict::new(py);
    let ess_bulk_dict = PyDict::new(py);
    let ess_tail_dict = PyDict::new(py);
    for p in 0..n_params {
        r_hat_dict.set_item(&param_names[p], diag.r_hat[p])?;
        ess_bulk_dict.set_item(&param_names[p], diag.ess_bulk[p])?;
        ess_tail_dict.set_item(&param_names[p], diag.ess_tail[p])?;
    }
    diagnostics_dict.set_item("r_hat", r_hat_dict)?;
    diagnostics_dict.set_item("ess_bulk", ess_bulk_dict)?;
    diagnostics_dict.set_item("ess_tail", ess_tail_dict)?;
    diagnostics_dict.set_item("divergence_rate", diag.divergence_rate)?;
    diagnostics_dict.set_item("max_treedepth_rate", diag.max_treedepth_rate)?;
    diagnostics_dict.set_item("ebfmi", diag.ebfmi.clone())?;

    // Non-slow quality summary (conservative gates).
    let gates = QualityGates::default();
    let qs = quality_summary(&diag, n_chains, n_samples, &gates);
    let quality = PyDict::new(py);
    quality.set_item("status", qs.status.to_string())?;
    quality.set_item("enabled", qs.enabled)?;
    quality.set_item("warnings", qs.warnings)?;
    quality.set_item("failures", qs.failures)?;
    quality.set_item("total_draws", qs.total_draws)?;
    quality.set_item("max_r_hat", qs.max_r_hat)?;
    quality.set_item("min_ess_bulk", qs.min_ess_bulk)?;
    quality.set_item("min_ess_tail", qs.min_ess_tail)?;
    quality.set_item("min_ebfmi", qs.min_ebfmi)?;
    diagnostics_dict.set_item("quality", quality)?;

    // Assemble top-level dict
    let out = PyDict::new(py);
    out.set_item("posterior", posterior)?;
    out.set_item("sample_stats", sample_stats)?;
    out.set_item("diagnostics", diagnostics_dict)?;
    out.set_item("param_names", param_names)?;
    out.set_item("n_chains", n_chains)?;
    out.set_item("n_warmup", n_warmup)?;
    out.set_item("n_samples", n_samples)?;

    Ok(out.into_any().unbind())
}

fn dmatrix_from_nested(name: &str, rows: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    if rows.is_empty() {
        return Err(PyValueError::new_err(format!("{name} must be non-empty")));
    }
    let nrows = rows.len();
    let ncols = rows[0].len();
    if ncols == 0 {
        return Err(PyValueError::new_err(format!("{name} rows must be non-empty")));
    }
    for (i, r) in rows.iter().enumerate() {
        if r.len() != ncols {
            return Err(PyValueError::new_err(format!(
                "{name} must be rectangular: row {i} has len {}, expected {}",
                r.len(),
                ncols
            )));
        }
        if r.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
        }
    }
    let mut flat = Vec::with_capacity(nrows * ncols);
    for r in rows {
        flat.extend_from_slice(&r);
    }
    Ok(DMatrix::from_row_slice(nrows, ncols, &flat))
}

fn dvector_from_vec(name: &str, v: Vec<f64>) -> PyResult<DVector<f64>> {
    if v.is_empty() {
        return Err(PyValueError::new_err(format!("{name} must be non-empty")));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
    }
    Ok(DVector::from_vec(v))
}

fn dvector_from_opt_vec(name: &str, v: Vec<Option<f64>>) -> PyResult<DVector<f64>> {
    if v.is_empty() {
        return Err(PyValueError::new_err(format!("{name} must be non-empty")));
    }
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        match x {
            Some(v) if v.is_finite() => out.push(v),
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain only finite values or None"
                )));
            }
            None => out.push(f64::NAN),
        }
    }
    Ok(DVector::from_vec(out))
}

fn dvector_to_vec(v: &DVector<f64>) -> Vec<f64> {
    v.iter().copied().collect()
}

fn dmatrix_to_nested(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut out = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let mut row = Vec::with_capacity(ncols);
        for j in 0..ncols {
            row.push(m[(i, j)]);
        }
        out.push(row);
    }
    out
}

// ---------------------------------------------------------------------------
// Bayesian posterior surface (Phase 3/5 standards: explicit Posterior API).
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PosteriorModel {
    HistFactory(RustModel),
    Unbinned(RustUnbinnedModel),
    Hybrid(RustHybridModel),
    GaussianMean(GaussianMeanModel),
    Funnel(FunnelModel),
    StdNormal(StdNormalModel),
    LinearRegression(RustLinearRegressionModel),
    LogisticRegression(RustLogisticRegressionModel),
    OrderedLogit(RustOrderedLogitModel),
    OrderedProbit(RustOrderedProbitModel),
    PoissonRegression(RustPoissonRegressionModel),
    NegativeBinomialRegression(RustNegativeBinomialRegressionModel),
    ComposedGlm(RustComposedGlmModel),
    LmmMarginal(RustLmmMarginalModel),
    ExponentialSurvival(RustExponentialSurvivalModel),
    WeibullSurvival(RustWeibullSurvivalModel),
    LogNormalAft(RustLogNormalAftModel),
    CoxPh(RustCoxPhModel),
    OneCompartmentOralPk(RustOneCompartmentOralPkModel),
    OneCompartmentOralPkNlme(RustOneCompartmentOralPkNlmeModel),
    GammaRegression(RustGammaRegressionModel),
    TweedieRegression(RustTweedieRegressionModel),
    Gev(RustGevModel),
    Gpd(RustGpdModel),
    EightSchools(RustEightSchoolsModel),
}

impl PosteriorModel {
    fn dim(&self) -> usize {
        match self {
            PosteriorModel::HistFactory(m) => m.dim(),
            PosteriorModel::Unbinned(m) => m.dim(),
            PosteriorModel::Hybrid(m) => m.dim(),
            PosteriorModel::GaussianMean(m) => m.dim(),
            PosteriorModel::Funnel(m) => m.dim(),
            PosteriorModel::StdNormal(m) => m.dim(),
            PosteriorModel::LinearRegression(m) => m.dim(),
            PosteriorModel::LogisticRegression(m) => m.dim(),
            PosteriorModel::OrderedLogit(m) => m.dim(),
            PosteriorModel::OrderedProbit(m) => m.dim(),
            PosteriorModel::PoissonRegression(m) => m.dim(),
            PosteriorModel::NegativeBinomialRegression(m) => m.dim(),
            PosteriorModel::ComposedGlm(m) => m.dim(),
            PosteriorModel::LmmMarginal(m) => m.dim(),
            PosteriorModel::ExponentialSurvival(m) => m.dim(),
            PosteriorModel::WeibullSurvival(m) => m.dim(),
            PosteriorModel::LogNormalAft(m) => m.dim(),
            PosteriorModel::CoxPh(m) => m.dim(),
            PosteriorModel::OneCompartmentOralPk(m) => m.dim(),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.dim(),
            PosteriorModel::GammaRegression(m) => m.dim(),
            PosteriorModel::TweedieRegression(m) => m.dim(),
            PosteriorModel::Gev(m) => m.dim(),
            PosteriorModel::Gpd(m) => m.dim(),
            PosteriorModel::EightSchools(m) => m.dim(),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_names(),
            PosteriorModel::Unbinned(m) => m.parameter_names(),
            PosteriorModel::Hybrid(m) => m.parameter_names(),
            PosteriorModel::GaussianMean(m) => m.parameter_names(),
            PosteriorModel::Funnel(m) => m.parameter_names(),
            PosteriorModel::StdNormal(m) => m.parameter_names(),
            PosteriorModel::LinearRegression(m) => m.parameter_names(),
            PosteriorModel::LogisticRegression(m) => m.parameter_names(),
            PosteriorModel::OrderedLogit(m) => m.parameter_names(),
            PosteriorModel::OrderedProbit(m) => m.parameter_names(),
            PosteriorModel::PoissonRegression(m) => m.parameter_names(),
            PosteriorModel::NegativeBinomialRegression(m) => m.parameter_names(),
            PosteriorModel::ComposedGlm(m) => m.parameter_names(),
            PosteriorModel::LmmMarginal(m) => m.parameter_names(),
            PosteriorModel::ExponentialSurvival(m) => m.parameter_names(),
            PosteriorModel::WeibullSurvival(m) => m.parameter_names(),
            PosteriorModel::LogNormalAft(m) => m.parameter_names(),
            PosteriorModel::CoxPh(m) => m.parameter_names(),
            PosteriorModel::OneCompartmentOralPk(m) => m.parameter_names(),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.parameter_names(),
            PosteriorModel::GammaRegression(m) => m.parameter_names(),
            PosteriorModel::TweedieRegression(m) => m.parameter_names(),
            PosteriorModel::Gev(m) => m.parameter_names(),
            PosteriorModel::Gpd(m) => m.parameter_names(),
            PosteriorModel::EightSchools(m) => m.parameter_names(),
        }
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_bounds(),
            PosteriorModel::Unbinned(m) => m.parameter_bounds(),
            PosteriorModel::Hybrid(m) => m.parameter_bounds(),
            PosteriorModel::GaussianMean(m) => m.parameter_bounds(),
            PosteriorModel::Funnel(m) => m.parameter_bounds(),
            PosteriorModel::StdNormal(m) => m.parameter_bounds(),
            PosteriorModel::LinearRegression(m) => m.parameter_bounds(),
            PosteriorModel::LogisticRegression(m) => m.parameter_bounds(),
            PosteriorModel::OrderedLogit(m) => m.parameter_bounds(),
            PosteriorModel::OrderedProbit(m) => m.parameter_bounds(),
            PosteriorModel::PoissonRegression(m) => m.parameter_bounds(),
            PosteriorModel::NegativeBinomialRegression(m) => m.parameter_bounds(),
            PosteriorModel::ComposedGlm(m) => m.parameter_bounds(),
            PosteriorModel::LmmMarginal(m) => m.parameter_bounds(),
            PosteriorModel::ExponentialSurvival(m) => m.parameter_bounds(),
            PosteriorModel::WeibullSurvival(m) => m.parameter_bounds(),
            PosteriorModel::LogNormalAft(m) => m.parameter_bounds(),
            PosteriorModel::CoxPh(m) => m.parameter_bounds(),
            PosteriorModel::OneCompartmentOralPk(m) => m.parameter_bounds(),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.parameter_bounds(),
            PosteriorModel::GammaRegression(m) => m.parameter_bounds(),
            PosteriorModel::TweedieRegression(m) => m.parameter_bounds(),
            PosteriorModel::Gev(m) => m.parameter_bounds(),
            PosteriorModel::Gpd(m) => m.parameter_bounds(),
            PosteriorModel::EightSchools(m) => m.parameter_bounds(),
        }
    }

    fn parameter_init(&self) -> Vec<f64> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_init(),
            PosteriorModel::Unbinned(m) => m.parameter_init(),
            PosteriorModel::Hybrid(m) => m.parameter_init(),
            PosteriorModel::GaussianMean(m) => m.parameter_init(),
            PosteriorModel::Funnel(m) => m.parameter_init(),
            PosteriorModel::StdNormal(m) => m.parameter_init(),
            PosteriorModel::LinearRegression(m) => m.parameter_init(),
            PosteriorModel::LogisticRegression(m) => m.parameter_init(),
            PosteriorModel::OrderedLogit(m) => m.parameter_init(),
            PosteriorModel::OrderedProbit(m) => m.parameter_init(),
            PosteriorModel::PoissonRegression(m) => m.parameter_init(),
            PosteriorModel::NegativeBinomialRegression(m) => m.parameter_init(),
            PosteriorModel::ComposedGlm(m) => m.parameter_init(),
            PosteriorModel::LmmMarginal(m) => m.parameter_init(),
            PosteriorModel::ExponentialSurvival(m) => m.parameter_init(),
            PosteriorModel::WeibullSurvival(m) => m.parameter_init(),
            PosteriorModel::LogNormalAft(m) => m.parameter_init(),
            PosteriorModel::CoxPh(m) => m.parameter_init(),
            PosteriorModel::OneCompartmentOralPk(m) => m.parameter_init(),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.parameter_init(),
            PosteriorModel::GammaRegression(m) => m.parameter_init(),
            PosteriorModel::TweedieRegression(m) => m.parameter_init(),
            PosteriorModel::Gev(m) => m.parameter_init(),
            PosteriorModel::Gpd(m) => m.parameter_init(),
            PosteriorModel::EightSchools(m) => m.parameter_init(),
        }
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        match self {
            PosteriorModel::HistFactory(m) => m.nll(params),
            PosteriorModel::Unbinned(m) => m.nll(params),
            PosteriorModel::Hybrid(m) => m.nll(params),
            PosteriorModel::GaussianMean(m) => m.nll(params),
            PosteriorModel::Funnel(m) => m.nll(params),
            PosteriorModel::StdNormal(m) => m.nll(params),
            PosteriorModel::LinearRegression(m) => m.nll(params),
            PosteriorModel::LogisticRegression(m) => m.nll(params),
            PosteriorModel::OrderedLogit(m) => m.nll(params),
            PosteriorModel::OrderedProbit(m) => m.nll(params),
            PosteriorModel::PoissonRegression(m) => m.nll(params),
            PosteriorModel::NegativeBinomialRegression(m) => m.nll(params),
            PosteriorModel::ComposedGlm(m) => m.nll(params),
            PosteriorModel::LmmMarginal(m) => m.nll(params),
            PosteriorModel::ExponentialSurvival(m) => m.nll(params),
            PosteriorModel::WeibullSurvival(m) => m.nll(params),
            PosteriorModel::LogNormalAft(m) => m.nll(params),
            PosteriorModel::CoxPh(m) => m.nll(params),
            PosteriorModel::OneCompartmentOralPk(m) => m.nll(params),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.nll(params),
            PosteriorModel::GammaRegression(m) => m.nll(params),
            PosteriorModel::TweedieRegression(m) => m.nll(params),
            PosteriorModel::Gev(m) => m.nll(params),
            PosteriorModel::Gpd(m) => m.nll(params),
            PosteriorModel::EightSchools(m) => m.nll(params),
        }
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        match self {
            PosteriorModel::HistFactory(m) => m.grad_nll(params),
            PosteriorModel::Unbinned(m) => m.grad_nll(params),
            PosteriorModel::Hybrid(m) => m.grad_nll(params),
            PosteriorModel::GaussianMean(m) => m.grad_nll(params),
            PosteriorModel::Funnel(m) => m.grad_nll(params),
            PosteriorModel::StdNormal(m) => m.grad_nll(params),
            PosteriorModel::LinearRegression(m) => m.grad_nll(params),
            PosteriorModel::LogisticRegression(m) => m.grad_nll(params),
            PosteriorModel::OrderedLogit(m) => m.grad_nll(params),
            PosteriorModel::OrderedProbit(m) => m.grad_nll(params),
            PosteriorModel::PoissonRegression(m) => m.grad_nll(params),
            PosteriorModel::NegativeBinomialRegression(m) => m.grad_nll(params),
            PosteriorModel::ComposedGlm(m) => m.grad_nll(params),
            PosteriorModel::LmmMarginal(m) => m.grad_nll(params),
            PosteriorModel::ExponentialSurvival(m) => m.grad_nll(params),
            PosteriorModel::WeibullSurvival(m) => m.grad_nll(params),
            PosteriorModel::LogNormalAft(m) => m.grad_nll(params),
            PosteriorModel::CoxPh(m) => m.grad_nll(params),
            PosteriorModel::OneCompartmentOralPk(m) => m.grad_nll(params),
            PosteriorModel::OneCompartmentOralPkNlme(m) => m.grad_nll(params),
            PosteriorModel::GammaRegression(m) => m.grad_nll(params),
            PosteriorModel::TweedieRegression(m) => m.grad_nll(params),
            PosteriorModel::Gev(m) => m.grad_nll(params),
            PosteriorModel::Gpd(m) => m.grad_nll(params),
            PosteriorModel::EightSchools(m) => m.grad_nll(params),
        }
    }

    fn fit_mle(&self, mle: &RustMLE) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => mle.fit(m),
            PosteriorModel::Unbinned(m) => mle.fit(m),
            PosteriorModel::Hybrid(m) => mle.fit(m),
            PosteriorModel::GaussianMean(m) => mle.fit(m),
            PosteriorModel::Funnel(m) => mle.fit(m),
            PosteriorModel::StdNormal(m) => mle.fit(m),
            PosteriorModel::LinearRegression(m) => mle.fit(m),
            PosteriorModel::LogisticRegression(m) => mle.fit(m),
            PosteriorModel::OrderedLogit(m) => mle.fit(m),
            PosteriorModel::OrderedProbit(m) => mle.fit(m),
            PosteriorModel::PoissonRegression(m) => mle.fit(m),
            PosteriorModel::NegativeBinomialRegression(m) => mle.fit(m),
            PosteriorModel::ComposedGlm(m) => mle.fit(m),
            PosteriorModel::LmmMarginal(m) => mle.fit(m),
            PosteriorModel::ExponentialSurvival(m) => mle.fit(m),
            PosteriorModel::WeibullSurvival(m) => mle.fit(m),
            PosteriorModel::LogNormalAft(m) => mle.fit(m),
            PosteriorModel::CoxPh(m) => mle.fit(m),
            PosteriorModel::OneCompartmentOralPk(m) => mle.fit(m),
            PosteriorModel::OneCompartmentOralPkNlme(m) => mle.fit(m),
            PosteriorModel::GammaRegression(m) => mle.fit(m),
            PosteriorModel::TweedieRegression(m) => mle.fit(m),
            PosteriorModel::Gev(m) => mle.fit(m),
            PosteriorModel::Gpd(m) => mle.fit(m),
            PosteriorModel::EightSchools(m) => mle.fit(m),
        }
    }

    fn fit_mle_from(&self, mle: &RustMLE, init_pars: &[f64]) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => mle.fit_from(m, init_pars),
            PosteriorModel::Unbinned(m) => mle.fit_from(m, init_pars),
            PosteriorModel::Hybrid(m) => mle.fit_from(m, init_pars),
            PosteriorModel::GaussianMean(m) => mle.fit_from(m, init_pars),
            PosteriorModel::Funnel(m) => mle.fit_from(m, init_pars),
            PosteriorModel::StdNormal(m) => mle.fit_from(m, init_pars),
            PosteriorModel::LinearRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::LogisticRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::OrderedLogit(m) => mle.fit_from(m, init_pars),
            PosteriorModel::OrderedProbit(m) => mle.fit_from(m, init_pars),
            PosteriorModel::PoissonRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::NegativeBinomialRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::ComposedGlm(m) => mle.fit_from(m, init_pars),
            PosteriorModel::LmmMarginal(m) => mle.fit_from(m, init_pars),
            PosteriorModel::ExponentialSurvival(m) => mle.fit_from(m, init_pars),
            PosteriorModel::WeibullSurvival(m) => mle.fit_from(m, init_pars),
            PosteriorModel::LogNormalAft(m) => mle.fit_from(m, init_pars),
            PosteriorModel::CoxPh(m) => mle.fit_from(m, init_pars),
            PosteriorModel::OneCompartmentOralPk(m) => mle.fit_from(m, init_pars),
            PosteriorModel::OneCompartmentOralPkNlme(m) => mle.fit_from(m, init_pars),
            PosteriorModel::GammaRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::TweedieRegression(m) => mle.fit_from(m, init_pars),
            PosteriorModel::Gev(m) => mle.fit_from(m, init_pars),
            PosteriorModel::Gpd(m) => mle.fit_from(m, init_pars),
            PosteriorModel::EightSchools(m) => mle.fit_from(m, init_pars),
        }
    }

    fn sample_nuts_multichain(
        &self,
        n_chains: usize,
        n_warmup: usize,
        n_samples: usize,
        seed: u64,
        config: NutsConfig,
    ) -> NsResult<RustSamplerResult> {
        let seeds: Vec<u64> =
            (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
        match self {
            PosteriorModel::HistFactory(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Unbinned(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Hybrid(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::GaussianMean(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Funnel(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::StdNormal(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LinearRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LogisticRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OrderedLogit(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OrderedProbit(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::PoissonRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::NegativeBinomialRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::ComposedGlm(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LmmMarginal(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::ExponentialSurvival(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::WeibullSurvival(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LogNormalAft(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::CoxPh(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OneCompartmentOralPk(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OneCompartmentOralPkNlme(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::GammaRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::TweedieRegression(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Gev(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Gpd(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::EightSchools(m) => {
                sample_nuts_multichain_with_seeds(m, n_warmup, n_samples, &seeds, config)
            }
        }
    }

    fn fit_map(&self, mle: &RustMLE, priors: Vec<Prior>) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::Unbinned(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::Hybrid(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::GaussianMean(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::Funnel(m) => {
                let w = WithPriors { model: *m, priors };
                mle.fit(&w)
            }
            PosteriorModel::StdNormal(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::LinearRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::LogisticRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::OrderedLogit(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::OrderedProbit(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::PoissonRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::NegativeBinomialRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::ComposedGlm(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::LmmMarginal(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::ExponentialSurvival(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::WeibullSurvival(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::LogNormalAft(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::CoxPh(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::OneCompartmentOralPk(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::OneCompartmentOralPkNlme(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::GammaRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::TweedieRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::Gev(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::Gpd(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
            PosteriorModel::EightSchools(m) => {
                let w = WithPriors { model: m.clone(), priors };
                mle.fit(&w)
            }
        }
    }

    fn sample_nuts_multichain_map(
        &self,
        n_chains: usize,
        n_warmup: usize,
        n_samples: usize,
        seed: u64,
        config: NutsConfig,
        priors: Vec<Prior>,
    ) -> NsResult<RustSamplerResult> {
        let seeds: Vec<u64> =
            (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
        match self {
            PosteriorModel::HistFactory(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Unbinned(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Hybrid(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::GaussianMean(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Funnel(m) => {
                let w = WithPriors { model: *m, priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::StdNormal(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LinearRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LogisticRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OrderedLogit(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OrderedProbit(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::PoissonRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::NegativeBinomialRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::ComposedGlm(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LmmMarginal(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::ExponentialSurvival(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::WeibullSurvival(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::LogNormalAft(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::CoxPh(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OneCompartmentOralPk(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::OneCompartmentOralPkNlme(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::GammaRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::TweedieRegression(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Gev(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::Gpd(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
            PosteriorModel::EightSchools(m) => {
                let w = WithPriors { model: m.clone(), priors };
                sample_nuts_multichain_with_seeds(&w, n_warmup, n_samples, &seeds, config)
            }
        }
    }
}

fn extract_posterior_model(model: &Bound<'_, PyAny>) -> PyResult<PosteriorModel> {
    if let Ok(hf) = model.extract::<PyRef<'_, PyHistFactoryModel>>() {
        Ok(PosteriorModel::HistFactory(hf.inner.clone()))
    } else if let Ok(ub) = model.extract::<PyRef<'_, PyUnbinnedModel>>() {
        Ok(PosteriorModel::Unbinned(ub.inner.clone()))
    } else if let Ok(hm) = model.extract::<PyRef<'_, PyHybridModel>>() {
        Ok(PosteriorModel::Hybrid(hm.inner.clone()))
    } else if let Ok(gm) = model.extract::<PyRef<'_, PyGaussianMeanModel>>() {
        Ok(PosteriorModel::GaussianMean(gm.inner.clone()))
    } else if let Ok(fm) = model.extract::<PyRef<'_, PyFunnelModel>>() {
        Ok(PosteriorModel::Funnel(fm.inner))
    } else if let Ok(sm) = model.extract::<PyRef<'_, PyStdNormalModel>>() {
        Ok(PosteriorModel::StdNormal(sm.inner.clone()))
    } else if let Ok(lr) = model.extract::<PyRef<'_, PyLinearRegressionModel>>() {
        Ok(PosteriorModel::LinearRegression(lr.inner.clone()))
    } else if let Ok(logit) = model.extract::<PyRef<'_, PyLogisticRegressionModel>>() {
        Ok(PosteriorModel::LogisticRegression(logit.inner.clone()))
    } else if let Ok(ord) = model.extract::<PyRef<'_, PyOrderedLogitModel>>() {
        Ok(PosteriorModel::OrderedLogit(ord.inner.clone()))
    } else if let Ok(ord) = model.extract::<PyRef<'_, PyOrderedProbitModel>>() {
        Ok(PosteriorModel::OrderedProbit(ord.inner.clone()))
    } else if let Ok(pois) = model.extract::<PyRef<'_, PyPoissonRegressionModel>>() {
        Ok(PosteriorModel::PoissonRegression(pois.inner.clone()))
    } else if let Ok(nb) = model.extract::<PyRef<'_, PyNegativeBinomialRegressionModel>>() {
        Ok(PosteriorModel::NegativeBinomialRegression(nb.inner.clone()))
    } else if let Ok(glm) = model.extract::<PyRef<'_, PyComposedGlmModel>>() {
        Ok(PosteriorModel::ComposedGlm(glm.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyLmmMarginalModel>>() {
        Ok(PosteriorModel::LmmMarginal(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyExponentialSurvivalModel>>() {
        Ok(PosteriorModel::ExponentialSurvival(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyWeibullSurvivalModel>>() {
        Ok(PosteriorModel::WeibullSurvival(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyLogNormalAftModel>>() {
        Ok(PosteriorModel::LogNormalAft(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyCoxPhModel>>() {
        Ok(PosteriorModel::CoxPh(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyOneCompartmentOralPkModel>>() {
        Ok(PosteriorModel::OneCompartmentOralPk(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyOneCompartmentOralPkNlmeModel>>() {
        Ok(PosteriorModel::OneCompartmentOralPkNlme(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyGammaRegressionModel>>() {
        Ok(PosteriorModel::GammaRegression(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyTweedieRegressionModel>>() {
        Ok(PosteriorModel::TweedieRegression(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyGevModel>>() {
        Ok(PosteriorModel::Gev(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyGpdModel>>() {
        Ok(PosteriorModel::Gpd(m.inner.clone()))
    } else if let Ok(m) = model.extract::<PyRef<'_, PyEightSchoolsModel>>() {
        Ok(PosteriorModel::EightSchools(m.inner.clone()))
    } else {
        Err(PyValueError::new_err(
            "Unsupported model type. Expected HistFactoryModel, UnbinnedModel, GaussianMeanModel, FunnelModel, StdNormalModel, a regression model, OrderedLogitModel, OrderedProbitModel, ComposedGlmModel, LmmMarginalModel, a survival model, a PK model, GammaRegressionModel, TweedieRegressionModel, GevModel, GpdModel, or EightSchoolsModel.",
        ))
    }
}

fn extract_posterior_model_with_data(
    model: &Bound<'_, PyAny>,
    data: Option<Vec<f64>>,
) -> PyResult<PosteriorModel> {
    if let Ok(hf) = model.extract::<PyRef<'_, PyHistFactoryModel>>() {
        let m = if let Some(obs_main) = data {
            hf.inner
                .with_observed_main(&obs_main)
                .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
        } else {
            hf.inner.clone()
        };
        Ok(PosteriorModel::HistFactory(m))
    } else {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        extract_posterior_model(model)
    }
}

fn validate_f64_vec(name: &str, xs: &[f64], dim: usize) -> PyResult<()> {
    if xs.len() != dim {
        return Err(PyValueError::new_err(format!(
            "expected {name} length = model.dim() = {dim}, got {}",
            xs.len()
        )));
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
    }
    Ok(())
}

fn validate_bounds(name: &str, bounds: &[(f64, f64)], dim: usize) -> PyResult<()> {
    if bounds.len() != dim {
        return Err(PyValueError::new_err(format!(
            "expected {name} length = model.dim() = {dim}, got {}",
            bounds.len()
        )));
    }
    for (i, (lo, hi)) in bounds.iter().copied().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name}[{i}] bounds must be finite, got ({lo}, {hi})"
            )));
        }
        if lo > hi {
            return Err(PyValueError::new_err(format!(
                "{name}[{i}] invalid bounds: lo > hi, got ({lo}, {hi})"
            )));
        }
    }
    Ok(())
}

fn validate_nuts_config(
    n_chains: usize,
    _n_warmup: usize,
    n_samples: usize,
    max_treedepth: usize,
    target_accept: f64,
    init_jitter: f64,
    init_jitter_rel: Option<f64>,
    init_overdispersed_rel: Option<f64>,
) -> PyResult<()> {
    if n_chains == 0 {
        return Err(PyValueError::new_err("n_chains must be >= 1"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be >= 1"));
    }
    if max_treedepth == 0 {
        return Err(PyValueError::new_err("max_treedepth must be >= 1"));
    }
    if !(target_accept.is_finite() && 0.0 < target_accept && target_accept < 1.0) {
        return Err(PyValueError::new_err("target_accept must be finite and in (0,1)"));
    }

    if !init_jitter.is_finite() || init_jitter < 0.0 {
        return Err(PyValueError::new_err("init_jitter must be finite and >= 0"));
    }
    if let Some(v) = init_jitter_rel
        && (!v.is_finite() || v <= 0.0)
    {
        return Err(PyValueError::new_err("init_jitter_rel must be finite and > 0"));
    }
    if let Some(v) = init_overdispersed_rel
        && (!v.is_finite() || v <= 0.0)
    {
        return Err(PyValueError::new_err("init_overdispersed_rel must be finite and > 0"));
    }

    let init_modes = (init_jitter > 0.0) as u8
        + init_jitter_rel.is_some() as u8
        + init_overdispersed_rel.is_some() as u8;
    if init_modes > 1 {
        return Err(PyValueError::new_err(
            "init_jitter, init_jitter_rel, init_overdispersed_rel are mutually exclusive",
        ));
    }
    Ok(())
}

/// Posterior wrapper: provides constrained and unconstrained log-density evaluation.
///
/// Notes:
/// - Uses the model's `nll`/`grad_nll`. For HistFactory, constraint terms are already included
///   in `nll` (pyhf parity baseline), and Posterior does not double-count them.
#[pyclass(name = "Posterior")]
struct PyPosterior {
    model: PosteriorModel,
    bounds: Vec<(f64, f64)>,
    transform: ParameterTransform,
    dim: usize,
    names: Vec<String>,
    name_to_idx: HashMap<String, usize>,
    priors: Vec<Prior>,
}

#[derive(Debug, Clone)]
enum Prior {
    Flat,
    Normal { center: f64, width: f64 },
}

impl Prior {
    fn logpdf(&self, theta: f64) -> PyResult<f64> {
        match self {
            Prior::Flat => Ok(0.0),
            Prior::Normal { center, width } => {
                if !width.is_finite() || *width <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "Normal prior width must be finite and > 0, got {}",
                        width
                    )));
                }
                let pull = (theta - center) / width;
                Ok(-0.5 * pull * pull)
            }
        }
    }

    fn grad(&self, theta: f64) -> PyResult<f64> {
        match self {
            Prior::Flat => Ok(0.0),
            Prior::Normal { center, width } => {
                if !width.is_finite() || *width <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "Normal prior width must be finite and > 0, got {}",
                        width
                    )));
                }
                Ok(-(theta - center) / (width * width))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WithPriors<M> {
    model: M,
    priors: Vec<Prior>,
}

impl<M: LogDensityModel> LogDensityModel for WithPriors<M> {
    type Prepared<'a>
        = ns_core::traits::PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.model.parameter_names()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.model.parameter_bounds()
    }

    fn parameter_init(&self) -> Vec<f64> {
        self.model.parameter_init()
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        if self.priors.len() != params.len() {
            return Err(NsError::Validation(format!(
                "priors length must match params length: {} != {}",
                self.priors.len(),
                params.len()
            )));
        }
        let mut out = self.model.nll(params)?;
        for (i, pr) in self.priors.iter().enumerate() {
            match pr {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    if !width.is_finite() || *width <= 0.0 {
                        return Err(NsError::Validation(format!(
                            "Normal prior width must be finite and > 0, got {}",
                            width
                        )));
                    }
                    let pull = (params[i] - center) / width;
                    out += 0.5 * pull * pull;
                }
            }
        }
        Ok(out)
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        if self.priors.len() != params.len() {
            return Err(NsError::Validation(format!(
                "priors length must match params length: {} != {}",
                self.priors.len(),
                params.len()
            )));
        }
        let mut g = self.model.grad_nll(params)?;
        for (i, pr) in self.priors.iter().enumerate() {
            match pr {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    if !width.is_finite() || *width <= 0.0 {
                        return Err(NsError::Validation(format!(
                            "Normal prior width must be finite and > 0, got {}",
                            width
                        )));
                    }
                    g[i] += (params[i] - center) / (width * width);
                }
            }
        }
        Ok(g)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        ns_core::traits::PreparedModelRef::new(self)
    }
}

#[pymethods]
impl PyPosterior {
    #[new]
    fn new(model: &Bound<'_, PyAny>) -> PyResult<Self> {
        let model = extract_posterior_model(model)?;
        let dim = model.dim();

        // Defensive: keep Posterior well-defined even if bounds length != dim.
        let mut bounds: Vec<(f64, f64)> = model.parameter_bounds();
        if bounds.len() > dim {
            bounds.truncate(dim);
        } else if bounds.len() < dim {
            bounds.resize(dim, (f64::NEG_INFINITY, f64::INFINITY));
        }

        let transform = ParameterTransform::from_bounds(&bounds);
        let names = model.parameter_names();
        let mut name_to_idx = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        let priors = vec![Prior::Flat; dim];
        Ok(Self { model, bounds, transform, dim, names, name_to_idx, priors })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn parameter_names(&self) -> Vec<String> {
        self.names.clone()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.model.parameter_init()
    }

    fn clear_priors(&mut self) {
        self.priors.fill(Prior::Flat);
    }

    fn set_prior_flat(&mut self, name: String) -> PyResult<()> {
        let idx = self.param_idx(&name)?;
        self.priors[idx] = Prior::Flat;
        Ok(())
    }

    fn set_prior_normal(&mut self, name: String, center: f64, width: f64) -> PyResult<()> {
        let idx = self.param_idx(&name)?;
        if !center.is_finite() {
            return Err(PyValueError::new_err("Normal prior center must be finite"));
        }
        if !width.is_finite() || width <= 0.0 {
            return Err(PyValueError::new_err("Normal prior width must be finite and > 0"));
        }
        self.priors[idx] = Prior::Normal { center, width };
        Ok(())
    }

    fn priors<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let out = PyDict::new(py);
        for (i, name) in self.names.iter().enumerate() {
            let d = PyDict::new(py);
            match &self.priors[i] {
                Prior::Flat => {
                    d.set_item("type", "flat")?;
                }
                Prior::Normal { center, width } => {
                    d.set_item("type", "normal")?;
                    d.set_item("center", *center)?;
                    d.set_item("width", *width)?;
                }
            }
            out.set_item(name, d)?;
        }
        Ok(out.into_any().unbind())
    }

    fn logpdf(&self, theta: &Bound<'_, PyAny>) -> PyResult<f64> {
        let theta = extract_f64_vec(theta)?;
        validate_f64_vec("theta", &theta, self.dim)?;
        let nll = self
            .model
            .nll(&theta)
            .map_err(|e| PyValueError::new_err(format!("logpdf failed: {}", e)))?;
        let lp_prior = self.prior_logpdf(&theta)?;
        Ok(-nll + lp_prior)
    }

    fn grad(&self, theta: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let theta = extract_f64_vec(theta)?;
        validate_f64_vec("theta", &theta, self.dim)?;
        let mut g = self
            .model
            .grad_nll(&theta)
            .map_err(|e| PyValueError::new_err(format!("grad failed: {}", e)))?;
        for gi in g.iter_mut() {
            *gi = -*gi;
        }
        self.add_prior_grad(&theta, &mut g)?;
        Ok(g)
    }

    fn to_unconstrained(&self, theta: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let theta = extract_f64_vec(theta)?;
        validate_f64_vec("theta", &theta, self.dim)?;
        Ok(self.transform.inverse(&theta))
    }

    fn to_constrained(&self, z: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let z = extract_f64_vec(z)?;
        validate_f64_vec("z", &z, self.dim)?;
        Ok(self.transform.forward(&z))
    }

    fn logpdf_unconstrained(&self, z: &Bound<'_, PyAny>) -> PyResult<f64> {
        let z = extract_f64_vec(z)?;
        validate_f64_vec("z", &z, self.dim)?;
        let theta = self.transform.forward(&z);
        let nll = self
            .model
            .nll(&theta)
            .map_err(|e| PyValueError::new_err(format!("logpdf_unconstrained failed: {}", e)))?;
        let lp_prior = self.prior_logpdf(&theta)?;
        let log_jac = self.transform.log_abs_det_jacobian(&z);
        Ok(-nll + lp_prior + log_jac)
    }

    fn grad_unconstrained(&self, z: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let z = extract_f64_vec(z)?;
        validate_f64_vec("z", &z, self.dim)?;
        let theta = self.transform.forward(&z);

        // grad(logpdf) = -grad(nll)
        let mut grad_theta = self
            .model
            .grad_nll(&theta)
            .map_err(|e| PyValueError::new_err(format!("grad_unconstrained failed: {}", e)))?;
        for gi in grad_theta.iter_mut() {
            *gi = -*gi;
        }
        self.add_prior_grad(&theta, &mut grad_theta)?;

        let jac_diag = self.transform.jacobian_diag(&z);
        let grad_log_jac = self.transform.grad_log_abs_det_jacobian(&z);
        let grad_z: Vec<f64> = grad_theta
            .iter()
            .zip(jac_diag.iter())
            .zip(grad_log_jac.iter())
            .map(|((&gt, &jd), &glj)| gt * jd + glj)
            .collect();
        Ok(grad_z)
    }
}

impl PyPosterior {
    fn param_idx(&self, name: &str) -> PyResult<usize> {
        self.name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Unknown parameter name {name:?}")))
    }

    fn prior_logpdf(&self, theta: &[f64]) -> PyResult<f64> {
        let mut lp = 0.0;
        for (th, pr) in theta.iter().copied().zip(self.priors.iter()) {
            lp += pr.logpdf(th)?;
        }
        Ok(lp)
    }

    fn add_prior_grad(&self, theta: &[f64], grad_theta: &mut [f64]) -> PyResult<()> {
        debug_assert_eq!(theta.len(), grad_theta.len());
        for i in 0..theta.len() {
            grad_theta[i] += self.priors[i].grad(theta[i])?;
        }
        Ok(())
    }
}

/// Python wrapper for HistFactoryModel
#[pyclass(name = "HistFactoryModel")]
struct PyHistFactoryModel {
    inner: RustModel,
}

#[pymethods]
impl PyHistFactoryModel {
    /// Create model from workspace JSON string (auto-detects pyhf vs HS3 format).
    #[staticmethod]
    fn from_workspace(json_str: &str) -> PyResult<Self> {
        let format = ns_translate::hs3::detect::detect_format(json_str);
        match format {
            ns_translate::hs3::detect::WorkspaceFormat::Hs3 => Self::from_hs3(json_str, None, None),
            _ => {
                let workspace: RustWorkspace = serde_json::from_str(json_str).map_err(|e| {
                    PyValueError::new_err(format!("Failed to parse workspace: {}", e))
                })?;
                let model = RustModel::from_workspace(&workspace)
                    .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;
                Ok(PyHistFactoryModel { inner: model })
            }
        }
    }

    /// Load model from HS3 JSON string.
    ///
    /// Args:
    ///     json_str: HS3 JSON string.
    ///     analysis: Optional analysis name (default: first analysis).
    ///     param_points: Optional parameter points set name (default: "default_values").
    #[staticmethod]
    #[pyo3(signature = (json_str, analysis=None, param_points=None))]
    fn from_hs3(
        json_str: &str,
        analysis: Option<&str>,
        param_points: Option<&str>,
    ) -> PyResult<Self> {
        let model = ns_translate::hs3::convert::from_hs3(
            json_str,
            analysis,
            param_points,
            ns_translate::pyhf::NormSysInterpCode::Code1,
            ns_translate::pyhf::HistoSysInterpCode::Code0,
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to load HS3: {}", e)))?;
        Ok(PyHistFactoryModel { inner: model })
    }

    /// Create model from HistFactory XML (combination.xml + ROOT files)
    #[staticmethod]
    fn from_xml(xml_path: &str) -> PyResult<Self> {
        let path = std::path::Path::new(xml_path);
        let workspace = histfactory_from_xml(path).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse HistFactory XML: {}", e))
        })?;

        let model = RustModel::from_workspace(&workspace)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        Ok(PyHistFactoryModel { inner: model })
    }

    /// Number of parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Alias: `dim()` matches the universal `LogDensityModel` naming.
    fn dim(&self) -> usize {
        self.n_params()
    }

    /// Compute negative log-likelihood
    fn nll(&self, params: &Bound<'_, PyAny>) -> PyResult<f64> {
        let params = extract_f64_vec(params)?;
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    /// Gradient of negative log-likelihood.
    fn grad_nll(&self, params: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    /// Expected data matching `pyhf.Model.expected_data`.
    #[pyo3(signature = (params, *, include_auxdata=true))]
    fn expected_data(
        &self,
        params: &Bound<'_, PyAny>,
        include_auxdata: bool,
    ) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;
        let out = if include_auxdata {
            self.inner.expected_data_pyhf(&params)
        } else {
            self.inner.expected_data_pyhf_main(&params)
        };
        out.map_err(|e| PyValueError::new_err(format!("expected_data failed: {}", e)))
    }

    /// Return a copy of the model with overridden **main observations** (main bins only).
    ///
    /// Auxiliary constraints remain unchanged (consistent with Phase 1 toy smoke tests policy).
    fn with_observed_main(&self, observed_main: Vec<f64>) -> PyResult<Self> {
        let updated = self
            .inner
            .with_observed_main(&observed_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?;
        Ok(PyHistFactoryModel { inner: updated })
    }

    /// Get parameter names in NextStat order.
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameters().iter().map(|p| p.name.clone()).collect()
    }

    /// Get suggested initial values in NextStat order.
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameters().iter().map(|p| p.init).collect()
    }

    /// Get suggested bounds in NextStat order.
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameters().iter().map(|p| p.bounds).collect()
    }

    /// Get POI index in NextStat order.
    fn poi_index(&self) -> Option<usize> {
        self.inner.poi_index()
    }

    /// Observed main-bin counts per channel.
    ///
    /// Returns a list of dicts:
    /// - channel_name: str
    /// - y: list[float]  (main bins only)
    fn observed_main_by_channel<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyAny>>> {
        let rows: Vec<ObservedChannelData> = self.inner.observed_main_by_channel();
        rows.into_iter()
            .map(|row| {
                let d = PyDict::new(py);
                d.set_item("channel_name", row.channel_name)?;
                d.set_item("y", row.y)?;
                Ok(d.into_any().unbind())
            })
            .collect()
    }

    /// Expected main-bin yields per channel and per sample, without auxdata.
    ///
    /// Returns a list of dicts:
    /// - channel_name: str
    /// - samples: list[dict{sample_name: str, y: list[float]}]
    /// - total: list[float]
    fn expected_main_by_channel_sample<'py>(
        &self,
        py: Python<'py>,
        params: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let params = extract_f64_vec(params)?;
        let rows: Vec<ExpectedChannelSampleYields> =
            self.inner.expected_main_by_channel_sample(&params).map_err(|e| {
                PyValueError::new_err(format!("expected_main_by_channel_sample failed: {}", e))
            })?;

        rows.into_iter()
            .map(|row| {
                let d = PyDict::new(py);
                d.set_item("channel_name", row.channel_name)?;
                let samples: Vec<Py<PyAny>> = row
                    .samples
                    .into_iter()
                    .map(|s: ExpectedSampleYields| {
                        let sd = PyDict::new(py);
                        sd.set_item("sample_name", s.sample_name)?;
                        sd.set_item("y", s.y)?;
                        Ok(sd.into_any().unbind())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                d.set_item("samples", samples)?;
                d.set_item("total", row.total)?;
                Ok(d.into_any().unbind())
            })
            .collect()
    }

    /// Override the **nominal yields** of a single sample (main bins only) in-place.
    ///
    /// This is intended for ML/RL loops where you want to repeatedly evaluate the model
    /// while varying a single sample's nominal histogram (e.g. signal yields from a NN).
    ///
    /// Notes:
    /// - The override is applied to main bins only (auxdata unchanged).
    /// - The sample must be override-safe (linear) as validated by the core model.
    #[pyo3(signature = (*, channel, sample, nominal))]
    fn set_sample_nominal(
        &mut self,
        channel: &str,
        sample: &str,
        nominal: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let nominal = extract_f64_vec(nominal)?;
        if nominal.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("nominal must contain only finite values"));
        }
        self.inner
            .set_sample_nominal_by_name(channel, sample, &nominal)
            .map_err(|e| PyValueError::new_err(format!("Failed to set sample nominal: {}", e)))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HEP / Unbinned (event-level) models (Phase 1+).
// ---------------------------------------------------------------------------

/// Python wrapper for `ns-unbinned::UnbinnedModel` (event-level likelihood).
#[pyclass(name = "UnbinnedModel")]
struct PyUnbinnedModel {
    inner: RustUnbinnedModel,
    schema_version: String,
}

#[pymethods]
impl PyUnbinnedModel {
    /// Compile an UnbinnedModel from a JSON/YAML spec file (`unbinned_spec_v0`).
    #[staticmethod]
    fn from_config(path: &str) -> PyResult<Self> {
        let path = std::path::Path::new(path);
        let spec = unbinned_spec::read_unbinned_spec(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read unbinned spec: {e}")))?;
        let model = unbinned_spec::compile_unbinned_model(&spec, path)
            .map_err(|e| PyValueError::new_err(format!("Failed to compile unbinned model: {e}")))?;
        Ok(Self { inner: model, schema_version: spec.schema_version })
    }

    /// Number of parameters.
    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    /// Alias: `dim()` matches the universal `LogDensityModel` naming.
    fn dim(&self) -> usize {
        self.n_params()
    }

    /// Spec `schema_version` used to construct this model.
    fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// Compute negative log-likelihood.
    fn nll(&self, params: &Bound<'_, PyAny>) -> PyResult<f64> {
        let params = extract_f64_vec(params)?;
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {e}")))
    }

    /// Gradient of negative log-likelihood.
    fn grad_nll(&self, params: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {e}")))
    }

    /// Get parameter names in NextStat order.
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    /// Get suggested initial values in NextStat order.
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    /// Get suggested bounds in NextStat order.
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }

    /// Get POI index in NextStat order (if defined in the spec).
    fn poi_index(&self) -> Option<usize> {
        self.inner.poi_index()
    }

    /// Return a copy of the model with a fixed parameter (bounds clamped to `(value, value)`).
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        Self {
            inner: self.inner.with_fixed_param(param_idx, value),
            schema_version: self.schema_version.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Hybrid (binned+unbinned) likelihood (Phase 4).
// ---------------------------------------------------------------------------

/// Python wrapper for `HybridLikelihood<HistFactoryModel, UnbinnedModel>`.
///
/// Combines a binned (HistFactory) and an unbinned (event-level) model into a
/// single likelihood with shared parameters matched by name.
#[pyclass(name = "HybridModel")]
struct PyHybridModel {
    inner: RustHybridModel,
}

#[pymethods]
impl PyHybridModel {
    /// Build a hybrid model from a binned workspace and an unbinned spec.
    ///
    /// Args:
    ///     binned: `HistFactoryModel` (binned component).
    ///     unbinned: `UnbinnedModel` (unbinned component).
    ///     poi_from: Which model provides the POI: `"binned"` (default) or `"unbinned"`.
    #[staticmethod]
    #[pyo3(signature = (binned, unbinned, poi_from = "binned"))]
    fn from_models(
        binned: &PyHistFactoryModel,
        unbinned: &PyUnbinnedModel,
        poi_from: &str,
    ) -> PyResult<Self> {
        let map = SharedParameterMap::build(&binned.inner, &unbinned.inner)
            .map_err(|e| PyValueError::new_err(format!("SharedParameterMap build failed: {e}")))?;

        let map = match poi_from {
            "binned" => {
                if let Some(poi) = binned.inner.poi_index() {
                    map.with_poi_from_a(poi)
                } else {
                    map
                }
            }
            "unbinned" => {
                if let Some(poi) = unbinned.inner.poi_index() {
                    map.with_poi_from_b(poi)
                } else {
                    map
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "poi_from must be 'binned' or 'unbinned', got '{other}'"
                )));
            }
        };

        let hybrid = HybridLikelihood::new(binned.inner.clone(), unbinned.inner.clone(), map);
        Ok(Self { inner: hybrid })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: &Bound<'_, PyAny>) -> PyResult<f64> {
        let params = extract_f64_vec(params)?;
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {e}")))
    }

    fn grad_nll(&self, params: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {e}")))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }

    fn poi_index(&self) -> Option<usize> {
        self.inner.poi_index()
    }

    /// Number of shared parameters between binned and unbinned components.
    fn n_shared(&self) -> usize {
        self.inner.parameter_map().n_shared()
    }

    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        Self { inner: self.inner.with_fixed_param(param_idx, value) }
    }
}

// ---------------------------------------------------------------------------
// Neural PDFs (Phase 3): FlowPdf + DcrSurrogate (feature-gated).
// ---------------------------------------------------------------------------

#[cfg(feature = "neural")]
use ns_unbinned::DcrSurrogate as RustDcrSurrogate;
#[cfg(feature = "neural")]
use ns_unbinned::FlowPdf as RustFlowPdf;
#[cfg(feature = "neural")]
use ns_unbinned::{EventStore, ObservableSpec};

/// Python wrapper for `ns-unbinned::FlowPdf` (ONNX normalizing flow).
///
/// Requires the `neural` feature.
#[cfg(feature = "neural")]
#[pyclass(name = "FlowPdf")]
struct PyFlowPdf {
    inner: RustFlowPdf,
}

#[cfg(feature = "neural")]
#[pymethods]
impl PyFlowPdf {
    /// Load a flow PDF from a manifest file.
    ///
    /// Args:
    ///     manifest_path: Path to `flow_manifest.json`.
    ///     context_param_indices: Maps each context feature to a global parameter index.
    ///         For unconditional flows, pass an empty list.
    #[staticmethod]
    fn from_manifest(manifest_path: &str, context_param_indices: Vec<usize>) -> PyResult<Self> {
        let path = std::path::Path::new(manifest_path);
        let flow = RustFlowPdf::from_manifest(path, &context_param_indices)
            .map_err(|e| PyValueError::new_err(format!("Failed to load FlowPdf: {e}")))?;
        Ok(Self { inner: flow })
    }

    /// Number of context (shape) parameters.
    fn n_context(&self) -> usize {
        ns_unbinned::UnbinnedPdf::n_params(&self.inner)
    }

    /// Observable names (length = features).
    fn observable_names(&self) -> Vec<String> {
        ns_unbinned::UnbinnedPdf::observables(&self.inner).to_vec()
    }

    /// Current log-normalization correction (approx 0 for well-trained flows).
    fn log_norm_correction(&self) -> f64 {
        self.inner.log_norm_correction()
    }

    /// Recompute the normalization correction for given parameters.
    fn update_normalization(&mut self, params: &Bound<'_, PyAny>) -> PyResult<()> {
        let params = extract_f64_vec(params)?;
        self.inner
            .update_normalization(&params)
            .map_err(|e| PyValueError::new_err(format!("update_normalization failed: {e}")))
    }

    /// Compute log-probabilities for a batch of events.
    ///
    /// Args:
    ///     events: dict mapping observable name -> list/array of float values.
    ///     bounds: dict mapping observable name -> (lo, hi) support bounds.
    ///     params: parameter vector (global model parameters).
    ///
    /// Returns:
    ///     List of log-probabilities (one per event).
    fn log_prob_batch(
        &self,
        events: &Bound<'_, PyDict>,
        bounds: &Bound<'_, PyDict>,
        params: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;

        let mut obs_specs = Vec::new();
        let mut columns = Vec::new();

        for (key, val) in events.iter() {
            let name: String = key.extract()?;
            let col: Vec<f64> = val.extract()?;

            let b: (f64, f64) = bounds
                .get_item(&name)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!("missing bounds for observable '{name}'"))
                })?
                .extract()?;

            obs_specs.push(ObservableSpec::branch(name.clone(), b));
            columns.push((name, col));
        }

        let store = EventStore::from_columns(obs_specs, columns, None)
            .map_err(|e| PyValueError::new_err(format!("EventStore construction failed: {e}")))?;

        let n = store.n_events();
        let mut out = vec![0.0f64; n];

        ns_unbinned::UnbinnedPdf::log_prob_batch(&self.inner, &store, &params, &mut out)
            .map_err(|e| PyValueError::new_err(format!("log_prob_batch failed: {e}")))?;

        Ok(out)
    }
}

/// Python wrapper for `ns-unbinned::DcrSurrogate` (neural DCR surrogate).
///
/// Requires the `neural` feature.
#[cfg(feature = "neural")]
#[pyclass(name = "DcrSurrogate")]
struct PyDcrSurrogate {
    inner: RustDcrSurrogate,
}

#[cfg(feature = "neural")]
#[pymethods]
impl PyDcrSurrogate {
    /// Load a DCR surrogate from a flow manifest.
    ///
    /// Args:
    ///     manifest_path: Path to `flow_manifest.json`.
    ///     systematic_param_indices: Maps each systematic nuisance parameter to
    ///         its global parameter index.
    ///     systematic_names: Names of the systematic parameters.
    ///     process_name: Name of the process this surrogate replaces.
    #[staticmethod]
    fn from_manifest(
        manifest_path: &str,
        systematic_param_indices: Vec<usize>,
        systematic_names: Vec<String>,
        process_name: String,
    ) -> PyResult<Self> {
        let path = std::path::Path::new(manifest_path);
        let dcr = RustDcrSurrogate::from_manifest(
            path,
            &systematic_param_indices,
            systematic_names,
            process_name,
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to load DcrSurrogate: {e}")))?;
        Ok(Self { inner: dcr })
    }

    /// Name of the process this surrogate replaces.
    fn process_name(&self) -> &str {
        self.inner.process_name()
    }

    /// Names of the systematic nuisance parameters.
    fn systematic_names(&self) -> Vec<String> {
        self.inner.systematic_names().to_vec()
    }

    /// Normalization tolerance.
    fn norm_tolerance(&self) -> f64 {
        self.inner.norm_tolerance()
    }

    /// Recompute normalization correction for current nuisance parameter values.
    fn update_normalization(&mut self, params: &Bound<'_, PyAny>) -> PyResult<()> {
        let params = extract_f64_vec(params)?;
        self.inner
            .update_normalization(&params)
            .map_err(|e| PyValueError::new_err(format!("update_normalization failed: {e}")))
    }

    /// Validate normalization at the nominal point (all systematics = 0).
    ///
    /// Returns:
    ///     Tuple (integral, deviation_from_1).
    fn validate_nominal_normalization(
        &mut self,
        params: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, f64)> {
        let params = extract_f64_vec(params)?;
        self.inner.validate_nominal_normalization(&params).map_err(|e| {
            PyValueError::new_err(format!("validate_nominal_normalization failed: {e}"))
        })
    }

    /// Compute log-probabilities for a batch of events.
    ///
    /// Args:
    ///     events: dict mapping observable name -> list/array of float values.
    ///     bounds: dict mapping observable name -> (lo, hi) support bounds.
    ///     params: parameter vector (global model parameters).
    ///
    /// Returns:
    ///     List of log-probabilities (one per event).
    fn log_prob_batch(
        &self,
        events: &Bound<'_, PyDict>,
        bounds: &Bound<'_, PyDict>,
        params: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<f64>> {
        let params = extract_f64_vec(params)?;

        let mut obs_specs = Vec::new();
        let mut columns = Vec::new();

        for (key, val) in events.iter() {
            let name: String = key.extract()?;
            let col: Vec<f64> = val.extract()?;

            let b: (f64, f64) = bounds
                .get_item(&name)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!("missing bounds for observable '{name}'"))
                })?
                .extract()?;

            obs_specs.push(ObservableSpec::branch(name.clone(), b));
            columns.push((name, col));
        }

        let store = EventStore::from_columns(obs_specs, columns, None)
            .map_err(|e| PyValueError::new_err(format!("EventStore construction failed: {e}")))?;

        let n = store.n_events();
        let mut out = vec![0.0f64; n];

        ns_unbinned::UnbinnedPdf::log_prob_batch(&self.inner, &store, &params, &mut out)
            .map_err(|e| PyValueError::new_err(format!("log_prob_batch failed: {e}")))?;

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Time series (Phase 8): Kalman filter / RTS smoother.
// ---------------------------------------------------------------------------

/// Python wrapper for `ns_inference::timeseries::kalman::KalmanModel`.
#[pyclass(name = "KalmanModel")]
struct PyKalmanModel {
    inner: RustKalmanModel,
}

#[pymethods]
impl PyKalmanModel {
    #[new]
    fn new(
        f: Vec<Vec<f64>>,
        q: Vec<Vec<f64>>,
        h: Vec<Vec<f64>>,
        r: Vec<Vec<f64>>,
        m0: Vec<f64>,
        p0: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let f = dmatrix_from_nested("F", f)?;
        let q = dmatrix_from_nested("Q", q)?;
        let h = dmatrix_from_nested("H", h)?;
        let r = dmatrix_from_nested("R", r)?;
        let m0 = dvector_from_vec("m0", m0)?;
        let p0 = dmatrix_from_nested("P0", p0)?;

        let model = RustKalmanModel::new(f, q, h, r, m0, p0)
            .map_err(|e| PyValueError::new_err(format!("Failed to build KalmanModel: {}", e)))?;
        Ok(Self { inner: model })
    }

    fn n_state(&self) -> usize {
        self.inner.n_state()
    }

    fn n_obs(&self) -> usize {
        self.inner.n_obs()
    }
}

#[pyfunction]
fn kalman_filter(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
) -> PyResult<Py<PyAny>> {
    let ys: Vec<DVector<f64>> = ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<PyResult<Vec<_>>>()?;

    let fr = rust_kalman_filter(&model.inner, &ys)
        .map_err(|e| PyValueError::new_err(format!("kalman_filter failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("log_likelihood", fr.log_likelihood)?;
    out.set_item(
        "predicted_means",
        fr.predicted_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "predicted_covs",
        fr.predicted_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "filtered_means",
        fr.filtered_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "filtered_covs",
        fr.filtered_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    )?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
fn kalman_smooth(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
) -> PyResult<Py<PyAny>> {
    let ys: Vec<DVector<f64>> = ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<PyResult<Vec<_>>>()?;

    let fr = rust_kalman_filter(&model.inner, &ys)
        .map_err(|e| PyValueError::new_err(format!("kalman_filter failed: {}", e)))?;
    let sr = rust_rts_smoother(&model.inner, &fr)
        .map_err(|e| PyValueError::new_err(format!("rts_smoother failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("log_likelihood", fr.log_likelihood)?;
    out.set_item(
        "filtered_means",
        fr.filtered_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "filtered_covs",
        fr.filtered_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "smoothed_means",
        sr.smoothed_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "smoothed_covs",
        sr.smoothed_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    )?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (model, ys, *, max_iter=50, tol=1e-6, estimate_q=true, estimate_r=true, estimate_f=false, estimate_h=false, min_diag=1e-12))]
fn kalman_em(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
    max_iter: usize,
    tol: f64,
    estimate_q: bool,
    estimate_r: bool,
    estimate_f: bool,
    estimate_h: bool,
    min_diag: f64,
) -> PyResult<Py<PyAny>> {
    let ys: Vec<DVector<f64>> = ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<PyResult<Vec<_>>>()?;

    let cfg = RustKalmanEmConfig {
        max_iter,
        tol,
        estimate_q,
        estimate_r,
        estimate_f,
        estimate_h,
        min_diag,
    };

    let res = rust_kalman_em(&model.inner, &ys, cfg)
        .map_err(|e| PyValueError::new_err(format!("kalman_em failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("converged", res.converged)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("f", dmatrix_to_nested(&res.model.f))?;
    out.set_item("h", dmatrix_to_nested(&res.model.h))?;
    out.set_item("q", dmatrix_to_nested(&res.model.q))?;
    out.set_item("r", dmatrix_to_nested(&res.model.r))?;
    out.set_item("model", Py::new(py, PyKalmanModel { inner: res.model })?)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (model, ys, *, steps=1, alpha=None))]
fn kalman_forecast(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
    steps: usize,
    alpha: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let ys: Vec<DVector<f64>> = ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<PyResult<Vec<_>>>()?;

    let fr = rust_kalman_filter(&model.inner, &ys)
        .map_err(|e| PyValueError::new_err(format!("kalman_filter failed: {}", e)))?;
    let fc = rust_kalman_forecast(&model.inner, &fr, steps)
        .map_err(|e| PyValueError::new_err(format!("kalman_forecast failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("state_means", fc.state_means.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
    out.set_item("state_covs", fc.state_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>())?;
    out.set_item("obs_means", fc.obs_means.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
    out.set_item("obs_covs", fc.obs_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>())?;
    if let Some(alpha) = alpha {
        let iv = rust_kalman_forecast_intervals(&fc, alpha).map_err(|e| {
            PyValueError::new_err(format!("kalman_forecast_intervals failed: {}", e))
        })?;
        out.set_item("alpha", iv.alpha)?;
        out.set_item("z", iv.z)?;
        out.set_item("obs_lower", iv.obs_lower.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
        out.set_item("obs_upper", iv.obs_upper.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
    }

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (model, *, t_max, seed=42, init="sample", x0=None))]
fn kalman_simulate(
    py: Python<'_>,
    model: &PyKalmanModel,
    t_max: usize,
    seed: u64,
    init: &str,
    x0: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    let x0 = if let Some(x0) = x0 {
        Some(dvector_from_vec("x0", x0)?)
    } else {
        match init {
            "sample" => None,
            "mean" => Some(model.inner.m0.clone()),
            _ => {
                return Err(PyValueError::new_err(
                    "init must be one of {'sample','mean'} (or pass x0=...)",
                ));
            }
        }
    };

    let sim = if let Some(x0) = x0 {
        rust_kalman_simulate_with_x0(&model.inner, t_max, seed, Some(x0))
    } else {
        rust_kalman_simulate(&model.inner, t_max, seed)
    }
    .map_err(|e| PyValueError::new_err(format!("kalman_simulate failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("xs", sim.xs.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
    out.set_item("ys", sim.ys.iter().map(dvector_to_vec).collect::<Vec<_>>())?;
    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Volatility (Econometrics): GARCH(1,1) + approximate SV (log-chi2 + Kalman).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (ys, *, max_iter=1000, tol=1e-6, alpha_beta_max=0.999, min_var=1e-18))]
fn garch11_fit(
    py: Python<'_>,
    ys: Vec<f64>,
    max_iter: u64,
    tol: f64,
    alpha_beta_max: f64,
    min_var: f64,
) -> PyResult<Py<PyAny>> {
    let cfg = RustGarch11Config {
        alpha_beta_max,
        min_var,
        optimizer: ns_inference::optimizer::OptimizerConfig { max_iter, tol, ..Default::default() },
        ..Default::default()
    };

    let fit = rust_garch11_fit(&ys, cfg)
        .map_err(|e| PyValueError::new_err(format!("garch11_fit failed: {e}")))?;

    let params = PyDict::new(py);
    params.set_item("mu", fit.params.mu)?;
    params.set_item("omega", fit.params.omega)?;
    params.set_item("alpha", fit.params.alpha)?;
    params.set_item("beta", fit.params.beta)?;

    let out = PyDict::new(py);
    out.set_item("params", params)?;
    out.set_item("log_likelihood", fit.log_likelihood)?;
    out.set_item("conditional_variance", fit.conditional_variance.clone())?;
    out.set_item(
        "conditional_sigma",
        fit.conditional_variance.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>(),
    )?;
    out.set_item("converged", fit.optimization.converged)?;
    out.set_item("n_iter", fit.optimization.n_iter)?;
    out.set_item("n_fev", fit.optimization.n_fev)?;
    out.set_item("n_gev", fit.optimization.n_gev)?;
    out.set_item("fval", fit.optimization.fval)?;
    out.set_item("message", fit.optimization.message)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (ys, *, max_iter=1000, tol=1e-6, log_eps=1e-12))]
fn sv_logchi2_fit(
    py: Python<'_>,
    ys: Vec<f64>,
    max_iter: u64,
    tol: f64,
    log_eps: f64,
) -> PyResult<Py<PyAny>> {
    let cfg = RustSvLogChi2Config {
        log_eps,
        optimizer: ns_inference::optimizer::OptimizerConfig { max_iter, tol, ..Default::default() },
        ..Default::default()
    };

    let fit = rust_sv_logchi2_fit(&ys, cfg)
        .map_err(|e| PyValueError::new_err(format!("sv_logchi2_fit failed: {e}")))?;

    let params = PyDict::new(py);
    params.set_item("mu", fit.params.mu)?;
    params.set_item("phi", fit.params.phi)?;
    params.set_item("sigma", fit.params.sigma)?;

    let out = PyDict::new(py);
    out.set_item("params", params)?;
    out.set_item("log_likelihood", fit.log_likelihood)?;
    out.set_item("smoothed_h", fit.smoothed_h)?;
    out.set_item("smoothed_sigma", fit.smoothed_sigma)?;
    out.set_item("converged", fit.optimization.converged)?;
    out.set_item("n_iter", fit.optimization.n_iter)?;
    out.set_item("n_fev", fit.optimization.n_fev)?;
    out.set_item("n_gev", fit.optimization.n_gev)?;
    out.set_item("fval", fit.optimization.fval)?;
    out.set_item("message", fit.optimization.message)?;

    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Econometrics & Causal Inference (Phase 12).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (entity_ids, x, y, p, *, cluster_ids=None))]
fn panel_fe(
    py: Python<'_>,
    entity_ids: Vec<u64>,
    x: Vec<f64>,
    y: Vec<f64>,
    p: usize,
    cluster_ids: Option<Vec<u64>>,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::econometrics::panel::panel_fe_fit(
        &entity_ids,
        &x,
        &y,
        p,
        cluster_ids.as_deref(),
    )
    .map_err(|e| PyValueError::new_err(format!("panel_fe failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("coefficients", res.coefficients)?;
    out.set_item("se_ols", res.se_ols)?;
    out.set_item("se_cluster", res.se_cluster)?;
    out.set_item("r_squared_within", res.r_squared_within)?;
    out.set_item("n_obs", res.n_obs)?;
    out.set_item("n_entities", res.n_entities)?;
    out.set_item("n_regressors", res.n_regressors)?;
    out.set_item("rss", res.rss)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (y, treat, post, cluster_ids))]
fn did(
    py: Python<'_>,
    y: Vec<f64>,
    treat: Vec<u8>,
    post: Vec<u8>,
    cluster_ids: Vec<u64>,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::econometrics::did::did_canonical(&y, &treat, &post, &cluster_ids)
        .map_err(|e| PyValueError::new_err(format!("did failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("att", res.att)?;
    out.set_item("se", res.se)?;
    out.set_item("se_cluster", res.se_cluster)?;
    out.set_item("t_stat", res.t_stat)?;
    out.set_item("mean_treated_post", res.mean_treated_post)?;
    out.set_item("mean_treated_pre", res.mean_treated_pre)?;
    out.set_item("mean_control_post", res.mean_control_post)?;
    out.set_item("mean_control_pre", res.mean_control_pre)?;
    out.set_item("n_obs", res.n_obs)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (y, entity_ids, time_ids, relative_time, min_lag, max_lag, reference_period, cluster_ids))]
fn event_study(
    py: Python<'_>,
    y: Vec<f64>,
    entity_ids: Vec<u64>,
    time_ids: Vec<u64>,
    relative_time: Vec<i64>,
    min_lag: i64,
    max_lag: i64,
    reference_period: i64,
    cluster_ids: Vec<u64>,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::econometrics::did::event_study(
        &y,
        &entity_ids,
        &time_ids,
        &relative_time,
        min_lag,
        max_lag,
        reference_period,
        &cluster_ids,
    )
    .map_err(|e| PyValueError::new_err(format!("event_study failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("relative_times", res.relative_times)?;
    out.set_item("coefficients", res.coefficients)?;
    out.set_item("se_cluster", res.se_cluster)?;
    out.set_item("ci_lower", res.ci_lower)?;
    out.set_item("ci_upper", res.ci_upper)?;
    out.set_item("n_obs", res.n_obs)?;
    out.set_item("reference_period", res.reference_period)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (y, x_exog, k_exog, x_endog, k_endog, z, m, *, exog_names=None, endog_names=None, cluster_ids=None))]
fn iv_2sls(
    py: Python<'_>,
    y: Vec<f64>,
    x_exog: Vec<f64>,
    k_exog: usize,
    x_endog: Vec<f64>,
    k_endog: usize,
    z: Vec<f64>,
    m: usize,
    exog_names: Option<Vec<String>>,
    endog_names: Option<Vec<String>>,
    cluster_ids: Option<Vec<u64>>,
) -> PyResult<Py<PyAny>> {
    let exog_n = exog_names.unwrap_or_default();
    let endog_n = endog_names.unwrap_or_default();

    let res = ns_inference::econometrics::iv::iv_2sls(
        &y,
        &x_exog,
        k_exog,
        &x_endog,
        k_endog,
        &z,
        m,
        &exog_n,
        &endog_n,
        cluster_ids.as_deref(),
    )
    .map_err(|e| PyValueError::new_err(format!("iv_2sls failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("coefficients", res.coefficients)?;
    out.set_item("names", res.names)?;
    out.set_item("se", res.se)?;
    out.set_item("se_cluster", res.se_cluster)?;
    out.set_item("n_obs", res.n_obs)?;
    out.set_item("n_instruments", res.n_instruments)?;

    let first_stage_list = PyList::empty(py);
    for fs in &res.first_stage {
        let d = PyDict::new(py);
        d.set_item("f_stat", fs.f_stat)?;
        d.set_item("r_squared", fs.r_squared)?;
        d.set_item("partial_r_squared", fs.partial_r_squared)?;
        d.set_item("passes_stock_yogo_10", fs.passes_stock_yogo_10)?;
        first_stage_list.append(d)?;
    }
    out.set_item("first_stage", first_stage_list)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (y, treat, propensity, mu1, mu0, *, trim=0.01))]
fn aipw_ate(
    py: Python<'_>,
    y: Vec<f64>,
    treat: Vec<u8>,
    propensity: Vec<f64>,
    mu1: Vec<f64>,
    mu0: Vec<f64>,
    trim: f64,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::econometrics::aipw::aipw_ate(&y, &treat, &propensity, &mu1, &mu0, trim)
        .map_err(|e| PyValueError::new_err(format!("aipw_ate failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("ate", res.ate)?;
    out.set_item("se", res.se)?;
    out.set_item("ci_lower", res.ci_lower)?;
    out.set_item("ci_upper", res.ci_upper)?;
    out.set_item("n_treated", res.n_treated)?;
    out.set_item("n_control", res.n_control)?;
    out.set_item("mean_propensity", res.mean_propensity)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (y_treated, y_control, gammas))]
fn rosenbaum_bounds(
    py: Python<'_>,
    y_treated: Vec<f64>,
    y_control: Vec<f64>,
    gammas: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::econometrics::aipw::rosenbaum_bounds(&y_treated, &y_control, &gammas)
        .map_err(|e| PyValueError::new_err(format!("rosenbaum_bounds failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("gammas", res.gammas)?;
    out.set_item("p_upper", res.p_upper)?;
    out.set_item("p_lower", res.p_lower)?;
    out.set_item("gamma_critical", res.gamma_critical)?;
    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Survival: Kaplan-Meier + Log-rank (non-parametric)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (times, events, *, conf_level=0.95))]
fn kaplan_meier(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let est = ns_inference::kaplan_meier(&times, &events, conf_level)
        .map_err(|e| PyValueError::new_err(format!("kaplan_meier failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("n", est.n)?;
    out.set_item("n_events", est.n_events)?;
    out.set_item("conf_level", est.conf_level)?;
    out.set_item("median", est.median)?;

    let step_times = PyList::new(py, est.steps.iter().map(|s| s.time))?;
    let step_n_risk = PyList::new(py, est.steps.iter().map(|s| s.n_risk))?;
    let step_n_events = PyList::new(py, est.steps.iter().map(|s| s.n_events))?;
    let step_n_censored = PyList::new(py, est.steps.iter().map(|s| s.n_censored))?;
    let step_survival = PyList::new(py, est.steps.iter().map(|s| s.survival))?;
    let step_variance = PyList::new(py, est.steps.iter().map(|s| s.variance))?;
    let step_ci_lower = PyList::new(py, est.steps.iter().map(|s| s.ci_lower))?;
    let step_ci_upper = PyList::new(py, est.steps.iter().map(|s| s.ci_upper))?;

    out.set_item("time", step_times)?;
    out.set_item("n_risk", step_n_risk)?;
    out.set_item("n_event", step_n_events)?;
    out.set_item("n_censored", step_n_censored)?;
    out.set_item("survival", step_survival)?;
    out.set_item("variance", step_variance)?;
    out.set_item("ci_lower", step_ci_lower)?;
    out.set_item("ci_upper", step_ci_upper)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, groups))]
fn log_rank_test(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
) -> PyResult<Py<PyAny>> {
    let res = ns_inference::log_rank_test(&times, &events, &groups)
        .map_err(|e| PyValueError::new_err(format!("log_rank_test failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("n", res.n)?;
    out.set_item("chi_squared", res.chi_squared)?;
    out.set_item("df", res.df)?;
    out.set_item("p_value", res.p_value)?;

    let grp_ids = PyList::new(py, res.group_summaries.iter().map(|(g, _, _)| *g))?;
    let grp_obs = PyList::new(py, res.group_summaries.iter().map(|(_, o, _)| *o))?;
    let grp_exp = PyList::new(py, res.group_summaries.iter().map(|(_, _, e)| *e))?;

    out.set_item("group_ids", grp_ids)?;
    out.set_item("observed", grp_obs)?;
    out.set_item("expected", grp_exp)?;

    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Churn / Subscription vertical
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (*, n_customers=2000, n_cohorts=6, max_time=24.0, treatment_fraction=0.3, seed=42))]
fn churn_generate_data(
    py: Python<'_>,
    n_customers: usize,
    n_cohorts: usize,
    max_time: f64,
    treatment_fraction: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let config = ns_inference::ChurnDataConfig {
        n_customers,
        n_cohorts,
        max_time,
        treatment_fraction,
        seed,
        ..Default::default()
    };
    let ds = ns_inference::generate_churn_dataset(&config)
        .map_err(|e| PyValueError::new_err(format!("churn_generate_data failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("n", ds.records.len())?;
    out.set_item("n_events", ds.events.iter().filter(|&&e| e).count())?;
    out.set_item("times", PyList::new(py, &ds.times)?)?;
    out.set_item("events", PyList::new(py, &ds.events)?)?;
    out.set_item("groups", PyList::new(py, &ds.groups)?)?;
    out.set_item("treated", PyList::new(py, ds.records.iter().map(|r| r.treated))?)?;
    out.set_item("covariates", &ds.covariates)?;
    out.set_item(
        "covariate_names",
        vec!["plan_basic", "plan_premium", "usage_score", "support_tickets"],
    )?;

    let plans = PyList::new(py, ds.records.iter().map(|r| r.plan))?;
    let regions = PyList::new(py, ds.records.iter().map(|r| r.region))?;
    let cohorts = PyList::new(py, ds.records.iter().map(|r| r.cohort))?;
    let usage = PyList::new(py, ds.records.iter().map(|r| r.usage_score))?;
    out.set_item("plan", plans)?;
    out.set_item("region", regions)?;
    out.set_item("cohort", cohorts)?;
    out.set_item("usage_score", usage)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, groups, *, conf_level=0.95))]
fn churn_retention(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let ra = ns_inference::retention_analysis(&times, &events, &groups, conf_level)
        .map_err(|e| PyValueError::new_err(format!("churn_retention failed: {e}")))?;

    let out = PyDict::new(py);

    // Overall KM summary.
    let overall = PyDict::new(py);
    overall.set_item("n", ra.overall.n)?;
    overall.set_item("n_events", ra.overall.n_events)?;
    overall.set_item("median", ra.overall.median)?;
    overall.set_item("time", PyList::new(py, ra.overall.steps.iter().map(|s| s.time))?)?;
    overall.set_item("survival", PyList::new(py, ra.overall.steps.iter().map(|s| s.survival))?)?;
    out.set_item("overall", overall)?;

    // Per-group.
    let by_group = PyList::empty(py);
    for (g, km) in &ra.by_group {
        let gd = PyDict::new(py);
        gd.set_item("group", *g)?;
        gd.set_item("n", km.n)?;
        gd.set_item("n_events", km.n_events)?;
        gd.set_item("median", km.median)?;
        gd.set_item("time", PyList::new(py, km.steps.iter().map(|s| s.time))?)?;
        gd.set_item("survival", PyList::new(py, km.steps.iter().map(|s| s.survival))?)?;
        by_group.append(gd)?;
    }
    out.set_item("by_group", by_group)?;

    // Log-rank.
    let lr = PyDict::new(py);
    lr.set_item("chi_squared", ra.log_rank.chi_squared)?;
    lr.set_item("df", ra.log_rank.df)?;
    lr.set_item("p_value", ra.log_rank.p_value)?;
    out.set_item("log_rank", lr)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, covariates, names, *, conf_level=0.95))]
fn churn_risk_model(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    covariates: Vec<Vec<f64>>,
    names: Vec<String>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let result = ns_inference::churn_risk_model(&times, &events, &covariates, &names, conf_level)
        .map_err(|e| PyValueError::new_err(format!("churn_risk_model failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("n", result.n)?;
    out.set_item("n_events", result.n_events)?;
    out.set_item("nll", result.nll)?;
    out.set_item("names", &result.names)?;
    out.set_item("coefficients", &result.coefficients)?;
    out.set_item("se", &result.se)?;
    out.set_item("hazard_ratios", &result.hazard_ratios)?;
    out.set_item("hr_ci_lower", &result.hr_ci_lower)?;
    out.set_item("hr_ci_upper", &result.hr_ci_upper)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, treated, covariates, *, horizon=12.0))]
fn churn_uplift(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    treated: Vec<u8>,
    covariates: Vec<Vec<f64>>,
    horizon: f64,
) -> PyResult<Py<PyAny>> {
    let result = ns_inference::churn_uplift(&times, &events, &treated, &covariates, horizon)
        .map_err(|e| PyValueError::new_err(format!("churn_uplift failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("ate", result.ate)?;
    out.set_item("se", result.se)?;
    out.set_item("ci_lower", result.ci_lower)?;
    out.set_item("ci_upper", result.ci_upper)?;
    out.set_item("n_treated", result.n_treated)?;
    out.set_item("n_control", result.n_control)?;
    out.set_item("gamma_critical", result.gamma_critical)?;
    out.set_item("horizon", horizon)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, groups, *, treated=vec![], covariates=vec![], covariate_names=vec![], trim=0.01))]
fn churn_diagnostics(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
    treated: Vec<u8>,
    covariates: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    trim: f64,
) -> PyResult<Py<PyAny>> {
    let r = ns_inference::churn_diagnostics_report(
        &times,
        &events,
        &groups,
        &treated,
        &covariates,
        &covariate_names,
        trim,
    )
    .map_err(|e| PyValueError::new_err(format!("churn_diagnostics failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("n", r.n)?;
    out.set_item("n_events", r.n_events)?;
    out.set_item("overall_censoring_frac", r.overall_censoring_frac)?;
    out.set_item("trust_gate_passed", r.trust_gate_passed)?;

    // Censoring by segment.
    let cens_list = PyList::empty(py);
    for seg in &r.censoring_by_segment {
        let d = PyDict::new(py);
        d.set_item("group", seg.group)?;
        d.set_item("n", seg.n)?;
        d.set_item("n_events", seg.n_events)?;
        d.set_item("n_censored", seg.n_censored)?;
        d.set_item("frac_censored", seg.frac_censored)?;
        cens_list.append(d)?;
    }
    out.set_item("censoring_by_segment", cens_list)?;

    // Covariate balance.
    let bal_list = PyList::empty(py);
    for b in &r.covariate_balance {
        let d = PyDict::new(py);
        d.set_item("name", &b.name)?;
        d.set_item("smd_raw", b.smd_raw)?;
        d.set_item("mean_treated", b.mean_treated)?;
        d.set_item("mean_control", b.mean_control)?;
        bal_list.append(d)?;
    }
    out.set_item("covariate_balance", bal_list)?;

    // Propensity overlap (optional).
    if let Some(ref po) = r.propensity_overlap {
        let pod = PyDict::new(py);
        pod.set_item("quantiles", po.quantiles.to_vec())?;
        pod.set_item("mean", po.mean)?;
        pod.set_item("n_trimmed_low", po.n_trimmed_low)?;
        pod.set_item("n_trimmed_high", po.n_trimmed_high)?;
        pod.set_item("trim", po.trim)?;
        out.set_item("propensity_overlap", pod)?;
    } else {
        out.set_item("propensity_overlap", py.None())?;
    }

    // Warnings.
    let warn_list = PyList::empty(py);
    for w in &r.warnings {
        let d = PyDict::new(py);
        d.set_item("category", &w.category)?;
        d.set_item("severity", &w.severity)?;
        d.set_item("message", &w.message)?;
        warn_list.append(d)?;
    }
    out.set_item("warnings", warn_list)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, groups, period_boundaries))]
fn churn_cohort_matrix(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
    period_boundaries: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let r = ns_inference::cohort_retention_matrix(&times, &events, &groups, &period_boundaries)
        .map_err(|e| PyValueError::new_err(format!("churn_cohort_matrix failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("period_boundaries", &r.period_boundaries)?;

    // Cohort rows.
    let rows_list = PyList::empty(py);
    for row in &r.cohorts {
        let rd = PyDict::new(py);
        rd.set_item("cohort", row.cohort)?;
        rd.set_item("n_total", row.n_total)?;
        rd.set_item("n_events", row.n_events)?;
        let periods = PyList::empty(py);
        for cell in &row.periods {
            let cd = PyDict::new(py);
            cd.set_item("n_at_risk", cell.n_at_risk)?;
            cd.set_item("n_events", cell.n_events)?;
            cd.set_item("n_censored", cell.n_censored)?;
            cd.set_item("retention_rate", cell.retention_rate)?;
            cd.set_item("cumulative_retention", cell.cumulative_retention)?;
            periods.append(cd)?;
        }
        rd.set_item("periods", periods)?;
        rows_list.append(rd)?;
    }
    out.set_item("cohorts", rows_list)?;

    // Overall row.
    let od = PyDict::new(py);
    od.set_item("cohort", r.overall.cohort)?;
    od.set_item("n_total", r.overall.n_total)?;
    od.set_item("n_events", r.overall.n_events)?;
    let op = PyList::empty(py);
    for cell in &r.overall.periods {
        let cd = PyDict::new(py);
        cd.set_item("n_at_risk", cell.n_at_risk)?;
        cd.set_item("n_events", cell.n_events)?;
        cd.set_item("n_censored", cell.n_censored)?;
        cd.set_item("retention_rate", cell.retention_rate)?;
        cd.set_item("cumulative_retention", cell.cumulative_retention)?;
        op.append(cd)?;
    }
    od.set_item("periods", op)?;
    out.set_item("overall", od)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, groups, *, conf_level=0.95, correction="benjamini_hochberg", alpha=0.05))]
fn churn_compare(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
    conf_level: f64,
    correction: &str,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    let corr = match correction {
        "bonferroni" => ns_inference::CorrectionMethod::Bonferroni,
        "benjamini_hochberg" | "bh" => ns_inference::CorrectionMethod::BenjaminiHochberg,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown correction method '{other}'; use 'bonferroni' or 'benjamini_hochberg'"
            )));
        }
    };

    let r =
        ns_inference::segment_comparison_report(&times, &events, &groups, conf_level, corr, alpha)
            .map_err(|e| PyValueError::new_err(format!("churn_compare failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("overall_chi_squared", r.overall_chi_squared)?;
    out.set_item("overall_p_value", r.overall_p_value)?;
    out.set_item("overall_df", r.overall_df)?;
    out.set_item("alpha", r.alpha)?;
    out.set_item("n", r.n)?;
    out.set_item("n_events", r.n_events)?;
    out.set_item("correction_method", correction)?;

    // Segments.
    let seg_list = PyList::empty(py);
    for s in &r.segments {
        let sd = PyDict::new(py);
        sd.set_item("group", s.group)?;
        sd.set_item("n", s.n)?;
        sd.set_item("n_events", s.n_events)?;
        sd.set_item("median", s.median)?;
        sd.set_item("observed", s.observed)?;
        sd.set_item("expected", s.expected)?;
        seg_list.append(sd)?;
    }
    out.set_item("segments", seg_list)?;

    // Pairwise comparisons.
    let pw_list = PyList::empty(py);
    for pw in &r.pairwise {
        let pd = PyDict::new(py);
        pd.set_item("group_a", pw.group_a)?;
        pd.set_item("group_b", pw.group_b)?;
        pd.set_item("chi_squared", pw.chi_squared)?;
        pd.set_item("p_value", pw.p_value)?;
        pd.set_item("p_adjusted", pw.p_adjusted)?;
        pd.set_item("hazard_ratio_proxy", pw.hazard_ratio_proxy)?;
        pd.set_item("median_diff", pw.median_diff)?;
        pd.set_item("significant", pw.significant)?;
        pw_list.append(pd)?;
    }
    out.set_item("pairwise", pw_list)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, treated, *, covariates=vec![], horizon=12.0, eval_horizons=vec![3.0, 6.0, 12.0, 24.0], trim=0.01))]
fn churn_uplift_survival(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    treated: Vec<u8>,
    covariates: Vec<Vec<f64>>,
    horizon: f64,
    eval_horizons: Vec<f64>,
    trim: f64,
) -> PyResult<Py<PyAny>> {
    let r = ns_inference::survival_uplift_report(
        &times,
        &events,
        &treated,
        &covariates,
        horizon,
        &eval_horizons,
        trim,
    )
    .map_err(|e| PyValueError::new_err(format!("churn_uplift_survival failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("rmst_treated", r.rmst_treated)?;
    out.set_item("rmst_control", r.rmst_control)?;
    out.set_item("delta_rmst", r.delta_rmst)?;
    out.set_item("horizon", r.horizon)?;
    out.set_item("ipw_applied", r.ipw_applied)?;

    // Arms.
    let arms_list = PyList::empty(py);
    for arm in &r.arms {
        let ad = PyDict::new(py);
        ad.set_item("arm", &arm.arm)?;
        ad.set_item("n", arm.n)?;
        ad.set_item("n_events", arm.n_events)?;
        ad.set_item("rmst", arm.rmst)?;
        ad.set_item("median", arm.median)?;
        arms_list.append(ad)?;
    }
    out.set_item("arms", arms_list)?;

    // Survival diffs.
    let sd_list = PyList::empty(py);
    for sd in &r.survival_diffs {
        let dd = PyDict::new(py);
        dd.set_item("horizon", sd.horizon)?;
        dd.set_item("survival_treated", sd.survival_treated)?;
        dd.set_item("survival_control", sd.survival_control)?;
        dd.set_item("delta_survival", sd.delta_survival)?;
        sd_list.append(dd)?;
    }
    out.set_item("survival_diffs", sd_list)?;

    // Overlap.
    let ol = PyDict::new(py);
    ol.set_item("n_total", r.overlap.n_total)?;
    ol.set_item("n_after_trim", r.overlap.n_after_trim)?;
    ol.set_item("n_trimmed", r.overlap.n_trimmed)?;
    ol.set_item("mean_propensity", r.overlap.mean_propensity)?;
    ol.set_item("min_propensity", r.overlap.min_propensity)?;
    ol.set_item("max_propensity", r.overlap.max_propensity)?;
    ol.set_item("ess_treated", r.overlap.ess_treated)?;
    ol.set_item("ess_control", r.overlap.ess_control)?;
    out.set_item("overlap", ol)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, covariates, names, *, n_bootstrap=1000, seed=42, conf_level=0.95))]
fn churn_bootstrap_hr(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    covariates: Vec<Vec<f64>>,
    names: Vec<String>,
    n_bootstrap: usize,
    seed: u64,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let r = ns_inference::bootstrap_hazard_ratios(
        &times,
        &events,
        &covariates,
        &names,
        n_bootstrap,
        seed,
        conf_level,
    )
    .map_err(|e| PyValueError::new_err(format!("churn_bootstrap_hr failed: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("names", &r.names)?;
    out.set_item("hr_point", &r.hr_point)?;
    out.set_item("hr_ci_lower", &r.hr_ci_lower)?;
    out.set_item("hr_ci_upper", &r.hr_ci_upper)?;
    out.set_item("n_bootstrap", r.n_bootstrap)?;
    out.set_item("n_converged", r.n_converged)?;
    out.set_item("elapsed_s", r.elapsed_s)?;
    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (times, events, *, groups=None, treated=None, covariates=vec![], covariate_names=vec![], observation_end=None))]
fn churn_ingest(
    py: Python<'_>,
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Option<Vec<i64>>,
    treated: Option<Vec<u8>>,
    covariates: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    observation_end: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let r = ns_inference::ingest_churn_arrays(
        &times,
        &events,
        groups.as_deref(),
        treated.as_deref(),
        &covariates,
        &covariate_names,
        observation_end,
    )
    .map_err(|e| PyValueError::new_err(format!("churn_ingest failed: {e}")))?;

    let out = PyDict::new(py);
    let ds = &r.dataset;
    out.set_item("n", ds.records.len())?;
    out.set_item("n_events", ds.events.iter().filter(|&&e| e).count())?;
    out.set_item("times", PyList::new(py, &ds.times)?)?;
    out.set_item("events", PyList::new(py, &ds.events)?)?;
    out.set_item("groups", PyList::new(py, &ds.groups)?)?;
    out.set_item("treated", PyList::new(py, ds.records.iter().map(|r| r.treated))?)?;
    out.set_item("covariates", &ds.covariates)?;
    out.set_item("covariate_names", &r.covariate_names)?;
    out.set_item("n_dropped", r.n_dropped)?;
    out.set_item("warnings", &r.warnings)?;
    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Minimal non-HEP model to exercise the generic sampling surface.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GaussianMeanModel {
    y: Vec<f64>,
    inv_var: f64,
    mean_y: f64,
}

impl GaussianMeanModel {
    fn new(y: Vec<f64>, sigma: f64) -> NsResult<Self> {
        if y.is_empty() {
            return Err(NsError::Validation("y must be non-empty".to_string()));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(NsError::Validation("sigma must be finite and > 0".to_string()));
        }
        let mean_y = y.iter().sum::<f64>() / (y.len() as f64);
        Ok(Self { y, inv_var: 1.0 / (sigma * sigma), mean_y })
    }
}

#[derive(Clone, Copy)]
struct PreparedGaussianMeanModel<'a> {
    model: &'a GaussianMeanModel,
}

impl PreparedNll for PreparedGaussianMeanModel<'_> {
    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        self.model.nll(params)
    }
}

impl LogDensityModel for GaussianMeanModel {
    type Prepared<'a>
        = PreparedGaussianMeanModel<'a>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        1
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["mu".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![self.mean_y]
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        if params.len() != 1 {
            return Err(NsError::Validation("expected 1 parameter".to_string()));
        }
        let mu = params[0];
        if !mu.is_finite() {
            return Err(NsError::Validation("mu must be finite".to_string()));
        }
        // Constant terms don't matter for inference; keep a stable, simple NLL.
        let mut nll = 0.0;
        for &yi in &self.y {
            let r = yi - mu;
            nll += 0.5 * r * r * self.inv_var;
        }
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        if params.len() != 1 {
            return Err(NsError::Validation("expected 1 parameter".to_string()));
        }
        let mu = params[0];
        if !mu.is_finite() {
            return Err(NsError::Validation("mu must be finite".to_string()));
        }
        // d/dmu 0.5 * sum ((yi - mu)^2 / sigma^2) = sum ((mu - yi) / sigma^2)
        let mut g = 0.0;
        for &yi in &self.y {
            g += (mu - yi) * self.inv_var;
        }
        Ok(vec![g])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedGaussianMeanModel { model: self }
    }
}

// ---------------------------------------------------------------------------
// Stress toy model: Neal's funnel (2D) for NUTS stability checks.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FunnelModel;

#[derive(Clone, Copy)]
struct PreparedFunnelModel<'a> {
    model: &'a FunnelModel,
}

impl PreparedNll for PreparedFunnelModel<'_> {
    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        self.model.nll(params)
    }
}

impl LogDensityModel for FunnelModel {
    type Prepared<'a>
        = PreparedFunnelModel<'a>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        2
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["y".to_string(), "x".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        if params.len() != 2 {
            return Err(NsError::Validation("expected 2 parameters".to_string()));
        }
        let y = params[0];
        let x = params[1];
        if !(y.is_finite() && x.is_finite()) {
            return Err(NsError::Validation("params must be finite".to_string()));
        }

        // Neal's funnel:
        // y ~ Normal(0, 3)
        // x | y ~ Normal(0, exp(y/2))
        //
        // Include stable constants (doesn't affect inference) to avoid accidental drift.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let nll_y = 0.5 * (y * y) / 9.0 + 3.0_f64.ln() + 0.5 * ln2pi;
        let nll_x = 0.5 * x * x * (-y).exp() + 0.5 * y + 0.5 * ln2pi;
        Ok(nll_y + nll_x)
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        if params.len() != 2 {
            return Err(NsError::Validation("expected 2 parameters".to_string()));
        }
        let y = params[0];
        let x = params[1];
        if !(y.is_finite() && x.is_finite()) {
            return Err(NsError::Validation("params must be finite".to_string()));
        }

        // d/dy: y/9 + 0.5 - 0.5*x^2*exp(-y)
        // d/dx: x*exp(-y)
        let exp_neg_y = (-y).exp();
        let dy = y / 9.0 - 0.5 * x * x * exp_neg_y + 0.5;
        let dx = x * exp_neg_y;
        Ok(vec![dy, dx])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedFunnelModel { model: self }
    }
}

/// Python wrapper for FunnelModel (Neal's funnel, 2D).
#[pyclass(name = "FunnelModel")]
struct PyFunnelModel {
    inner: FunnelModel,
}

#[pymethods]
impl PyFunnelModel {
    #[new]
    fn new() -> Self {
        Self { inner: FunnelModel }
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for GaussianMeanModel.
#[pyclass(name = "GaussianMeanModel")]
struct PyGaussianMeanModel {
    inner: GaussianMeanModel,
}

#[pymethods]
impl PyGaussianMeanModel {
    #[new]
    fn new(y: Vec<f64>, sigma: f64) -> PyResult<Self> {
        let inner =
            GaussianMeanModel::new(y, sigma).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

// ---------------------------------------------------------------------------
// Minimal non-HEP model: standard normal in R^d (useful for perf/profiling).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct StdNormalModel {
    dim: usize,
}

impl StdNormalModel {
    fn new(dim: usize) -> NsResult<Self> {
        if dim == 0 {
            return Err(NsError::Validation("dim must be >= 1".to_string()));
        }
        Ok(Self { dim })
    }
}

#[derive(Clone, Copy)]
struct PreparedStdNormalModel<'a> {
    model: &'a StdNormalModel,
}

impl PreparedNll for PreparedStdNormalModel<'_> {
    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        self.model.nll(params)
    }
}

impl LogDensityModel for StdNormalModel {
    type Prepared<'a>
        = PreparedStdNormalModel<'a>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.dim
    }

    fn parameter_names(&self) -> Vec<String> {
        (0..self.dim).map(|i| format!("x{i}")).collect()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; self.dim]
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        if params.len() != self.dim {
            return Err(NsError::Validation(format!(
                "expected {} parameters, got {}",
                self.dim,
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(NsError::Validation("params must be finite".to_string()));
        }
        // Standard normal: 0.5 * sum(x_i^2). Constants don't affect inference.
        Ok(0.5 * params.iter().map(|&v| v * v).sum::<f64>())
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        if params.len() != self.dim {
            return Err(NsError::Validation(format!(
                "expected {} parameters, got {}",
                self.dim,
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(NsError::Validation("params must be finite".to_string()));
        }
        Ok(params.to_vec())
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedStdNormalModel { model: self }
    }
}

/// Python wrapper for `StdNormalModel` (standard normal in R^d).
#[pyclass(name = "StdNormalModel")]
struct PyStdNormalModel {
    inner: StdNormalModel,
}

#[pymethods]
impl PyStdNormalModel {
    #[new]
    #[pyo3(signature = (dim=1))]
    fn new(dim: usize) -> PyResult<Self> {
        let inner = StdNormalModel::new(dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

// ---------------------------------------------------------------------------
// Regression models (Phase 6).
// ---------------------------------------------------------------------------

/// Python wrapper for `LinearRegressionModel` (Gaussian, sigma fixed to 1).
#[pyclass(name = "LinearRegressionModel")]
struct PyLinearRegressionModel {
    inner: RustLinearRegressionModel,
}

#[pymethods]
impl PyLinearRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true))]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> PyResult<Self> {
        let inner = RustLinearRegressionModel::new(x, y, include_intercept)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `LogisticRegressionModel`.
#[pyclass(name = "LogisticRegressionModel")]
struct PyLogisticRegressionModel {
    inner: RustLogisticRegressionModel,
}

#[pymethods]
impl PyLogisticRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true))]
    fn new(x: Vec<Vec<f64>>, y: Vec<u8>, include_intercept: bool) -> PyResult<Self> {
        let inner = RustLogisticRegressionModel::new(x, y, include_intercept)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `OrderedLogitModel` (ordinal logistic regression).
#[pyclass(name = "OrderedLogitModel")]
struct PyOrderedLogitModel {
    inner: RustOrderedLogitModel,
}

#[pymethods]
impl PyOrderedLogitModel {
    #[new]
    #[pyo3(signature = (x, y, *, n_levels))]
    fn new(x: Vec<Vec<f64>>, y: Vec<u8>, n_levels: usize) -> PyResult<Self> {
        let inner = RustOrderedLogitModel::new(x, y, n_levels)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `OrderedProbitModel` (ordinal probit regression).
#[pyclass(name = "OrderedProbitModel")]
struct PyOrderedProbitModel {
    inner: RustOrderedProbitModel,
}

#[pymethods]
impl PyOrderedProbitModel {
    #[new]
    #[pyo3(signature = (x, y, *, n_levels))]
    fn new(x: Vec<Vec<f64>>, y: Vec<u8>, n_levels: usize) -> PyResult<Self> {
        let inner = RustOrderedProbitModel::new(x, y, n_levels)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `PoissonRegressionModel`.
#[pyclass(name = "PoissonRegressionModel")]
struct PyPoissonRegressionModel {
    inner: RustPoissonRegressionModel,
}

#[pymethods]
impl PyPoissonRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true, offset=None))]
    fn new(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = RustPoissonRegressionModel::new(x, y, include_intercept, offset)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `NegativeBinomialRegressionModel` (NB2 mean/dispersion).
#[pyclass(name = "NegativeBinomialRegressionModel")]
struct PyNegativeBinomialRegressionModel {
    inner: RustNegativeBinomialRegressionModel,
}

#[pymethods]
impl PyNegativeBinomialRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true, offset=None))]
    fn new(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = RustNegativeBinomialRegressionModel::new(x, y, include_intercept, offset)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `GammaRegressionModel` (Gamma GLM, log link).
#[pyclass(name = "GammaRegressionModel")]
struct PyGammaRegressionModel {
    inner: RustGammaRegressionModel,
}

#[pymethods]
impl PyGammaRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true))]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> PyResult<Self> {
        let inner = RustGammaRegressionModel::new(x, y, include_intercept)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.grad_nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `TweedieRegressionModel` (Tweedie GLM, log link, p ∈ (1,2)).
#[pyclass(name = "TweedieRegressionModel")]
struct PyTweedieRegressionModel {
    inner: RustTweedieRegressionModel,
}

#[pymethods]
impl PyTweedieRegressionModel {
    #[new]
    #[pyo3(signature = (x, y, *, p=1.5, include_intercept=true))]
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>, p: f64, include_intercept: bool) -> PyResult<Self> {
        let inner = RustTweedieRegressionModel::new(x, y, include_intercept, p)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn power(&self) -> f64 {
        self.inner.power()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.grad_nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `GevModel` (Generalized Extreme Value).
#[pyclass(name = "GevModel")]
struct PyGevModel {
    inner: RustGevModel,
}

#[pymethods]
impl PyGevModel {
    #[new]
    fn new(data: Vec<f64>) -> PyResult<Self> {
        let inner = RustGevModel::new(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.grad_nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }

    #[staticmethod]
    fn return_level(params: Vec<f64>, return_period: f64) -> PyResult<f64> {
        RustGevModel::return_level(&params, return_period)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Python wrapper for `GpdModel` (Generalized Pareto Distribution).
#[pyclass(name = "GpdModel")]
struct PyGpdModel {
    inner: RustGpdModel,
}

#[pymethods]
impl PyGpdModel {
    #[new]
    fn new(exceedances: Vec<f64>) -> PyResult<Self> {
        let inner =
            RustGpdModel::new(exceedances).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.grad_nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }

    #[staticmethod]
    fn quantile(params: Vec<f64>, p: f64) -> PyResult<f64> {
        RustGpdModel::quantile(&params, p).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Python wrapper for `EightSchoolsModel` (hierarchical, non-centered).
#[pyclass(name = "EightSchoolsModel")]
struct PyEightSchoolsModel {
    inner: RustEightSchoolsModel,
}

#[pymethods]
impl PyEightSchoolsModel {
    #[new]
    #[pyo3(signature = (y, sigma, *, prior_mu_sigma=5.0, prior_tau_scale=5.0))]
    fn new(
        y: Vec<f64>,
        sigma: Vec<f64>,
        prior_mu_sigma: f64,
        prior_tau_scale: f64,
    ) -> PyResult<Self> {
        let inner = RustEightSchoolsModel::new(y, sigma, prior_mu_sigma, prior_tau_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.grad_nll(&params).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }
    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }
    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Fixed-effects meta-analysis (inverse-variance method).
#[pyfunction]
#[pyo3(signature = (estimates, standard_errors, *, labels=None, conf_level=0.95))]
fn meta_fixed(
    py: Python<'_>,
    estimates: Vec<f64>,
    standard_errors: Vec<f64>,
    labels: Option<Vec<String>>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let studies = build_study_effects(&estimates, &standard_errors, labels.as_deref())?;
    let res =
        rust_meta_fixed(&studies, conf_level).map_err(|e| PyValueError::new_err(e.to_string()))?;
    meta_result_to_pydict(py, &res)
}

/// Random-effects meta-analysis (DerSimonian–Laird method).
#[pyfunction]
#[pyo3(signature = (estimates, standard_errors, *, labels=None, conf_level=0.95))]
fn meta_random(
    py: Python<'_>,
    estimates: Vec<f64>,
    standard_errors: Vec<f64>,
    labels: Option<Vec<String>>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let studies = build_study_effects(&estimates, &standard_errors, labels.as_deref())?;
    let res =
        rust_meta_random(&studies, conf_level).map_err(|e| PyValueError::new_err(e.to_string()))?;
    meta_result_to_pydict(py, &res)
}

fn build_study_effects(
    estimates: &[f64],
    standard_errors: &[f64],
    labels: Option<&[String]>,
) -> PyResult<Vec<RustStudyEffect>> {
    if estimates.len() != standard_errors.len() {
        return Err(PyValueError::new_err("estimates and standard_errors must have same length"));
    }
    Ok(estimates
        .iter()
        .zip(standard_errors.iter())
        .enumerate()
        .map(|(i, (&est, &se))| RustStudyEffect {
            label: labels
                .and_then(|l| l.get(i))
                .cloned()
                .unwrap_or_else(|| format!("Study {}", i + 1)),
            estimate: est,
            se,
        })
        .collect())
}

fn meta_result_to_pydict(
    py: Python<'_>,
    res: &ns_inference::meta_analysis::MetaAnalysisResult,
) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    out.set_item("estimate", res.estimate)?;
    out.set_item("se", res.se)?;
    out.set_item("ci_lower", res.ci_lower)?;
    out.set_item("ci_upper", res.ci_upper)?;
    out.set_item("z", res.z)?;
    out.set_item("p_value", res.p_value)?;
    out.set_item("method", res.method)?;
    out.set_item("conf_level", res.conf_level)?;
    out.set_item("k", res.k)?;

    let het = PyDict::new(py);
    het.set_item("q", res.heterogeneity.q)?;
    het.set_item("df", res.heterogeneity.df)?;
    het.set_item("p_value", res.heterogeneity.p_value)?;
    het.set_item("i_squared", res.heterogeneity.i_squared)?;
    het.set_item("h_squared", res.heterogeneity.h_squared)?;
    het.set_item("tau_squared", res.heterogeneity.tau_squared)?;
    out.set_item("heterogeneity", het)?;

    let forest: Vec<_> = res
        .forest
        .iter()
        .map(|r| {
            let d = PyDict::new(py);
            d.set_item("label", &r.label).unwrap();
            d.set_item("estimate", r.estimate).unwrap();
            d.set_item("se", r.se).unwrap();
            d.set_item("ci_lower", r.ci_lower).unwrap();
            d.set_item("ci_upper", r.ci_upper).unwrap();
            d.set_item("weight", r.weight).unwrap();
            d
        })
        .collect();
    out.set_item("forest", forest)?;

    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Chain Ladder / Mack bindings
// ---------------------------------------------------------------------------

/// Deterministic Chain Ladder reserving from a cumulative claims triangle.
#[pyfunction]
#[pyo3(signature = (triangle))]
fn chain_ladder(py: Python<'_>, triangle: Vec<Vec<f64>>) -> PyResult<Py<PyAny>> {
    let tri = ClaimsTriangle::new(triangle).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let res = rust_chain_ladder(&tri).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("development_factors", &res.development_factors)?;
    out.set_item("cumulative_factors", &res.cumulative_factors)?;

    let ultimates: Vec<f64> = res.rows.iter().map(|r| r.ultimate).collect();
    let ibnr: Vec<f64> = res.rows.iter().map(|r| r.ibnr).collect();
    let latest: Vec<f64> = res.rows.iter().map(|r| r.latest).collect();
    out.set_item("ultimates", &ultimates)?;
    out.set_item("ibnr", &ibnr)?;
    out.set_item("latest", &latest)?;
    out.set_item("total_ibnr", res.total_ibnr)?;
    out.set_item("projected", &res.projected)?;

    Ok(out.into_any().unbind())
}

/// Mack Chain Ladder with prediction standard errors.
#[pyfunction]
#[pyo3(signature = (triangle, *, conf_level=0.95))]
fn mack_chain_ladder(
    py: Python<'_>,
    triangle: Vec<Vec<f64>>,
    conf_level: f64,
) -> PyResult<Py<PyAny>> {
    let tri = ClaimsTriangle::new(triangle).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let res = rust_mack_chain_ladder(&tri, conf_level)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("development_factors", &res.development_factors)?;
    out.set_item("sigma_sq", &res.sigma_sq)?;

    let ultimates: Vec<f64> = res.rows.iter().map(|r| r.ultimate).collect();
    let ibnr: Vec<f64> = res.rows.iter().map(|r| r.ibnr).collect();
    let latest: Vec<f64> = res.rows.iter().map(|r| r.latest).collect();
    let se: Vec<f64> = res.rows.iter().map(|r| r.se).collect();
    let cv: Vec<f64> = res.rows.iter().map(|r| r.cv).collect();
    let pi_lower: Vec<f64> = res.rows.iter().map(|r| r.pi_lower).collect();
    let pi_upper: Vec<f64> = res.rows.iter().map(|r| r.pi_upper).collect();

    out.set_item("ultimates", &ultimates)?;
    out.set_item("ibnr", &ibnr)?;
    out.set_item("latest", &latest)?;
    out.set_item("se", &se)?;
    out.set_item("cv", &cv)?;
    out.set_item("pi_lower", &pi_lower)?;
    out.set_item("pi_upper", &pi_upper)?;
    out.set_item("total_ibnr", res.total_ibnr)?;
    out.set_item("total_se", res.total_se)?;
    out.set_item("conf_level", res.conf_level)?;

    Ok(out.into_any().unbind())
}

/// Python wrapper for `ComposedGlmModel` built via `ModelBuilder`.
#[pyclass(name = "ComposedGlmModel")]
struct PyComposedGlmModel {
    inner: RustComposedGlmModel,
}

#[pymethods]
impl PyComposedGlmModel {
    /// Build a composed Gaussian linear regression model (optionally with inferred sigma_y).
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false, obs_sigma_prior_m=None, obs_sigma_prior_s=None, random_intercept_non_centered=false, random_slope_feature_idx=None, random_slope_non_centered=false, correlated_feature_idx=None, lkj_eta=1.0))]
    fn linear_regression(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        include_intercept: bool,
        group_idx: Option<Vec<usize>>,
        n_groups: Option<usize>,
        coef_prior_mu: f64,
        coef_prior_sigma: f64,
        penalize_intercept: bool,
        obs_sigma_prior_m: Option<f64>,
        obs_sigma_prior_s: Option<f64>,
        random_intercept_non_centered: bool,
        random_slope_feature_idx: Option<usize>,
        random_slope_non_centered: bool,
        correlated_feature_idx: Option<usize>,
        lkj_eta: f64,
    ) -> PyResult<Self> {
        if group_idx.is_none() && n_groups.is_some() {
            return Err(PyValueError::new_err("n_groups requires group_idx"));
        }

        let mut b = RustModelBuilder::linear_regression(x, y, include_intercept)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b
            .with_coef_prior_normal(coef_prior_mu, coef_prior_sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b.with_penalize_intercept(penalize_intercept);

        if obs_sigma_prior_m.is_some() != obs_sigma_prior_s.is_some() {
            return Err(PyValueError::new_err(
                "obs_sigma_prior_m and obs_sigma_prior_s must be set together",
            ));
        }
        if let (Some(m), Some(s)) = (obs_sigma_prior_m, obs_sigma_prior_s) {
            b = b
                .with_gaussian_obs_sigma_prior_lognormal((m, s))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        if correlated_feature_idx.is_some()
            && (random_slope_feature_idx.is_some()
                || random_intercept_non_centered
                || random_slope_non_centered)
        {
            return Err(PyValueError::new_err(
                "correlated_feature_idx cannot be combined with random_slope_feature_idx or non-centered toggles",
            ));
        }

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);

            if let Some(feature_idx) = correlated_feature_idx {
                b = b
                    .with_correlated_random_intercept_slope(feature_idx, group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                b = b
                    .with_correlated_lkj_eta(lkj_eta)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                b = b
                    .with_random_intercept(group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                if random_intercept_non_centered {
                    b = b
                        .with_random_intercept_non_centered(true)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }

                if let Some(feature_idx) = random_slope_feature_idx {
                    b = b
                        .with_random_slope(feature_idx, group_idx, ng)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    if random_slope_non_centered {
                        b = b
                            .with_random_slope_non_centered(true)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    }
                }
            }
        } else if correlated_feature_idx.is_some() || random_slope_feature_idx.is_some() {
            return Err(PyValueError::new_err(
                "random slopes / correlated effects require group_idx",
            ));
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a composed logistic regression model (Bernoulli-logit).
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false, random_intercept_non_centered=false, random_slope_feature_idx=None, random_slope_non_centered=false, correlated_feature_idx=None, lkj_eta=1.0))]
    fn logistic_regression(
        x: Vec<Vec<f64>>,
        y: Vec<u8>,
        include_intercept: bool,
        group_idx: Option<Vec<usize>>,
        n_groups: Option<usize>,
        coef_prior_mu: f64,
        coef_prior_sigma: f64,
        penalize_intercept: bool,
        random_intercept_non_centered: bool,
        random_slope_feature_idx: Option<usize>,
        random_slope_non_centered: bool,
        correlated_feature_idx: Option<usize>,
        lkj_eta: f64,
    ) -> PyResult<Self> {
        if group_idx.is_none() && n_groups.is_some() {
            return Err(PyValueError::new_err("n_groups requires group_idx"));
        }

        let mut b = RustModelBuilder::logistic_regression(x, y, include_intercept)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b
            .with_coef_prior_normal(coef_prior_mu, coef_prior_sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b.with_penalize_intercept(penalize_intercept);

        if correlated_feature_idx.is_some()
            && (random_slope_feature_idx.is_some()
                || random_intercept_non_centered
                || random_slope_non_centered)
        {
            return Err(PyValueError::new_err(
                "correlated_feature_idx cannot be combined with random_slope_feature_idx or non-centered toggles",
            ));
        }

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);

            if let Some(feature_idx) = correlated_feature_idx {
                b = b
                    .with_correlated_random_intercept_slope(feature_idx, group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                b = b
                    .with_correlated_lkj_eta(lkj_eta)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                b = b
                    .with_random_intercept(group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                if random_intercept_non_centered {
                    b = b
                        .with_random_intercept_non_centered(true)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }

                if let Some(feature_idx) = random_slope_feature_idx {
                    b = b
                        .with_random_slope(feature_idx, group_idx, ng)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    if random_slope_non_centered {
                        b = b
                            .with_random_slope_non_centered(true)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    }
                }
            }
        } else if correlated_feature_idx.is_some() || random_slope_feature_idx.is_some() {
            return Err(PyValueError::new_err(
                "random slopes / correlated effects require group_idx",
            ));
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a composed Poisson regression model (log link) with optional offset.
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, offset=None, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false, random_intercept_non_centered=false, random_slope_feature_idx=None, random_slope_non_centered=false, correlated_feature_idx=None, lkj_eta=1.0))]
    fn poisson_regression(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
        group_idx: Option<Vec<usize>>,
        n_groups: Option<usize>,
        coef_prior_mu: f64,
        coef_prior_sigma: f64,
        penalize_intercept: bool,
        random_intercept_non_centered: bool,
        random_slope_feature_idx: Option<usize>,
        random_slope_non_centered: bool,
        correlated_feature_idx: Option<usize>,
        lkj_eta: f64,
    ) -> PyResult<Self> {
        if group_idx.is_none() && n_groups.is_some() {
            return Err(PyValueError::new_err("n_groups requires group_idx"));
        }

        let mut b = RustModelBuilder::poisson_regression(x, y, include_intercept, offset)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b
            .with_coef_prior_normal(coef_prior_mu, coef_prior_sigma)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        b = b.with_penalize_intercept(penalize_intercept);

        if correlated_feature_idx.is_some()
            && (random_slope_feature_idx.is_some()
                || random_intercept_non_centered
                || random_slope_non_centered)
        {
            return Err(PyValueError::new_err(
                "correlated_feature_idx cannot be combined with random_slope_feature_idx or non-centered toggles",
            ));
        }

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);

            if let Some(feature_idx) = correlated_feature_idx {
                b = b
                    .with_correlated_random_intercept_slope(feature_idx, group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                b = b
                    .with_correlated_lkj_eta(lkj_eta)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                b = b
                    .with_random_intercept(group_idx.clone(), ng)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                if random_intercept_non_centered {
                    b = b
                        .with_random_intercept_non_centered(true)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                }

                if let Some(feature_idx) = random_slope_feature_idx {
                    b = b
                        .with_random_slope(feature_idx, group_idx, ng)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    if random_slope_non_centered {
                        b = b
                            .with_random_slope_non_centered(true)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    }
                }
            }
        } else if correlated_feature_idx.is_some() || random_slope_feature_idx.is_some() {
            return Err(PyValueError::new_err(
                "random slopes / correlated effects require group_idx",
            ));
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

// ---------------------------------------------------------------------------
// Survival models (Phase 9 Pack A).
// ---------------------------------------------------------------------------

fn parse_cox_ties(ties: &str) -> PyResult<RustCoxTies> {
    match ties.to_ascii_lowercase().as_str() {
        "breslow" => Ok(RustCoxTies::Breslow),
        "efron" => Ok(RustCoxTies::Efron),
        _ => Err(PyValueError::new_err("Invalid ties policy: expected 'efron' or 'breslow'")),
    }
}

/// Python wrapper for `ExponentialSurvivalModel` (right-censoring).
#[pyclass(name = "ExponentialSurvivalModel")]
struct PyExponentialSurvivalModel {
    inner: RustExponentialSurvivalModel,
}

#[pymethods]
impl PyExponentialSurvivalModel {
    #[new]
    fn new(times: Vec<f64>, events: Vec<bool>) -> PyResult<Self> {
        let inner = RustExponentialSurvivalModel::new(times, events)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `WeibullSurvivalModel` (right-censoring).
#[pyclass(name = "WeibullSurvivalModel")]
struct PyWeibullSurvivalModel {
    inner: RustWeibullSurvivalModel,
}

#[pymethods]
impl PyWeibullSurvivalModel {
    #[new]
    fn new(times: Vec<f64>, events: Vec<bool>) -> PyResult<Self> {
        let inner = RustWeibullSurvivalModel::new(times, events)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `LogNormalAftModel` (right-censoring).
#[pyclass(name = "LogNormalAftModel")]
struct PyLogNormalAftModel {
    inner: RustLogNormalAftModel,
}

#[pymethods]
impl PyLogNormalAftModel {
    #[new]
    fn new(times: Vec<f64>, events: Vec<bool>) -> PyResult<Self> {
        let inner = RustLogNormalAftModel::new(times, events)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `CoxPhModel` (partial likelihood; right-censoring).
#[pyclass(name = "CoxPhModel")]
struct PyCoxPhModel {
    inner: RustCoxPhModel,
}

#[pymethods]
impl PyCoxPhModel {
    #[new]
    #[pyo3(signature = (times, events, x, *, ties = "efron"))]
    fn new(times: Vec<f64>, events: Vec<bool>, x: Vec<Vec<f64>>, ties: &str) -> PyResult<Self> {
        let ties = parse_cox_ties(ties)?;
        let inner = RustCoxPhModel::new(times, events, x, ties)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

// ---------------------------------------------------------------------------
// Pharmacometrics (Phase 13): PK / NLME baselines.
// ---------------------------------------------------------------------------

fn parse_lloq_policy(policy: &str) -> PyResult<RustLloqPolicy> {
    match policy.to_ascii_lowercase().as_str() {
        "ignore" => Ok(RustLloqPolicy::Ignore),
        "replace_half" => Ok(RustLloqPolicy::ReplaceHalf),
        "censored" => Ok(RustLloqPolicy::Censored),
        _ => Err(PyValueError::new_err(
            "Invalid lloq_policy: expected 'ignore', 'replace_half', or 'censored'",
        )),
    }
}

#[inline]
fn conc_oral(dose: f64, bioavailability: f64, cl: f64, v: f64, ka: f64, t: f64) -> f64 {
    let ke = cl / v;
    let d = ka - ke;
    let d_amt = dose * bioavailability;
    let pref = d_amt / v;

    let eke = (-ke * t).exp();
    let s = if d.abs() < 1e-10 { t } else { (-(-d * t).exp_m1()) / d };

    pref * ka * eke * s
}

/// Python wrapper for `OneCompartmentOralPkModel` (oral, first-order absorption).
#[pyclass(name = "OneCompartmentOralPkModel")]
struct PyOneCompartmentOralPkModel {
    inner: RustOneCompartmentOralPkModel,
    times: Vec<f64>,
    dose: f64,
    bioavailability: f64,
}

#[pymethods]
impl PyOneCompartmentOralPkModel {
    #[new]
    #[pyo3(signature = (times, y, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored"))]
    fn new(
        times: Vec<f64>,
        y: Vec<f64>,
        dose: f64,
        bioavailability: f64,
        sigma: f64,
        lloq: Option<f64>,
        lloq_policy: &str,
    ) -> PyResult<Self> {
        let lloq_policy = parse_lloq_policy(lloq_policy)?;
        let times_for_pred = times.clone();
        let inner = RustOneCompartmentOralPkModel::new(
            times,
            y,
            dose,
            bioavailability,
            sigma,
            lloq,
            lloq_policy,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner, times: times_for_pred, dose, bioavailability })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn predict(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        if params.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "expected 3 parameters, got {}",
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(PyValueError::new_err("params must be finite and > 0"));
        }
        let cl = params[0];
        let v = params[1];
        let ka = params[2];
        Ok(self
            .times
            .iter()
            .map(|&t| conc_oral(self.dose, self.bioavailability, cl, v, ka, t))
            .collect())
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for `OneCompartmentOralPkNlmeModel` (population + random effects).
#[pyclass(name = "OneCompartmentOralPkNlmeModel")]
struct PyOneCompartmentOralPkNlmeModel {
    inner: RustOneCompartmentOralPkNlmeModel,
}

#[pymethods]
impl PyOneCompartmentOralPkNlmeModel {
    #[new]
    #[pyo3(signature = (times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored"))]
    fn new(
        times: Vec<f64>,
        y: Vec<f64>,
        subject_idx: Vec<usize>,
        n_subjects: usize,
        dose: f64,
        bioavailability: f64,
        sigma: f64,
        lloq: Option<f64>,
        lloq_policy: &str,
    ) -> PyResult<Self> {
        let lloq_policy = parse_lloq_policy(lloq_policy)?;
        let inner = RustOneCompartmentOralPkNlmeModel::new(
            times,
            y,
            subject_idx,
            n_subjects,
            dose,
            bioavailability,
            sigma,
            lloq,
            lloq_policy,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

// ---------------------------------------------------------------------------
// LMM (Phase 9 Pack B).
// ---------------------------------------------------------------------------

/// Python wrapper for `LmmMarginalModel` (Gaussian mixed model, marginal likelihood).
#[pyclass(name = "LmmMarginalModel")]
struct PyLmmMarginalModel {
    inner: RustLmmMarginalModel,
}

#[pymethods]
impl PyLmmMarginalModel {
    #[new]
    #[pyo3(signature = (x, y, *, include_intercept=true, group_idx, n_groups=None, random_slope_feature_idx=None))]
    fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        include_intercept: bool,
        group_idx: Vec<usize>,
        n_groups: Option<usize>,
        random_slope_feature_idx: Option<usize>,
    ) -> PyResult<Self> {
        let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);
        let re = if let Some(k) = random_slope_feature_idx {
            RustLmmRandomEffects::InterceptSlope { feature_idx: k }
        } else {
            RustLmmRandomEffects::Intercept
        };
        let inner = RustLmmMarginalModel::new(x, y, include_intercept, group_idx, ng, re)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
    }

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    fn suggested_init(&self) -> Vec<f64> {
        self.inner.parameter_init()
    }

    fn suggested_bounds(&self) -> Vec<(f64, f64)> {
        self.inner.parameter_bounds()
    }
}

/// Python wrapper for FitResult
#[pyclass(name = "FitResult")]
struct PyFitResult {
    #[pyo3(get)]
    parameters: Vec<f64>,
    #[pyo3(get)]
    uncertainties: Vec<f64>,
    #[pyo3(get)]
    nll: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    n_iter: usize,
    #[pyo3(get)]
    n_fev: usize,
    #[pyo3(get)]
    n_gev: usize,
    #[pyo3(get)]
    termination_reason: String,
    #[pyo3(get)]
    final_grad_norm: f64,
    #[pyo3(get)]
    initial_nll: f64,
    #[pyo3(get)]
    n_active_bounds: usize,
}

#[pymethods]
impl PyFitResult {
    /// Alias: `bestfit` (pyhf-style).
    #[getter]
    fn bestfit(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    /// Alias: `twice_nll = 2 * nll` (pyhf-style).
    #[getter]
    fn twice_nll(&self) -> f64 {
        2.0 * self.nll
    }

    /// Alias: `success` (pyhf-style).
    #[getter]
    fn success(&self) -> bool {
        self.converged
    }

    /// Back-compat alias: `n_evaluations` historically contained argmin iterations.
    #[getter]
    fn n_evaluations(&self) -> usize {
        self.n_iter
    }
}

impl From<ns_core::FitResult> for PyFitResult {
    fn from(r: ns_core::FitResult) -> Self {
        PyFitResult {
            parameters: r.parameters,
            uncertainties: r.uncertainties,
            nll: r.nll,
            converged: r.converged,
            n_iter: r.n_iter,
            n_fev: r.n_fev,
            n_gev: r.n_gev,
            termination_reason: r.termination_reason,
            final_grad_norm: r.final_grad_norm,
            initial_nll: r.initial_nll,
            n_active_bounds: r.n_active_bounds,
        }
    }
}

/// Python wrapper for OptimizationResult (fast-path: no covariance/Hessian).
#[pyclass(name = "FitMinimumResult")]
struct PyFitMinimumResult {
    #[pyo3(get)]
    parameters: Vec<f64>,
    /// Objective value at the minimum (NLL for likelihood models).
    #[pyo3(get)]
    nll: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    n_iter: u64,
    #[pyo3(get)]
    n_fev: usize,
    #[pyo3(get)]
    n_gev: usize,
    /// Termination message from the optimizer.
    #[pyo3(get)]
    message: String,
    /// Objective value at the initial point (before optimisation).
    #[pyo3(get)]
    initial_nll: f64,
    /// Final gradient vector (None for gradient-free paths).
    #[pyo3(get)]
    final_gradient: Option<Vec<f64>>,
}

#[pymethods]
impl PyFitMinimumResult {
    /// Alias: `bestfit` (pyhf-style).
    #[getter]
    fn bestfit(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    /// Alias: `twice_nll = 2 * nll` (pyhf-style).
    #[getter]
    fn twice_nll(&self) -> f64 {
        2.0 * self.nll
    }

    /// Alias: `success` (pyhf-style).
    #[getter]
    fn success(&self) -> bool {
        self.converged
    }
}

impl From<RustOptimizationResult> for PyFitMinimumResult {
    fn from(r: RustOptimizationResult) -> Self {
        PyFitMinimumResult {
            parameters: r.parameters,
            nll: r.fval,
            converged: r.converged,
            n_iter: r.n_iter,
            n_fev: r.n_fev,
            n_gev: r.n_gev,
            message: r.message,
            initial_nll: r.initial_cost,
            final_gradient: r.final_gradient,
        }
    }
}

/// Python wrapper for MaximumLikelihoodEstimator
#[pyclass(name = "MaximumLikelihoodEstimator")]
struct PyMaximumLikelihoodEstimator {
    inner: RustMLE,
}

#[pymethods]
impl PyMaximumLikelihoodEstimator {
    #[new]
    #[pyo3(signature = (*, max_iter=1000, tol=1e-6, m=0, smooth_bounds=false))]
    fn new(max_iter: u64, tol: f64, m: usize, smooth_bounds: bool) -> PyResult<Self> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(PyValueError::new_err("tol must be finite and >= 0"));
        }
        let cfg = OptimizerConfig { max_iter, tol, m, smooth_bounds };
        Ok(PyMaximumLikelihoodEstimator { inner: RustMLE::with_config(cfg) })
    }

    /// Fit any supported model (generic `LogDensityModel`) using MLE.
    ///
    /// `data=` is only supported for `HistFactoryModel` (overrides main-bin observations).
    /// `init_pars=` overrides the model's default initial parameters (warm-start).
    #[pyo3(signature = (model, *, data=None, init_pars=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        data: Option<Vec<f64>>,
        init_pars: Option<Vec<f64>>,
    ) -> PyResult<PyFitResult> {
        let fit_model = extract_posterior_model_with_data(model, data)?;

        if let Some(ref ip) = init_pars {
            validate_f64_vec("init_pars", ip, fit_model.dim())?;
        }

        let mle = self.inner.clone();
        let result = py
            .detach(move || {
                if let Some(ip) = init_pars {
                    fit_model.fit_mle_from(&mle, &ip)
                } else {
                    fit_model.fit_mle(&mle)
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;

        Ok(PyFitResult::from(result))
    }

    /// Minimize NLL and return the fast-path optimizer result (no covariance/Hessian).
    ///
    /// This is the recommended entrypoint for profile likelihood scans and conditional fits.
    ///
    /// `data=` is only supported for `HistFactoryModel` (overrides main-bin observations).
    /// `init_pars=` overrides the model's default initial parameters (warm-start).
    /// `bounds=` (HistFactoryModel only) overrides parameter bounds; clamp a parameter to
    /// `(value, value)` to fix it.
    #[pyo3(signature = (model, *, data=None, init_pars=None, bounds=None))]
    fn fit_minimum<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        data: Option<Vec<f64>>,
        init_pars: Option<Vec<f64>>,
        bounds: Option<Vec<(f64, f64)>>,
    ) -> PyResult<PyFitMinimumResult> {
        let fit_model = extract_posterior_model_with_data(model, data)?;

        if let Some(ref ip) = init_pars {
            validate_f64_vec("init_pars", ip, fit_model.dim())?;
        }
        if let Some(ref b) = bounds {
            validate_bounds("bounds", b, fit_model.dim())?;
        }
        if bounds.is_some() && !matches!(fit_model, PosteriorModel::HistFactory(_)) {
            return Err(PyValueError::new_err(
                "bounds= is currently only supported for HistFactoryModel",
            ));
        }

        let mle = self.inner.clone();
        let result = py
            .detach(move || {
                let init_slice = init_pars.as_deref();
                match fit_model {
                    PosteriorModel::HistFactory(m) => {
                        let init_storage;
                        let init: &[f64] = match init_slice {
                            Some(ip) => ip,
                            None => {
                                init_storage = m.parameter_init();
                                &init_storage
                            }
                        };

                        let bounds_storage;
                        let b: &[(f64, f64)] = match bounds.as_deref() {
                            Some(bs) => bs,
                            None => {
                                bounds_storage = m.parameter_bounds();
                                &bounds_storage
                            }
                        };

                        let mut tape = AdTape::with_capacity(m.n_params() * 20);
                        mle.fit_minimum_histfactory_from_with_bounds_with_tape(
                            &m, init, b, &mut tape,
                        )
                    }
                    PosteriorModel::Unbinned(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::Hybrid(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::GaussianMean(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::Funnel(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::StdNormal(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::LinearRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::LogisticRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::OrderedLogit(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::OrderedProbit(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::PoissonRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::NegativeBinomialRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::ComposedGlm(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::LmmMarginal(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::ExponentialSurvival(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::WeibullSurvival(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::LogNormalAft(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::CoxPh(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::OneCompartmentOralPk(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::OneCompartmentOralPkNlme(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::GammaRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::TweedieRegression(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::Gev(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::Gpd(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                    PosteriorModel::EightSchools(m) => match init_slice {
                        Some(ip) => mle.fit_minimum_from(&m, ip),
                        None => mle.fit_minimum(&m),
                    },
                }
            })
            .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;

        Ok(PyFitMinimumResult::from(result))
    }

    /// Fit multiple models in parallel (Rayon).
    ///
    /// Two call forms:
    /// - `fit_batch(models)` — list of models, each with its own observations
    /// - `fit_batch(model, datasets)` — HistFactoryModel + list of observation vectors
    ///
    /// Notes:
    /// - Lists must be homogeneous (all items of the same supported model type).
    /// - `datasets=` is only supported for HistFactoryModel.
    #[pyo3(signature = (models_or_model, datasets=None))]
    fn fit_batch<'py>(
        &self,
        py: Python<'py>,
        models_or_model: &Bound<'py, PyAny>,
        datasets: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<PyFitResult>> {
        let mle = self.inner.clone();

        if let Some(datasets) = datasets {
            // Single model + multiple datasets
            let hf = models_or_model.extract::<PyRef<'_, PyHistFactoryModel>>().map_err(|_| {
                PyValueError::new_err(
                    "When datasets is given, first argument must be a HistFactoryModel",
                )
            })?;
            let base = hf.inner.clone();
            let models: Vec<RustModel> = datasets
                .into_iter()
                .map(|ds| {
                    base.with_observed_main(&ds).map_err(|e| {
                        PyValueError::new_err(format!("Failed to set observed data: {}", e))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;

            let results = py.detach(move || mle.fit_batch(&models));

            results
                .into_iter()
                .map(|r| {
                    let r = r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                    Ok(PyFitResult::from(r))
                })
                .collect()
        } else {
            // List of models
            let list = models_or_model.cast::<PyList>().map_err(|_| {
                PyValueError::new_err(
                    "Expected a list of models or (HistFactoryModel, datasets=...)",
                )
            })?;

            if list.is_empty() {
                return Err(PyValueError::new_err("models list must be non-empty"));
            }

            let first = list.get_item(0)?;

            // Determine the concrete model type from the first item, then enforce homogeneity.
            if first.extract::<PyRef<'_, PyHistFactoryModel>>().is_ok() {
                let mut models: Vec<RustModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let hf = item.extract::<PyRef<'_, PyHistFactoryModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be HistFactoryModel")
                    })?;
                    models.push(hf.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyUnbinnedModel>>().is_ok() {
                let mut models: Vec<RustUnbinnedModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let ub = item.extract::<PyRef<'_, PyUnbinnedModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be UnbinnedModel")
                    })?;
                    models.push(ub.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyGaussianMeanModel>>().is_ok() {
                let mut models: Vec<GaussianMeanModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let gm = item.extract::<PyRef<'_, PyGaussianMeanModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be GaussianMeanModel")
                    })?;
                    models.push(gm.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyLinearRegressionModel>>().is_ok() {
                let mut models: Vec<RustLinearRegressionModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let lr =
                        item.extract::<PyRef<'_, PyLinearRegressionModel>>().map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be LinearRegressionModel",
                            )
                        })?;
                    models.push(lr.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyLogisticRegressionModel>>().is_ok() {
                let mut models: Vec<RustLogisticRegressionModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let logit =
                        item.extract::<PyRef<'_, PyLogisticRegressionModel>>().map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be LogisticRegressionModel",
                            )
                        })?;
                    models.push(logit.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyOrderedLogitModel>>().is_ok() {
                let mut models: Vec<RustOrderedLogitModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyOrderedLogitModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be OrderedLogitModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyOrderedProbitModel>>().is_ok() {
                let mut models: Vec<RustOrderedProbitModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyOrderedProbitModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be OrderedProbitModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyPoissonRegressionModel>>().is_ok() {
                let mut models: Vec<RustPoissonRegressionModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let pois =
                        item.extract::<PyRef<'_, PyPoissonRegressionModel>>().map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be PoissonRegressionModel",
                            )
                        })?;
                    models.push(pois.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyNegativeBinomialRegressionModel>>().is_ok() {
                let mut models: Vec<RustNegativeBinomialRegressionModel> =
                    Vec::with_capacity(list.len());
                for item in list.iter() {
                    let nb = item
                        .extract::<PyRef<'_, PyNegativeBinomialRegressionModel>>()
                        .map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be NegativeBinomialRegressionModel",
                            )
                        })?;
                    models.push(nb.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyComposedGlmModel>>().is_ok() {
                let mut models: Vec<RustComposedGlmModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let glm = item.extract::<PyRef<'_, PyComposedGlmModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be ComposedGlmModel")
                    })?;
                    models.push(glm.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyLmmMarginalModel>>().is_ok() {
                let mut models: Vec<RustLmmMarginalModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyLmmMarginalModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be LmmMarginalModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyExponentialSurvivalModel>>().is_ok() {
                let mut models: Vec<RustExponentialSurvivalModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m =
                        item.extract::<PyRef<'_, PyExponentialSurvivalModel>>().map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be ExponentialSurvivalModel",
                            )
                        })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyWeibullSurvivalModel>>().is_ok() {
                let mut models: Vec<RustWeibullSurvivalModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyWeibullSurvivalModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be WeibullSurvivalModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyLogNormalAftModel>>().is_ok() {
                let mut models: Vec<RustLogNormalAftModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyLogNormalAftModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be LogNormalAftModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyCoxPhModel>>().is_ok() {
                let mut models: Vec<RustCoxPhModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyCoxPhModel>>().map_err(|_| {
                        PyValueError::new_err("All items in the list must be CoxPhModel")
                    })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyOneCompartmentOralPkModel>>().is_ok() {
                let mut models: Vec<RustOneCompartmentOralPkModel> = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m =
                        item.extract::<PyRef<'_, PyOneCompartmentOralPkModel>>().map_err(|_| {
                            PyValueError::new_err(
                                "All items in the list must be OneCompartmentOralPkModel",
                            )
                        })?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            if first.extract::<PyRef<'_, PyOneCompartmentOralPkNlmeModel>>().is_ok() {
                let mut models: Vec<RustOneCompartmentOralPkNlmeModel> =
                    Vec::with_capacity(list.len());
                for item in list.iter() {
                    let m = item.extract::<PyRef<'_, PyOneCompartmentOralPkNlmeModel>>().map_err(
                        |_| {
                            PyValueError::new_err(
                                "All items in the list must be OneCompartmentOralPkNlmeModel",
                            )
                        },
                    )?;
                    models.push(m.inner.clone());
                }

                let results = py.detach(move || mle.fit_batch(&models));
                return results
                    .into_iter()
                    .map(|r| {
                        let r =
                            r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                        Ok(PyFitResult::from(r))
                    })
                    .collect();
            }

            Err(PyValueError::new_err(
                "Unsupported model type. Expected HistFactoryModel, GaussianMeanModel, a regression model, ComposedGlmModel, LmmMarginalModel, a survival model, or a PK model.",
            ))
        }
    }

    /// Generate Poisson toy pseudo-experiments and fit each in parallel.
    #[pyo3(signature = (model, params, *, n_toys=1000, seed=42))]
    fn fit_toys(
        &self,
        py: Python<'_>,
        model: &PyHistFactoryModel,
        params: Vec<f64>,
        n_toys: usize,
        seed: u64,
    ) -> PyResult<Vec<PyFitResult>> {
        let mle = self.inner.clone();
        let m = model.inner.clone();

        let results = py.detach(move || mle.fit_toys(&m, &params, n_toys, seed));

        results
            .into_iter()
            .map(|r| {
                let r = r.map_err(|e| PyValueError::new_err(format!("Toy fit failed: {}", e)))?;
                Ok(PyFitResult::from(r))
            })
            .collect()
    }

    /// Compute nuisance-parameter ranking (impact on POI).
    ///
    /// Returns a list of dicts with keys: name, delta_mu_up, delta_mu_down, pull, constraint.
    fn ranking<'py>(
        &self,
        py: Python<'py>,
        model: &PyHistFactoryModel,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mle = self.inner.clone();
        let m = model.inner.clone();

        let entries: Vec<RankingEntry> = py
            .detach(move || mle.ranking(&m))
            .map_err(|e| PyValueError::new_err(format!("Ranking failed: {}", e)))?;

        entries
            .into_iter()
            .map(|e| {
                let d = PyDict::new(py);
                d.set_item("name", e.name)?;
                d.set_item("delta_mu_up", e.delta_mu_up)?;
                d.set_item("delta_mu_down", e.delta_mu_down)?;
                d.set_item("pull", e.pull)?;
                d.set_item("constraint", e.constraint)?;
                Ok(d.into_any().unbind())
            })
            .collect()
    }

    /// Discovery-style `q0` and gradient w.r.t. one sample's nominal yields (main bins).
    ///
    /// This mutates the provided `HistFactoryModel` in place by overriding the specified
    /// sample's nominal yields before computing the profiled test statistic.
    ///
    /// Notes:
    /// - The overridden sample must have only multiplicative modifiers (NormFactor/NormSys/Lumi).
    /// - This is intended for ML/training-loop use; prefer reusing a single model instance.
    #[pyo3(signature = (model, *, channel, sample, nominal))]
    fn q0_like_loss_and_grad_nominal(
        &self,
        model: &mut PyHistFactoryModel,
        channel: &str,
        sample: &str,
        nominal: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let nominal = extract_f64_vec(nominal)?;
        if nominal.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("nominal must contain only finite values"));
        }

        let (ch_idx, s_idx) = model
            .inner
            .set_sample_nominal_by_name(channel, sample, &nominal)
            .map_err(|e| PyValueError::new_err(format!("Failed to set sample nominal: {}", e)))?;

        let (q0, grad) = self
            .inner
            .q0_like_loss_and_grad_sample_nominal(&model.inner, ch_idx, s_idx)
            .map_err(|e| PyValueError::new_err(format!("q0_like failed: {}", e)))?;

        Ok((q0, grad))
    }

    /// Upper-limit-style `q_mu` and gradient w.r.t. one sample's nominal yields (main bins).
    ///
    /// This mutates the provided `HistFactoryModel` in place by overriding the specified
    /// sample's nominal yields before computing the profiled test statistic.
    #[pyo3(signature = (model, *, mu_test, channel, sample, nominal))]
    fn qmu_like_loss_and_grad_nominal(
        &self,
        model: &mut PyHistFactoryModel,
        mu_test: f64,
        channel: &str,
        sample: &str,
        nominal: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Vec<f64>)> {
        if !mu_test.is_finite() {
            return Err(PyValueError::new_err("mu_test must be finite"));
        }

        let nominal = extract_f64_vec(nominal)?;
        if nominal.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("nominal must contain only finite values"));
        }

        let (ch_idx, s_idx) = model
            .inner
            .set_sample_nominal_by_name(channel, sample, &nominal)
            .map_err(|e| PyValueError::new_err(format!("Failed to set sample nominal: {}", e)))?;

        let (q, grad) = self
            .inner
            .qmu_like_loss_and_grad_sample_nominal(&model.inner, mu_test, ch_idx, s_idx)
            .map_err(|e| PyValueError::new_err(format!("qmu_like failed: {}", e)))?;

        Ok((q, grad))
    }
}

/// Convenience wrapper: create model from pyhf JSON string.
#[pyfunction]
fn from_pyhf(json_str: &str) -> PyResult<PyHistFactoryModel> {
    PyHistFactoryModel::from_workspace(json_str)
}

// ---------------------------------------------------------------------------
// Arrow / Parquet integration
// ---------------------------------------------------------------------------

/// Create a HistFactoryModel from Arrow IPC bytes.
///
/// The IPC stream must contain a table with columns:
/// `channel` (Utf8), `sample` (Utf8), `yields` (List<Float64>),
/// optionally `stat_error` (List<Float64>).
///
/// Args:
///     ipc_bytes: bytes from `table.serialize()` (PyArrow IPC stream).
///     poi: parameter of interest name (default "mu").
///     observations: optional dict {channel_name: [obs_counts]}.
///
/// Returns:
///     HistFactoryModel
#[pyfunction]
#[pyo3(signature = (ipc_bytes, poi="mu", observations=None))]
fn from_arrow_ipc(
    ipc_bytes: &[u8],
    poi: &str,
    observations: Option<std::collections::HashMap<String, Vec<f64>>>,
) -> PyResult<PyHistFactoryModel> {
    let config = ns_translate::arrow::ingest::ArrowIngestConfig {
        poi: poi.to_string(),
        observations,
        ..Default::default()
    };

    let workspace = ns_translate::arrow::ingest::from_arrow_ipc(ipc_bytes, &config)
        .map_err(|e| PyValueError::new_err(format!("Arrow ingest failed: {e}")))?;

    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
        .map_err(|e| PyValueError::new_err(format!("Model construction failed: {e}")))?;

    Ok(PyHistFactoryModel { inner: model })
}

/// Create a HistFactoryModel from a Parquet file.
///
/// The Parquet file must contain the histogram table schema:
/// `channel` (Utf8), `sample` (Utf8), `yields` (List<Float64>),
/// optionally `stat_error` (List<Float64>).
#[pyfunction]
#[pyo3(signature = (path, poi="mu", observations=None))]
fn from_parquet(
    path: &str,
    poi: &str,
    observations: Option<std::collections::HashMap<String, Vec<f64>>>,
) -> PyResult<PyHistFactoryModel> {
    let config = ns_translate::arrow::ingest::ArrowIngestConfig {
        poi: poi.to_string(),
        observations,
        ..Default::default()
    };

    let workspace = ns_translate::arrow::parquet::from_parquet(std::path::Path::new(path), &config)
        .map_err(|e| PyValueError::new_err(format!("Parquet ingest failed: {e}")))?;

    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
        .map_err(|e| PyValueError::new_err(format!("Model construction failed: {e}")))?;

    Ok(PyHistFactoryModel { inner: model })
}

/// Create a HistFactoryModel from Parquet yields + modifiers files (binned Parquet v2).
///
/// The yields file must contain the histogram yields table schema:
/// `channel` (Utf8), `sample` (Utf8), `yields` (List<Float64>),
/// optionally `stat_error` (List<Float64>).
///
/// The modifiers file must contain the modifiers table schema:
/// `channel` (Utf8), `sample` (Utf8), `modifier_name` (Utf8), `modifier_type` (Utf8),
/// optionally `data_hi`/`data_lo`/`data` (List<Float64>).
#[pyfunction]
#[pyo3(signature = (yields_path, modifiers_path, poi="mu", observations=None))]
fn from_parquet_with_modifiers(
    yields_path: &str,
    modifiers_path: &str,
    poi: &str,
    observations: Option<std::collections::HashMap<String, Vec<f64>>>,
) -> PyResult<PyHistFactoryModel> {
    let config = ns_translate::arrow::ingest::ArrowIngestConfig {
        poi: poi.to_string(),
        observations,
        ..Default::default()
    };

    let workspace = ns_translate::arrow::parquet::from_parquet_with_modifiers(
        std::path::Path::new(yields_path),
        std::path::Path::new(modifiers_path),
        &config,
    )
    .map_err(|e| PyValueError::new_err(format!("Parquet ingest failed: {e}")))?;

    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
        .map_err(|e| PyValueError::new_err(format!("Model construction failed: {e}")))?;

    Ok(PyHistFactoryModel { inner: model })
}

/// Export model expected yields as Arrow IPC bytes.
///
/// Returns bytes that can be deserialized with:
///     `pyarrow.ipc.open_stream(bytes).read_all()`
///
/// Schema: channel (Utf8), sample (Utf8), yields (List<Float64>).
#[pyfunction]
#[pyo3(signature = (model, params=None))]
fn to_arrow_yields_ipc<'py>(
    py: Python<'py>,
    model: &PyHistFactoryModel,
    params: Option<Vec<f64>>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let p = params.unwrap_or_else(|| model.inner.parameters().iter().map(|p| p.init).collect());

    let ipc = ns_translate::arrow::export::yields_to_ipc(&model.inner, &p)
        .map_err(|e| PyValueError::new_err(format!("Arrow export failed: {e}")))?;

    Ok(pyo3::types::PyBytes::new(py, &ipc).unbind())
}

/// Export model parameters as Arrow IPC bytes.
///
/// Schema: name (Utf8), index (UInt32), value (Float64),
///         bound_lo (Float64), bound_hi (Float64), init (Float64).
#[pyfunction]
#[pyo3(signature = (model, params=None))]
fn to_arrow_params_ipc<'py>(
    py: Python<'py>,
    model: &PyHistFactoryModel,
    params: Option<Vec<f64>>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let p_ref = params.as_deref();

    let ipc = ns_translate::arrow::export::parameters_to_ipc(&model.inner, p_ref)
        .map_err(|e| PyValueError::new_err(format!("Arrow export failed: {e}")))?;

    Ok(pyo3::types::PyBytes::new(py, &ipc).unbind())
}

/// Audit a pyhf workspace JSON string for compatibility.
///
/// Returns a dict with channels, modifier types, unsupported features, etc.
#[pyfunction]
fn workspace_audit(py: Python<'_>, json_str: &str) -> PyResult<Py<PyAny>> {
    let json: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let audit = ns_translate::pyhf::audit::workspace_audit(&json);
    let audit_json = serde_json::to_value(&audit)
        .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;
    json_value_to_py(py, &audit_json)
}

fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyDict, PyFloat, PyList};
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            Ok(b.into_pyobject(py).unwrap().to_owned().into_any().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().into_any().unbind())
            } else {
                Ok(PyFloat::new(py, n.as_f64().unwrap_or(0.0)).into_any().unbind())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py).unwrap().into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Py<PyAny>> =
                arr.iter().map(|v| json_value_to_py(py, v)).collect::<PyResult<_>>()?;
            Ok(PyList::new(py, &items).unwrap().into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Apply a pyhf PatchSet (HEPData) to a base workspace JSON string.
///
/// Returns the patched workspace as a pretty JSON string.
#[pyfunction]
#[pyo3(signature = (workspace_json, patchset_json, *, patch_name=None))]
fn apply_patchset(
    workspace_json: &str,
    patchset_json: &str,
    patch_name: Option<String>,
) -> PyResult<String> {
    let base: serde_json::Value =
        serde_json::from_str(workspace_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ps: ns_translate::pyhf::PatchSet =
        serde_json::from_str(patchset_json).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let patched = ps
        .apply_to_value(&base, patch_name.as_deref())
        .map_err(|e| PyValueError::new_err(format!("apply_patchset failed: {}", e)))?;

    serde_json::to_string_pretty(&patched).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convenience wrapper: create model from HistFactory XML (combination.xml + ROOT files).
#[pyfunction]
fn from_histfactory(xml_path: &str) -> PyResult<PyHistFactoryModel> {
    PyHistFactoryModel::from_xml(xml_path)
}

/// Extract per-channel bin edges from a HistFactory export (`combination.xml` + ROOT files).
///
/// Uses each channel's `data` histogram as the canonical binning source.
#[pyfunction]
fn histfactory_bin_edges_by_channel<'py>(py: Python<'py>, xml_path: &str) -> PyResult<Py<PyAny>> {
    let path = std::path::Path::new(xml_path);
    let map = ns_translate::histfactory::bin_edges_by_channel_from_xml(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to extract bin edges: {}", e)))?;

    let d = PyDict::new(py);
    for (k, v) in map {
        d.set_item(k, v)?;
    }
    Ok(d.into_any().unbind())
}

/// Read a TH1 histogram from a ROOT file, including sumw2 and under/overflow bins.
///
/// Returns a dict with keys:
/// - name, title
/// - bin_edges, bin_content, sumw2
/// - underflow, overflow, underflow_sumw2, overflow_sumw2
#[pyfunction]
fn read_root_histogram<'py>(
    py: Python<'py>,
    root_path: &str,
    hist_path: &str,
) -> PyResult<Py<PyAny>> {
    let root_path = root_path.to_string();
    let hist_path = hist_path.to_string();

    let wf = py
        .detach(move || {
            let f = RootFile::open(&root_path)?;
            f.get_histogram_with_flows(&hist_path)
        })
        .map_err(|e| PyValueError::new_err(format!("ROOT histogram read failed: {}", e)))?;

    let d = PyDict::new(py);
    d.set_item("name", wf.histogram.name)?;
    d.set_item("title", wf.histogram.title)?;
    d.set_item("bin_edges", wf.histogram.bin_edges)?;
    d.set_item("bin_content", wf.histogram.bin_content)?;
    d.set_item("sumw2", wf.histogram.sumw2)?;
    d.set_item("underflow", wf.underflow)?;
    d.set_item("overflow", wf.overflow)?;
    d.set_item("underflow_sumw2", wf.underflow_sumw2)?;
    d.set_item("overflow_sumw2", wf.overflow_sumw2)?;
    Ok(d.into_any().unbind())
}

/// Convenience wrapper: fit model with optional overridden observations.
///
/// Pass `device="cuda"` to use GPU-accelerated NLL+gradient (requires CUDA build).
#[pyfunction]
#[pyo3(signature = (model, *, data=None, init_pars=None, device="cpu"))]
fn fit<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    data: Option<Vec<f64>>,
    init_pars: Option<Vec<f64>>,
    device: &str,
) -> PyResult<PyFitResult> {
    if device == "cuda" {
        // GPU path: requires HistFactoryModel
        let hf = model
            .extract::<PyRef<'_, PyHistFactoryModel>>()
            .map_err(|_| PyValueError::new_err("device='cuda' requires a HistFactoryModel"))?;
        let mut m = hf.inner.clone();
        if let Some(ref obs) = data {
            m = m
                .with_observed_main(obs)
                .map_err(|e| PyValueError::new_err(format!("Failed to set data: {}", e)))?;
        }
        let ip = init_pars.clone();
        let result = py
            .detach(move || -> ns_core::Result<ns_core::FitResult> {
                let mle = RustMLE::new();
                #[cfg(feature = "cuda")]
                {
                    if let Some(ip) = ip { mle.fit_gpu_from(&m, &ip) } else { mle.fit_gpu(&m) }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (mle, m, ip);
                    Err(ns_core::Error::Computation(
                        "CUDA support not compiled in. Build with --features cuda".to_string(),
                    ))
                }
            })
            .map_err(|e| PyValueError::new_err(format!("GPU fit failed: {}", e)))?;

        return Ok(PyFitResult::from(result));
    }

    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.fit(py, model, data, init_pars)
}

/// Convenience wrapper: fit a Posterior (MAP) by minimizing negative log-posterior.
#[pyfunction]
fn map_fit<'py>(py: Python<'py>, posterior: &Bound<'py, PyAny>) -> PyResult<PyFitResult> {
    let post = posterior
        .extract::<PyRef<'_, PyPosterior>>()
        .map_err(|_| PyValueError::new_err("map_fit expects a Posterior"))?;

    let mle = RustMLE::new();
    let model = post.model.clone();
    let priors = post.priors.clone();

    let result = py
        .detach(move || model.fit_map(&mle, priors))
        .map_err(|e| PyValueError::new_err(format!("MAP fit failed: {}", e)))?;

    Ok(PyFitResult::from(result))
}

/// Convenience wrapper: fit multiple models in parallel (Rayon).
///
/// Two call forms:
/// - `fit_batch(models)` — list of models, each with its own observations
/// - `fit_batch(model, datasets)` — HistFactoryModel + list of observation vectors
#[pyfunction]
#[pyo3(signature = (models_or_model, datasets=None))]
fn fit_batch<'py>(
    py: Python<'py>,
    models_or_model: &Bound<'py, PyAny>,
    datasets: Option<Vec<Vec<f64>>>,
) -> PyResult<Vec<PyFitResult>> {
    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.fit_batch(py, models_or_model, datasets)
}

/// Convenience wrapper: generate Poisson toys and fit each in parallel.
#[pyfunction]
#[pyo3(signature = (model, params, *, n_toys=1000, seed=42))]
fn fit_toys(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    params: Vec<f64>,
    n_toys: usize,
    seed: u64,
) -> PyResult<Vec<PyFitResult>> {
    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.fit_toys(py, model, params, n_toys, seed)
}

/// Generate Poisson-fluctuated toys and fit each for an unbinned model.
///
/// Uses warm-start (from `init_params` or observed-data MLE), retry with jitter,
/// and Rayon parallelism for ≥95% convergence on typical unbinned models.
#[pyfunction]
#[pyo3(signature = (model, params, *, n_toys=1000, seed=42, init_params=None, max_retries=3, max_iter=5000, compute_hessian=false))]
fn unbinned_fit_toys(
    py: Python<'_>,
    model: &PyUnbinnedModel,
    params: Vec<f64>,
    n_toys: usize,
    seed: u64,
    init_params: Option<Vec<f64>>,
    max_retries: usize,
    max_iter: u64,
    compute_hessian: bool,
) -> PyResult<Vec<PyFitResult>> {
    let m = model.inner.clone();

    let results = py.detach(move || {
        let toy_config = ns_inference::ToyFitConfig {
            optimizer: OptimizerConfig { max_iter, ..OptimizerConfig::default() },
            max_retries,
            compute_hessian,
            ..ns_inference::ToyFitConfig::default()
        };

        let batch = ns_inference::fit_unbinned_toys_batch_cpu(
            &m,
            &params,
            n_toys,
            seed,
            init_params.as_deref(),
            toy_config,
            |m, p, s| m.sample_poisson_toy(p, s),
        );
        batch.fits
    });

    results
        .into_iter()
        .map(|r| {
            let r =
                r.map_err(|e| PyValueError::new_err(format!("Unbinned toy fit failed: {}", e)))?;
            Ok(PyFitResult::from(r))
        })
        .collect()
}

/// Batch toy fitting (skips Hessian/covariance for speed).
///
/// Uses Accelerate (vDSP/vForce) for Poisson NLL when built with `--features accelerate`.
/// Returns FitResult with parameters + NLL only (uncertainties = 0).
#[pyfunction]
#[pyo3(signature = (model, params, *, n_toys=1000, seed=42))]
fn fit_toys_batch(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    params: Vec<f64>,
    n_toys: usize,
    seed: u64,
) -> PyResult<Vec<PyFitResult>> {
    let m = model.inner.clone();

    let results =
        py.detach(move || ns_inference::batch::fit_toys_batch(&m, &params, n_toys, seed, None));

    results
        .into_iter()
        .map(|r| {
            let r = r.map_err(|e| PyValueError::new_err(format!("Batch toy fit failed: {}", e)))?;
            Ok(PyFitResult::from(r))
        })
        .collect()
}

/// Set the process-wide evaluation mode.
///
/// - `"fast"` (default): maximum speed, naive summation, multi-threaded.
/// - `"parity"`: Kahan summation, Accelerate disabled, single-thread recommended.
///   Used for deterministic pyhf parity validation.
#[pyfunction]
fn set_eval_mode(mode: &str) -> PyResult<()> {
    let m = match mode {
        "fast" => ns_compute::EvalMode::Fast,
        "parity" => ns_compute::EvalMode::Parity,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown eval mode '{}'. Use 'fast' or 'parity'.",
                mode
            )));
        }
    };
    ns_compute::set_eval_mode(m);
    Ok(())
}

/// Best-effort: configure the global Rayon thread pool.
///
/// Returns `True` if the requested configuration was applied, `False` if Rayon
/// was already initialized (or threads==0, meaning "do not force").
///
/// Notes:
/// - Rayon global thread pool can only be configured once per process.
/// - This is primarily intended for deterministic / parity-mode automation.
#[pyfunction]
fn set_threads(threads: usize) -> PyResult<bool> {
    if threads == 0 {
        return Ok(false);
    }
    let r = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    Ok(r.is_ok())
}

/// Get the current evaluation mode ("fast" or "parity").
#[pyfunction]
fn get_eval_mode() -> &'static str {
    match ns_compute::eval_mode() {
        ns_compute::EvalMode::Fast => "fast",
        ns_compute::EvalMode::Parity => "parity",
    }
}

/// Check if Accelerate backend is available.
#[pyfunction]
fn has_accelerate() -> bool {
    ns_inference::batch::is_accelerate_available()
}

/// Check if CUDA GPU batch backend is available.
///
/// Returns True if:
/// - the `cuda` feature was enabled at compile time, AND
/// - a CUDA-capable GPU is present at runtime.
#[pyfunction]
fn has_cuda() -> bool {
    ns_inference::batch::is_cuda_batch_available()
}

/// Check if Metal GPU batch backend is available.
///
/// Returns True if:
/// - the `metal` feature was enabled at compile time, AND
/// - a Metal-capable GPU is present at runtime (Apple Silicon).
#[pyfunction]
fn has_metal() -> bool {
    ns_inference::batch::is_metal_batch_available()
}

/// GPU-accelerated batch toy fitting (requires CUDA).
///
/// All toys are optimized in lockstep on the GPU.
/// Falls back to CPU if CUDA is not available.
#[pyfunction]
#[pyo3(signature = (model, params, *, n_toys=1000, seed=42, device="cpu"))]
fn fit_toys_batch_gpu(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    params: Vec<f64>,
    n_toys: usize,
    seed: u64,
    device: &str,
) -> PyResult<Vec<PyFitResult>> {
    let m = model.inner.clone();

    let device_str = device.to_string();

    let results = py.detach(move || match device_str.as_str() {
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                match ns_inference::gpu_batch::fit_toys_batch_gpu(&m, &params, n_toys, seed, None) {
                    Ok(r) => r,
                    Err(e) => vec![Err(e)],
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                vec![Err(ns_core::Error::Computation(
                    "CUDA support not compiled in. Build with --features cuda".to_string(),
                ))]
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                match ns_inference::metal_batch::fit_toys_batch_metal(
                    &m, &params, n_toys, seed, None,
                ) {
                    Ok(r) => r,
                    Err(e) => vec![Err(e)],
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                vec![Err(ns_core::Error::Computation(
                    "Metal support not compiled in. Build with --features metal".to_string(),
                ))]
            }
        }
        _ => ns_inference::batch::fit_toys_batch(&m, &params, n_toys, seed, None),
    });

    results
        .into_iter()
        .map(|r| {
            let r =
                r.map_err(|e| PyValueError::new_err(format!("GPU batch toy fit failed: {}", e)))?;
            Ok(PyFitResult::from(r))
        })
        .collect()
}

/// Generate an Asimov (deterministic expected) **main** dataset for a HistFactory model.
///
/// Returns a flat vector of main-bin expectations (no auxdata), suitable for passing to
/// `nextstat.fit(model, data=...)` or `model.with_observed_main(...)`.
#[pyfunction]
fn asimov_data(model: &PyHistFactoryModel, params: Vec<f64>) -> PyResult<Vec<f64>> {
    ns_inference::toys::asimov_main(&model.inner, &params)
        .map_err(|e| PyValueError::new_err(format!("asimov_data failed: {}", e)))
}

/// Generate Poisson-fluctuated **main** toy datasets for a HistFactory model.
///
/// Returns `n_toys` flat vectors of main-bin observations (no auxdata).
#[pyfunction]
#[pyo3(signature = (model, params, *, n_toys=1000, seed=42))]
fn poisson_toys(
    model: &PyHistFactoryModel,
    params: Vec<f64>,
    n_toys: usize,
    seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    ns_inference::toys::poisson_main_toys(&model.inner, &params, n_toys, seed)
        .map_err(|e| PyValueError::new_err(format!("poisson_toys failed: {}", e)))
}

/// Convenience wrapper: nuisance-parameter ranking (impact on POI).
#[pyfunction]
fn ranking<'py>(py: Python<'py>, model: &PyHistFactoryModel) -> PyResult<Vec<Py<PyAny>>> {
    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.ranking(py, model)
}

/// Convenience wrapper: nuisance-parameter ranking (impact on POI) for `UnbinnedModel`.
#[pyfunction]
fn unbinned_ranking<'py>(py: Python<'py>, model: &PyUnbinnedModel) -> PyResult<Vec<Py<PyAny>>> {
    let poi_idx = model.inner.poi_index().ok_or_else(|| {
        PyValueError::new_err("UnbinnedModel has no POI (spec.model.poi is required for ranking)")
    })?;

    let mle = RustMLE::new();
    let m = model.inner.clone();

    let entries: Vec<RankingEntry> = py
        .detach(move || -> NsResult<Vec<RankingEntry>> {
            // Nominal fit (WITH Hessian) to get pull/constraint.
            let nominal = mle.fit(&m)?;
            let mu_hat = nominal.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);

            let base_bounds: Vec<(f64, f64)> = m.parameters().iter().map(|p| p.bounds).collect();

            let mut entries = Vec::<RankingEntry>::new();
            for (np_idx, p) in m.parameters().iter().enumerate() {
                if np_idx == poi_idx {
                    continue;
                }
                let Some(constraint) = p.constraint.clone() else { continue };
                if (p.bounds.0 - p.bounds.1).abs() < 1e-12 {
                    continue;
                }

                let (center, sigma) = match constraint {
                    ns_unbinned::Constraint::Gaussian { mean, sigma } => (mean, sigma),
                };
                if !sigma.is_finite() || sigma <= 0.0 {
                    continue;
                }

                let (b_lo, b_hi) = base_bounds[np_idx];
                let val_up = (center + sigma).min(b_hi);
                let val_down = (center - sigma).max(b_lo);

                let theta_hat = nominal.parameters[np_idx];
                let pull = (theta_hat - center) / sigma;
                let constraint = nominal.uncertainties[np_idx] / sigma;

                // --- +1σ refit ---
                let m_up = m.with_fixed_param(np_idx, val_up);
                let mut warm_up = nominal.parameters.clone();
                warm_up[np_idx] = val_up;
                let r_up = mle.fit_minimum_from(&m_up, &warm_up);
                let mu_up = r_up
                    .ok()
                    .filter(|r| r.converged)
                    .and_then(|r| r.parameters.get(poi_idx).copied());

                // --- -1σ refit ---
                let m_down = m.with_fixed_param(np_idx, val_down);
                let mut warm_down = nominal.parameters.clone();
                warm_down[np_idx] = val_down;
                let r_down = mle.fit_minimum_from(&m_down, &warm_down);
                let mu_down = r_down
                    .ok()
                    .filter(|r| r.converged)
                    .and_then(|r| r.parameters.get(poi_idx).copied());

                let (Some(mu_up), Some(mu_down)) = (mu_up, mu_down) else { continue };
                entries.push(RankingEntry {
                    name: p.name.clone(),
                    delta_mu_up: mu_up - mu_hat,
                    delta_mu_down: mu_down - mu_hat,
                    pull,
                    constraint,
                });
            }

            // Sort by |impact|
            entries.sort_by(|a, b| {
                let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
                let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
                impact_b
                    .partial_cmp(&impact_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.name.cmp(&b.name))
            });

            Ok(entries)
        })
        .map_err(|e| PyValueError::new_err(format!("Unbinned ranking failed: {}", e)))?;

    entries
        .into_iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("delta_mu_up", e.delta_mu_up)?;
            d.set_item("delta_mu_down", e.delta_mu_down)?;
            d.set_item("pull", e.pull)?;
            d.set_item("constraint", e.constraint)?;
            Ok(d.into_any().unbind())
        })
        .collect()
}

/// GPU-accelerated nuisance-parameter ranking (impact on POI).
///
/// Nominal fit uses CPU (needs Hessian for pull/constraint). Per-NP refits
/// use GPU with shared session, warm-start, and bounds-clamping.
#[cfg(feature = "cuda")]
#[pyfunction]
fn ranking_gpu<'py>(py: Python<'py>, model: &PyHistFactoryModel) -> PyResult<Vec<Py<PyAny>>> {
    let mle = RustMLE::new();
    let m = model.inner.clone();

    let entries = py
        .detach(move || ns_inference::ranking_gpu(&mle, &m))
        .map_err(|e| PyValueError::new_err(format!("GPU ranking failed: {}", e)))?;

    entries
        .into_iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("delta_mu_up", e.delta_mu_up)?;
            d.set_item("delta_mu_down", e.delta_mu_down)?;
            d.set_item("pull", e.pull)?;
            d.set_item("constraint", e.constraint)?;
            Ok(d.into_any().unbind())
        })
        .collect()
}

/// Fixed-step RK4 integration for a linear ODE system `dy/dt = A y`.
///
/// Returns a dict: {"t": [...], "y": [[...], ...]}.
#[pyfunction]
#[pyo3(signature = (a, y0, t0, t1, dt, *, max_steps=100000))]
fn rk4_linear(
    py: Python<'_>,
    a: Vec<Vec<f64>>,
    y0: Vec<f64>,
    t0: f64,
    t1: f64,
    dt: f64,
    max_steps: usize,
) -> PyResult<Py<PyAny>> {
    let a = dmatrix_from_nested("A", a)?;
    let sol = ns_inference::ode::rk4_linear(&a, &y0, t0, t1, dt, max_steps)
        .map_err(|e| PyValueError::new_err(format!("rk4_linear failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("t", sol.t)?;
    out.set_item("y", sol.y)?;
    Ok(out.into_any().unbind())
}

/// Closed-form OLS fit for linear regression fixtures and baselines.
#[pyfunction]
#[pyo3(signature = (x, y, *, include_intercept=true))]
fn ols_fit(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> PyResult<Vec<f64>> {
    rust_ols_fit(x, y, include_intercept).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Frequentist hypotest (asymptotics, qtilde) returning CLs (pyhf-compatible).
#[pyfunction]
#[pyo3(signature = (poi_test, model, *, data=None, return_tail_probs=false))]
fn hypotest(
    py: Python<'_>,
    poi_test: f64,
    model: &PyHistFactoryModel,
    data: Option<Vec<f64>>,
    return_tail_probs: bool,
) -> PyResult<Py<PyAny>> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let ctx = RustCLsCtx::new(&mle, &fit_model)
        .map_err(|e| PyValueError::new_err(format!("Failed to build asymptotic context: {}", e)))?;
    let r = ctx
        .hypotest_qtilde(&mle, poi_test)
        .map_err(|e| PyValueError::new_err(format!("Hypotest failed: {}", e)))?;

    if return_tail_probs {
        (r.cls, vec![r.clsb, r.clb]).into_py_any(py)
    } else {
        r.cls.into_py_any(py)
    }
}

/// Frequentist hypotest (toy-based CLs, qtilde) returning CLs.
#[pyfunction]
#[pyo3(signature = (poi_test, model, *, n_toys=1000, seed=42, expected_set=false, data=None, return_tail_probs=false, return_meta=false))]
fn hypotest_toys(
    py: Python<'_>,
    poi_test: f64,
    model: &PyHistFactoryModel,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
    data: Option<Vec<f64>>,
    return_tail_probs: bool,
    return_meta: bool,
) -> PyResult<Py<PyAny>> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    if expected_set {
        let r = ns_inference::hypotest_qtilde_toys_expected_set(
            &mle, &fit_model, poi_test, n_toys, seed,
        )
        .map_err(|e| PyValueError::new_err(format!("Toy-based hypotest failed: {}", e)))?;

        if return_meta {
            let d = PyDict::new(py);
            d.set_item("mu_test", r.observed.mu_test)?;
            d.set_item("cls", r.observed.cls)?;
            d.set_item("clsb", r.observed.clsb)?;
            d.set_item("clb", r.observed.clb)?;
            d.set_item("q_obs", r.observed.q_obs)?;
            d.set_item("mu_hat", r.observed.mu_hat)?;
            d.set_item("n_toys_b", r.observed.n_toys_b)?;
            d.set_item("n_toys_sb", r.observed.n_toys_sb)?;
            d.set_item("n_error_b", r.observed.n_error_b)?;
            d.set_item("n_error_sb", r.observed.n_error_sb)?;
            d.set_item("n_nonconverged_b", r.observed.n_nonconverged_b)?;
            d.set_item("n_nonconverged_sb", r.observed.n_nonconverged_sb)?;
            d.set_item("expected", r.expected.to_vec())?;
            return Ok(d.into_any().unbind());
        }

        if return_tail_probs {
            (r.observed.cls, r.expected.to_vec(), vec![r.observed.clsb, r.observed.clb])
                .into_py_any(py)
        } else {
            (r.observed.cls, r.expected.to_vec()).into_py_any(py)
        }
    } else {
        let r = ns_inference::hypotest_qtilde_toys(&mle, &fit_model, poi_test, n_toys, seed)
            .map_err(|e| PyValueError::new_err(format!("Toy-based hypotest failed: {}", e)))?;

        if return_meta {
            let d = PyDict::new(py);
            d.set_item("mu_test", r.mu_test)?;
            d.set_item("cls", r.cls)?;
            d.set_item("clsb", r.clsb)?;
            d.set_item("clb", r.clb)?;
            d.set_item("q_obs", r.q_obs)?;
            d.set_item("mu_hat", r.mu_hat)?;
            d.set_item("n_toys_b", r.n_toys_b)?;
            d.set_item("n_toys_sb", r.n_toys_sb)?;
            d.set_item("n_error_b", r.n_error_b)?;
            d.set_item("n_error_sb", r.n_error_sb)?;
            d.set_item("n_nonconverged_b", r.n_nonconverged_b)?;
            d.set_item("n_nonconverged_sb", r.n_nonconverged_sb)?;
            return Ok(d.into_any().unbind());
        }

        if return_tail_probs {
            (r.cls, vec![r.clsb, r.clb]).into_py_any(py)
        } else {
            r.cls.into_py_any(py)
        }
    }
}

/// Unbinned hypotest (toy-based CLs, qtilde) returning CLs.
#[pyfunction]
#[pyo3(signature = (poi_test, model, *, n_toys=1000, seed=42, expected_set=false, return_tail_probs=false, return_meta=false))]
fn unbinned_hypotest_toys(
    py: Python<'_>,
    poi_test: f64,
    model: &PyUnbinnedModel,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
    return_tail_probs: bool,
    return_meta: bool,
) -> PyResult<Py<PyAny>> {
    let mle = RustMLE::new();
    let fit_model = model.inner.clone();

    let sample_toy =
        |m: &RustUnbinnedModel, params: &[f64], seed: u64| m.sample_poisson_toy(params, seed);

    if expected_set {
        let r = ns_inference::hypotest_qtilde_toys_expected_set_with_sampler(
            &mle, &fit_model, poi_test, n_toys, seed, sample_toy,
        )
        .map_err(|e| PyValueError::new_err(format!("Toy-based hypotest failed: {}", e)))?;

        if return_meta {
            let d = PyDict::new(py);
            d.set_item("mu_test", r.observed.mu_test)?;
            d.set_item("cls", r.observed.cls)?;
            d.set_item("clsb", r.observed.clsb)?;
            d.set_item("clb", r.observed.clb)?;
            d.set_item("q_obs", r.observed.q_obs)?;
            d.set_item("mu_hat", r.observed.mu_hat)?;
            d.set_item("n_toys_b", r.observed.n_toys_b)?;
            d.set_item("n_toys_sb", r.observed.n_toys_sb)?;
            d.set_item("n_error_b", r.observed.n_error_b)?;
            d.set_item("n_error_sb", r.observed.n_error_sb)?;
            d.set_item("n_nonconverged_b", r.observed.n_nonconverged_b)?;
            d.set_item("n_nonconverged_sb", r.observed.n_nonconverged_sb)?;
            d.set_item("expected", r.expected.to_vec())?;
            return Ok(d.into_any().unbind());
        }

        if return_tail_probs {
            (r.observed.cls, r.expected.to_vec(), vec![r.observed.clsb, r.observed.clb])
                .into_py_any(py)
        } else {
            (r.observed.cls, r.expected.to_vec()).into_py_any(py)
        }
    } else {
        let r = ns_inference::hypotest_qtilde_toys_with_sampler(
            &mle, &fit_model, poi_test, n_toys, seed, sample_toy,
        )
        .map_err(|e| PyValueError::new_err(format!("Toy-based hypotest failed: {}", e)))?;

        if return_meta {
            let d = PyDict::new(py);
            d.set_item("mu_test", r.mu_test)?;
            d.set_item("cls", r.cls)?;
            d.set_item("clsb", r.clsb)?;
            d.set_item("clb", r.clb)?;
            d.set_item("q_obs", r.q_obs)?;
            d.set_item("mu_hat", r.mu_hat)?;
            d.set_item("n_toys_b", r.n_toys_b)?;
            d.set_item("n_toys_sb", r.n_toys_sb)?;
            d.set_item("n_error_b", r.n_error_b)?;
            d.set_item("n_error_sb", r.n_error_sb)?;
            d.set_item("n_nonconverged_b", r.n_nonconverged_b)?;
            d.set_item("n_nonconverged_sb", r.n_nonconverged_sb)?;
            return Ok(d.into_any().unbind());
        }

        if return_tail_probs {
            (r.cls, vec![r.clsb, r.clb]).into_py_any(py)
        } else {
            r.cls.into_py_any(py)
        }
    }
}

/// Unbinned profile likelihood scan over POI values (q_mu).
#[pyfunction]
#[pyo3(signature = (model, mu_values))]
fn unbinned_profile_scan(
    py: Python<'_>,
    model: &PyUnbinnedModel,
    mu_values: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let mle = RustMLE::new();
    let scan = pl::scan(&mle, &model.inner, &mu_values)
        .map_err(|e| PyValueError::new_err(format!("Profile scan failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("input_schema_version", model.schema_version())?;
    out.set_item("poi_index", scan.poi_index)?;
    out.set_item("mu_hat", scan.mu_hat)?;
    out.set_item("nll_hat", scan.nll_hat)?;

    let mut point_objs: Vec<Py<PyAny>> = Vec::with_capacity(scan.points.len());
    for p in scan.points {
        let d = PyDict::new(py);
        d.set_item("mu", p.mu)?;
        d.set_item("q_mu", p.q_mu)?;
        d.set_item("nll_mu", p.nll_mu)?;
        d.set_item("converged", p.converged)?;
        d.set_item("n_iter", p.n_iter)?;
        point_objs.push(d.into_any().unbind());
    }
    out.set_item("points", PyList::new(py, point_objs)?)?;

    Ok(out.into_any().unbind())
}

/// Unbinned `q_mu` computation (upper-limit style, one-sided).
#[pyfunction]
#[pyo3(signature = (mu_test, model))]
fn unbinned_hypotest(py: Python<'_>, mu_test: f64, model: &PyUnbinnedModel) -> PyResult<Py<PyAny>> {
    let poi_idx = model.inner.poi_index().ok_or_else(|| {
        PyValueError::new_err("UnbinnedModel has no POI (spec.model.poi is required for hypotest)")
    })?;

    let mle = RustMLE::new();
    let m = model.inner.clone();

    let (free, fixed_mu, q0, nll_mu0) = py
        .detach(move || -> NsResult<(RustOptimizationResult, RustOptimizationResult, Option<f64>, Option<f64>)> {
            let free = mle.fit_minimum(&m)?;
            let fixed_model = m.with_fixed_param(poi_idx, mu_test);
            let mut warm_mu = free.parameters.clone();
            warm_mu[poi_idx] = mu_test;
            let fixed_mu = mle.fit_minimum_from(&fixed_model, &warm_mu)?;

            let (poi_lo, poi_hi) = m.parameter_bounds()[poi_idx];
            let mut q0: Option<f64> = None;
            let mut nll_mu0: Option<f64> = None;
            if poi_lo <= 0.0 && 0.0 <= poi_hi {
                let fixed0_model = m.with_fixed_param(poi_idx, 0.0);
                let mut warm0 = free.parameters.clone();
                warm0[poi_idx] = 0.0;
                let fixed0 = mle.fit_minimum_from(&fixed0_model, &warm0)?;
                let llr0 = 2.0 * (fixed0.fval - free.fval);
                let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
                q0 = Some(if mu_hat < 0.0 { 0.0 } else { llr0.max(0.0) });
                nll_mu0 = Some(fixed0.fval);
            }

            Ok((free, fixed_mu, q0, nll_mu0))
        })
        .map_err(|e| PyValueError::new_err(format!("Unbinned hypotest failed: {}", e)))?;

    let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
    let nll_hat = free.fval;
    let nll_mu = fixed_mu.fval;

    // Upper-limit style q_mu (one-sided).
    let llr = 2.0 * (nll_mu - nll_hat);
    let mut q_mu = llr.max(0.0);
    if mu_hat > mu_test {
        q_mu = 0.0;
    }

    let out = PyDict::new(py);
    out.set_item("input_schema_version", model.schema_version())?;
    out.set_item("poi_index", poi_idx)?;
    out.set_item("mu_test", mu_test)?;
    out.set_item("mu_hat", mu_hat)?;
    out.set_item("nll_hat", nll_hat)?;
    out.set_item("nll_mu", nll_mu)?;
    out.set_item("q_mu", q_mu)?;
    out.set_item("q0", q0)?;
    out.set_item("nll_mu0", nll_mu0)?;
    out.set_item("converged_hat", free.converged)?;
    out.set_item("converged_mu", fixed_mu.converged)?;
    out.set_item("n_iter_hat", free.n_iter)?;
    out.set_item("n_iter_mu", fixed_mu.n_iter)?;

    Ok(out.into_any().unbind())
}

/// Profile likelihood scan over POI values (q_mu).
///
/// Pass `device="cuda"` to use GPU-accelerated NLL+gradient (requires CUDA build).
#[pyfunction]
#[pyo3(signature = (model, mu_values, *, data=None, device="cpu", return_params=false))]
fn profile_scan(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    mu_values: Vec<f64>,
    data: Option<Vec<f64>>,
    device: &str,
    return_params: bool,
) -> PyResult<Py<PyAny>> {
    // Profile scans are used as a parity surface against ROOT/HistFactory. In practice,
    // ROOT's Minuit stopping criteria are looser than our strict-gradient defaults; an
    // overly strict tolerance can push the fit into slightly different basins and produce
    // small but systematic q(mu) differences on large exports.
    let mle = RustMLE::with_config(OptimizerConfig {
        max_iter: 20000,
        tol: 1e-6,
        m: 20,
        smooth_bounds: false,
    });
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    if return_params && device == "cuda" {
        return Err(PyValueError::new_err(
            "return_params is not supported for device='cuda' yet (cpu-only debug surface)",
        ));
    }

    let scan = if device == "cuda" {
        #[cfg(feature = "cuda")]
        {
            pl::scan_gpu(&mle, &fit_model, &mu_values)
                .map_err(|e| PyValueError::new_err(format!("GPU profile scan failed: {}", e)))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(PyValueError::new_err(
                "CUDA support not compiled in. Build with --features cuda",
            ));
        }
    } else if return_params {
        pl::scan_histfactory_diag(&mle, &fit_model, &mu_values)
            .map_err(|e| PyValueError::new_err(format!("Profile scan failed: {}", e)))?
    } else {
        pl::scan_histfactory(&mle, &fit_model, &mu_values)
            .map_err(|e| PyValueError::new_err(format!("Profile scan failed: {}", e)))?
    };

    let out = PyDict::new(py);
    out.set_item("poi_index", scan.poi_index)?;
    out.set_item("mu_hat", scan.mu_hat)?;
    out.set_item("nll_hat", scan.nll_hat)?;

    let mut point_objs: Vec<Py<PyAny>> = Vec::with_capacity(scan.points.len());
    for p in scan.points {
        let d = PyDict::new(py);
        d.set_item("mu", p.mu)?;
        d.set_item("q_mu", p.q_mu)?;
        d.set_item("nll_mu", p.nll_mu)?;
        d.set_item("converged", p.converged)?;
        d.set_item("n_iter", p.n_iter)?;
        if return_params && let Some(diag) = p.diag {
            d.set_item("params", diag.parameters)?;
            d.set_item("message", diag.message)?;
            d.set_item("n_fev", diag.n_fev)?;
            d.set_item("n_gev", diag.n_gev)?;
            d.set_item("initial_cost", diag.initial_cost)?;
            d.set_item("grad_l2", diag.grad_l2)?;
        }
        point_objs.push(d.into_any().unbind());
    }
    let points = PyList::new(py, point_objs)?;
    out.set_item("points", points)?;

    Ok(out.into_any().unbind())
}

/// Observed upper limit via asymptotic CLs (qtilde) and bisection.
#[pyfunction]
#[pyo3(signature = (model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None))]
fn upper_limit(
    model: &PyHistFactoryModel,
    alpha: f64,
    lo: f64,
    hi: Option<f64>,
    rtol: f64,
    max_iter: usize,
    data: Option<Vec<f64>>,
) -> PyResult<f64> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let ctx = RustCLsCtx::new(&mle, &fit_model)
        .map_err(|e| PyValueError::new_err(format!("Failed to build asymptotic context: {}", e)))?;

    let hi0 = hi.unwrap_or_else(|| {
        fit_model
            .poi_index()
            .and_then(|idx| fit_model.parameters().get(idx).map(|p| p.bounds.1))
            .unwrap_or(10.0)
    });

    ctx.upper_limit_qtilde(&mle, alpha, lo, hi0, rtol, max_iter)
        .map_err(|e| PyValueError::new_err(format!("Upper limit failed: {}", e)))
}

/// Observed and expected upper limits via linear scan + interpolation (pyhf-compatible).
#[pyfunction]
#[pyo3(signature = (model, scan, *, alpha=0.05, data=None))]
fn upper_limits(
    model: &PyHistFactoryModel,
    scan: Vec<f64>,
    alpha: f64,
    data: Option<Vec<f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let ctx = RustCLsCtx::new(&mle, &fit_model)
        .map_err(|e| PyValueError::new_err(format!("Failed to build asymptotic context: {}", e)))?;
    let (obs, exp) = ctx
        .upper_limits_qtilde_linear_scan(&mle, alpha, &scan)
        .map_err(|e| PyValueError::new_err(format!("Upper limits failed: {}", e)))?;

    Ok((obs, exp.to_vec()))
}

/// Observed and expected upper limits via bisection root-finding (pyhf scan=None analogue).
#[pyfunction]
#[pyo3(signature = (model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None))]
fn upper_limits_root(
    model: &PyHistFactoryModel,
    alpha: f64,
    lo: f64,
    hi: Option<f64>,
    rtol: f64,
    max_iter: usize,
    data: Option<Vec<f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let ctx = RustCLsCtx::new(&mle, &fit_model)
        .map_err(|e| PyValueError::new_err(format!("Failed to build asymptotic context: {}", e)))?;

    let hi0 = hi.unwrap_or_else(|| {
        fit_model
            .poi_index()
            .and_then(|idx| fit_model.parameters().get(idx).map(|p| p.bounds.1))
            .unwrap_or(10.0)
    });

    let (obs, exp) = ctx
        .upper_limits_qtilde_bisection(&mle, alpha, lo, hi0, rtol, max_iter)
        .map_err(|e| PyValueError::new_err(format!("Upper limits root failed: {}", e)))?;

    Ok((obs, exp.to_vec()))
}

/// Bayesian NUTS/HMC sampling with ArviZ-compatible output.
#[pyfunction]
#[pyo3(signature = (model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, target_accept=0.8, init_strategy="random", metric="diagonal", init_jitter=0.0, init_jitter_rel=None, init_overdispersed_rel=None, stepsize_jitter=0.0, data=None))]
fn sample<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    max_treedepth: usize,
    target_accept: f64,
    init_strategy: &str,
    metric: &str,
    init_jitter: f64,
    init_jitter_rel: Option<f64>,
    init_overdispersed_rel: Option<f64>,
    stepsize_jitter: f64,
    data: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    validate_nuts_config(
        n_chains,
        n_warmup,
        n_samples,
        max_treedepth,
        target_accept,
        init_jitter,
        init_jitter_rel,
        init_overdispersed_rel,
    )?;

    let init_strategy = match init_strategy {
        "random" => InitStrategy::Random,
        "mle" => InitStrategy::Mle,
        other => {
            return Err(PyValueError::new_err(format!(
                "init must be 'random' or 'mle', got '{other}'"
            )));
        }
    };

    let metric_type = match metric {
        "diagonal" | "diag" | "diag_e" => ns_inference::MetricType::Diagonal,
        "dense" | "dense_e" => ns_inference::MetricType::Dense,
        "auto" => ns_inference::MetricType::Auto,
        other => {
            return Err(PyValueError::new_err(format!(
                "metric must be 'diagonal', 'dense', or 'auto', got '{other}'"
            )));
        }
    };

    let config = NutsConfig {
        max_treedepth,
        target_accept,
        init_strategy,
        metric_type,
        init_jitter,
        init_jitter_rel,
        init_overdispersed_rel,
        stepsize_jitter,
    };

    // Accept `Posterior` objects (MAP sampling), otherwise sample the model (ML posterior).
    let result = if let Ok(post) = model.extract::<PyRef<'_, PyPosterior>>() {
        if data.is_some() {
            return Err(PyValueError::new_err(
                "data= is not supported when sampling a Posterior; build the Posterior from the desired model/data",
            ));
        }
        let priors = post.priors.clone();
        let m = post.model.clone();
        py.detach(move || {
            m.sample_nuts_multichain_map(n_chains, n_warmup, n_samples, seed, config, priors)
        })
        .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
    } else {
        let sample_model = extract_posterior_model_with_data(model, data)?;
        py.detach(move || {
            sample_model.sample_nuts_multichain(n_chains, n_warmup, n_samples, seed, config)
        })
        .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
    };

    sampler_result_to_py(py, &result, n_chains, n_warmup, n_samples)
}

/// Plot-friendly CLs curve + Brazil band artifact over a scan grid.
#[pyfunction]
#[pyo3(signature = (model, scan, *, alpha=0.05, data=None))]
fn cls_curve(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    scan: Vec<f64>,
    alpha: f64,
    data: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    if scan.len() < 2 {
        return Err(PyValueError::new_err("scan must have at least 2 points"));
    }

    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let ctx = RustCLsCtx::new(&mle, &fit_model)
        .map_err(|e| PyValueError::new_err(format!("Failed to build asymptotic context: {}", e)))?;
    let art = ClsCurveArtifact::from_scan(&ctx, &mle, alpha, &scan)
        .map_err(|e| PyValueError::new_err(format!("CLs curve failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("alpha", art.alpha)?;
    out.set_item("nsigma_order", art.nsigma_order.to_vec())?;
    out.set_item("obs_limit", art.obs_limit)?;
    out.set_item("exp_limits", art.exp_limits.to_vec())?;
    out.set_item("mu_values", art.mu_values.clone())?;
    out.set_item("cls_obs", art.cls_obs.clone())?;
    out.set_item("cls_exp", art.cls_exp.to_vec())?;

    let points: Vec<Py<PyAny>> = art
        .points
        .iter()
        .map(|p| -> PyResult<Py<PyAny>> {
            let d = PyDict::new(py);
            d.set_item("mu", p.mu)?;
            d.set_item("cls", p.cls)?;
            d.set_item("expected", p.expected.to_vec())?;
            Ok(d.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("points", PyList::new(py, points)?)?;

    Ok(out.into_any().unbind())
}

/// Plot-friendly profile likelihood scan artifact.
#[pyfunction]
#[pyo3(signature = (model, mu_values, *, data=None))]
fn profile_curve(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    mu_values: Vec<f64>,
    data: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let scan = pl::scan_histfactory(&mle, &fit_model, &mu_values)
        .map_err(|e| PyValueError::new_err(format!("Profile scan failed: {}", e)))?;
    let art: ProfileCurveArtifact = scan.into();

    let out = PyDict::new(py);
    out.set_item("poi_index", art.poi_index)?;
    out.set_item("mu_hat", art.mu_hat)?;
    out.set_item("nll_hat", art.nll_hat)?;
    out.set_item("mu_values", art.mu_values.clone())?;
    out.set_item("q_mu_values", art.q_mu_values.clone())?;
    out.set_item("twice_delta_nll", art.twice_delta_nll.clone())?;

    let points: Vec<Py<PyAny>> = art
        .points
        .iter()
        .map(|p| -> PyResult<Py<PyAny>> {
            let d = PyDict::new(py);
            d.set_item("mu", p.mu)?;
            d.set_item("q_mu", p.q_mu)?;
            d.set_item("nll_mu", p.nll_mu)?;
            d.set_item("converged", p.converged)?;
            d.set_item("n_iter", p.n_iter)?;
            Ok(d.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("points", PyList::new(py, points)?)?;

    Ok(out.into_any().unbind())
}

// ---------------------------------------------------------------------------
// DifferentiableSession — PyTorch zero-copy differentiable NLL (CUDA only)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[pyclass(name = "DifferentiableSession")]
struct PyDiffSession {
    inner: ns_inference::DifferentiableSession,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyDiffSession {
    /// Create a differentiable GPU session.
    ///
    /// Args:
    ///     model: HistFactoryModel
    ///     signal_sample_name: name of the signal sample in the workspace
    #[new]
    fn new(model: &PyHistFactoryModel, signal_sample_name: &str) -> PyResult<Self> {
        let inner = ns_inference::DifferentiableSession::new(&model.inner, signal_sample_name)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Compute NLL and write signal gradient into PyTorch grad tensor (zero-copy).
    ///
    /// Args:
    ///     params: list[float] — nuisance parameter values
    ///     signal_ptr: int — from torch.Tensor.data_ptr() (CUDA device pointer)
    ///     grad_signal_ptr: int — from torch.Tensor.data_ptr() (CUDA device pointer, pre-zeroed)
    ///
    /// Returns:
    ///     float — NLL value
    fn nll_grad_signal(
        &mut self,
        params: Vec<f64>,
        signal_ptr: u64,
        grad_signal_ptr: u64,
    ) -> PyResult<f64> {
        self.inner
            .nll_grad_signal(&params, signal_ptr, grad_signal_ptr)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Number of signal bins.
    fn signal_n_bins(&self) -> usize {
        self.inner.signal_n_bins()
    }

    /// Number of model parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Default initial parameter values.
    fn parameter_init(&self) -> Vec<f64> {
        self.inner.parameter_init().to_vec()
    }
}

// ---------------------------------------------------------------------------
// ProfiledDifferentiableSession — profiled q₀/qμ with envelope gradient (CUDA)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[pyclass(name = "ProfiledDifferentiableSession")]
struct PyProfiledDiffSession {
    inner: ns_inference::ProfiledDifferentiableSession,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyProfiledDiffSession {
    /// Create a profiled differentiable GPU session.
    ///
    /// Args:
    ///     model: HistFactoryModel
    ///     signal_sample_name: name of the signal sample in the workspace
    #[new]
    fn new(model: &PyHistFactoryModel, signal_sample_name: &str) -> PyResult<Self> {
        let inner =
            ns_inference::ProfiledDifferentiableSession::new(&model.inner, signal_sample_name)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Compute profiled q₀ and its gradient w.r.t. signal bins.
    ///
    /// Args:
    ///     signal_ptr: int — from torch.Tensor.data_ptr() (CUDA device pointer)
    ///
    /// Returns:
    ///     tuple[float, list[float]] — (q0, grad_signal)
    fn profiled_q0_and_grad(&mut self, signal_ptr: u64) -> PyResult<(f64, Vec<f64>)> {
        self.inner
            .profiled_q0_and_grad(signal_ptr)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Compute profiled qμ and its gradient w.r.t. signal bins.
    ///
    /// Args:
    ///     mu_test: float — signal strength hypothesis to test
    ///     signal_ptr: int — from torch.Tensor.data_ptr() (CUDA device pointer)
    ///
    /// Returns:
    ///     tuple[float, list[float]] — (qmu, grad_signal)
    fn profiled_qmu_and_grad(
        &mut self,
        mu_test: f64,
        signal_ptr: u64,
    ) -> PyResult<(f64, Vec<f64>)> {
        self.inner
            .profiled_qmu_and_grad(mu_test, signal_ptr)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Number of signal bins.
    fn signal_n_bins(&self) -> usize {
        self.inner.signal_n_bins()
    }

    /// Number of model parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Default initial parameter values.
    fn parameter_init(&self) -> Vec<f64> {
        self.inner.parameter_init().to_vec()
    }

    /// Compute profiled qμ for multiple mu_test values (sequential, GPU session reuse).
    ///
    /// Args:
    ///     signal_ptr: int — from torch.Tensor.data_ptr() (CUDA device pointer)
    ///     mu_values: list[float] — signal strength hypotheses
    ///
    /// Returns:
    ///     list[tuple[float, list[float]]] — [(qmu, grad_signal), ...]
    fn batch_profiled_qmu(
        &mut self,
        signal_ptr: u64,
        mu_values: Vec<f64>,
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        self.inner
            .batch_profiled_qmu(signal_ptr, &mu_values)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }
}

// ---------------------------------------------------------------------------
// GpuFlowSession — GPU-accelerated flow PDF NLL reduction (CUDA only)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[pyclass(name = "GpuFlowSession")]
struct PyGpuFlowSession {
    inner: ns_inference::gpu_flow_session::GpuFlowSession,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyGpuFlowSession {
    /// Create a GPU flow session for unbinned NLL reduction.
    ///
    /// Args:
    ///     n_events: int — number of events
    ///     n_params: int — number of global parameters
    ///     processes: list[dict] — process descriptors, each with keys:
    ///         - process_index: int
    ///         - base_yield: float
    ///         - yield_param_idx: int | None
    ///         - yield_is_scaled: bool (default False)
    ///     gauss_constraints: list[dict] — Gaussian constraint entries, each with keys:
    ///         - center: float
    ///         - sigma: float (> 0)
    ///         - param_idx: int
    ///     constraint_const: float (default 0.0)
    #[new]
    #[pyo3(signature = (n_events, n_params, processes, gauss_constraints=None, constraint_const=0.0))]
    fn new(
        n_events: usize,
        n_params: usize,
        processes: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        gauss_constraints: Option<Vec<pyo3::Bound<'_, pyo3::types::PyDict>>>,
        constraint_const: f64,
    ) -> PyResult<Self> {
        use ns_compute::unbinned_types::GpuUnbinnedGaussConstraintEntry;
        use ns_inference::gpu_flow_session::{FlowProcessDesc, GpuFlowSessionConfig};

        let procs: Vec<FlowProcessDesc> = processes
            .iter()
            .map(|d| {
                let process_index: usize = d.get_item("process_index")?.unwrap().extract()?;
                let base_yield: f64 = d.get_item("base_yield")?.unwrap().extract()?;
                let yield_param_idx: Option<usize> =
                    d.get_item("yield_param_idx")?.and_then(|v| v.extract().ok());
                let yield_is_scaled: bool = d
                    .get_item("yield_is_scaled")
                    .ok()
                    .and_then(|v| v)
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(false);
                let context_param_indices: Vec<usize> = d
                    .get_item("context_param_indices")
                    .ok()
                    .and_then(|v| v)
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                Ok(FlowProcessDesc {
                    process_index,
                    base_yield,
                    yield_param_idx,
                    yield_is_scaled,
                    context_param_indices,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let gauss: Vec<GpuUnbinnedGaussConstraintEntry> = gauss_constraints
            .unwrap_or_default()
            .iter()
            .map(|d| {
                let center: f64 = d.get_item("center")?.unwrap().extract()?;
                let sigma: f64 = d.get_item("sigma")?.unwrap().extract()?;
                let param_idx: u32 = d.get_item("param_idx")?.unwrap().extract()?;
                Ok(GpuUnbinnedGaussConstraintEntry {
                    center,
                    inv_width: 1.0 / sigma,
                    param_idx,
                    _pad: 0,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        let n_context = procs.first().map(|p| p.context_param_indices.len()).unwrap_or(0);
        let config = GpuFlowSessionConfig {
            processes: procs,
            n_events,
            n_params,
            n_context,
            gauss_constraints: gauss,
            constraint_const,
        };

        let inner = ns_inference::gpu_flow_session::GpuFlowSession::new(config)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Compute NLL from host-side f64 log-prob values.
    ///
    /// Args:
    ///     logp_flat: list[float] — [n_procs × n_events] row-major
    ///     params: list[float] — [n_params] current parameter values
    ///
    /// Returns:
    ///     float — NLL value
    fn nll(&mut self, logp_flat: Vec<f64>, params: Vec<f64>) -> PyResult<f64> {
        self.inner.nll(&logp_flat, &params).map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Compute NLL from a GPU-resident float (f32) log-prob buffer (zero-copy CUDA EP path).
    ///
    /// This is the zero-copy path for ONNX Runtime CUDA Execution Provider:
    /// the log_prob output tensor stays on device, no D2H copy, no f32→f64 conversion.
    ///
    /// Args:
    ///     d_logp_flat_ptr: int — raw CUDA device pointer (from ort I/O binding or
    ///         torch.Tensor.data_ptr()) to float[n_procs × n_events]
    ///     params: list[float] — [n_params] current parameter values
    ///
    /// Returns:
    ///     float — NLL value
    fn nll_device_ptr_f32(&mut self, d_logp_flat_ptr: u64, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll_device_ptr_f32(d_logp_flat_ptr, &params)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Compute yields from parameters.
    ///
    /// Args:
    ///     params: list[float] — [n_params]
    ///
    /// Returns:
    ///     list[float] — [n_procs] computed yields
    fn compute_yields(&self, params: Vec<f64>) -> Vec<f64> {
        self.inner.compute_yields(&params)
    }

    /// Number of events.
    fn n_events(&self) -> usize {
        self.inner.n_events()
    }

    /// Number of processes.
    fn n_procs(&self) -> usize {
        self.inner.n_procs()
    }

    /// Number of parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }
}

// ---------------------------------------------------------------------------
// MetalProfiledDifferentiableSession — profiled q₀/qμ on Apple Silicon (Metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
#[pyclass(name = "MetalProfiledDifferentiableSession")]
struct PyMetalProfiledDiffSession {
    inner: ns_inference::MetalProfiledDifferentiableSession,
}

#[cfg(feature = "metal")]
#[pymethods]
impl PyMetalProfiledDiffSession {
    /// Create a Metal profiled differentiable GPU session.
    ///
    /// Args:
    ///     model: HistFactoryModel
    ///     signal_sample_name: name of the signal sample in the workspace
    #[new]
    fn new(model: &PyHistFactoryModel, signal_sample_name: &str) -> PyResult<Self> {
        let inner =
            ns_inference::MetalProfiledDifferentiableSession::new(&model.inner, signal_sample_name)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Upload signal histogram from CPU (list or numpy array of f64).
    fn upload_signal(&mut self, signal: Vec<f64>) -> PyResult<()> {
        self.inner.upload_signal(&signal).map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Compute profiled q₀ and its gradient w.r.t. signal bins.
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    ///
    /// Returns:
    ///     tuple[float, list[float]] — (q0, grad_signal)
    fn profiled_q0_and_grad(&mut self) -> PyResult<(f64, Vec<f64>)> {
        self.inner.profiled_q0_and_grad().map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Compute profiled qμ and its gradient w.r.t. signal bins.
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    ///
    /// Args:
    ///     mu_test: float — signal strength hypothesis to test
    ///
    /// Returns:
    ///     tuple[float, list[float]] — (qmu, grad_signal)
    fn profiled_qmu_and_grad(&mut self, mu_test: f64) -> PyResult<(f64, Vec<f64>)> {
        self.inner.profiled_qmu_and_grad(mu_test).map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Number of signal bins.
    fn signal_n_bins(&self) -> usize {
        self.inner.signal_n_bins()
    }

    /// Number of model parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Default initial parameter values.
    fn parameter_init(&self) -> Vec<f64> {
        self.inner.parameter_init().to_vec()
    }

    /// Compute profiled qμ for multiple mu_test values.
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    ///
    /// Args:
    ///     mu_values: list[float] — signal strength hypotheses
    ///
    /// Returns:
    ///     list[tuple[float, list[float]]] — [(qmu, grad_signal), ...]
    fn batch_profiled_qmu(&mut self, mu_values: Vec<f64>) -> PyResult<Vec<(f64, Vec<f64>)>> {
        self.inner.batch_profiled_qmu(&mu_values).map_err(|e| PyValueError::new_err(format!("{e}")))
    }
}

/// Python submodule: nextstat._core
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ns_core::VERSION)?;

    // Convenience functions (pyhf-style API).
    m.add_function(wrap_pyfunction!(from_pyhf, m)?)?;
    m.add_function(wrap_pyfunction!(workspace_audit, m)?)?;
    m.add_function(wrap_pyfunction!(apply_patchset, m)?)?;
    m.add_function(wrap_pyfunction!(from_histfactory, m)?)?;
    m.add_function(wrap_pyfunction!(histfactory_bin_edges_by_channel, m)?)?;
    m.add_function(wrap_pyfunction!(read_root_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(map_fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fit_toys, m)?)?;
    m.add_function(wrap_pyfunction!(unbinned_fit_toys, m)?)?;
    m.add_function(wrap_pyfunction!(fit_toys_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_eval_mode, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_eval_mode, m)?)?;
    m.add_function(wrap_pyfunction!(has_accelerate, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(has_metal, m)?)?;
    m.add_function(wrap_pyfunction!(fit_toys_batch_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(asimov_data, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_toys, m)?)?;
    m.add_function(wrap_pyfunction!(ranking, m)?)?;
    m.add_function(wrap_pyfunction!(unbinned_ranking, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(ranking_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_linear, m)?)?;
    m.add_function(wrap_pyfunction!(ols_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest_toys, m)?)?;
    m.add_function(wrap_pyfunction!(unbinned_hypotest, m)?)?;
    m.add_function(wrap_pyfunction!(unbinned_hypotest_toys, m)?)?;
    m.add_function(wrap_pyfunction!(profile_scan, m)?)?;
    m.add_function(wrap_pyfunction!(unbinned_profile_scan, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limit, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limits, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limits_root, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(cls_curve, m)?)?;
    m.add_function(wrap_pyfunction!(profile_curve, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_em, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(garch11_fit, m)?)?;
    m.add_function(wrap_pyfunction!(sv_logchi2_fit, m)?)?;

    // Econometrics & Causal Inference
    m.add_function(wrap_pyfunction!(panel_fe, m)?)?;
    m.add_function(wrap_pyfunction!(did, m)?)?;
    m.add_function(wrap_pyfunction!(event_study, m)?)?;
    m.add_function(wrap_pyfunction!(iv_2sls, m)?)?;
    m.add_function(wrap_pyfunction!(aipw_ate, m)?)?;
    m.add_function(wrap_pyfunction!(rosenbaum_bounds, m)?)?;

    // Survival: Kaplan-Meier + Log-rank
    m.add_function(wrap_pyfunction!(kaplan_meier, m)?)?;
    m.add_function(wrap_pyfunction!(log_rank_test, m)?)?;

    // Churn / Subscription vertical
    m.add_function(wrap_pyfunction!(churn_generate_data, m)?)?;
    m.add_function(wrap_pyfunction!(churn_retention, m)?)?;
    m.add_function(wrap_pyfunction!(churn_risk_model, m)?)?;
    m.add_function(wrap_pyfunction!(churn_uplift, m)?)?;
    m.add_function(wrap_pyfunction!(churn_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(churn_cohort_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(churn_compare, m)?)?;
    m.add_function(wrap_pyfunction!(churn_uplift_survival, m)?)?;
    m.add_function(wrap_pyfunction!(churn_bootstrap_hr, m)?)?;
    m.add_function(wrap_pyfunction!(churn_ingest, m)?)?;

    // Arrow / Parquet
    m.add_function(wrap_pyfunction!(from_arrow_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(from_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(from_parquet_with_modifiers, m)?)?;
    m.add_function(wrap_pyfunction!(to_arrow_yields_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(to_arrow_params_ipc, m)?)?;

    // Add classes
    m.add_class::<PyHistFactoryModel>()?;
    m.add_class::<PyUnbinnedModel>()?;
    m.add_class::<PyHybridModel>()?;
    m.add_class::<PyPosterior>()?;
    m.add_class::<PyKalmanModel>()?;
    m.add_class::<PyGaussianMeanModel>()?;
    m.add_class::<PyFunnelModel>()?;
    m.add_class::<PyStdNormalModel>()?;
    m.add_class::<PyLinearRegressionModel>()?;
    m.add_class::<PyLogisticRegressionModel>()?;
    m.add_class::<PyOrderedLogitModel>()?;
    m.add_class::<PyOrderedProbitModel>()?;
    m.add_class::<PyPoissonRegressionModel>()?;
    m.add_class::<PyNegativeBinomialRegressionModel>()?;
    m.add_class::<PyGammaRegressionModel>()?;
    m.add_class::<PyTweedieRegressionModel>()?;
    m.add_class::<PyGevModel>()?;
    m.add_class::<PyGpdModel>()?;
    m.add_class::<PyEightSchoolsModel>()?;
    m.add_wrapped(wrap_pyfunction!(meta_fixed))?;
    m.add_wrapped(wrap_pyfunction!(meta_random))?;
    m.add_wrapped(wrap_pyfunction!(chain_ladder))?;
    m.add_wrapped(wrap_pyfunction!(mack_chain_ladder))?;
    m.add_class::<PyComposedGlmModel>()?;
    m.add_class::<PyLmmMarginalModel>()?;
    m.add_class::<PyExponentialSurvivalModel>()?;
    m.add_class::<PyWeibullSurvivalModel>()?;
    m.add_class::<PyLogNormalAftModel>()?;
    m.add_class::<PyCoxPhModel>()?;
    m.add_class::<PyOneCompartmentOralPkModel>()?;
    m.add_class::<PyOneCompartmentOralPkNlmeModel>()?;
    m.add_class::<PyMaximumLikelihoodEstimator>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyFitMinimumResult>()?;

    // Neural PDFs (FlowPdf + DcrSurrogate, neural only)
    #[cfg(feature = "neural")]
    m.add_class::<PyFlowPdf>()?;
    #[cfg(feature = "neural")]
    m.add_class::<PyDcrSurrogate>()?;

    // DifferentiableSession (CUDA only)
    #[cfg(feature = "cuda")]
    m.add_class::<PyDiffSession>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<PyProfiledDiffSession>()?;
    // GpuFlowSession (CUDA only)
    #[cfg(feature = "cuda")]
    m.add_class::<PyGpuFlowSession>()?;
    // MetalProfiledDifferentiableSession (Metal only)
    #[cfg(feature = "metal")]
    m.add_class::<PyMetalProfiledDiffSession>()?;

    // Back-compat aliases used in plans/docs.
    let model_cls = m.getattr("HistFactoryModel")?;
    m.add("PyModel", model_cls)?;
    let fit_cls = m.getattr("FitResult")?;
    m.add("PyFitResult", fit_cls)?;
    let fit_min_cls = m.getattr("FitMinimumResult")?;
    m.add("PyFitMinimumResult", fit_min_cls)?;

    Ok(())
}
