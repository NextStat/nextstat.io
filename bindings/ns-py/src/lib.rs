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
use ns_core::traits::{LogDensityModel, PreparedNll};
use ns_core::{Error as NsError, Result as NsResult};
use ns_inference::OptimizerConfig;
use ns_inference::chain::{Chain as RustChain, SamplerResult as RustSamplerResult};
use ns_inference::diagnostics::{QualityGates, compute_diagnostics, quality_summary};
use ns_inference::lmm::{
    LmmMarginalModel as RustLmmMarginalModel, RandomEffects as RustLmmRandomEffects,
};
use ns_inference::mle::{MaximumLikelihoodEstimator as RustMLE, RankingEntry};
use ns_inference::nuts::{NutsConfig, sample_nuts};
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
use ns_inference::transforms::ParameterTransform;
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
    let mut chains: Vec<RustChain> = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let chain = sample_nuts(model, n_warmup, n_samples, seed, config.clone())?;
        chains.push(chain);
    }
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
    sample_stats.set_item("diverging", diverging)?;
    sample_stats.set_item("tree_depth", tree_depth)?;
    sample_stats.set_item("accept_prob", accept_prob)?;
    sample_stats.set_item("energy", energy)?;
    sample_stats.set_item("step_size", step_sizes)?;

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

#[derive(Debug, Clone)]
enum PosteriorModel {
    HistFactory(RustModel),
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
}

impl PosteriorModel {
    fn dim(&self) -> usize {
        match self {
            PosteriorModel::HistFactory(m) => m.dim(),
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
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_names(),
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
        }
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_bounds(),
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
        }
    }

    fn parameter_init(&self) -> Vec<f64> {
        match self {
            PosteriorModel::HistFactory(m) => m.parameter_init(),
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
        }
    }

    fn nll(&self, params: &[f64]) -> NsResult<f64> {
        match self {
            PosteriorModel::HistFactory(m) => m.nll(params),
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
        }
    }

    fn grad_nll(&self, params: &[f64]) -> NsResult<Vec<f64>> {
        match self {
            PosteriorModel::HistFactory(m) => m.grad_nll(params),
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
        }
    }

    fn fit_mle(&self, mle: &RustMLE) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => mle.fit(m),
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
        }
    }

    fn fit_mle_from(&self, mle: &RustMLE, init_pars: &[f64]) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => mle.fit_from(m, init_pars),
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
        }
    }

    fn fit_map(&self, mle: &RustMLE, priors: Vec<Prior>) -> NsResult<ns_core::FitResult> {
        match self {
            PosteriorModel::HistFactory(m) => {
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
        }
    }
}

fn extract_posterior_model(model: &Bound<'_, PyAny>) -> PyResult<PosteriorModel> {
    if let Ok(hf) = model.extract::<PyRef<'_, PyHistFactoryModel>>() {
        Ok(PosteriorModel::HistFactory(hf.inner.clone()))
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
    } else {
        Err(PyValueError::new_err(
            "Unsupported model type. Expected HistFactoryModel, GaussianMeanModel, FunnelModel, StdNormalModel, a regression model, OrderedLogitModel, OrderedProbitModel, ComposedGlmModel, LmmMarginalModel, a survival model, or a PK model.",
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
    /// Create model from workspace JSON string
    #[staticmethod]
    fn from_workspace(json_str: &str) -> PyResult<Self> {
        let workspace: RustWorkspace = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse workspace: {}", e)))?;

        let model = RustModel::from_workspace(&workspace)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

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

/// Python wrapper for MaximumLikelihoodEstimator
#[pyclass(name = "MaximumLikelihoodEstimator")]
struct PyMaximumLikelihoodEstimator {
    inner: RustMLE,
}

#[pymethods]
impl PyMaximumLikelihoodEstimator {
    #[new]
    #[pyo3(signature = (*, max_iter=1000, tol=1e-6, m=10))]
    fn new(max_iter: u64, tol: f64, m: usize) -> PyResult<Self> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(PyValueError::new_err("tol must be finite and >= 0"));
        }
        if m == 0 {
            return Err(PyValueError::new_err("m must be >= 1"));
        }
        let cfg = OptimizerConfig { max_iter, tol, m };
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

/// Audit a pyhf workspace JSON string for compatibility.
///
/// Returns a dict with channels, modifier types, unsupported features, etc.
#[pyfunction]
fn workspace_audit(py: Python<'_>, json_str: &str) -> PyResult<Py<PyAny>> {
    let json: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let audit = ns_translate::pyhf::audit::workspace_audit(&json);
    let audit_json = serde_json::to_value(&audit)
        .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;
    json_value_to_py(py, &audit_json)
}

fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyDict, PyFloat, PyList};
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py).unwrap().to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().into_any().unbind())
            } else {
                Ok(PyFloat::new(py, n.as_f64().unwrap_or(0.0)).into_any().unbind())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py).unwrap().into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Py<PyAny>> = arr.iter().map(|v| json_value_to_py(py, v)).collect::<PyResult<_>>()?;
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
                    if let Some(ip) = ip {
                        mle.fit_gpu_from(&m, &ip)
                    } else {
                        mle.fit_gpu(&m)
                    }
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

/// Profile likelihood scan over POI values (q_mu).
///
/// Pass `device="cuda"` to use GPU-accelerated NLL+gradient (requires CUDA build).
#[pyfunction]
#[pyo3(signature = (model, mu_values, *, data=None, device="cpu"))]
fn profile_scan(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    mu_values: Vec<f64>,
    data: Option<Vec<f64>>,
    device: &str,
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
#[pyo3(signature = (model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, target_accept=0.8, init_jitter=0.0, init_jitter_rel=None, init_overdispersed_rel=None, data=None))]
fn sample<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    max_treedepth: usize,
    target_accept: f64,
    init_jitter: f64,
    init_jitter_rel: Option<f64>,
    init_overdispersed_rel: Option<f64>,
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

    let config = NutsConfig {
        max_treedepth,
        target_accept,
        init_jitter,
        init_jitter_rel,
        init_overdispersed_rel,
        ..Default::default()
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
    m.add_function(wrap_pyfunction!(fit_toys_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_eval_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_eval_mode, m)?)?;
    m.add_function(wrap_pyfunction!(has_accelerate, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(has_metal, m)?)?;
    m.add_function(wrap_pyfunction!(fit_toys_batch_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(asimov_data, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_toys, m)?)?;
    m.add_function(wrap_pyfunction!(ranking, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_linear, m)?)?;
    m.add_function(wrap_pyfunction!(ols_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest_toys, m)?)?;
    m.add_function(wrap_pyfunction!(profile_scan, m)?)?;
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

    // Add classes
    m.add_class::<PyHistFactoryModel>()?;
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

    // DifferentiableSession (CUDA only)
    #[cfg(feature = "cuda")]
    m.add_class::<PyDiffSession>()?;

    // Back-compat aliases used in plans/docs.
    let model_cls = m.getattr("HistFactoryModel")?;
    m.add("PyModel", model_cls)?;
    let fit_cls = m.getattr("FitResult")?;
    m.add("PyFitResult", fit_cls)?;

    Ok(())
}
