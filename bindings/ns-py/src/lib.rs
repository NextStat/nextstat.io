//! Python bindings for NextStat

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

use nalgebra::{DMatrix, DVector};

// Re-export types from core crates
use ns_core::traits::{LogDensityModel, PreparedNll};
use ns_core::{Error as NsError, Result as NsResult};
use ns_inference::chain::{Chain as RustChain, SamplerResult as RustSamplerResult};
use ns_inference::diagnostics::{QualityGates, compute_diagnostics, quality_summary};
use ns_inference::mle::{MaximumLikelihoodEstimator as RustMLE, RankingEntry};
use ns_inference::nuts::{NutsConfig, sample_nuts};
use ns_inference::{
    ComposedGlmModel as RustComposedGlmModel, LinearRegressionModel as RustLinearRegressionModel,
    LogisticRegressionModel as RustLogisticRegressionModel, ModelBuilder as RustModelBuilder,
    PoissonRegressionModel as RustPoissonRegressionModel,
    hypotest::AsymptoticCLsContext as RustCLsCtx, ols_fit as rust_ols_fit,
    profile_likelihood as pl,
};
use ns_inference::regression::NegativeBinomialRegressionModel as RustNegativeBinomialRegressionModel;
use ns_inference::timeseries::kalman::{
    KalmanModel as RustKalmanModel, kalman_filter as rust_kalman_filter,
    rts_smoother as rust_rts_smoother,
};
use ns_inference::timeseries::em::{
    KalmanEmConfig as RustKalmanEmConfig, kalman_em as rust_kalman_em,
};
use ns_inference::timeseries::forecast::kalman_forecast as rust_kalman_forecast;
use ns_inference::timeseries::simulate::kalman_simulate as rust_kalman_simulate;
use ns_translate::pyhf::{HistFactoryModel as RustModel, Workspace as RustWorkspace};
use ns_viz::{ClsCurveArtifact, ProfileCurveArtifact};

fn sample_nuts_multichain_with_seeds(
    model: &(impl LogDensityModel + Sync),
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

    /// Number of parameters.
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// Compute negative log-likelihood
    fn nll(&self, params: Vec<f64>) -> PyResult<f64> {
        self.inner
            .nll(&params)
            .map_err(|e| PyValueError::new_err(format!("NLL computation failed: {}", e)))
    }

    /// Gradient of negative log-likelihood.
    fn grad_nll(&self, params: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .grad_nll(&params)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))
    }

    /// Expected data matching `pyhf.Model.expected_data`.
    #[pyo3(signature = (params, *, include_auxdata=true))]
    fn expected_data(&self, params: Vec<f64>, include_auxdata: bool) -> PyResult<Vec<f64>> {
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
#[pyo3(signature = (model, ys, *, max_iter=50, tol=1e-6, estimate_q=true, estimate_r=true, min_diag=1e-12))]
fn kalman_em(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
    max_iter: usize,
    tol: f64,
    estimate_q: bool,
    estimate_r: bool,
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
        min_diag,
    };

    let res = rust_kalman_em(&model.inner, &ys, cfg)
        .map_err(|e| PyValueError::new_err(format!("kalman_em failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("converged", res.converged)?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("loglik_trace", res.loglik_trace)?;
    out.set_item("q", dmatrix_to_nested(&res.model.q))?;
    out.set_item("r", dmatrix_to_nested(&res.model.r))?;
    out.set_item("model", Py::new(py, PyKalmanModel { inner: res.model })?)?;

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (model, ys, *, steps=1))]
fn kalman_forecast(
    py: Python<'_>,
    model: &PyKalmanModel,
    ys: Vec<Vec<Option<f64>>>,
    steps: usize,
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

    Ok(out.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (model, *, t_max, seed=42))]
fn kalman_simulate(py: Python<'_>, model: &PyKalmanModel, t_max: usize, seed: u64) -> PyResult<Py<PyAny>> {
    let sim = rust_kalman_simulate(&model.inner, t_max, seed)
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
    /// Build a composed Gaussian linear regression model (sigma fixed to 1).
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false))]
    fn linear_regression(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        include_intercept: bool,
        group_idx: Option<Vec<usize>>,
        n_groups: Option<usize>,
        coef_prior_mu: f64,
        coef_prior_sigma: f64,
        penalize_intercept: bool,
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

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);
            b = b
                .with_random_intercept(group_idx, ng)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a composed logistic regression model (Bernoulli-logit).
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false))]
    fn logistic_regression(
        x: Vec<Vec<f64>>,
        y: Vec<u8>,
        include_intercept: bool,
        group_idx: Option<Vec<usize>>,
        n_groups: Option<usize>,
        coef_prior_mu: f64,
        coef_prior_sigma: f64,
        penalize_intercept: bool,
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

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);
            b = b
                .with_random_intercept(group_idx, ng)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a composed Poisson regression model (log link) with optional offset.
    #[staticmethod]
    #[pyo3(signature = (x, y, *, include_intercept=true, offset=None, group_idx=None, n_groups=None, coef_prior_mu=0.0, coef_prior_sigma=10.0, penalize_intercept=false))]
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

        if let Some(group_idx) = group_idx {
            let ng = n_groups.unwrap_or_else(|| group_idx.iter().copied().max().unwrap_or(0) + 1);
            b = b
                .with_random_intercept(group_idx, ng)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let inner = b.build().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn n_params(&self) -> usize {
        self.inner.dim()
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

/// Python wrapper for MaximumLikelihoodEstimator
#[pyclass(name = "MaximumLikelihoodEstimator")]
struct PyMaximumLikelihoodEstimator {
    inner: RustMLE,
}

#[pymethods]
impl PyMaximumLikelihoodEstimator {
    #[new]
    fn new() -> Self {
        PyMaximumLikelihoodEstimator { inner: RustMLE::new() }
    }

    /// Fit any supported model (generic `LogDensityModel`) using MLE.
    ///
    /// `data=` is only supported for `HistFactoryModel` (overrides main-bin observations).
    #[pyo3(signature = (model, *, data=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        data: Option<Vec<f64>>,
    ) -> PyResult<PyFitResult> {
        // Extract a Rust-owned model before releasing the GIL.
        enum FitModel {
            HistFactory(RustModel),
            GaussianMean(GaussianMeanModel),
            LinearRegression(RustLinearRegressionModel),
            LogisticRegression(RustLogisticRegressionModel),
            PoissonRegression(RustPoissonRegressionModel),
            NegativeBinomialRegression(RustNegativeBinomialRegressionModel),
            ComposedGlm(RustComposedGlmModel),
        }

        let fit_model = if let Ok(hf) = model.extract::<PyRef<'_, PyHistFactoryModel>>() {
            let m = if let Some(obs_main) = data {
                hf.inner.with_observed_main(&obs_main).map_err(|e| {
                    PyValueError::new_err(format!("Failed to set observed data: {}", e))
                })?
            } else {
                hf.inner.clone()
            };
            FitModel::HistFactory(m)
        } else if let Ok(gm) = model.extract::<PyRef<'_, PyGaussianMeanModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::GaussianMean(gm.inner.clone())
        } else if let Ok(lr) = model.extract::<PyRef<'_, PyLinearRegressionModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::LinearRegression(lr.inner.clone())
        } else if let Ok(logit) = model.extract::<PyRef<'_, PyLogisticRegressionModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::LogisticRegression(logit.inner.clone())
        } else if let Ok(pois) = model.extract::<PyRef<'_, PyPoissonRegressionModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::PoissonRegression(pois.inner.clone())
        } else if let Ok(nb) = model.extract::<PyRef<'_, PyNegativeBinomialRegressionModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::NegativeBinomialRegression(nb.inner.clone())
        } else if let Ok(glm) = model.extract::<PyRef<'_, PyComposedGlmModel>>() {
            if data.is_some() {
                return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
            }
            FitModel::ComposedGlm(glm.inner.clone())
        } else {
            return Err(PyValueError::new_err(
                "Unsupported model type. Expected HistFactoryModel, GaussianMeanModel, a regression model, or ComposedGlmModel.",
            ));
        };

        let mle = self.inner.clone();
        let result = match fit_model {
            FitModel::HistFactory(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::GaussianMean(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::LinearRegression(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::LogisticRegression(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::PoissonRegression(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::NegativeBinomialRegression(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
            FitModel::ComposedGlm(m) => {
                let mle = mle.clone();
                py.detach(move || mle.fit(&m))
                    .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?
            }
        };

        Ok(PyFitResult {
            parameters: result.parameters,
            uncertainties: result.uncertainties,
            nll: result.nll,
            converged: result.converged,
            n_iter: result.n_iter,
            n_fev: result.n_fev,
            n_gev: result.n_gev,
        })
    }

    /// Fit multiple HistFactoryModel instances in parallel (Rayon).
    ///
    /// Two call forms:
    /// - `fit_batch(models)` — list of models, each with its own observations
    /// - `fit_batch(model, datasets)` — single model + list of observation vectors
    #[pyo3(signature = (models_or_model, datasets=None))]
    fn fit_batch<'py>(
        &self,
        py: Python<'py>,
        models_or_model: &Bound<'py, PyAny>,
        datasets: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<PyFitResult>> {
        let mle = self.inner.clone();

        let models: Vec<RustModel> = if let Some(datasets) = datasets {
            // Single model + multiple datasets
            let hf = models_or_model
                .extract::<PyRef<'_, PyHistFactoryModel>>()
                .map_err(|_| {
                    PyValueError::new_err(
                        "When datasets is given, first argument must be a HistFactoryModel",
                    )
                })?;
            let base = hf.inner.clone();
            datasets
                .into_iter()
                .map(|ds| {
                    base.with_observed_main(&ds).map_err(|e| {
                        PyValueError::new_err(format!("Failed to set observed data: {}", e))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            // List of models
            let list = models_or_model
                .cast::<PyList>()
                .map_err(|_| {
                    PyValueError::new_err(
                        "Expected a list of HistFactoryModel or (model, datasets=...)",
                    )
                })?;
            let mut ms = Vec::with_capacity(list.len());
            for item in list.iter() {
                let hf = item.extract::<PyRef<'_, PyHistFactoryModel>>().map_err(|_| {
                    PyValueError::new_err("All items in the list must be HistFactoryModel")
                })?;
                ms.push(hf.inner.clone());
            }
            ms
        };

        let results = py.detach(move || mle.fit_batch(&models));

        results
            .into_iter()
            .map(|r| {
                let r = r.map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;
                Ok(PyFitResult {
                    parameters: r.parameters,
                    uncertainties: r.uncertainties,
                    nll: r.nll,
                    converged: r.converged,
                    n_iter: r.n_iter,
                    n_fev: r.n_fev,
                    n_gev: r.n_gev,
                })
            })
            .collect()
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
                Ok(PyFitResult {
                    parameters: r.parameters,
                    uncertainties: r.uncertainties,
                    nll: r.nll,
                    converged: r.converged,
                    n_iter: r.n_iter,
                    n_fev: r.n_fev,
                    n_gev: r.n_gev,
                })
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
}

/// Convenience wrapper: create model from pyhf JSON string.
#[pyfunction]
fn from_pyhf(json_str: &str) -> PyResult<PyHistFactoryModel> {
    PyHistFactoryModel::from_workspace(json_str)
}

/// Convenience wrapper: fit model with optional overridden observations.
#[pyfunction]
#[pyo3(signature = (model, *, data=None))]
fn fit<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    data: Option<Vec<f64>>,
) -> PyResult<PyFitResult> {
    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.fit(py, model, data)
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

/// Convenience wrapper: nuisance-parameter ranking (impact on POI).
#[pyfunction]
fn ranking<'py>(py: Python<'py>, model: &PyHistFactoryModel) -> PyResult<Vec<Py<PyAny>>> {
    let mle = PyMaximumLikelihoodEstimator { inner: RustMLE::new() };
    mle.ranking(py, model)
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

/// Profile likelihood scan over POI values (q_mu).
#[pyfunction]
#[pyo3(signature = (model, mu_values, *, data=None))]
fn profile_scan(
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

    let scan = pl::scan(&mle, &fit_model, &mu_values)
        .map_err(|e| PyValueError::new_err(format!("Profile scan failed: {}", e)))?;

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
    let config = NutsConfig {
        max_treedepth,
        target_accept,
        init_jitter,
        init_jitter_rel,
        init_overdispersed_rel,
        ..Default::default()
    };

    // Extract a Rust-owned model before releasing the GIL.
    enum SampleModel {
        HistFactory(RustModel),
        GaussianMean(GaussianMeanModel),
        LinearRegression(RustLinearRegressionModel),
        LogisticRegression(RustLogisticRegressionModel),
        PoissonRegression(RustPoissonRegressionModel),
        NegativeBinomialRegression(RustNegativeBinomialRegressionModel),
        ComposedGlm(RustComposedGlmModel),
    }

    let sample_model = if let Ok(hf) = model.extract::<PyRef<'_, PyHistFactoryModel>>() {
        let m = if let Some(obs_main) = data {
            hf.inner
                .with_observed_main(&obs_main)
                .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
        } else {
            hf.inner.clone()
        };
        SampleModel::HistFactory(m)
    } else if let Ok(gm) = model.extract::<PyRef<'_, PyGaussianMeanModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::GaussianMean(gm.inner.clone())
    } else if let Ok(lr) = model.extract::<PyRef<'_, PyLinearRegressionModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::LinearRegression(lr.inner.clone())
    } else if let Ok(logit) = model.extract::<PyRef<'_, PyLogisticRegressionModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::LogisticRegression(logit.inner.clone())
    } else if let Ok(pois) = model.extract::<PyRef<'_, PyPoissonRegressionModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::PoissonRegression(pois.inner.clone())
    } else if let Ok(nb) = model.extract::<PyRef<'_, PyNegativeBinomialRegressionModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::NegativeBinomialRegression(nb.inner.clone())
    } else if let Ok(glm) = model.extract::<PyRef<'_, PyComposedGlmModel>>() {
        if data.is_some() {
            return Err(PyValueError::new_err("data= is only supported for HistFactoryModel"));
        }
        SampleModel::ComposedGlm(glm.inner.clone())
    } else {
        return Err(PyValueError::new_err(
            "Unsupported model type. Expected HistFactoryModel, GaussianMeanModel, a regression model, or ComposedGlmModel.",
        ));
    };

    // Release GIL during sampling (Rayon-parallel for multi-chain).
    let result = match sample_model {
        SampleModel::HistFactory(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::GaussianMean(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::LinearRegression(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::LogisticRegression(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::PoissonRegression(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::NegativeBinomialRegression(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
        SampleModel::ComposedGlm(m) => {
            let config = config.clone();
            py.detach(move || {
                let seeds: Vec<u64> =
                    (0..n_chains).map(|chain_id| seed.wrapping_add(chain_id as u64)).collect();
                sample_nuts_multichain_with_seeds(&m, n_warmup, n_samples, &seeds, config)
            })
            .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?
        }
    };

    let diag = compute_diagnostics(&result);
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

    let scan = pl::scan(&mle, &fit_model, &mu_values)
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

/// Python submodule: nextstat._core
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ns_core::VERSION)?;

    // Convenience functions (pyhf-style API).
    m.add_function(wrap_pyfunction!(from_pyhf, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_toys, m)?)?;
    m.add_function(wrap_pyfunction!(ranking, m)?)?;
    m.add_function(wrap_pyfunction!(ols_fit, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest, m)?)?;
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
    m.add_class::<PyKalmanModel>()?;
    m.add_class::<PyGaussianMeanModel>()?;
    m.add_class::<PyLinearRegressionModel>()?;
    m.add_class::<PyLogisticRegressionModel>()?;
    m.add_class::<PyPoissonRegressionModel>()?;
    m.add_class::<PyNegativeBinomialRegressionModel>()?;
    m.add_class::<PyComposedGlmModel>()?;
    m.add_class::<PyMaximumLikelihoodEstimator>()?;
    m.add_class::<PyFitResult>()?;

    // Back-compat aliases used in plans/docs.
    let model_cls = m.getattr("HistFactoryModel")?;
    m.add("PyModel", model_cls)?;
    let fit_cls = m.getattr("FitResult")?;
    m.add("PyFitResult", fit_cls)?;

    Ok(())
}
