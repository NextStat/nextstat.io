//! Python bindings for NextStat

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

// Re-export types from core crates
use ns_inference::mle::MaximumLikelihoodEstimator as RustMLE;
use ns_inference::{hypotest::AsymptoticCLsContext as RustCLsCtx, profile_likelihood as pl};
use ns_translate::pyhf::{HistFactoryModel as RustModel, Workspace as RustWorkspace};

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
    n_evaluations: usize,
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

    /// Fit a model to data
    #[pyo3(signature = (model, *, data=None))]
    fn fit(&self, model: &PyHistFactoryModel, data: Option<Vec<f64>>) -> PyResult<PyFitResult> {
        let fit_model = if let Some(obs_main) = data {
            model
                .inner
                .with_observed_main(&obs_main)
                .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
        } else {
            model.inner.clone()
        };

        let result = self
            .inner
            .fit(&fit_model)
            .map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;

        Ok(PyFitResult {
            parameters: result.parameters,
            uncertainties: result.uncertainties,
            nll: result.nll,
            converged: result.converged,
            n_evaluations: result.n_evaluations,
        })
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
fn fit(model: &PyHistFactoryModel, data: Option<Vec<f64>>) -> PyResult<PyFitResult> {
    let mle = RustMLE::new();
    let fit_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let result =
        mle.fit(&fit_model).map_err(|e| PyValueError::new_err(format!("Fit failed: {}", e)))?;

    Ok(PyFitResult {
        parameters: result.parameters,
        uncertainties: result.uncertainties,
        nll: result.nll,
        converged: result.converged,
        n_evaluations: result.n_evaluations,
    })
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

/// Python submodule: nextstat._core
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ns_core::VERSION)?;

    // Convenience functions (pyhf-style API).
    m.add_function(wrap_pyfunction!(from_pyhf, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(hypotest, m)?)?;
    m.add_function(wrap_pyfunction!(profile_scan, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limit, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limits, m)?)?;

    // Add classes
    m.add_class::<PyHistFactoryModel>()?;
    m.add_class::<PyMaximumLikelihoodEstimator>()?;
    m.add_class::<PyFitResult>()?;

    // Back-compat aliases used in plans/docs.
    let model_cls = m.getattr("HistFactoryModel")?;
    m.add("PyModel", model_cls)?;
    let fit_cls = m.getattr("FitResult")?;
    m.add("PyFitResult", fit_cls)?;

    Ok(())
}
