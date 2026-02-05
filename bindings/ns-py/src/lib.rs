//! Python bindings for NextStat

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

// Re-export types from core crates
use ns_inference::chain::sample_nuts_multichain;
use ns_inference::diagnostics::compute_diagnostics;
use ns_inference::mle::MaximumLikelihoodEstimator as RustMLE;
use ns_inference::nuts::NutsConfig;
use ns_inference::{hypotest::AsymptoticCLsContext as RustCLsCtx, profile_likelihood as pl};
use ns_translate::pyhf::{HistFactoryModel as RustModel, Workspace as RustWorkspace};
use ns_viz::{ClsCurveArtifact, ProfileCurveArtifact};

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
#[pyo3(signature = (model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, target_accept=0.8, init_jitter=0.5, data=None))]
fn sample(
    py: Python<'_>,
    model: &PyHistFactoryModel,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    max_treedepth: usize,
    target_accept: f64,
    init_jitter: f64,
    data: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    let sample_model = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("Failed to set observed data: {}", e)))?
    } else {
        model.inner.clone()
    };

    let config = NutsConfig { max_treedepth, target_accept, init_jitter };

    // Release GIL during Rayon-parallel sampling.
    let result = py
        .detach(|| {
            sample_nuts_multichain(&sample_model, n_chains, n_warmup, n_samples, seed, config)
        })
        .map_err(|e| PyValueError::new_err(format!("Sampling failed: {}", e)))?;

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
    m.add_function(wrap_pyfunction!(hypotest, m)?)?;
    m.add_function(wrap_pyfunction!(profile_scan, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limit, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limits, m)?)?;
    m.add_function(wrap_pyfunction!(upper_limits_root, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(cls_curve, m)?)?;
    m.add_function(wrap_pyfunction!(profile_curve, m)?)?;

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
