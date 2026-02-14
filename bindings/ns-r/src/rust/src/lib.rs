use extendr_api::prelude::*;
use nalgebra::{DMatrix, DVector};
use ns_core::traits::LogDensityModel;

fn normal_logpdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln();
    let z = (x - mu) / sigma;
    log_norm - 0.5 * z * z
}

fn setup_runtime(threads: usize) {
    if threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }
}

fn histfactory_model_from_json(json_str: &str) -> extendr_api::Result<ns_translate::pyhf::HistFactoryModel> {
    let format = ns_translate::hs3::detect::detect_format(json_str);
    match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => ns_translate::hs3::convert::from_hs3_default(json_str)
            .map_err(|e| Error::Other(format!("HS3 loading failed: {e}"))),
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(json_str)
                .map_err(|e| Error::Other(format!("invalid pyhf JSON: {e}")))?;
            ns_translate::pyhf::HistFactoryModel::from_workspace_with_settings(
                &workspace,
                ns_translate::pyhf::NormSysInterpCode::Code1,
                ns_translate::pyhf::HistoSysInterpCode::Code0,
            )
            .map_err(|e| Error::Other(format!("failed to build HistFactory model: {e}")))
        }
    }
}

#[extendr]
fn ns_normal_logpdf(x: Vec<f64>, mu: f64, sigma: f64) -> extendr_api::Result<Vec<f64>> {
    if x.is_empty() {
        return Ok(vec![]);
    }
    if !mu.is_finite() {
        return Err(Error::Other("mu must be finite".into()));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Other("sigma must be finite and > 0".into()));
    }
    Ok(x.into_iter().map(|xi| normal_logpdf_scalar(xi, mu, sigma)).collect())
}

#[extendr]
fn ns_ols_fit(x: RMatrix<f64>, y: Vec<f64>, include_intercept: bool) -> extendr_api::Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err(Error::Other("x must have positive dimensions".into()));
    }
    if y.len() != n {
        return Err(Error::Other("length(y) must match nrow(x)".into()));
    }

    // R matrices are column-major. Convert to row-wise Vec<Vec<f64>> expected by ns-inference.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        for j in 0..p {
            row.push(x[[i, j]]);
        }
        rows.push(row);
    }

    let beta = ns_inference::regression::ols_fit(rows, y, include_intercept)
        .map_err(|e| Error::Other(format!("ols_fit: {e}")))?;
    Ok(beta)
}

#[extendr]
fn nextstat_fit(workspace_json: &str) -> extendr_api::Result<List> {
    let json_str = workspace_json.trim();
    if json_str.is_empty() {
        return Err(Error::Other("workspace_json must be a non-empty string".into()));
    }

    // Default to single-thread for R-friendliness and deterministic behavior.
    setup_runtime(1);

    let model = histfactory_model_from_json(json_str)?;

    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let result = mle
        .fit(&model)
        .map_err(|e| Error::Other(format!("MLE fit failed: {e}")))?;

    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();
    let poi_index_1based: Option<i32> = model.poi_index().map(|i| (i + 1) as i32);
    let (mu_hat, mu_sigma, poi_name) = model
        .poi_index()
        .map(|i| {
            (
                result.parameters.get(i).copied(),
                result.uncertainties.get(i).copied(),
                parameter_names.get(i).cloned(),
            )
        })
        .unwrap_or((None, None, None));

    Ok(list!(
        parameter_names = parameter_names,
        poi_index = poi_index_1based,
        poi_name = poi_name,
        bestfit = result.parameters,
        uncertainties = result.uncertainties,
        nll = result.nll,
        twice_nll = 2.0 * result.nll,
        converged = result.converged,
        n_iter = result.n_iter,
        n_fev = result.n_fev,
        n_gev = result.n_gev,
        mu_hat = mu_hat,
        mu_sigma = mu_sigma
    ))
}

#[extendr]
fn nextstat_hypotest(workspace_json: &str, mu_test: f64) -> extendr_api::Result<List> {
    let json_str = workspace_json.trim();
    if json_str.is_empty() {
        return Err(Error::Other("workspace_json must be a non-empty string".into()));
    }
    if !mu_test.is_finite() || mu_test < 0.0 {
        return Err(Error::Other("mu_test must be finite and >= 0".into()));
    }

    setup_runtime(1);
    let model = histfactory_model_from_json(json_str)?;

    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, &model)
        .map_err(|e| Error::Other(format!("hypotest context init failed: {e}")))?;
    let r = ctx
        .hypotest_qtilde(&mle, mu_test)
        .map_err(|e| Error::Other(format!("hypotest failed: {e}")))?;

    Ok(list!(
        mu_test = r.mu_test,
        cls = r.cls,
        clb = r.clb,
        clsb = r.clsb,
        teststat = r.teststat,
        q_mu = r.q_mu,
        q_mu_a = r.q_mu_a,
        mu_hat = r.mu_hat
    ))
}

#[extendr]
fn nextstat_upper_limit(
    workspace_json: &str,
    cl: f64,
    mu_range: Vec<f64>,
    points: i32,
) -> extendr_api::Result<List> {
    let json_str = workspace_json.trim();
    if json_str.is_empty() {
        return Err(Error::Other("workspace_json must be a non-empty string".into()));
    }
    if !(cl.is_finite() && 0.0 < cl && cl < 1.0) {
        return Err(Error::Other("cl must be in (0,1)".into()));
    }
    if mu_range.len() != 2 {
        return Err(Error::Other("mu_range must be a numeric vector of length 2".into()));
    }
    let mu_lo = mu_range[0];
    let mu_hi = mu_range[1];
    if !mu_lo.is_finite() || !mu_hi.is_finite() || mu_lo < 0.0 || mu_hi < 0.0 || mu_hi <= mu_lo {
        return Err(Error::Other("mu_range must satisfy 0 <= mu_lo < mu_hi".into()));
    }
    let n = points.max(2) as usize;

    setup_runtime(1);
    let model = histfactory_model_from_json(json_str)?;

    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, &model)
        .map_err(|e| Error::Other(format!("upper limit context init failed: {e}")))?;

    let alpha = 1.0 - cl;
    let step = (mu_hi - mu_lo) / (n as f64 - 1.0);
    let scan: Vec<f64> = (0..n).map(|i| mu_lo + step * i as f64).collect();

    let r = ctx
        .upper_limits_qtilde_linear_scan_result(&mle, alpha, &scan)
        .map_err(|e| Error::Other(format!("upper limit scan failed: {e}")))?;

    // `expected_cls` is a list-of-length-5 numeric vectors (one per scan point).
    let expected_cls: List = List::from_values(
        r.expected_cls
            .iter()
            .map(|a| Robj::from(a.to_vec()))
            .collect::<Vec<Robj>>(),
    );

    Ok(list!(
        cl = cl,
        alpha = alpha,
        nsigma_order = ns_inference::hypotest::NSIGMA_ORDER.to_vec(),
        scan = r.scan,
        observed_cls = r.observed_cls,
        expected_cls = expected_cls,
        observed_limit = r.observed_limit,
        expected_limits = r.expected_limits.to_vec()
    ))
}

fn rmatrix_to_rows(x: &RMatrix<f64>) -> extendr_api::Result<Vec<Vec<f64>>> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err(Error::Other("x must have positive dimensions".into()));
    }
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(p);
        for j in 0..p {
            let v = x[[i, j]];
            if !v.is_finite() {
                return Err(Error::Other("x must contain only finite values".into()));
            }
            row.push(v);
        }
        rows.push(row);
    }
    Ok(rows)
}

#[extendr]
fn nextstat_glm_logistic(x: RMatrix<f64>, y: Vec<f64>, include_intercept: bool) -> extendr_api::Result<List> {
    let rows = rmatrix_to_rows(&x)?;
    let n = rows.len();
    if y.len() != n {
        return Err(Error::Other("length(y) must match nrow(x)".into()));
    }
    let y_u8: Vec<u8> = y
        .into_iter()
        .map(|v| {
            if v == 0.0 {
                Ok(0u8)
            } else if v == 1.0 {
                Ok(1u8)
            } else {
                Err(Error::Other("y must contain only 0/1 values for logistic regression".into()))
            }
        })
        .collect::<extendr_api::Result<Vec<u8>>>()?;

    setup_runtime(1);
    let model = ns_inference::regression::LogisticRegressionModel::new(rows, y_u8, include_intercept)
        .map_err(|e| Error::Other(format!("logistic model: {e}")))?;

    let parameter_names = model.parameter_names();
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).map_err(|e| Error::Other(format!("logistic fit: {e}")))?;

    let k = fit.parameters.len() as f64;
    let deviance = 2.0 * fit.nll;
    let aic = 2.0 * k + deviance;

    Ok(list!(
        parameter_names = parameter_names,
        coefficients = fit.parameters,
        se = fit.uncertainties,
        nll = fit.nll,
        deviance = deviance,
        aic = aic,
        converged = fit.converged,
        n_iter = fit.n_iter,
        n_fev = fit.n_fev,
        n_gev = fit.n_gev
    ))
}

#[extendr]
fn nextstat_glm_poisson(x: RMatrix<f64>, y: Vec<f64>, include_intercept: bool) -> extendr_api::Result<List> {
    let rows = rmatrix_to_rows(&x)?;
    let n = rows.len();
    if y.len() != n {
        return Err(Error::Other("length(y) must match nrow(x)".into()));
    }
    let y_u64: Vec<u64> = y
        .into_iter()
        .map(|v| {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Other("y must be finite and >= 0 for Poisson regression".into()));
            }
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(Error::Other("y must be integer-valued for Poisson regression".into()));
            }
            Ok(r as u64)
        })
        .collect::<extendr_api::Result<Vec<u64>>>()?;

    setup_runtime(1);
    let model = ns_inference::regression::PoissonRegressionModel::new(rows, y_u64, include_intercept, None)
        .map_err(|e| Error::Other(format!("poisson model: {e}")))?;

    let parameter_names = model.parameter_names();
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).map_err(|e| Error::Other(format!("poisson fit: {e}")))?;

    let k = fit.parameters.len() as f64;
    let deviance = 2.0 * fit.nll;
    let aic = 2.0 * k + deviance;

    Ok(list!(
        parameter_names = parameter_names,
        coefficients = fit.parameters,
        se = fit.uncertainties,
        nll = fit.nll,
        deviance = deviance,
        aic = aic,
        converged = fit.converged,
        n_iter = fit.n_iter,
        n_fev = fit.n_fev,
        n_gev = fit.n_gev
    ))
}

#[extendr]
fn nextstat_glm_negbin(x: RMatrix<f64>, y: Vec<f64>, include_intercept: bool) -> extendr_api::Result<List> {
    let rows = rmatrix_to_rows(&x)?;
    let n = rows.len();
    if y.len() != n {
        return Err(Error::Other("length(y) must match nrow(x)".into()));
    }
    let y_u64: Vec<u64> = y
        .into_iter()
        .map(|v| {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Other("y must be finite and >= 0 for negbin regression".into()));
            }
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(Error::Other("y must be integer-valued for negbin regression".into()));
            }
            Ok(r as u64)
        })
        .collect::<extendr_api::Result<Vec<u64>>>()?;

    setup_runtime(1);
    let model = ns_inference::regression::NegativeBinomialRegressionModel::new(rows, y_u64, include_intercept, None)
        .map_err(|e| Error::Other(format!("negbin model: {e}")))?;

    let parameter_names = model.parameter_names();
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).map_err(|e| Error::Other(format!("negbin fit: {e}")))?;

    let k = fit.parameters.len() as f64;
    let deviance = 2.0 * fit.nll;
    let aic = 2.0 * k + deviance;

    Ok(list!(
        parameter_names = parameter_names,
        coefficients = fit.parameters,
        se = fit.uncertainties,
        nll = fit.nll,
        deviance = deviance,
        aic = aic,
        converged = fit.converged,
        n_iter = fit.n_iter,
        n_fev = fit.n_fev,
        n_gev = fit.n_gev
    ))
}

fn rmatrix_to_dmatrix(x: &RMatrix<f64>, name: &str) -> extendr_api::Result<DMatrix<f64>> {
    let n = x.nrows();
    let m = x.ncols();
    if n == 0 || m == 0 {
        return Err(Error::Other(format!("{name} must have positive dimensions")));
    }
    let mut data = Vec::with_capacity(n * m);
    for i in 0..n {
        for j in 0..m {
            let v = x[[i, j]];
            if !v.is_finite() {
                return Err(Error::Other(format!("{name} must contain only finite values")));
            }
            data.push(v);
        }
    }
    Ok(DMatrix::from_row_slice(n, m, &data))
}

fn dvector_list(vs: Vec<DVector<f64>>) -> List {
    List::from_values(vs.into_iter().map(|v| Robj::from(v.iter().copied().collect::<Vec<f64>>())).collect::<Vec<Robj>>())
}

#[extendr]
fn nextstat_kalman(y: Vec<f64>, f: RMatrix<f64>, h: RMatrix<f64>, q: RMatrix<f64>, r: RMatrix<f64>) -> extendr_api::Result<List> {
    if y.is_empty() {
        return Err(Error::Other("y must be non-empty".into()));
    }
    if y.iter().any(|v| !v.is_finite() && !v.is_nan()) {
        return Err(Error::Other("y must be finite or NaN (NaN means missing)".into()));
    }

    let f = rmatrix_to_dmatrix(&f, "F")?;
    let h = rmatrix_to_dmatrix(&h, "H")?;
    let q = rmatrix_to_dmatrix(&q, "Q")?;
    let r = rmatrix_to_dmatrix(&r, "R")?;

    let n_state = f.nrows();
    if f.ncols() != n_state {
        return Err(Error::Other("F must be square (n_state x n_state)".into()));
    }
    if h.nrows() != 1 || h.ncols() != n_state {
        return Err(Error::Other("H must be 1 x n_state (this wrapper currently supports 1D observations)".into()));
    }
    if q.nrows() != n_state || q.ncols() != n_state {
        return Err(Error::Other("Q must be n_state x n_state".into()));
    }
    if r.nrows() != 1 || r.ncols() != 1 {
        return Err(Error::Other("R must be 1 x 1 (this wrapper currently supports 1D observations)".into()));
    }

    // Simple defaults for initial state.
    let m0 = DVector::from_element(n_state, 0.0);
    let mut p0 = DMatrix::identity(n_state, n_state);
    // Slightly diffuse prior to reduce sensitivity to m0.
    p0.scale_mut(1e3);

    let model = ns_inference::timeseries::kalman::KalmanModel::new(f, q, h, r, m0, p0)
        .map_err(|e| Error::Other(format!("kalman model: {e}")))?;

    let ys: Vec<DVector<f64>> = y.into_iter().map(|v| DVector::from_row_slice(&[v])).collect();
    let fr = ns_inference::timeseries::kalman::kalman_filter(&model, ys.as_slice())
        .map_err(|e| Error::Other(format!("kalman_filter: {e}")))?;
    let sr = ns_inference::timeseries::kalman::rts_smoother(&model, &fr)
        .map_err(|e| Error::Other(format!("rts_smoother: {e}")))?;

    Ok(list!(
        log_likelihood = fr.log_likelihood,
        filtered_means = dvector_list(fr.filtered_means),
        smoothed_means = dvector_list(sr.smoothed_means)
    ))
}

#[extendr]
fn nextstat_garch(y: Vec<f64>) -> extendr_api::Result<List> {
    let fit = ns_inference::timeseries::volatility::garch11_fit(&y, ns_inference::timeseries::volatility::Garch11Config::default())
        .map_err(|e| Error::Other(format!("garch11_fit: {e}")))?;
    let p = fit.params;
    Ok(list!(
        mu = p.mu,
        omega = p.omega,
        alpha = p.alpha,
        beta = p.beta,
        log_likelihood = fit.log_likelihood,
        conditional_variance = fit.conditional_variance,
        converged = fit.optimization.converged,
        n_iter = fit.optimization.n_iter,
        n_fev = fit.optimization.n_fev,
        n_gev = fit.optimization.n_gev
    ))
}

#[extendr]
fn nextstat_sv(y: Vec<f64>, log_eps: f64) -> extendr_api::Result<List> {
    let cfg = ns_inference::timeseries::volatility::SvLogChi2Config {
        log_eps,
        ..Default::default()
    };
    let fit = ns_inference::timeseries::volatility::sv_logchi2_fit(&y, cfg)
        .map_err(|e| Error::Other(format!("sv_logchi2_fit: {e}")))?;
    let p = fit.params;
    Ok(list!(
        mu = p.mu,
        phi = p.phi,
        sigma = p.sigma,
        log_likelihood = fit.log_likelihood,
        smoothed_h = fit.smoothed_h,
        smoothed_sigma = fit.smoothed_sigma,
        converged = fit.optimization.converged,
        n_iter = fit.optimization.n_iter,
        n_fev = fit.optimization.n_fev,
        n_gev = fit.optimization.n_gev
    ))
}

// ---------------------------------------------------------------------------
// Pharma helpers
// ---------------------------------------------------------------------------

fn parse_error_model_r(error_model: &str, sigma: &[f64]) -> extendr_api::Result<ns_inference::pk::ErrorModel> {
    use ns_inference::pk::ErrorModel;
    match error_model {
        "additive" => {
            let s = sigma.first().copied().unwrap_or(1.0);
            Ok(ErrorModel::Additive(s))
        }
        "proportional" => {
            let s = sigma.first().copied().unwrap_or(0.1);
            Ok(ErrorModel::Proportional(s))
        }
        "combined" => {
            if sigma.len() < 2 {
                return Err(Error::Other("combined error model requires sigma of length 2: c(sigma_add, sigma_prop)".into()));
            }
            Ok(ErrorModel::Combined { sigma_add: sigma[0], sigma_prop: sigma[1] })
        }
        _ => Err(Error::Other(format!("unknown error_model: {error_model}; expected additive/proportional/combined"))),
    }
}

// ---------------------------------------------------------------------------
// ns_foce — FOCE/FOCEI population PK estimation
// ---------------------------------------------------------------------------

#[extendr]
fn ns_foce(
    times: Vec<f64>,
    dv: Vec<f64>,
    id: Vec<i32>,
    n_subjects: i32,
    dose: f64,
    bioav: f64,
    error_model: &str,
    sigma: Vec<f64>,
    theta_init: Vec<f64>,
    omega_init: Vec<f64>,
    interaction: bool,
    max_outer_iter: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    setup_runtime(1);
    let em = parse_error_model_r(error_model, &sigma)?;
    let subject_idx: Vec<usize> = id.iter().map(|&i| i as usize).collect();

    let config = ns_inference::foce::FoceConfig {
        max_outer_iter: max_outer_iter as usize,
        max_inner_iter: 20,
        tol,
        interaction,
    };
    let estimator = ns_inference::foce::FoceEstimator::new(config);
    let result = estimator
        .fit_1cpt_oral(&times, &dv, &subject_idx, n_subjects as usize, dose, bioav, em, &theta_init, &omega_init)
        .map_err(|e| Error::Other(format!("FOCE failed: {e}")))?;

    // Convert eta (Vec<Vec<f64>>) to flat matrix for R (n_subjects x 3).
    let eta_flat: Vec<f64> = result.eta.iter().flat_map(|row| row.iter().copied()).collect();
    let n_sub = result.eta.len();
    let n_params = if n_sub > 0 { result.eta[0].len() } else { 3 };

    // Correlation matrix → flat list of vectors.
    let corr: List = List::from_values(
        result.correlation.iter().map(|row| Robj::from(row.clone())).collect::<Vec<Robj>>(),
    );

    Ok(list!(
        theta = result.theta,
        omega = result.omega,
        eta = RMatrix::new_matrix(n_sub, n_params, |r, c| eta_flat[r * n_params + c]),
        ofv = result.ofv,
        converged = result.converged,
        n_iter = result.n_iter as i32,
        correlation = corr
    ))
}

// ---------------------------------------------------------------------------
// ns_saem — SAEM population PK estimation
// ---------------------------------------------------------------------------

#[extendr]
fn ns_saem(
    times: Vec<f64>,
    dv: Vec<f64>,
    id: Vec<i32>,
    n_subjects: i32,
    dose: f64,
    bioav: f64,
    error_model: &str,
    sigma: Vec<f64>,
    theta_init: Vec<f64>,
    omega_init: Vec<f64>,
    n_burn: i32,
    n_iter: i32,
    n_chains: i32,
    seed: i32,
    tol: f64,
) -> extendr_api::Result<List> {
    setup_runtime(1);
    let em = parse_error_model_r(error_model, &sigma)?;
    let subject_idx: Vec<usize> = id.iter().map(|&i| i as usize).collect();

    let config = ns_inference::saem::SaemConfig {
        n_burn: n_burn as usize,
        n_iter: n_iter as usize,
        n_chains: n_chains as usize,
        seed: seed as u64,
        tol,
        ..ns_inference::saem::SaemConfig::default()
    };
    let estimator = ns_inference::saem::SaemEstimator::new(config);
    let (result, diagnostics) = estimator
        .fit_1cpt_oral(&times, &dv, &subject_idx, n_subjects as usize, dose, bioav, em, &theta_init, &omega_init)
        .map_err(|e| Error::Other(format!("SAEM failed: {e}")))?;

    let eta_flat: Vec<f64> = result.eta.iter().flat_map(|row| row.iter().copied()).collect();
    let n_sub = result.eta.len();
    let n_params = if n_sub > 0 { result.eta[0].len() } else { 3 };

    Ok(list!(
        theta = result.theta,
        omega = result.omega,
        eta = RMatrix::new_matrix(n_sub, n_params, |r, c| eta_flat[r * n_params + c]),
        ofv = result.ofv,
        converged = result.converged,
        n_iter = result.n_iter as i32,
        acceptance_rates = diagnostics.acceptance_rates,
        ofv_trace = diagnostics.ofv_trace
    ))
}

// ---------------------------------------------------------------------------
// ns_vpc — Visual Predictive Check
// ---------------------------------------------------------------------------

#[extendr]
fn ns_vpc(
    times: Vec<f64>,
    dv: Vec<f64>,
    id: Vec<i32>,
    n_subjects: i32,
    dose: f64,
    theta: Vec<f64>,
    omega: Vec<f64>,
    sigma: f64,
    n_rep: i32,
    n_bins: i32,
) -> extendr_api::Result<List> {
    setup_runtime(1);
    let subject_idx: Vec<usize> = id.iter().map(|&i| i as usize).collect();
    let om = ns_inference::foce::OmegaMatrix::from_diagonal(&omega)
        .map_err(|e| Error::Other(format!("omega: {e}")))?;
    let em = ns_inference::pk::ErrorModel::Additive(sigma);
    let config = ns_inference::VpcConfig {
        n_sim: n_rep as usize,
        n_bins: n_bins as usize,
        quantiles: vec![0.05, 0.50, 0.95],
        seed: 42,
        pi_level: 0.90,
    };
    let result = ns_inference::vpc_1cpt_oral(
        &times, &dv, &subject_idx, n_subjects as usize, dose, 1.0, &theta, &om, &em, &config,
    )
    .map_err(|e| Error::Other(format!("VPC failed: {e}")))?;

    // Convert bins to R list of lists.
    let bins: List = List::from_values(
        result.bins.iter().map(|b| {
            list!(
                time = b.time,
                n_obs = b.n_obs as i32,
                obs_quantiles = b.obs_quantiles.clone(),
                sim_pi_lower = b.sim_pi_lower.clone(),
                sim_pi_median = b.sim_pi_median.clone(),
                sim_pi_upper = b.sim_pi_upper.clone()
            ).into_robj()
        }).collect::<Vec<Robj>>(),
    );

    Ok(list!(
        bins = bins,
        quantiles = result.quantiles,
        n_sim = result.n_sim as i32
    ))
}

// ---------------------------------------------------------------------------
// ns_gof — Goodness-of-Fit diagnostics
// ---------------------------------------------------------------------------

#[extendr]
fn ns_gof(
    times: Vec<f64>,
    dv: Vec<f64>,
    id: Vec<i32>,
    _n_subjects: i32,
    dose: f64,
    theta: Vec<f64>,
    _omega: Vec<f64>,
    eta: RMatrix<f64>,
    sigma: f64,
) -> extendr_api::Result<List> {
    setup_runtime(1);
    let subject_idx: Vec<usize> = id.iter().map(|&i| i as usize).collect();
    let em = ns_inference::pk::ErrorModel::Additive(sigma);

    // Convert R matrix to Vec<Vec<f64>>.
    let n_sub = eta.nrows();
    let n_par = eta.ncols();
    let mut etas: Vec<Vec<f64>> = Vec::with_capacity(n_sub);
    for i in 0..n_sub {
        let mut row = Vec::with_capacity(n_par);
        for j in 0..n_par {
            row.push(eta[[i, j]]);
        }
        etas.push(row);
    }

    let records = ns_inference::gof_1cpt_oral(
        &times, &dv, &subject_idx, dose, 1.0, &theta, &etas, &em,
    )
    .map_err(|e| Error::Other(format!("GOF failed: {e}")))?;

    // Return as parallel vectors (R data.frame-friendly).
    let mut r_time = Vec::with_capacity(records.len());
    let mut r_dv = Vec::with_capacity(records.len());
    let mut r_pred = Vec::with_capacity(records.len());
    let mut r_ipred = Vec::with_capacity(records.len());
    let mut r_iwres = Vec::with_capacity(records.len());
    let mut r_cwres = Vec::with_capacity(records.len());
    let mut r_subject = Vec::with_capacity(records.len());

    for rec in &records {
        r_subject.push(rec.subject as i32);
        r_time.push(rec.time);
        r_dv.push(rec.dv);
        r_pred.push(rec.pred);
        r_ipred.push(rec.ipred);
        r_iwres.push(rec.iwres);
        r_cwres.push(rec.cwres);
    }

    Ok(list!(
        SUBJECT = r_subject,
        TIME = r_time,
        DV = r_dv,
        PRED = r_pred,
        IPRED = r_ipred,
        IWRES = r_iwres,
        CWRES = r_cwres
    ))
}

// ---------------------------------------------------------------------------
// ns_idr — Indirect Response Model simulation
// ---------------------------------------------------------------------------

#[extendr]
fn ns_idr(
    conc_times: Vec<f64>,
    conc_values: Vec<f64>,
    output_times: Vec<f64>,
    idr_type: &str,
    kin: f64,
    kout: f64,
    max_effect: f64,
    c50: f64,
    r0: f64,
) -> extendr_api::Result<Vec<f64>> {
    use ns_inference::pd::{IndirectResponseModel, IndirectResponseType};

    let ty = match idr_type {
        "inhibit_production" => IndirectResponseType::InhibitProduction,
        "inhibit_loss" => IndirectResponseType::InhibitLoss,
        "stimulate_production" => IndirectResponseType::StimulateProduction,
        "stimulate_loss" => IndirectResponseType::StimulateLoss,
        _ => return Err(Error::Other(format!("unknown IDR type: {idr_type}"))),
    };

    let model = IndirectResponseModel::new(ty, kin, kout, max_effect, c50)
        .map_err(|e| Error::Other(format!("IDR model: {e}")))?;

    let conc_profile: Vec<(f64, f64)> = conc_times
        .into_iter()
        .zip(conc_values.into_iter())
        .collect();

    model
        .simulate(&conc_profile, &output_times, Some(r0), None)
        .map_err(|e| Error::Other(format!("IDR simulate: {e}")))
}

// ---------------------------------------------------------------------------
// ns_scm — Stepwise Covariate Modeling (stub — requires complex infrastructure)
// ---------------------------------------------------------------------------

#[extendr]
fn ns_scm(
    _times: Vec<f64>,
    _dv: Vec<f64>,
    _id: Vec<i32>,
    _n_subjects: i32,
    _dose: f64,
    _theta: Vec<f64>,
    _omega: Vec<f64>,
    _sigma: f64,
    _covariates: List,
    _alpha_forward: f64,
    _alpha_backward: f64,
) -> extendr_api::Result<List> {
    Err(Error::Other("ns_scm is not yet implemented in Rust. Use R-native SCM packages (e.g., Pmx, nlme) for covariate modeling.".into()))
}

extendr_module! {
    mod nextstat;
    fn ns_normal_logpdf;
    fn ns_ols_fit;
    fn nextstat_fit;
    fn nextstat_hypotest;
    fn nextstat_upper_limit;
    fn nextstat_glm_logistic;
    fn nextstat_glm_poisson;
    fn nextstat_glm_negbin;
    fn nextstat_kalman;
    fn nextstat_garch;
    fn nextstat_sv;
    fn ns_foce;
    fn ns_saem;
    fn ns_vpc;
    fn ns_gof;
    fn ns_idr;
    fn ns_scm;
}

/// R looks for `R_init_<package>` when loading a package shared library.
///
/// `extendr_module!` generates `R_init_<package>_extendr`, so we provide a thin
/// alias with the expected name.
///
/// # Safety
///
/// `info` must be a valid, non-null pointer to `DllInfo` provided by R during
/// package loading. This function must only be called by R's dynamic loader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn R_init_nextstat(info: *mut extendr_api::DllInfo) {
    R_init_nextstat_extendr(info);
}
