//! Maximum Likelihood Estimation

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizerConfig};
use nalgebra::DMatrix;
use ns_core::traits::{LogDensityModel, PreparedNll};
use ns_core::{FitResult, Result};
use ns_translate::pyhf::HistFactoryModel;

/// Maximum Likelihood Estimator
///
/// Fits statistical models by minimizing negative log-likelihood.
#[derive(Clone)]
pub struct MaximumLikelihoodEstimator {
    config: OptimizerConfig,
}

impl MaximumLikelihoodEstimator {
    /// Create a new MLE with default configuration
    pub fn new() -> Self {
        Self { config: OptimizerConfig::default() }
    }

    /// Create MLE with custom optimizer configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Fit any [`LogDensityModel`] by minimizing negative log-likelihood.
    ///
    /// # Arguments
    /// * `model` - Statistical model to fit
    ///
    /// # Returns
    /// FitResult with best-fit parameters, uncertainties, covariance, and fit quality
    pub fn fit<M: LogDensityModel>(&self, model: &M) -> Result<FitResult> {
        let result = self.fit_minimum(model)?;

        // Compute full Hessian and covariance matrix
        let hessian = self.compute_hessian(model, &result.parameters)?;
        let n = result.parameters.len();
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                // Uncertainties from diagonal of covariance matrix
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }

                if all_variances_ok {
                    // Store covariance as row-major flat Vec
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();

                    Ok(FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    ))
                } else {
                    log::warn!("Invalid covariance diagonal; omitting covariance matrix");
                    Ok(FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    ))
                }
            }
            None => {
                // Hessian inversion failed; fall back to diagonal estimate
                log::warn!("Hessian inversion failed, using diagonal approximation");
                let uncertainties = self.diagonal_uncertainties(&hessian, n);
                Ok(FitResult::new(
                    result.parameters,
                    uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                ))
            }
        }
    }

    /// Minimize NLL and return the optimizer result.
    ///
    /// Fast path: does not compute Hessian/covariance. Intended for repeated minimizations
    /// (profile likelihood scans, hypotest/limits).
    pub fn fit_minimum(
        &self,
        model: &impl LogDensityModel,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let initial_params: Vec<f64> = model.parameter_init();
        self.fit_minimum_from(model, &initial_params)
    }

    /// Minimize NLL from an explicit starting point (warm-start).
    ///
    /// This is important for profile scans / CLs scans where consecutive points
    /// are highly correlated and re-starting from `parameter_init()` is slow.
    pub fn fit_minimum_from(
        &self,
        model: &impl LogDensityModel,
        initial_params: &[f64],
    ) -> Result<crate::optimizer::OptimizationResult> {
        if initial_params.len() != model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "fit_minimum_from: initial_params length {} != model.dim() {}",
                initial_params.len(),
                model.dim()
            )));
        }
        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        let prepared = model.prepared();

        struct ModelObjective<'a, M: LogDensityModel + ?Sized, P: PreparedNll> {
            prepared: P,
            model: &'a M,
        }

        impl<M: LogDensityModel + ?Sized, P: PreparedNll> ObjectiveFunction for ModelObjective<'_, M, P> {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                self.prepared.nll(params)
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                self.model.grad_nll(params)
            }
        }

        let objective = ModelObjective { prepared, model };
        let optimizer = LbfgsbOptimizer::new(self.config.clone());
        optimizer.minimize(&objective, initial_params, &bounds)
    }

    /// Compute full Hessian matrix using forward differences of analytical gradient.
    ///
    /// H_{ij} = (∂g_i/∂x_j) ≈ (g_i(x + ε·e_j) − g_i(x)) / ε
    ///
    /// Cost: N+1 gradient evaluations (each O(1) via reverse-mode AD).
    fn compute_hessian(
        &self,
        model: &impl LogDensityModel,
        best_params: &[f64],
    ) -> Result<DMatrix<f64>> {
        let n = best_params.len();
        let grad_center = model.grad_nll(best_params)?;

        let mut hessian = DMatrix::zeros(n, n);

        for j in 0..n {
            let eps = 1e-4 * best_params[j].abs().max(1.0);

            let mut params_plus = best_params.to_vec();
            params_plus[j] += eps;
            let grad_plus = model.grad_nll(&params_plus)?;

            for i in 0..n {
                hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
            }
        }

        // Symmetrise: H = (H + H^T) / 2
        let ht = hessian.transpose();
        hessian = (&hessian + &ht) * 0.5;

        Ok(hessian)
    }

    /// Invert Hessian to get covariance matrix via Cholesky decomposition.
    ///
    /// Returns `None` if the Hessian is not positive definite.
    fn invert_hessian(&self, hessian: &DMatrix<f64>, n: usize) -> Option<DMatrix<f64>> {
        // We want a positive-(semi)definite covariance; even at a valid minimum the
        // numerically estimated Hessian can be slightly indefinite. Prefer a damped
        // Cholesky solve to avoid negative variances (which then explode to 1e6).
        let identity = DMatrix::identity(n, n);

        // Scale damping to the Hessian diagonal to be unit-ish across models.
        let diag_scale = (0..n).map(|i| hessian[(i, i)].abs()).fold(0.0_f64, f64::max).max(1.0);

        let mut h_damped = hessian.clone();
        let mut damping = 0.0_f64;
        let max_attempts = 10;

        for attempt in 0..max_attempts {
            if let Some(chol) = nalgebra::linalg::Cholesky::new(h_damped.clone()) {
                return Some(chol.solve(&identity));
            }

            if attempt + 1 == max_attempts {
                break;
            }

            // Increase diagonal damping geometrically.
            let next_damping = if damping == 0.0 { diag_scale * 1e-9 } else { damping * 10.0 };
            let add = next_damping - damping;
            for i in 0..n {
                h_damped[(i, i)] += add;
            }
            damping = next_damping;
        }

        let cov = h_damped.lu().try_inverse()?;
        // Reject clearly-bad inverses (negative/NaN variances).
        for i in 0..n {
            let v = cov[(i, i)];
            if !(v.is_finite() && v > 0.0) {
                return None;
            }
        }
        Some(cov)
    }

    /// Extract uncertainties from Hessian diagonal (fallback).
    fn diagonal_uncertainties(&self, hessian: &DMatrix<f64>, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let hess_ii = hessian[(i, i)];
                let denom = hess_ii.abs().max(1e-12);
                1.0 / denom.sqrt()
            })
            .collect()
    }

    /// Run multiple independent fits in parallel using Rayon.
    ///
    /// Returns one `FitResult` per model.
    pub fn fit_batch<M: LogDensityModel + Sync>(&self, models: &[M]) -> Vec<Result<FitResult>> {
        use rayon::prelude::*;

        models.par_iter().map(|model| self.fit(model)).collect()
    }

    /// Generate toy pseudo-experiments and fit each one.
    ///
    /// # Arguments
    /// * `model` - Base model (used for expected data and parameter structure)
    /// * `params` - Parameters to generate toys at (e.g., best-fit or Asimov)
    /// * `n_toys` - Number of pseudo-experiments
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// Vector of fit results, one per toy
    pub fn fit_toys(
        &self,
        model: &HistFactoryModel,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> Vec<Result<FitResult>> {
        use rayon::prelude::*;

        // Generate expected main data at given parameters (pyhf ordering)
        let expected = match model.expected_data_pyhf_main(params) {
            Ok(e) => e,
            Err(e) => return vec![Err(e)],
        };

        // Generate toy datasets in parallel with deterministic seeds
        (0..n_toys)
            .into_par_iter()
            .map(|toy_idx| {
                let toy_seed = seed.wrapping_add(toy_idx as u64);
                let toy_data = crate::toys::poisson_main_from_expected(&expected, toy_seed);

                // Create toy model with fluctuated data
                let toy_model = model.with_observed_main(&toy_data)?;
                self.fit(&toy_model)
            })
            .collect()
    }

    /// Compute ranking: impact of each nuisance parameter on POI.
    ///
    /// For each NP, fixes it at ±1σ and re-fits. The shift in POI
    /// measures the NP's impact.
    ///
    /// # Returns
    /// Vector of `(param_name, delta_mu_up, delta_mu_down, pull, constraint)`
    /// sorted by |impact| descending.
    pub fn ranking(&self, model: &HistFactoryModel) -> Result<Vec<RankingEntry>> {
        use rayon::prelude::*;

        // First: unconditional fit
        let nominal_result = self.fit(model)?;
        let poi_idx = model
            .poi_index()
            .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;
        let mu_hat = nominal_result.parameters[poi_idx];

        // Identify nuisance parameters (all non-POI parameters)
        let np_indices: Vec<usize> = model
            .parameters()
            .iter()
            .enumerate()
            .filter(|(i, p)| *i != poi_idx && p.bounds.0 != p.bounds.1)
            .map(|(i, _)| i)
            .collect();

        // For each NP, fit with NP fixed at ±1σ
        let entries: Vec<Result<RankingEntry>> = np_indices
            .par_iter()
            .map(|&np_idx| {
                let param = &model.parameters()[np_idx];
                let center = param.constraint_center.unwrap_or(param.init);
                // For Barlow-Beeston gammas: use sqrt(1/tau) ≈ relative uncertainty
                let sigma =
                    param.constraint_width.unwrap_or(if center > 0.0 { 0.1 * center } else { 1.0 });

                // Fix NP at +1σ
                let model_up = model.with_fixed_param(np_idx, center + sigma);
                let result_up = self.fit(&model_up)?;
                let mu_up = result_up.parameters[poi_idx];

                // Fix NP at -1σ
                let model_down = model.with_fixed_param(np_idx, center - sigma);
                let result_down = self.fit(&model_down)?;
                let mu_down = result_down.parameters[poi_idx];

                // Pull: (θ̂ - θ₀) / σ
                let theta_hat = nominal_result.parameters[np_idx];
                let pull = (theta_hat - center) / sigma;

                // Constraint: σ̂ / σ (should be ≤ 1)
                let constraint = nominal_result.uncertainties[np_idx] / sigma;

                Ok(RankingEntry {
                    name: param.name.clone(),
                    delta_mu_up: mu_up - mu_hat,
                    delta_mu_down: mu_down - mu_hat,
                    pull,
                    constraint,
                })
            })
            .collect::<Vec<_>>();

        // Collect results, sort by |impact|
        let mut ranking: Vec<RankingEntry> = entries.into_iter().filter_map(|r| r.ok()).collect();

        ranking.sort_by(|a, b| {
            let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
            let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
            impact_b.partial_cmp(&impact_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ranking)
    }
}

/// Entry in ranking plot: impact of a nuisance parameter on the POI.
#[derive(Debug, Clone)]
pub struct RankingEntry {
    /// Parameter name
    pub name: String,
    /// POI shift when NP fixed at +1σ
    pub delta_mu_up: f64,
    /// POI shift when NP fixed at -1σ
    pub delta_mu_down: f64,
    /// Pull: (θ̂ − θ₀) / σ
    pub pull: f64,
    /// Constraint: σ̂ / σ
    pub constraint: f64,
}

impl Default for MaximumLikelihoodEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_mle_fit_simple() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        println!("Fit result:");
        println!("  Parameters: {:?}", result.parameters);
        println!("  Uncertainties: {:?}", result.uncertainties);
        println!("  NLL: {:.6}", result.nll);
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.n_iter);

        assert!(result.converged, "Fit should converge");

        let poi = result.parameters[0];
        assert!(poi > 0.0 && poi < 2.0, "POI should be reasonable: {}", poi);

        for (i, &unc) in result.uncertainties.iter().enumerate() {
            assert!(unc > 0.0, "Uncertainty[{}] should be positive: {}", i, unc);
            assert!(unc.is_finite(), "Uncertainty[{}] should be finite: {}", i, unc);
            assert!(
                unc < 1e5,
                "Uncertainty[{}] looks like a numerical fallback (too large): {}",
                i,
                unc
            );
        }

        assert!(result.nll > 0.0 && result.nll < 100.0);
    }

    #[test]
    fn test_mle_poi_extraction() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let poi_idx = model.poi_index().expect("POI index should exist");
        assert_eq!(poi_idx, 0, "POI should be first parameter");

        let poi = result.parameters[poi_idx];
        let poi_unc = result.uncertainties[poi_idx];

        println!("Best-fit POI: {} ± {}", poi, poi_unc);

        assert!(poi > 0.0 && poi < 2.0);
        assert!(poi_unc > 0.0 && poi_unc < 1.0);
    }

    #[test]
    fn test_mle_covariance_matrix() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let n = result.parameters.len();

        // Covariance matrix should be present
        assert!(result.covariance.is_some(), "Covariance matrix should be computed");
        let cov = result.covariance.as_ref().unwrap();
        assert_eq!(cov.len(), n * n, "Covariance should be N×N");

        println!("Covariance matrix ({}×{}):", n, n);
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| format!("{:+.4e}", cov[i * n + j])).collect();
            println!("  [{}]", row.join(", "));
        }

        // Diagonal elements should be positive (variances)
        for i in 0..n {
            let var_i = cov[i * n + i];
            assert!(var_i > 0.0, "Variance[{}] should be positive: {}", i, var_i);
        }

        // Diagonal should match uncertainties^2
        for i in 0..n {
            let var_from_cov = cov[i * n + i];
            let unc_sq = result.uncertainties[i].powi(2);
            let rel_diff = ((var_from_cov - unc_sq) / unc_sq).abs();
            assert!(
                rel_diff < 1e-10,
                "Cov diagonal[{}] should match uncertainty²: cov={}, unc²={}",
                i,
                var_from_cov,
                unc_sq
            );
        }

        // Matrix should be approximately symmetric (it is by construction)
        for i in 0..n {
            for j in 0..n {
                let diff = (cov[i * n + j] - cov[j * n + i]).abs();
                assert!(diff < 1e-15, "Covariance should be symmetric: [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn test_mle_correlations() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let n = result.parameters.len();

        println!("Correlation matrix:");
        for i in 0..n {
            let row: Vec<String> = (0..n)
                .map(|j| {
                    let rho = result.correlation(i, j).unwrap();
                    format!("{:+.4}", rho)
                })
                .collect();
            println!("  [{}]", row.join(", "));
        }

        // Diagonal correlations should be 1.0
        for i in 0..n {
            let rho_ii = result.correlation(i, i).unwrap();
            assert!(
                (rho_ii - 1.0).abs() < 1e-10,
                "Diagonal correlation[{}] should be 1.0: {}",
                i,
                rho_ii
            );
        }

        // All correlations should be in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                let rho = result.correlation(i, j).unwrap();
                assert!(
                    (-1.0 - 1e-10..=1.0 + 1e-10).contains(&rho),
                    "Correlation[{},{}] out of range: {}",
                    i,
                    j,
                    rho
                );
            }
        }
    }

    #[test]
    fn test_hessian_computation() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        // Compute Hessian at best-fit point
        let hessian = mle.compute_hessian(&model, &result.parameters).unwrap();
        let n = result.parameters.len();

        println!("Hessian matrix ({}×{}):", n, n);
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| format!("{:+.4e}", hessian[(i, j)])).collect();
            println!("  [{}]", row.join(", "));
        }

        // Hessian should be symmetric
        for i in 0..n {
            for j in i + 1..n {
                let diff = (hessian[(i, j)] - hessian[(j, i)]).abs();
                let scale = hessian[(i, j)].abs().max(1e-10);
                assert!(
                    diff / scale < 1e-6,
                    "Hessian not symmetric: H[{},{}]={}, H[{},{}]={}",
                    i,
                    j,
                    hessian[(i, j)],
                    j,
                    i,
                    hessian[(j, i)]
                );
            }
        }

        // Diagonal should be positive (at a minimum, Hessian is positive definite)
        for i in 0..n {
            assert!(
                hessian[(i, i)] > 0.0,
                "Hessian diagonal[{}] should be positive: {}",
                i,
                hessian[(i, i)]
            );
        }
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_fit_toys -- --ignored`"]
    fn test_fit_toys() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let nominal = mle.fit(&model).unwrap();

        // Run 20 toy fits
        let n_toys = 20;
        let results = mle.fit_toys(&model, &nominal.parameters, n_toys, 42);

        assert_eq!(results.len(), n_toys);

        let mut converged = 0;
        let mut poi_values = Vec::new();
        for result in &results {
            match result {
                Ok(r) => {
                    if r.converged {
                        converged += 1;
                        poi_values.push(r.parameters[0]);
                    }
                }
                Err(e) => println!("Toy failed: {}", e),
            }
        }

        println!("Toys: {}/{} converged", converged, n_toys);
        println!("POI values: {:?}", poi_values);

        // Most toys should converge
        assert!(converged >= n_toys / 2, "Too few toys converged: {}/{}", converged, n_toys);

        // POI should scatter around nominal value
        let poi_mean: f64 = poi_values.iter().sum::<f64>() / poi_values.len() as f64;
        println!("Mean POI: {:.4} (nominal: {:.4})", poi_mean, nominal.parameters[0]);
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_fit_toys_reproducible -- --ignored`"]
    fn test_fit_toys_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let mle = MaximumLikelihoodEstimator::new();

        // Same seed => same results
        let results1 = mle.fit_toys(&model, &params, 5, 123);
        let results2 = mle.fit_toys(&model, &params, 5, 123);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            if let (Ok(a), Ok(b)) = (r1, r2) {
                assert_eq!(a.parameters, b.parameters, "Toys should be reproducible");
            }
        }
    }

    #[test]
    fn test_fit_toys_smoke_fast_and_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let mle = MaximumLikelihoodEstimator::new();

        // Keep this test fast: 2 toys only.
        let results1 = mle.fit_toys(&model, &params, 2, 123);
        let results2 = mle.fit_toys(&model, &params, 2, 123);

        assert_eq!(results1.len(), 2);
        assert_eq!(results2.len(), 2);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            match (r1, r2) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.parameters, b.parameters, "Toy best-fits should be reproducible");
                    assert!(a.nll.is_finite(), "Toy NLL should be finite");
                    assert!(b.nll.is_finite(), "Toy NLL should be finite");
                }
                (Err(e1), Err(e2)) => {
                    // If both fail, at least ensure deterministic failure mode.
                    assert_eq!(e1.to_string(), e2.to_string());
                }
                _ => panic!("Toy results should be deterministically Ok/Err for a fixed seed"),
            }
        }
    }

    #[test]
    #[ignore = "very slow (~10min release, argmin+tape AD); run with `cargo test -p ns-inference --release test_fit_toys_pull_distribution -- --ignored`"]
    fn test_fit_toys_pull_distribution() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Poisson};

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let poi_idx = model.poi_index().expect("POI index should exist");
        let mu_true = 1.0;

        // Generate at POI = mu_true, nuisances at suggested init
        let mut truth: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        truth[poi_idx] = mu_true;

        let expected = model.expected_data(&truth).unwrap();

        // Cap iterations to bound runtime on pathological toys
        let config = OptimizerConfig { max_iter: 50, ..OptimizerConfig::default() };
        let mle = MaximumLikelihoodEstimator::with_config(config);
        let n_toys = 100;
        let seed = 0u64;

        let mut pulls = Vec::new();
        let mut n_converged = 0usize;
        let mut n_covered = 0usize;

        for toy_idx in 0..n_toys {
            let toy_seed = seed.wrapping_add(toy_idx as u64);
            let mut rng = rand::rngs::StdRng::seed_from_u64(toy_seed);

            let toy_data: Vec<f64> = expected
                .iter()
                .map(|&lam| {
                    let pois = Poisson::new(lam.max(1e-10)).unwrap();
                    pois.sample(&mut rng)
                })
                .collect();

            let toy_model = match model.with_observed_main(&toy_data) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let fit = match mle.fit(&toy_model) {
                Ok(f) => f,
                Err(_) => continue,
            };

            if !fit.converged {
                continue;
            }
            n_converged += 1;
            let mu_hat = fit.parameters[poi_idx];
            let sigma_mu = fit.uncertainties[poi_idx];
            if sigma_mu <= 0.0 || !sigma_mu.is_finite() {
                continue;
            }
            let pull = (mu_hat - mu_true) / sigma_mu;
            pulls.push(pull);
            if pull.abs() <= 1.0 {
                n_covered += 1;
            }
        }

        let n = pulls.len() as f64;
        assert!(n >= 50.0, "Need at least 50 converged toys, got {}", n as usize);

        let mean: f64 = pulls.iter().sum::<f64>() / n;
        let var: f64 = pulls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let coverage = n_covered as f64 / pulls.len() as f64;

        // Print JSON summary for CI capture
        println!(
            "{{\"test\":\"pull_distribution\",\"n_toys\":{},\"n_converged\":{},\"n_pulls\":{},\
             \"pull_mean\":{:.4},\"pull_std\":{:.4},\"coverage_1sigma\":{:.4}}}",
            n_toys,
            n_converged,
            pulls.len(),
            mean,
            std,
            coverage
        );

        assert!(mean.abs() < 0.15, "Pull mean should be near 0: {:.4}", mean);
        assert!((std - 1.0).abs() < 0.15, "Pull std should be near 1: {:.4}", std);
        assert!((coverage - 0.68).abs() < 0.08, "1σ coverage should be near 68%: {:.4}", coverage);
    }

    #[test]
    fn test_ranking() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let ranking = mle.ranking(&model).unwrap();

        println!("Ranking ({} NPs):", ranking.len());
        for entry in &ranking {
            println!(
                "  {}: Δμ_up={:+.4}, Δμ_down={:+.4}, pull={:+.3}, constraint={:.3}",
                entry.name, entry.delta_mu_up, entry.delta_mu_down, entry.pull, entry.constraint
            );
        }

        // Should have entries for constrained NPs (not POI)
        assert!(!ranking.is_empty(), "Should have ranking entries");

        // All entries should have finite values
        for entry in &ranking {
            assert!(entry.delta_mu_up.is_finite(), "delta_mu_up should be finite");
            assert!(entry.delta_mu_down.is_finite(), "delta_mu_down should be finite");
            assert!(entry.pull.is_finite(), "pull should be finite");
            assert!(entry.constraint.is_finite(), "constraint should be finite");
            assert!(entry.constraint > 0.0, "constraint should be positive");
        }
    }

    #[test]
    fn test_diagonal_uncertainties_abs_diag() {
        let mle = MaximumLikelihoodEstimator::new();
        let h = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_vec(vec![-4.0, 9.0]));
        let u = mle.diagonal_uncertainties(&h, 2);
        assert!((u[0] - 0.5).abs() < 1e-12);
        assert!((u[1] - (1.0 / 3.0)).abs() < 1e-12);
    }
}
