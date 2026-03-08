//! Internal CUDA-backed HamiltonianPotential prototypes for linear-predictor targets.
//!
//! This is a seam-validation slice only. It intentionally does not expose a
//! public sampler API and currently supports:
//! - CUDA runtime only
//! - linear regression with fixed sigma=1 and a standard normal prior on all
//!   coefficients
//! - logistic regression with a standard normal prior on all coefficients
//! - Poisson regression with optional offsets and with a standard normal prior
//!   on all coefficients
//! - Negative Binomial regression with optional offsets and with a standard
//!   normal prior on all coefficients, including `log_alpha`
//! - interval-censored Weibull AFT survival with covariates and a standard
//!   normal prior on `[log_k, beta...]`
//! - diagonal Euclidean metric on the host-side leapfrog integrator

#![cfg(feature = "cuda")]

use crate::hmc::{HamiltonianPotential, HmcState, LeapfrogIntegrator, Metric};
use crate::posterior::{Posterior, Prior};
use crate::regression::{
    LinearRegressionModel, LogisticRegressionModel, NegativeBinomialRegressionModel,
    PoissonRegressionModel,
};
use crate::survival::{CensoringType, IntervalCensoredWeibullAftModel};
use crate::transforms::ParameterTransform;
use crate::walnuts::{WalnutsConfig, walnuts_transition};
use ns_compute::cuda_glm_cublas::{CudaGlmCublasEvaluator, CudaGlmFamily};
use ns_compute::cuda_weibull_aft_cublas::CudaWeibullAftCublasEvaluator;
use ns_core::traits::LogDensityModel;
use ns_core::{Error, Result};
use rand::SeedableRng;
use serde::Serialize;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;

/// CUDA-backed potential evaluator for supported GLM families with `N(0, 1)` priors.
pub(crate) struct CudaStdNormalPriorGlmPotential {
    dim: usize,
    family_name: &'static str,
    transform: ParameterTransform,
    eval: Mutex<CudaGlmCublasEvaluator>,
}

impl CudaStdNormalPriorGlmPotential {
    pub(crate) fn is_available() -> bool {
        std::panic::catch_unwind(|| ns_compute::cuda_driver::CudaContext::new(0).is_ok())
            .unwrap_or(false)
    }

    pub(crate) fn new_logistic_on_device(
        model: &LogisticRegressionModel,
        device_id: usize,
    ) -> Result<Self> {
        let (x_col, y, n, p_total) = model.cuda_glm_design_colmajor();
        let eval = CudaGlmCublasEvaluator::new_on_device_with_family(
            &x_col,
            &y,
            n,
            p_total,
            1,
            CudaGlmFamily::Logistic,
            device_id,
        )?;
        let transform = ParameterTransform::from_bounds(&model.parameter_bounds());
        Ok(Self { dim: p_total, family_name: "logistic", transform, eval: Mutex::new(eval) })
    }

    pub(crate) fn new_linear_on_device(
        model: &LinearRegressionModel,
        device_id: usize,
    ) -> Result<Self> {
        let (x_col, y, n, p_total) = model.cuda_glm_design_colmajor();
        let eval = CudaGlmCublasEvaluator::new_on_device_with_family(
            &x_col,
            &y,
            n,
            p_total,
            1,
            CudaGlmFamily::Linear,
            device_id,
        )?;
        let transform = ParameterTransform::from_bounds(&model.parameter_bounds());
        Ok(Self { dim: p_total, family_name: "linear", transform, eval: Mutex::new(eval) })
    }

    pub(crate) fn new_poisson_on_device(
        model: &PoissonRegressionModel,
        device_id: usize,
    ) -> Result<Self> {
        let (x_col, y, offset, n, p_total) = model.cuda_glm_design_colmajor()?;
        let family_name = if offset.is_some() { "poisson_with_offset" } else { "poisson" };
        let eval = CudaGlmCublasEvaluator::new_on_device_with_family_and_offset(
            &x_col,
            &y,
            offset.as_deref(),
            n,
            p_total,
            1,
            CudaGlmFamily::Poisson,
            device_id,
        )?;
        let transform = ParameterTransform::from_bounds(&model.parameter_bounds());
        Ok(Self { dim: p_total, family_name, transform, eval: Mutex::new(eval) })
    }

    pub(crate) fn new_negbin_on_device(
        model: &NegativeBinomialRegressionModel,
        device_id: usize,
    ) -> Result<Self> {
        let (x_col, y, offset, n, beta_dim, param_dim) = model.cuda_glm_design_colmajor();
        let family_name = if offset.is_some() { "negbin_with_offset" } else { "negbin" };
        let eval = CudaGlmCublasEvaluator::new_on_device_with_family_and_offset(
            &x_col,
            &y,
            offset.as_deref(),
            n,
            beta_dim,
            1,
            CudaGlmFamily::NegativeBinomial,
            device_id,
        )?;
        let transform = ParameterTransform::from_bounds(&model.parameter_bounds());
        Ok(Self { dim: param_dim, family_name, transform, eval: Mutex::new(eval) })
    }
}

impl HamiltonianPotential for CudaStdNormalPriorGlmPotential {
    fn dim(&self) -> usize {
        self.dim
    }

    fn potential_grad(&self, q: &[f64]) -> Result<(f64, Vec<f64>)> {
        if q.len() != self.dim {
            return Err(Error::Validation(format!(
                "CUDA {} potential expected {} parameters, got {}",
                self.family_name,
                self.dim,
                q.len()
            )));
        }
        let (theta, jac_diag, grad_log_jac, log_jac) = if self.transform.is_all_identity() {
            (q.to_vec(), Vec::new(), Vec::new(), 0.0)
        } else {
            (
                self.transform.forward(q),
                self.transform.jacobian_diag(q),
                self.transform.grad_log_abs_det_jacobian(q),
                self.transform.log_abs_det_jacobian(q),
            )
        };
        let mut eval = self.eval.lock().map_err(|_| {
            Error::Computation(format!("CUDA {} potential lock poisoned", self.family_name))
        })?;
        let (mut grad_theta, nll) = eval.evaluate_host(&theta)?;
        if self.transform.is_all_identity() {
            return Ok((nll[0], grad_theta));
        }
        for ((g, jd), glj) in grad_theta.iter_mut().zip(jac_diag).zip(grad_log_jac) {
            *g = *g * jd - glj;
        }
        Ok((nll[0] - log_jac, grad_theta))
    }
}

/// CUDA-backed potential evaluator for interval-censored Weibull AFT targets
/// with `N(0, 1)` priors on `[log_k, beta...]`.
pub(crate) struct CudaStdNormalPriorIcWeibullAftPotential {
    dim: usize,
    transform: ParameterTransform,
    eval: Mutex<CudaWeibullAftCublasEvaluator>,
}

impl CudaStdNormalPriorIcWeibullAftPotential {
    pub(crate) fn is_available() -> bool {
        CudaStdNormalPriorGlmPotential::is_available()
    }

    pub(crate) fn new_on_device(
        model: &IntervalCensoredWeibullAftModel,
        device_id: usize,
    ) -> Result<Self> {
        let (
            x_col,
            time_lower,
            time_upper,
            ln_time_lower,
            ln_time_upper,
            censor_code,
            n,
            beta_dim,
            dim,
        ) = model.cuda_weibull_aft_design_colmajor();
        let eval = CudaWeibullAftCublasEvaluator::new_on_device(
            &x_col,
            &time_lower,
            &time_upper,
            &ln_time_lower,
            &ln_time_upper,
            &censor_code,
            n,
            beta_dim,
            1,
            device_id,
        )?;
        let transform = ParameterTransform::from_bounds(&model.parameter_bounds());
        Ok(Self { dim, transform, eval: Mutex::new(eval) })
    }
}

impl HamiltonianPotential for CudaStdNormalPriorIcWeibullAftPotential {
    fn dim(&self) -> usize {
        self.dim
    }

    fn potential_grad(&self, q: &[f64]) -> Result<(f64, Vec<f64>)> {
        if q.len() != self.dim {
            return Err(Error::Validation(format!(
                "CUDA weibull_ic_aft potential expected {} parameters, got {}",
                self.dim,
                q.len()
            )));
        }
        let (theta, jac_diag, grad_log_jac, log_jac) = if self.transform.is_all_identity() {
            (q.to_vec(), Vec::new(), Vec::new(), 0.0)
        } else {
            (
                self.transform.forward(q),
                self.transform.jacobian_diag(q),
                self.transform.grad_log_abs_det_jacobian(q),
                self.transform.log_abs_det_jacobian(q),
            )
        };
        let mut eval = self.eval.lock().map_err(|_| {
            Error::Computation("CUDA weibull_ic_aft potential lock poisoned".to_string())
        })?;
        let (mut grad_theta, nll) = eval.evaluate_host(&theta)?;
        if self.transform.is_all_identity() {
            return Ok((nll[0], grad_theta));
        }
        for ((g, jd), glj) in grad_theta.iter_mut().zip(jac_diag).zip(grad_log_jac) {
            *g = *g * jd - glj;
        }
        Ok((nll[0] - log_jac, grad_theta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_model(n: usize, p: usize, include_intercept: bool) -> LinearRegressionModel {
        let mut x = vec![vec![0.0; p]; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut eta = if include_intercept { 0.15 } else { 0.0 };
            for j in 0..p {
                let value = (((i + 2) * 19 + (j + 1) * 13) as f64 * 0.012).sin();
                x[i][j] = value;
                eta += value * (((j + 1) as f64) * 0.05).cos();
            }
            y[i] = eta + (((i + 1) as f64) * 0.03).sin() * 0.15;
        }
        LinearRegressionModel::new(x, y, include_intercept).unwrap()
    }

    fn logistic_model(n: usize, p: usize, include_intercept: bool) -> LogisticRegressionModel {
        let mut x = vec![vec![0.0; p]; n];
        let mut y = vec![0u8; n];
        for i in 0..n {
            let mut eta = if include_intercept { -0.25 } else { 0.0 };
            for j in 0..p {
                let value = (((i + 1) * 17 + (j + 3) * 29) as f64 * 0.013).sin();
                x[i][j] = value;
                eta += value * (((j + 1) as f64) * 0.09).cos();
            }
            y[i] = if eta > 0.0 { 1 } else { 0 };
        }
        LogisticRegressionModel::new(x, y, include_intercept).unwrap()
    }

    fn poisson_model(
        n: usize,
        p: usize,
        include_intercept: bool,
        with_offset: bool,
    ) -> PoissonRegressionModel {
        let mut x = vec![vec![0.0; p]; n];
        let mut y = vec![0u64; n];
        let offset = if with_offset {
            Some((0..n).map(|i| (((i + 11) as f64) * 0.19).sin() * 0.25).collect::<Vec<_>>())
        } else {
            None
        };
        for i in 0..n {
            let mut eta = if include_intercept { 0.35 } else { 0.0 };
            for j in 0..p {
                let value = (((i + 5) * 13 + (j + 7) * 19) as f64 * 0.011).cos();
                x[i][j] = value;
                eta += value * (((j + 1) as f64) * 0.07).sin();
            }
            if let Some(offset) = &offset {
                eta += offset[i];
            }
            let mu = eta.clamp(-2.0, 2.0).exp();
            y[i] = mu.round().max(0.0) as u64;
        }
        PoissonRegressionModel::new(x, y, include_intercept, offset).unwrap()
    }

    fn negbin_model(
        n: usize,
        p: usize,
        include_intercept: bool,
        with_offset: bool,
    ) -> NegativeBinomialRegressionModel {
        let mut x = vec![vec![0.0; p]; n];
        let offset = if with_offset {
            Some((0..n).map(|i| (((i + 17) as f64) * 0.13).cos() * 0.35).collect::<Vec<_>>())
        } else {
            None
        };
        let mut y = vec![0u64; n];
        for i in 0..n {
            let mut eta = if include_intercept { 0.2 } else { 0.0 };
            for j in 0..p {
                let value = (((i + 3) * 23 + (j + 5) * 11) as f64 * 0.009).sin();
                x[i][j] = value;
                eta += value * (((j + 2) as f64) * 0.11).cos();
            }
            if let Some(offset) = &offset {
                eta += offset[i];
            }
            let mu = eta.clamp(-2.2, 2.2).exp();
            let overdisp = 0.4 * (1.0 + (((i + 1) as f64) * 0.17).sin());
            y[i] =
                (mu * (1.0 + overdisp) + overdisp * mu * mu / (1.0 + mu)).round().max(0.0) as u64;
        }
        NegativeBinomialRegressionModel::new(x, y, include_intercept, offset).unwrap()
    }

    fn weibull_aft_model(n: usize, p: usize) -> IntervalCensoredWeibullAftModel {
        let mut covariates = vec![0.0; n * p];
        let mut time_lower = vec![0.0; n];
        let mut time_upper = vec![0.0; n];
        let mut censor_type = vec![CensoringType::Exact; n];

        for i in 0..n {
            let mut log_lambda = 0.0;
            for j in 0..p {
                let value = (((i + 1) * 19 + (j + 3) * 11) as f64 * 0.009).sin();
                covariates[i * p + j] = value;
                log_lambda += value * (((j + 1) as f64) * 0.025).cos();
            }
            let log_lambda = log_lambda.clamp(-0.6, 0.6);
            let base = (0.15 + 0.04 * ((i as f64) * 0.17).sin() + 0.35 * log_lambda).exp();
            match i % 4 {
                0 => {
                    censor_type[i] = CensoringType::Exact;
                    time_lower[i] = base.max(1e-3);
                    time_upper[i] = time_lower[i];
                }
                1 => {
                    censor_type[i] = CensoringType::Right;
                    time_lower[i] = (base * 0.9).max(0.0);
                    time_upper[i] = (time_lower[i] + 1.0).max(time_lower[i]);
                }
                2 => {
                    censor_type[i] = CensoringType::Left;
                    time_lower[i] = 0.0;
                    time_upper[i] = (base * 1.2 + 0.25).max(1e-3);
                }
                _ => {
                    censor_type[i] = CensoringType::Interval;
                    let lo = if i % 8 == 3 { 0.0 } else { (base * 0.65).max(0.0) };
                    let hi = (base * 1.45 + 0.25).max(lo + 1e-3);
                    time_lower[i] = lo;
                    time_upper[i] = hi;
                }
            }
        }

        IntervalCensoredWeibullAftModel::new(time_lower, time_upper, censor_type, covariates, p)
            .unwrap()
    }

    fn gaussian_prior_posterior<'a, M>(model: &'a M) -> Posterior<'a, M>
    where
        M: LogDensityModel,
    {
        Posterior::new(model)
            .with_priors(vec![Prior::Normal { center: 0.0, width: 1.0 }; model.dim()])
            .unwrap()
    }

    fn seeded_state_from_potential(
        potential: &impl HamiltonianPotential,
        q_shift: f64,
    ) -> HmcState {
        let dim = potential.dim();
        let q = (0..dim).map(|i| ((i as f64 + 1.0) * 0.05) + q_shift).collect::<Vec<_>>();
        let p = (0..dim).map(|i| -0.2 + i as f64 * 0.03).collect::<Vec<_>>();
        let (u, grad_u) = potential.potential_grad(&q).unwrap();
        HmcState { potential: u, grad_potential: grad_u, q, p }
    }

    fn assert_potential_matches_cpu(
        cpu: &impl HamiltonianPotential,
        gpu: &impl HamiltonianPotential,
        q: &[f64],
    ) {
        let (u_cpu, g_cpu) = cpu.potential_grad(q).unwrap();
        let (u_gpu, g_gpu) = gpu.potential_grad(q).unwrap();

        assert!((u_cpu - u_gpu).abs() < 1e-8, "potential mismatch: cpu={u_cpu}, gpu={u_gpu}");
        for (idx, (lhs, rhs)) in g_cpu.iter().zip(&g_gpu).enumerate() {
            assert!((lhs - rhs).abs() < 1e-8, "grad mismatch at {idx}: cpu={lhs}, gpu={rhs}");
        }
    }

    fn assert_leapfrog_matches_cpu(
        cpu: &impl HamiltonianPotential,
        gpu: &impl HamiltonianPotential,
        eps: f64,
    ) {
        let dim = cpu.dim();
        let q = (0..dim).map(|i| (i as f64 + 1.0) * 0.05).collect::<Vec<_>>();
        let p = (0..dim).map(|i| -0.2 + i as f64 * 0.03).collect::<Vec<_>>();
        assert_leapfrog_matches_cpu_with_state(cpu, gpu, eps, &q, &p);
    }

    fn assert_leapfrog_matches_cpu_with_state(
        cpu: &impl HamiltonianPotential,
        gpu: &impl HamiltonianPotential,
        eps: f64,
        q: &[f64],
        p: &[f64],
    ) {
        let metric = Metric::Diag(vec![1.0; cpu.dim()]);
        let cpu_integrator = LeapfrogIntegrator::new(cpu, eps, metric.clone());
        let gpu_integrator = LeapfrogIntegrator::new(gpu, eps, metric);

        let (u_cpu, grad_cpu) = cpu.potential_grad(q).unwrap();
        let (u_gpu, grad_gpu) = gpu.potential_grad(q).unwrap();
        let mut cpu_state =
            HmcState { potential: u_cpu, grad_potential: grad_cpu, q: q.to_vec(), p: p.to_vec() };
        let mut gpu_state =
            HmcState { potential: u_gpu, grad_potential: grad_gpu, q: q.to_vec(), p: p.to_vec() };

        cpu_integrator.step(&mut cpu_state).unwrap();
        gpu_integrator.step(&mut gpu_state).unwrap();

        for (idx, (lhs, rhs)) in cpu_state.q.iter().zip(&gpu_state.q).enumerate() {
            assert!((lhs - rhs).abs() < 1e-8, "q mismatch at {idx}: cpu={lhs}, gpu={rhs}");
        }
        for (idx, (lhs, rhs)) in cpu_state.p.iter().zip(&gpu_state.p).enumerate() {
            assert!((lhs - rhs).abs() < 1e-8, "p mismatch at {idx}: cpu={lhs}, gpu={rhs}");
        }
        assert!(
            (cpu_state.potential - gpu_state.potential).abs() < 1e-8,
            "potential mismatch after leapfrog: cpu={}, gpu={}",
            cpu_state.potential,
            gpu_state.potential
        );
    }

    #[derive(Serialize)]
    struct EvalMetric {
        iterations: usize,
        dim: usize,
        wall_s: f64,
        evals_per_sec: f64,
    }

    #[derive(Serialize)]
    struct TransitionMetric {
        iterations: usize,
        dim: usize,
        wall_s: f64,
        transitions_per_sec: f64,
        total_leapfrogs: usize,
        leapfrogs_per_sec: f64,
    }

    #[derive(Serialize)]
    struct CpuGpuComparison<T> {
        cpu: T,
        gpu: T,
        gpu_over_cpu_throughput: f64,
    }

    #[derive(Serialize)]
    struct CudaLinearPredictorWalnutsBenchReport {
        schema_version: &'static str,
        family: &'static str,
        n_obs: usize,
        n_features: usize,
        potential_grad: CpuGpuComparison<EvalMetric>,
        walnuts_transition: CpuGpuComparison<TransitionMetric>,
    }

    fn bench_cpu_vs_gpu(
        family: &'static str,
        schema_version: &'static str,
        n_obs: usize,
        n_features: usize,
        cpu: &impl HamiltonianPotential,
        gpu: &impl HamiltonianPotential,
    ) -> CudaLinearPredictorWalnutsBenchReport {
        let dim = cpu.dim();
        let q = (0..dim).map(|i| -0.2 + i as f64 * 0.015).collect::<Vec<_>>();
        let eval_iterations = 400usize;

        let cpu_eval_started = Instant::now();
        for _ in 0..eval_iterations {
            black_box(cpu.potential_grad(&q).unwrap());
        }
        let cpu_eval_wall = cpu_eval_started.elapsed().as_secs_f64();

        let gpu_eval_started = Instant::now();
        for _ in 0..eval_iterations {
            black_box(gpu.potential_grad(&q).unwrap());
        }
        let gpu_eval_wall = gpu_eval_started.elapsed().as_secs_f64();

        let metric = Metric::Diag(vec![1.0; dim]);
        let cpu_integrator = LeapfrogIntegrator::new(cpu, 0.025, metric.clone());
        let gpu_integrator = LeapfrogIntegrator::new(gpu, 0.025, metric);
        let state_cpu = seeded_state_from_potential(cpu, 0.0);
        let state_gpu = seeded_state_from_potential(gpu, 0.0);
        let config = WalnutsConfig::default();
        let transition_iterations = 80usize;

        let cpu_started = Instant::now();
        let mut cpu_total_leapfrogs = 0usize;
        for i in 0..transition_iterations {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0xBAD5EED + i as u64);
            let transition =
                walnuts_transition(&cpu_integrator, &state_cpu, &config, &mut rng).unwrap();
            cpu_total_leapfrogs += transition.n_leapfrog;
            black_box(transition);
        }
        let cpu_wall = cpu_started.elapsed().as_secs_f64();

        let gpu_started = Instant::now();
        let mut gpu_total_leapfrogs = 0usize;
        for i in 0..transition_iterations {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0xBAD5EED + i as u64);
            let transition =
                walnuts_transition(&gpu_integrator, &state_gpu, &config, &mut rng).unwrap();
            gpu_total_leapfrogs += transition.n_leapfrog;
            black_box(transition);
        }
        let gpu_wall = gpu_started.elapsed().as_secs_f64();

        let cpu_eval = EvalMetric {
            iterations: eval_iterations,
            dim,
            wall_s: cpu_eval_wall,
            evals_per_sec: eval_iterations as f64 / cpu_eval_wall,
        };
        let gpu_eval = EvalMetric {
            iterations: eval_iterations,
            dim,
            wall_s: gpu_eval_wall,
            evals_per_sec: eval_iterations as f64 / gpu_eval_wall,
        };
        let cpu_transition = TransitionMetric {
            iterations: transition_iterations,
            dim,
            wall_s: cpu_wall,
            transitions_per_sec: transition_iterations as f64 / cpu_wall,
            total_leapfrogs: cpu_total_leapfrogs,
            leapfrogs_per_sec: cpu_total_leapfrogs as f64 / cpu_wall,
        };
        let gpu_transition = TransitionMetric {
            iterations: transition_iterations,
            dim,
            wall_s: gpu_wall,
            transitions_per_sec: transition_iterations as f64 / gpu_wall,
            total_leapfrogs: gpu_total_leapfrogs,
            leapfrogs_per_sec: gpu_total_leapfrogs as f64 / gpu_wall,
        };

        CudaLinearPredictorWalnutsBenchReport {
            schema_version,
            family,
            n_obs,
            n_features,
            potential_grad: CpuGpuComparison {
                gpu_over_cpu_throughput: gpu_eval.evals_per_sec / cpu_eval.evals_per_sec,
                cpu: cpu_eval,
                gpu: gpu_eval,
            },
            walnuts_transition: CpuGpuComparison {
                gpu_over_cpu_throughput: gpu_transition.transitions_per_sec
                    / cpu_transition.transitions_per_sec,
                cpu: cpu_transition,
                gpu: gpu_transition,
            },
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_logistic_potential_matches_cpu_potential_grad() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = logistic_model(256, 8, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_logistic_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA logistic potential: {err}"));
        let q = (0..model.dim()).map(|i| -0.15 + i as f64 * 0.07).collect::<Vec<_>>();

        assert_potential_matches_cpu(&cpu, &gpu, &q);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_linear_potential_matches_cpu_potential_grad() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = linear_model(256, 8, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_linear_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA linear potential: {err}"));
        let q = (0..model.dim()).map(|i| -0.18 + i as f64 * 0.06).collect::<Vec<_>>();

        assert_potential_matches_cpu(&cpu, &gpu, &q);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_linear_leapfrog_matches_cpu_one_step() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = linear_model(512, 12, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_linear_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA linear potential: {err}"));

        assert_leapfrog_matches_cpu(&cpu, &gpu, 0.03);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_logistic_leapfrog_matches_cpu_one_step() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = logistic_model(512, 12, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_logistic_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA logistic potential: {err}"));

        assert_leapfrog_matches_cpu(&cpu, &gpu, 0.03);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_poisson_offset_potential_matches_cpu_potential_grad() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = poisson_model(256, 8, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_poisson_on_device(&model, 0).unwrap_or_else(
            |err| panic!("failed to create CUDA poisson-with-offset potential: {err}"),
        );
        let q = (0..model.dim()).map(|i| -0.12 + i as f64 * 0.05).collect::<Vec<_>>();

        assert_potential_matches_cpu(&cpu, &gpu, &q);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_poisson_offset_leapfrog_matches_cpu_one_step() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = poisson_model(512, 12, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_poisson_on_device(&model, 0).unwrap_or_else(
            |err| panic!("failed to create CUDA poisson-with-offset potential: {err}"),
        );

        assert_leapfrog_matches_cpu(&cpu, &gpu, 0.02);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_negbin_offset_potential_matches_cpu_potential_grad() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = negbin_model(256, 8, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu =
            CudaStdNormalPriorGlmPotential::new_negbin_on_device(&model, 0).unwrap_or_else(|err| {
                panic!("failed to create CUDA negbin-with-offset potential: {err}")
            });
        let q = (0..model.dim()).map(|i| -0.09 + i as f64 * 0.04).collect::<Vec<_>>();

        assert_potential_matches_cpu(&cpu, &gpu, &q);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_negbin_offset_leapfrog_matches_cpu_one_step() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = negbin_model(512, 12, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu =
            CudaStdNormalPriorGlmPotential::new_negbin_on_device(&model, 0).unwrap_or_else(|err| {
                panic!("failed to create CUDA negbin-with-offset potential: {err}")
            });

        assert_leapfrog_matches_cpu(&cpu, &gpu, 0.015);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_weibull_aft_potential_matches_cpu_potential_grad() {
        if !CudaStdNormalPriorIcWeibullAftPotential::is_available() {
            return;
        }

        let model = weibull_aft_model(256, 8);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorIcWeibullAftPotential::new_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA weibull_ic_aft potential: {err}"));
        let q = (0..model.dim()).map(|i| -0.16 + i as f64 * 0.045).collect::<Vec<_>>();

        assert_potential_matches_cpu(&cpu, &gpu, &q);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_weibull_aft_leapfrog_matches_cpu_one_step() {
        if !CudaStdNormalPriorIcWeibullAftPotential::is_available() {
            return;
        }

        let model = weibull_aft_model(512, 12);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorIcWeibullAftPotential::new_on_device(&model, 0)
            .unwrap_or_else(|err| panic!("failed to create CUDA weibull_ic_aft potential: {err}"));
        let dim = model.dim();
        let q = vec![0.0; dim];
        let p = vec![0.01; dim];

        assert_leapfrog_matches_cpu_with_state(&cpu, &gpu, 0.001, &q, &p);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "internal GPU benchmark hook; run explicitly on a CUDA host"]
    fn bench_cuda_logistic_walnuts_cpu_vs_gpu() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = logistic_model(4096, 32, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_logistic_on_device(&model, 0).unwrap();
        let report = bench_cpu_vs_gpu(
            "logistic",
            "nextstat.walnuts_cuda_logistic_bench.v1",
            model.n_obs(),
            model.n_features(),
            &cpu,
            &gpu,
        );
        println!(
            "NEXTSTAT_WALNUTS_GPU_LOGISTIC_BENCH_JSON={}",
            serde_json::to_string(&report).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "internal GPU benchmark hook; run explicitly on a CUDA host"]
    fn bench_cuda_linear_walnuts_cpu_vs_gpu() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = linear_model(4096, 32, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_linear_on_device(&model, 0).unwrap();
        let report = bench_cpu_vs_gpu(
            "linear",
            "nextstat.walnuts_cuda_linear_bench.v1",
            model.n_obs(),
            model.n_features(),
            &cpu,
            &gpu,
        );
        println!(
            "NEXTSTAT_WALNUTS_GPU_LINEAR_BENCH_JSON={}",
            serde_json::to_string(&report).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "internal GPU benchmark hook; run explicitly on a CUDA host"]
    fn bench_cuda_poisson_offset_walnuts_cpu_vs_gpu() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = poisson_model(4096, 32, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_poisson_on_device(&model, 0).unwrap();
        let report = bench_cpu_vs_gpu(
            "poisson_with_offset",
            "nextstat.walnuts_cuda_poisson_offset_bench.v1",
            model.n_obs(),
            model.n_features(),
            &cpu,
            &gpu,
        );
        println!(
            "NEXTSTAT_WALNUTS_GPU_POISSON_OFFSET_BENCH_JSON={}",
            serde_json::to_string(&report).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "internal GPU benchmark hook; run explicitly on a CUDA host"]
    fn bench_cuda_negbin_offset_walnuts_cpu_vs_gpu() {
        if !CudaStdNormalPriorGlmPotential::is_available() {
            return;
        }

        let model = negbin_model(4096, 32, true, true);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorGlmPotential::new_negbin_on_device(&model, 0).unwrap();
        let report = bench_cpu_vs_gpu(
            "negbin_with_offset",
            "nextstat.walnuts_cuda_negbin_offset_bench.v1",
            model.n_obs(),
            model.n_features(),
            &cpu,
            &gpu,
        );
        println!(
            "NEXTSTAT_WALNUTS_GPU_NEGBIN_OFFSET_BENCH_JSON={}",
            serde_json::to_string(&report).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "internal GPU benchmark hook; run explicitly on a CUDA host"]
    fn bench_cuda_weibull_aft_walnuts_cpu_vs_gpu() {
        if !CudaStdNormalPriorIcWeibullAftPotential::is_available() {
            return;
        }

        let model = weibull_aft_model(4096, 32);
        let cpu = gaussian_prior_posterior(&model);
        let gpu = CudaStdNormalPriorIcWeibullAftPotential::new_on_device(&model, 0).unwrap();
        let report = bench_cpu_vs_gpu(
            "weibull_ic_aft",
            "nextstat.walnuts_cuda_weibull_ic_aft_bench.v1",
            model.n_obs(),
            model.n_covariates(),
            &cpu,
            &gpu,
        );
        println!(
            "NEXTSTAT_WALNUTS_GPU_WEIBULL_AFT_BENCH_JSON={}",
            serde_json::to_string(&report).unwrap()
        );
    }
}
