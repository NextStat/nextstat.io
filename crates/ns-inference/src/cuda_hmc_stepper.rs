//! Internal CUDA-backed HamiltonianStepper prototype for StdNormal targets.
//!
//! This is a seam-validation slice only. It intentionally does not expose a
//! public sampler API and currently supports:
//! - CUDA runtime only
//! - Standard Normal target `U(q)=0.5*||q||^2`
//! - diagonal Euclidean metric only

use crate::hmc::{HamiltonianStepper, HmcState, Metric, StepProbeOutcome, StepSequenceOutcome};
use ns_compute::cuda_hmc_stdnormal::CudaStdNormalLeapfrog;
use ns_core::{Error, Result};
use std::sync::Mutex;

/// Minimal CUDA-backed stepper for `StdNormal` tests and seam validation.
pub(crate) struct CudaStdNormalHamiltonianStepper {
    step_size: f64,
    metric: Metric,
    accel: Mutex<CudaStdNormalLeapfrog>,
}

impl CudaStdNormalHamiltonianStepper {
    pub(crate) fn is_available() -> bool {
        CudaStdNormalLeapfrog::is_available()
    }

    pub(crate) fn new_on_device(step_size: f64, metric: Metric, device_id: usize) -> Result<Self> {
        if !step_size.is_finite() || step_size == 0.0 {
            return Err(Error::Validation(
                "CUDA StdNormal stepper requires finite non-zero step size".to_string(),
            ));
        }
        let inv_mass = match &metric {
            Metric::Diag(diag) => diag.clone(),
            Metric::DenseCholesky { .. } => {
                return Err(Error::Validation(
                    "CUDA StdNormal stepper supports diagonal metric only".to_string(),
                ));
            }
        };
        let accel = CudaStdNormalLeapfrog::new_on_device(&inv_mass, device_id)?;
        Ok(Self { step_size, metric, accel: Mutex::new(accel) })
    }
}

impl HamiltonianStepper for CudaStdNormalHamiltonianStepper {
    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn metric(&self) -> &Metric {
        &self.metric
    }

    fn step_with_eps(&self, state: &mut HmcState, eps: f64) -> Result<()> {
        self.step_many_with_eps(state, eps, 1).into_result()
    }

    fn step_many_with_eps(
        &self,
        state: &mut HmcState,
        eps: f64,
        n_steps: usize,
    ) -> StepSequenceOutcome {
        let dim = self.metric.dim();
        if state.q.len() != dim || state.p.len() != dim || state.grad_potential.len() != dim {
            return StepSequenceOutcome::Failed {
                attempted_steps: 0,
                error: Error::Validation(format!(
                    "CUDA StdNormal stepper dimension mismatch: metric={}, q={}, p={}, grad={}",
                    dim,
                    state.q.len(),
                    state.p.len(),
                    state.grad_potential.len()
                )),
            };
        }

        let accel = self
            .accel
            .lock()
            .map_err(|_| Error::Computation("CUDA StdNormal stepper lock poisoned".to_string()));
        let mut accel = match accel {
            Ok(accel) => accel,
            Err(error) => {
                return StepSequenceOutcome::Failed { attempted_steps: 0, error };
            }
        };
        if let Err((error, attempted_steps)) =
            accel.step_many_attempted(&mut state.q, &mut state.p, eps, n_steps)
        {
            return StepSequenceOutcome::Failed { attempted_steps, error };
        }
        state.potential = 0.5 * state.q.iter().map(|v| v * v).sum::<f64>();
        state.grad_potential.clone_from(&state.q);
        StepSequenceOutcome::Complete { attempted_steps: n_steps }
    }

    fn probe_log_joint_with_eps(
        &self,
        initial: &HmcState,
        eps: f64,
        n_steps: usize,
    ) -> StepProbeOutcome {
        let dim = self.metric.dim();
        if initial.q.len() != dim || initial.p.len() != dim || initial.grad_potential.len() != dim {
            return StepProbeOutcome::Failed {
                attempted_steps: 0,
                error: Error::Validation(format!(
                    "CUDA StdNormal stepper dimension mismatch: metric={}, q={}, p={}, grad={}",
                    dim,
                    initial.q.len(),
                    initial.p.len(),
                    initial.grad_potential.len()
                )),
            };
        }

        let accel = self
            .accel
            .lock()
            .map_err(|_| Error::Computation("CUDA StdNormal stepper lock poisoned".to_string()));
        let mut accel = match accel {
            Ok(accel) => accel,
            Err(error) => {
                return StepProbeOutcome::Failed { attempted_steps: 0, error };
            }
        };
        match accel.probe_log_joint_many_attempted(&initial.q, &initial.p, eps, n_steps) {
            Ok((log_joint, attempted_steps)) => {
                StepProbeOutcome::Complete { attempted_steps, log_joint }
            }
            Err((error, attempted_steps)) => StepProbeOutcome::Failed { attempted_steps, error },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmc::LeapfrogIntegrator;
    use crate::posterior::Posterior;
    use crate::walnuts::{WalnutsConfig, walnuts_transition};
    use ns_core::traits::{LogDensityModel, PreparedModelRef};
    use rand::SeedableRng;
    use serde::Serialize;
    use std::hint::black_box;
    use std::time::Instant;

    struct StdNormalModel {
        dim: usize,
    }

    impl LogDensityModel for StdNormalModel {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            self.dim
        }

        fn parameter_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{i}]")).collect()
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0; self.dim]
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            Ok(0.5 * params.iter().map(|x| x * x).sum::<f64>())
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            Ok(params.to_vec())
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    fn seeded_state(dim: usize) -> HmcState {
        shifted_state(dim, 0.0)
    }

    fn shifted_state(dim: usize, q_shift: f64) -> HmcState {
        let q = (0..dim).map(|i| (i as f64 + 1.0) * 0.1).collect::<Vec<_>>();
        let p = (0..dim).map(|i| -0.3 + i as f64 * 0.05).collect::<Vec<_>>();
        let q = q.into_iter().map(|value| value + q_shift).collect::<Vec<_>>();
        HmcState {
            potential: 0.5 * q.iter().map(|v| v * v).sum::<f64>(),
            grad_potential: q.clone(),
            q,
            p,
        }
    }

    #[derive(Serialize)]
    struct StepMetric {
        iterations: usize,
        dim: usize,
        wall_s: f64,
        steps_per_sec: f64,
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
    struct TransitionConfigReport {
        max_treedepth: usize,
        max_step_halvings: usize,
        min_micro_steps: usize,
        max_energy_error: f64,
    }

    #[derive(Serialize)]
    struct CpuGpuComparison<T> {
        cpu: T,
        gpu: T,
        gpu_over_cpu_throughput: f64,
    }

    #[derive(Serialize)]
    struct CudaStdNormalBenchReport {
        schema_version: &'static str,
        one_step: CpuGpuComparison<StepMetric>,
        walnuts_transition: CpuGpuComparison<TransitionMetric>,
        transition_config: TransitionConfigReport,
    }

    #[test]
    fn cuda_stdnormal_stepper_rejects_dense_metric() {
        let metric = Metric::DenseCholesky { dim: 2, l: vec![1.0, 0.0, 0.0, 1.0] };
        let err = match CudaStdNormalHamiltonianStepper::new_on_device(0.1, metric, 0) {
            Ok(_) => panic!("expected dense metric to be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("diagonal metric only"), "unexpected error: {err}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_stepper_matches_cpu_one_step() {
        if !CudaStdNormalHamiltonianStepper::is_available() {
            return;
        }

        let model = StdNormalModel { dim: 4 };
        let posterior = Posterior::new(&model);
        let metric = Metric::Diag(vec![1.0, 0.5, 2.0, 1.5]);
        let cpu = LeapfrogIntegrator::new(&posterior, 0.1, metric.clone());
        let gpu = CudaStdNormalHamiltonianStepper::new_on_device(0.1, metric, 0).unwrap();

        let mut cpu_state = seeded_state(4);
        let mut gpu_state = seeded_state(4);

        cpu.step_with_eps(&mut cpu_state, -0.2).unwrap();
        gpu.step_with_eps(&mut gpu_state, -0.2).unwrap();

        for i in 0..4 {
            assert!(
                (cpu_state.q[i] - gpu_state.q[i]).abs() < 1e-12,
                "q mismatch at {i}: cpu={}, gpu={}",
                cpu_state.q[i],
                gpu_state.q[i]
            );
            assert!(
                (cpu_state.p[i] - gpu_state.p[i]).abs() < 1e-12,
                "p mismatch at {i}: cpu={}, gpu={}",
                cpu_state.p[i],
                gpu_state.p[i]
            );
            assert!(
                (cpu_state.grad_potential[i] - gpu_state.grad_potential[i]).abs() < 1e-12,
                "grad mismatch at {i}: cpu={}, gpu={}",
                cpu_state.grad_potential[i],
                gpu_state.grad_potential[i]
            );
        }
        assert!((cpu_state.potential - gpu_state.potential).abs() < 1e-12);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_stepper_matches_cpu_many_steps() {
        if !CudaStdNormalHamiltonianStepper::is_available() {
            return;
        }

        let model = StdNormalModel { dim: 4 };
        let posterior = Posterior::new(&model);
        let metric = Metric::Diag(vec![1.0, 0.5, 2.0, 1.5]);
        let cpu = LeapfrogIntegrator::new(&posterior, 0.1, metric.clone());
        let gpu = CudaStdNormalHamiltonianStepper::new_on_device(0.1, metric, 0).unwrap();

        let mut cpu_state = seeded_state(4);
        let mut gpu_state = seeded_state(4);

        cpu.step_many_with_eps(&mut cpu_state, -0.2, 5).into_result().unwrap();
        gpu.step_many_with_eps(&mut gpu_state, -0.2, 5).into_result().unwrap();

        for i in 0..4 {
            assert!(
                (cpu_state.q[i] - gpu_state.q[i]).abs() < 1e-12,
                "q mismatch at {i}: cpu={}, gpu={}",
                cpu_state.q[i],
                gpu_state.q[i]
            );
            assert!(
                (cpu_state.p[i] - gpu_state.p[i]).abs() < 1e-12,
                "p mismatch at {i}: cpu={}, gpu={}",
                cpu_state.p[i],
                gpu_state.p[i]
            );
            assert!(
                (cpu_state.grad_potential[i] - gpu_state.grad_potential[i]).abs() < 1e-12,
                "grad mismatch at {i}: cpu={}, gpu={}",
                cpu_state.grad_potential[i],
                gpu_state.grad_potential[i]
            );
        }
        assert!((cpu_state.potential - gpu_state.potential).abs() < 1e-12);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_probe_log_joint_matches_cpu_many_steps() {
        if !CudaStdNormalHamiltonianStepper::is_available() {
            return;
        }

        let model = StdNormalModel { dim: 4 };
        let posterior = Posterior::new(&model);
        let metric = Metric::Diag(vec![1.0, 0.5, 2.0, 1.5]);
        let cpu = LeapfrogIntegrator::new(&posterior, 0.1, metric.clone());
        let gpu = CudaStdNormalHamiltonianStepper::new_on_device(0.1, metric, 0).unwrap();
        let initial = shifted_state(4, 0.03);

        let cpu_probe = cpu.probe_log_joint_with_eps(&initial, -0.2, 3).into_result().unwrap();
        let gpu_probe = gpu.probe_log_joint_with_eps(&initial, -0.2, 3).into_result().unwrap();
        assert!((cpu_probe - gpu_probe).abs() < 1e-12);

        let mut cpu_state = initial.clone();
        let mut gpu_state = initial.clone();
        cpu.step_many_with_eps(&mut cpu_state, -0.2, 3).into_result().unwrap();
        gpu.step_many_with_eps(&mut gpu_state, -0.2, 3).into_result().unwrap();
        for i in 0..4 {
            assert!((cpu_state.q[i] - gpu_state.q[i]).abs() < 1e-12);
            assert!((cpu_state.p[i] - gpu_state.p[i]).abs() < 1e-12);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_walnuts_transition_matches_cpu() {
        if !CudaStdNormalHamiltonianStepper::is_available() {
            return;
        }

        let model = StdNormalModel { dim: 4 };
        let posterior = Posterior::new(&model);
        let metric = Metric::Diag(vec![1.0, 0.75, 1.5, 0.9]);
        let cpu = LeapfrogIntegrator::new(&posterior, 0.08, metric.clone());
        let gpu = CudaStdNormalHamiltonianStepper::new_on_device(0.08, metric, 0).unwrap();
        let current = seeded_state(4);
        let config = WalnutsConfig {
            max_treedepth: 3,
            max_step_halvings: 2,
            min_micro_steps: 1,
            max_energy_error: 2.0,
        };

        let mut rng_cpu = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng_gpu = rand::rngs::StdRng::seed_from_u64(42);
        let cpu_transition = walnuts_transition(&cpu, &current, &config, &mut rng_cpu).unwrap();
        let gpu_transition = walnuts_transition(&gpu, &current, &config, &mut rng_gpu).unwrap();

        assert_eq!(cpu_transition.depth, gpu_transition.depth);
        assert_eq!(cpu_transition.divergent, gpu_transition.divergent);
        assert_eq!(cpu_transition.n_leapfrog, gpu_transition.n_leapfrog);
        assert!((cpu_transition.accept_prob - gpu_transition.accept_prob).abs() < 1e-12);
        assert!((cpu_transition.energy - gpu_transition.energy).abs() < 1e-12);
        for i in 0..4 {
            assert!(
                (cpu_transition.q[i] - gpu_transition.q[i]).abs() < 1e-12,
                "q mismatch at {i}: cpu={}, gpu={}",
                cpu_transition.q[i],
                gpu_transition.q[i]
            );
            assert!(
                (cpu_transition.grad_potential[i] - gpu_transition.grad_potential[i]).abs() < 1e-12,
                "grad mismatch at {i}: cpu={}, gpu={}",
                cpu_transition.grad_potential[i],
                gpu_transition.grad_potential[i]
            );
        }
        assert!((cpu_transition.potential - gpu_transition.potential).abs() < 1e-12);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA GPU; internal runner parses NEXTSTAT_WALNUTS_GPU_CERT_JSON"]
    fn bench_cuda_stdnormal_walnuts_cpu_vs_gpu() {
        assert!(
            CudaStdNormalHamiltonianStepper::is_available(),
            "CUDA StdNormal stepper unavailable on this host"
        );

        let one_step_dim = 256;
        let one_step_iterations = 2048usize;
        let transition_dim = 32;
        let transition_iterations = 256usize;
        let step_eps = 0.05;
        let transition_eps = 0.08;
        let transition_config = WalnutsConfig {
            max_treedepth: 5,
            max_step_halvings: 2,
            min_micro_steps: 1,
            max_energy_error: 2.0,
        };

        let one_step_model = StdNormalModel { dim: one_step_dim };
        let one_step_posterior = Posterior::new(&one_step_model);
        let one_step_metric = Metric::Diag(vec![1.0; one_step_dim]);
        let cpu_one_step =
            LeapfrogIntegrator::new(&one_step_posterior, step_eps, one_step_metric.clone());
        let gpu_one_step =
            CudaStdNormalHamiltonianStepper::new_on_device(step_eps, one_step_metric, 0).unwrap();

        let mut cpu_state = shifted_state(one_step_dim, 0.01);
        let mut gpu_state = shifted_state(one_step_dim, 0.01);
        cpu_one_step.step_with_eps(&mut cpu_state, step_eps).unwrap();
        gpu_one_step.step_with_eps(&mut gpu_state, step_eps).unwrap();

        let cpu_one_step_started = Instant::now();
        for _ in 0..one_step_iterations {
            cpu_one_step.step_with_eps(&mut cpu_state, step_eps).unwrap();
        }
        let cpu_one_step_wall = cpu_one_step_started.elapsed().as_secs_f64();
        black_box(cpu_state.potential);

        let gpu_one_step_started = Instant::now();
        for _ in 0..one_step_iterations {
            gpu_one_step.step_with_eps(&mut gpu_state, step_eps).unwrap();
        }
        let gpu_one_step_wall = gpu_one_step_started.elapsed().as_secs_f64();
        black_box(gpu_state.potential);

        let transition_model = StdNormalModel { dim: transition_dim };
        let transition_posterior = Posterior::new(&transition_model);
        let transition_metric = Metric::Diag(vec![1.0; transition_dim]);
        let cpu_transition_stepper = LeapfrogIntegrator::new(
            &transition_posterior,
            transition_eps,
            transition_metric.clone(),
        );
        let gpu_transition_stepper =
            CudaStdNormalHamiltonianStepper::new_on_device(transition_eps, transition_metric, 0)
                .unwrap();

        let mut cpu_total_leapfrogs = 0usize;
        for iter in 0..16usize {
            let current = shifted_state(transition_dim, iter as f64 * 1e-4);
            let seed = 1234 + iter as u64;
            let mut rng_cpu = rand::rngs::StdRng::seed_from_u64(seed);
            let mut rng_gpu = rand::rngs::StdRng::seed_from_u64(seed);
            let cpu_transition = walnuts_transition(
                &cpu_transition_stepper,
                &current,
                &transition_config,
                &mut rng_cpu,
            )
            .unwrap();
            let gpu_transition = walnuts_transition(
                &gpu_transition_stepper,
                &current,
                &transition_config,
                &mut rng_gpu,
            )
            .unwrap();
            assert_eq!(cpu_transition.depth, gpu_transition.depth);
            assert_eq!(cpu_transition.divergent, gpu_transition.divergent);
            assert_eq!(cpu_transition.n_leapfrog, gpu_transition.n_leapfrog);
            assert!((cpu_transition.accept_prob - gpu_transition.accept_prob).abs() < 1e-12);
            assert!((cpu_transition.energy - gpu_transition.energy).abs() < 1e-12);
            assert!((cpu_transition.q[0] - gpu_transition.q[0]).abs() < 1e-12);
        }

        let cpu_transition_started = Instant::now();
        for iter in 0..transition_iterations {
            let current = shifted_state(transition_dim, iter as f64 * 1e-4);
            let seed = 1234 + iter as u64;
            let mut rng_cpu = rand::rngs::StdRng::seed_from_u64(seed);
            let cpu_transition = walnuts_transition(
                &cpu_transition_stepper,
                &current,
                &transition_config,
                &mut rng_cpu,
            )
            .unwrap();
            cpu_total_leapfrogs += cpu_transition.n_leapfrog;
            black_box(cpu_transition.potential);
        }
        let cpu_transition_wall = cpu_transition_started.elapsed().as_secs_f64();

        let gpu_transition_started = Instant::now();
        let mut gpu_total_leapfrogs = 0usize;
        for iter in 0..transition_iterations {
            let current = shifted_state(transition_dim, iter as f64 * 1e-4);
            let seed = 1234 + iter as u64;
            let mut rng_gpu = rand::rngs::StdRng::seed_from_u64(seed);
            let gpu_transition = walnuts_transition(
                &gpu_transition_stepper,
                &current,
                &transition_config,
                &mut rng_gpu,
            )
            .unwrap();
            gpu_total_leapfrogs += gpu_transition.n_leapfrog;
            black_box(gpu_transition.potential);
        }
        let gpu_transition_wall = gpu_transition_started.elapsed().as_secs_f64();
        assert_eq!(
            cpu_total_leapfrogs, gpu_total_leapfrogs,
            "CPU/GPU leapfrog totals differ in timing loop"
        );

        let report = CudaStdNormalBenchReport {
            schema_version: "nextstat.walnuts_cuda_stdnormal_bench.v1",
            one_step: CpuGpuComparison {
                cpu: StepMetric {
                    iterations: one_step_iterations,
                    dim: one_step_dim,
                    wall_s: cpu_one_step_wall,
                    steps_per_sec: one_step_iterations as f64 / cpu_one_step_wall,
                },
                gpu: StepMetric {
                    iterations: one_step_iterations,
                    dim: one_step_dim,
                    wall_s: gpu_one_step_wall,
                    steps_per_sec: one_step_iterations as f64 / gpu_one_step_wall,
                },
                gpu_over_cpu_throughput: cpu_one_step_wall / gpu_one_step_wall,
            },
            walnuts_transition: CpuGpuComparison {
                cpu: TransitionMetric {
                    iterations: transition_iterations,
                    dim: transition_dim,
                    wall_s: cpu_transition_wall,
                    transitions_per_sec: transition_iterations as f64 / cpu_transition_wall,
                    total_leapfrogs: cpu_total_leapfrogs,
                    leapfrogs_per_sec: cpu_total_leapfrogs as f64 / cpu_transition_wall,
                },
                gpu: TransitionMetric {
                    iterations: transition_iterations,
                    dim: transition_dim,
                    wall_s: gpu_transition_wall,
                    transitions_per_sec: transition_iterations as f64 / gpu_transition_wall,
                    total_leapfrogs: gpu_total_leapfrogs,
                    leapfrogs_per_sec: gpu_total_leapfrogs as f64 / gpu_transition_wall,
                },
                gpu_over_cpu_throughput: cpu_transition_wall / gpu_transition_wall,
            },
            transition_config: TransitionConfigReport {
                max_treedepth: transition_config.max_treedepth,
                max_step_halvings: transition_config.max_step_halvings,
                min_micro_steps: transition_config.min_micro_steps,
                max_energy_error: transition_config.max_energy_error,
            },
        };

        println!("NEXTSTAT_WALNUTS_GPU_CERT_JSON={}", serde_json::to_string(&report).unwrap());
    }
}
