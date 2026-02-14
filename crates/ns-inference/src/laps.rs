//! LAPS: Late-Adjusted Parallel Sampler — GPU MAMS on CUDA.
//!
//! Runs 4096+ chains simultaneously on GPU. The entire leapfrog trajectory
//! (all `n_steps = ⌈L/ε⌉`) is computed in a single kernel launch per transition.
//! Fixed trajectory length L guarantees zero warp divergence.
//!
//! Two-phase sampling:
//! - **Phase 1 (unadjusted MCLMC)**: No MH correction → biased but fast warmup.
//! - **Phase 2 (exact MAMS)**: MH accept/reject → asymptotically exact samples.
//!
//! Host handles warmup adaptation (step size, mass matrix) and convergence diagnostics.

use crate::adapt::{DualAveraging, WelfordVariance};
use crate::chain::{Chain, SamplerResult};
use ns_core::Result;

/// LAPS model specification for GPU sampling.
///
/// Each variant carries model-specific data that gets uploaded to the GPU.
/// The model gradient is computed inline in the CUDA kernel.
#[derive(Debug, Clone)]
pub enum LapsModel {
    /// Standard normal in `dim` dimensions: NLL = 0.5 * sum(x_i^2).
    StdNormal { dim: usize },
    /// Eight Schools hierarchical (non-centered).
    /// `y` = observed effects, `sigma` = known standard errors.
    EightSchools { y: Vec<f64>, sigma: Vec<f64>, prior_mu_sigma: f64, prior_tau_scale: f64 },
    /// Neal's funnel: v ~ N(0,9), x_i|v ~ N(0, exp(v)).
    NealFunnel { dim: usize },
    /// Logistic regression: X is `n×p` row-major, y is `n` binary labels.
    GlmLogistic { x_data: Vec<f64>, y_data: Vec<f64>, n: usize, p: usize },
    /// User-defined model with raw CUDA C source for NLL and gradient.
    ///
    /// The `cuda_src` must define:
    /// ```c
    /// __device__ double user_nll(const double* x, int dim, const double* model_data);
    /// __device__ void   user_grad(const double* x, double* grad, int dim, const double* model_data);
    /// ```
    ///
    /// Compiled at runtime via NVRTC (cached to disk).
    Custom { dim: usize, param_names: Vec<String>, model_data: Vec<f64>, cuda_src: String },
}

impl LapsModel {
    /// Parameter dimensionality.
    pub fn dim(&self) -> usize {
        match self {
            LapsModel::StdNormal { dim } => *dim,
            LapsModel::EightSchools { y, .. } => 2 + y.len(),
            LapsModel::NealFunnel { dim } => *dim,
            LapsModel::GlmLogistic { p, .. } => *p,
            LapsModel::Custom { dim, .. } => *dim,
        }
    }

    /// CUDA model ID (-1 for Custom/JIT models).
    fn model_id(&self) -> i32 {
        match self {
            LapsModel::StdNormal { .. } => 0,
            LapsModel::EightSchools { .. } => 1,
            LapsModel::NealFunnel { .. } => 2,
            LapsModel::GlmLogistic { .. } => 3,
            LapsModel::Custom { .. } => -1,
        }
    }

    /// Whether this model uses JIT compilation (Custom variant).
    fn is_jit(&self) -> bool {
        matches!(self, LapsModel::Custom { .. })
    }

    /// Flatten model data for GPU upload.
    fn model_data(&self) -> Vec<f64> {
        match self {
            LapsModel::StdNormal { .. } => vec![],
            LapsModel::EightSchools { y, sigma, prior_mu_sigma, prior_tau_scale } => {
                let j = y.len();
                let inv_var: Vec<f64> = sigma.iter().map(|s| 1.0 / (s * s)).collect();
                let mut data = Vec::with_capacity(1 + 2 * j + 2);
                data.push(j as f64);
                data.extend_from_slice(y);
                data.extend_from_slice(&inv_var);
                data.push(*prior_mu_sigma);
                data.push(*prior_tau_scale);
                data
            }
            LapsModel::NealFunnel { .. } => vec![],
            LapsModel::GlmLogistic { x_data, y_data, n, p } => {
                let mut data = Vec::with_capacity(2 + n * p + n);
                data.push(*n as f64);
                data.push(*p as f64);
                data.extend_from_slice(x_data);
                data.extend_from_slice(y_data);
                data
            }
            LapsModel::Custom { model_data, .. } => model_data.clone(),
        }
    }

    /// Parameter names.
    fn param_names(&self) -> Vec<String> {
        match self {
            LapsModel::StdNormal { dim } => (0..*dim).map(|i| format!("x[{}]", i)).collect(),
            LapsModel::EightSchools { y, .. } => {
                let j = y.len();
                let mut names = vec!["mu".into(), "tau".into()];
                for i in 0..j {
                    names.push(format!("theta_raw[{}]", i));
                }
                names
            }
            LapsModel::NealFunnel { dim } => {
                let mut names = vec!["v".into()];
                for i in 1..*dim {
                    names.push(format!("x[{}]", i));
                }
                names
            }
            LapsModel::GlmLogistic { p, .. } => (0..*p).map(|i| format!("beta[{}]", i)).collect(),
            LapsModel::Custom { param_names, dim, .. } => {
                if param_names.is_empty() {
                    (0..*dim).map(|i| format!("x[{}]", i)).collect()
                } else {
                    param_names.clone()
                }
            }
        }
    }
}

/// LAPS sampler configuration.
#[derive(Debug, Clone)]
pub struct LapsConfig {
    /// Number of GPU chains (default: 4096).
    pub n_chains: usize,
    /// Number of warmup iterations — Phase 1 + Phase 2 (default: 500).
    pub n_warmup: usize,
    /// Number of post-warmup samples (default: 2000).
    pub n_samples: usize,
    /// Target MH acceptance rate (default: 0.9).
    pub target_accept: f64,
    /// Initial step size; 0 = auto-detect (default: 0).
    pub init_step_size: f64,
    /// Initial decoherence length L; 0 = auto π√d (default: 0).
    pub init_l: f64,
    /// Maximum leapfrog steps per trajectory (default: 1024).
    pub max_leapfrog: usize,
    /// PRNG seed.
    pub seed: u64,
    /// GPU device IDs. `None` = auto-detect all GPUs. `Some(vec![0])` = single GPU.
    pub device_ids: Option<Vec<usize>>,
}

impl Default for LapsConfig {
    fn default() -> Self {
        Self {
            n_chains: 4096,
            n_warmup: 500,
            n_samples: 2000,
            target_accept: 0.9,
            init_step_size: 0.0,
            init_l: 0.0,
            max_leapfrog: 1024,
            seed: 42,
            device_ids: None,
        }
    }
}

/// LAPS sampling result with timing.
#[derive(Debug, Clone)]
pub struct LapsResult {
    /// Standard sampler result (chains, diagnostics, etc.)
    pub sampler_result: SamplerResult,
    /// Wall-clock time in seconds.
    pub wall_time_s: f64,
    /// Total GPU kernel launches.
    pub n_kernel_launches: usize,
    /// Phase breakdown: init, warmup, sampling (seconds).
    pub phase_times: [f64; 3],
    /// Number of GPU devices used.
    pub n_devices: usize,
    /// GPU device IDs used.
    pub device_ids: Vec<usize>,
}

/// Detect available CUDA devices. Returns sorted device IDs.
#[cfg(feature = "cuda")]
fn detect_cuda_devices() -> Vec<usize> {
    let n = ns_compute::cuda_device_count();
    (0..n).collect()
}

/// Run LAPS (GPU MAMS) sampling.
///
/// Dispatches to single-GPU or multi-GPU path based on `config.device_ids`.
/// Returns a [`LapsResult`] containing chains from `n_report_chains` chains
/// (default: min(n_chains, 16)) subsampled from the full GPU chain set,
/// plus wall time and kernel launch count.
#[cfg(feature = "cuda")]
pub fn sample_laps(model: &LapsModel, config: LapsConfig) -> Result<LapsResult> {
    let device_ids = config.device_ids.clone().unwrap_or_else(detect_cuda_devices);
    if device_ids.is_empty() {
        return Err(ns_core::Error::Computation("No CUDA devices available".into()));
    }
    if device_ids.len() == 1 {
        sample_laps_single_gpu(model, config, device_ids[0])
    } else {
        sample_laps_multi_gpu(model, config, &device_ids)
    }
}

/// Generate random initial chain state for `n_chains` chains of `dim` dimensions.
#[cfg(feature = "cuda")]
fn generate_init_state(
    n_chains: usize,
    dim: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    use rand::Rng;
    use rand::SeedableRng;

    let total = n_chains * dim;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x_init = vec![0.0f64; total];
    let mut u_init = vec![0.0f64; total];
    let potential_init = vec![0.0f64; n_chains];
    let grad_init = vec![0.0f64; total];

    for c in 0..n_chains {
        for d in 0..dim {
            x_init[c * dim + d] = rng.random::<f64>() * 4.0 - 2.0;
        }
        let mut norm_sq = 0.0;
        for d in 0..dim {
            let z: f64 = rng.random::<f64>() * 2.0 - 1.0;
            u_init[c * dim + d] = z;
            norm_sq += z * z;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for d in 0..dim {
                u_init[c * dim + d] /= norm;
            }
        }
    }

    (x_init, u_init, potential_init, grad_init)
}

/// Compute initial trajectory length and step size.
#[cfg(feature = "cuda")]
fn compute_initial_params(dim: usize, config: &LapsConfig) -> (f64, f64) {
    let l = if config.init_l > 0.0 {
        config.init_l
    } else {
        std::f64::consts::PI * (dim as f64).sqrt()
    };
    let eps = if config.init_step_size > 0.0 { config.init_step_size } else { l / 10.0 };
    (l, eps)
}

/// Compute inv_mass from Welford variance with Stan regularization.
#[cfg(feature = "cuda")]
fn regularize_inv_mass(welford: &WelfordVariance, dim: usize) -> Option<Vec<f64>> {
    if welford.count() < 10 {
        return None;
    }
    let var = welford.variance();
    let count = welford.count() as f64;
    let alpha = count / (count + 5.0);
    let inv_mass: Vec<f64> =
        (0..dim).map(|i| (alpha * var[i] + 1e-3 * (1.0 - alpha)).max(1e-10)).collect();
    Some(inv_mass)
}

/// Create a CudaMamsAccelerator for the given model on the specified device.
#[cfg(feature = "cuda")]
fn create_accelerator(
    model: &LapsModel,
    n_chains: usize,
    dim: usize,
    model_data: &[f64],
    seed: u64,
    device_id: usize,
) -> Result<ns_compute::cuda_mams::CudaMamsAccelerator> {
    use ns_compute::cuda_mams::CudaMamsAccelerator;
    if model.is_jit() {
        let cuda_src = match model {
            LapsModel::Custom { cuda_src, .. } => cuda_src.as_str(),
            _ => unreachable!(),
        };
        CudaMamsAccelerator::new_jit_on_device(n_chains, dim, model_data, cuda_src, seed, device_id)
    } else {
        CudaMamsAccelerator::new_on_device(
            n_chains,
            dim,
            model.model_id(),
            model_data,
            seed,
            device_id,
        )
    }
}

/// Assemble SamplerResult from per-chain collected data.
#[cfg(feature = "cuda")]
fn assemble_result(
    chain_draws: Vec<Vec<Vec<f64>>>,
    chain_accept_probs: Vec<Vec<f64>>,
    chain_energies: Vec<Vec<f64>>,
    chain_divergences: Vec<Vec<bool>>,
    chain_leapfrogs: Vec<Vec<usize>>,
    inv_mass: &[f64],
    eps: f64,
    n_samples: usize,
    n_warmup: usize,
    param_names: Vec<String>,
) -> SamplerResult {
    let mass_diag: Vec<f64> = inv_mass.iter().map(|&im| 1.0 / im.max(1e-30)).collect();
    let n_report_chains = chain_draws.len();

    let chains: Vec<Chain> = (0..n_report_chains)
        .map(|c| Chain {
            draws_unconstrained: chain_draws[c].clone(),
            draws_constrained: chain_draws[c].clone(),
            divergences: chain_divergences[c].clone(),
            tree_depths: vec![0; n_samples],
            accept_probs: chain_accept_probs[c].clone(),
            energies: chain_energies[c].clone(),
            n_leapfrog: chain_leapfrogs[c].clone(),
            max_treedepth: 0,
            step_size: eps,
            mass_diag: mass_diag.clone(),
        })
        .collect();

    SamplerResult { chains, param_names, n_warmup, n_samples, diagnostics: None }
}

/// Single-GPU LAPS sampling (original path, zero overhead for single device).
#[cfg(feature = "cuda")]
fn sample_laps_single_gpu(
    model: &LapsModel,
    config: LapsConfig,
    device_id: usize,
) -> Result<LapsResult> {
    let start = std::time::Instant::now();
    let dim = model.dim();
    let n_chains = config.n_chains;
    let model_data = model.model_data();

    let mut accel = create_accelerator(model, n_chains, dim, &model_data, config.seed, device_id)?;

    let (x_init, u_init, potential_init, grad_init) =
        generate_init_state(n_chains, dim, config.seed);
    accel.upload_state(&x_init, &u_init, &potential_init, &grad_init)?;
    let t_init = start.elapsed().as_secs_f64();

    let (l, mut eps) = compute_initial_params(dim, &config);
    let mut inv_mass = vec![1.0f64; dim];
    let mut n_kernel_launches = 0usize;

    // ---------- Phase 1 warmup: unadjusted (no MH) ----------
    let phase1_iters = (config.n_warmup as f64 * 0.4) as usize;
    let phase2_iters = (config.n_warmup as f64 * 0.3) as usize;
    let phase_final_iters = config.n_warmup.saturating_sub(phase1_iters + phase2_iters);

    let mut da = DualAveraging::new(config.target_accept, eps);
    let mut welford = WelfordVariance::new(dim);

    for iter in 0..phase1_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        accel.transition(eps, l, n_steps, false)?;
        n_kernel_launches += 1;

        if (iter + 1) % 50 == 0 || iter == phase1_iters - 1 {
            let diag = accel.download_diagnostics()?;
            let mean_ap = compute_mean_accept_prob(&diag.energy_error);
            da.update(mean_ap);
            eps = da.current_step_size();

            let positions = accel.download_positions()?;
            for c in 0..n_chains.min(256) {
                welford.update(&positions[c * dim..(c + 1) * dim]);
            }
        }
    }

    if let Some(new_mass) = regularize_inv_mass(&welford, dim) {
        inv_mass = new_mass;
        accel.set_inv_mass(&inv_mass)?;
        eps = da.adapted_step_size();
        da = DualAveraging::new(config.target_accept, eps);
    }

    // ---------- Phase 2 warmup: MH enabled ----------
    let mut welford2 = WelfordVariance::new(dim);
    for iter in 0..phase2_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        accel.transition(eps, l, n_steps, true)?;
        n_kernel_launches += 1;

        if (iter + 1) % 50 == 0 || iter == phase2_iters - 1 {
            let diag = accel.download_diagnostics()?;
            let mean_ap = compute_mean_accept_prob(&diag.energy_error);
            da.update(mean_ap);
            eps = da.current_step_size();

            let positions = accel.download_positions()?;
            for c in 0..n_chains.min(256) {
                welford2.update(&positions[c * dim..(c + 1) * dim]);
            }
        }
    }

    if let Some(new_mass) = regularize_inv_mass(&welford2, dim) {
        inv_mass = new_mass;
        accel.set_inv_mass(&inv_mass)?;
    }
    eps = da.adapted_step_size();

    // ---------- Phase final: equilibrate ----------
    for _iter in 0..phase_final_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        accel.transition(eps, l, n_steps, true)?;
        n_kernel_launches += 1;
    }

    let t_warmup = start.elapsed().as_secs_f64() - t_init;

    // ---------- Sampling (batched GPU accumulation) ----------
    let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
    let n_report_chains = n_chains.min(16);
    let batch_size = 500usize;
    accel.configure_batch(batch_size, n_report_chains)?;

    let mut chain_draws: Vec<Vec<Vec<f64>>> =
        (0..n_report_chains).map(|_| Vec::with_capacity(config.n_samples)).collect();
    let mut chain_accept_probs: Vec<Vec<f64>> =
        (0..n_report_chains).map(|_| Vec::with_capacity(config.n_samples)).collect();
    let mut chain_energies: Vec<Vec<f64>> =
        (0..n_report_chains).map(|_| Vec::with_capacity(config.n_samples)).collect();
    let mut chain_divergences: Vec<Vec<bool>> =
        (0..n_report_chains).map(|_| Vec::with_capacity(config.n_samples)).collect();
    let mut chain_leapfrogs: Vec<Vec<usize>> =
        (0..n_report_chains).map(|_| Vec::with_capacity(config.n_samples)).collect();

    let mut samples_collected = 0usize;
    while samples_collected < config.n_samples {
        let this_batch = batch_size.min(config.n_samples - samples_collected);
        let batch = accel.transition_batch(eps, l, n_steps, this_batch)?;
        n_kernel_launches += batch.n_launches;

        for s in 0..this_batch {
            for c in 0..n_report_chains {
                let draw: Vec<f64> = batch.positions[s][c * dim..(c + 1) * dim].to_vec();
                chain_draws[c].push(draw);
                chain_accept_probs[c].push(if batch.accepted[s][c] != 0 { 1.0 } else { 0.0 });
                chain_energies[c].push(batch.potentials[s][c]);
                let is_divergent =
                    !batch.energy_error[s][c].is_finite() || batch.energy_error[s][c] > 1000.0;
                chain_divergences[c].push(is_divergent);
                chain_leapfrogs[c].push(n_steps);
            }
        }
        samples_collected += this_batch;
    }

    let sampler_result = assemble_result(
        chain_draws,
        chain_accept_probs,
        chain_energies,
        chain_divergences,
        chain_leapfrogs,
        &inv_mass,
        eps,
        config.n_samples,
        config.n_warmup,
        model.param_names(),
    );

    let wall_time_s = start.elapsed().as_secs_f64();
    let t_sampling = wall_time_s - t_init - t_warmup;

    Ok(LapsResult {
        sampler_result,
        wall_time_s,
        n_kernel_launches,
        phase_times: [t_init, t_warmup, t_sampling],
        n_devices: 1,
        device_ids: vec![device_id],
    })
}

/// Per-device result returned from each GPU thread.
#[cfg(feature = "cuda")]
struct DeviceLapsResult {
    chains: Vec<Chain>,
    n_kernel_launches: usize,
    warmup_secs: f64,
    sampling_secs: f64,
}

/// Multi-GPU LAPS sampling with synchronized warmup across devices.
///
/// Each device runs a shard of chains. During warmup, devices synchronize every
/// 50 iterations via barriers to aggregate diagnostics and compute global
/// step size and inverse mass matrix. During sampling, devices run independently.
#[cfg(feature = "cuda")]
fn sample_laps_multi_gpu(
    model: &LapsModel,
    config: LapsConfig,
    device_ids: &[usize],
) -> Result<LapsResult> {
    use std::sync::{Barrier, Mutex};

    let start = std::time::Instant::now();
    let dim = model.dim();
    let n_chains = config.n_chains;
    let n_devices = device_ids.len();
    let model_data = model.model_data();

    // Distribute chains across devices
    let base_chains = n_chains / n_devices;
    let remainder = n_chains % n_devices;
    let chains_per_device: Vec<usize> =
        (0..n_devices).map(|i| base_chains + if i < remainder { 1 } else { 0 }).collect();

    // Report chains: distribute min(n_chains, 16) across devices
    let total_report = n_chains.min(16);
    let base_report = total_report / n_devices;
    let report_remainder = total_report % n_devices;
    let report_per_device: Vec<usize> = (0..n_devices)
        .map(|i| {
            let r = base_report + if i < report_remainder { 1 } else { 0 };
            r.min(chains_per_device[i])
        })
        .collect();

    let (l, init_eps) = compute_initial_params(dim, &config);

    // Shared warmup state (protected by mutex, synchronized by barriers)
    let shared_eps = Mutex::new(init_eps);
    let shared_inv_mass = Mutex::new(vec![1.0f64; dim]);

    // Aggregation buffers: each device publishes its energy_errors and positions
    let shared_energy_errors: Mutex<Vec<Vec<f64>>> =
        Mutex::new((0..n_devices).map(|_| Vec::new()).collect());
    let shared_positions: Mutex<Vec<Vec<f64>>> =
        Mutex::new((0..n_devices).map(|_| Vec::new()).collect());

    // Warmup phase counts
    let phase1_iters = (config.n_warmup as f64 * 0.4) as usize;
    let phase2_iters = (config.n_warmup as f64 * 0.3) as usize;
    let phase_final_iters = config.n_warmup.saturating_sub(phase1_iters + phase2_iters);

    // Count sync points for warmup phases
    fn count_sync_points(total_iters: usize) -> usize {
        if total_iters == 0 {
            return 0;
        }
        let mut count = 0;
        for iter in 0..total_iters {
            if (iter + 1) % 50 == 0 || iter == total_iters - 1 {
                count += 1;
            }
        }
        count
    }
    let phase1_syncs = count_sync_points(phase1_iters);
    let phase2_syncs = count_sync_points(phase2_iters);
    // Each sync point = 1 barrier pair (upload + aggregated).
    // +1 pair for post-phase1 mass update, +1 pair for post-phase2 mass update.
    let total_barrier_pairs = phase1_syncs + 1 + phase2_syncs + 1;

    let barrier_upload = Barrier::new(n_devices);
    let barrier_aggregated = Barrier::new(n_devices);

    // Global DualAveraging and Welford (only device 0 writes)
    let shared_da = Mutex::new(DualAveraging::new(config.target_accept, init_eps));
    let shared_welford = Mutex::new(WelfordVariance::new(dim));
    let shared_welford2 = Mutex::new(WelfordVariance::new(dim));

    let device_results: Vec<Result<DeviceLapsResult>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_devices)
            .map(|dev_idx| {
                let dev_chains = chains_per_device[dev_idx];
                let dev_report = report_per_device[dev_idx];
                let device_id = device_ids[dev_idx];
                let dev_seed = config.seed + dev_idx as u64 * 1_000_000;

                // References to shared state
                let shared_eps = &shared_eps;
                let shared_inv_mass = &shared_inv_mass;
                let shared_energy_errors = &shared_energy_errors;
                let shared_positions = &shared_positions;
                let barrier_upload = &barrier_upload;
                let barrier_aggregated = &barrier_aggregated;
                let shared_da = &shared_da;
                let shared_welford = &shared_welford;
                let shared_welford2 = &shared_welford2;
                let config = &config;
                let model_data = &model_data;

                scope.spawn(move || -> Result<DeviceLapsResult> {
                    // Track remaining barrier pairs for error-path cleanup.
                    // Each pair = one barrier_upload.wait() + one barrier_aggregated.wait().
                    let mut remaining_barriers = total_barrier_pairs;

                    // Helper: hit remaining barriers and return error
                    macro_rules! bail_with_barriers {
                        ($e:expr) => {{
                            let err = $e;
                            while remaining_barriers > 0 {
                                barrier_upload.wait();
                                barrier_aggregated.wait();
                                remaining_barriers -= 1;
                            }
                            return Err(err);
                        }};
                    }

                    let mut accel = match create_accelerator(
                        model, dev_chains, dim, model_data, dev_seed, device_id,
                    ) {
                        Ok(a) => a,
                        Err(e) => bail_with_barriers!(e),
                    };

                    let (x_init, u_init, potential_init, grad_init) =
                        generate_init_state(dev_chains, dim, dev_seed);
                    if let Err(e) =
                        accel.upload_state(&x_init, &u_init, &potential_init, &grad_init)
                    {
                        bail_with_barriers!(e);
                    }

                    let mut eps = *shared_eps.lock().unwrap();
                    let mut n_kernel_launches = 0usize;

                    // ===== Phase 1: unadjusted warmup =====
                    for iter in 0..phase1_iters {
                        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
                        if let Err(e) = accel.transition(eps, l, n_steps, false) {
                            bail_with_barriers!(e);
                        }
                        n_kernel_launches += 1;

                        if (iter + 1) % 50 == 0 || iter == phase1_iters - 1 {
                            // Download local diagnostics
                            let diag = match accel.download_diagnostics() {
                                Ok(d) => d,
                                Err(e) => bail_with_barriers!(e),
                            };
                            let positions = match accel.download_positions() {
                                Ok(p) => p,
                                Err(e) => bail_with_barriers!(e),
                            };

                            // Publish to shared buffers
                            {
                                let mut ee = shared_energy_errors.lock().unwrap();
                                ee[dev_idx] = diag.energy_error;
                            }
                            {
                                let mut pp = shared_positions.lock().unwrap();
                                pp[dev_idx] = positions;
                            }

                            barrier_upload.wait();

                            // Device 0: aggregate and update global state
                            if dev_idx == 0 {
                                let ee = shared_energy_errors.lock().unwrap();
                                let all_errors: Vec<f64> =
                                    ee.iter().flat_map(|v| v.iter().copied()).collect();
                                let mean_ap = compute_mean_accept_prob(&all_errors);

                                let mut da = shared_da.lock().unwrap();
                                da.update(mean_ap);
                                *shared_eps.lock().unwrap() = da.current_step_size();

                                let pp = shared_positions.lock().unwrap();
                                let mut welford = shared_welford.lock().unwrap();
                                for dev_pos in pp.iter() {
                                    let dev_n = dev_pos.len() / dim;
                                    for c in 0..dev_n.min(64) {
                                        welford.update(&dev_pos[c * dim..(c + 1) * dim]);
                                    }
                                }
                            }

                            barrier_aggregated.wait();
                            remaining_barriers -= 1; // one pair consumed

                            // All devices read updated eps
                            eps = *shared_eps.lock().unwrap();
                        }
                    }

                    // Post-phase 1: mass matrix update
                    {
                        barrier_upload.wait();

                        if dev_idx == 0 {
                            let welford = shared_welford.lock().unwrap();
                            if let Some(new_mass) = regularize_inv_mass(&welford, dim) {
                                *shared_inv_mass.lock().unwrap() = new_mass;
                                let adapted = {
                                    let da = shared_da.lock().unwrap();
                                    da.adapted_step_size()
                                };
                                *shared_eps.lock().unwrap() = adapted;
                                *shared_da.lock().unwrap() =
                                    DualAveraging::new(config.target_accept, adapted);
                            }
                        }

                        barrier_aggregated.wait();
                        remaining_barriers -= 1; // one pair consumed

                        let new_mass = shared_inv_mass.lock().unwrap().clone();
                        if let Err(e) = accel.set_inv_mass(&new_mass) {
                            bail_with_barriers!(e);
                        }
                        eps = *shared_eps.lock().unwrap();
                    }

                    // ===== Phase 2: MH-enabled warmup =====
                    for iter in 0..phase2_iters {
                        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
                        if let Err(e) = accel.transition(eps, l, n_steps, true) {
                            bail_with_barriers!(e);
                        }
                        n_kernel_launches += 1;

                        if (iter + 1) % 50 == 0 || iter == phase2_iters - 1 {
                            let diag = match accel.download_diagnostics() {
                                Ok(d) => d,
                                Err(e) => bail_with_barriers!(e),
                            };
                            let positions = match accel.download_positions() {
                                Ok(p) => p,
                                Err(e) => bail_with_barriers!(e),
                            };

                            {
                                let mut ee = shared_energy_errors.lock().unwrap();
                                ee[dev_idx] = diag.energy_error;
                            }
                            {
                                let mut pp = shared_positions.lock().unwrap();
                                pp[dev_idx] = positions;
                            }

                            barrier_upload.wait();

                            if dev_idx == 0 {
                                let ee = shared_energy_errors.lock().unwrap();
                                let all_errors: Vec<f64> =
                                    ee.iter().flat_map(|v| v.iter().copied()).collect();
                                let mean_ap = compute_mean_accept_prob(&all_errors);

                                let mut da = shared_da.lock().unwrap();
                                da.update(mean_ap);
                                *shared_eps.lock().unwrap() = da.current_step_size();

                                let pp = shared_positions.lock().unwrap();
                                let mut welford2 = shared_welford2.lock().unwrap();
                                for dev_pos in pp.iter() {
                                    let dev_n = dev_pos.len() / dim;
                                    for c in 0..dev_n.min(64) {
                                        welford2.update(&dev_pos[c * dim..(c + 1) * dim]);
                                    }
                                }
                            }

                            barrier_aggregated.wait();
                            remaining_barriers -= 1; // one pair consumed

                            eps = *shared_eps.lock().unwrap();
                        }
                    }

                    // Post-phase 2: refine mass matrix
                    {
                        barrier_upload.wait();

                        if dev_idx == 0 {
                            let welford2 = shared_welford2.lock().unwrap();
                            if let Some(new_mass) = regularize_inv_mass(&welford2, dim) {
                                *shared_inv_mass.lock().unwrap() = new_mass;
                            }
                            let da = shared_da.lock().unwrap();
                            *shared_eps.lock().unwrap() = da.adapted_step_size();
                        }

                        barrier_aggregated.wait();
                        remaining_barriers -= 1; // one pair consumed

                        let new_mass = shared_inv_mass.lock().unwrap().clone();
                        if let Err(e) = accel.set_inv_mass(&new_mass) {
                            // No more barrier pairs after this point
                            return Err(e);
                        }
                        eps = *shared_eps.lock().unwrap();
                    }

                    debug_assert_eq!(remaining_barriers, 0);

                    // ===== Phase final: equilibrate (independent) =====
                    for _iter in 0..phase_final_iters {
                        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
                        accel.transition(eps, l, n_steps, true)?;
                        n_kernel_launches += 1;
                    }

                    let warmup_secs = start.elapsed().as_secs_f64();

                    // ===== Sampling (independent, batched) =====
                    let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
                    let batch_size = 500usize;

                    if dev_report == 0 {
                        // This device contributes no report chains; just run transitions
                        let mut remaining = config.n_samples;
                        while remaining > 0 {
                            let this_batch = batch_size.min(remaining);
                            accel.configure_batch(this_batch, 0)?;
                            // Run transitions without accumulation
                            for _ in 0..this_batch {
                                accel.transition(eps, l, n_steps, true)?;
                                n_kernel_launches += 1;
                            }
                            remaining -= this_batch;
                        }
                        let sampling_secs = start.elapsed().as_secs_f64() - warmup_secs;
                        return Ok(DeviceLapsResult { chains: Vec::new(), n_kernel_launches, warmup_secs, sampling_secs });
                    }

                    accel.configure_batch(batch_size, dev_report)?;

                    let mut chain_draws: Vec<Vec<Vec<f64>>> =
                        (0..dev_report).map(|_| Vec::with_capacity(config.n_samples)).collect();
                    let mut chain_accept_probs: Vec<Vec<f64>> =
                        (0..dev_report).map(|_| Vec::with_capacity(config.n_samples)).collect();
                    let mut chain_energies: Vec<Vec<f64>> =
                        (0..dev_report).map(|_| Vec::with_capacity(config.n_samples)).collect();
                    let mut chain_divergences: Vec<Vec<bool>> =
                        (0..dev_report).map(|_| Vec::with_capacity(config.n_samples)).collect();
                    let mut chain_leapfrogs: Vec<Vec<usize>> =
                        (0..dev_report).map(|_| Vec::with_capacity(config.n_samples)).collect();

                    let mut samples_collected = 0usize;
                    while samples_collected < config.n_samples {
                        let this_batch = batch_size.min(config.n_samples - samples_collected);
                        let batch = accel.transition_batch(eps, l, n_steps, this_batch)?;
                        n_kernel_launches += batch.n_launches;

                        for s in 0..this_batch {
                            for c in 0..dev_report {
                                let draw: Vec<f64> =
                                    batch.positions[s][c * dim..(c + 1) * dim].to_vec();
                                chain_draws[c].push(draw);
                                chain_accept_probs[c].push(if batch.accepted[s][c] != 0 {
                                    1.0
                                } else {
                                    0.0
                                });
                                chain_energies[c].push(batch.potentials[s][c]);
                                let is_divergent = !batch.energy_error[s][c].is_finite()
                                    || batch.energy_error[s][c] > 1000.0;
                                chain_divergences[c].push(is_divergent);
                                chain_leapfrogs[c].push(n_steps);
                            }
                        }
                        samples_collected += this_batch;
                    }

                    let inv_mass_local = shared_inv_mass.lock().unwrap().clone();
                    let mass_diag: Vec<f64> =
                        inv_mass_local.iter().map(|&im| 1.0 / im.max(1e-30)).collect();

                    let chains: Vec<Chain> = (0..dev_report)
                        .map(|c| Chain {
                            draws_unconstrained: chain_draws[c].clone(),
                            draws_constrained: chain_draws[c].clone(),
                            divergences: chain_divergences[c].clone(),
                            tree_depths: vec![0; config.n_samples],
                            accept_probs: chain_accept_probs[c].clone(),
                            energies: chain_energies[c].clone(),
                            n_leapfrog: chain_leapfrogs[c].clone(),
                            max_treedepth: 0,
                            step_size: eps,
                            mass_diag: mass_diag.clone(),
                        })
                        .collect();

                    let sampling_secs = start.elapsed().as_secs_f64() - warmup_secs;
                    Ok(DeviceLapsResult { chains, n_kernel_launches, warmup_secs, sampling_secs })
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge results from all devices
    let mut all_chains = Vec::new();
    let mut total_kernel_launches = 0usize;
    let mut ok_device_ids = Vec::new();
    let mut max_warmup_secs = 0.0f64;
    let mut max_sampling_secs = 0.0f64;

    for (i, dev_result) in device_results.into_iter().enumerate() {
        match dev_result {
            Ok(dr) => {
                all_chains.extend(dr.chains);
                total_kernel_launches += dr.n_kernel_launches;
                ok_device_ids.push(device_ids[i]);
                max_warmup_secs = max_warmup_secs.max(dr.warmup_secs);
                max_sampling_secs = max_sampling_secs.max(dr.sampling_secs);
            }
            Err(e) => {
                log::warn!("LAPS device {} failed: {}", device_ids[i], e);
            }
        }
    }

    if ok_device_ids.is_empty() {
        return Err(ns_core::Error::Computation(
            "All GPU devices failed during LAPS multi-GPU sampling".into(),
        ));
    }

    let wall_time_s = start.elapsed().as_secs_f64();

    let sampler_result = SamplerResult {
        chains: all_chains,
        param_names: model.param_names(),
        n_warmup: config.n_warmup,
        n_samples: config.n_samples,
        diagnostics: None,
    };

    Ok(LapsResult {
        sampler_result,
        wall_time_s,
        n_kernel_launches: total_kernel_launches,
        phase_times: [0.0, max_warmup_secs, max_sampling_secs],
        n_devices: ok_device_ids.len(),
        device_ids: ok_device_ids,
    })
}

/// Compute mean acceptance probability from energy errors.
fn compute_mean_accept_prob(energy_errors: &[f64]) -> f64 {
    if energy_errors.is_empty() {
        return 0.5;
    }
    let sum: f64 =
        energy_errors.iter().map(|&w| if w.is_finite() { (-w).exp().min(1.0) } else { 0.0 }).sum();
    sum / energy_errors.len() as f64
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_laps_std_normal_smoke() {
        let model = LapsModel::StdNormal { dim: 10 };
        let config = LapsConfig {
            n_chains: 256,
            n_warmup: 100,
            n_samples: 200,
            seed: 42,
            ..Default::default()
        };
        let result = sample_laps(&model, config).unwrap();
        assert_eq!(result.sampler_result.chains.len(), 16);
        assert_eq!(result.sampler_result.chains[0].draws_constrained.len(), 200);
        assert!(result.wall_time_s > 0.0);
        assert!(result.n_kernel_launches > 0);
    }

    #[test]
    fn test_laps_eight_schools_smoke() {
        let model = LapsModel::EightSchools {
            y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
            prior_mu_sigma: 5.0,
            prior_tau_scale: 5.0,
        };
        let config = LapsConfig {
            n_chains: 256,
            n_warmup: 100,
            n_samples: 100,
            seed: 42,
            ..Default::default()
        };
        let result = sample_laps(&model, config).unwrap();
        assert_eq!(result.sampler_result.chains[0].draws_constrained[0].len(), 10);
    }

    #[test]
    fn test_laps_neal_funnel_smoke() {
        let model = LapsModel::NealFunnel { dim: 5 };
        let config = LapsConfig {
            n_chains: 256,
            n_warmup: 100,
            n_samples: 100,
            seed: 42,
            ..Default::default()
        };
        let result = sample_laps(&model, config).unwrap();
        assert_eq!(result.sampler_result.chains[0].draws_constrained[0].len(), 5);
    }
}
