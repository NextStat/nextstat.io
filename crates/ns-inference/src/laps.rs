//! LAPS: Late-Adjusted Parallel Sampler — GPU MAMS on CUDA/Metal.
//!
//! Runs 4096+ chains simultaneously on GPU. The entire leapfrog trajectory
//! (all `n_steps = ⌈L/ε⌉`) is computed in a single kernel launch per transition.
//! Fixed trajectory length L guarantees zero warp divergence.
//!
//! Four-phase warmup (matching CPU MAMS):
//! - **Phase 1 (15%)**: Fast DA — adapt ε only, MH enabled.
//! - **Phase 2 (40%)**: DA + Welford — adapt ε + collect mass matrix, MH enabled.
//!   → mass update + reset DA + find new ε via binary search.
//! - **Phase 3 (15%)**: DA with new metric — re-adapt ε after mass update, MH enabled.
//! - **Phase 4 (30%)**: L tuning + equilibrate — fixed ε, tuned L, MH enabled.
//!
//! Host handles warmup adaptation (step size, mass matrix, L tuning) and convergence diagnostics.
//!
//! Backends: CUDA (f64, NVIDIA) and Metal (f32, Apple Silicon). The warmup and
//! sampling logic is generic over [`ns_compute::mams_trait::MamsAccelerator`].

use crate::adapt::{DualAveraging, WelfordVariance, percentile};
use crate::chain::{Chain, SamplerResult};
use ns_compute::mams_trait::MamsAccelerator;
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
    /// Neal's funnel — non-centered parameterization.
    /// Samples (v, z_1..z_{d-1}) where z_i ~ N(0,1), then transforms x_i = z_i * exp(v/2).
    NealFunnelNcp { dim: usize },
    /// Neal's funnel — Riemannian MAMS with Fisher metric.
    ///
    /// G = diag(1/9, exp(-v), ..., exp(-v)). The kernel computes the effective
    /// potential U_eff and natural gradient internally, with position-dependent
    /// metric scaling in B-step and A-step.
    NealFunnelRiemannian { dim: usize },
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
            LapsModel::NealFunnelNcp { dim } => *dim,
            LapsModel::NealFunnelRiemannian { dim } => *dim,
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
            LapsModel::NealFunnelNcp { .. } => 4,
            LapsModel::NealFunnelRiemannian { .. } => 5,
            LapsModel::GlmLogistic { .. } => 3,
            LapsModel::Custom { .. } => -1,
        }
    }

    /// Whether this model uses Riemannian (position-dependent metric) dynamics.
    fn is_riemannian(&self) -> bool {
        matches!(self, LapsModel::NealFunnelRiemannian { .. })
    }

    /// Whether this model uses JIT compilation (Custom variant).
    fn is_jit(&self) -> bool {
        matches!(self, LapsModel::Custom { .. })
    }

    /// Whether this model is supported on Metal (built-in models only, no JIT).
    fn is_metal_supported(&self) -> bool {
        !self.is_jit()
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
            LapsModel::NealFunnelNcp { .. } => vec![],
            LapsModel::NealFunnelRiemannian { .. } => vec![],
            LapsModel::GlmLogistic { x_data, y_data, n, p } => {
                let np = n * p;
                let mut x_col = vec![0.0f64; np];
                for row in 0..*n {
                    for col in 0..*p {
                        x_col[col * *n + row] = x_data[row * *p + col];
                    }
                }
                // Layout:
                // [n, p, X_row(n*p), y(n), X_col(n*p)]
                let mut data = Vec::with_capacity(2 + np + n + np);
                data.push(*n as f64);
                data.push(*p as f64);
                data.extend_from_slice(x_data);
                data.extend_from_slice(y_data);
                data.extend_from_slice(&x_col);
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
            LapsModel::NealFunnelNcp { dim } => {
                let mut names = vec!["v".into()];
                for i in 1..*dim {
                    names.push(format!("z[{}]", i));
                }
                names
            }
            LapsModel::NealFunnelRiemannian { dim } => {
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
    /// Maximum leapfrog steps per trajectory (default: 8192).
    pub max_leapfrog: usize,
    /// PRNG seed.
    pub seed: u64,
    /// GPU device IDs. `None` = auto-detect all GPUs. `Some(vec![0])` = single GPU.
    pub device_ids: Option<Vec<usize>>,
    /// Warmup sync interval: download diagnostics every N iterations (default: 100).
    /// Lower values = more host-device sync overhead but faster adaptation.
    pub sync_interval: usize,
    /// Welford chains per device for mass matrix estimation (default: 256).
    pub welford_chains_per_device: usize,
    /// Batch size for sampling phase: transitions per GPU-side accumulation (default: 1000).
    pub batch_size: usize,
    /// Fused transitions per kernel launch (default: same as batch_size).
    /// When >0, a single kernel launch executes N transitions keeping state in registers.
    /// Set to 0 to disable fused kernel and use individual-launch batch path.
    pub fused_transitions: usize,
    /// Number of chains retained for diagnostics/returned posterior (default: 32).
    ///
    /// Returning too few chains can make max R-hat unstable when one chain sticks.
    pub report_chains: usize,
    /// Use diagonal mass matrix from Welford variance (default: true).
    /// With per-chain ε adaptation, Welford provides rough scale normalization
    /// and per-chain DA fine-tunes — works well even for multi-scale models
    /// like Neal's funnel. Only disable for debugging.
    pub use_diagonal_precond: bool,
    /// Number of mass matrix adaptation windows in warmup (default: 3).
    ///
    /// Each window collects Welford statistics, computes a new `inv_mass`,
    /// resets dual averaging, and re-searches ε. Window sizes double
    /// progressively (Stan-style). Set to 1 for single-pass (original behavior).
    pub n_mass_windows: usize,
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
            max_leapfrog: 8192,
            seed: 42,
            device_ids: None,
            sync_interval: 50,
            welford_chains_per_device: 256,
            batch_size: 1000,
            fused_transitions: 1000,
            report_chains: 32,
            use_diagonal_precond: true,
            n_mass_windows: 3,
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
/// Dispatches to CUDA (f64, multi-GPU) or Metal (f32, single-GPU) backend.
/// CUDA is preferred when available. Metal is used on Apple Silicon when the
/// model is a built-in type (Custom/JIT models require CUDA).
///
/// Returns a [`LapsResult`] containing chains from `n_report_chains` chains
/// (default: min(n_chains, report_chains=32)) subsampled from the full GPU chain set,
/// plus wall time and kernel launch count.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn sample_laps(model: &LapsModel, config: LapsConfig) -> Result<LapsResult> {
    // CUDA path: priority (f64 precision + JIT + multi-GPU)
    #[cfg(feature = "cuda")]
    {
        let device_ids = config.device_ids.clone().unwrap_or_else(detect_cuda_devices);
        if !device_ids.is_empty() {
            return if device_ids.len() == 1 {
                sample_laps_single_gpu_cuda(model, config, device_ids[0])
            } else {
                sample_laps_multi_gpu(model, config, &device_ids)
            };
        }
    }

    // Metal path: fallback on Apple Silicon (built-in models only)
    #[cfg(feature = "metal")]
    {
        if model.is_metal_supported() {
            return sample_laps_single_gpu_metal(model, config);
        }
    }

    Err(ns_core::Error::Computation(
        "No GPU backend available for LAPS (need CUDA or Metal with a built-in model)".into(),
    ))
}

/// Metal single-GPU wrapper: creates MetalMamsAccelerator and calls generic.
#[cfg(feature = "metal")]
fn sample_laps_single_gpu_metal(model: &LapsModel, config: LapsConfig) -> Result<LapsResult> {
    let n_chains = config.n_chains;
    let dim = model.dim();
    let model_data = model.model_data();
    let accel = ns_compute::metal_mams::MetalMamsAccelerator::new(
        n_chains,
        dim,
        model.model_id(),
        &model_data,
        config.seed,
    )?;
    sample_laps_single_gpu_generic(model, config, accel, 0)
}

/// Generate random initial chain state for `n_chains` chains of `dim` dimensions.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn generate_init_state(
    n_chains: usize,
    dim: usize,
    seed: u64,
    init_scale: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    use rand::Rng;
    use rand::SeedableRng;

    let total = n_chains * dim;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x_init = vec![0.0f64; total];
    let mut u_init = vec![0.0f64; total];
    let potential_init = vec![f64::NAN; n_chains];
    let grad_init = vec![0.0f64; total];

    // Scale initial positions by init_scale ≈ posterior_std_est.
    // For GLM with large n: posterior σ ≈ 2/√n ≈ 0.06, so chains start
    // within U(-0.12, 0.12) — near the typical set instead of U(-2,2).
    // For std_normal: init_scale ≈ 1.0 → U(-2,2) — unchanged.
    for c in 0..n_chains {
        for d in 0..dim {
            x_init[c * dim + d] = (rng.random::<f64>() * 4.0 - 2.0) * init_scale;
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
#[cfg(any(feature = "cuda", feature = "metal"))]
fn compute_initial_params(model: &LapsModel, config: &LapsConfig) -> (f64, f64) {
    let dim = model.dim();
    let l_default = std::f64::consts::PI * (dim as f64).sqrt();
    let l = if config.init_l > 0.0 {
        config.init_l
    } else {
        match model {
            // Data-aware default for logistic GLM:
            // posterior scale per coefficient is roughly O(1/sqrt(n)).
            // Use sigma_post ≈ 2/sqrt(n), so L = pi*sqrt(dim)*sigma_post.
            // Empirically:
            // - n_warmup=0 needs more aggressive small L for stability/mixing.
            // - with warmup, a too-small L can hurt mixing quality; keep a floor.
            LapsModel::GlmLogistic { n, .. } => {
                let sigma_post = 2.0 / (*n as f64).sqrt();
                let l_model = l_default * sigma_post;
                if config.n_warmup == 0 {
                    l_model.clamp(0.25, l_default)
                } else {
                    l_model.clamp(0.2 * l_default, l_default)
                }
            }
            _ => l_default,
        }
    };
    let eps = if config.init_step_size > 0.0 { config.init_step_size } else { l / 10.0 };
    (l, eps)
}

/// Find a reasonable initial step size for GPU MAMS by testing short trajectories.
///
/// Uses uniform eps (same for all chains) during binary search.
/// Returns a scalar eps that can be used to initialize per-chain DA.
///
/// Key design: search uses SHORT trajectories (≤50 leapfrog steps) for speed.
/// The full trajectory length (L/eps steps, up to max_leapfrog) is only used
/// during warmup and sampling.
///
/// For cold start: first finds a stable burn-in eps (adapts to gradient scale),
/// runs burn-in, then searches UP from that eps. This handles both easy models
/// (std_normal, eps~1) and data-heavy models (GLM n=5000, eps~0.001).
#[cfg(any(feature = "cuda", feature = "metal"))]
fn find_initial_eps_gpu<A: MamsAccelerator>(
    accel: &mut A,
    l: f64,
    _dim: usize,
    config: &LapsConfig,
    cold_start: bool,
    fast_glm: bool,
) -> Result<(f64, usize)> {
    let search_max_steps: usize = if fast_glm { 32 } else { 50 };
    let eps_floor = (l / (config.max_leapfrog.max(1) as f64)).max(1e-6);
    let stable_probe_iters: usize = if fast_glm { 5 } else { 8 };
    let search_iters: usize = if fast_glm { 12 } else { 20 };
    let search_repeats: usize = if fast_glm { 2 } else { 3 };
    let mut launches = 0usize;

    // Phase A: find a stable eps (one that doesn't immediately diverge).
    // Start at l/10, drop by 10x until acceptance > 1%. For std_normal l/10~1.4
    // works immediately. For GLM n=5000, drops to ~1e-4 after 3 iterations.
    let mut stable_eps = l / 10.0;
    for _ in 0..stable_probe_iters {
        accel.set_uniform_eps(stable_eps)?;
        accel.transition_auto(l, 1, true)?;
        launches += 1;
        let diag = accel.download_diagnostics()?;
        let acc = compute_mean_accept_prob(&diag.energy_error);
        if acc > 0.01 {
            break;
        }
        stable_eps *= 0.1;
        if stable_eps < 1e-8 {
            stable_eps = 1e-8;
            break;
        }
    }

    // Phase B (cold only): burn-in at stable_eps to move toward typical set.
    // With init_scale ≈ posterior_std, chains start within ~2σ of origin.
    // For no-warmup runs we use a shorter burn-in to reduce startup overhead
    // while keeping the robust cold-start behavior.
    if cold_start {
        accel.set_uniform_eps(stable_eps)?;
        let n_steps = ((l / stable_eps).round() as usize).clamp(1, search_max_steps);
        let burnin_iters = if config.n_warmup == 0 {
            if fast_glm { 8 } else { 12 }
        } else if fast_glm {
            20
        } else {
            50
        };
        for _ in 0..burnin_iters {
            accel.transition_auto(l, n_steps, false)?;
            launches += 1;
        }
    }

    // Phase C: search UP from stable_eps to find optimal eps.
    // Increase by 1.5x while acceptance stays > 0.65. When it drops below 0.5,
    // back off. This is a monotonic search — no need for bidirectional.
    let mut eps = stable_eps;
    let mut best_eps = eps;

    for _ in 0..search_iters {
        accel.set_uniform_eps(eps)?;
        let n_steps = ((l / eps).round() as usize).clamp(1, search_max_steps);
        let mut total_accept = 0.0;
        for _ in 0..search_repeats {
            accel.transition_auto(l, n_steps, true)?;
            launches += 1;
            let diag = accel.download_diagnostics()?;
            total_accept += compute_mean_accept_prob(&diag.energy_error);
        }
        let acc = total_accept / search_repeats as f64;

        if acc >= 0.65 {
            best_eps = eps;
            eps *= 1.5;
        } else if acc >= 0.5 {
            best_eps = eps;
            break;
        } else {
            // Overshot — revert to last good eps
            break;
        }
        eps = eps.clamp(eps_floor, l * 0.5);
        if eps >= l * 0.5 {
            best_eps = best_eps.min(l * 0.5);
            break;
        }
    }

    Ok((best_eps.clamp(eps_floor, l * 0.5), launches))
}

/// Tune decoherence length L on GPU using median of per-chain eps.
///
/// Per-chain eps are already uploaded to GPU. We use the median eps
/// to compute n_steps for each L candidate.
/// Returns `(best_l, n_kernel_launches)`.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn tune_l_gpu<A: MamsAccelerator>(
    accel: &mut A,
    eps_median: f64,
    n_chains: usize,
    _dim: usize,
    config: &LapsConfig,
    fast_glm: bool,
) -> Result<(f64, usize)> {
    use crate::adapt::estimate_ess_simple;

    let (n_trials, n_candidates) = if fast_glm {
        if config.n_warmup < 100 { (8, 3) } else { (12, 4) }
    } else if config.n_warmup < 100 {
        (16, 4)
    } else {
        (40, 7)
    };
    // Robust L scoring: aggregate ESS/grad over multiple chains instead of
    // relying on chain 0 only (which is brittle on funnel-like geometries).
    let n_eval_chains =
        n_chains.min(config.welford_chains_per_device.max(1)).min(config.report_chains.max(1));
    let l_min = (std::f64::consts::FRAC_PI_2).max(eps_median * 4.0);

    let mut best_l = l_min;
    let mut best_ess_per_grad = 0.0f64;
    let mut total_launches = 0usize;

    accel.configure_batch(n_trials, n_eval_chains)?;

    let mut l_candidate = l_min;
    for _ in 0..n_candidates {
        // max_leapfrog cap — per-chain n_steps computed in kernel
        let max_leapfrog =
            ((l_candidate / eps_median).round() as usize).clamp(1, config.max_leapfrog) * 2;
        let max_leapfrog = max_leapfrog.min(config.max_leapfrog);

        let batch = accel.transition_batch_auto(l_candidate, max_leapfrog, n_trials, false)?;
        total_launches += batch.n_launches;

        let n_steps_approx =
            ((l_candidate / eps_median).round() as usize).clamp(1, config.max_leapfrog);
        let total_leapfrog = n_trials * n_steps_approx;
        let mut chain_scores: Vec<f64> = Vec::with_capacity(n_eval_chains);

        for c in 0..n_eval_chains {
            let draws: Vec<f64> =
                (0..n_trials).map(|s| batch.potentials[s][c]).filter(|v| v.is_finite()).collect();
            if draws.len() >= 4 {
                let ess = estimate_ess_simple(&draws);
                if ess.is_finite() && ess > 0.0 {
                    chain_scores.push(ess / total_leapfrog as f64);
                }
            }
        }

        if !chain_scores.is_empty() {
            let ess_per_grad = percentile(&chain_scores, 0.5);
            if ess_per_grad.is_finite() && ess_per_grad > best_ess_per_grad {
                best_ess_per_grad = ess_per_grad;
                best_l = l_candidate;
            }
        }

        l_candidate *= 2.0;
    }

    Ok((best_l, total_launches))
}

/// Compute inv_mass from Welford variance with Stan regularization.
#[cfg(any(feature = "cuda", feature = "metal"))]
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
#[cfg(any(feature = "cuda", feature = "metal"))]
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
            // LAPS/MAMS are not tree-based samplers; avoid false max-treedepth failures.
            max_treedepth: usize::MAX,
            step_size: eps,
            mass_diag: mass_diag.clone(),
        })
        .collect();

    SamplerResult { chains, param_names, n_warmup, n_samples, diagnostics: None }
}

/// Compute max_leapfrog cap from per-chain eps and L.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn compute_max_leapfrog(eps_vec: &[f64], l: f64, cap: usize) -> usize {
    eps_vec.iter().map(|&e| ((l / e).round() as usize).clamp(1, cap)).max().unwrap_or(10)
}

/// Compute number of chains retained for diagnostics/result payload.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn compute_report_chains(config: &LapsConfig, n_chains: usize) -> usize {
    n_chains.min(config.report_chains.max(1))
}

/// For short warmup schedules on data-heavy GLM, skip expensive L tuning.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn should_skip_l_tuning(model: &LapsModel, config: &LapsConfig) -> bool {
    config.n_warmup <= 256 && matches!(model, LapsModel::GlmLogistic { .. })
}

/// Decide whether to use fused multi-step kernel in sampling.
///
/// Empirically, for large-n GLM logistic with warp support, the fused path can
/// be slower than warp batch launches despite fewer kernel launches.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn should_use_fused_sampling<A: MamsAccelerator>(
    model: &LapsModel,
    accel: &A,
    config: &LapsConfig,
) -> bool {
    if config.fused_transitions == 0 || !accel.supports_fused() {
        return false;
    }
    match model {
        LapsModel::GlmLogistic { n, .. } if *n >= 1024 && accel.supports_warp() => false,
        _ => true,
    }
}

/// Per-chain DA warmup helper: update DA instances from energy errors
/// and upload new per-chain eps to GPU.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn update_per_chain_da<A: MamsAccelerator>(
    da_vec: &mut [DualAveraging],
    energy_errors: &[f64],
    accel: &mut A,
    n_chains: usize,
    l: f64,
    max_leapfrog: usize,
) -> Result<()> {
    for c in 0..da_vec.len().min(n_chains) {
        let ap_c =
            if energy_errors[c].is_finite() { (-energy_errors[c]).exp().min(1.0) } else { 0.0 };
        da_vec[c].update(ap_c);
    }

    let current_eps: Vec<f64> = da_vec.iter().map(|da| da.current_step_size()).collect();
    let median_eps = if current_eps.is_empty() { l / 10.0 } else { percentile(&current_eps, 0.5) };
    // Robust epsilon floor:
    // 1) avoid single-chain collapse far below population scale
    // 2) avoid tiny effective trajectory length when n_steps hits cap
    let cap_floor = ((l / max_leapfrog.max(1) as f64) * 0.1).max(1e-6);
    let min_eps = (median_eps * 0.01).max(cap_floor).min(l * 0.5);

    let eps_vec: Vec<f64> = (0..n_chains)
        .map(|c| {
            if c < da_vec.len() {
                da_vec[c].current_step_size().min(l * 0.5).max(min_eps)
            } else {
                // For chains beyond DA count, use median.
                median_eps.min(l * 0.5).max(min_eps)
            }
        })
        .collect();
    accel.set_per_chain_eps(&eps_vec)?;
    Ok(())
}

/// Compute LAPS windowed mass adaptation schedule.
///
/// Returns a Vec of window sizes that sum to `total_iters`.
/// Sizes double progressively: `init, 2*init, 4*init, ...`, with
/// the last window absorbing the remainder.
///
/// `init_window = max(total_iters / (2^n_windows - 1), 1)` ensures
/// the geometric series fills the budget, or `max(total/20, 25)` whichever
/// is larger for stability.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn compute_laps_window_schedule(total_iters: usize, n_windows: usize) -> Vec<usize> {
    if total_iters == 0 || n_windows == 0 {
        return vec![];
    }
    if n_windows == 1 {
        return vec![total_iters];
    }

    // Geometric series: init * (2^n - 1) ≈ total → init ≈ total / (2^n - 1)
    let denom = (1usize << n_windows).saturating_sub(1).max(1);
    let init_window = (total_iters / denom).max(1).max(25.min(total_iters));

    let mut sizes = Vec::with_capacity(n_windows);
    let mut remaining = total_iters;
    let mut size = init_window;

    for i in 0..n_windows {
        if i == n_windows - 1 {
            // Last window absorbs remainder
            sizes.push(remaining);
        } else if size >= remaining {
            // Can't fit more windows — give all remaining to this one
            sizes.push(remaining);
            break;
        } else {
            sizes.push(size);
            remaining -= size;
            size *= 2;
        }
    }

    sizes
}

/// Single-GPU LAPS sampling with per-chain step size adaptation.
///
/// Generic over `MamsAccelerator` — the caller creates the backend-specific accelerator
/// and passes the `device_id` for result metadata.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn sample_laps_single_gpu_generic<A: MamsAccelerator>(
    model: &LapsModel,
    config: LapsConfig,
    mut accel: A,
    device_id: usize,
) -> Result<LapsResult> {
    let start = std::time::Instant::now();
    let dim = model.dim();
    let n_chains = config.n_chains;
    let fast_glm = matches!(model, LapsModel::GlmLogistic { .. });

    let (mut l, _init_eps) = compute_initial_params(model, &config);
    // Scale chain initialization: init_scale ≈ posterior_std from init_l.
    // Puts chains within ~2 posterior σ of the origin instead of U(-2,2).
    let init_scale = l / (std::f64::consts::PI * (dim as f64).sqrt());
    let (x_init, u_init, potential_init, grad_init) =
        generate_init_state(n_chains, dim, config.seed, init_scale);
    accel.upload_state(&x_init, &u_init, &potential_init, &grad_init)?;
    let t_init = start.elapsed().as_secs_f64();
    let mut inv_mass = vec![1.0f64; dim];
    let mut n_kernel_launches = 0usize;

    // ---------- Windowed warmup with PER-CHAIN dual averaging ----------
    let phase1_iters = (config.n_warmup as f64 * 0.15) as usize;
    let phase2_iters = (config.n_warmup as f64 * 0.40) as usize;
    let phase3_iters = (config.n_warmup as f64 * 0.15) as usize;
    let phase_final_iters =
        config.n_warmup.saturating_sub(phase1_iters + phase2_iters + phase3_iters);

    // Initial ε binary search (uniform across chains)
    // Cold start: ~8 stability probes + 200 burn-in + up to 20*3 = 60 search ≈ 270
    let (eps, eps_launches) = find_initial_eps_gpu(&mut accel, l, dim, &config, true, fast_glm)?;
    n_kernel_launches += eps_launches;

    // Per-chain DA: one DualAveraging instance per chain
    let mut da_vec: Vec<DualAveraging> =
        (0..n_chains).map(|_| DualAveraging::new(config.target_accept, eps)).collect();
    accel.set_uniform_eps(eps)?;

    // ---------- Phase 1: Per-chain DA — adapt ε per chain, MH enabled ----------
    for iter in 0..phase1_iters {
        let eps_vec: Vec<f64> = da_vec.iter().map(|da| da.current_step_size()).collect();
        let max_lf = compute_max_leapfrog(&eps_vec, l, config.max_leapfrog);
        accel.transition_auto(l, max_lf, true)?;
        n_kernel_launches += 1;

        if (iter + 1) % config.sync_interval == 0 || iter == phase1_iters - 1 {
            let diag = accel.download_diagnostics()?;
            update_per_chain_da(
                &mut da_vec,
                &diag.energy_error,
                &mut accel,
                n_chains,
                l,
                config.max_leapfrog,
            )?;
        }
    }

    // ---------- Phase 2+3: Windowed mass adaptation ----------
    //
    // Instead of a single Welford pass + one DA reset, we run K windows
    // (Stan-style doubling schedule). Each window:
    //   1. Runs DA + Welford for `window_size` iterations
    //   2. Computes new inv_mass from Welford variance
    //   3. Uploads inv_mass to GPU
    //   4. Re-searches eps via binary search
    //   5. Resets DA instances with the new eps
    //   6. Resets Welford
    //
    // The windowed iterations consume Phase 2 + Phase 3 budget combined
    // (55% of warmup). n_mass_windows=1 gives original single-pass behavior.
    let windowed_iters = phase2_iters + phase3_iters;

    let final_eps: Vec<f64>;
    let eps_median: f64;
    let max_lf: usize;
    if config.n_warmup == 0 {
        // Fast path for no-warmup runs:
        // keep stable eps from initial cold search and skip expensive
        // second eps search + L tuning phases.
        final_eps = vec![eps.min(l * 0.5).max(1e-6); n_chains];
        eps_median = final_eps[0];
        accel.set_per_chain_eps(&final_eps)?;
        max_lf = compute_max_leapfrog(&final_eps, l, config.max_leapfrog);
    } else {
        // Compute window schedule: sizes double, last window absorbs remainder.
        let n_windows = config.n_mass_windows.max(1);
        let window_sizes = compute_laps_window_schedule(windowed_iters, n_windows);

        let mut welford = WelfordVariance::new(dim);

        for (_win_idx, &win_size) in window_sizes.iter().enumerate() {
            // Run DA + Welford for this window
            for iter in 0..win_size {
                let eps_vec: Vec<f64> = da_vec.iter().map(|da| da.current_step_size()).collect();
                let max_lf = compute_max_leapfrog(&eps_vec, l, config.max_leapfrog);
                accel.transition_auto(l, max_lf, true)?;
                n_kernel_launches += 1;

                if (iter + 1) % config.sync_interval == 0 || iter == win_size - 1 {
                    let diag = accel.download_diagnostics()?;
                    update_per_chain_da(
                        &mut da_vec,
                        &diag.energy_error,
                        &mut accel,
                        n_chains,
                        l,
                        config.max_leapfrog,
                    )?;

                    if config.use_diagonal_precond {
                        let positions = accel.download_positions()?;
                        for c in 0..n_chains.min(config.welford_chains_per_device) {
                            welford.update(&positions[c * dim..(c + 1) * dim]);
                        }
                    }
                }
            }

            // End of window: update mass matrix, re-search eps, reset DA + Welford
            if config.use_diagonal_precond {
                if let Some(new_mass) = regularize_inv_mass(&welford, dim) {
                    inv_mass = new_mass;
                    accel.set_inv_mass(&inv_mass)?;
                }
            }

            let (new_eps, eps_launches) =
                find_initial_eps_gpu(&mut accel, l, dim, &config, false, fast_glm)?;
            n_kernel_launches += eps_launches;

            da_vec =
                (0..n_chains).map(|_| DualAveraging::new(config.target_accept, new_eps)).collect();
            accel.set_uniform_eps(new_eps)?;

            welford.reset();
        }

        // Finalize per-chain eps (smoothed adapted step sizes)
        final_eps = da_vec.iter().map(|da| da.adapted_step_size().min(l * 0.5).max(1e-6)).collect();
        eps_median = percentile(&final_eps, 0.5);
        accel.set_per_chain_eps(&final_eps)?;

        // ---------- Phase 4: L tuning + equilibrate ----------
        if !should_skip_l_tuning(model, &config) {
            let (tuned_l, l_launches) =
                tune_l_gpu(&mut accel, eps_median, n_chains, dim, &config, fast_glm)?;
            l = tuned_l;
            n_kernel_launches += l_launches;
        }

        max_lf = compute_max_leapfrog(&final_eps, l, config.max_leapfrog);
        for _iter in 0..phase_final_iters {
            accel.transition_auto(l, max_lf, true)?;
            n_kernel_launches += 1;
        }
    }

    let t_warmup = start.elapsed().as_secs_f64() - t_init;

    // ---------- Sampling (batched or fused GPU accumulation) ----------
    let n_report_chains = compute_report_chains(&config, n_chains);
    let use_fused = should_use_fused_sampling(model, &accel, &config);
    let batch_size =
        if use_fused { config.fused_transitions.min(config.batch_size) } else { config.batch_size };
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

    let n_steps_median = ((l / eps_median).round() as usize).clamp(1, config.max_leapfrog);
    let mut samples_collected = 0usize;
    while samples_collected < config.n_samples {
        let this_batch = batch_size.min(config.n_samples - samples_collected);
        let batch = accel.transition_batch_auto(l, max_lf, this_batch, use_fused)?;
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
                chain_leapfrogs[c].push(n_steps_median);
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
        eps_median,
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

/// CUDA single-GPU wrapper: creates CudaMamsAccelerator and calls generic.
#[cfg(feature = "cuda")]
fn sample_laps_single_gpu_cuda(
    model: &LapsModel,
    config: LapsConfig,
    device_id: usize,
) -> Result<LapsResult> {
    let n_chains = config.n_chains;
    let dim = model.dim();
    let model_data = model.model_data();
    let accel = create_accelerator(model, n_chains, dim, &model_data, config.seed, device_id)?;
    sample_laps_single_gpu_generic(model, config, accel, device_id)
}

/// Per-device result returned from each GPU thread.
#[cfg(feature = "cuda")]
struct DeviceLapsResult {
    chains: Vec<Chain>,
    n_kernel_launches: usize,
    warmup_secs: f64,
    sampling_secs: f64,
}

/// Multi-GPU LAPS sampling with per-chain step size adaptation.
///
/// Each device runs a shard of chains with its own `Vec<DualAveraging>`.
/// During warmup, devices synchronize via barriers only for:
///   - Phase 2 Welford mass matrix aggregation
///   - Post-phase 2 mass update + eps binary search broadcast
///   - Post-phase 3 L tuning broadcast
/// Per-chain DA runs independently per device (no cross-device DA sync needed).
/// During sampling, devices run independently.
#[cfg(feature = "cuda")]
fn sample_laps_multi_gpu(
    model: &LapsModel,
    config: LapsConfig,
    device_ids: &[usize],
) -> Result<LapsResult> {
    use std::sync::{Barrier, Mutex};

    let start = std::time::Instant::now();
    let dim = model.dim();
    let fast_glm = matches!(model, LapsModel::GlmLogistic { .. });
    let n_chains = config.n_chains;
    let n_devices = device_ids.len();
    let model_data = model.model_data();

    // Distribute chains across devices
    let base_chains = n_chains / n_devices;
    let remainder = n_chains % n_devices;
    let chains_per_device: Vec<usize> =
        (0..n_devices).map(|i| base_chains + if i < remainder { 1 } else { 0 }).collect();

    // Report chains: distribute min(n_chains, report_chains) across devices
    let total_report = compute_report_chains(&config, n_chains);
    let base_report = total_report / n_devices;
    let report_remainder = total_report % n_devices;
    let report_per_device: Vec<usize> = (0..n_devices)
        .map(|i| {
            let r = base_report + if i < report_remainder { 1 } else { 0 };
            r.min(chains_per_device[i])
        })
        .collect();

    let (l, init_eps) = compute_initial_params(model, &config);
    let init_scale = l / (std::f64::consts::PI * (dim as f64).sqrt());

    // Shared state: inv_mass (from Welford), L (from tuning), init_eps (from binary search)
    let shared_inv_mass = Mutex::new(vec![1.0f64; dim]);
    let shared_init_eps = Mutex::new(init_eps);
    let shared_l = Mutex::new(l);

    // Aggregation buffers for Welford (phase 2 only)
    let shared_positions: Mutex<Vec<Vec<f64>>> =
        Mutex::new((0..n_devices).map(|_| Vec::new()).collect());

    // 4-phase warmup
    let phase1_iters = (config.n_warmup as f64 * 0.15) as usize;
    let phase2_iters = (config.n_warmup as f64 * 0.40) as usize;
    let phase3_iters = (config.n_warmup as f64 * 0.15) as usize;
    let phase_final_iters =
        config.n_warmup.saturating_sub(phase1_iters + phase2_iters + phase3_iters);

    // Count sync points for phase 2 Welford aggregation
    let sync_interval = config.sync_interval;
    fn count_sync_points(total_iters: usize, interval: usize) -> usize {
        if total_iters == 0 {
            return 0;
        }
        let mut count = 0;
        for iter in 0..total_iters {
            if (iter + 1) % interval == 0 || iter == total_iters - 1 {
                count += 1;
            }
        }
        count
    }
    let phase2_syncs = count_sync_points(phase2_iters, sync_interval);
    // Barriers: phase2 Welford syncs + post-phase2 mass update + post-phase3 L tuning
    let total_barrier_pairs = phase2_syncs + 1 + 1;

    let barrier_upload = Barrier::new(n_devices);
    let barrier_aggregated = Barrier::new(n_devices);

    let shared_welford = Mutex::new(WelfordVariance::new(dim));

    let device_results: Vec<Result<DeviceLapsResult>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_devices)
            .map(|dev_idx| {
                let dev_chains = chains_per_device[dev_idx];
                let dev_report = report_per_device[dev_idx];
                let device_id = device_ids[dev_idx];
                let dev_seed = config.seed + dev_idx as u64 * 1_000_000;

                let shared_inv_mass = &shared_inv_mass;
                let shared_init_eps = &shared_init_eps;
                let shared_positions = &shared_positions;
                let barrier_upload = &barrier_upload;
                let barrier_aggregated = &barrier_aggregated;
                let shared_welford = &shared_welford;
                let shared_l = &shared_l;
                let config = &config;
                let model_data = &model_data;
                let fast_glm = fast_glm;

                scope.spawn(move || -> Result<DeviceLapsResult> {
                    let mut remaining_barriers = total_barrier_pairs;

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
                        generate_init_state(dev_chains, dim, dev_seed, init_scale);
                    if let Err(e) =
                        accel.upload_state(&x_init, &u_init, &potential_init, &grad_init)
                    {
                        bail_with_barriers!(e);
                    }

                    let mut n_kernel_launches = 0usize;

                    // Initial ε binary search on device 0, broadcast to all
                    if dev_idx == 0 {
                        if let Ok((found_eps, launches)) =
                            find_initial_eps_gpu(&mut accel, l, dim, config, true, fast_glm)
                        {
                            *shared_init_eps.lock().unwrap() = found_eps;
                            n_kernel_launches += launches;
                        }
                    }
                    let eps = *shared_init_eps.lock().unwrap();

                    // Per-chain DA: one DualAveraging per chain on this device
                    let mut da_vec: Vec<DualAveraging> = (0..dev_chains)
                        .map(|_| DualAveraging::new(config.target_accept, eps))
                        .collect();
                    if let Err(e) = accel.set_uniform_eps(eps) {
                        bail_with_barriers!(e);
                    }

                    // ===== Phase 1 (15%): Per-chain DA — adapt ε per chain =====
                    for iter in 0..phase1_iters {
                        let eps_vec: Vec<f64> =
                            da_vec.iter().map(|da| da.current_step_size()).collect();
                        let max_lf = compute_max_leapfrog(&eps_vec, l, config.max_leapfrog);
                        if let Err(e) = accel.transition_auto(l, max_lf, true) {
                            bail_with_barriers!(e);
                        }
                        n_kernel_launches += 1;

                        if (iter + 1) % sync_interval == 0 || iter == phase1_iters - 1 {
                            let diag = match accel.download_diagnostics() {
                                Ok(d) => d,
                                Err(e) => bail_with_barriers!(e),
                            };
                            if let Err(e) = update_per_chain_da(
                                &mut da_vec,
                                &diag.energy_error,
                                &mut accel,
                                dev_chains,
                                l,
                                config.max_leapfrog,
                            ) {
                                bail_with_barriers!(e);
                            }
                        }
                    }

                    // ===== Phase 2 (40%): Per-chain DA + Welford =====
                    for iter in 0..phase2_iters {
                        let eps_vec: Vec<f64> =
                            da_vec.iter().map(|da| da.current_step_size()).collect();
                        let max_lf = compute_max_leapfrog(&eps_vec, l, config.max_leapfrog);
                        if let Err(e) = accel.transition_auto(l, max_lf, true) {
                            bail_with_barriers!(e);
                        }
                        n_kernel_launches += 1;

                        if (iter + 1) % sync_interval == 0 || iter == phase2_iters - 1 {
                            let diag = match accel.download_diagnostics() {
                                Ok(d) => d,
                                Err(e) => bail_with_barriers!(e),
                            };
                            if let Err(e) = update_per_chain_da(
                                &mut da_vec,
                                &diag.energy_error,
                                &mut accel,
                                dev_chains,
                                l,
                                config.max_leapfrog,
                            ) {
                                bail_with_barriers!(e);
                            }

                            // Upload positions for Welford aggregation
                            let positions = match accel.download_positions() {
                                Ok(p) => p,
                                Err(e) => bail_with_barriers!(e),
                            };
                            {
                                let mut pp = shared_positions.lock().unwrap();
                                pp[dev_idx] = positions;
                            }

                            barrier_upload.wait();

                            if dev_idx == 0 && config.use_diagonal_precond {
                                let pp = shared_positions.lock().unwrap();
                                let mut welford = shared_welford.lock().unwrap();
                                for dev_pos in pp.iter() {
                                    let dev_n = dev_pos.len() / dim;
                                    for c in 0..dev_n.min(config.welford_chains_per_device) {
                                        welford.update(&dev_pos[c * dim..(c + 1) * dim]);
                                    }
                                }
                            }

                            barrier_aggregated.wait();
                            remaining_barriers -= 1;
                        }
                    }

                    // Post-phase 2: mass update (device 0), broadcast, reset DA
                    {
                        barrier_upload.wait();

                        if dev_idx == 0 {
                            if config.use_diagonal_precond {
                                let welford = shared_welford.lock().unwrap();
                                if let Some(new_mass) = regularize_inv_mass(&welford, dim) {
                                    *shared_inv_mass.lock().unwrap() = new_mass;
                                }
                            }
                            // Device 0 runs binary search for new eps after mass update
                        }

                        barrier_aggregated.wait();
                        remaining_barriers -= 1;

                        if config.use_diagonal_precond {
                            let new_mass = shared_inv_mass.lock().unwrap().clone();
                            if let Err(e) = accel.set_inv_mass(&new_mass) {
                                bail_with_barriers!(e);
                            }
                        }

                        // Device 0 finds new eps, broadcasts as init for per-chain DA reset
                        if dev_idx == 0 {
                            let new_eps = match find_initial_eps_gpu(
                                &mut accel, l, dim, config, false, fast_glm,
                            ) {
                                Ok((e, launches)) => {
                                    n_kernel_launches += launches;
                                    e
                                }
                                Err(_) => percentile(
                                    &da_vec
                                        .iter()
                                        .map(|da| da.adapted_step_size())
                                        .collect::<Vec<_>>(),
                                    0.5,
                                ),
                            };
                            *shared_init_eps.lock().unwrap() = new_eps;
                        }
                        let new_eps = *shared_init_eps.lock().unwrap();

                        // Reset per-chain DA with new eps from binary search
                        da_vec = (0..dev_chains)
                            .map(|_| DualAveraging::new(config.target_accept, new_eps))
                            .collect();
                        if let Err(e) = accel.set_uniform_eps(new_eps) {
                            bail_with_barriers!(e);
                        }
                    }

                    // ===== Phase 3 (15%): Per-chain DA with new metric =====
                    let p3_sync = sync_interval.min(phase3_iters.max(1) / 3).max(1);
                    for iter in 0..phase3_iters {
                        let eps_vec: Vec<f64> =
                            da_vec.iter().map(|da| da.current_step_size()).collect();
                        let max_lf = compute_max_leapfrog(&eps_vec, l, config.max_leapfrog);
                        if let Err(e) = accel.transition_auto(l, max_lf, true) {
                            bail_with_barriers!(e);
                        }
                        n_kernel_launches += 1;

                        if (iter + 1) % p3_sync == 0 || iter == phase3_iters - 1 {
                            let diag = match accel.download_diagnostics() {
                                Ok(d) => d,
                                Err(e) => bail_with_barriers!(e),
                            };
                            if let Err(e) = update_per_chain_da(
                                &mut da_vec,
                                &diag.energy_error,
                                &mut accel,
                                dev_chains,
                                l,
                                config.max_leapfrog,
                            ) {
                                bail_with_barriers!(e);
                            }
                        }
                    }

                    // Finalize per-chain eps
                    let final_eps: Vec<f64> = da_vec
                        .iter()
                        .map(|da| da.adapted_step_size().min(l * 0.5).max(1e-6))
                        .collect();
                    let eps_median = percentile(&final_eps, 0.5);
                    if let Err(e) = accel.set_per_chain_eps(&final_eps) {
                        bail_with_barriers!(e);
                    }

                    // Post-phase 3: L tuning on device 0, broadcast
                    {
                        barrier_upload.wait();

                        if dev_idx == 0 && !should_skip_l_tuning(model, config) {
                            match tune_l_gpu(
                                &mut accel, eps_median, dev_chains, dim, config, fast_glm,
                            ) {
                                Ok((tuned_l, launches)) => {
                                    *shared_l.lock().unwrap() = tuned_l;
                                    n_kernel_launches += launches;
                                }
                                Err(_) => {} // keep default L
                            }
                        }

                        barrier_aggregated.wait();
                        remaining_barriers -= 1;
                    }

                    debug_assert_eq!(remaining_barriers, 0);

                    let l = *shared_l.lock().unwrap();
                    let max_lf = compute_max_leapfrog(&final_eps, l, config.max_leapfrog);

                    // ===== Phase 4: equilibrate (independent, per-chain eps) =====
                    for _iter in 0..phase_final_iters {
                        accel.transition_auto(l, max_lf, true)?;
                        n_kernel_launches += 1;
                    }

                    let warmup_secs = start.elapsed().as_secs_f64();

                    // ===== Sampling (independent, batched or fused, per-chain eps) =====
                    let n_steps_median =
                        ((l / eps_median).round() as usize).clamp(1, config.max_leapfrog);
                    let use_fused = should_use_fused_sampling(model, &accel, config);
                    let batch_size = if use_fused {
                        config.fused_transitions.min(config.batch_size)
                    } else {
                        config.batch_size
                    };

                    if dev_report == 0 {
                        let mut remaining = config.n_samples;
                        while remaining > 0 {
                            let this_batch = batch_size.min(remaining);
                            if use_fused || accel.supports_warp() {
                                accel.configure_batch(this_batch, 1)?;
                                let batch = accel
                                    .transition_batch_auto(l, max_lf, this_batch, use_fused)?;
                                n_kernel_launches += batch.n_launches;
                            } else {
                                for _ in 0..this_batch {
                                    accel.transition_auto(l, max_lf, true)?;
                                    n_kernel_launches += 1;
                                }
                            }
                            remaining -= this_batch;
                        }
                        let sampling_secs = start.elapsed().as_secs_f64() - warmup_secs;
                        return Ok(DeviceLapsResult {
                            chains: Vec::new(),
                            n_kernel_launches,
                            warmup_secs,
                            sampling_secs,
                        });
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
                        let batch =
                            accel.transition_batch_auto(l, max_lf, this_batch, use_fused)?;
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
                                chain_leapfrogs[c].push(n_steps_median);
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
                            // LAPS/MAMS are not tree-based samplers; avoid false max-treedepth failures.
                            max_treedepth: usize::MAX,
                            step_size: eps_median,
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

/// Compute mean acceptance probability from energy errors (standard).
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
        assert_eq!(result.sampler_result.chains.len(), 32);
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

#[cfg(test)]
#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod tests_metal {
    use super::*;

    #[test]
    fn test_laps_metal_std_normal_smoke() {
        let model = LapsModel::StdNormal { dim: 10 };
        let config = LapsConfig {
            n_chains: 256,
            n_warmup: 100,
            n_samples: 200,
            seed: 42,
            ..Default::default()
        };
        let result = sample_laps(&model, config).unwrap();
        assert_eq!(result.sampler_result.chains.len(), 32);
        assert_eq!(result.sampler_result.chains[0].draws_constrained.len(), 200);
        assert!(result.wall_time_s > 0.0);
        assert!(result.n_kernel_launches > 0);

        // Verify posterior moments: mean ≈ 0, var ≈ 1 (f32 tolerance)
        let chain = &result.sampler_result.chains[0];
        let n_draws = chain.draws_constrained.len();
        let dim = chain.draws_constrained[0].len();
        for d in 0..dim {
            let mean: f64 =
                chain.draws_constrained.iter().map(|s| s[d]).sum::<f64>() / n_draws as f64;
            assert!(mean.abs() < 0.5, "dim {d} mean = {mean}, expected ≈ 0");
        }
    }

    #[test]
    fn test_laps_metal_eight_schools_smoke() {
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
    fn test_laps_metal_neal_funnel_smoke() {
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

    #[test]
    fn test_laps_metal_custom_rejected() {
        // Custom (JIT) models are not supported on Metal — should return error.
        let model = LapsModel::Custom {
            dim: 2,
            param_names: vec!["x".into(), "y".into()],
            model_data: vec![],
            cuda_src: "// dummy".into(),
        };
        let config = LapsConfig {
            n_chains: 64,
            n_warmup: 10,
            n_samples: 10,
            seed: 42,
            ..Default::default()
        };
        let result = sample_laps(&model, config);
        assert!(result.is_err(), "Custom model should fail on Metal-only");
    }
}
