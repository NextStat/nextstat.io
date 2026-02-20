//! CUDA accelerator for MAMS (Metropolis-Adjusted Microcanonical Sampler).
//!
//! Wraps the `mams_transition` CUDA kernel. Manages persistent per-chain state
//! on device (positions, velocities, potentials, gradients). The host drives
//! warmup adaptation and convergence diagnostics; the GPU executes trajectories.
//!
//! Architecture: 1 CUDA thread = 1 chain. For d ≤ 128, all state fits in registers.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::cuda_glm_cublas::CudaGlmCublasEvaluator;

const PTX_SRC: &str = include_str!(env!("CUDA_MAMS_PTX_PATH"));
const MAMS_CUDA_SRC: &str = include_str!("../kernels/mams_leapfrog.cu");
const MAMS_KERNEL_INCLUDE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels");
const WARP_SHMEM_LIMIT_BYTES: usize = 48 * 1024;

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA mams: {msg}"))
}

fn detect_cuda_compute_capability(device_id: usize) -> ns_core::Result<(i32, i32)> {
    use cudarc::driver::result;
    use cudarc::driver::sys;

    unsafe {
        result::init().map_err(|e| cuda_err(format!("cuInit: {e}")))?;
        let dev = result::device::get(device_id as i32)
            .map_err(|e| cuda_err(format!("cuDeviceGet({device_id}): {e}")))?;
        let major = result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|e| cuda_err(format!("get CC major (device {device_id}): {e}")))?;
        let minor = result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|e| cuda_err(format!("get CC minor (device {device_id}): {e}")))?;
        Ok((major, minor))
    }
}

fn load_precompiled_mams_kernels(
    ctx: &Arc<CudaContext>,
) -> ns_core::Result<(CudaFunction, Option<CudaFunction>, Option<CudaFunction>, Option<CudaFunction>)> {
    let ptx = Ptx::from_src(PTX_SRC);
    let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load module: {e}")))?;
    let kernel = module
        .load_function("mams_transition")
        .map_err(|e| cuda_err(format!("load function mams_transition: {e}")))?;
    let kernel_fused = module.load_function("mams_transition_fused").ok();
    let kernel_warp = module.load_function("mams_transition_warp").ok();
    let kernel_warp_hi = module.load_function("mams_transition_warp_hi").ok();
    Ok((kernel, kernel_fused, kernel_warp, kernel_warp_hi))
}

fn load_nvrtc_mams_kernels(
    ctx: &Arc<CudaContext>,
    device_id: usize,
) -> ns_core::Result<(CudaFunction, Option<CudaFunction>, Option<CudaFunction>, Option<CudaFunction>)> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    let (major, minor) = detect_cuda_compute_capability(device_id)?;
    let include_opt = format!("--include-path={MAMS_KERNEL_INCLUDE_DIR}");
    let mut last_err = String::new();

    // CUDA 13 toolchains can drop support for older arch flags (e.g. sm_70 on V100 images).
    // Try a short cascade and end with NVRTC default arch as final fallback.
    let mut arch_attempts = vec![
        Some(format!("--gpu-architecture=compute_{major}{minor}")),
        Some(format!("--gpu-architecture=sm_{major}{minor}")),
        Some("--gpu-architecture=compute_75".to_string()),
        Some("--gpu-architecture=sm_75".to_string()),
        None, // let NVRTC choose default virtual arch
    ];
    arch_attempts.dedup();

    for arch_opt in arch_attempts {
        let mut options = vec![include_opt.clone()];
        if let Some(opt) = arch_opt.clone() {
            options.push(opt);
        }
        let opts = CompileOptions {
            prec_sqrt: Some(true),
            prec_div: Some(true),
            arch: None,
            options,
            ..Default::default()
        };

        let arch_label = arch_opt.clone().unwrap_or_else(|| "default-arch".to_string());

        let ptx = match compile_ptx_with_opts(MAMS_CUDA_SRC, opts) {
            Ok(ptx) => ptx,
            Err(e) => {
                last_err = format!("{arch_label} compile: {e}");
                continue;
            }
        };

        let module = match ctx.load_module(ptx) {
            Ok(module) => module,
            Err(e) => {
                last_err = format!("{arch_label} load_module: {e}");
                continue;
            }
        };

        let kernel = match module.load_function("mams_transition") {
            Ok(kernel) => kernel,
            Err(e) => {
                last_err = format!("{arch_label} load_function(mams_transition): {e}");
                continue;
            }
        };

        let kernel_fused = module.load_function("mams_transition_fused").ok();
        let kernel_warp = module.load_function("mams_transition_warp").ok();
        let kernel_warp_hi = module.load_function("mams_transition_warp_hi").ok();
        return Ok((kernel, kernel_fused, kernel_warp, kernel_warp_hi));
    }

    Err(cuda_err(format!("NVRTC mams_leapfrog failed after arch fallback cascade: {last_err}")))
}

/// CUDA accelerator for MAMS transitions.
///
/// Persistent GPU buffers for chain state. 1 kernel launch = 1 transition
/// for all chains simultaneously.
///
/// Supports batch mode: `configure_batch()` allocates GPU-side accumulation
/// buffers, then `transition_batch()` launches multiple kernels without
/// intermediate host-device sync. One bulk download per batch.
pub struct CudaMamsAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    kernel_fused: Option<CudaFunction>,
    kernel_warp: Option<CudaFunction>,
    kernel_warp_hi: Option<CudaFunction>,

    // Per-chain state [n_chains × dim]
    d_x: CudaSlice<f64>,
    d_u: CudaSlice<f64>,
    d_potential: CudaSlice<f64>,
    d_grad: CudaSlice<f64>,

    // Sampler parameters
    d_inv_mass: CudaSlice<f64>,
    d_model_data: CudaSlice<f64>,

    // Per-chain output (overwritten each transition)
    d_accepted: CudaSlice<i32>,
    d_energy_error: CudaSlice<f64>,

    // Accumulation buffers for batch sampling
    d_sample_buf: CudaSlice<f64>,      // [batch_stride × n_report × dim]
    d_accum_potential: CudaSlice<f64>, // [batch_stride × n_report]
    d_accum_accepted: CudaSlice<i32>,  // [batch_stride × n_report]
    d_accum_energy: CudaSlice<f64>,    // [batch_stride × n_report]
    batch_stride: usize,
    n_report: usize,

    // Per-chain step sizes [n_chains]
    d_eps_per_chain: CudaSlice<f64>,

    n_chains: usize,
    dim: usize,
    model_id: i32,
    seed: u64,
    iteration: i32,

    // Warp kernel data dimensions (for shared memory sizing)
    n_obs: usize,
    n_feat: usize,
    // Optional cuBLAS evaluator for GLM initialization.
    glm_cublas_eval: Option<CudaGlmCublasEvaluator>,
    // Energy error threshold for divergence detection (default: 1000.0).
    divergence_threshold: f64,
}

pub use crate::mams_trait::{BatchResult, TransitionDiagnostics};

impl CudaMamsAccelerator {
    /// Check CUDA availability.
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Create a MAMS accelerator on a specific GPU device.
    pub fn new_on_device(
        n_chains: usize,
        dim: usize,
        model_id: i32,
        model_data: &[f64],
        seed: u64,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        let ctx = CudaContext::new(device_id)
            .map_err(|e| cuda_err(format!("context (device {device_id}): {e}")))?;
        let stream = ctx.default_stream();

        let (kernel, kernel_fused, kernel_warp, kernel_warp_hi) = match load_precompiled_mams_kernels(&ctx) {
            Ok(kernels) => kernels,
            Err(pre_err) => {
                log::warn!(
                    "CUDA mams: precompiled PTX unavailable on device {} ({}), falling back to NVRTC JIT",
                    device_id,
                    pre_err
                );
                load_nvrtc_mams_kernels(&ctx, device_id)?
            }
        };

        // Extract data dimensions for warp kernel shared memory sizing.
        // GLM models: model_data starts with [n, p, ...] (id 3,6) or [n, p, offset_flag, ...] (id 7,8)
        // or [n, p, G, ...] (id 9).
        let (n_obs, n_feat) = if [3, 6, 7, 8, 9].contains(&model_id) && model_data.len() >= 2 {
            (model_data[0] as usize, model_data[1] as usize)
        } else {
            (0, 0)
        };
        let mut glm_cublas_eval = None;
        if (model_id == 3 || model_id == 6) && n_obs > 0 && n_feat > 0 && dim == n_feat {
            let np = n_obs * n_feat;
            let y_off = 2 + np;
            let x_col_off = y_off + n_obs;
            let has_layout = model_data.len() >= x_col_off + np;
            let disable = std::env::var_os("NEXTSTAT_DISABLE_LAPS_GLM_CUBLAS_INIT").is_some();
            let strict = std::env::var_os("NEXTSTAT_STRICT_LAPS_GLM_CUBLAS_INIT").is_some();
            if has_layout && !disable {
                let y = &model_data[y_off..y_off + n_obs];
                let x_col = &model_data[x_col_off..x_col_off + np];
                match CudaGlmCublasEvaluator::new_on_device(
                    x_col, y, n_obs, n_feat, n_chains, device_id,
                ) {
                    Ok(eval) => {
                        log::info!(
                            "LAPS GLM cuBLAS evaluator enabled (device={}, n={}, p={}, chains={})",
                            device_id,
                            n_obs,
                            n_feat,
                            n_chains
                        );
                        glm_cublas_eval = Some(eval);
                    }
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        log::warn!(
                            "LAPS GLM cuBLAS evaluator unavailable on device {}: {}",
                            device_id,
                            e
                        );
                    }
                }
            }
        }

        let total = n_chains * dim;

        // Allocate persistent buffers (zeroed)
        let d_x = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_u = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_potential = stream.memcpy_stod(&vec![0.0f64; n_chains]).map_err(cuda_err)?;
        let d_grad = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_inv_mass = stream.memcpy_stod(&vec![1.0f64; dim]).map_err(cuda_err)?;
        let d_model_data = if model_data.is_empty() {
            stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?
        } else {
            stream.memcpy_stod(model_data).map_err(cuda_err)?
        };
        let d_accepted = stream.memcpy_stod(&vec![0i32; n_chains]).map_err(cuda_err)?;
        let d_energy_error = stream.memcpy_stod(&vec![0.0f64; n_chains]).map_err(cuda_err)?;

        // Accumulation buffers (minimal — reallocated by configure_batch)
        let d_sample_buf = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;
        let d_accum_potential = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;
        let d_accum_accepted = stream.memcpy_stod(&[0i32]).map_err(cuda_err)?;
        let d_accum_energy = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;

        // Per-chain step sizes (initialized to 1.0, overwritten by host)
        let d_eps_per_chain = stream.memcpy_stod(&vec![1.0f64; n_chains]).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            kernel_fused,
            kernel_warp,
            kernel_warp_hi,
            d_x,
            d_u,
            d_potential,
            d_grad,
            d_inv_mass,
            d_model_data,
            d_accepted,
            d_energy_error,
            d_sample_buf,
            d_accum_potential,
            d_accum_accepted,
            d_accum_energy,
            batch_stride: 0,
            n_report: 0,
            d_eps_per_chain,
            n_chains,
            dim,
            model_id,
            seed,
            iteration: 0,
            n_obs,
            n_feat,
            glm_cublas_eval,
            divergence_threshold: 1000.0,
        })
    }

    /// Create a new MAMS accelerator on GPU 0 (backward-compatible).
    pub fn new(
        n_chains: usize,
        dim: usize,
        model_id: i32,
        model_data: &[f64],
        seed: u64,
    ) -> ns_core::Result<Self> {
        Self::new_on_device(n_chains, dim, model_id, model_data, seed, 0)
    }

    /// Create a MAMS accelerator with a JIT-compiled user model on a specific device.
    pub fn new_jit_on_device(
        n_chains: usize,
        dim: usize,
        model_data: &[f64],
        user_cuda_src: &str,
        seed: u64,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        use crate::nvrtc_mams::MamsJitCompiler;

        let compiler = MamsJitCompiler::new_for_device(device_id)?;
        let ptx_str = compiler.compile(user_cuda_src)?;

        let ctx = CudaContext::new(device_id)
            .map_err(|e| cuda_err(format!("context (device {device_id}): {e}")))?;
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(ptx_str);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load JIT module: {e}")))?;
        let kernel = module
            .load_function("mams_transition")
            .map_err(|e| cuda_err(format!("load JIT function: {e}")))?;
        let kernel_fused = module.load_function("mams_transition_fused").ok();
        let kernel_warp = module.load_function("mams_transition_warp").ok();
        let kernel_warp_hi = module.load_function("mams_transition_warp_hi").ok();

        let total = n_chains * dim;

        let d_x = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_u = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_potential = stream.memcpy_stod(&vec![0.0f64; n_chains]).map_err(cuda_err)?;
        let d_grad = stream.memcpy_stod(&vec![0.0f64; total]).map_err(cuda_err)?;
        let d_inv_mass = stream.memcpy_stod(&vec![1.0f64; dim]).map_err(cuda_err)?;
        let d_model_data = if model_data.is_empty() {
            stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?
        } else {
            stream.memcpy_stod(model_data).map_err(cuda_err)?
        };
        let d_accepted = stream.memcpy_stod(&vec![0i32; n_chains]).map_err(cuda_err)?;
        let d_energy_error = stream.memcpy_stod(&vec![0.0f64; n_chains]).map_err(cuda_err)?;

        let d_sample_buf = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;
        let d_accum_potential = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;
        let d_accum_accepted = stream.memcpy_stod(&[0i32]).map_err(cuda_err)?;
        let d_accum_energy = stream.memcpy_stod(&[0.0f64]).map_err(cuda_err)?;

        let d_eps_per_chain = stream.memcpy_stod(&vec![1.0f64; n_chains]).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            kernel_fused,
            kernel_warp,
            kernel_warp_hi,
            d_x,
            d_u,
            d_potential,
            d_grad,
            d_inv_mass,
            d_model_data,
            d_accepted,
            d_energy_error,
            d_sample_buf,
            d_accum_potential,
            d_accum_accepted,
            d_accum_energy,
            batch_stride: 0,
            n_report: 0,
            d_eps_per_chain,
            n_chains,
            dim,
            model_id: -1, // unused for JIT models
            seed,
            iteration: 0,
            n_obs: 0,
            n_feat: 0,
            glm_cublas_eval: None,
            divergence_threshold: 1000.0,
        })
    }

    /// Create a MAMS accelerator with a JIT-compiled user model on GPU 0 (backward-compatible).
    pub fn new_jit(
        n_chains: usize,
        dim: usize,
        model_data: &[f64],
        user_cuda_src: &str,
        seed: u64,
    ) -> ns_core::Result<Self> {
        Self::new_jit_on_device(n_chains, dim, model_data, user_cuda_src, seed, 0)
    }

    /// Upload chain state from host.
    ///
    /// - `x`: positions `[n_chains × dim]` (row-major)
    /// - `u`: unit velocities `[n_chains × dim]`
    /// - `potential`: potentials `[n_chains]`
    /// - `grad`: gradients `[n_chains × dim]`
    pub fn upload_state(
        &mut self,
        x: &[f64],
        u: &[f64],
        potential: &[f64],
        grad: &[f64],
    ) -> ns_core::Result<()> {
        let total = self.n_chains * self.dim;
        assert_eq!(x.len(), total);
        assert_eq!(u.len(), total);
        assert_eq!(potential.len(), self.n_chains);
        assert_eq!(grad.len(), total);

        let mut potential_upload = potential.to_vec();
        let mut grad_upload = grad.to_vec();

        // GLM-specific startup fast-path:
        // if host provides non-finite initial potential (typical cold start),
        // seed potential/grad from cuBLAS evaluator before first transition.
        if potential_upload.iter().any(|v| !v.is_finite()) {
            if let Some(eval) = self.glm_cublas_eval.as_mut() {
                if let Ok((grad0, nll0)) = eval.evaluate_host(x) {
                    if grad0.len() == total
                        && nll0.len() == self.n_chains
                        && nll0.iter().all(|v| v.is_finite())
                    {
                        grad_upload = grad0;
                        potential_upload = nll0;
                    }
                }
            }
        }

        self.stream.memcpy_htod(x, &mut self.d_x).map_err(cuda_err)?;
        self.stream.memcpy_htod(u, &mut self.d_u).map_err(cuda_err)?;
        self.stream.memcpy_htod(&potential_upload, &mut self.d_potential).map_err(cuda_err)?;
        self.stream.memcpy_htod(&grad_upload, &mut self.d_grad).map_err(cuda_err)?;

        Ok(())
    }

    /// Upload diagonal inverse mass matrix `[dim]`.
    pub fn set_inv_mass(&mut self, inv_mass: &[f64]) -> ns_core::Result<()> {
        assert_eq!(inv_mass.len(), self.dim);
        self.stream.memcpy_htod(inv_mass, &mut self.d_inv_mass).map_err(cuda_err)?;
        Ok(())
    }

    /// Upload per-chain step sizes `[n_chains]`.
    pub fn set_per_chain_eps(&mut self, eps_vec: &[f64]) -> ns_core::Result<()> {
        assert_eq!(eps_vec.len(), self.n_chains);
        self.stream.memcpy_htod(eps_vec, &mut self.d_eps_per_chain).map_err(cuda_err)?;
        Ok(())
    }

    /// Fill per-chain eps buffer with a uniform value (convenience for warmup).
    pub fn set_uniform_eps(&mut self, eps: f64) -> ns_core::Result<()> {
        let uniform = vec![eps; self.n_chains];
        self.set_per_chain_eps(&uniform)
    }

    /// Optional GLM pre-solve using cuBLAS batched gradients.
    ///
    /// Runs a short normalized gradient-descent loop on host-side `x` using the
    /// cuBLAS evaluator, then uploads `(x, grad, potential)` back to GPU state.
    /// This is model-specific (GLM logistic only) and intended to reduce warmup
    /// transients for large-`n` data-heavy runs.
    pub fn glm_presolve(&mut self, max_iters: usize) -> ns_core::Result<bool> {
        if max_iters == 0 || (self.model_id != 3 && self.model_id != 6) {
            return Ok(false);
        }
        if std::env::var_os("NEXTSTAT_DISABLE_LAPS_GLM_PRESOLVE").is_some() {
            return Ok(false);
        }
        let Some(eval) = self.glm_cublas_eval.as_mut() else {
            return Ok(false);
        };

        let total = self.n_chains * self.dim;
        let mut x = vec![0.0f64; total];
        self.stream.memcpy_dtoh(&self.d_x, &mut x).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        let sigma_post =
            if self.n_obs > 0 { (2.0 / (self.n_obs as f64).sqrt()).clamp(0.02, 1.0) } else { 1.0 };
        let base_step = std::env::var("NEXTSTAT_LAPS_GLM_PRESOLVE_STEP")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.20)
            .max(1e-6);
        let min_step = std::env::var("NEXTSTAT_LAPS_GLM_PRESOLVE_MIN_STEP")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.02)
            .max(1e-6);
        let x_clip = std::env::var("NEXTSTAT_LAPS_GLM_PRESOLVE_X_CLIP")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(20.0)
            .max(1.0);

        let mut grad_rms_prev = f64::INFINITY;
        let mut applied_iters = 0usize;

        for iter in 0..max_iters {
            let (grad, nll) = eval.evaluate_host(&x)?;
            if grad.len() != total || nll.len() != self.n_chains {
                break;
            }
            if nll.iter().any(|v| !v.is_finite()) || grad.iter().any(|g| !g.is_finite()) {
                break;
            }

            let t = if max_iters > 1 { iter as f64 / (max_iters - 1) as f64 } else { 0.0 };
            // Normalized GD is already normalized by ||grad|| — no need to scale
            // by sigma_post. But for small-n models (sigma_post < 1), clamp step
            // to avoid overshooting the mode.
            let raw_step = (1.0 - t) * base_step + t * min_step;
            let step = if sigma_post < 1.0 { raw_step.min(2.0 * sigma_post) } else { raw_step };

            let mut grad_norm_sq_sum = 0.0f64;
            for c in 0..self.n_chains {
                let row = &grad[c * self.dim..(c + 1) * self.dim];
                let mut g2 = 0.0f64;
                for &g in row {
                    g2 += g * g;
                }
                let gnorm = g2.sqrt();
                grad_norm_sq_sum += g2;
                if gnorm > 1e-14 {
                    let scale = step / gnorm;
                    for d in 0..self.dim {
                        let idx = c * self.dim + d;
                        x[idx] = (x[idx] - scale * row[d]).clamp(-x_clip, x_clip);
                    }
                }
            }

            applied_iters += 1;
            let grad_rms = (grad_norm_sq_sum / (total.max(1) as f64)).sqrt();
            let rel_improve = (grad_rms_prev - grad_rms).abs() / grad_rms_prev.max(1e-12);
            grad_rms_prev = grad_rms;
            if grad_rms < 1e-3 || (iter >= 12 && rel_improve < 5e-4) {
                break;
            }
        }

        if applied_iters == 0 {
            return Ok(false);
        }

        // Perturb chains by ~σ_post so they don't all start at the same mode.
        // Uses xorshift64 + Box-Muller to avoid adding rand dependency.
        {
            let mut rng_state: u64 = 0xCAFE_BABE_DEAD_BEEF;
            let next_u64 = |s: &mut u64| -> u64 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                *s
            };
            let noise_scale = std::env::var("NEXTSTAT_LAPS_GLM_PRESOLVE_JITTER")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.20)
                .max(0.0);
            let noise_std = noise_scale * sigma_post;
            for idx in 0..total {
                let u1 = (next_u64(&mut rng_state) as f64 / u64::MAX as f64).max(1e-12);
                let u2 = next_u64(&mut rng_state) as f64 / u64::MAX as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                x[idx] = (x[idx] + noise_std * z).clamp(-x_clip, x_clip);
            }
        }

        let (grad_final, nll_final) = eval.evaluate_host(&x)?;
        if grad_final.len() != total || nll_final.len() != self.n_chains {
            return Ok(false);
        }
        if nll_final.iter().any(|v| !v.is_finite()) || grad_final.iter().any(|g| !g.is_finite()) {
            return Ok(false);
        }

        self.stream.memcpy_htod(&x, &mut self.d_x).map_err(cuda_err)?;
        self.stream.memcpy_htod(&grad_final, &mut self.d_grad).map_err(cuda_err)?;
        self.stream.memcpy_htod(&nll_final, &mut self.d_potential).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(true)
    }

    /// Run one MAMS transition for all chains.
    ///
    /// Uses per-chain eps from the GPU buffer (set via `set_per_chain_eps` or `set_uniform_eps`).
    /// - `l`: decoherence length
    /// - `max_leapfrog`: cap on per-chain leapfrog steps
    /// - `enable_mh`: false = Phase 1 (unadjusted), true = Phase 2 (exact)
    pub fn transition(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let block_dim = 256u32;
        let grid_dim = ((self.n_chains as u32 + block_dim - 1) / block_dim).min(65535);

        let config = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let max_leapfrog_arg = max_leapfrog as i32;
        let dim_arg = self.dim as i32;
        let enable_mh_arg: i32 = if enable_mh { 1 } else { 0 };
        let model_id_arg = self.model_id;
        let seed_arg = self.seed;
        let iteration_arg = self.iteration;
        let n_chains_arg = self.n_chains as i32;

        let store_idx_arg: i32 = 0;
        let n_report_arg: i32 = 0; // disabled for single transition

        let mut builder = self.stream.launch_builder(&self.kernel);
        builder.arg(&mut self.d_x);
        builder.arg(&mut self.d_u);
        builder.arg(&mut self.d_potential);
        builder.arg(&mut self.d_grad);
        builder.arg(&self.d_eps_per_chain);
        builder.arg(&l);
        builder.arg(&max_leapfrog_arg);
        builder.arg(&dim_arg);
        builder.arg(&self.d_inv_mass);
        builder.arg(&enable_mh_arg);
        builder.arg(&self.d_model_data);
        builder.arg(&model_id_arg);
        builder.arg(&seed_arg);
        builder.arg(&iteration_arg);
        builder.arg(&n_chains_arg);
        builder.arg(&mut self.d_accepted);
        builder.arg(&mut self.d_energy_error);
        // Accumulation buffers (disabled: n_report=0)
        builder.arg(&mut self.d_sample_buf);
        builder.arg(&mut self.d_accum_potential);
        builder.arg(&mut self.d_accum_accepted);
        builder.arg(&mut self.d_accum_energy);
        builder.arg(&store_idx_arg);
        builder.arg(&n_report_arg);
        builder.arg(&self.divergence_threshold);

        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch: {e}")))?;
        }

        self.iteration += 1;

        Ok(())
    }

    /// Configure GPU-side accumulation buffers for batch sampling.
    ///
    /// - `batch_size`: number of transitions per batch (e.g. 100)
    /// - `n_report`: number of chains to accumulate per transition (first n_report chains)
    pub fn configure_batch(&mut self, batch_size: usize, n_report: usize) -> ns_core::Result<()> {
        assert!(n_report <= self.n_chains);
        self.batch_stride = batch_size;
        self.n_report = n_report;

        let total_slots = batch_size * n_report;
        let total_pos = total_slots * self.dim;

        self.d_sample_buf = self.stream.memcpy_stod(&vec![0.0f64; total_pos]).map_err(cuda_err)?;
        self.d_accum_potential =
            self.stream.memcpy_stod(&vec![0.0f64; total_slots]).map_err(cuda_err)?;
        self.d_accum_accepted =
            self.stream.memcpy_stod(&vec![0i32; total_slots]).map_err(cuda_err)?;
        self.d_accum_energy =
            self.stream.memcpy_stod(&vec![0.0f64; total_slots]).map_err(cuda_err)?;

        Ok(())
    }

    /// Run `batch_size` MAMS transitions without intermediate host-device sync.
    ///
    /// Uses per-chain eps from the GPU buffer.
    /// Must call `configure_batch()` first.
    pub fn transition_batch(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let block_dim = 256u32;
        let grid_dim = ((self.n_chains as u32 + block_dim - 1) / block_dim).min(65535);
        let config = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let max_leapfrog_arg = max_leapfrog as i32;
        let dim_arg = self.dim as i32;
        let enable_mh_arg: i32 = 1; // batch = always Phase 2
        let model_id_arg = self.model_id;
        let seed_arg = self.seed;
        let n_chains_arg = self.n_chains as i32;
        let n_report_arg = self.n_report as i32;

        // Launch batch_size kernels without intermediate sync
        for s in 0..batch_size {
            let iteration_arg = self.iteration;
            let store_idx_arg = s as i32;

            let mut builder = self.stream.launch_builder(&self.kernel);
            builder.arg(&mut self.d_x);
            builder.arg(&mut self.d_u);
            builder.arg(&mut self.d_potential);
            builder.arg(&mut self.d_grad);
            builder.arg(&self.d_eps_per_chain);
            builder.arg(&l);
            builder.arg(&max_leapfrog_arg);
            builder.arg(&dim_arg);
            builder.arg(&self.d_inv_mass);
            builder.arg(&enable_mh_arg);
            builder.arg(&self.d_model_data);
            builder.arg(&model_id_arg);
            builder.arg(&seed_arg);
            builder.arg(&iteration_arg);
            builder.arg(&n_chains_arg);
            builder.arg(&mut self.d_accepted);
            builder.arg(&mut self.d_energy_error);
            // Accumulation buffers (enabled)
            builder.arg(&mut self.d_sample_buf);
            builder.arg(&mut self.d_accum_potential);
            builder.arg(&mut self.d_accum_accepted);
            builder.arg(&mut self.d_accum_energy);
            builder.arg(&store_idx_arg);
            builder.arg(&n_report_arg);
            builder.arg(&self.divergence_threshold);

            unsafe {
                builder.launch(config).map_err(|e| cuda_err(format!("launch batch[{s}]: {e}")))?;
            }

            self.iteration += 1;
        }

        // Single sync + bulk download
        self.stream.synchronize().map_err(cuda_err)?;

        let nr = self.n_report;
        let dim = self.dim;
        // Download full GPU buffers (batch_stride), then use only batch_size slots
        let gpu_slots = self.batch_stride * nr;
        let gpu_pos = gpu_slots * dim;

        let mut pos_flat = vec![0.0f64; gpu_pos];
        let mut pot_flat = vec![0.0f64; gpu_slots];
        let mut acc_flat = vec![0i32; gpu_slots];
        let mut eng_flat = vec![0.0f64; gpu_slots];

        self.stream.memcpy_dtoh(&self.d_sample_buf, &mut pos_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_potential, &mut pot_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_accepted, &mut acc_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_energy, &mut eng_flat).map_err(cuda_err)?;

        // Reshape into per-sample vectors
        let mut positions = Vec::with_capacity(batch_size);
        let mut potentials = Vec::with_capacity(batch_size);
        let mut accepted = Vec::with_capacity(batch_size);
        let mut energy_error = Vec::with_capacity(batch_size);

        for s in 0..batch_size {
            let slot_start = s * nr;
            let pos_start = slot_start * dim;
            positions.push(pos_flat[pos_start..pos_start + nr * dim].to_vec());
            potentials.push(pot_flat[slot_start..slot_start + nr].to_vec());
            accepted.push(acc_flat[slot_start..slot_start + nr].to_vec());
            energy_error.push(eng_flat[slot_start..slot_start + nr].to_vec());
        }

        Ok(BatchResult { positions, potentials, accepted, energy_error, n_launches: batch_size })
    }

    /// Whether the fused multi-step kernel is available.
    pub fn supports_fused(&self) -> bool {
        self.kernel_fused.is_some()
    }

    /// Whether the warp-cooperative kernel is available and beneficial.
    pub fn supports_warp(&self) -> bool {
        if self.kernel_warp.is_none() || self.n_obs < 1024 {
            return false;
        }
        if self.dim > 32 {
            return false;
        }
        if self.n_feat == 0 {
            return false;
        }
        true
    }

    /// Whether the high-dim warp kernel (dim ≤ 96) is available and beneficial.
    pub fn supports_warp_hi(&self) -> bool {
        if self.kernel_warp_hi.is_none() || self.n_obs < 1024 {
            return false;
        }
        if self.dim <= 32 || self.dim > 96 {
            return false;
        }
        if self.n_feat == 0 {
            return false;
        }
        true
    }

    /// Internal: launch one warp transition with the given kernel function.
    fn transition_warp_impl(
        &mut self,
        kernel: &CudaFunction,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
        label: &str,
    ) -> ns_core::Result<()> {
        let block_dim = 256u32;
        let total_threads = self.n_chains as u32 * 32;
        let grid_dim = ((total_threads + block_dim - 1) / block_dim).min(65535);

        let full_shmem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * std::mem::size_of::<f64>()
        } else {
            0usize
        };
        let (warp_use_shmem_arg, shmem_bytes) =
            if full_shmem_bytes > 0 && full_shmem_bytes <= WARP_SHMEM_LIMIT_BYTES {
                (1i32, full_shmem_bytes as u32)
            } else {
                (0i32, 0u32)
            };

        let config = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shmem_bytes,
        };

        let max_leapfrog_arg = max_leapfrog as i32;
        let dim_arg = self.dim as i32;
        let enable_mh_arg: i32 = if enable_mh { 1 } else { 0 };
        let model_id_arg = self.model_id;
        let seed_arg = self.seed;
        let iteration_arg = self.iteration;
        let n_chains_arg = self.n_chains as i32;
        let store_idx_arg: i32 = 0;
        let n_report_arg: i32 = 0;
        let n_obs_arg = self.n_obs as i32;
        let n_feat_arg = self.n_feat as i32;

        let mut builder = self.stream.launch_builder(kernel);
        builder.arg(&mut self.d_x);
        builder.arg(&mut self.d_u);
        builder.arg(&mut self.d_potential);
        builder.arg(&mut self.d_grad);
        builder.arg(&self.d_eps_per_chain);
        builder.arg(&l);
        builder.arg(&max_leapfrog_arg);
        builder.arg(&dim_arg);
        builder.arg(&self.d_inv_mass);
        builder.arg(&enable_mh_arg);
        builder.arg(&self.d_model_data);
        builder.arg(&model_id_arg);
        builder.arg(&seed_arg);
        builder.arg(&iteration_arg);
        builder.arg(&n_chains_arg);
        builder.arg(&mut self.d_accepted);
        builder.arg(&mut self.d_energy_error);
        builder.arg(&mut self.d_sample_buf);
        builder.arg(&mut self.d_accum_potential);
        builder.arg(&mut self.d_accum_accepted);
        builder.arg(&mut self.d_accum_energy);
        builder.arg(&store_idx_arg);
        builder.arg(&n_report_arg);
        builder.arg(&n_obs_arg);
        builder.arg(&n_feat_arg);
        builder.arg(&warp_use_shmem_arg);
        builder.arg(&self.divergence_threshold);

        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch {label}: {e}")))?;
        }

        self.iteration += 1;
        Ok(())
    }

    /// Internal: launch batch warp transitions with the given kernel function.
    fn transition_batch_warp_impl(
        &mut self,
        kernel: &CudaFunction,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
        label: &str,
    ) -> ns_core::Result<BatchResult> {
        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let block_dim = 256u32;
        let total_threads = self.n_chains as u32 * 32;
        let grid_dim = ((total_threads + block_dim - 1) / block_dim).min(65535);

        let full_shmem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * std::mem::size_of::<f64>()
        } else {
            0usize
        };
        let (warp_use_shmem_arg, shmem_bytes) =
            if full_shmem_bytes > 0 && full_shmem_bytes <= WARP_SHMEM_LIMIT_BYTES {
                (1i32, full_shmem_bytes as u32)
            } else {
                (0i32, 0u32)
            };

        let config = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shmem_bytes,
        };

        let max_leapfrog_arg = max_leapfrog as i32;
        let dim_arg = self.dim as i32;
        let enable_mh_arg: i32 = 1;
        let model_id_arg = self.model_id;
        let seed_arg = self.seed;
        let n_chains_arg = self.n_chains as i32;
        let n_report_arg = self.n_report as i32;
        let n_obs_arg = self.n_obs as i32;
        let n_feat_arg = self.n_feat as i32;

        for s in 0..batch_size {
            let iteration_arg = self.iteration;
            let store_idx_arg = s as i32;

            let mut builder = self.stream.launch_builder(kernel);
            builder.arg(&mut self.d_x);
            builder.arg(&mut self.d_u);
            builder.arg(&mut self.d_potential);
            builder.arg(&mut self.d_grad);
            builder.arg(&self.d_eps_per_chain);
            builder.arg(&l);
            builder.arg(&max_leapfrog_arg);
            builder.arg(&dim_arg);
            builder.arg(&self.d_inv_mass);
            builder.arg(&enable_mh_arg);
            builder.arg(&self.d_model_data);
            builder.arg(&model_id_arg);
            builder.arg(&seed_arg);
            builder.arg(&iteration_arg);
            builder.arg(&n_chains_arg);
            builder.arg(&mut self.d_accepted);
            builder.arg(&mut self.d_energy_error);
            builder.arg(&mut self.d_sample_buf);
            builder.arg(&mut self.d_accum_potential);
            builder.arg(&mut self.d_accum_accepted);
            builder.arg(&mut self.d_accum_energy);
            builder.arg(&store_idx_arg);
            builder.arg(&n_report_arg);
            builder.arg(&n_obs_arg);
            builder.arg(&n_feat_arg);
            builder.arg(&warp_use_shmem_arg);
            builder.arg(&self.divergence_threshold);

            unsafe {
                builder
                    .launch(config)
                    .map_err(|e| cuda_err(format!("launch {label} batch[{s}]: {e}")))?;
            }

            self.iteration += 1;
        }

        // Single sync + bulk download
        self.stream.synchronize().map_err(cuda_err)?;

        let nr = self.n_report;
        let dim = self.dim;
        let gpu_slots = self.batch_stride * nr;
        let gpu_pos = gpu_slots * dim;

        let mut pos_flat = vec![0.0f64; gpu_pos];
        let mut pot_flat = vec![0.0f64; gpu_slots];
        let mut acc_flat = vec![0i32; gpu_slots];
        let mut eng_flat = vec![0.0f64; gpu_slots];

        self.stream.memcpy_dtoh(&self.d_sample_buf, &mut pos_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_potential, &mut pot_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_accepted, &mut acc_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_energy, &mut eng_flat).map_err(cuda_err)?;

        let mut positions = Vec::with_capacity(batch_size);
        let mut potentials = Vec::with_capacity(batch_size);
        let mut accepted = Vec::with_capacity(batch_size);
        let mut energy_error = Vec::with_capacity(batch_size);

        for s in 0..batch_size {
            let slot_start = s * nr;
            let pos_start = slot_start * dim;
            positions.push(pos_flat[pos_start..pos_start + nr * dim].to_vec());
            potentials.push(pot_flat[slot_start..slot_start + nr].to_vec());
            accepted.push(acc_flat[slot_start..slot_start + nr].to_vec());
            energy_error.push(eng_flat[slot_start..slot_start + nr].to_vec());
        }

        Ok(BatchResult { positions, potentials, accepted, energy_error, n_launches: batch_size })
    }

    /// Run one MAMS transition using the warp-cooperative kernel (1 warp = 1 chain).
    pub fn transition_warp(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let kernel = self.kernel_warp.as_ref()
            .ok_or_else(|| cuda_err("warp kernel not available"))?
            .clone();
        self.transition_warp_impl(&kernel, l, max_leapfrog, enable_mh, "warp")
    }

    /// Run one MAMS transition using the high-dim warp kernel (dim ≤ 96).
    pub fn transition_warp_hi(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let kernel = self.kernel_warp_hi.as_ref()
            .ok_or_else(|| cuda_err("warp_hi kernel not available"))?
            .clone();
        self.transition_warp_impl(&kernel, l, max_leapfrog, enable_mh, "warp_hi")
    }

    /// Run `batch_size` MAMS transitions using the warp-cooperative kernel.
    pub fn transition_batch_warp(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        let kernel = self.kernel_warp.as_ref()
            .ok_or_else(|| cuda_err("warp kernel not available"))?
            .clone();
        self.transition_batch_warp_impl(&kernel, l, max_leapfrog, batch_size, "warp")
    }

    /// Run `batch_size` MAMS transitions using the high-dim warp kernel.
    pub fn transition_batch_warp_hi(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        let kernel = self.kernel_warp_hi.as_ref()
            .ok_or_else(|| cuda_err("warp_hi kernel not available"))?
            .clone();
        self.transition_batch_warp_impl(&kernel, l, max_leapfrog, batch_size, "warp_hi")
    }

    /// Run `n_transitions` MAMS transitions in a single kernel launch (fused).
    pub fn transition_fused(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        n_transitions: usize,
    ) -> ns_core::Result<BatchResult> {
        let kernel_fused =
            self.kernel_fused.as_ref().ok_or_else(|| cuda_err("fused kernel not available"))?;

        assert!(
            n_transitions <= self.batch_stride,
            "n_transitions exceeds configured batch_stride"
        );
        assert!(self.n_report > 0, "call configure_batch() first");

        let block_dim = 256u32;
        let grid_dim = ((self.n_chains as u32 + block_dim - 1) / block_dim).min(65535);
        let config = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let max_leapfrog_arg = max_leapfrog as i32;
        let dim_arg = self.dim as i32;
        let enable_mh_arg: i32 = 1;
        let model_id_arg = self.model_id;
        let seed_arg = self.seed;
        let iteration_start_arg = self.iteration;
        let n_chains_arg = self.n_chains as i32;
        let n_report_arg = self.n_report as i32;
        let n_transitions_arg = n_transitions as i32;

        let mut builder = self.stream.launch_builder(kernel_fused);
        builder.arg(&mut self.d_x);
        builder.arg(&mut self.d_u);
        builder.arg(&mut self.d_potential);
        builder.arg(&mut self.d_grad);
        builder.arg(&self.d_eps_per_chain);
        builder.arg(&l);
        builder.arg(&max_leapfrog_arg);
        builder.arg(&dim_arg);
        builder.arg(&self.d_inv_mass);
        builder.arg(&enable_mh_arg);
        builder.arg(&self.d_model_data);
        builder.arg(&model_id_arg);
        builder.arg(&seed_arg);
        builder.arg(&iteration_start_arg);
        builder.arg(&n_chains_arg);
        builder.arg(&mut self.d_accepted);
        builder.arg(&mut self.d_energy_error);
        builder.arg(&mut self.d_sample_buf);
        builder.arg(&mut self.d_accum_potential);
        builder.arg(&mut self.d_accum_accepted);
        builder.arg(&mut self.d_accum_energy);
        builder.arg(&n_report_arg);
        builder.arg(&n_transitions_arg);
        builder.arg(&self.divergence_threshold);

        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch fused: {e}")))?;
        }

        self.iteration += n_transitions as i32;

        // Sync + bulk download
        self.stream.synchronize().map_err(cuda_err)?;

        let nr = self.n_report;
        let dim = self.dim;
        let gpu_slots = self.batch_stride * nr;
        let gpu_pos = gpu_slots * dim;

        let mut pos_flat = vec![0.0f64; gpu_pos];
        let mut pot_flat = vec![0.0f64; gpu_slots];
        let mut acc_flat = vec![0i32; gpu_slots];
        let mut eng_flat = vec![0.0f64; gpu_slots];

        self.stream.memcpy_dtoh(&self.d_sample_buf, &mut pos_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_potential, &mut pot_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_accepted, &mut acc_flat).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_accum_energy, &mut eng_flat).map_err(cuda_err)?;

        let mut positions = Vec::with_capacity(n_transitions);
        let mut potentials = Vec::with_capacity(n_transitions);
        let mut accepted = Vec::with_capacity(n_transitions);
        let mut energy_error = Vec::with_capacity(n_transitions);

        for s in 0..n_transitions {
            let slot_start = s * nr;
            let pos_start = slot_start * dim;
            positions.push(pos_flat[pos_start..pos_start + nr * dim].to_vec());
            potentials.push(pot_flat[slot_start..slot_start + nr].to_vec());
            accepted.push(acc_flat[slot_start..slot_start + nr].to_vec());
            energy_error.push(eng_flat[slot_start..slot_start + nr].to_vec());
        }

        Ok(BatchResult { positions, potentials, accepted, energy_error, n_launches: 1 })
    }

    /// Auto-dispatch: uses warp kernel for data-heavy models, scalar kernel otherwise.
    pub fn transition_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        if self.supports_warp() {
            self.transition_warp(l, max_leapfrog, enable_mh)
        } else if self.supports_warp_hi() {
            self.transition_warp_hi(l, max_leapfrog, enable_mh)
        } else {
            self.transition(l, max_leapfrog, enable_mh)
        }
    }

    /// Auto-dispatch batch.
    ///
    /// For sampling, callers can set `prefer_fused=true` to force the fused
    /// multi-step path when available, even on models that also support warp.
    /// This ensures `fused_transitions` is not silently ignored for large-`n`
    /// GLM cases where warp kernels are available but launch overhead still
    /// matters.
    pub fn transition_batch_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
        prefer_fused: bool,
    ) -> ns_core::Result<BatchResult> {
        if prefer_fused && self.supports_fused() {
            self.transition_fused(l, max_leapfrog, batch_size)
        } else if self.supports_warp() {
            self.transition_batch_warp(l, max_leapfrog, batch_size)
        } else if self.supports_warp_hi() {
            self.transition_batch_warp_hi(l, max_leapfrog, batch_size)
        } else {
            self.transition_batch(l, max_leapfrog, batch_size)
        }
    }

    /// Download positions from GPU. Returns `[n_chains × dim]` (row-major).
    pub fn download_positions(&self) -> ns_core::Result<Vec<f64>> {
        let total = self.n_chains * self.dim;
        let mut host = vec![0.0f64; total];
        self.stream.memcpy_dtoh(&self.d_x, &mut host).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(host)
    }

    /// Download potentials from GPU. Returns `[n_chains]`.
    pub fn download_potentials(&self) -> ns_core::Result<Vec<f64>> {
        let mut host = vec![0.0f64; self.n_chains];
        self.stream.memcpy_dtoh(&self.d_potential, &mut host).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(host)
    }

    /// Download diagnostics (accepted flags + energy errors).
    pub fn download_diagnostics(&self) -> ns_core::Result<TransitionDiagnostics> {
        let mut accepted = vec![0i32; self.n_chains];
        let mut energy_error = vec![0.0f64; self.n_chains];
        self.stream.memcpy_dtoh(&self.d_accepted, &mut accepted).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_energy_error, &mut energy_error).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(TransitionDiagnostics { accepted, energy_error })
    }

    /// Number of chains.
    pub fn n_chains(&self) -> usize {
        self.n_chains
    }

    /// Parameter dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl crate::mams_trait::MamsAccelerator for CudaMamsAccelerator {
    fn upload_state(
        &mut self,
        x: &[f64],
        u: &[f64],
        potential: &[f64],
        grad: &[f64],
    ) -> ns_core::Result<()> {
        self.upload_state(x, u, potential, grad)
    }

    fn download_positions(&self) -> ns_core::Result<Vec<f64>> {
        self.download_positions()
    }

    fn download_potentials(&self) -> ns_core::Result<Vec<f64>> {
        self.download_potentials()
    }

    fn download_diagnostics(&self) -> ns_core::Result<TransitionDiagnostics> {
        self.download_diagnostics()
    }

    fn set_inv_mass(&mut self, inv_mass: &[f64]) -> ns_core::Result<()> {
        self.set_inv_mass(inv_mass)
    }

    fn set_per_chain_eps(&mut self, eps_vec: &[f64]) -> ns_core::Result<()> {
        self.set_per_chain_eps(eps_vec)
    }

    fn set_uniform_eps(&mut self, eps: f64) -> ns_core::Result<()> {
        self.set_uniform_eps(eps)
    }

    fn transition_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        self.transition_auto(l, max_leapfrog, enable_mh)
    }

    fn transition_batch_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
        prefer_fused: bool,
    ) -> ns_core::Result<BatchResult> {
        self.transition_batch_auto(l, max_leapfrog, batch_size, prefer_fused)
    }

    fn glm_presolve(&mut self, max_iters: usize) -> ns_core::Result<bool> {
        self.glm_presolve(max_iters)
    }

    fn configure_batch(&mut self, batch_size: usize, n_report: usize) -> ns_core::Result<()> {
        self.configure_batch(batch_size, n_report)
    }

    fn supports_fused(&self) -> bool {
        self.supports_fused()
    }

    fn supports_warp(&self) -> bool {
        self.supports_warp()
    }

    fn supports_warp_hi(&self) -> bool {
        self.supports_warp_hi()
    }

    fn n_chains(&self) -> usize {
        self.n_chains()
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn set_divergence_threshold(&mut self, threshold: f64) {
        self.divergence_threshold = threshold;
    }
}
