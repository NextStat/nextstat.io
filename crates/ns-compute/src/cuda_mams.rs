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

const PTX_SRC: &str = include_str!(env!("CUDA_MAMS_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA mams: {msg}"))
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
}

/// Diagnostics from one transition.
pub struct TransitionDiagnostics {
    /// Per-chain acceptance flags.
    pub accepted: Vec<i32>,
    /// Per-chain energy errors (ΔV + ΔK).
    pub energy_error: Vec<f64>,
}

/// Result from a batch of transitions (GPU-side accumulation).
pub struct BatchResult {
    /// Positions `[batch_size][n_report × dim]` (flat per-sample).
    pub positions: Vec<Vec<f64>>,
    /// Potentials `[batch_size][n_report]`.
    pub potentials: Vec<Vec<f64>>,
    /// Acceptance flags `[batch_size][n_report]`.
    pub accepted: Vec<Vec<i32>>,
    /// Energy errors `[batch_size][n_report]`.
    pub energy_error: Vec<Vec<f64>>,
    /// Number of kernel launches in this batch.
    pub n_launches: usize,
}

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

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load module: {e}")))?;
        let kernel = module
            .load_function("mams_transition")
            .map_err(|e| cuda_err(format!("load function: {e}")))?;
        let kernel_fused = module.load_function("mams_transition_fused").ok();
        let kernel_warp = module.load_function("mams_transition_warp").ok();

        // Extract data dimensions for warp kernel shared memory sizing.
        // GLM logistic (model_id=3): model_data = [n, p, X(n*p), y(n)]
        let (n_obs, n_feat) = if model_id == 3 && model_data.len() >= 2 {
            (model_data[0] as usize, model_data[1] as usize)
        } else {
            (0, 0)
        };

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

        self.stream.memcpy_htod(x, &mut self.d_x).map_err(cuda_err)?;
        self.stream.memcpy_htod(u, &mut self.d_u).map_err(cuda_err)?;
        self.stream.memcpy_htod(potential, &mut self.d_potential).map_err(cuda_err)?;
        self.stream.memcpy_htod(grad, &mut self.d_grad).map_err(cuda_err)?;

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
        let shmem_bytes = (self.n_obs * self.n_feat + self.n_obs) * std::mem::size_of::<f64>();
        shmem_bytes <= 48 * 1024
    }

    /// Run one MAMS transition using the warp-cooperative kernel (1 warp = 1 chain).
    pub fn transition_warp(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let kernel_warp = self
            .kernel_warp
            .as_ref()
            .ok_or_else(|| cuda_err("warp kernel not available"))?;

        let block_dim = 256u32;
        let total_threads = self.n_chains as u32 * 32;
        let grid_dim = ((total_threads + block_dim - 1) / block_dim).min(65535);

        let shmem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            ((self.n_obs * self.n_feat + self.n_obs) * std::mem::size_of::<f64>()) as u32
        } else {
            0u32
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

        let mut builder = self.stream.launch_builder(kernel_warp);
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

        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch warp: {e}")))?;
        }

        self.iteration += 1;
        Ok(())
    }

    /// Run `batch_size` MAMS transitions using the warp-cooperative kernel.
    pub fn transition_batch_warp(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        let kernel_warp = self
            .kernel_warp
            .as_ref()
            .ok_or_else(|| cuda_err("warp kernel not available"))?;

        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let block_dim = 256u32;
        let total_threads = self.n_chains as u32 * 32;
        let grid_dim = ((total_threads + block_dim - 1) / block_dim).min(65535);

        let shmem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            ((self.n_obs * self.n_feat + self.n_obs) * std::mem::size_of::<f64>()) as u32
        } else {
            0u32
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

            let mut builder = self.stream.launch_builder(kernel_warp);
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

            unsafe {
                builder
                    .launch(config)
                    .map_err(|e| cuda_err(format!("launch warp batch[{s}]: {e}")))?;
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

    /// Run `n_transitions` MAMS transitions in a single kernel launch (fused).
    pub fn transition_fused(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        n_transitions: usize,
    ) -> ns_core::Result<BatchResult> {
        let kernel_fused = self
            .kernel_fused
            .as_ref()
            .ok_or_else(|| cuda_err("fused kernel not available"))?;

        assert!(n_transitions <= self.batch_stride, "n_transitions exceeds configured batch_stride");
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
        } else {
            self.transition(l, max_leapfrog, enable_mh)
        }
    }

    /// Auto-dispatch batch: warp > fused > scalar.
    pub fn transition_batch_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
        prefer_fused: bool,
    ) -> ns_core::Result<BatchResult> {
        if self.supports_warp() {
            self.transition_batch_warp(l, max_leapfrog, batch_size)
        } else if prefer_fused && self.supports_fused() {
            self.transition_fused(l, max_leapfrog, batch_size)
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
