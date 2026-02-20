//! Metal accelerator for MAMS (Metropolis-Adjusted Microcanonical Sampler).
//!
//! Wraps the `mams_transition` MSL kernel. Manages persistent per-chain state
//! on device (positions, velocities, potentials, gradients). The host drives
//! warmup adaptation and convergence diagnostics; the GPU executes trajectories.
//!
//! Architecture: 1 Metal thread = 1 chain (scalar kernel), or
//!               32 threads (1 SIMD group) = 1 chain (simdgroup kernel).
//! All computation in f32. Conversion f64↔f32 happens at the API boundary.

use crate::mams_trait::{BatchResult, MamsAccelerator, TransitionDiagnostics};
use metal::*;
use std::mem;

/// MSL source compiled from `kernels/mams_leapfrog.metal`.
const MSL_SRC: &str = include_str!("../kernels/mams_leapfrog.metal");

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal mams: {msg}"))
}

/// Scalar arguments passed to the kernel via `constant` buffer.
/// Must match `MamsArgs` in `mams_leapfrog.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MamsArgs {
    l: f32,
    max_leapfrog: i32,
    dim: i32,
    enable_mh: i32,
    model_id: i32,
    seed_lo: u32,
    seed_hi: u32,
    iteration: i32,
    n_chains: i32,
    store_idx: i32,
    n_report: i32,
    n_obs: i32,
    n_feat: i32,
    riemannian: i32,
    divergence_threshold: f32,
}

/// Metal accelerator for MAMS transitions.
///
/// Persistent GPU buffers for chain state (f32, `StorageModeShared`).
/// 1 kernel launch = 1 transition for all chains simultaneously.
///
/// Supports batch mode: `configure_batch()` allocates GPU-side accumulation
/// buffers, then `transition_batch()` encodes multiple dispatches in one
/// command buffer without intermediate host-device sync.
pub struct MetalMamsAccelerator {
    device: Device,
    queue: CommandQueue,
    pipeline_transition: ComputePipelineState,
    pipeline_fused: ComputePipelineState,
    pipeline_simdgroup: Option<ComputePipelineState>,
    pipeline_simdgroup_hi: Option<ComputePipelineState>,

    // Per-chain state [n_chains × dim] — f32
    buf_x: Buffer,
    buf_u: Buffer,
    buf_potential: Buffer,
    buf_grad: Buffer,

    // Sampler parameters — f32
    buf_inv_mass: Buffer,
    buf_model_data: Buffer,
    buf_eps_per_chain: Buffer,

    // Per-chain output
    buf_accepted: Buffer,     // [n_chains] i32
    buf_energy_error: Buffer, // [n_chains] f32

    // Accumulation buffers for batch sampling
    buf_sample_buf: Buffer,      // [batch_stride × n_report × dim] f32
    buf_accum_potential: Buffer, // [batch_stride × n_report] f32
    buf_accum_accepted: Buffer,  // [batch_stride × n_report] i32
    buf_accum_energy: Buffer,    // [batch_stride × n_report] f32
    batch_stride: usize,
    n_report: usize,

    n_chains: usize,
    dim: usize,
    model_id: i32,
    seed: u64,
    iteration: i32,

    // Simdgroup kernel data dimensions (for threadgroup memory sizing)
    n_obs: usize,
    n_feat: usize,

    // Whether this model uses Riemannian (position-dependent metric) dynamics.
    riemannian: bool,

    // Energy error threshold for divergence detection (default: 1000.0).
    divergence_threshold: f32,

    // CPU scratch buffer for f64→f32 conversion (reused)
    scratch_f32: Vec<f32>,
}

impl MetalMamsAccelerator {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create a new MAMS accelerator on the default Metal device.
    ///
    /// - `model_data`: f64 values from the host — converted to f32 for GPU.
    pub fn new(
        n_chains: usize,
        dim: usize,
        model_id: i32,
        model_data: &[f64],
        seed: u64,
    ) -> ns_core::Result<Self> {
        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        // Compile MSL pipelines via thread-local cache.
        let pipeline_transition =
            crate::metal_kernel_cache::get_pipeline(&device, "mams", MSL_SRC, "mams_transition")?;
        let pipeline_fused = crate::metal_kernel_cache::get_pipeline(
            &device,
            "mams",
            MSL_SRC,
            "mams_transition_fused",
        )?;
        let pipeline_simdgroup = crate::metal_kernel_cache::get_pipeline(
            &device,
            "mams",
            MSL_SRC,
            "mams_transition_simdgroup",
        )
        .ok();
        let pipeline_simdgroup_hi = crate::metal_kernel_cache::get_pipeline(
            &device,
            "mams",
            MSL_SRC,
            "mams_transition_simdgroup_hi",
        )
        .ok();

        let opts = MTLResourceOptions::StorageModeShared;
        let total = n_chains * dim;

        // Allocate persistent chain state buffers (zeroed)
        let buf_x = device.new_buffer((total * mem::size_of::<f32>()) as u64, opts);
        let buf_u = device.new_buffer((total * mem::size_of::<f32>()) as u64, opts);
        let buf_potential = device.new_buffer((n_chains * mem::size_of::<f32>()) as u64, opts);
        let buf_grad = device.new_buffer((total * mem::size_of::<f32>()) as u64, opts);

        // Inverse mass matrix [dim] — initialized to 1.0
        let inv_mass_f32: Vec<f32> = vec![1.0f32; dim];
        let buf_inv_mass = Self::create_buffer_from_f32_slice(&device, &inv_mass_f32, opts);

        // Model data: f64→f32
        let model_data_f32: Vec<f32> = if model_data.is_empty() {
            vec![0.0f32]
        } else {
            model_data.iter().map(|&v| v as f32).collect()
        };
        let buf_model_data = Self::create_buffer_from_f32_slice(&device, &model_data_f32, opts);

        // Per-chain step sizes [n_chains] — initialized to 1.0
        let eps_f32: Vec<f32> = vec![1.0f32; n_chains];
        let buf_eps_per_chain = Self::create_buffer_from_f32_slice(&device, &eps_f32, opts);

        // Diagnostics output
        let buf_accepted = device.new_buffer((n_chains * mem::size_of::<i32>()) as u64, opts);
        let buf_energy_error = device.new_buffer((n_chains * mem::size_of::<f32>()) as u64, opts);

        // Accumulation buffers (minimal — reallocated by configure_batch)
        let buf_sample_buf = device.new_buffer(mem::size_of::<f32>().max(4) as u64, opts);
        let buf_accum_potential = device.new_buffer(mem::size_of::<f32>().max(4) as u64, opts);
        let buf_accum_accepted = device.new_buffer(mem::size_of::<i32>().max(4) as u64, opts);
        let buf_accum_energy = device.new_buffer(mem::size_of::<f32>().max(4) as u64, opts);

        // Extract data dimensions for simdgroup kernel (GLM models)
        let (n_obs, n_feat) = if [3, 6, 7, 8, 9].contains(&model_id) && model_data.len() >= 2 {
            (model_data[0] as usize, model_data[1] as usize)
        } else {
            (0, 0)
        };

        // Initialize potential buffers to infinity (signals first-call logic in kernel)
        let inf_f32 = vec![f32::INFINITY; n_chains];
        Self::copy_to_buffer(&buf_potential, &inf_f32);

        // Pre-allocate scratch buffer (sized for max common upload)
        let scratch_size = total.max(n_chains);
        let scratch_f32 = vec![0.0f32; scratch_size];

        let riemannian = model_id == 5;

        Ok(Self {
            device,
            queue,
            pipeline_transition,
            pipeline_fused,
            pipeline_simdgroup,
            pipeline_simdgroup_hi,
            buf_x,
            buf_u,
            buf_potential,
            buf_grad,
            buf_inv_mass,
            buf_model_data,
            buf_eps_per_chain,
            buf_accepted,
            buf_energy_error,
            buf_sample_buf,
            buf_accum_potential,
            buf_accum_accepted,
            buf_accum_energy,
            batch_stride: 0,
            n_report: 0,
            n_chains,
            dim,
            model_id,
            seed,
            iteration: 0,
            n_obs,
            n_feat,
            riemannian,
            divergence_threshold: 1000.0f32,
            scratch_f32,
        })
    }

    /// Upload chain state from host (f64→f32).
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

        // Ensure scratch is large enough
        if self.scratch_f32.len() < total {
            self.scratch_f32.resize(total, 0.0);
        }

        // x
        for (i, &v) in x.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_x, &self.scratch_f32[..total]);

        // u
        for (i, &v) in u.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_u, &self.scratch_f32[..total]);

        // potential [n_chains]
        for (i, &v) in potential.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_potential, &self.scratch_f32[..self.n_chains]);

        // grad
        for (i, &v) in grad.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_grad, &self.scratch_f32[..total]);

        Ok(())
    }

    /// Download positions from GPU (f32→f64). Returns `[n_chains × dim]`.
    pub fn download_positions(&self) -> ns_core::Result<Vec<f64>> {
        Ok(Self::read_buffer_f32_to_f64(&self.buf_x, self.n_chains * self.dim))
    }

    /// Download potentials from GPU (f32→f64). Returns `[n_chains]`.
    pub fn download_potentials(&self) -> ns_core::Result<Vec<f64>> {
        Ok(Self::read_buffer_f32_to_f64(&self.buf_potential, self.n_chains))
    }

    /// Download diagnostics (accepted flags + energy errors, f32→f64).
    pub fn download_diagnostics(&self) -> ns_core::Result<TransitionDiagnostics> {
        let accepted = Self::read_buffer_i32(&self.buf_accepted, self.n_chains);
        let energy_error = Self::read_buffer_f32_to_f64(&self.buf_energy_error, self.n_chains);
        Ok(TransitionDiagnostics { accepted, energy_error })
    }

    /// Upload diagonal inverse mass matrix `[dim]` (f64→f32).
    pub fn set_inv_mass(&mut self, inv_mass: &[f64]) -> ns_core::Result<()> {
        assert_eq!(inv_mass.len(), self.dim);
        if self.scratch_f32.len() < self.dim {
            self.scratch_f32.resize(self.dim, 0.0);
        }
        for (i, &v) in inv_mass.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_inv_mass, &self.scratch_f32[..self.dim]);
        Ok(())
    }

    /// Upload per-chain step sizes `[n_chains]` (f64→f32).
    pub fn set_per_chain_eps(&mut self, eps_vec: &[f64]) -> ns_core::Result<()> {
        assert_eq!(eps_vec.len(), self.n_chains);
        if self.scratch_f32.len() < self.n_chains {
            self.scratch_f32.resize(self.n_chains, 0.0);
        }
        for (i, &v) in eps_vec.iter().enumerate() {
            self.scratch_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_eps_per_chain, &self.scratch_f32[..self.n_chains]);
        Ok(())
    }

    /// Fill per-chain eps buffer with a uniform value (f64→f32).
    pub fn set_uniform_eps(&mut self, eps: f64) -> ns_core::Result<()> {
        let uniform = vec![eps; self.n_chains];
        self.set_per_chain_eps(&uniform)
    }

    /// Run one MAMS transition for all chains (scalar kernel).
    pub fn transition(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let args = MamsArgs {
            l: l as f32,
            max_leapfrog: max_leapfrog as i32,
            dim: self.dim as i32,
            enable_mh: if enable_mh { 1 } else { 0 },
            model_id: self.model_id,
            seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
            seed_hi: (self.seed >> 32) as u32,
            iteration: self.iteration,
            n_chains: self.n_chains as i32,
            store_idx: 0,
            n_report: 0,
            n_obs: self.n_obs as i32,
            n_feat: self.n_feat as i32,
            riemannian: if self.riemannian { 1 } else { 0 },
            divergence_threshold: self.divergence_threshold,
        };

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pipeline_transition);
        self.set_chain_buffers(enc);
        enc.set_bytes(
            13,
            mem::size_of::<MamsArgs>() as u64,
            &args as *const MamsArgs as *const std::ffi::c_void,
        );

        let grid = MTLSize::new(self.n_chains as u64, 1, 1);
        let tg_size = 256u64.min(self.n_chains as u64).max(1);
        let tg = MTLSize::new(tg_size, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.iteration += 1;
        Ok(())
    }

    /// Configure GPU-side accumulation buffers for batch sampling.
    pub fn configure_batch(&mut self, batch_size: usize, n_report: usize) -> ns_core::Result<()> {
        assert!(n_report <= self.n_chains);
        self.batch_stride = batch_size;
        self.n_report = n_report;

        let opts = MTLResourceOptions::StorageModeShared;
        let total_slots = batch_size * n_report;
        let total_pos = total_slots * self.dim;

        self.buf_sample_buf =
            self.device.new_buffer((total_pos * mem::size_of::<f32>()).max(4) as u64, opts);
        self.buf_accum_potential =
            self.device.new_buffer((total_slots * mem::size_of::<f32>()).max(4) as u64, opts);
        self.buf_accum_accepted =
            self.device.new_buffer((total_slots * mem::size_of::<i32>()).max(4) as u64, opts);
        self.buf_accum_energy =
            self.device.new_buffer((total_slots * mem::size_of::<f32>()).max(4) as u64, opts);

        Ok(())
    }

    /// Run `batch_size` MAMS transitions without intermediate host-device sync.
    pub fn transition_batch(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let cmd = self.queue.new_command_buffer();

        let grid = MTLSize::new(self.n_chains as u64, 1, 1);
        let tg_size = 256u64.min(self.n_chains as u64).max(1);
        let tg = MTLSize::new(tg_size, 1, 1);

        for s in 0..batch_size {
            let args = MamsArgs {
                l: l as f32,
                max_leapfrog: max_leapfrog as i32,
                dim: self.dim as i32,
                enable_mh: 1,
                model_id: self.model_id,
                seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
                seed_hi: (self.seed >> 32) as u32,
                iteration: self.iteration,
                n_chains: self.n_chains as i32,
                store_idx: s as i32,
                n_report: self.n_report as i32,
                n_obs: self.n_obs as i32,
                n_feat: self.n_feat as i32,
                riemannian: if self.riemannian { 1 } else { 0 },
                divergence_threshold: self.divergence_threshold,
            };

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_transition);
            self.set_chain_buffers(enc);
            enc.set_bytes(
                13,
                mem::size_of::<MamsArgs>() as u64,
                &args as *const MamsArgs as *const std::ffi::c_void,
            );
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            self.iteration += 1;
        }

        cmd.commit();
        cmd.wait_until_completed();

        self.download_batch_result(batch_size)
    }

    /// Whether the fused multi-step kernel is available.
    pub fn supports_fused(&self) -> bool {
        true // fused kernel is always compiled in the MSL source
    }

    /// Whether the simdgroup-cooperative kernel is available and beneficial.
    pub fn supports_warp(&self) -> bool {
        if self.pipeline_simdgroup.is_none() || self.n_obs < 1024 {
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

    /// Whether the high-dim simdgroup kernel (dim ≤ 96) is available and beneficial.
    pub fn supports_warp_hi(&self) -> bool {
        if self.pipeline_simdgroup_hi.is_none() || self.n_obs < 1024 {
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

    /// Run one MAMS transition using the simdgroup-cooperative kernel.
    pub fn transition_simdgroup(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let pipeline = self
            .pipeline_simdgroup
            .as_ref()
            .ok_or_else(|| metal_err("simdgroup kernel not available"))?;

        let args = MamsArgs {
            l: l as f32,
            max_leapfrog: max_leapfrog as i32,
            dim: self.dim as i32,
            enable_mh: if enable_mh { 1 } else { 0 },
            model_id: self.model_id,
            seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
            seed_hi: (self.seed >> 32) as u32,
            iteration: self.iteration,
            n_chains: self.n_chains as i32,
            store_idx: 0,
            n_report: 0,
            n_obs: self.n_obs as i32,
            n_feat: self.n_feat as i32,
            riemannian: if self.riemannian { 1 } else { 0 },
            divergence_threshold: self.divergence_threshold,
        };

        // Threadgroup memory for GLM data: X_col[p*n] + y[n] in f32
        let tg_mem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * mem::size_of::<f32>()
        } else {
            0
        };

        // 32 threads per chain (SIMD group)
        let total_threads = self.n_chains as u64 * 32;
        let grid = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(32, 1, 1);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pipeline);
        self.set_chain_buffers(enc);
        enc.set_bytes(
            13,
            mem::size_of::<MamsArgs>() as u64,
            &args as *const MamsArgs as *const std::ffi::c_void,
        );
        if tg_mem_bytes > 0 {
            enc.set_threadgroup_memory_length(0, tg_mem_bytes as u64);
        }

        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.iteration += 1;
        Ok(())
    }

    /// Run `batch_size` MAMS transitions using the simdgroup-cooperative kernel.
    pub fn transition_batch_simdgroup(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        let pipeline = self
            .pipeline_simdgroup
            .as_ref()
            .ok_or_else(|| metal_err("simdgroup kernel not available"))?
            .clone();

        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let tg_mem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * mem::size_of::<f32>()
        } else {
            0
        };

        let total_threads = self.n_chains as u64 * 32;
        let grid = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(32, 1, 1);

        let cmd = self.queue.new_command_buffer();

        for s in 0..batch_size {
            let args = MamsArgs {
                l: l as f32,
                max_leapfrog: max_leapfrog as i32,
                dim: self.dim as i32,
                enable_mh: 1,
                model_id: self.model_id,
                seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
                seed_hi: (self.seed >> 32) as u32,
                iteration: self.iteration,
                n_chains: self.n_chains as i32,
                store_idx: s as i32,
                n_report: self.n_report as i32,
                n_obs: self.n_obs as i32,
                n_feat: self.n_feat as i32,
                riemannian: if self.riemannian { 1 } else { 0 },
                divergence_threshold: self.divergence_threshold,
            };

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            self.set_chain_buffers(enc);
            enc.set_bytes(
                13,
                mem::size_of::<MamsArgs>() as u64,
                &args as *const MamsArgs as *const std::ffi::c_void,
            );
            if tg_mem_bytes > 0 {
                enc.set_threadgroup_memory_length(0, tg_mem_bytes as u64);
            }

            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            self.iteration += 1;
        }

        cmd.commit();
        cmd.wait_until_completed();

        self.download_batch_result(batch_size)
    }

    /// Run one MAMS transition using the high-dim simdgroup kernel (dim ≤ 96).
    pub fn transition_simdgroup_hi(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        let pipeline = self
            .pipeline_simdgroup_hi
            .as_ref()
            .ok_or_else(|| metal_err("simdgroup_hi kernel not available"))?;

        let args = MamsArgs {
            l: l as f32,
            max_leapfrog: max_leapfrog as i32,
            dim: self.dim as i32,
            enable_mh: if enable_mh { 1 } else { 0 },
            model_id: self.model_id,
            seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
            seed_hi: (self.seed >> 32) as u32,
            iteration: self.iteration,
            n_chains: self.n_chains as i32,
            store_idx: 0,
            n_report: 0,
            n_obs: self.n_obs as i32,
            n_feat: self.n_feat as i32,
            riemannian: if self.riemannian { 1 } else { 0 },
            divergence_threshold: self.divergence_threshold,
        };

        let tg_mem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * mem::size_of::<f32>()
        } else {
            0
        };

        let total_threads = self.n_chains as u64 * 32;
        let grid = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(32, 1, 1);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pipeline);
        self.set_chain_buffers(enc);
        enc.set_bytes(
            13,
            mem::size_of::<MamsArgs>() as u64,
            &args as *const MamsArgs as *const std::ffi::c_void,
        );
        if tg_mem_bytes > 0 {
            enc.set_threadgroup_memory_length(0, tg_mem_bytes as u64);
        }

        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.iteration += 1;
        Ok(())
    }

    /// Run `batch_size` MAMS transitions using the high-dim simdgroup kernel.
    pub fn transition_batch_simdgroup_hi(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
    ) -> ns_core::Result<BatchResult> {
        let pipeline = self
            .pipeline_simdgroup_hi
            .as_ref()
            .ok_or_else(|| metal_err("simdgroup_hi kernel not available"))?
            .clone();

        assert!(batch_size <= self.batch_stride, "batch_size exceeds configured batch_stride");
        assert!(self.n_report > 0, "call configure_batch() first");

        let tg_mem_bytes = if self.n_obs > 0 && self.n_feat > 0 {
            (self.n_obs * self.n_feat + self.n_obs) * mem::size_of::<f32>()
        } else {
            0
        };

        let total_threads = self.n_chains as u64 * 32;
        let grid = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(32, 1, 1);

        let cmd = self.queue.new_command_buffer();

        for s in 0..batch_size {
            let args = MamsArgs {
                l: l as f32,
                max_leapfrog: max_leapfrog as i32,
                dim: self.dim as i32,
                enable_mh: 1,
                model_id: self.model_id,
                seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
                seed_hi: (self.seed >> 32) as u32,
                iteration: self.iteration,
                n_chains: self.n_chains as i32,
                store_idx: s as i32,
                n_report: self.n_report as i32,
                n_obs: self.n_obs as i32,
                n_feat: self.n_feat as i32,
                riemannian: if self.riemannian { 1 } else { 0 },
                divergence_threshold: self.divergence_threshold,
            };

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            self.set_chain_buffers(enc);
            enc.set_bytes(
                13,
                mem::size_of::<MamsArgs>() as u64,
                &args as *const MamsArgs as *const std::ffi::c_void,
            );
            if tg_mem_bytes > 0 {
                enc.set_threadgroup_memory_length(0, tg_mem_bytes as u64);
            }

            enc.dispatch_threads(grid, tg);
            enc.end_encoding();

            self.iteration += 1;
        }

        cmd.commit();
        cmd.wait_until_completed();

        self.download_batch_result(batch_size)
    }

    /// Run `n_transitions` MAMS transitions in a single kernel launch (fused).
    pub fn transition_fused(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        n_transitions: usize,
    ) -> ns_core::Result<BatchResult> {
        assert!(
            n_transitions <= self.batch_stride,
            "n_transitions exceeds configured batch_stride"
        );
        assert!(self.n_report > 0, "call configure_batch() first");

        let args = MamsArgs {
            l: l as f32,
            max_leapfrog: max_leapfrog as i32,
            dim: self.dim as i32,
            enable_mh: 1,
            model_id: self.model_id,
            seed_lo: (self.seed & 0xFFFF_FFFF) as u32,
            seed_hi: (self.seed >> 32) as u32,
            iteration: self.iteration,
            n_chains: self.n_chains as i32,
            store_idx: 0, // unused by fused — iterates internally
            n_report: self.n_report as i32,
            n_obs: self.n_obs as i32,
            n_feat: self.n_feat as i32,
            riemannian: if self.riemannian { 1 } else { 0 },
            divergence_threshold: self.divergence_threshold,
        };
        let n_transitions_arg = n_transitions as i32;

        let grid = MTLSize::new(self.n_chains as u64, 1, 1);
        let tg_size = 256u64.min(self.n_chains as u64).max(1);
        let tg = MTLSize::new(tg_size, 1, 1);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pipeline_fused);
        self.set_chain_buffers(enc);
        enc.set_bytes(
            13,
            mem::size_of::<MamsArgs>() as u64,
            &args as *const MamsArgs as *const std::ffi::c_void,
        );
        enc.set_bytes(
            14,
            mem::size_of::<i32>() as u64,
            &n_transitions_arg as *const i32 as *const std::ffi::c_void,
        );

        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.iteration += n_transitions as i32;

        self.download_batch_result_with_launches(n_transitions, 1)
    }

    /// Auto-dispatch single transition.
    pub fn transition_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        enable_mh: bool,
    ) -> ns_core::Result<()> {
        if self.supports_warp() {
            self.transition_simdgroup(l, max_leapfrog, enable_mh)
        } else if self.supports_warp_hi() {
            self.transition_simdgroup_hi(l, max_leapfrog, enable_mh)
        } else {
            self.transition(l, max_leapfrog, enable_mh)
        }
    }

    /// Auto-dispatch batch of transitions.
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
            self.transition_batch_simdgroup(l, max_leapfrog, batch_size)
        } else if self.supports_warp_hi() {
            self.transition_batch_simdgroup_hi(l, max_leapfrog, batch_size)
        } else {
            self.transition_batch(l, max_leapfrog, batch_size)
        }
    }

    /// Number of chains.
    pub fn n_chains(&self) -> usize {
        self.n_chains
    }

    /// Parameter dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    // --- Private helpers ---

    fn set_chain_buffers(&self, enc: &ComputeCommandEncoderRef) {
        enc.set_buffer(0, Some(&self.buf_x), 0);
        enc.set_buffer(1, Some(&self.buf_u), 0);
        enc.set_buffer(2, Some(&self.buf_potential), 0);
        enc.set_buffer(3, Some(&self.buf_grad), 0);
        enc.set_buffer(4, Some(&self.buf_eps_per_chain), 0);
        enc.set_buffer(5, Some(&self.buf_inv_mass), 0);
        enc.set_buffer(6, Some(&self.buf_model_data), 0);
        enc.set_buffer(7, Some(&self.buf_accepted), 0);
        enc.set_buffer(8, Some(&self.buf_energy_error), 0);
        enc.set_buffer(9, Some(&self.buf_sample_buf), 0);
        enc.set_buffer(10, Some(&self.buf_accum_potential), 0);
        enc.set_buffer(11, Some(&self.buf_accum_accepted), 0);
        enc.set_buffer(12, Some(&self.buf_accum_energy), 0);
    }

    fn download_batch_result(&self, batch_size: usize) -> ns_core::Result<BatchResult> {
        self.download_batch_result_with_launches(batch_size, batch_size)
    }

    fn download_batch_result_with_launches(
        &self,
        batch_size: usize,
        n_launches: usize,
    ) -> ns_core::Result<BatchResult> {
        let nr = self.n_report;
        let dim = self.dim;
        let gpu_slots = self.batch_stride * nr;
        let gpu_pos = gpu_slots * dim;

        let pos_flat = Self::read_buffer_f32_to_f64(&self.buf_sample_buf, gpu_pos);
        let pot_flat = Self::read_buffer_f32_to_f64(&self.buf_accum_potential, gpu_slots);
        let acc_flat = Self::read_buffer_i32(&self.buf_accum_accepted, gpu_slots);
        let eng_flat = Self::read_buffer_f32_to_f64(&self.buf_accum_energy, gpu_slots);

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

        Ok(BatchResult { positions, potentials, accepted, energy_error, n_launches })
    }

    fn create_buffer_from_f32_slice(
        device: &Device,
        data: &[f32],
        opts: MTLResourceOptions,
    ) -> Buffer {
        if data.is_empty() {
            return device.new_buffer(mem::size_of::<f32>().max(4) as u64, opts);
        }
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(data) as u64,
            opts,
        )
    }

    fn copy_to_buffer<T>(buffer: &Buffer, data: &[T]) {
        if data.is_empty() {
            return;
        }
        let ptr = buffer.contents() as *mut T;
        // SAFETY: buffer capacity >= data.len() * size_of::<T>() — upheld by caller.
        // contents() returns valid mapped pointer for StorageModeShared.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn read_buffer_f32_to_f64(buffer: &Buffer, count: usize) -> Vec<f64> {
        if count == 0 {
            return Vec::new();
        }
        let ptr = buffer.contents() as *const f32;
        // SAFETY: buffer capacity >= count * size_of::<f32>().
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.iter().map(|&v| v as f64).collect()
    }

    fn read_buffer_i32(buffer: &Buffer, count: usize) -> Vec<i32> {
        if count == 0 {
            return Vec::new();
        }
        let ptr = buffer.contents() as *const i32;
        // SAFETY: buffer capacity >= count * size_of::<i32>().
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }
}

// --- MamsAccelerator trait implementation ---

impl MamsAccelerator for MetalMamsAccelerator {
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
        self.divergence_threshold = threshold as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_mams_std_normal_smoke() {
        if !MetalMamsAccelerator::is_available() {
            eprintln!("Metal not available, skipping");
            return;
        }

        let n_chains = 64;
        let dim = 4;
        let model_id = 0; // StdNormal
        let mut accel = MetalMamsAccelerator::new(n_chains, dim, model_id, &[], 42).unwrap();

        // Upload initial state: x=0, u=random-ish, potential=inf, grad=0
        let x = vec![0.1f64; n_chains * dim];
        let grad = vec![0.1f64; n_chains * dim]; // grad of 0.5*x^2 at x=0.1
        let potential = vec![f64::INFINITY; n_chains]; // signals first call

        // Generate unit velocities (normalized random)
        let mut u = vec![0.0f64; n_chains * dim];
        for c in 0..n_chains {
            let norm = (dim as f64).sqrt();
            for d in 0..dim {
                u[c * dim + d] = 1.0 / norm;
            }
        }

        accel.upload_state(&x, &u, &potential, &grad).unwrap();
        accel.set_uniform_eps(0.1).unwrap();

        // Run 10 transitions
        for _ in 0..10 {
            accel.transition(1.0, 10, true).unwrap();
        }

        // Download and verify finite
        let positions = accel.download_positions().unwrap();
        assert_eq!(positions.len(), n_chains * dim);
        for &v in &positions {
            assert!(v.is_finite(), "position is not finite: {v}");
        }

        let potentials = accel.download_potentials().unwrap();
        assert_eq!(potentials.len(), n_chains);
        for &v in &potentials {
            assert!(v.is_finite(), "potential is not finite: {v}");
            assert!(v >= 0.0, "StdNormal NLL should be >= 0, got {v}");
        }

        let diag = accel.download_diagnostics().unwrap();
        assert_eq!(diag.accepted.len(), n_chains);
        assert_eq!(diag.energy_error.len(), n_chains);
    }

    #[test]
    fn test_metal_mams_fused_kernel() {
        if !MetalMamsAccelerator::is_available() {
            eprintln!("Metal not available, skipping");
            return;
        }

        let n_chains = 32;
        let dim = 2;
        let mut accel = MetalMamsAccelerator::new(n_chains, dim, 0, &[], 123).unwrap();

        let x = vec![0.5f64; n_chains * dim];
        let grad = vec![0.5f64; n_chains * dim];
        let potential = vec![f64::INFINITY; n_chains];
        let norm = (dim as f64).sqrt();
        let u: Vec<f64> = (0..n_chains * dim).map(|_| 1.0 / norm).collect();

        accel.upload_state(&x, &u, &potential, &grad).unwrap();
        accel.set_uniform_eps(0.05).unwrap();

        // Configure batch and run fused
        let n_transitions = 20;
        let n_report = 16;
        accel.configure_batch(n_transitions, n_report).unwrap();

        let result = accel.transition_fused(1.0, 20, n_transitions).unwrap();

        assert_eq!(result.positions.len(), n_transitions);
        assert_eq!(result.potentials.len(), n_transitions);
        assert_eq!(result.accepted.len(), n_transitions);
        assert_eq!(result.energy_error.len(), n_transitions);
        assert_eq!(result.n_launches, 1);

        // Verify all positions are finite
        for (s, pos) in result.positions.iter().enumerate() {
            assert_eq!(pos.len(), n_report * dim);
            for &v in pos {
                assert!(v.is_finite(), "fused pos[{s}] not finite: {v}");
            }
        }
    }

    #[test]
    fn test_metal_mams_batch_transition() {
        if !MetalMamsAccelerator::is_available() {
            eprintln!("Metal not available, skipping");
            return;
        }

        let n_chains = 32;
        let dim = 3;
        let mut accel = MetalMamsAccelerator::new(n_chains, dim, 0, &[], 99).unwrap();

        let x = vec![0.2f64; n_chains * dim];
        let grad = vec![0.2f64; n_chains * dim];
        let potential = vec![f64::INFINITY; n_chains];
        let norm = (dim as f64).sqrt();
        let u: Vec<f64> = (0..n_chains * dim).map(|_| 1.0 / norm).collect();

        accel.upload_state(&x, &u, &potential, &grad).unwrap();
        accel.set_uniform_eps(0.08).unwrap();

        let batch_size = 10;
        let n_report = 8;
        accel.configure_batch(batch_size, n_report).unwrap();

        let result = accel.transition_batch(1.0, 15, batch_size).unwrap();

        assert_eq!(result.positions.len(), batch_size);
        for pos in &result.positions {
            assert_eq!(pos.len(), n_report * dim);
        }
    }

    #[test]
    fn test_metal_mams_trait_api() {
        if !MetalMamsAccelerator::is_available() {
            eprintln!("Metal not available, skipping");
            return;
        }

        // Verify trait methods work via dynamic dispatch
        let n_chains = 16;
        let dim = 2;
        let mut accel: Box<dyn MamsAccelerator> =
            Box::new(MetalMamsAccelerator::new(n_chains, dim, 0, &[], 77).unwrap());

        assert_eq!(accel.n_chains(), n_chains);
        assert_eq!(accel.dim(), dim);
        assert!(accel.supports_fused());

        let x = vec![0.0f64; n_chains * dim];
        let u = vec![1.0f64 / (dim as f64).sqrt(); n_chains * dim];
        let potential = vec![f64::INFINITY; n_chains];
        let grad = vec![0.0f64; n_chains * dim];

        accel.upload_state(&x, &u, &potential, &grad).unwrap();
        accel.set_uniform_eps(0.1).unwrap();
        accel.transition_auto(1.0, 10, false).unwrap();

        let pos = accel.download_positions().unwrap();
        assert_eq!(pos.len(), n_chains * dim);
    }
}
