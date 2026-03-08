//! CUDA leapfrog kernel wrapper for an internal StdNormal HMC prototype.
//!
//! This module intentionally implements only the smallest useful slice needed
//! to validate the tree-HMC stepper seam on GPU:
//! - single-chain host-driven execution
//! - Standard Normal potential `U(q) = 0.5 * ||q||^2`
//! - diagonal inverse mass matrix only
//! - device-resident batched leapfrog sequences with H2D / D2H at sequence
//!   boundaries
//!
//! It is not a public WALNUTS backend. Higher-level sampler integration lives
//! in `ns-inference`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const PTX_SRC: &str = include_str!(env!("CUDA_HMC_STDNORMAL_PTX_PATH"));
const KERNEL_SRC: &str = include_str!("../kernels/hmc_stdnormal_step.cu");

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA hmc stdnormal: {msg}"))
}

fn embedded_ptx_is_stub() -> bool {
    PTX_SRC.contains("STUB PTX")
}

fn load_kernel_from_ptx(
    ctx: &Arc<CudaContext>,
    ptx_src: &str,
) -> ns_core::Result<(Arc<CudaStream>, CudaFunction, CudaFunction)> {
    let stream = ctx.default_stream();
    let ptx = Ptx::from_src(ptx_src);
    let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
    let kernel_step = module
        .load_function("hmc_stdnormal_leapfrog_diag")
        .map_err(|e| cuda_err(format!("load hmc_stdnormal_leapfrog_diag: {e}")))?;
    let kernel_log_joint = module
        .load_function("hmc_stdnormal_log_joint_diag")
        .map_err(|e| cuda_err(format!("load hmc_stdnormal_log_joint_diag: {e}")))?;
    Ok((stream, kernel_step, kernel_log_joint))
}

fn detect_gpu_arch_for_device(device_id: usize) -> ns_core::Result<String> {
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

        Ok(format!("sm_{major}{minor}"))
    }
}

fn compile_kernel_ptx_for_arch(arch: &str) -> ns_core::Result<String> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    let inferred_arch = if let Some(cc) = arch.strip_prefix("sm_") {
        format!("compute_{cc}")
    } else {
        arch.to_string()
    };
    let opts = CompileOptions {
        prec_sqrt: Some(true),
        prec_div: Some(true),
        fmad: Some(true),
        arch: None,
        options: vec![format!("--gpu-architecture={inferred_arch}")],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
        .map_err(|e| cuda_err(format!("NVRTC compile hmc_stdnormal_step failed: {e}")))?;
    Ok(ptx.to_src())
}

fn load_kernel(
    ctx: &Arc<CudaContext>,
    device_id: usize,
) -> ns_core::Result<(Arc<CudaStream>, CudaFunction, CudaFunction)> {
    if !embedded_ptx_is_stub()
        && let Ok(loaded) = load_kernel_from_ptx(ctx, PTX_SRC)
    {
        return Ok(loaded);
    }

    let arch = detect_gpu_arch_for_device(device_id)?;
    let ptx_src = compile_kernel_ptx_for_arch(&arch)?;
    load_kernel_from_ptx(ctx, &ptx_src)
}

/// Minimal CUDA leapfrog wrapper for a StdNormal target with diagonal metric.
pub struct CudaStdNormalLeapfrog {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_step: CudaFunction,
    kernel_log_joint: CudaFunction,
    d_q_in: CudaSlice<f64>,
    d_p_in: CudaSlice<f64>,
    d_q_out: CudaSlice<f64>,
    d_p_out: CudaSlice<f64>,
    d_inv_mass: CudaSlice<f64>,
    d_log_joint_partials: CudaSlice<f64>,
    dim: usize,
    n_partial_blocks: usize,
    host_q_cache: Vec<f64>,
    host_p_cache: Vec<f64>,
    host_log_joint_partials: Vec<f64>,
    host_cache_valid: bool,
}

impl CudaStdNormalLeapfrog {
    fn sequence_err(error: ns_core::Error, attempted_steps: usize) -> (ns_core::Error, usize) {
        (error, attempted_steps)
    }

    /// True when a CUDA device is available and the step kernel can be loaded.
    pub fn is_available() -> bool {
        let ctx = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(Ok(ctx)) => ctx,
            _ => return false,
        };
        load_kernel(&ctx, 0).is_ok()
    }

    /// Create a wrapper on a specific device for a fixed diagonal inverse mass.
    pub fn new_on_device(inv_mass: &[f64], device_id: usize) -> ns_core::Result<Self> {
        if inv_mass.is_empty() {
            return Err(ns_core::Error::Validation(
                "CUDA StdNormal HMC requires dim > 0".to_string(),
            ));
        }
        if inv_mass.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err(ns_core::Error::Validation(
                "CUDA StdNormal HMC requires finite positive diagonal inverse mass".to_string(),
            ));
        }

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(device_id)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context (device {device_id}): {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let (stream, kernel_step, kernel_log_joint) = load_kernel(&ctx, device_id)?;

        let dim = inv_mass.len();
        let n_partial_blocks = dim.div_ceil(256);
        let zeros = vec![0.0f64; dim];
        let d_q_in = stream.clone_htod(&zeros).map_err(cuda_err)?;
        let d_p_in = stream.clone_htod(&zeros).map_err(cuda_err)?;
        let d_q_out = stream.clone_htod(&zeros).map_err(cuda_err)?;
        let d_p_out = stream.clone_htod(&zeros).map_err(cuda_err)?;
        let d_inv_mass = stream.clone_htod(inv_mass).map_err(cuda_err)?;
        let d_log_joint_partials = stream.alloc_zeros::<f64>(n_partial_blocks).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_step,
            kernel_log_joint,
            d_q_in,
            d_p_in,
            d_q_out,
            d_p_out,
            d_inv_mass,
            d_log_joint_partials,
            dim,
            n_partial_blocks,
            host_q_cache: zeros.clone(),
            host_p_cache: zeros,
            host_log_joint_partials: vec![0.0; n_partial_blocks],
            host_cache_valid: false,
        })
    }

    /// Dimension of the state carried by this wrapper.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn launch_step(&mut self, eps: f64) -> ns_core::Result<()> {
        let n_arg = self.dim as i32;
        let config = LaunchConfig {
            grid_dim: (((self.dim as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.kernel_step);
        builder.arg(&self.d_q_in);
        builder.arg(&self.d_p_in);
        builder.arg(&self.d_inv_mass);
        builder.arg(&eps);
        builder.arg(&n_arg);
        builder.arg(&mut self.d_q_out);
        builder.arg(&mut self.d_p_out);
        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch: {e}")))?;
        }

        std::mem::swap(&mut self.d_q_in, &mut self.d_q_out);
        std::mem::swap(&mut self.d_p_in, &mut self.d_p_out);
        Ok(())
    }

    fn launch_log_joint(&mut self) -> ns_core::Result<f64> {
        let n_arg = self.dim as i32;
        let config = LaunchConfig {
            grid_dim: (self.n_partial_blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f64>() as u32,
        };

        let mut builder = self.stream.launch_builder(&self.kernel_log_joint);
        builder.arg(&self.d_q_in);
        builder.arg(&self.d_p_in);
        builder.arg(&self.d_inv_mass);
        builder.arg(&n_arg);
        builder.arg(&mut self.d_log_joint_partials);
        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch log-joint: {e}")))?;
        }

        self.stream
            .memcpy_dtoh(&self.d_log_joint_partials, &mut self.host_log_joint_partials)
            .map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(self.host_log_joint_partials.iter().sum())
    }

    fn cache_matches(&self, q: &[f64], p: &[f64]) -> bool {
        self.host_cache_valid
            && self.host_q_cache.as_slice() == q
            && self.host_p_cache.as_slice() == p
    }

    fn upload_state_if_needed(&mut self, q: &[f64], p: &[f64]) -> ns_core::Result<()> {
        if self.cache_matches(q, p) {
            return Ok(());
        }
        self.stream.memcpy_htod(q, &mut self.d_q_in).map_err(cuda_err)?;
        self.stream.memcpy_htod(p, &mut self.d_p_in).map_err(cuda_err)?;
        Ok(())
    }

    fn download_state(&mut self, q: &mut [f64], p: &mut [f64]) -> ns_core::Result<()> {
        self.stream.memcpy_dtoh(&self.d_q_in, q).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_p_in, p).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        self.host_q_cache.clone_from_slice(q);
        self.host_p_cache.clone_from_slice(p);
        self.host_cache_valid = true;
        Ok(())
    }

    fn step_many_counted(
        &mut self,
        q: &mut [f64],
        p: &mut [f64],
        eps: f64,
        n_steps: usize,
    ) -> std::result::Result<usize, (ns_core::Error, usize)> {
        if q.len() != self.dim || p.len() != self.dim {
            return Err(Self::sequence_err(
                ns_core::Error::Validation(format!(
                    "CUDA StdNormal HMC dimension mismatch: expected {}, got q={}, p={}",
                    self.dim,
                    q.len(),
                    p.len()
                )),
                0,
            ));
        }
        if !eps.is_finite() {
            return Err(Self::sequence_err(
                ns_core::Error::Validation(
                    "CUDA StdNormal HMC requires finite step size".to_string(),
                ),
                0,
            ));
        }
        if n_steps == 0 {
            return Ok(0);
        }

        self.upload_state_if_needed(q, p).map_err(|error| Self::sequence_err(error, 0))?;
        for attempt_idx in 0..n_steps {
            self.launch_step(eps).map_err(|error| Self::sequence_err(error, attempt_idx + 1))?;
        }
        self.download_state(q, p).map_err(|error| Self::sequence_err(error, n_steps))?;
        Ok(n_steps)
    }

    fn probe_log_joint_many_counted(
        &mut self,
        q: &[f64],
        p: &[f64],
        eps: f64,
        n_steps: usize,
    ) -> std::result::Result<(f64, usize), (ns_core::Error, usize)> {
        if q.len() != self.dim || p.len() != self.dim {
            return Err(Self::sequence_err(
                ns_core::Error::Validation(format!(
                    "CUDA StdNormal HMC dimension mismatch: expected {}, got q={}, p={}",
                    self.dim,
                    q.len(),
                    p.len()
                )),
                0,
            ));
        }
        if !eps.is_finite() {
            return Err(Self::sequence_err(
                ns_core::Error::Validation(
                    "CUDA StdNormal HMC requires finite step size".to_string(),
                ),
                0,
            ));
        }

        self.upload_state_if_needed(q, p).map_err(|error| Self::sequence_err(error, 0))?;
        for attempt_idx in 0..n_steps {
            self.launch_step(eps).map_err(|error| Self::sequence_err(error, attempt_idx + 1))?;
        }
        let log_joint =
            self.launch_log_joint().map_err(|error| Self::sequence_err(error, n_steps))?;
        if n_steps > 0 {
            self.host_cache_valid = false;
        }
        Ok((log_joint, n_steps))
    }

    /// Advance one or more explicit leapfrog steps.
    pub fn step_many(
        &mut self,
        q: &mut [f64],
        p: &mut [f64],
        eps: f64,
        n_steps: usize,
    ) -> ns_core::Result<()> {
        self.step_many_counted(q, p, eps, n_steps).map(|_| ()).map_err(|(error, _)| error)
    }

    /// Advance one explicit leapfrog step.
    pub fn step(&mut self, q: &mut [f64], p: &mut [f64], eps: f64) -> ns_core::Result<()> {
        self.step_many(q, p, eps, 1)
    }

    /// Advance one or more explicit leapfrog steps and report the exact number of attempts.
    pub fn step_many_attempted(
        &mut self,
        q: &mut [f64],
        p: &mut [f64],
        eps: f64,
        n_steps: usize,
    ) -> std::result::Result<usize, (ns_core::Error, usize)> {
        self.step_many_counted(q, p, eps, n_steps)
    }

    /// Advance explicit leapfrog steps on device and return only the final log-joint.
    pub fn probe_log_joint_many_attempted(
        &mut self,
        q: &[f64],
        p: &[f64],
        eps: f64,
        n_steps: usize,
    ) -> std::result::Result<(f64, usize), (ns_core::Error, usize)> {
        self.probe_log_joint_many_counted(q, p, eps, n_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_step_matches_closed_form_diag() {
        if !CudaStdNormalLeapfrog::is_available() {
            return;
        }

        let inv_mass = vec![1.0, 0.5, 2.0];
        let mut accel = CudaStdNormalLeapfrog::new_on_device(&inv_mass, 0).unwrap();
        let mut q = vec![1.0, -0.5, 0.25];
        let mut p = vec![0.2, -0.4, 0.6];
        let eps = 0.1;

        let expected: Vec<(f64, f64)> = q
            .iter()
            .zip(p.iter())
            .zip(inv_mass.iter())
            .map(|((&q0, &p0), &inv_m)| {
                let p_half = p0 - 0.5 * eps * q0;
                let q1 = q0 + eps * inv_m * p_half;
                let p1 = p_half - 0.5 * eps * q1;
                (q1, p1)
            })
            .collect();

        accel.step(&mut q, &mut p, eps).unwrap();

        for (i, ((q_expected, p_expected), (&q_got, &p_got))) in
            expected.iter().zip(q.iter().zip(p.iter())).enumerate()
        {
            assert!(
                (q_expected - q_got).abs() < 1e-12,
                "q mismatch at {i}: expected {}, got {}",
                q_expected,
                q_got
            );
            assert!(
                (p_expected - p_got).abs() < 1e-12,
                "p mismatch at {i}: expected {}, got {}",
                p_expected,
                p_got
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_step_many_matches_repeated_diag() {
        if !CudaStdNormalLeapfrog::is_available() {
            return;
        }

        let inv_mass = vec![1.0, 0.5, 2.0];
        let mut accel = CudaStdNormalLeapfrog::new_on_device(&inv_mass, 0).unwrap();
        let mut q_many = vec![1.0, -0.5, 0.25];
        let mut p_many = vec![0.2, -0.4, 0.6];
        let mut q_repeated = q_many.clone();
        let mut p_repeated = p_many.clone();
        let eps = 0.1;

        accel.step_many(&mut q_many, &mut p_many, eps, 5).unwrap();

        for _ in 0..5 {
            let expected: Vec<(f64, f64)> = q_repeated
                .iter()
                .zip(p_repeated.iter())
                .zip(inv_mass.iter())
                .map(|((&q0, &p0), &inv_m)| {
                    let p_half = p0 - 0.5 * eps * q0;
                    let q1 = q0 + eps * inv_m * p_half;
                    let p1 = p_half - 0.5 * eps * q1;
                    (q1, p1)
                })
                .collect();
            for (i, (q_next, p_next)) in expected.iter().copied().enumerate() {
                q_repeated[i] = q_next;
                p_repeated[i] = p_next;
            }
        }

        for (i, ((&q_expected, &p_expected), (&q_got, &p_got))) in q_repeated
            .iter()
            .zip(p_repeated.iter())
            .zip(q_many.iter().zip(p_many.iter()))
            .enumerate()
        {
            assert!(
                (q_expected - q_got).abs() < 1e-12,
                "q mismatch at {i}: expected {}, got {}",
                q_expected,
                q_got
            );
            assert!(
                (p_expected - p_got).abs() < 1e-12,
                "p mismatch at {i}: expected {}, got {}",
                p_expected,
                p_got
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stdnormal_probe_log_joint_matches_repeated_diag() {
        if !CudaStdNormalLeapfrog::is_available() {
            return;
        }

        let inv_mass = vec![1.0, 0.5, 2.0];
        let mut accel = CudaStdNormalLeapfrog::new_on_device(&inv_mass, 0).unwrap();
        let q0 = vec![1.0, -0.5, 0.25];
        let p0 = vec![0.2, -0.4, 0.6];
        let eps = -0.1;
        let n_steps = 3;

        let mut q_expected = q0.clone();
        let mut p_expected = p0.clone();
        for _ in 0..n_steps {
            for i in 0..q_expected.len() {
                let p_half = p_expected[i] - 0.5 * eps * q_expected[i];
                let q1 = q_expected[i] + eps * inv_mass[i] * p_half;
                let p1 = p_half - 0.5 * eps * q1;
                q_expected[i] = q1;
                p_expected[i] = p1;
            }
        }
        let expected_log_joint = -0.5
            * q_expected
                .iter()
                .zip(p_expected.iter())
                .zip(inv_mass.iter())
                .map(|((&q, &p), &inv_m)| q * q + p * p * inv_m)
                .sum::<f64>();

        let (log_joint, attempted_steps) =
            accel.probe_log_joint_many_attempted(&q0, &p0, eps, n_steps).unwrap();
        assert_eq!(attempted_steps, n_steps);
        assert!((log_joint - expected_log_joint).abs() < 1e-12);

        // Probe-only execution must not poison the next host-driven sequence.
        let mut q_after = q0.clone();
        let mut p_after = p0.clone();
        accel.step_many(&mut q_after, &mut p_after, eps, n_steps).unwrap();
        for i in 0..q_after.len() {
            assert!((q_after[i] - q_expected[i]).abs() < 1e-12);
            assert!((p_after[i] - p_expected[i]).abs() < 1e-12);
        }
    }
}
