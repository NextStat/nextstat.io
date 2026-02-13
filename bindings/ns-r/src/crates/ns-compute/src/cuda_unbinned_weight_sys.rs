//! CUDA helper kernel: per-event WeightSys morphing for unbinned workflows.
//!
//! Exposes `unbinned_weight_sys_apply`, which morphs per-event weights under multiple
//! WeightSys nuisances and returns both the morphed weights and their per-nuisance
//! derivatives.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/unbinned_weight_sys.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_UNBINNED_WEIGHT_SYS_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (unbinned weight sys): {msg}"))
}

/// CUDA kernel wrapper for per-event WeightSys interpolation.
pub struct CudaUnbinnedWeightSysKernel {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_apply: CudaFunction,

    d_params: CudaSlice<f64>,
    d_w_nom: CudaSlice<f64>,
    d_w_down: CudaSlice<f64>,
    d_w_up: CudaSlice<f64>,
    d_alpha_param_idx: CudaSlice<u32>,
    d_interp_code: CudaSlice<u32>,
    d_w_out: CudaSlice<f64>,
    d_dw_out: CudaSlice<f64>,

    n_params: usize,
    n_events: usize,
    n_sys: usize,
}

impl CudaUnbinnedWeightSysKernel {
    /// Check if CUDA is available at runtime (driver loaded, GPU present).
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Create a kernel wrapper with fixed `(n_params, n_events, n_sys)` and static
    /// nuisance index + interpolation code arrays.
    pub fn new(
        n_params: usize,
        n_events: usize,
        alpha_param_idx: &[u32],
        interp_code: &[u32],
    ) -> ns_core::Result<Self> {
        if n_params == 0 {
            return Err(ns_core::Error::Validation("n_params must be > 0".into()));
        }
        if n_events == 0 {
            return Err(ns_core::Error::Validation("n_events must be > 0".into()));
        }
        if alpha_param_idx.len() != interp_code.len() {
            return Err(ns_core::Error::Validation(format!(
                "alpha_param_idx/interp_code length mismatch: {} != {}",
                alpha_param_idx.len(),
                interp_code.len()
            )));
        }
        let n_sys = alpha_param_idx.len();
        if n_sys == 0 {
            return Err(ns_core::Error::Validation("need at least 1 weight systematic".into()));
        }
        if alpha_param_idx.iter().any(|&i| i as usize >= n_params) {
            return Err(ns_core::Error::Validation(
                "alpha_param_idx contains out-of-range index".into(),
            ));
        }

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context: {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_apply = module
            .load_function("unbinned_weight_sys_apply")
            .map_err(|e| cuda_err(format!("load unbinned_weight_sys_apply: {e}")))?;

        let d_alpha_param_idx = stream.clone_htod(alpha_param_idx).map_err(cuda_err)?;
        let d_interp_code = stream.clone_htod(interp_code).map_err(cuda_err)?;

        let d_params = stream.alloc_zeros::<f64>(n_params).map_err(cuda_err)?;
        let d_w_nom = stream.alloc_zeros::<f64>(n_events).map_err(cuda_err)?;
        let d_w_down = stream.alloc_zeros::<f64>(n_sys * n_events).map_err(cuda_err)?;
        let d_w_up = stream.alloc_zeros::<f64>(n_sys * n_events).map_err(cuda_err)?;
        let d_w_out = stream.alloc_zeros::<f64>(n_events).map_err(cuda_err)?;
        let d_dw_out = stream.alloc_zeros::<f64>(n_sys * n_events).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_apply,
            d_params,
            d_w_nom,
            d_w_down,
            d_w_up,
            d_alpha_param_idx,
            d_interp_code,
            d_w_out,
            d_dw_out,
            n_params,
            n_events,
            n_sys,
        })
    }

    fn launch_config(&self) -> LaunchConfig {
        let block = 256u32;
        let grid = ((self.n_events as u32) + block - 1) / block;
        LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
    }

    /// Apply all configured WeightSys nuisances to weights.
    pub fn apply(
        &mut self,
        params: &[f64],
        w_nom: &[f64],
        w_down: &[f64],
        w_up: &[f64],
    ) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length mismatch: expected {}, got {}",
                self.n_params,
                params.len()
            )));
        }
        if w_nom.len() != self.n_events {
            return Err(ns_core::Error::Validation(format!(
                "w_nom length mismatch: expected {}, got {}",
                self.n_events,
                w_nom.len()
            )));
        }
        let expected_flat = self.n_sys * self.n_events;
        if w_down.len() != expected_flat || w_up.len() != expected_flat {
            return Err(ns_core::Error::Validation(format!(
                "w_down/w_up length mismatch: expected {} (n_sys*n_events), got down={} up={}",
                expected_flat,
                w_down.len(),
                w_up.len()
            )));
        }

        self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        self.stream.memcpy_htod(w_nom, &mut self.d_w_nom).map_err(cuda_err)?;
        self.stream.memcpy_htod(w_down, &mut self.d_w_down).map_err(cuda_err)?;
        self.stream.memcpy_htod(w_up, &mut self.d_w_up).map_err(cuda_err)?;

        let cfg = self.launch_config();
        let np = self.n_params as u32;
        let ne = self.n_events as u32;
        let ns = self.n_sys as u32;

        let mut builder = self.stream.launch_builder(&self.kernel_apply);
        builder.arg(&self.d_params);
        builder.arg(&self.d_w_nom);
        builder.arg(&self.d_w_down);
        builder.arg(&self.d_w_up);
        builder.arg(&self.d_alpha_param_idx);
        builder.arg(&self.d_interp_code);
        builder.arg(&mut self.d_w_out);
        builder.arg(&mut self.d_dw_out);
        builder.arg(&np);
        builder.arg(&ne);
        builder.arg(&ns);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the compiled kernel signature, launch config is within limits.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| cuda_err(format!("launch unbinned_weight_sys_apply: {e}")))?;
        }

        let mut w_out = vec![0.0f64; self.n_events];
        let mut dw_out = vec![0.0f64; expected_flat];
        self.stream.memcpy_dtoh(&self.d_w_out, &mut w_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_dw_out, &mut dw_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok((w_out, dw_out))
    }
}
