//! cuBLAS-backed evaluator for interval-censored Weibull AFT models on CUDA.
//!
//! This evaluator mirrors the existing GLM pattern:
//! - `eta = X @ beta` for many chains in parallel via strided batched GEMM
//! - a family-specific CUDA kernel computes the interval-censored Weibull AFT
//!   data NLL, `d(nll)/d(log_lambda_i)`, and `d(nll)/d(log_k)`
//! - `grad_beta = X^T @ diff_log_lambda` via strided batched GEMM
//! - an optional standard normal prior can be added on the full packed parameter vector

#![cfg(feature = "cuda")]

use core::ffi::c_int;
use cudarc::cublas::safe::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::{CudaBlas, result::CublasError, sys as cublas_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use std::sync::Arc;

const AUX_KERNEL_SRC: &str = include_str!("../kernels/weibull_aft_cublas_aux.cu");

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA Weibull AFT cuBLAS: {msg}"))
}

fn cublas_err(msg: impl std::fmt::Display, err: CublasError) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA Weibull AFT cuBLAS: {msg}: {err:?}"))
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

fn compile_aux_ptx_for_arch(arch: &str) -> ns_core::Result<String> {
    if let Ok(override_ptx) = std::env::var("NS_WEIBULL_AFT_CUBLAS_AUX_PTX_OVERRIDE") {
        if !override_ptx.trim().is_empty() {
            let src = std::fs::read_to_string(&override_ptx).map_err(|e| {
                cuda_err(format!("read NS_WEIBULL_AFT_CUBLAS_AUX_PTX_OVERRIDE={override_ptx}: {e}"))
            })?;
            if src.trim().is_empty() {
                return Err(cuda_err(format!(
                    "NS_WEIBULL_AFT_CUBLAS_AUX_PTX_OVERRIDE is empty: {override_ptx}"
                )));
            }
            return Ok(src);
        }
    }

    let inferred_arch = if let Some(cc) = arch.strip_prefix("sm_") {
        format!("compute_{cc}")
    } else {
        arch.to_string()
    };

    let mut try_arches = Vec::new();
    if let Ok(override_arch) = std::env::var("NS_WEIBULL_AFT_CUBLAS_NVRTC_ARCH") {
        if !override_arch.trim().is_empty() {
            try_arches.push(override_arch);
        }
    }
    if try_arches.is_empty() {
        try_arches.push(inferred_arch.clone());
        if inferred_arch == "compute_70" {
            try_arches.push("compute_75".to_string());
        }
    }

    let mut errs = Vec::new();
    for nvrtc_arch in try_arches {
        let opts = CompileOptions {
            prec_sqrt: Some(true),
            prec_div: Some(true),
            fmad: Some(true),
            arch: None,
            options: vec![format!("--gpu-architecture={nvrtc_arch}")],
            ..Default::default()
        };
        match compile_ptx_with_opts(AUX_KERNEL_SRC, opts) {
            Ok(ptx) => return Ok(ptx.to_src()),
            Err(e) => errs.push(format!("{nvrtc_arch}: {e}")),
        }
    }

    Err(cuda_err(format!("NVRTC compile weibull_aft_cublas_aux failed:\n{}", errs.join("\n"))))
}

/// CUDA evaluator for interval-censored Weibull AFT models with a shared
/// standard normal prior on `[log_k, beta...]`.
pub struct CudaWeibullAftCublasEvaluator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k_diff_nll: CudaFunction,
    k_scatter_beta_grad: CudaFunction,
    k_add_prior: CudaFunction,
    d_x_col: CudaSlice<f64>,
    d_time_lower: CudaSlice<f64>,
    d_time_upper: CudaSlice<f64>,
    d_ln_time_lower: CudaSlice<f64>,
    d_ln_time_upper: CudaSlice<f64>,
    d_censor_code: CudaSlice<u8>,
    d_params: CudaSlice<f64>,
    d_beta_params: CudaSlice<f64>,
    d_eta: CudaSlice<f64>,
    d_diff_log_lambda: CudaSlice<f64>,
    d_beta_grad: CudaSlice<f64>,
    d_grad: CudaSlice<f64>,
    d_nll: CudaSlice<f64>,
    n: usize,
    beta_dim: usize,
    param_dim: usize,
    n_chains: usize,
    zeros_beta_grad: Vec<f64>,
    zeros_grad: Vec<f64>,
    zeros_nll: Vec<f64>,
    apply_standard_normal_prior: bool,
}

impl CudaWeibullAftCublasEvaluator {
    /// Create a CUDA evaluator for packed interval-censored Weibull AFT data on
    /// a specific device.
    pub fn new_on_device(
        x_col: &[f64],
        time_lower: &[f64],
        time_upper: &[f64],
        ln_time_lower: &[f64],
        ln_time_upper: &[f64],
        censor_code: &[u8],
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(
            x_col,
            time_lower,
            time_upper,
            ln_time_lower,
            ln_time_upper,
            censor_code,
            n,
            beta_dim,
            n_chains,
            device_id,
            true,
        )
    }

    /// Create a CUDA evaluator without a prior term.
    pub fn new_on_device_no_prior(
        x_col: &[f64],
        time_lower: &[f64],
        time_upper: &[f64],
        ln_time_lower: &[f64],
        ln_time_upper: &[f64],
        censor_code: &[u8],
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(
            x_col,
            time_lower,
            time_upper,
            ln_time_lower,
            ln_time_upper,
            censor_code,
            n,
            beta_dim,
            n_chains,
            device_id,
            false,
        )
    }

    fn new_on_device_impl(
        x_col: &[f64],
        time_lower: &[f64],
        time_upper: &[f64],
        ln_time_lower: &[f64],
        ln_time_upper: &[f64],
        censor_code: &[u8],
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        device_id: usize,
        apply_standard_normal_prior: bool,
    ) -> ns_core::Result<Self> {
        if n == 0 || beta_dim == 0 || n_chains == 0 {
            return Err(ns_core::Error::Validation("n, beta_dim and n_chains must be > 0".into()));
        }
        if x_col.len() != n * beta_dim {
            return Err(ns_core::Error::Validation(format!(
                "x_col length mismatch: expected {}, got {}",
                n * beta_dim,
                x_col.len()
            )));
        }
        for (name, values) in [
            ("time_lower", time_lower),
            ("time_upper", time_upper),
            ("ln_time_lower", ln_time_lower),
            ("ln_time_upper", ln_time_upper),
        ] {
            if values.len() != n {
                return Err(ns_core::Error::Validation(format!(
                    "{name} length mismatch: expected {n}, got {}",
                    values.len()
                )));
            }
        }
        if censor_code.len() != n {
            return Err(ns_core::Error::Validation(format!(
                "censor_code length mismatch: expected {n}, got {}",
                censor_code.len()
            )));
        }
        if censor_code.iter().any(|code| *code > 3) {
            return Err(ns_core::Error::Validation(
                "censor_code must use 0=exact,1=right,2=left,3=interval".into(),
            ));
        }

        let ctx = CudaContext::new(device_id)
            .map_err(|e| cuda_err(format!("context (device {device_id}): {e}")))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| cublas_err("create handle", e))?;

        let arch = detect_gpu_arch_for_device(device_id)?;
        let ptx_src = compile_aux_ptx_for_arch(&arch)?;
        let module = ctx
            .load_module(Ptx::from_src(ptx_src))
            .map_err(|e| cuda_err(format!("load module: {e}")))?;
        let k_diff_nll = module
            .load_function("weibull_aft_diff_nll")
            .map_err(|e| cuda_err(format!("load weibull_aft_diff_nll: {e}")))?;
        let k_scatter_beta_grad = module
            .load_function("weibull_aft_scatter_beta_grad")
            .map_err(|e| cuda_err(format!("load weibull_aft_scatter_beta_grad: {e}")))?;
        let k_add_prior = module
            .load_function("weibull_aft_add_prior")
            .map_err(|e| cuda_err(format!("load weibull_aft_add_prior: {e}")))?;

        let d_x_col = stream.clone_htod(x_col).map_err(cuda_err)?;
        let d_time_lower = stream.clone_htod(time_lower).map_err(cuda_err)?;
        let d_time_upper = stream.clone_htod(time_upper).map_err(cuda_err)?;
        let d_ln_time_lower = stream.clone_htod(ln_time_lower).map_err(cuda_err)?;
        let d_ln_time_upper = stream.clone_htod(ln_time_upper).map_err(cuda_err)?;
        let d_censor_code = stream.clone_htod(censor_code).map_err(cuda_err)?;

        let param_dim = beta_dim + 1;
        let d_params = stream.alloc_zeros::<f64>(n_chains * param_dim).map_err(cuda_err)?;
        let d_beta_params = stream.alloc_zeros::<f64>(n_chains * beta_dim).map_err(cuda_err)?;
        let d_eta = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_diff_log_lambda = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_beta_grad = stream.alloc_zeros::<f64>(n_chains * beta_dim).map_err(cuda_err)?;
        let d_grad = stream.alloc_zeros::<f64>(n_chains * param_dim).map_err(cuda_err)?;
        let d_nll = stream.alloc_zeros::<f64>(n_chains).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            blas,
            k_diff_nll,
            k_scatter_beta_grad,
            k_add_prior,
            d_x_col,
            d_time_lower,
            d_time_upper,
            d_ln_time_lower,
            d_ln_time_upper,
            d_censor_code,
            d_params,
            d_beta_params,
            d_eta,
            d_diff_log_lambda,
            d_beta_grad,
            d_grad,
            d_nll,
            n,
            beta_dim,
            param_dim,
            n_chains,
            zeros_beta_grad: vec![0.0; n_chains * beta_dim],
            zeros_grad: vec![0.0; n_chains * param_dim],
            zeros_nll: vec![0.0; n_chains],
            apply_standard_normal_prior,
        })
    }

    /// Evaluate batched gradients and NLL for packed per-chain parameters
    /// `[log_k, beta...]`.
    pub fn evaluate_host(&mut self, params: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params.len() != self.n_chains * self.param_dim {
            return Err(ns_core::Error::Validation(format!(
                "parameter length mismatch: expected {}, got {}",
                self.n_chains * self.param_dim,
                params.len()
            )));
        }

        let mut beta_params = vec![0.0; self.n_chains * self.beta_dim];
        for chain in 0..self.n_chains {
            let src = &params[(chain * self.param_dim + 1)..((chain + 1) * self.param_dim)];
            let dst = &mut beta_params[(chain * self.beta_dim)..((chain + 1) * self.beta_dim)];
            dst.copy_from_slice(src);
        }

        self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        self.stream.memcpy_htod(&beta_params, &mut self.d_beta_params).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_beta_grad, &mut self.d_beta_grad).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_grad, &mut self.d_grad).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_nll, &mut self.d_nll).map_err(cuda_err)?;

        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            m: self.n as c_int,
                            n: 1,
                            k: self.beta_dim as c_int,
                            alpha: 1.0f64,
                            lda: self.n as c_int,
                            ldb: self.beta_dim as c_int,
                            beta: 0.0f64,
                            ldc: self.n as c_int,
                        },
                        batch_size: self.n_chains as c_int,
                        stride_a: 0,
                        stride_b: self.beta_dim as i64,
                        stride_c: self.n as i64,
                    },
                    &self.d_x_col,
                    &self.d_beta_params,
                    &mut self.d_eta,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X @ beta)", e))?;
        }

        let total = self.n * self.n_chains;
        let block = 256u32;
        let grid = (total as u32).div_ceil(block).min(65535);
        let cfg =
            LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        let n_arg = self.n as c_int;
        let beta_dim_arg = self.beta_dim as c_int;
        let param_dim_arg = self.param_dim as c_int;
        let n_chains_arg = self.n_chains as c_int;

        let mut builder = self.stream.launch_builder(&self.k_diff_nll);
        builder.arg(&self.d_eta);
        builder.arg(&self.d_time_lower);
        builder.arg(&self.d_time_upper);
        builder.arg(&self.d_ln_time_lower);
        builder.arg(&self.d_ln_time_upper);
        builder.arg(&self.d_censor_code);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_diff_log_lambda);
        builder.arg(&mut self.d_grad);
        builder.arg(&mut self.d_nll);
        builder.arg(&n_arg);
        builder.arg(&beta_dim_arg);
        builder.arg(&param_dim_arg);
        builder.arg(&n_chains_arg);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| cuda_err(format!("launch weibull_aft_diff_nll: {e}")))?;
        }

        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            m: self.beta_dim as c_int,
                            n: 1,
                            k: self.n as c_int,
                            alpha: 1.0f64,
                            lda: self.n as c_int,
                            ldb: self.n as c_int,
                            beta: 0.0f64,
                            ldc: self.beta_dim as c_int,
                        },
                        batch_size: self.n_chains as c_int,
                        stride_a: 0,
                        stride_b: self.n as i64,
                        stride_c: self.beta_dim as i64,
                    },
                    &self.d_x_col,
                    &self.d_diff_log_lambda,
                    &mut self.d_beta_grad,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X^T @ diff_log_lambda)", e))?;
        }

        let total_beta = self.n_chains * self.beta_dim;
        let grid_beta = (total_beta as u32).div_ceil(block).min(65535);
        let cfg_beta = LaunchConfig {
            grid_dim: (grid_beta, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.k_scatter_beta_grad);
        builder.arg(&self.d_beta_grad);
        builder.arg(&mut self.d_grad);
        builder.arg(&beta_dim_arg);
        builder.arg(&param_dim_arg);
        builder.arg(&n_chains_arg);
        unsafe {
            builder
                .launch(cfg_beta)
                .map_err(|e| cuda_err(format!("launch weibull_aft_scatter_beta_grad: {e}")))?;
        }

        if self.apply_standard_normal_prior {
            let total_params = self.n_chains * self.param_dim;
            let grid_prior = (total_params as u32).div_ceil(block).min(65535);
            let cfg_prior = LaunchConfig {
                grid_dim: (grid_prior, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.stream.launch_builder(&self.k_add_prior);
            builder.arg(&self.d_params);
            builder.arg(&mut self.d_grad);
            builder.arg(&mut self.d_nll);
            builder.arg(&param_dim_arg);
            builder.arg(&n_chains_arg);
            unsafe {
                builder
                    .launch(cfg_prior)
                    .map_err(|e| cuda_err(format!("launch weibull_aft_add_prior: {e}")))?;
            }
        }

        self.stream.synchronize().map_err(cuda_err)?;

        let mut grad = vec![0.0f64; self.n_chains * self.param_dim];
        let mut nll = vec![0.0f64; self.n_chains];
        self.stream.memcpy_dtoh(&self.d_grad, &mut grad).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_nll, &mut nll).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok((grad, nll))
    }
}
