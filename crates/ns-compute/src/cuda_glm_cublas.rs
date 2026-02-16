//! cuBLAS-based batched GLM logistic evaluator on CUDA.
//!
//! This module provides a focused primitive for large-`n` logistic GLM:
//! - `eta = X @ beta` for many chains in parallel (strided batched GEMM)
//! - `grad = X^T @ (sigmoid(eta) - y)` (strided batched GEMM)
//! - `nll = sum(log(1+exp(eta)) - y*eta) + 0.5*||beta||^2`
//!
//! It is intentionally kept separate from MAMS transition kernels so we can
//! benchmark and validate a cuBLAS path before deeper integrator refactors.

#![cfg(feature = "cuda")]

use core::ffi::c_int;
use cudarc::cublas::safe::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::{CudaBlas, result::CublasError, sys as cublas_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const AUX_KERNEL_SRC: &str = include_str!("../kernels/glm_cublas_aux.cu");

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA GLM cuBLAS: {msg}"))
}

fn cublas_err(msg: impl std::fmt::Display, err: CublasError) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA GLM cuBLAS: {msg}: {err:?}"))
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
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    if let Ok(override_ptx) = std::env::var("NS_GLM_CUBLAS_AUX_PTX_OVERRIDE")
        && !override_ptx.trim().is_empty()
    {
        let src = std::fs::read_to_string(&override_ptx).map_err(|e| {
            cuda_err(format!("read NS_GLM_CUBLAS_AUX_PTX_OVERRIDE={override_ptx}: {e}"))
        })?;
        if src.trim().is_empty() {
            return Err(cuda_err(format!(
                "NS_GLM_CUBLAS_AUX_PTX_OVERRIDE is empty: {override_ptx}"
            )));
        }
        return Ok(src);
    }

    let inferred_arch = if let Some(cc) = arch.strip_prefix("sm_") {
        format!("compute_{cc}")
    } else {
        arch.to_string()
    };

    let mut try_arches = Vec::new();
    if let Ok(override_arch) = std::env::var("NS_GLM_CUBLAS_NVRTC_ARCH") {
        if !override_arch.trim().is_empty() {
            try_arches.push(override_arch);
        }
    }
    if try_arches.is_empty() {
        try_arches.push(inferred_arch.clone());
        // CUDA 13 toolchains may drop compute_70 from NVRTC options.
        // Keep a fallback for diagnostic probing on legacy devices.
        if inferred_arch == "compute_70" {
            try_arches.push("compute_75".to_string());
        }
    }

    let mut errs = Vec::new();
    for nvrtc_arch in try_arches {
        let opts = CompileOptions {
            prec_sqrt: Some(true),
            prec_div: Some(true),
            // Keep numerics stable for logistic + NLL path.
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

    Err(cuda_err(format!("NVRTC compile glm_cublas_aux failed:\n{}", errs.join("\n"))))
}

/// Batched GLM logistic evaluator backed by cuBLAS.
pub struct CudaGlmCublasEvaluator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k_diff_nll: CudaFunction,
    k_add_prior: CudaFunction,
    d_x_col: CudaSlice<f64>,
    d_y: CudaSlice<f64>,
    d_beta: CudaSlice<f64>,
    d_eta: CudaSlice<f64>,
    d_diff: CudaSlice<f64>,
    d_grad: CudaSlice<f64>,
    d_nll: CudaSlice<f64>,
    n: usize,
    p: usize,
    n_chains: usize,
    zeros_grad: Vec<f64>,
    zeros_nll: Vec<f64>,
}

impl CudaGlmCublasEvaluator {
    /// Create evaluator on a specific GPU device.
    pub fn new_on_device(
        x_col: &[f64], // column-major [p * n]
        y: &[f64],     // [n]
        n: usize,
        p: usize,
        n_chains: usize,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        if n == 0 || p == 0 || n_chains == 0 {
            return Err(ns_core::Error::Validation("n, p and n_chains must be > 0".into()));
        }
        if x_col.len() != n * p {
            return Err(ns_core::Error::Validation(format!(
                "x_col length mismatch: expected {}, got {}",
                n * p,
                x_col.len()
            )));
        }
        if y.len() != n {
            return Err(ns_core::Error::Validation(format!(
                "y length mismatch: expected {}, got {}",
                n,
                y.len()
            )));
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
            .load_function("glm_logistic_diff_nll")
            .map_err(|e| cuda_err(format!("load glm_logistic_diff_nll: {e}")))?;
        let k_add_prior = module
            .load_function("glm_add_prior")
            .map_err(|e| cuda_err(format!("load glm_add_prior: {e}")))?;

        let d_x_col = stream.clone_htod(x_col).map_err(cuda_err)?;
        let d_y = stream.clone_htod(y).map_err(cuda_err)?;
        let d_beta = stream.alloc_zeros::<f64>(n_chains * p).map_err(cuda_err)?;
        let d_eta = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_diff = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_grad = stream.alloc_zeros::<f64>(n_chains * p).map_err(cuda_err)?;
        let d_nll = stream.alloc_zeros::<f64>(n_chains).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            blas,
            k_diff_nll,
            k_add_prior,
            d_x_col,
            d_y,
            d_beta,
            d_eta,
            d_diff,
            d_grad,
            d_nll,
            n,
            p,
            n_chains,
            zeros_grad: vec![0.0; n_chains * p],
            zeros_nll: vec![0.0; n_chains],
        })
    }

    /// Create evaluator on GPU 0.
    pub fn new(
        x_col: &[f64],
        y: &[f64],
        n: usize,
        p: usize,
        n_chains: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device(x_col, y, n, p, n_chains, 0)
    }

    /// Evaluate batched GLM logistic grad and NLL for `beta` (shape `[n_chains * p]`).
    ///
    /// Returns `(grad_flat, nll)` with shapes `[n_chains * p]` and `[n_chains]`.
    pub fn evaluate_host(&mut self, beta: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if beta.len() != self.n_chains * self.p {
            return Err(ns_core::Error::Validation(format!(
                "beta length mismatch: expected {}, got {}",
                self.n_chains * self.p,
                beta.len()
            )));
        }

        self.stream.memcpy_htod(beta, &mut self.d_beta).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_grad, &mut self.d_grad).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_nll, &mut self.d_nll).map_err(cuda_err)?;

        // eta = X @ beta for each chain (X shared across batch).
        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            m: self.n as c_int,
                            n: 1,
                            k: self.p as c_int,
                            alpha: 1.0f64,
                            lda: self.n as c_int,
                            ldb: self.p as c_int,
                            beta: 0.0f64,
                            ldc: self.n as c_int,
                        },
                        batch_size: self.n_chains as c_int,
                        stride_a: 0,
                        stride_b: self.p as i64,
                        stride_c: self.n as i64,
                    },
                    &self.d_x_col,
                    &self.d_beta,
                    &mut self.d_eta,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X @ beta)", e))?;
        }

        // diff = sigmoid(eta)-y, nll_data = sum(log1pexp - y*eta)
        let total = self.n * self.n_chains;
        let block = 256u32;
        let grid = (total as u32).div_ceil(block).min(65535);
        let cfg =
            LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        let n_arg = self.n as c_int;
        let n_chains_arg = self.n_chains as c_int;
        let mut builder = self.stream.launch_builder(&self.k_diff_nll);
        builder.arg(&self.d_eta);
        builder.arg(&self.d_y);
        builder.arg(&mut self.d_diff);
        builder.arg(&mut self.d_nll);
        builder.arg(&n_arg);
        builder.arg(&n_chains_arg);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| cuda_err(format!("launch glm_logistic_diff_nll: {e}")))?;
        }

        // grad_data = X^T @ diff for each chain.
        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            m: self.p as c_int,
                            n: 1,
                            k: self.n as c_int,
                            alpha: 1.0f64,
                            lda: self.n as c_int,
                            ldb: self.n as c_int,
                            beta: 0.0f64,
                            ldc: self.p as c_int,
                        },
                        batch_size: self.n_chains as c_int,
                        stride_a: 0,
                        stride_b: self.n as i64,
                        stride_c: self.p as i64,
                    },
                    &self.d_x_col,
                    &self.d_diff,
                    &mut self.d_grad,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X^T @ diff)", e))?;
        }

        // Add Gaussian prior: grad += beta; nll += 0.5 * ||beta||^2
        let total_beta = self.n_chains * self.p;
        let grid_prior = (total_beta as u32).div_ceil(block).min(65535);
        let cfg_prior = LaunchConfig {
            grid_dim: (grid_prior, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let p_arg = self.p as c_int;
        let mut builder = self.stream.launch_builder(&self.k_add_prior);
        builder.arg(&self.d_beta);
        builder.arg(&mut self.d_grad);
        builder.arg(&mut self.d_nll);
        builder.arg(&p_arg);
        builder.arg(&n_chains_arg);
        unsafe {
            builder
                .launch(cfg_prior)
                .map_err(|e| cuda_err(format!("launch glm_add_prior: {e}")))?;
        }

        self.stream.synchronize().map_err(cuda_err)?;

        let mut grad = vec![0.0f64; self.n_chains * self.p];
        let mut nll = vec![0.0f64; self.n_chains];
        self.stream.memcpy_dtoh(&self.d_grad, &mut grad).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_nll, &mut nll).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok((grad, nll))
    }
}
