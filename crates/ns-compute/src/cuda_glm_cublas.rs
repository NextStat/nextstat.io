//! cuBLAS-based batched GLM evaluator on CUDA.
//!
//! This module provides a focused primitive for large-`n` GLM families:
//! - `eta = X @ beta` for many chains in parallel (strided batched GEMM)
//! - `grad = X^T @ diff(eta, y)` (strided batched GEMM)
//! - `nll = sum(data_nll(eta, y))`, with optional standard-normal prior
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

/// GLM family supported by [`CudaGlmCublasEvaluator`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaGlmFamily {
    /// Gaussian linear regression with fixed sigma=1.
    Linear,
    /// Bernoulli-logit GLM.
    Logistic,
    /// Poisson GLM with log link and no offset term inside the evaluator.
    Poisson,
    /// Negative binomial GLM with log link and trailing `log_alpha`.
    NegativeBinomial,
}

impl CudaGlmFamily {
    fn kernel_name(self) -> &'static str {
        match self {
            Self::Linear => "glm_linear_diff_nll",
            Self::Logistic => "glm_logistic_diff_nll",
            Self::Poisson => "glm_poisson_diff_nll",
            Self::NegativeBinomial => "glm_negbin_diff_nll",
        }
    }

    fn extra_param_dim(self) -> usize {
        match self {
            Self::Linear | Self::Logistic | Self::Poisson => 0,
            Self::NegativeBinomial => 1,
        }
    }
}

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

    if let Ok(override_ptx) = std::env::var("NS_GLM_CUBLAS_AUX_PTX_OVERRIDE") {
        if !override_ptx.trim().is_empty() {
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

/// Batched GLM evaluator backed by cuBLAS.
pub struct CudaGlmCublasEvaluator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k_diff_nll: CudaFunction,
    k_add_prior: CudaFunction,
    d_x_col: CudaSlice<f64>,
    d_y: CudaSlice<f64>,
    d_offset: CudaSlice<f64>,
    d_params: CudaSlice<f64>,
    d_eta: CudaSlice<f64>,
    d_diff: CudaSlice<f64>,
    d_grad: CudaSlice<f64>,
    d_nll: CudaSlice<f64>,
    n: usize,
    beta_dim: usize,
    param_dim: usize,
    n_chains: usize,
    zeros_grad: Vec<f64>,
    zeros_nll: Vec<f64>,
    family: CudaGlmFamily,
    apply_standard_normal_prior: bool,
}

impl CudaGlmCublasEvaluator {
    /// Create evaluator for a specific GLM family on a specific GPU device.
    pub fn new_on_device_with_family(
        x_col: &[f64], // column-major [p * n]
        y: &[f64],     // [n]
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, None, n, beta_dim, n_chains, family, device_id, true)
    }

    /// Create evaluator for a specific GLM family with an optional observation offset.
    pub fn new_on_device_with_family_and_offset(
        x_col: &[f64],          // column-major [p * n]
        y: &[f64],              // [n]
        offset: Option<&[f64]>, // [n]
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, offset, n, beta_dim, n_chains, family, device_id, true)
    }

    /// Create evaluator for a specific GLM family on a specific GPU device without a prior term.
    pub fn new_on_device_with_family_no_prior(
        x_col: &[f64], // column-major [p * n]
        y: &[f64],     // [n]
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, None, n, beta_dim, n_chains, family, device_id, false)
    }

    /// Create evaluator for a specific GLM family with an optional observation offset and no prior.
    pub fn new_on_device_with_family_and_offset_no_prior(
        x_col: &[f64],          // column-major [p * n]
        y: &[f64],              // [n]
        offset: Option<&[f64]>, // [n]
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, offset, n, beta_dim, n_chains, family, device_id, false)
    }

    /// Create evaluator on a specific GPU device.
    pub fn new_on_device(
        x_col: &[f64], // column-major [p * n]
        y: &[f64],     // [n]
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(
            x_col,
            y,
            None,
            n,
            beta_dim,
            n_chains,
            CudaGlmFamily::Logistic,
            device_id,
            true,
        )
    }

    fn new_on_device_impl(
        x_col: &[f64], // column-major [p * n]
        y: &[f64],     // [n]
        offset: Option<&[f64]>,
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
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
        if y.len() != n {
            return Err(ns_core::Error::Validation(format!(
                "y length mismatch: expected {}, got {}",
                n,
                y.len()
            )));
        }
        if let Some(off) = offset {
            if off.len() != n {
                return Err(ns_core::Error::Validation(format!(
                    "offset length mismatch: expected {}, got {}",
                    n,
                    off.len()
                )));
            }
            if off.iter().any(|v| !v.is_finite()) {
                return Err(ns_core::Error::Validation(
                    "offset must contain only finite values".into(),
                ));
            }
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
            .load_function(family.kernel_name())
            .map_err(|e| cuda_err(format!("load {}: {e}", family.kernel_name())))?;
        let k_add_prior = module
            .load_function("glm_add_prior")
            .map_err(|e| cuda_err(format!("load glm_add_prior: {e}")))?;

        let d_x_col = stream.clone_htod(x_col).map_err(cuda_err)?;
        let d_y = stream.clone_htod(y).map_err(cuda_err)?;
        let host_offset = offset.map_or_else(|| vec![0.0; n], |off| off.to_vec());
        let d_offset = stream.clone_htod(&host_offset).map_err(cuda_err)?;
        let param_dim = beta_dim + family.extra_param_dim();
        let d_params = stream.alloc_zeros::<f64>(n_chains * param_dim).map_err(cuda_err)?;
        let d_eta = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_diff = stream.alloc_zeros::<f64>(n_chains * n).map_err(cuda_err)?;
        let d_grad = stream.alloc_zeros::<f64>(n_chains * param_dim).map_err(cuda_err)?;
        let d_nll = stream.alloc_zeros::<f64>(n_chains).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            blas,
            k_diff_nll,
            k_add_prior,
            d_x_col,
            d_y,
            d_offset,
            d_params,
            d_eta,
            d_diff,
            d_grad,
            d_nll,
            n,
            beta_dim,
            param_dim,
            n_chains,
            zeros_grad: vec![0.0; n_chains * param_dim],
            zeros_nll: vec![0.0; n_chains],
            family,
            apply_standard_normal_prior,
        })
    }

    /// Create evaluator for a specific GLM family on GPU 0.
    pub fn new_with_family(
        x_col: &[f64],
        y: &[f64],
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, None, n, beta_dim, n_chains, family, 0, true)
    }

    /// Create evaluator for a specific GLM family with an optional observation offset on GPU 0.
    pub fn new_with_family_and_offset(
        x_col: &[f64],
        y: &[f64],
        offset: Option<&[f64]>,
        n: usize,
        beta_dim: usize,
        n_chains: usize,
        family: CudaGlmFamily,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(x_col, y, offset, n, beta_dim, n_chains, family, 0, true)
    }

    /// Create evaluator on GPU 0.
    pub fn new(
        x_col: &[f64],
        y: &[f64],
        n: usize,
        beta_dim: usize,
        n_chains: usize,
    ) -> ns_core::Result<Self> {
        Self::new_on_device_impl(
            x_col,
            y,
            None,
            n,
            beta_dim,
            n_chains,
            CudaGlmFamily::Logistic,
            0,
            true,
        )
    }

    /// Evaluate batched GLM grad and NLL for packed per-chain parameters.
    ///
    /// Logistic and Poisson pack only the coefficient vector. Negative binomial
    /// packs `[beta..., log_alpha]` per chain.
    ///
    /// Returns `(grad_flat, nll)` with shapes `[n_chains * param_dim]` and
    /// `[n_chains]`.
    pub fn evaluate_host(&mut self, params: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params.len() != self.n_chains * self.param_dim {
            return Err(ns_core::Error::Validation(format!(
                "parameter length mismatch: expected {}, got {}",
                self.n_chains * self.param_dim,
                params.len()
            )));
        }

        self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_grad, &mut self.d_grad).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.zeros_nll, &mut self.d_nll).map_err(cuda_err)?;

        // eta = X @ beta for each chain (X shared across batch). Family-specific
        // trailing parameters are skipped by the batch stride.
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
                        stride_b: self.param_dim as i64,
                        stride_c: self.n as i64,
                    },
                    &self.d_x_col,
                    &self.d_params,
                    &mut self.d_eta,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X @ beta)", e))?;
        }

        // diff = family-specific residual, nll_data = family-specific data NLL,
        // and any trailing-parameter likelihood gradients are accumulated
        // directly into d_grad.
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
        builder.arg(&self.d_y);
        builder.arg(&self.d_offset);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_diff);
        builder.arg(&mut self.d_grad);
        builder.arg(&mut self.d_nll);
        builder.arg(&n_arg);
        builder.arg(&beta_dim_arg);
        builder.arg(&param_dim_arg);
        builder.arg(&n_chains_arg);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| cuda_err(format!("launch {}: {e}", self.family.kernel_name())))?;
        }

        // grad_data = X^T @ diff for each chain.
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
                        stride_c: self.param_dim as i64,
                    },
                    &self.d_x_col,
                    &self.d_diff,
                    &mut self.d_grad,
                )
                .map_err(|e| cublas_err("gemm_strided_batched(X^T @ diff)", e))?;
        }

        if self.apply_standard_normal_prior {
            // Add Gaussian prior: grad += params; nll += 0.5 * ||params||^2
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
                    .map_err(|e| cuda_err(format!("launch glm_add_prior: {e}")))?;
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
