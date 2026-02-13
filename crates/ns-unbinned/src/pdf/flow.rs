//! ONNX-backed normalizing flow PDF.
//!
//! Feature-gated behind `neural`. Loads a pre-trained normalizing flow from ONNX
//! files described by a [`FlowManifest`] and implements [`UnbinnedPdf`] for use in
//! unbinned likelihood fits.
//!
//! # Architecture
//!
//! A normalizing flow defines a bijection `f: z → x` with `z ~ base_dist`.
//! Two ONNX models are exported:
//!
//! - **`log_prob`**: `(x [, c]) → log p(x | c)` — hot path, called per optimizer step.
//! - **`sample`** (optional): `(z [, c]) → x` — cold path, for toy generation.
//!
//! Conditional flows accept a context vector `c` (e.g. nuisance parameters).
//!
//! # Normalization
//!
//! Well-trained flows produce a properly normalized density by construction
//! (the change-of-variables formula guarantees `∫ p(x) dx = 1`). However, we
//! optionally verify normalization via Gauss-Legendre quadrature and apply a
//! correction if needed.

#[cfg(feature = "neural")]
use std::path::{Path, PathBuf};
#[cfg(feature = "neural")]
use std::sync::Mutex;

#[cfg(feature = "neural")]
use ns_core::{Error, Result};
#[cfg(feature = "neural")]
use ort::session::Session;
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
use ort::{
    ep,
    io_binding::IoBinding,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    value::{Tensor, TensorValueType},
};
#[cfg(feature = "neural")]
use rand::RngCore;

#[cfg(feature = "neural")]
use crate::event_store::{EventStore, ObservableSpec};
#[cfg(feature = "neural")]
use crate::normalize::{NormalizationCache, QuadratureGrid};
#[cfg(feature = "neural")]
use crate::pdf::UnbinnedPdf;
#[cfg(feature = "neural")]
use crate::pdf::flow_manifest::FlowManifest;

/// Which GPU execution provider is active for the flow session.
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowGpuEpKind {
    /// ONNX Runtime CUDA Execution Provider.
    CudaEp,
    /// ONNX Runtime TensorRT Execution Provider.
    TensorRtEp,
}

/// Runtime configuration for GPU execution providers.
///
/// Passed to [`FlowPdf::from_manifest_with_config`] to control TensorRT engine
/// caching, FP16 inference, and dynamic batch-size profiles.
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
#[derive(Debug, Clone)]
pub struct FlowGpuConfig {
    /// Enable TensorRT FP16 inference (default: `true`).
    /// When TRT EP is unavailable, this setting is ignored.
    pub fp16: bool,
    /// Path to the TensorRT engine cache directory.
    /// When `Some`, compiled TRT engines are cached to disk so subsequent
    /// loads skip the expensive compilation step.
    pub engine_cache_path: Option<String>,
    /// TensorRT optimization profile: minimum batch size (default: 1).
    pub profile_min_batch: usize,
    /// TensorRT optimization profile: optimal batch size (default: 1024).
    pub profile_opt_batch: usize,
    /// TensorRT optimization profile: maximum batch size (default: 65536).
    pub profile_max_batch: usize,
}

#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
impl Default for FlowGpuConfig {
    fn default() -> Self {
        Self {
            fp16: true,
            engine_cache_path: default_trt_cache_path(),
            profile_min_batch: 1,
            profile_opt_batch: 1024,
            profile_max_batch: 65536,
        }
    }
}

#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
fn default_trt_cache_path() -> Option<String> {
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))?;
    let dir = format!("{base}/nextstat/tensorrt");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
struct FlowGpuLogProbState {
    session: Session,
    binding: IoBinding,
    input_name: String,
    output_name: String,
    ep_kind: FlowGpuEpKind,
}

/// A CUDA device-resident log-prob buffer produced by `FlowPdf` via CUDA EP I/O binding.
///
/// Holds the underlying ORT tensor alive; use [`FlowCudaLogProb::device_ptr_u64`] to pass it
/// into CUDA kernels.
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
pub struct FlowCudaLogProb {
    tensor: Tensor<f32>,
}

#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
impl FlowCudaLogProb {
    /// CUDA device pointer (as u64) to the underlying `float` buffer.
    pub fn device_ptr_u64(&self) -> u64 {
        self.tensor.data_ptr() as u64
    }

    /// Number of elements (events) in the output tensor.
    pub fn len(&self) -> usize {
        self.tensor.shape().iter().map(|&d| d as usize).product()
    }
}

/// GPU EP state for the analytical Jacobian (`log_prob_grad`) ONNX model.
///
/// Similar to [`FlowGpuLogProbState`] but binds **two** outputs to CUDA device memory:
/// - output 0: `log_prob [batch]`
/// - output 1: `d_log_prob_d_context [batch, n_context]`
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
struct FlowGpuLogProbGradState {
    session: Session,
    binding: IoBinding,
    input_name: String,
    output_name_logp: String,
    output_name_jac: String,
}

/// CUDA device-resident log-prob + Jacobian buffers produced by the `log_prob_grad` model
/// via CUDA EP I/O binding.
///
/// Use [`device_ptr_logp_u64`](FlowCudaLogProbGrad::device_ptr_logp_u64) and
/// [`device_ptr_jac_u64`](FlowCudaLogProbGrad::device_ptr_jac_u64) to pass raw pointers
/// into CUDA kernels.
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
pub struct FlowCudaLogProbGrad {
    tensor_logp: Tensor<f32>,
    tensor_jac: Tensor<f32>,
}

#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
impl FlowCudaLogProbGrad {
    /// CUDA device pointer (as u64) to `float[batch]` log-prob output.
    pub fn device_ptr_logp_u64(&self) -> u64 {
        self.tensor_logp.data_ptr() as u64
    }

    /// CUDA device pointer (as u64) to `float[batch, n_context]` Jacobian output.
    pub fn device_ptr_jac_u64(&self) -> u64 {
        self.tensor_jac.data_ptr() as u64
    }

    /// Number of events (batch size).
    pub fn n_events(&self) -> usize {
        self.tensor_logp.shape().iter().map(|&d| d as usize).product()
    }

    /// Number of context parameters (from Jacobian shape[1]).
    pub fn n_context(&self) -> usize {
        let shape = self.tensor_jac.shape();
        if shape.len() >= 2 { shape[1] as usize } else { 0 }
    }
}

/// An ONNX-backed normalizing flow PDF.
///
/// Implements [`UnbinnedPdf`] so it can be used as a process PDF in unbinned
/// likelihood fits alongside parametric PDFs.
///
/// # Conditional flows
///
/// When `n_context > 0`, the flow expects context features appended to its input.
/// The `context_param_indices` field maps global model parameter indices to the
/// flow's context vector. During `log_prob_batch`, the context values are read from
/// `params[context_param_indices[i]]` and broadcast across all events.
#[cfg(feature = "neural")]
pub struct FlowPdf {
    /// ONNX session for `log_prob(x [, c]) -> [batch]`.
    /// Wrapped in `Mutex` because `ort::Session::run` requires `&mut self`
    /// while `UnbinnedPdf` trait methods take `&self`.
    session_logprob: Mutex<Session>,
    /// ONNX session for `sample(z [, c]) -> [batch, features]`. Optional.
    session_sample: Option<Mutex<Session>>,
    /// Observable names (length = features).
    observable_names: Vec<String>,
    /// Number of context features (0 for unconditional).
    n_context: usize,
    /// Maps model parameter indices → flow context vector positions.
    /// Length = `n_context`. Entry `i` means `context[i] = params[context_param_indices[i]]`.
    context_param_indices: Vec<usize>,
    /// Per-observable support bounds.
    support: Vec<(f64, f64)>,
    /// Parameter-keyed normalization cache with auto-selected quadrature grid.
    ///
    /// Covers all dimensionalities: 1-3D (Gauss-Legendre), 4-5D (low-order tensor
    /// product), 6D+ (Halton quasi-Monte Carlo). The cache avoids recomputing the
    /// integral when parameters haven't changed (rounded to 6 significant digits).
    norm_cache: Option<NormalizationCache>,
    /// Cached log-normalization constant (recomputed when params change).
    /// For well-trained flows this should be ≈ 0.
    log_norm_correction: f64,
    /// Base directory for resolving relative ONNX paths.
    _base_dir: PathBuf,

    /// Optional ONNX session for the analytical Jacobian model.
    ///
    /// When present, `log_prob_grad_batch` uses a single forward pass through this model
    /// instead of `2 × n_context` finite-difference evaluations.
    /// Outputs: `(log_prob [batch], d_log_prob_d_context [batch, n_context])`.
    session_logprob_grad: Option<Mutex<Session>>,

    /// Optional GPU EP log_prob runner with output bound to CUDA device memory.
    ///
    /// Constructed when the crate is built with `neural-cuda` or `neural-tensorrt`
    /// and the corresponding EP is available at runtime.
    /// TensorRT EP is preferred when both are available.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    gpu_logprob: Option<Mutex<FlowGpuLogProbState>>,

    /// Optional GPU EP log_prob_grad runner with **two** outputs bound to CUDA device memory:
    /// `(log_prob [batch], d_log_prob_d_context [batch, n_context])`.
    ///
    /// When present, `log_prob_grad_batch_cuda` evaluates the analytical Jacobian model
    /// on the GPU in a single forward pass, keeping both outputs device-resident.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    gpu_logprob_grad: Option<Mutex<FlowGpuLogProbGradState>>,
}

#[cfg(feature = "neural")]
impl FlowPdf {
    /// Load a flow PDF from a manifest file.
    ///
    /// `context_param_indices` maps each context feature to a global parameter index.
    /// For unconditional flows, pass an empty slice.
    pub fn from_manifest(
        manifest_path: &Path,
        context_param_indices: &[usize],
    ) -> anyhow::Result<Self> {
        let manifest = FlowManifest::from_path(manifest_path)?;
        let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();

        if context_param_indices.len() != manifest.context_features {
            anyhow::bail!(
                "FlowPdf: context_param_indices length {} != manifest.context_features {}",
                context_param_indices.len(),
                manifest.context_features
            );
        }

        let logprob_path = base_dir.join(&manifest.models.log_prob);
        let session_logprob = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&logprob_path))
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to load ONNX log_prob model {}: {e}",
                    logprob_path.display()
                )
            })?;

        let session_sample = match &manifest.models.sample {
            Some(sample_file) => {
                let sample_path = base_dir.join(sample_file);
                let s = Session::builder()
                    .and_then(|b| b.with_intra_threads(1))
                    .and_then(|b| b.commit_from_file(&sample_path))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load ONNX sample model {}: {e}",
                            sample_path.display()
                        )
                    })?;
                Some(Mutex::new(s))
            }
            None => None,
        };

        let session_logprob_grad = match &manifest.models.log_prob_grad {
            Some(grad_file) => {
                let grad_path = base_dir.join(grad_file);
                let s = Session::builder()
                    .and_then(|b| b.with_intra_threads(1))
                    .and_then(|b| b.commit_from_file(&grad_path))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load ONNX log_prob_grad model {}: {e}",
                            grad_path.display()
                        )
                    })?;
                // Validate: must have exactly 2 outputs.
                if s.outputs().len() != 2 {
                    anyhow::bail!(
                        "log_prob_grad ONNX model must have exactly 2 outputs (log_prob, d_log_prob_d_context), got {}",
                        s.outputs().len()
                    );
                }
                Some(Mutex::new(s))
            }
            None => None,
        };

        let support: Vec<(f64, f64)> = manifest.support.iter().map(|b| (b[0], b[1])).collect();

        // Build normalization cache with auto-selected quadrature grid (all dimensions).
        let norm_cache = {
            let grid = QuadratureGrid::auto(&manifest.observable_names, &support)?;
            Some(NormalizationCache::with_default_precision(grid))
        };

        #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
        let gpu_logprob = {
            let n_input_cols = manifest.features + manifest.context_features;
            Self::try_load_gpu_ep(&logprob_path, &FlowGpuConfig::default(), n_input_cols)
        };

        #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
        let gpu_logprob_grad = manifest.models.log_prob_grad.as_ref().and_then(|grad_file| {
            let grad_path = base_dir.join(grad_file);
            Self::try_load_cuda_ep_grad(&grad_path)
        });

        Ok(Self {
            session_logprob: Mutex::new(session_logprob),
            session_sample,
            observable_names: manifest.observable_names,
            n_context: manifest.context_features,
            context_param_indices: context_param_indices.to_vec(),
            support,
            norm_cache,
            log_norm_correction: 0.0,
            _base_dir: base_dir,
            session_logprob_grad,
            #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
            gpu_logprob,
            #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
            gpu_logprob_grad,
        })
    }

    /// Load a flow PDF from a manifest file with custom GPU EP configuration.
    ///
    /// Same as [`from_manifest`](Self::from_manifest) but allows configuring TensorRT
    /// engine caching, FP16 inference, and dynamic batch-size profiles.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    pub fn from_manifest_with_config(
        manifest_path: &Path,
        context_param_indices: &[usize],
        gpu_config: &FlowGpuConfig,
    ) -> anyhow::Result<Self> {
        let manifest = FlowManifest::from_path(manifest_path)?;
        let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();

        if context_param_indices.len() != manifest.context_features {
            anyhow::bail!(
                "FlowPdf: context_param_indices length {} != manifest.context_features {}",
                context_param_indices.len(),
                manifest.context_features
            );
        }

        let logprob_path = base_dir.join(&manifest.models.log_prob);
        let session_logprob = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&logprob_path))
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to load ONNX log_prob model {}: {e}",
                    logprob_path.display()
                )
            })?;

        let session_sample = match &manifest.models.sample {
            Some(sample_file) => {
                let sample_path = base_dir.join(sample_file);
                let s = Session::builder()
                    .and_then(|b| b.with_intra_threads(1))
                    .and_then(|b| b.commit_from_file(&sample_path))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load ONNX sample model {}: {e}",
                            sample_path.display()
                        )
                    })?;
                Some(Mutex::new(s))
            }
            None => None,
        };

        let session_logprob_grad = match &manifest.models.log_prob_grad {
            Some(grad_file) => {
                let grad_path = base_dir.join(grad_file);
                let s = Session::builder()
                    .and_then(|b| b.with_intra_threads(1))
                    .and_then(|b| b.commit_from_file(&grad_path))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to load ONNX log_prob_grad model {}: {e}",
                            grad_path.display()
                        )
                    })?;
                if s.outputs().len() != 2 {
                    anyhow::bail!(
                        "log_prob_grad ONNX model must have exactly 2 outputs (log_prob, d_log_prob_d_context), got {}",
                        s.outputs().len()
                    );
                }
                Some(Mutex::new(s))
            }
            None => None,
        };

        let support: Vec<(f64, f64)> = manifest.support.iter().map(|b| (b[0], b[1])).collect();

        let norm_cache = {
            let grid = QuadratureGrid::auto(&manifest.observable_names, &support)?;
            Some(NormalizationCache::with_default_precision(grid))
        };

        let n_input_cols = manifest.features + manifest.context_features;
        let gpu_logprob = Self::try_load_gpu_ep(&logprob_path, gpu_config, n_input_cols);

        let gpu_logprob_grad = manifest.models.log_prob_grad.as_ref().and_then(|grad_file| {
            let grad_path = base_dir.join(grad_file);
            Self::try_load_cuda_ep_grad(&grad_path)
        });

        Ok(Self {
            session_logprob: Mutex::new(session_logprob),
            session_sample,
            observable_names: manifest.observable_names,
            n_context: manifest.context_features,
            context_param_indices: context_param_indices.to_vec(),
            support,
            norm_cache,
            log_norm_correction: 0.0,
            _base_dir: base_dir,
            session_logprob_grad,
            gpu_logprob,
            gpu_logprob_grad,
        })
    }

    /// Try loading GPU execution providers in priority order: TensorRT → CUDA → None.
    ///
    /// Returns `Some(Mutex<FlowGpuLogProbState>)` on success, `None` on failure.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    fn try_load_gpu_ep(
        logprob_path: &Path,
        config: &FlowGpuConfig,
        n_input_cols: usize,
    ) -> Option<Mutex<FlowGpuLogProbState>> {
        // Try TensorRT EP first (if feature enabled).
        #[cfg(feature = "neural-tensorrt")]
        {
            if let Some(state) = Self::try_load_tensorrt_ep(logprob_path, config, n_input_cols) {
                return Some(Mutex::new(state));
            }
        }

        // Fallback to CUDA EP.
        #[cfg(feature = "neural-cuda")]
        {
            if let Some(state) = Self::try_load_cuda_ep(logprob_path) {
                return Some(Mutex::new(state));
            }
        }

        let _ = (config, n_input_cols);
        None
    }

    /// Try loading CUDA Execution Provider.
    #[cfg(feature = "neural-cuda")]
    fn try_load_cuda_ep(logprob_path: &Path) -> Option<FlowGpuLogProbState> {
        let result = (|| -> anyhow::Result<FlowGpuLogProbState> {
            let session = Session::builder()
                .and_then(|b| b.with_execution_providers([ep::CUDA::default().build()]))
                .and_then(|b| b.with_intra_threads(1))
                .and_then(|b| b.commit_from_file(logprob_path))
                .map_err(|e| anyhow::anyhow!("CUDA EP: {e}"))?;

            let input_name = session
                .inputs()
                .first()
                .ok_or_else(|| anyhow::anyhow!("ONNX log_prob model has no inputs"))?
                .name()
                .to_string();
            let output_name = session
                .outputs()
                .first()
                .ok_or_else(|| anyhow::anyhow!("ONNX log_prob model has no outputs"))?
                .name()
                .to_string();

            let mem_info = MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?;

            let mut binding = session.create_binding()?;
            binding.bind_output_to_device(output_name.clone(), &mem_info)?;

            Ok(FlowGpuLogProbState {
                session,
                binding,
                input_name,
                output_name,
                ep_kind: FlowGpuEpKind::CudaEp,
            })
        })();

        result.ok()
    }

    /// Try loading TensorRT Execution Provider with engine caching and FP16.
    #[cfg(feature = "neural-tensorrt")]
    fn try_load_tensorrt_ep(
        logprob_path: &Path,
        config: &FlowGpuConfig,
        n_input_cols: usize,
    ) -> Option<FlowGpuLogProbState> {
        let result = (|| -> anyhow::Result<FlowGpuLogProbState> {
            let mut trt = ep::TensorRT::default().with_fp16(config.fp16);

            if let Some(ref cache_path) = config.engine_cache_path {
                trt = trt.with_engine_cache(true).with_engine_cache_path(cache_path.as_str());
            }

            // Set dynamic batch profile shapes for TensorRT.
            // Flow models have input shape [batch, n_input_cols].
            // We need the ONNX input tensor name — load a lightweight CPU session to inspect.
            if config.profile_max_batch > 1 && n_input_cols > 0 {
                let temp_session = Session::builder()
                    .and_then(|b| b.with_intra_threads(1))
                    .and_then(|b| b.commit_from_file(logprob_path))
                    .map_err(|e| anyhow::anyhow!("TRT profile: failed to inspect model: {e}"))?;

                let input_name = temp_session
                    .inputs()
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("TRT profile: model has no inputs"))?
                    .name()
                    .to_string();

                let d1 = n_input_cols;
                let min_shape = format!("{input_name}:{}x{d1}", config.profile_min_batch);
                let opt_shape = format!("{input_name}:{}x{d1}", config.profile_opt_batch);
                let max_shape = format!("{input_name}:{}x{d1}", config.profile_max_batch);

                trt = trt
                    .with_profile_min_shapes(min_shape)
                    .with_profile_opt_shapes(opt_shape)
                    .with_profile_max_shapes(max_shape);

                drop(temp_session);
            }

            let session = Session::builder()
                .and_then(|b| b.with_execution_providers([trt.build()]))
                .and_then(|b| b.with_intra_threads(1))
                .and_then(|b| b.commit_from_file(logprob_path))
                .map_err(|e| anyhow::anyhow!("TensorRT EP: {e}"))?;

            let input_name = session
                .inputs()
                .first()
                .ok_or_else(|| anyhow::anyhow!("ONNX log_prob model has no inputs"))?
                .name()
                .to_string();
            let output_name = session
                .outputs()
                .first()
                .ok_or_else(|| anyhow::anyhow!("ONNX log_prob model has no outputs"))?
                .name()
                .to_string();

            let mem_info = MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?;

            let mut binding = session.create_binding()?;
            binding.bind_output_to_device(output_name.clone(), &mem_info)?;

            Ok(FlowGpuLogProbState {
                session,
                binding,
                input_name,
                output_name,
                ep_kind: FlowGpuEpKind::TensorRtEp,
            })
        })();

        result.ok()
    }

    /// Try loading the analytical Jacobian ONNX model with CUDA EP.
    ///
    /// The model must have exactly 2 outputs: `log_prob [batch]` and
    /// `d_log_prob_d_context [batch, n_context]`. Both are bound to CUDA device memory.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    fn try_load_cuda_ep_grad(grad_path: &Path) -> Option<Mutex<FlowGpuLogProbGradState>> {
        let result = (|| -> anyhow::Result<FlowGpuLogProbGradState> {
            #[cfg(feature = "neural-cuda")]
            let session = Session::builder()
                .and_then(|b| b.with_execution_providers([ep::CUDA::default().build()]))
                .and_then(|b| b.with_intra_threads(1))
                .and_then(|b| b.commit_from_file(grad_path))
                .map_err(|e| anyhow::anyhow!("CUDA EP (grad): {e}"))?;

            #[cfg(not(feature = "neural-cuda"))]
            anyhow::bail!("CUDA EP not available (feature neural-cuda not enabled)");

            if session.outputs().len() != 2 {
                anyhow::bail!(
                    "log_prob_grad ONNX model must have exactly 2 outputs, got {}",
                    session.outputs().len()
                );
            }

            let input_name = session
                .inputs()
                .first()
                .ok_or_else(|| anyhow::anyhow!("ONNX log_prob_grad model has no inputs"))?
                .name()
                .to_string();
            let output_name_logp = session.outputs()[0].name().to_string();
            let output_name_jac = session.outputs()[1].name().to_string();

            let mem_info = MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?;

            let mut binding = session.create_binding()?;
            binding.bind_output_to_device(output_name_logp.clone(), &mem_info)?;
            binding.bind_output_to_device(output_name_jac.clone(), &mem_info)?;

            Ok(FlowGpuLogProbGradState {
                session,
                binding,
                input_name,
                output_name_logp,
                output_name_jac,
            })
        })();

        result.ok().map(Mutex::new)
    }

    /// Whether the analytical Jacobian model is available on GPU (CUDA EP).
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    pub fn has_gpu_grad(&self) -> bool {
        self.gpu_logprob_grad.is_some()
    }

    /// Which GPU execution provider is active, if any.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    pub fn gpu_ep_kind(&self) -> Option<FlowGpuEpKind> {
        self.gpu_logprob
            .as_ref()
            .map(|m| m.lock().map(|st| st.ep_kind).unwrap_or(FlowGpuEpKind::CudaEp))
    }

    /// Recompute the normalization correction for the given parameters.
    ///
    /// For well-trained flows this returns ≈ 0. If the integral deviates from 1.0
    /// by more than 0.1%, the correction is applied to `log_prob_batch`.
    pub fn update_normalization(&mut self, params: &[f64]) -> Result<()> {
        let shape_params = self.extract_context_params(params);
        if let Some(mut cache) = self.norm_cache.take() {
            let log_norm = cache.log_norm(self, &shape_params)?;
            self.log_norm_correction = log_norm;
            self.norm_cache = Some(cache);
        }
        Ok(())
    }

    /// Current log-normalization correction (≈ 0 for well-trained flows).
    pub fn log_norm_correction(&self) -> f64 {
        self.log_norm_correction
    }

    /// Extract context parameters from the global parameter vector.
    fn extract_context_params(&self, params: &[f64]) -> Vec<f64> {
        self.context_param_indices.iter().map(|&i| params[i]).collect()
    }

    /// Run the ONNX log_prob session.
    ///
    /// `x` is `[batch, features]` as a flat row-major array.
    /// `context` is the context vector (broadcast across batch).
    /// Returns `log_prob` as `[batch]`.
    fn run_logprob(&self, x_flat: &[f32], batch_size: usize, context: &[f64]) -> Result<Vec<f64>> {
        let features = self.observable_names.len();

        let input_tensor = if self.n_context == 0 {
            let shape = vec![batch_size, features];
            ort::value::Tensor::from_array((shape, x_flat.to_vec()))
                .map_err(|e| Error::Validation(format!("ONNX input tensor error: {e}")))?
        } else {
            let total_cols = features + self.n_context;
            let mut combined = Vec::with_capacity(batch_size * total_cols);
            let ctx_f32: Vec<f32> = context.iter().map(|&v| v as f32).collect();
            for i in 0..batch_size {
                let row_start = i * features;
                combined.extend_from_slice(&x_flat[row_start..row_start + features]);
                combined.extend_from_slice(&ctx_f32);
            }
            let shape = vec![batch_size, total_cols];
            ort::value::Tensor::from_array((shape, combined))
                .map_err(|e| Error::Validation(format!("ONNX input tensor error: {e}")))?
        };

        let mut session = self
            .session_logprob
            .lock()
            .map_err(|e| Error::Validation(format!("ONNX session lock poisoned: {e}")))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| Error::Validation(format!("ONNX log_prob inference error: {e}")))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Validation(format!("ONNX output extraction error: {e}")))?;

        let logp: Vec<f64> = data.iter().map(|&v| v as f64).collect();

        if logp.len() != batch_size {
            return Err(Error::Validation(format!(
                "ONNX log_prob output length {} != batch_size {}",
                logp.len(),
                batch_size
            )));
        }

        Ok(logp)
    }

    /// Run the analytical Jacobian ONNX model in a single forward pass.
    ///
    /// Returns `(logp [batch], d_logp_d_context [batch × n_context])` where the Jacobian
    /// is laid out row-major: `[event0_ctx0, event0_ctx1, ..., event1_ctx0, ...]`.
    ///
    /// Requires the `log_prob_grad` model to be present in the manifest.
    fn run_logprob_with_grad(
        &self,
        x_flat: &[f32],
        batch_size: usize,
        context: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let session_mutex = self.session_logprob_grad.as_ref().ok_or_else(|| {
            Error::NotImplemented(
                "FlowPdf::run_logprob_with_grad requires a log_prob_grad ONNX model in the manifest".into(),
            )
        })?;

        let features = self.observable_names.len();
        let nc = self.n_context;

        let input_tensor = if nc == 0 {
            let shape = vec![batch_size, features];
            ort::value::Tensor::from_array((shape, x_flat.to_vec()))
                .map_err(|e| Error::Validation(format!("ONNX grad input tensor error: {e}")))?
        } else {
            let total_cols = features + nc;
            let mut combined = Vec::with_capacity(batch_size * total_cols);
            let ctx_f32: Vec<f32> = context.iter().map(|&v| v as f32).collect();
            for i in 0..batch_size {
                let row_start = i * features;
                combined.extend_from_slice(&x_flat[row_start..row_start + features]);
                combined.extend_from_slice(&ctx_f32);
            }
            let shape = vec![batch_size, total_cols];
            ort::value::Tensor::from_array((shape, combined))
                .map_err(|e| Error::Validation(format!("ONNX grad input tensor error: {e}")))?
        };

        let mut session = session_mutex
            .lock()
            .map_err(|e| Error::Validation(format!("ONNX grad session lock poisoned: {e}")))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| Error::Validation(format!("ONNX log_prob_grad inference error: {e}")))?;

        // Output 0: log_prob [batch]
        let (_shape0, data0) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Validation(format!("ONNX grad output 0 extraction error: {e}")))?;
        let logp: Vec<f64> = data0.iter().map(|&v| v as f64).collect();
        if logp.len() != batch_size {
            return Err(Error::Validation(format!(
                "ONNX log_prob_grad output 0 length {} != batch_size {}",
                logp.len(),
                batch_size
            )));
        }

        // Output 1: d_log_prob_d_context [batch, n_context]
        let (_shape1, data1) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Validation(format!("ONNX grad output 1 extraction error: {e}")))?;
        let jacobian: Vec<f64> = data1.iter().map(|&v| v as f64).collect();
        let expected_jac_len = batch_size * nc;
        if jacobian.len() != expected_jac_len {
            return Err(Error::Validation(format!(
                "ONNX log_prob_grad output 1 length {} != expected {} (batch={} × n_context={})",
                jacobian.len(),
                expected_jac_len,
                batch_size,
                nc
            )));
        }

        Ok((logp, jacobian))
    }

    /// Whether the analytical Jacobian model is available.
    pub fn has_analytical_grad(&self) -> bool {
        self.session_logprob_grad.is_some()
    }

    /// Run the ONNX `log_prob` model on CUDA EP and keep the output tensor device-resident.
    ///
    /// The normalization correction is **not applied** here. If needed, incorporate it
    /// as a yield scaling per-process: `nu' = nu * exp(-log_norm_correction)`.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    pub fn log_prob_batch_cuda(
        &self,
        events: &EventStore,
        params: &[f64],
    ) -> Result<FlowCudaLogProb> {
        let n = events.n_events();
        if n == 0 {
            return Err(Error::Validation(
                "FlowPdf::log_prob_batch_cuda: n_events must be > 0".into(),
            ));
        }

        let state = self.gpu_logprob.as_ref().ok_or_else(|| {
            Error::NotImplemented(
                "FlowPdf GPU EP path not available (build with --features neural-cuda or neural-tensorrt and ensure EP loads)"
                    .into(),
            )
        })?;

        let features = self.observable_names.len();

        // Build flat [batch, features] input in f32.
        let mut x_flat = Vec::with_capacity(n * features);
        for obs_name in &self.observable_names {
            let col = events.column(obs_name).ok_or_else(|| {
                Error::Validation(format!("FlowPdf: EventStore missing column '{obs_name}'"))
            })?;
            x_flat.extend(col.iter().map(|&v| v as f32));
        }

        // Convert to row-major [event0_obs0, event0_obs1, ...].
        let mut x_row_major = vec![0.0f32; n * features];
        for f in 0..features {
            for i in 0..n {
                x_row_major[i * features + f] = x_flat[f * n + i];
            }
        }

        let context = self.extract_context_params(params);

        // Build input tensor (CPU). ORT handles staging/copies to CUDA EP.
        let input_tensor = if self.n_context == 0 {
            let shape = vec![n, features];
            Tensor::from_array((shape, x_row_major))
                .map_err(|e| Error::Validation(format!("ONNX input tensor error: {e}")))?
        } else {
            let total_cols = features + self.n_context;
            let mut combined = Vec::with_capacity(n * total_cols);
            let ctx_f32: Vec<f32> = context.iter().map(|&v| v as f32).collect();
            for i in 0..n {
                let row_start = i * features;
                combined.extend_from_slice(&x_row_major[row_start..row_start + features]);
                combined.extend_from_slice(&ctx_f32);
            }
            let shape = vec![n, total_cols];
            Tensor::from_array((shape, combined))
                .map_err(|e| Error::Validation(format!("ONNX input tensor error: {e}")))?
        };

        let mut st = state
            .lock()
            .map_err(|e| Error::Validation(format!("ONNX GPU session lock poisoned: {e}")))?;

        let FlowGpuLogProbState { session, binding, input_name, output_name, ep_kind: _ } =
            &mut *st;

        // Clone names first to avoid borrow conflicts with `SessionOutputs<'_>` lifetimes.
        let input_name = input_name.clone();
        let output_name = output_name.clone();

        binding
            .bind_input(input_name, &input_tensor)
            .map_err(|e| Error::Validation(format!("ONNX bind_input error: {e}")))?;

        let mut outputs = session
            .run_binding(binding)
            .map_err(|e| Error::Validation(format!("ONNX CUDA inference error: {e}")))?;

        // Ensure CUDA EP finished writing outputs before any external CUDA kernel reads them.
        binding
            .synchronize_outputs()
            .map_err(|e| Error::Validation(format!("ONNX synchronize_outputs error: {e}")))?;

        let out_val = outputs.remove(output_name.as_str()).ok_or_else(|| {
            Error::Validation(format!("ONNX CUDA binding missing output '{output_name}'",))
        })?;

        let tensor = out_val
            .downcast::<TensorValueType<f32>>()
            .map_err(|e| Error::Validation(format!("ONNX output downcast error: {e}")))?;

        if tensor.memory_info().allocation_device() != AllocationDevice::CUDA {
            return Err(Error::Validation(format!(
                "ONNX CUDA output is not on CUDA device (got {:?})",
                tensor.memory_info().allocation_device()
            )));
        }

        let out_len: usize = tensor.shape().iter().map(|&d| d as usize).product();
        if out_len != n {
            return Err(Error::Validation(format!(
                "ONNX CUDA log_prob output length {} != n_events {}",
                out_len, n
            )));
        }

        Ok(FlowCudaLogProb { tensor })
    }

    /// Run the ONNX `log_prob_grad` model on CUDA EP and keep both outputs device-resident.
    ///
    /// Returns a [`FlowCudaLogProbGrad`] holding two device-resident `float` tensors:
    /// - `log_prob [batch]`
    /// - `d_log_prob_d_context [batch, n_context]`
    ///
    /// This is the analytical gradient counterpart to [`log_prob_batch_cuda`]: a single
    /// forward pass replaces `2 × n_context` finite-difference evaluations.
    #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
    pub fn log_prob_grad_batch_cuda(
        &self,
        events: &EventStore,
        params: &[f64],
    ) -> Result<FlowCudaLogProbGrad> {
        let n = events.n_events();
        if n == 0 {
            return Err(Error::Validation(
                "FlowPdf::log_prob_grad_batch_cuda: n_events must be > 0".into(),
            ));
        }

        let state = self.gpu_logprob_grad.as_ref().ok_or_else(|| {
            Error::NotImplemented(
                "FlowPdf GPU EP grad path not available (no log_prob_grad model in manifest or CUDA EP failed to load)"
                    .into(),
            )
        })?;

        let features = self.observable_names.len();

        // Build flat [batch, features] input in f32.
        let mut x_flat = Vec::with_capacity(n * features);
        for obs_name in &self.observable_names {
            let col = events.column(obs_name).ok_or_else(|| {
                Error::Validation(format!("FlowPdf: EventStore missing column '{obs_name}'"))
            })?;
            x_flat.extend(col.iter().map(|&v| v as f32));
        }

        // Convert to row-major [event0_obs0, event0_obs1, ...].
        let mut x_row_major = vec![0.0f32; n * features];
        for f in 0..features {
            for i in 0..n {
                x_row_major[i * features + f] = x_flat[f * n + i];
            }
        }

        let context = self.extract_context_params(params);

        // Build input tensor (CPU). ORT handles staging/copies to CUDA EP.
        let input_tensor = if self.n_context == 0 {
            let shape = vec![n, features];
            Tensor::from_array((shape, x_row_major))
                .map_err(|e| Error::Validation(format!("ONNX grad input tensor error: {e}")))?
        } else {
            let total_cols = features + self.n_context;
            let mut combined = Vec::with_capacity(n * total_cols);
            let ctx_f32: Vec<f32> = context.iter().map(|&v| v as f32).collect();
            for i in 0..n {
                let row_start = i * features;
                combined.extend_from_slice(&x_row_major[row_start..row_start + features]);
                combined.extend_from_slice(&ctx_f32);
            }
            let shape = vec![n, total_cols];
            Tensor::from_array((shape, combined))
                .map_err(|e| Error::Validation(format!("ONNX grad input tensor error: {e}")))?
        };

        let mut st = state
            .lock()
            .map_err(|e| Error::Validation(format!("ONNX GPU grad session lock poisoned: {e}")))?;

        let FlowGpuLogProbGradState {
            session,
            binding,
            input_name,
            output_name_logp,
            output_name_jac,
        } = &mut *st;

        let input_name = input_name.clone();
        let output_name_logp = output_name_logp.clone();
        let output_name_jac = output_name_jac.clone();

        binding
            .bind_input(input_name, &input_tensor)
            .map_err(|e| Error::Validation(format!("ONNX grad bind_input error: {e}")))?;

        let mut outputs = session
            .run_binding(binding)
            .map_err(|e| Error::Validation(format!("ONNX CUDA grad inference error: {e}")))?;

        binding
            .synchronize_outputs()
            .map_err(|e| Error::Validation(format!("ONNX grad synchronize_outputs error: {e}")))?;

        // Extract log_prob tensor (output 0).
        let logp_val = outputs.remove(output_name_logp.as_str()).ok_or_else(|| {
            Error::Validation(format!("ONNX CUDA grad binding missing output '{output_name_logp}'"))
        })?;
        let tensor_logp = logp_val
            .downcast::<TensorValueType<f32>>()
            .map_err(|e| Error::Validation(format!("ONNX grad logp downcast error: {e}")))?;

        if tensor_logp.memory_info().allocation_device() != AllocationDevice::CUDA {
            return Err(Error::Validation(format!(
                "ONNX CUDA grad logp output is not on CUDA device (got {:?})",
                tensor_logp.memory_info().allocation_device()
            )));
        }

        let logp_len: usize = tensor_logp.shape().iter().map(|&d| d as usize).product();
        if logp_len != n {
            return Err(Error::Validation(format!(
                "ONNX CUDA grad logp output length {} != n_events {}",
                logp_len, n
            )));
        }

        // Extract Jacobian tensor (output 1).
        let jac_val = outputs.remove(output_name_jac.as_str()).ok_or_else(|| {
            Error::Validation(format!("ONNX CUDA grad binding missing output '{output_name_jac}'"))
        })?;
        let tensor_jac = jac_val
            .downcast::<TensorValueType<f32>>()
            .map_err(|e| Error::Validation(format!("ONNX grad jac downcast error: {e}")))?;

        if tensor_jac.memory_info().allocation_device() != AllocationDevice::CUDA {
            return Err(Error::Validation(format!(
                "ONNX CUDA grad jac output is not on CUDA device (got {:?})",
                tensor_jac.memory_info().allocation_device()
            )));
        }

        let jac_shape = tensor_jac.shape();
        let expected_jac_len = n * self.n_context;
        let jac_len: usize = jac_shape.iter().map(|&d| d as usize).product();
        if jac_len != expected_jac_len {
            return Err(Error::Validation(format!(
                "ONNX CUDA grad jac output length {} != expected {} (n_events={} × n_context={})",
                jac_len, expected_jac_len, n, self.n_context
            )));
        }

        Ok(FlowCudaLogProbGrad { tensor_logp, tensor_jac })
    }

    /// Run the ONNX sample (inverse) session.
    ///
    /// `z` is `[batch, features]` from the base distribution (standard normal).
    /// Returns sampled `x` as `[batch, features]`.
    fn run_sample(&self, z_flat: &[f32], batch_size: usize, context: &[f64]) -> Result<Vec<f64>> {
        let session_mutex = self.session_sample.as_ref().ok_or_else(|| {
            Error::NotImplemented(
                "FlowPdf::sample requires a sample ONNX model in the manifest".into(),
            )
        })?;

        let features = self.observable_names.len();

        let input_tensor = if self.n_context == 0 {
            let shape = vec![batch_size, features];
            ort::value::Tensor::from_array((shape, z_flat.to_vec()))
                .map_err(|e| Error::Validation(format!("ONNX sample input tensor error: {e}")))?
        } else {
            let total_cols = features + self.n_context;
            let mut combined = Vec::with_capacity(batch_size * total_cols);
            let ctx_f32: Vec<f32> = context.iter().map(|&v| v as f32).collect();
            for i in 0..batch_size {
                let row_start = i * features;
                combined.extend_from_slice(&z_flat[row_start..row_start + features]);
                combined.extend_from_slice(&ctx_f32);
            }
            let shape = vec![batch_size, total_cols];
            ort::value::Tensor::from_array((shape, combined))
                .map_err(|e| Error::Validation(format!("ONNX sample input tensor error: {e}")))?
        };

        let mut session = session_mutex
            .lock()
            .map_err(|e| Error::Validation(format!("ONNX sample session lock poisoned: {e}")))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| Error::Validation(format!("ONNX sample inference error: {e}")))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Validation(format!("ONNX sample output extraction error: {e}")))?;

        let x: Vec<f64> = data.iter().map(|&v| v as f64).collect();

        if x.len() != batch_size * features {
            return Err(Error::Validation(format!(
                "ONNX sample output length {} != expected {}",
                x.len(),
                batch_size * features
            )));
        }

        Ok(x)
    }
}

#[cfg(feature = "neural")]
impl UnbinnedPdf for FlowPdf {
    fn n_params(&self) -> usize {
        self.n_context
    }

    fn observables(&self) -> &[String] {
        &self.observable_names
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "FlowPdf::log_prob_batch: out length {} != n_events {}",
                out.len(),
                n
            )));
        }
        if n == 0 {
            return Ok(());
        }

        let features = self.observable_names.len();

        // Build flat [batch, features] input in f32.
        let mut x_flat = Vec::with_capacity(n * features);
        for obs_name in &self.observable_names {
            let col = events.column(obs_name).ok_or_else(|| {
                Error::Validation(format!("FlowPdf: EventStore missing column '{obs_name}'"))
            })?;
            // We'll transpose below: first collect column-major, then rearrange.
            x_flat.extend(col.iter().map(|&v| v as f32));
        }

        // x_flat is currently column-major [obs0_event0..obs0_eventN, obs1_event0..].
        // ONNX expects row-major [event0_obs0, event0_obs1, ..., event1_obs0, ...].
        let mut x_row_major = vec![0.0f32; n * features];
        for f in 0..features {
            for i in 0..n {
                x_row_major[i * features + f] = x_flat[f * n + i];
            }
        }

        let context = self.extract_context_params(params);
        let logp = self.run_logprob(&x_row_major, n, &context)?;

        // Apply normalization correction.
        let correction = self.log_norm_correction;
        for (o, &lp) in out.iter_mut().zip(&logp) {
            *o = lp - correction;
        }

        Ok(())
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        let n = events.n_events();
        let np = self.n_context;

        if np == 0 {
            self.log_prob_batch(events, params, out_logp)?;
            for g in out_grad.iter_mut() {
                *g = 0.0;
            }
            return Ok(());
        }

        let features = self.observable_names.len();

        // Build row-major [batch, features] input (shared by both paths).
        let mut x_flat = Vec::with_capacity(n * features);
        for obs_name in &self.observable_names {
            let col = events.column(obs_name).ok_or_else(|| {
                Error::Validation(format!("FlowPdf: EventStore missing column '{obs_name}'"))
            })?;
            x_flat.extend(col.iter().map(|&v| v as f32));
        }

        let mut x_row_major = vec![0.0f32; n * features];
        for f in 0..features {
            for i in 0..n {
                x_row_major[i * features + f] = x_flat[f * n + i];
            }
        }

        let context = self.extract_context_params(params);

        // Prefer analytical Jacobian (1 forward pass) over finite differences (2×n_context).
        if self.session_logprob_grad.is_some() {
            let (logp, jacobian) = self.run_logprob_with_grad(&x_row_major, n, &context)?;

            let correction = self.log_norm_correction;
            for (o, &lp) in out_logp.iter_mut().zip(&logp) {
                *o = lp - correction;
            }

            // jacobian is already [batch × n_context] row-major, matching out_grad layout.
            out_grad.copy_from_slice(&jacobian);

            return Ok(());
        }

        // Fallback: central finite differences (2 × n_context forward passes).
        self.log_prob_batch(events, params, out_logp)?;

        let eps = 1e-4;
        for p_idx in 0..np {
            let mut ctx_plus = context.clone();
            let mut ctx_minus = context.clone();
            ctx_plus[p_idx] += eps;
            ctx_minus[p_idx] -= eps;

            let lp_plus = self.run_logprob(&x_row_major, n, &ctx_plus)?;
            let lp_minus = self.run_logprob(&x_row_major, n, &ctx_minus)?;

            for i in 0..n {
                out_grad[i * np + p_idx] = (lp_plus[i] - lp_minus[i]) / (2.0 * eps);
            }
        }

        Ok(())
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> Result<EventStore> {
        if self.session_sample.is_none() {
            return Err(Error::NotImplemented(
                "FlowPdf::sample requires a sample ONNX model in the manifest".into(),
            ));
        }

        let features = self.observable_names.len();
        if support.len() != features {
            return Err(Error::Validation(format!(
                "FlowPdf::sample: support length {} != features {}",
                support.len(),
                features
            )));
        }

        // Sample z ~ N(0, 1) for the base distribution.
        use rand_distr::{Distribution, StandardNormal};
        let mut z_flat = Vec::with_capacity(n_events * features);
        for _ in 0..(n_events * features) {
            let z: f64 = StandardNormal.sample(rng);
            z_flat.push(z as f32);
        }

        let context = self.extract_context_params(params);
        let x_flat = self.run_sample(&z_flat, n_events, &context)?;

        // Build EventStore from sampled data.
        // x_flat is row-major [event0_obs0, event0_obs1, ..., event1_obs0, ...].
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n_events); features];
        for i in 0..n_events {
            for f in 0..features {
                let val = x_flat[i * features + f];
                // Clamp to support bounds.
                let (lo, hi) = support[f];
                let clamped = val.clamp(lo, hi);
                columns[f].push(clamped);
            }
        }

        let obs: Vec<ObservableSpec> = self
            .observable_names
            .iter()
            .zip(support)
            .map(|(name, &(lo, hi))| ObservableSpec::branch(name.clone(), (lo, hi)))
            .collect();

        let col_pairs: Vec<(String, Vec<f64>)> = self
            .observable_names
            .iter()
            .zip(columns)
            .map(|(name, col)| (name.clone(), col))
            .collect();

        EventStore::from_columns(obs, col_pairs, None)
    }
}

#[cfg(feature = "neural")]
impl std::fmt::Debug for FlowPdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowPdf")
            .field("observable_names", &self.observable_names)
            .field("n_context", &self.n_context)
            .field("context_param_indices", &self.context_param_indices)
            .field("support", &self.support)
            .field("has_sample_model", &self.session_sample.is_some())
            .field("has_grad_model", &self.session_logprob_grad.is_some())
            .field("gpu_ep", &{
                #[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
                {
                    self.gpu_ep_kind()
                }
                #[cfg(not(any(feature = "neural-cuda", feature = "neural-tensorrt")))]
                {
                    Option::<&str>::None
                }
            })
            .field("n_features", &self.observable_names.len())
            .field("log_norm_correction", &self.log_norm_correction)
            .finish()
    }
}
