//! CUDA compute backend (NVIDIA).
//!
//! This module is feature-gated behind `cuda`.
//!
//! # Production GPU paths
//!
//! The real CUDA acceleration lives in dedicated accelerator modules:
//!
//! - [`CudaBatchAccelerator`] — lockstep batch NLL+gradient for toy fitting
//!   (used by `ns_inference::gpu_batch::fit_toys_batch_gpu`).
//! - [`DifferentiableAccelerator`] — zero-copy PyTorch integration for
//!   differentiable analysis (`ns_inference::differentiable`).
//!
//! These types do **not** implement the generic [`ComputeBackend`] trait;
//! they expose fused kernel APIs tailored to their respective workloads.
//!
//! [`ComputeBackend`]: ns_core::ComputeBackend
//! [`CudaBatchAccelerator`]: crate::cuda_batch::CudaBatchAccelerator
//! [`DifferentiableAccelerator`]: crate::differentiable::DifferentiableAccelerator

// Re-export for discoverability
pub use crate::cuda_batch::CudaBatchAccelerator;
pub use crate::differentiable::DifferentiableAccelerator;
