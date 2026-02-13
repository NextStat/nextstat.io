//! Metal compute backend (Apple Silicon).
//!
//! This module is feature-gated behind `metal`.
//!
//! # Production GPU paths
//!
//! The real Metal acceleration lives in dedicated accelerator modules:
//!
//! - [`MetalBatchAccelerator`] — lockstep batch NLL+gradient for toy fitting
//!   (used by `ns_inference::metal_batch::fit_toys_batch_metal`).
//! - [`MetalDifferentiableAccelerator`] — profiled fitting for differentiable
//!   analysis (`ns_inference::metal_differentiable`).
//!
//! These types do **not** implement the generic [`ComputeBackend`] trait;
//! they expose fused kernel APIs tailored to their respective workloads.
//!
//! [`ComputeBackend`]: ns_core::ComputeBackend
//! [`MetalBatchAccelerator`]: crate::metal_batch::MetalBatchAccelerator
//! [`MetalDifferentiableAccelerator`]: crate::metal_differentiable::MetalDifferentiableAccelerator

// Re-export for discoverability
pub use crate::metal_batch::MetalBatchAccelerator;
pub use crate::metal_differentiable::MetalDifferentiableAccelerator;
pub use crate::metal_unbinned::MetalUnbinnedAccelerator;
