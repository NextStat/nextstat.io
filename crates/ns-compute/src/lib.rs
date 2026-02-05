//! # ns-compute
//!
//! Compute backends for NextStat.
//!
//! This crate provides implementations of the `ComputeBackend` trait:
//! - **CPU backend** (Rayon + SIMD) - Priority P0, always available
//! - **Metal backend** (macOS) - Priority P1, feature-gated
//! - **CUDA backend** (NVIDIA) - Priority P1, feature-gated
//!
//! ## Architecture
//!
//! High-level inference code (ns-inference) depends on the `ComputeBackend`
//! trait from ns-core, NOT on concrete implementations. This allows:
//! - Testing with simple CPU backend
//! - Optional GPU acceleration
//! - Easy addition of new backends (TPU, etc.)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod cpu;
pub mod simd;

pub use cpu::CpuBackend;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
