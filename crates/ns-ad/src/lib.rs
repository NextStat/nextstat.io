//! # ns-ad
//!
//! Automatic differentiation (AD) primitives for NextStat.
//!
//! Provides:
//! - **Forward-mode AD** via [`dual::Dual`] numbers (efficient for few parameters)
//! - **Reverse-mode AD** via computation tape (planned, efficient for many parameters)
//! - [`Scalar`] trait for writing generic code over `f64` and `Dual`

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod dual;
pub mod scalar;
pub mod tape;
