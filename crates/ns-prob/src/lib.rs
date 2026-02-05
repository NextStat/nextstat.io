//! Probability building blocks for NextStat.
//!
//! This crate hosts reusable probability math used by multiple domains:
//! - base distributions (logpdf/cdf/etc.)
//! - transforms/bijectors (for constrained parameterizations)
//! - small numeric helpers (stable log/exp/sigmoid primitives)

pub mod math;
pub mod normal;
pub mod transforms;
