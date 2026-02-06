//! Probability building blocks for NextStat.
//!
//! This crate hosts reusable probability math used by multiple domains:
//! - base distributions (logpdf/cdf/etc.)
//! - transforms/bijectors (for constrained parameterizations)
//! - small numeric helpers (stable log/exp/sigmoid primitives)

pub mod math;
pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod distributions;
pub mod exponential;
pub mod gamma;
pub mod neg_binomial;
pub mod normal;
pub mod poisson;
pub mod student_t;
pub mod transforms;
pub mod weibull;

// Additional distributions are intentionally added incrementally (Phase 5).
