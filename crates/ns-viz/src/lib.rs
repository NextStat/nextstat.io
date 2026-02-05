//! # ns-viz
//!
//! Visualization data artifacts for NextStat.
//!
//! This crate is intentionally dependency-light and focuses on emitting
//! plot-friendly JSON structures (arrays instead of nested objects).

#![warn(missing_docs)]
#![warn(clippy::all)]

/// CLs curve artifacts (observed + Brazil bands).
pub mod cls;

/// Profile likelihood artifacts (q_mu curves).
pub mod profile;

pub use cls::{ClsCurveArtifact, ClsCurvePoint, NsSigmaOrder};
pub use profile::{ProfileCurveArtifact, ProfileCurvePoint};
