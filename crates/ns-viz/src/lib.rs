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

/// Nuisance-parameter ranking artifacts (impact on POI).
pub mod ranking;

/// Stacked distributions artifacts (prefit/postfit, ratio).
pub mod distributions;

/// Pulls + constraints artifacts (TREx-style).
pub mod pulls;

/// Correlation matrix artifacts (TREx-style).
pub mod corr;

/// Yields tables artifacts (TREx-style).
pub mod yields;

pub use cls::{ClsCurveArtifact, ClsCurvePoint, NsSigmaOrder};
pub use profile::{ProfileCurveArtifact, ProfileCurvePoint};
pub use ranking::RankingArtifact;
pub use distributions::{DistributionsArtifact, DistributionsChannelArtifact, RatioPolicy};
pub use pulls::{PullsArtifact, PullEntry};
pub use corr::CorrArtifact;
pub use yields::YieldsArtifact;
