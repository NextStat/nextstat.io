//! # ns-viz
//!
//! Visualization data artifacts for NextStat.
//!
//! This crate is intentionally dependency-light and focuses on emitting
//! plot-friendly JSON structures (arrays instead of nested objects).

#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

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

/// Gammas (staterror / Barlow-Beeston) artifacts.
pub mod gammas;

/// Correlation matrix artifacts (TREx-style).
pub mod corr;

/// Yields tables artifacts (TREx-style).
pub mod yields;

/// Separation plot artifacts (S vs B shape comparison).
pub mod separation;

/// Summary plot artifacts (multi-fit μ comparison).
pub mod summary;

/// Pie chart artifacts (sample composition per channel).
pub mod pie;

/// Uncertainty breakdown artifacts (TREx-style).
pub mod uncertainty;

/// Significance scan artifacts (p₀ / Z vs mass/parameter).
pub mod significance;

/// 2D likelihood contour artifacts (two-POI profile likelihood).
pub mod contour;

/// Unfolding plot artifacts (response matrix + unfolded spectrum).
pub mod unfolding;

/// Morphing validation artifacts (template interpolation visualisation).
pub mod morphing;

/// Signal injection / linearity test artifacts.
pub mod injection;

pub use cls::{ClsCurveArtifact, ClsCurvePoint, NsSigmaOrder};
pub use contour::{ContourArtifact, ContourGridPoint, ContourLine};
pub use corr::CorrArtifact;
pub use distributions::{
    BandEnvelope, DistributionsArtifact, DistributionsChannelArtifact, RatioPolicy,
};
pub use gammas::GammasArtifact;
pub use injection::{InjectionArtifact, InjectionPoint};
pub use morphing::{MorphingArtifact, MorphingTemplate};
pub use pie::PieArtifact;
pub use profile::{ProfileCurveArtifact, ProfileCurvePoint};
pub use pulls::{PullEntry, PullsArtifact};
pub use ranking::RankingArtifact;
pub use separation::SeparationArtifact;
pub use significance::{SignificanceScanArtifact, SignificanceScanPoint};
pub use summary::SummaryArtifact;
pub use uncertainty::UncertaintyBreakdownArtifact;
pub use unfolding::{ResponseMatrixArtifact, UnfoldedBin, UnfoldedSpectrumArtifact};
pub use yields::YieldsArtifact;
