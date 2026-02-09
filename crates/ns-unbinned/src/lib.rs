//! # ns-unbinned
//!
//! Event-level (unbinned) likelihood models for NextStat.
//!
//! This crate provides:
//! - A columnar [`EventStore`] (SoA layout) for observable vectors.
//! - Parametric unbinned PDFs (Phase 1) and a framework to add more.
//! - An extended unbinned mixture model with yields + constraints, implementing
//!   [`ns_core::traits::LogDensityModel`], so it can be optimized via `ns-inference`.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod event_store;
pub mod model;
pub mod pdf;

mod math;

pub use event_store::{EventStore, ObservableSpec};
pub use model::{
    Constraint, Parameter, Process, RateModifier, UnbinnedChannel, UnbinnedModel, YieldExpr,
};
pub use pdf::{
    ChebyshevPdf, CrystalBallPdf, DoubleCrystalBallPdf, ExponentialPdf, GaussianPdf, HistogramPdf,
    KdePdf, UnbinnedPdf,
};

#[cfg(test)]
mod tests;
