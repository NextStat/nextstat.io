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

#[cfg(feature = "arrow-io")]
pub mod event_parquet;
pub mod event_store;
pub mod model;
pub mod pdf;
pub mod spec;

pub mod normalize;

pub(crate) mod fused_kernel;
mod interp;
mod math;

pub use event_store::{EventStore, ObservableSpec, WeightSummary};
pub use interp::HistoSysInterpCode;
pub use model::{
    Constraint, Parameter, Process, RateModifier, UnbinnedChannel, UnbinnedModel, YieldExpr,
};
pub use pdf::{
    ArgusPdf, ChebyshevPdf, CrystalBallPdf, DoubleCrystalBallPdf, ExponentialPdf, GaussianPdf,
    HistogramPdf, HistogramSystematic, HorizontalMorphingKdePdf, KdeHorizontalSystematic, KdeNdPdf,
    KdePdf, KdeWeightSystematic, MorphingHistogramPdf, MorphingKdePdf, ProductPdf, SplinePdf,
    UnbinnedPdf, VoigtianPdf,
};
#[cfg(feature = "neural")]
pub use pdf::{DcrSurrogate, FlowManifest, FlowPdf};
#[cfg(any(feature = "neural-cuda", feature = "neural-tensorrt"))]
pub use pdf::{FlowCudaLogProb, FlowCudaLogProbGrad, FlowGpuConfig, FlowGpuEpKind};

#[cfg(test)]
mod tests;
