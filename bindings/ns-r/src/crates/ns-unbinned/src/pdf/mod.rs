//! Unbinned probability density functions (PDFs).

use crate::event_store::EventStore;
use ns_core::{Error, Result};
use rand::RngCore;

mod argus;
mod chebyshev;
mod crystal_ball;
mod exponential;
mod gaussian;
mod histogram;
mod horizontal_morphing_kde;
mod kde;
mod kde_nd;
mod morphing_histogram;
mod morphing_kde;
mod product;
mod spline;
mod voigtian;

#[cfg(feature = "neural")]
mod dcr;
#[cfg(feature = "neural")]
mod flow;
#[cfg(feature = "neural")]
pub mod flow_manifest;

pub use argus::ArgusPdf;
pub use chebyshev::ChebyshevPdf;
pub use crystal_ball::{CrystalBallPdf, DoubleCrystalBallPdf};
pub use exponential::ExponentialPdf;
pub use gaussian::GaussianPdf;
pub use histogram::HistogramPdf;
pub use horizontal_morphing_kde::{HorizontalMorphingKdePdf, KdeHorizontalSystematic};
pub use kde::KdePdf;
pub use kde_nd::KdeNdPdf;
pub use morphing_histogram::{HistogramSystematic, MorphingHistogramPdf};
pub use morphing_kde::{KdeWeightSystematic, MorphingKdePdf};
pub use product::ProductPdf;
pub use spline::SplinePdf;
pub use voigtian::VoigtianPdf;

#[cfg(feature = "neural")]
pub use dcr::DcrSurrogate;
#[cfg(feature = "neural")]
pub use flow::FlowPdf;
#[cfg(feature = "neural")]
pub use flow_manifest::FlowManifest;

/// Trait for normalized PDFs used in event-level likelihoods.
///
/// Implementations are expected to define a **proper density** on the observable support `Ω`
/// described by [`EventStore::bounds`].
pub trait UnbinnedPdf: Send + Sync {
    /// Number of shape parameters for this PDF.
    fn n_params(&self) -> usize;

    /// Observable names required by this PDF (stable order).
    ///
    /// Phase 1 PDFs are 1D and return a slice of length 1.
    fn observables(&self) -> &[String];

    /// Evaluate `log p(x | params)` for all events in the store.
    ///
    /// `out` must have length `events.n_events()`.
    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()>;

    /// Evaluate `log p(x|params)` and its gradient w.r.t the shape parameters.
    ///
    /// - `out_logp` must have length `events.n_events()`.
    /// - `out_grad` must have length `events.n_events() * self.n_params()` and is laid out as
    ///   row-major `[event0_param0, event0_param1, ..., event1_param0, ...]`.
    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()>;

    /// Internal tag for fused-kernel topology detection.
    ///
    /// Concrete PDFs that participate in fused evaluation paths override this
    /// to return a stable identifier (e.g. `"gaussian"`, `"exponential"`).
    /// Default returns `""` (no fused path available).
    fn pdf_tag(&self) -> &'static str {
        ""
    }

    /// Sample `n_events` from this PDF on the provided observable support `Ω`.
    ///
    /// `support` must be in the same order as [`Self::observables`]. For Phase 1 PDFs this has
    /// length 1: `[(low, high)]`.
    ///
    /// Default implementation returns [`Error::NotImplemented`].
    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> Result<EventStore> {
        let _ = (params, n_events, support, rng);
        Err(Error::NotImplemented("UnbinnedPdf::sample is not implemented for this PDF".into()))
    }
}
