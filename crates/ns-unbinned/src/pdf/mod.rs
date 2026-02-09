//! Unbinned probability density functions (PDFs).

use crate::event_store::EventStore;
use ns_core::Result;

mod chebyshev;
mod crystal_ball;
mod exponential;
mod gaussian;
mod histogram;
mod kde;

pub use chebyshev::ChebyshevPdf;
pub use crystal_ball::{CrystalBallPdf, DoubleCrystalBallPdf};
pub use exponential::ExponentialPdf;
pub use gaussian::GaussianPdf;
pub use histogram::HistogramPdf;
pub use kde::KdePdf;

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
}
