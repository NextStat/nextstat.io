use crate::event_store::EventStore;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Piecewise-constant histogram PDF normalized on the bin edges.
///
/// This is a pragmatic, interpretable baseline for modeling unknown shapes from MC
/// without introducing ML. The density is constant within each bin:
///
/// `p(x) = p_i / (x_{i+1} - x_i)` for `x ∈ [x_i, x_{i+1})`
///
/// where `p_i` is the probability mass of bin `i` derived from `bin_content`.
#[derive(Debug, Clone)]
pub struct HistogramPdf {
    observables: [String; 1],
    bin_edges: Vec<f64>,
    log_density: Vec<f64>,
}

impl HistogramPdf {
    /// Construct a histogram PDF from edges and non-negative bin contents.
    ///
    /// `pseudo_count` is added to every bin content before normalization. Use it to avoid
    /// hard zeros when the histogram is built from finite MC samples.
    pub fn from_edges_and_contents(
        observable: impl Into<String>,
        bin_edges: Vec<f64>,
        bin_content: Vec<f64>,
        pseudo_count: f64,
    ) -> Result<Self> {
        if bin_edges.len() < 2 {
            return Err(Error::Validation(format!(
                "HistogramPdf requires at least 2 bin edges, got {}",
                bin_edges.len()
            )));
        }
        if bin_content.len() + 1 != bin_edges.len() {
            return Err(Error::Validation(format!(
                "HistogramPdf bin_content length mismatch: expected {}, got {}",
                bin_edges.len() - 1,
                bin_content.len()
            )));
        }
        if !pseudo_count.is_finite() || pseudo_count < 0.0 {
            return Err(Error::Validation(format!(
                "HistogramPdf pseudo_count must be finite and >=0, got {pseudo_count}"
            )));
        }

        for (i, w) in bin_content.iter().enumerate() {
            if !w.is_finite() || *w < 0.0 {
                return Err(Error::Validation(format!(
                    "HistogramPdf bin_content[{i}] must be finite and >=0, got {w}"
                )));
            }
        }
        for i in 0..bin_edges.len() {
            let e = bin_edges[i];
            if !e.is_finite() {
                return Err(Error::Validation(format!(
                    "HistogramPdf bin_edges[{i}] must be finite, got {e}"
                )));
            }
            if i > 0 && bin_edges[i - 1] >= e {
                return Err(Error::Validation(format!(
                    "HistogramPdf bin edges must be strictly increasing, got edges[{}]={} and edges[{}]={}",
                    i - 1,
                    bin_edges[i - 1],
                    i,
                    e
                )));
            }
        }

        let mut total = 0.0f64;
        for &w in &bin_content {
            total += w + pseudo_count;
        }
        if !(total.is_finite() && total > 0.0) {
            return Err(Error::Validation(format!(
                "HistogramPdf total content must be finite and >0 after pseudo_count, got {total}"
            )));
        }
        let log_total = total.ln();

        let mut log_density = Vec::with_capacity(bin_content.len());
        for i in 0..bin_content.len() {
            let w = bin_content[i] + pseudo_count;
            let width = bin_edges[i + 1] - bin_edges[i];
            if !(width.is_finite() && width > 0.0) {
                return Err(Error::Validation(format!(
                    "HistogramPdf bin width must be finite and >0, got width={width} for bin {i}"
                )));
            }

            if w > 0.0 {
                log_density.push(w.ln() - log_total - width.ln());
            } else {
                log_density.push(f64::NEG_INFINITY);
            }
        }

        Ok(Self { observables: [observable.into()], bin_edges, log_density })
    }

    fn bin_index(&self, x: f64) -> Result<usize> {
        let x_min = self.bin_edges[0];
        let x_max = *self.bin_edges.last().unwrap_or(&x_min);
        if !(x.is_finite()) {
            return Err(Error::Validation("HistogramPdf requires finite x".into()));
        }
        if x < x_min || x > x_max {
            return Err(Error::Validation(format!(
                "HistogramPdf x out of range: x={x} not in [{x_min}, {x_max}]"
            )));
        }

        let n_bins = self.log_density.len();
        if n_bins == 0 {
            return Err(Error::Validation("HistogramPdf has 0 bins".into()));
        }

        if x >= x_max {
            return Ok(n_bins - 1);
        }

        // `k` is the number of edges <= x, so bin index is k-1.
        let k = self.bin_edges.partition_point(|e| *e <= x);
        if k == 0 {
            return Err(Error::Validation(format!(
                "HistogramPdf x below first edge unexpectedly: x={x}, x_min={x_min}"
            )));
        }
        let idx = k - 1;
        if idx >= n_bins {
            return Err(Error::Validation(format!(
                "HistogramPdf bin lookup failed for x={x}: idx={idx} >= n_bins={n_bins}"
            )));
        }
        Ok(idx)
    }
}

impl UnbinnedPdf for HistogramPdf {
    fn n_params(&self) -> usize {
        0
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, _params: &[f64], out: &mut [f64]) -> Result<()> {
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "HistogramPdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;

        let x_min = self.bin_edges[0];
        let x_max = *self.bin_edges.last().unwrap_or(&x_min);
        // Require that the histogram support matches the event support, so the PDF is properly
        // normalized on Ω.
        let eps = 1e-12;
        if (a - x_min).abs() > eps || (b - x_max).abs() > eps {
            return Err(Error::Validation(format!(
                "HistogramPdf support mismatch: pdf=[{x_min}, {x_max}], events=({a}, {b})"
            )));
        }

        for (i, &x) in xs.iter().enumerate() {
            let idx = self.bin_index(x)?;
            out[i] = self.log_density[idx];
        }

        Ok(())
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "HistogramPdf expects 0 params, got {}",
                params.len()
            )));
        }
        if !out_grad.is_empty() {
            return Err(Error::Validation(format!(
                "HistogramPdf out_grad must be empty (n_params=0), got len={}",
                out_grad.len()
            )));
        }
        self.log_prob_batch(events, params, out_logp)
    }
}
