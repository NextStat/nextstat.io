use std::sync::Arc;

use crate::event_store::EventStore;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Product of independent PDFs: `p(x₁, x₂, …) = p₁(x₁) × p₂(x₂) × …`
///
/// Each component PDF reads its own observable columns from the shared [`EventStore`].
/// Shape parameters are packed contiguously: `[comp0_params…, comp1_params…, …]`.
///
/// `log p = Σᵢ log pᵢ(xᵢ | θᵢ)`
pub struct ProductPdf {
    components: Vec<Arc<dyn UnbinnedPdf>>,
    /// Flattened observable names (deduplicated, stable order).
    all_observables: Vec<String>,
    /// Total number of shape parameters across all components.
    total_params: usize,
    /// Cumulative param offsets: component `i` owns params `[offsets[i]..offsets[i+1])`.
    param_offsets: Vec<usize>,
}

impl ProductPdf {
    /// Build a product PDF from a list of independent component PDFs.
    ///
    /// Observable names across components must be disjoint (otherwise the "independent"
    /// assumption is violated and the user should use a multi-D flow instead).
    pub fn new(components: Vec<Arc<dyn UnbinnedPdf>>) -> Result<Self> {
        if components.is_empty() {
            return Err(Error::Validation("ProductPdf requires at least one component".into()));
        }

        let mut all_observables = Vec::new();
        let mut param_offsets = Vec::with_capacity(components.len() + 1);
        let mut total_params = 0usize;

        for (i, comp) in components.iter().enumerate() {
            let obs = comp.observables();
            for name in obs {
                if all_observables.contains(name) {
                    return Err(Error::Validation(format!(
                        "ProductPdf: observable '{}' appears in component {} but was already \
                         claimed by an earlier component. ProductPdf requires disjoint observables.",
                        name, i
                    )));
                }
                all_observables.push(name.clone());
            }
            param_offsets.push(total_params);
            total_params += comp.n_params();
        }
        param_offsets.push(total_params);

        Ok(Self { components, all_observables, total_params, param_offsets })
    }

    /// Number of component PDFs.
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Access a component by index.
    pub fn component(&self, idx: usize) -> Option<&dyn UnbinnedPdf> {
        self.components.get(idx).map(|c| c.as_ref())
    }
}

impl UnbinnedPdf for ProductPdf {
    fn n_params(&self) -> usize {
        self.total_params
    }

    fn observables(&self) -> &[String] {
        &self.all_observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if params.len() != self.total_params {
            return Err(Error::Validation(format!(
                "ProductPdf expects {} params, got {}",
                self.total_params,
                params.len()
            )));
        }
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "ProductPdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        // Initialize output to zero; accumulate log-probs from each component.
        for v in out.iter_mut() {
            *v = 0.0;
        }

        let mut tmp = vec![0.0f64; n];

        for (i, comp) in self.components.iter().enumerate() {
            let p_start = self.param_offsets[i];
            let p_end = self.param_offsets[i + 1];
            let comp_params = &params[p_start..p_end];

            comp.log_prob_batch(events, comp_params, &mut tmp)?;

            for j in 0..n {
                out[j] += tmp[j];
            }
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
        if params.len() != self.total_params {
            return Err(Error::Validation(format!(
                "ProductPdf expects {} params, got {}",
                self.total_params,
                params.len()
            )));
        }
        let n = events.n_events();
        if out_logp.len() != n {
            return Err(Error::Validation(format!(
                "ProductPdf out_logp length mismatch: expected {n}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n * self.total_params;
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "ProductPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
                out_grad.len()
            )));
        }

        // Initialize accumulators.
        for v in out_logp.iter_mut() {
            *v = 0.0;
        }
        for v in out_grad.iter_mut() {
            *v = 0.0;
        }

        for (i, comp) in self.components.iter().enumerate() {
            let p_start = self.param_offsets[i];
            let p_end = self.param_offsets[i + 1];
            let comp_n_params = p_end - p_start;
            let comp_params = &params[p_start..p_end];

            let mut tmp_logp = vec![0.0f64; n];
            let mut tmp_grad = vec![0.0f64; n * comp_n_params];

            comp.log_prob_grad_batch(events, comp_params, &mut tmp_logp, &mut tmp_grad)?;

            // Accumulate log-prob.
            for j in 0..n {
                out_logp[j] += tmp_logp[j];
            }

            // Scatter component gradients into the correct columns of the full gradient.
            // out_grad layout: [event0_param0, event0_param1, …, event1_param0, …]
            // Component i owns global param indices [p_start..p_end).
            for j in 0..n {
                for k in 0..comp_n_params {
                    out_grad[j * self.total_params + p_start + k] = tmp_grad[j * comp_n_params + k];
                }
            }
        }

        Ok(())
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if params.len() != self.total_params {
            return Err(Error::Validation(format!(
                "ProductPdf expects {} params, got {}",
                self.total_params,
                params.len()
            )));
        }
        if support.len() != self.all_observables.len() {
            return Err(Error::Validation(format!(
                "ProductPdf sample expects {}D support, got {}D",
                self.all_observables.len(),
                support.len()
            )));
        }

        // Sample each component independently, then merge columns.
        let mut all_columns: Vec<(String, Vec<f64>)> = Vec::new();
        let mut obs_offset = 0usize;

        for (i, comp) in self.components.iter().enumerate() {
            let p_start = self.param_offsets[i];
            let p_end = self.param_offsets[i + 1];
            let comp_params = &params[p_start..p_end];

            let comp_n_obs = comp.observables().len();
            let comp_support = &support[obs_offset..obs_offset + comp_n_obs];

            let comp_events = comp.sample(comp_params, n_events, comp_support, rng)?;

            for obs_name in comp.observables() {
                let col = comp_events.column(obs_name).ok_or_else(|| {
                    Error::Computation(format!(
                        "ProductPdf: component {} sample() did not produce column '{}'",
                        i, obs_name
                    ))
                })?;
                all_columns.push((obs_name.clone(), col.to_vec()));
            }

            obs_offset += comp_n_obs;
        }

        // Build the merged EventStore.
        let obs_specs: Vec<crate::event_store::ObservableSpec> = self
            .all_observables
            .iter()
            .zip(support)
            .map(|(name, &bounds)| crate::event_store::ObservableSpec::branch(name.clone(), bounds))
            .collect();

        EventStore::from_columns(obs_specs, all_columns, None)
    }
}
