use crate::event_store::{EventStore, ObservableSpec};
use crate::math::standard_normal_logpdf;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Multi-dimensional Gaussian kernel density estimator (2-D and 3-D).
///
/// Uses a diagonal bandwidth matrix with per-dimension bandwidths computed via
/// Silverman's rule of thumb: `h_d = σ_d · (4 / ((d+2) · N))^(1/(d+4))`.
///
/// The KDE is evaluated on a bounded support with truncated Gaussian kernels
/// (same approach as the 1-D [`super::KdePdf`], extended to multiple dimensions).
///
/// This PDF has **no shape parameters** (n_params = 0).
pub struct KdeNdPdf {
    observable_names: Vec<String>,
    n_dim: usize,
    /// Per-kernel center coordinates: `centers[k][d]` = coordinate d of kernel k.
    centers: Vec<Vec<f64>>,
    /// Per-dimension bandwidth (diagonal bandwidth matrix).
    bandwidths: Vec<f64>,
    /// Per-dimension inverse bandwidth (precomputed).
    inv_bandwidths: Vec<f64>,
    /// Per-dimension support bounds.
    support: Vec<(f64, f64)>,
    /// Per-kernel log weight (after normalization by truncation factors).
    /// `kernel_log_prefactor[k]` = `ln(w_k) - Σ_d ln(h_d) - Σ_d ln(Z_{k,d})`
    kernel_log_prefactor: Vec<f64>,
    /// `ln(Σ w_k)`.
    log_sum_w: f64,
    /// Number of kernels.
    n_kernels: usize,
}

impl KdeNdPdf {
    /// Construct a multi-D KDE from sample points.
    ///
    /// - `observable_names`: names of the D observables (2 or 3).
    /// - `support`: per-dimension `(low, high)` bounds.
    /// - `centers`: `N × D` matrix of sample coordinates (row-major: `centers[k][d]`).
    /// - `weights`: optional per-sample non-negative weights (length N).
    /// - `bandwidths`: optional per-dimension bandwidths. If `None`, Silverman's rule is used.
    pub fn from_samples(
        observable_names: Vec<String>,
        support: Vec<(f64, f64)>,
        centers: Vec<Vec<f64>>,
        weights: Option<Vec<f64>>,
        bandwidths: Option<Vec<f64>>,
    ) -> Result<Self> {
        let n_dim = observable_names.len();
        if !(2..=3).contains(&n_dim) {
            return Err(Error::Validation(format!("KdeNdPdf supports 2-D or 3-D, got {n_dim}-D")));
        }
        if support.len() != n_dim {
            return Err(Error::Validation(format!(
                "KdeNdPdf: support length ({}) != n_dim ({n_dim})",
                support.len()
            )));
        }
        for (d, &(lo, hi)) in support.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                return Err(Error::Validation(format!(
                    "KdeNdPdf: invalid support for dim {d}: ({lo}, {hi})"
                )));
            }
        }
        let n_kernels = centers.len();
        if n_kernels == 0 {
            return Err(Error::Validation("KdeNdPdf requires at least one sample".into()));
        }
        for (k, row) in centers.iter().enumerate() {
            if row.len() != n_dim {
                return Err(Error::Validation(format!(
                    "KdeNdPdf: center[{k}] has {} dimensions, expected {n_dim}",
                    row.len()
                )));
            }
            for (d, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(Error::Validation(format!(
                        "KdeNdPdf: center[{k}][{d}] is not finite"
                    )));
                }
            }
        }

        if let Some(w) = &weights {
            if w.len() != n_kernels {
                return Err(Error::Validation(format!(
                    "KdeNdPdf: weights length ({}) != number of samples ({n_kernels})",
                    w.len()
                )));
            }
            if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(Error::Validation("KdeNdPdf: weights must be finite and >= 0".into()));
            }
        }

        // Compute bandwidths (Silverman's rule or user-provided).
        let bw = match bandwidths {
            Some(bw) => {
                if bw.len() != n_dim {
                    return Err(Error::Validation(format!(
                        "KdeNdPdf: bandwidths length ({}) != n_dim ({n_dim})",
                        bw.len()
                    )));
                }
                for (d, &h) in bw.iter().enumerate() {
                    if !h.is_finite() || h <= 0.0 {
                        return Err(Error::Validation(format!(
                            "KdeNdPdf: bandwidth[{d}] must be finite and > 0, got {h}"
                        )));
                    }
                }
                bw
            }
            None => silverman_bandwidths(&centers, weights.as_deref(), n_dim),
        };

        let inv_bw: Vec<f64> = bw.iter().map(|&h| 1.0 / h).collect();
        let log_h_sum: f64 = bw.iter().map(|h| h.ln()).sum();

        // Precompute per-kernel log prefactor.
        let cdf = crate::math::standard_normal_cdf;

        let mut kernel_log_prefactor = Vec::with_capacity(n_kernels);
        let mut sum_w = 0.0f64;

        for k in 0..n_kernels {
            let w = weights.as_ref().map(|v| v[k]).unwrap_or(1.0);
            sum_w += w;
            let log_w = if w > 0.0 { w.ln() } else { f64::NEG_INFINITY };

            // Product of per-dimension truncation factors.
            let mut log_trunc = 0.0f64;
            for d in 0..n_dim {
                let z_lo = (support[d].0 - centers[k][d]) * inv_bw[d];
                let z_hi = (support[d].1 - centers[k][d]) * inv_bw[d];
                let mut zd = cdf(z_hi) - cdf(z_lo);
                if !zd.is_finite() || zd <= 0.0 {
                    zd = f64::MIN_POSITIVE;
                }
                log_trunc += zd.ln();
            }

            kernel_log_prefactor.push(log_w - log_h_sum - log_trunc);
        }

        if !(sum_w.is_finite() && sum_w > 0.0) {
            return Err(Error::Validation(format!(
                "KdeNdPdf requires sum(weights) > 0, got {sum_w}"
            )));
        }

        Ok(Self {
            observable_names,
            n_dim,
            centers,
            bandwidths: bw,
            inv_bandwidths: inv_bw,
            support,
            kernel_log_prefactor,
            log_sum_w: sum_w.ln(),
            n_kernels,
        })
    }

    /// Evaluate the log-density (unnormalized by sum_w) at a single point.
    #[inline]
    fn log_density_at(&self, x: &[f64]) -> f64 {
        // log p(x) = logsumexp_k [ log_prefactor_k + Σ_d log φ(z_{k,d}) ] - log(sum_w)
        let mut max_val = f64::NEG_INFINITY;
        let mut terms = Vec::with_capacity(self.n_kernels);

        for k in 0..self.n_kernels {
            let lpref = self.kernel_log_prefactor[k];
            if !lpref.is_finite() {
                terms.push(f64::NEG_INFINITY);
                continue;
            }

            let mut log_kernel = lpref;
            for (d, &xi) in x.iter().enumerate().take(self.n_dim) {
                let z = (xi - self.centers[k][d]) * self.inv_bandwidths[d];
                log_kernel += standard_normal_logpdf(z);
            }

            terms.push(log_kernel);
            if log_kernel > max_val {
                max_val = log_kernel;
            }
        }

        if !max_val.is_finite() {
            return f64::NEG_INFINITY;
        }

        let mut s = 0.0f64;
        for &t in &terms {
            s += (t - max_val).exp();
        }

        max_val + s.ln() - self.log_sum_w
    }
}

impl UnbinnedPdf for KdeNdPdf {
    fn n_params(&self) -> usize {
        0
    }

    fn observables(&self) -> &[String] {
        &self.observable_names
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "KdeNdPdf expects 0 params, got {}",
                params.len()
            )));
        }
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "KdeNdPdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        // Gather column references.
        let cols: Vec<&[f64]> = self
            .observable_names
            .iter()
            .map(|name| {
                events
                    .column(name)
                    .ok_or_else(|| Error::Validation(format!("missing column '{name}'")))
            })
            .collect::<Result<_>>()?;

        let mut x_buf = vec![0.0f64; self.n_dim];
        for i in 0..n {
            for d in 0..self.n_dim {
                x_buf[d] = cols[d][i];
            }
            out[i] = self.log_density_at(&x_buf);
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
        if !out_grad.is_empty() {
            return Err(Error::Validation(format!(
                "KdeNdPdf out_grad must be empty (n_params=0), got len={}",
                out_grad.len()
            )));
        }
        self.log_prob_batch(events, params, out_logp)
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "KdeNdPdf expects 0 params, got {}",
                params.len()
            )));
        }
        if support.len() != self.n_dim {
            return Err(Error::Validation(format!(
                "KdeNdPdf sample expects {}D support, got {}D",
                self.n_dim,
                support.len()
            )));
        }

        use statrs::distribution::{ContinuousCDF, Normal};
        let stdn = Normal::new(0.0, 1.0)
            .map_err(|e| Error::Computation(format!("failed to construct standard normal: {e}")))?;

        // Build CDF over kernels for weighted selection.
        let mut kernel_cdf = Vec::with_capacity(self.n_kernels);
        let mut sum_w = 0.0f64;
        for k in 0..self.n_kernels {
            // Recover weight from log_prefactor (approximate: just use exp(log_prefactor)).
            // Simpler: recompute from raw sum_w. We stored log_sum_w = ln(sum_w).
            // For uniform weights, each kernel has equal probability.
            sum_w += self.kernel_log_prefactor[k].exp();
            kernel_cdf.push(sum_w);
        }
        for v in &mut kernel_cdf {
            *v /= sum_w;
        }
        if let Some(last) = kernel_cdf.last_mut() {
            *last = 1.0;
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let mut columns: Vec<Vec<f64>> =
            (0..self.n_dim).map(|_| Vec::with_capacity(n_events)).collect();

        for _ in 0..n_events {
            let u = u01(rng);
            let idx = kernel_cdf.partition_point(|p| *p < u).min(self.n_kernels - 1);

            for (d, col) in columns.iter_mut().enumerate().take(self.n_dim) {
                let x0 = self.centers[idx][d];
                let h = self.bandwidths[d];
                let (lo, hi) = self.support[d];

                let z_lo = (lo - x0) / h;
                let z_hi = (hi - x0) / h;
                let mut u_lo = stdn.cdf(z_lo);
                let mut u_hi = stdn.cdf(z_hi);
                let eps = 1e-15;
                u_lo = u_lo.clamp(eps, 1.0 - eps);
                u_hi = u_hi.clamp(eps, 1.0 - eps);
                if u_lo >= u_hi {
                    col.push(x0.clamp(lo, hi));
                    continue;
                }

                let uu = u_lo + (u_hi - u_lo) * u01(rng);
                let z = stdn.inverse_cdf(uu);
                let x = (x0 + h * z).clamp(lo, hi);
                col.push(x);
            }
        }

        let obs_specs: Vec<ObservableSpec> = self
            .observable_names
            .iter()
            .zip(&self.support)
            .map(|(name, &bounds)| ObservableSpec::branch(name.clone(), bounds))
            .collect();

        let col_pairs: Vec<(String, Vec<f64>)> = self
            .observable_names
            .iter()
            .zip(columns)
            .map(|(name, col)| (name.clone(), col))
            .collect();

        EventStore::from_columns(obs_specs, col_pairs, None)
    }
}

/// Silverman's rule of thumb for diagonal bandwidth in D dimensions.
///
/// `h_d = σ_d · (4 / ((d+2) · N))^(1/(d+4))`
fn silverman_bandwidths(centers: &[Vec<f64>], weights: Option<&[f64]>, n_dim: usize) -> Vec<f64> {
    let n = centers.len() as f64;
    let factor = (4.0 / ((n_dim as f64 + 2.0) * n)).powf(1.0 / (n_dim as f64 + 4.0));

    let mut bw = Vec::with_capacity(n_dim);
    for d in 0..n_dim {
        // Weighted standard deviation.
        let (_mean, var) = weighted_mean_var(centers.iter().map(|row| row[d]), weights);
        let sigma = var.sqrt().max(1e-10);
        bw.push(sigma * factor);
    }
    bw
}

/// Compute weighted mean and variance for a single dimension.
fn weighted_mean_var(
    values: impl Iterator<Item = f64> + Clone,
    weights: Option<&[f64]>,
) -> (f64, f64) {
    let mut sum_w = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut sum_wx2 = 0.0f64;

    for (i, x) in values.enumerate() {
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        sum_w += w;
        sum_wx += w * x;
        sum_wx2 += w * x * x;
    }

    if sum_w <= 0.0 {
        return (0.0, 1.0);
    }

    let mean = sum_wx / sum_w;
    let var = (sum_wx2 / sum_w - mean * mean).max(0.0);
    (mean, var)
}
