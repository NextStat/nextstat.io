//! Numerical normalization for PDFs without analytical integrals.
//!
//! Provides Gauss-Legendre quadrature on bounded 1-D supports. For multi-D flows
//! the product quadrature grid is constructed as a tensor product of 1-D rules.
//!
//! For high-dimensional PDFs (4D+), low-order tensor products (N8/N16) keep the
//! grid size tractable. For 6D+ a quasi-Monte Carlo grid using Sobol sequences
//! avoids the exponential curse of dimensionality.

use crate::event_store::{EventStore, ObservableSpec};
use crate::pdf::UnbinnedPdf;
use ns_core::Result;

/// Gauss-Legendre quadrature order (number of nodes per dimension).
///
/// Higher orders give better accuracy but cost more PDF evaluations.
/// For a smooth normalizing flow, 64 nodes typically give ~12 digits of accuracy in 1-D.
#[derive(Debug, Clone, Copy, Default)]
pub enum QuadratureOrder {
    /// 8 nodes per dimension (for 4-5D tensor products: 8⁴ = 4096, 8⁵ = 32768).
    N8,
    /// 16 nodes per dimension (for 4D tensor products: 16⁴ = 65536).
    N16,
    /// 32 nodes per dimension.
    N32,
    /// 64 nodes per dimension (default, recommended).
    #[default]
    N64,
    /// 128 nodes per dimension (high precision).
    N128,
}

impl QuadratureOrder {
    fn n(self) -> usize {
        match self {
            Self::N8 => 8,
            Self::N16 => 16,
            Self::N32 => 32,
            Self::N64 => 64,
            Self::N128 => 128,
        }
    }
}

/// Compute Gauss-Legendre nodes and weights on `[-1, 1]` for the given order.
///
/// Uses Newton iteration to find roots of the Legendre polynomial P_n(x),
/// then computes weights from the derivative P'_n at each root.
/// Exploits symmetry: only computes half the roots.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0f64; n];
    let mut weights = vec![0.0f64; n];

    if n == 0 {
        return (nodes, weights);
    }
    if n == 1 {
        nodes[0] = 0.0;
        weights[0] = 2.0;
        return (nodes, weights);
    }

    let nf = n as f64;
    let m = n.div_ceil(2); // number of unique positive roots (exploiting symmetry)

    for i in 0..m {
        // Initial guess via Chebyshev approximation.
        let mut x = ((std::f64::consts::PI * (i as f64 + 0.75)) / (nf + 0.5)).cos();

        // Newton iteration to refine root of P_n(x).
        for _ in 0..100 {
            // Evaluate P_n(x) and P'_n(x) via recurrence.
            let mut p0 = 1.0f64; // P_0(x)
            let mut p1 = x; // P_1(x)
            for j in 2..=n {
                let jf = j as f64;
                let p2 = ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf;
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(x), p0 = P_{n-1}(x)
            // Derivative: P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
            let dp = nf * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        // Final evaluation for weight computation.
        let mut p0 = 1.0f64;
        let mut p1 = x;
        for j in 2..=n {
            let jf = j as f64;
            let p2 = ((2.0 * jf - 1.0) * x * p1 - (jf - 1.0) * p0) / jf;
            p0 = p1;
            p1 = p2;
        }
        let dp = nf * (x * p1 - p0) / (x * x - 1.0);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);

        // Place symmetric pair.
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

/// Public (crate-internal) wrapper for [`gauss_legendre_nodes_weights`].
pub(crate) fn gauss_legendre_nodes_weights_pub(n: usize) -> (Vec<f64>, Vec<f64>) {
    gauss_legendre_nodes_weights(n)
}

/// Map Gauss-Legendre nodes from `[-1, 1]` to `[a, b]` and scale weights accordingly.
fn map_to_interval(nodes: &[f64], weights: &[f64], a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let half_len = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    let mapped_nodes: Vec<f64> = nodes.iter().map(|&x| mid + half_len * x).collect();
    let mapped_weights: Vec<f64> = weights.iter().map(|&w| w * half_len).collect();
    (mapped_nodes, mapped_weights)
}

/// Pre-computed quadrature grid for normalizing a PDF over its support.
#[derive(Debug, Clone)]
pub struct QuadratureGrid {
    /// The `EventStore` containing the quadrature nodes as "events".
    pub events: EventStore,
    /// Quadrature weights (length = total number of grid points).
    pub weights: Vec<f64>,
}

impl QuadratureGrid {
    /// Build a 1-D quadrature grid for a single observable.
    pub fn new_1d(
        observable_name: &str,
        bounds: (f64, f64),
        order: QuadratureOrder,
    ) -> Result<Self> {
        let n = order.n();
        let (ref_nodes, ref_weights) = gauss_legendre_nodes_weights(n);
        let (nodes, weights) = map_to_interval(&ref_nodes, &ref_weights, bounds.0, bounds.1);

        let obs = vec![ObservableSpec::branch(observable_name.to_string(), bounds)];
        let columns = vec![(observable_name.to_string(), nodes)];
        let events = EventStore::from_columns(obs, columns, None)?;

        Ok(Self { events, weights })
    }

    /// Build a multi-D tensor-product quadrature grid.
    ///
    /// For `d` observables with `n` nodes each, the grid has `n^d` points.
    pub fn new_tensor_product(
        observable_names: &[String],
        bounds: &[(f64, f64)],
        order: QuadratureOrder,
    ) -> Result<Self> {
        let d = observable_names.len();
        assert_eq!(d, bounds.len());
        if d == 0 {
            return Err(ns_core::Error::Validation(
                "QuadratureGrid requires at least one observable".into(),
            ));
        }

        let n = order.n();
        let (ref_nodes, ref_weights) = gauss_legendre_nodes_weights(n);

        // Per-dimension mapped nodes and weights.
        let mut dim_nodes = Vec::with_capacity(d);
        let mut dim_weights = Vec::with_capacity(d);
        for &(a, b) in bounds {
            let (mn, mw) = map_to_interval(&ref_nodes, &ref_weights, a, b);
            dim_nodes.push(mn);
            dim_weights.push(mw);
        }

        // Total grid points.
        let total: usize = dim_nodes.iter().map(|v| v.len()).product();

        // Build tensor product grid.
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(total); d];
        let mut weights = Vec::with_capacity(total);

        // Iterate over all combinations (tensor product).
        let mut indices = vec![0usize; d];
        for _ in 0..total {
            let mut w = 1.0;
            for dim in 0..d {
                columns[dim].push(dim_nodes[dim][indices[dim]]);
                w *= dim_weights[dim][indices[dim]];
            }
            weights.push(w);

            // Increment multi-index (odometer).
            let mut carry = true;
            for dim in (0..d).rev() {
                if carry {
                    indices[dim] += 1;
                    if indices[dim] >= n {
                        indices[dim] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        let obs: Vec<ObservableSpec> = observable_names
            .iter()
            .zip(bounds)
            .map(|(name, &b)| ObservableSpec::branch(name.clone(), b))
            .collect();

        let col_pairs: Vec<(String, Vec<f64>)> =
            observable_names.iter().zip(columns).map(|(name, col)| (name.clone(), col)).collect();

        let events = EventStore::from_columns(obs, col_pairs, None)?;

        Ok(Self { events, weights })
    }

    /// Build a quasi-Monte Carlo grid using Sobol sequences for high-dimensional PDFs (6D+).
    ///
    /// Unlike tensor products, the cost is O(N) regardless of dimension, avoiding the
    /// exponential curse of dimensionality. Each sample point is weighted equally:
    /// `w_i = volume / N`.
    ///
    /// `n_points` controls accuracy (default recommendation: 50_000 for 6-10D).
    pub fn new_sobol(
        observable_names: &[String],
        bounds: &[(f64, f64)],
        n_points: usize,
    ) -> Result<Self> {
        let d = observable_names.len();
        assert_eq!(d, bounds.len());
        if d == 0 {
            return Err(ns_core::Error::Validation(
                "QuadratureGrid requires at least one observable".into(),
            ));
        }
        if n_points == 0 {
            return Err(ns_core::Error::Validation("n_points must be > 0".into()));
        }

        // Compute volume = Π (b_i - a_i).
        let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
        let w = volume / n_points as f64;

        // Generate Sobol-like low-discrepancy sequence via the bit-reversal radical-inverse
        // (van der Corput) in different prime bases per dimension.
        // For dimensions 1..d we use bases 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...
        let primes = small_primes(d);

        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n_points); d];
        let weights = vec![w; n_points];

        for i in 1..=n_points {
            for dim in 0..d {
                let u = radical_inverse(i, primes[dim]);
                let (a, b) = bounds[dim];
                columns[dim].push(a + u * (b - a));
            }
        }

        let obs: Vec<ObservableSpec> = observable_names
            .iter()
            .zip(bounds)
            .map(|(name, &b)| ObservableSpec::branch(name.clone(), b))
            .collect();

        let col_pairs: Vec<(String, Vec<f64>)> =
            observable_names.iter().zip(columns).map(|(name, col)| (name.clone(), col)).collect();

        let events = EventStore::from_columns(obs, col_pairs, None)?;

        Ok(Self { events, weights })
    }

    /// Automatically choose the best grid for the given dimensionality.
    ///
    /// | D   | Strategy                              | Grid size       |
    /// |-----|---------------------------------------|-----------------|
    /// | 1   | Gauss-Legendre N128                   | 128             |
    /// | 2   | Gauss-Legendre N64                    | 4,096           |
    /// | 3   | Tensor product N32                    | 32,768          |
    /// | 4   | Tensor product N16                    | 65,536          |
    /// | 5   | Tensor product N8                     | 32,768          |
    /// | 6+  | Quasi-Monte Carlo (Halton) 65,536 pts | 65,536          |
    pub fn auto(observable_names: &[String], bounds: &[(f64, f64)]) -> Result<Self> {
        let d = observable_names.len();
        match d {
            0 => Err(ns_core::Error::Validation(
                "QuadratureGrid requires at least one observable".into(),
            )),
            1 => Self::new_1d(&observable_names[0], bounds[0], QuadratureOrder::N128),
            2 => Self::new_tensor_product(observable_names, bounds, QuadratureOrder::N64),
            3 => Self::new_tensor_product(observable_names, bounds, QuadratureOrder::N32),
            4 => Self::new_tensor_product(observable_names, bounds, QuadratureOrder::N16),
            5 => Self::new_tensor_product(observable_names, bounds, QuadratureOrder::N8),
            _ => Self::new_sobol(observable_names, bounds, 65_536),
        }
    }
}

/// Return the first `n` prime numbers.
fn small_primes(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let mut primes = Vec::with_capacity(n);
    let mut candidate = 2usize;
    while primes.len() < n {
        let is_prime = primes.iter().all(|&p| candidate % p != 0);
        if is_prime {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

/// Radical-inverse function (van der Corput sequence) in the given base.
///
/// Maps integer `i` to a value in `[0, 1)` with low discrepancy.
fn radical_inverse(mut i: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f64;
    while i > 0 {
        result += (i % base) as f64 * fraction;
        i /= base;
        fraction /= base as f64;
    }
    result
}

/// Compute the normalization integral `∫_Ω p(x|θ) dx` using a pre-built quadrature grid.
///
/// Returns `ln(∫ p dx)` for numerical stability.
pub fn log_normalize_quadrature(
    pdf: &dyn UnbinnedPdf,
    grid: &QuadratureGrid,
    params: &[f64],
) -> Result<f64> {
    let n = grid.events.n_events();
    let mut logp = vec![0.0; n];
    pdf.log_prob_batch(&grid.events, params, &mut logp)?;

    // Compute log(∫ exp(logp) * w dx) via log-sum-exp with weights.
    // integral = Σ_i exp(logp_i) * w_i
    // log(integral) = log(Σ_i exp(logp_i) * w_i)
    // We use the log-sum-exp trick: find max(logp_i + ln(w_i)), factor it out.
    let log_terms: Vec<f64> = logp
        .iter()
        .zip(&grid.weights)
        .map(|(&lp, &w)| if w <= 0.0 { f64::NEG_INFINITY } else { lp + w.ln() })
        .collect();

    let max_val = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_val.is_finite() {
        return Err(ns_core::Error::Validation(
            "normalization quadrature: all terms are -inf (PDF is zero on the support?)".into(),
        ));
    }

    let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_val).exp()).sum();
    let log_integral = max_val + sum_exp.ln();

    Ok(log_integral)
}

/// Compute the gradient of the log-normalization integral w.r.t. shape parameters.
///
/// Returns `(ln_integral, grad)` where `grad[k] = d/dθ_k ln(∫ p(x|θ) dx)`.
///
/// Uses the identity: `d/dθ ln(∫ p dx) = ∫ (d/dθ log p) · p dx / ∫ p dx`
/// which equals `E_p[d/dθ log p]` — the expected gradient under the PDF.
pub fn log_normalize_quadrature_grad(
    pdf: &dyn UnbinnedPdf,
    grid: &QuadratureGrid,
    params: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let n = grid.events.n_events();
    let n_params = pdf.n_params();

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * n_params];
    pdf.log_prob_grad_batch(&grid.events, params, &mut logp, &mut grad)?;

    // Compute log-normalization via log-sum-exp with quadrature weights.
    let log_terms: Vec<f64> = logp
        .iter()
        .zip(&grid.weights)
        .map(|(&lp, &w)| if w <= 0.0 { f64::NEG_INFINITY } else { lp + w.ln() })
        .collect();

    let max_val = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_val.is_finite() {
        return Err(ns_core::Error::Validation(
            "normalization quadrature grad: all terms are -inf".into(),
        ));
    }

    let exp_terms: Vec<f64> = log_terms.iter().map(|&lt| (lt - max_val).exp()).collect();
    let sum_exp: f64 = exp_terms.iter().sum();
    let log_integral = max_val + sum_exp.ln();

    // Gradient: d/dθ_k ln(∫ p dx) = Σ_i [w_i · p(x_i) · (d/dθ_k log p(x_i))] / (∫ p dx)
    // = Σ_i exp(logp_i + ln(w_i) - log_integral) · grad_ik
    let mut out_grad = vec![0.0f64; n_params];
    for i in 0..n {
        let weight_factor = (log_terms[i] - log_integral).exp();
        if !weight_factor.is_finite() {
            continue;
        }
        for k in 0..n_params {
            out_grad[k] += weight_factor * grad[i * n_params + k];
        }
    }

    Ok((log_integral, out_grad))
}

/// Cache for normalization integrals, keyed by discretized parameter values.
///
/// For PDFs with expensive normalization (e.g., numerical quadrature), this cache
/// avoids recomputing `∫ p(x|θ) dx` when parameters haven't changed significantly.
///
/// The cache uses a simple hash-map with parameter values rounded to a configurable
/// number of significant digits.
pub struct NormalizationCache {
    /// Number of decimal digits for rounding parameter values (key precision).
    precision_digits: u32,
    /// Cached `(ln_integral, grad)` indexed by rounded parameter key.
    cache: std::collections::HashMap<Vec<i64>, (f64, Vec<f64>)>,
    /// Pre-built quadrature grid (shared across all cache lookups).
    grid: QuadratureGrid,
}

impl NormalizationCache {
    /// Create a new normalization cache with a pre-built quadrature grid.
    ///
    /// `precision_digits` controls rounding: parameters are multiplied by `10^digits`
    /// and rounded to nearest integer for the cache key. Default: 6 (~1e-6 resolution).
    pub fn new(grid: QuadratureGrid, precision_digits: u32) -> Self {
        Self { precision_digits, cache: std::collections::HashMap::new(), grid }
    }

    /// Create a cache with default precision (6 digits).
    pub fn with_default_precision(grid: QuadratureGrid) -> Self {
        Self::new(grid, 6)
    }

    /// Look up or compute the log-normalization integral for the given parameters.
    pub fn log_norm(&mut self, pdf: &dyn UnbinnedPdf, params: &[f64]) -> Result<f64> {
        let key = self.make_key(params);
        if let Some((ln_int, _)) = self.cache.get(&key) {
            return Ok(*ln_int);
        }

        let ln_int = log_normalize_quadrature(pdf, &self.grid, params)?;
        self.cache.insert(key, (ln_int, Vec::new()));
        Ok(ln_int)
    }

    /// Look up or compute the log-normalization integral and its gradient.
    pub fn log_norm_grad(
        &mut self,
        pdf: &dyn UnbinnedPdf,
        params: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        let key = self.make_key(params);
        #[allow(clippy::collapsible_if)]
        if let Some((ln_int, grad)) = self.cache.get(&key) {
            if !grad.is_empty() {
                return Ok((*ln_int, grad.clone()));
            }
        }

        let (ln_int, grad) = log_normalize_quadrature_grad(pdf, &self.grid, params)?;
        self.cache.insert(key, (ln_int, grad.clone()));
        Ok((ln_int, grad))
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Access the underlying quadrature grid.
    pub fn grid(&self) -> &QuadratureGrid {
        &self.grid
    }

    fn make_key(&self, params: &[f64]) -> Vec<i64> {
        let scale = 10.0f64.powi(self.precision_digits as i32);
        params.iter().map(|&v| (v * scale).round() as i64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_legendre_integrates_polynomial() {
        // GL with n nodes integrates polynomials of degree ≤ 2n-1 exactly.
        // Test: ∫_{-1}^{1} x^2 dx = 2/3.
        let (nodes, weights) = gauss_legendre_nodes_weights(4);
        let integral: f64 = nodes.iter().zip(&weights).map(|(&x, &w)| x * x * w).sum();
        assert!((integral - 2.0 / 3.0).abs() < 1e-14, "got {integral}");
    }

    #[test]
    fn test_gauss_legendre_integrates_x4() {
        // ∫_{-1}^{1} x^4 dx = 2/5.
        let (nodes, weights) = gauss_legendre_nodes_weights(4);
        let integral: f64 = nodes.iter().zip(&weights).map(|(&x, &w)| x.powi(4) * w).sum();
        assert!((integral - 2.0 / 5.0).abs() < 1e-14, "got {integral}");
    }

    #[test]
    fn test_mapped_interval() {
        // ∫_{0}^{1} x dx = 0.5
        let (nodes, weights) = gauss_legendre_nodes_weights(8);
        let (mn, mw) = map_to_interval(&nodes, &weights, 0.0, 1.0);
        let integral: f64 = mn.iter().zip(&mw).map(|(&x, &w)| x * w).sum();
        assert!((integral - 0.5).abs() < 1e-14, "got {integral}");
    }

    #[test]
    fn test_quadrature_grid_1d() {
        let grid = QuadratureGrid::new_1d("x", (0.0, 1.0), QuadratureOrder::N32).unwrap();
        assert_eq!(grid.events.n_events(), 32);
        assert_eq!(grid.weights.len(), 32);
        // Sum of weights should equal interval length (1.0).
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "weight sum = {sum}");
    }

    #[test]
    fn test_quadrature_grid_2d() {
        let names = vec!["x".to_string(), "y".to_string()];
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let grid =
            QuadratureGrid::new_tensor_product(&names, &bounds, QuadratureOrder::N32).unwrap();
        assert_eq!(grid.events.n_events(), 32 * 32);
        assert_eq!(grid.weights.len(), 32 * 32);
        // Sum of weights = area = 1.0 * 2.0 = 2.0.
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-12, "weight sum = {sum}");
    }

    #[test]
    fn test_quadrature_grid_4d_n16() {
        let names: Vec<String> = (0..4).map(|i| format!("x{i}")).collect();
        let bounds = vec![(0.0, 1.0); 4];
        let grid =
            QuadratureGrid::new_tensor_product(&names, &bounds, QuadratureOrder::N16).unwrap();
        assert_eq!(grid.events.n_events(), 16usize.pow(4)); // 65536
        assert_eq!(grid.weights.len(), 16usize.pow(4));
        // Sum of weights = volume = 1^4 = 1.0.
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weight sum = {sum}");
    }

    #[test]
    fn test_quadrature_grid_5d_n8() {
        let names: Vec<String> = (0..5).map(|i| format!("x{i}")).collect();
        let bounds = vec![(0.0, 2.0); 5];
        let grid =
            QuadratureGrid::new_tensor_product(&names, &bounds, QuadratureOrder::N8).unwrap();
        assert_eq!(grid.events.n_events(), 8usize.pow(5)); // 32768
        assert_eq!(grid.weights.len(), 8usize.pow(5));
        // Sum of weights = volume = 2^5 = 32.
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - 32.0).abs() < 1e-8, "weight sum = {sum}");
    }

    #[test]
    fn test_sobol_grid_6d() {
        let names: Vec<String> = (0..6).map(|i| format!("x{i}")).collect();
        let bounds = vec![(0.0, 1.0); 6];
        let n_points = 10_000;
        let grid = QuadratureGrid::new_sobol(&names, &bounds, n_points).unwrap();
        assert_eq!(grid.events.n_events(), n_points);
        assert_eq!(grid.weights.len(), n_points);
        // Sum of weights = volume / N * N = volume = 1.0.
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weight sum = {sum}");
        // Each weight = 1/N.
        assert!((grid.weights[0] - 1.0 / n_points as f64).abs() < 1e-15);
    }

    #[test]
    fn test_sobol_grid_10d() {
        let names: Vec<String> = (0..10).map(|i| format!("x{i}")).collect();
        let bounds: Vec<(f64, f64)> = (0..10).map(|i| (0.0, (i + 1) as f64)).collect();
        let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
        let n_points = 50_000;
        let grid = QuadratureGrid::new_sobol(&names, &bounds, n_points).unwrap();
        assert_eq!(grid.events.n_events(), n_points);
        let sum: f64 = grid.weights.iter().sum();
        assert!((sum - volume).abs() / volume < 1e-10, "weight sum = {sum}, expected {volume}");
    }

    #[test]
    fn test_auto_selects_correct_strategy() {
        // 1D → N128
        let g1 = QuadratureGrid::auto(&["x".to_string()].to_vec(), &[(0.0, 1.0)]).unwrap();
        assert_eq!(g1.events.n_events(), 128);

        // 2D → N64 tensor product
        let names2: Vec<String> = (0..2).map(|i| format!("x{i}")).collect();
        let g2 = QuadratureGrid::auto(&names2, &[(0.0, 1.0); 2]).unwrap();
        assert_eq!(g2.events.n_events(), 64 * 64);

        // 3D → N32 tensor product
        let names3: Vec<String> = (0..3).map(|i| format!("x{i}")).collect();
        let g3 = QuadratureGrid::auto(&names3, &[(0.0, 1.0); 3]).unwrap();
        assert_eq!(g3.events.n_events(), 32usize.pow(3));

        // 4D → N16 tensor product
        let names4: Vec<String> = (0..4).map(|i| format!("x{i}")).collect();
        let g4 = QuadratureGrid::auto(&names4, &[(0.0, 1.0); 4]).unwrap();
        assert_eq!(g4.events.n_events(), 16usize.pow(4));

        // 5D → N8 tensor product
        let names5: Vec<String> = (0..5).map(|i| format!("x{i}")).collect();
        let g5 = QuadratureGrid::auto(&names5, &[(0.0, 1.0); 5]).unwrap();
        assert_eq!(g5.events.n_events(), 8usize.pow(5));

        // 6D → Sobol 65536
        let names6: Vec<String> = (0..6).map(|i| format!("x{i}")).collect();
        let g6 = QuadratureGrid::auto(&names6, &[(0.0, 1.0); 6]).unwrap();
        assert_eq!(g6.events.n_events(), 65_536);
    }

    #[test]
    fn test_radical_inverse_base2() {
        // radical_inverse(1, 2) = 0.5, (2, 2) = 0.25, (3, 2) = 0.75
        assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-15);
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-15);
        assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-15);
    }

    #[test]
    fn test_small_primes() {
        assert_eq!(small_primes(0), Vec::<usize>::new());
        assert_eq!(small_primes(1), vec![2]);
        assert_eq!(small_primes(5), vec![2, 3, 5, 7, 11]);
        assert_eq!(small_primes(10), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }
}
