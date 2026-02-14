//! Standalone L-BFGS-B state machine for GPU lockstep optimization.
//!
//! Shared between CUDA and Metal batch toy fitters.

/// Flat, cache-friendly ring buffer for L-BFGS s/y history pairs.
///
/// Stores `m` pairs of n-dimensional vectors (`s`, `y`) plus scalar `rho`
/// in contiguous memory. Replaces `Vec<Vec<f64>>` to eliminate per-pair
/// heap allocations and O(m) `remove(0)` shifts.
struct RingHistory {
    s_flat: Vec<f64>,
    y_flat: Vec<f64>,
    rho: Vec<f64>,
    n: usize,
    m: usize,
    head: usize,
    len: usize,
}

impl RingHistory {
    fn new(m: usize, n: usize) -> Self {
        debug_assert!(m > 0, "L-BFGS history depth m must be > 0");
        Self {
            s_flat: vec![0.0; m * n],
            y_flat: vec![0.0; m * n],
            rho: vec![0.0; m],
            n,
            m,
            head: 0,
            len: 0,
        }
    }

    /// Push a new (s, y, rho) entry. O(1) — overwrites oldest when full.
    fn push(&mut self, s: &[f64], y: &[f64], rho: f64) {
        let slot = if self.len < self.m {
            let slot = (self.head + self.len) % self.m;
            self.len += 1;
            slot
        } else {
            let slot = self.head;
            self.head = (self.head + 1) % self.m;
            slot
        };
        let off = slot * self.n;
        self.s_flat[off..off + self.n].copy_from_slice(s);
        self.y_flat[off..off + self.n].copy_from_slice(y);
        self.rho[slot] = rho;
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// Access s vector at logical index `i` (0 = oldest, len-1 = newest).
    #[inline]
    fn s(&self, i: usize) -> &[f64] {
        let slot = (self.head + i) % self.m;
        let off = slot * self.n;
        &self.s_flat[off..off + self.n]
    }

    /// Access y vector at logical index `i`.
    #[inline]
    fn y(&self, i: usize) -> &[f64] {
        let slot = (self.head + i) % self.m;
        let off = slot * self.n;
        &self.y_flat[off..off + self.n]
    }

    /// Access rho scalar at logical index `i`.
    #[inline]
    fn rho(&self, i: usize) -> f64 {
        let slot = (self.head + i) % self.m;
        self.rho[slot]
    }
}

/// Standalone L-BFGS-B state machine for one optimization problem.
///
/// Maintains the inverse Hessian approximation (via limited-memory pairs)
/// and supports bounded parameters with gradient projection.
pub(crate) struct LbfgsState {
    /// Current parameter values.
    pub x: Vec<f64>,
    /// Previous gradient.
    prev_grad: Option<Vec<f64>>,
    /// Previous parameters.
    prev_x: Option<Vec<f64>>,
    /// Current function value.
    pub fval: f64,
    /// Previous function value (for relative objective decrease checks).
    prev_fval: Option<f64>,
    /// L-BFGS history ring buffer (s, y, rho).
    history: RingHistory,
    /// Parameter bounds.
    bounds: Vec<(f64, f64)>,
    /// Base convergence tolerance (unscaled).
    tol: f64,
    /// Optional scaling for gradient tolerance (e.g. based on expected event count).
    ///
    /// If not set, we fall back to scaling based on `sqrt(|nll|)` at runtime.
    tol_scale: Option<f64>,
    /// Step counter.
    pub iter: usize,
    /// Number of function evaluations.
    pub n_fev: usize,
    /// Number of gradient evaluations.
    pub n_gev: usize,
    /// Whether converged.
    pub converged: bool,
    /// Whether optimization failed due to non-finite NLL/gradient.
    pub failed: bool,
}

impl LbfgsState {
    pub fn new(x0: Vec<f64>, bounds: Vec<(f64, f64)>, m: usize, tol: f64) -> Self {
        let n = x0.len();
        let x = Self::clamp_to_bounds(&x0, &bounds);
        Self {
            x,
            prev_grad: None,
            prev_x: None,
            fval: f64::INFINITY,
            prev_fval: None,
            history: RingHistory::new(m, n),
            bounds,
            tol,
            tol_scale: None,
            iter: 0,
            n_fev: 0,
            n_gev: 0,
            converged: false,
            failed: false,
        }
    }

    /// Scale the gradient-norm tolerance for large-N log-likelihoods.
    ///
    /// For statistical problems where the NLL is a sum over events, the gradient norm at
    /// "statistically good enough" solutions typically scales like `O(sqrt(N))`. Using an
    /// absolute tolerance can therefore lead to max-iter behavior at very large event counts.
    ///
    /// We scale the tolerance as: `tol_eff = tol * max(1, sqrt(n_events))`.
    pub fn set_expected_events(&mut self, n_events: usize) {
        let scale = (n_events as f64).sqrt().max(1.0);
        self.tol_scale = Some(scale);
    }

    fn effective_tol(&self, nll: f64) -> f64 {
        // Fall back to scaling with sqrt(|nll|) when no event-count scale is provided.
        // For sum-of-events NLLs, `|nll|` is O(N), so this is a cheap proxy for sqrt(N).
        let scale = self.tol_scale.unwrap_or_else(|| nll.abs().max(1.0).sqrt().max(1.0));
        self.tol * scale
    }

    fn relative_obj_change(prev: f64, curr: f64) -> f64 {
        (prev - curr).abs() / prev.abs().max(curr.abs()).max(1.0)
    }

    /// Begin an iteration at the current `x` given NLL+gradient evaluated at `x`.
    ///
    /// Returns a descent direction for proposing an `x_next`, or `None` if this state is done
    /// (converged or invalid numerics).
    pub fn begin_iter(&mut self, nll: f64, grad: &[f64]) -> Option<Vec<f64>> {
        let n = self.x.len();
        self.n_fev += 1;
        self.n_gev += 1;

        if !nll.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            self.failed = true;
            self.converged = true;
            self.fval = nll;
            return None;
        }
        self.fval = nll;

        let pg_norm = self.projected_gradient_norm(grad);
        if pg_norm < self.effective_tol(nll) {
            self.converged = true;
            return None;
        }

        if let Some(prev) = self.prev_fval
            && Self::relative_obj_change(prev, nll) < self.tol
        {
            self.converged = true;
            return None;
        }
        self.prev_fval = Some(nll);

        // Update L-BFGS memory (x_k - x_{k-1}, g_k - g_{k-1}).
        if let (Some(prev_x), Some(prev_g)) = (&self.prev_x, &self.prev_grad) {
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            let mut sy = 0.0;
            for i in 0..n {
                s[i] = self.x[i] - prev_x[i];
                y[i] = grad[i] - prev_g[i];
                sy += s[i] * y[i];
            }
            if sy > 1e-10 {
                self.history.push(&s, &y, 1.0 / sy);
            }
        }

        // Save current state for the next history update.
        self.prev_x = Some(self.x.clone());
        self.prev_grad = Some(grad.to_vec());

        let mut direction = self.lbfgs_direction(grad);
        let mut dir_deriv = 0.0;
        for i in 0..n {
            dir_deriv += grad[i] * direction[i];
        }
        // Ensure a descent direction (fall back to steepest descent).
        if !(dir_deriv < 0.0) {
            direction = grad.iter().map(|&g| -g).collect();
        }

        Some(direction)
    }

    /// Propose a step size that stays within bounds (guarded slightly away from hard bounds).
    pub fn propose_step_size(&self, direction: &[f64]) -> f64 {
        let n = self.x.len();
        let mut step: f64 = 1.0;

        for i in 0..n {
            if (self.bounds[i].1 - self.bounds[i].0).abs() <= 0.0 {
                continue;
            }
            let span = (self.bounds[i].1 - self.bounds[i].0).abs();
            let guard = span.max(1.0) * 1e-12;
            let lo = self.bounds[i].0 + guard;
            let hi = self.bounds[i].1 - guard;
            if lo >= hi {
                continue;
            }
            let d = direction[i];
            if d > 0.0 {
                let max_step = (hi - self.x[i]) / d;
                if max_step.is_finite() && max_step > 0.0 {
                    step = step.min(max_step);
                }
            } else if d < 0.0 {
                let max_step = (lo - self.x[i]) / d;
                if max_step.is_finite() && max_step > 0.0 {
                    step = step.min(max_step);
                }
            }
        }

        step.max(1e-20)
    }

    /// Compute a trial `x_next` from `direction` and `step_size`, clamped to hard bounds.
    pub fn propose_x(&self, direction: &[f64], step_size: f64) -> Vec<f64> {
        let mut x_next = self.x.clone();
        for (i, xi) in x_next.iter_mut().enumerate() {
            *xi += step_size * direction[i];
        }
        Self::clamp_to_bounds(&x_next, &self.bounds)
    }

    /// Commit an accepted step and advance the iteration counter.
    pub fn accept_x(&mut self, x_next: Vec<f64>) {
        self.x = x_next;
        self.iter += 1;
    }

    /// Perform one L-BFGS-B step given current NLL and gradient.
    ///
    /// Returns the new parameters to evaluate. Sets `converged = true` if
    /// the projected gradient norm is below tolerance.
    pub fn step(&mut self, nll: f64, grad: &[f64]) -> &[f64] {
        let n = self.x.len();
        self.n_fev += 1;
        self.n_gev += 1;

        // Bail out early if NLL or gradient contains NaN/Inf — continuing
        // would silently corrupt the inverse Hessian approximation.
        if !nll.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            self.failed = true;
            self.converged = true;
            self.fval = nll;
            return &self.x;
        }
        self.fval = nll;

        // Check convergence: projected gradient norm
        let pg_norm = self.projected_gradient_norm(grad);
        if pg_norm < self.effective_tol(nll) {
            self.converged = true;
            return &self.x;
        }

        if let Some(prev) = self.prev_fval
            && Self::relative_obj_change(prev, nll) < self.tol
        {
            self.converged = true;
            return &self.x;
        }
        self.prev_fval = Some(nll);

        // Update L-BFGS memory
        if let (Some(prev_x), Some(prev_g)) = (&self.prev_x, &self.prev_grad) {
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            let mut sy = 0.0;
            for i in 0..n {
                s[i] = self.x[i] - prev_x[i];
                y[i] = grad[i] - prev_g[i];
                sy += s[i] * y[i];
            }
            if sy > 1e-10 {
                self.history.push(&s, &y, 1.0 / sy);
            }
        }

        // L-BFGS two-loop recursion → search direction
        let mut direction = self.lbfgs_direction(grad);

        // Ensure descent direction (match begin_iter() safety check).
        let mut dir_deriv = 0.0;
        for i in 0..n {
            dir_deriv += grad[i] * direction[i];
        }
        if !(dir_deriv < 0.0) {
            direction = grad.iter().map(|&g| -g).collect();
        }

        // Backtracking line search with Armijo condition
        let step_size = self.line_search(grad, &direction);

        // Save current state
        self.prev_x = Some(self.x.clone());
        self.prev_grad = Some(grad.to_vec());

        // Update parameters
        for i in 0..n {
            self.x[i] += step_size * direction[i];
        }
        self.x = Self::clamp_to_bounds(&self.x, &self.bounds);

        self.iter += 1;
        &self.x
    }

    /// L-BFGS two-loop recursion to compute search direction.
    fn lbfgs_direction(&self, grad: &[f64]) -> Vec<f64> {
        let n = grad.len();
        let k = self.history.len();

        if k == 0 {
            return grad.iter().map(|&g| -g).collect();
        }

        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; k];

        // Backward pass
        for i in (0..k).rev() {
            let si = self.history.s(i);
            let mut dot = 0.0;
            for j in 0..n {
                dot += si[j] * q[j];
            }
            alpha[i] = self.history.rho(i) * dot;
            let yi = self.history.y(i);
            for j in 0..n {
                q[j] -= alpha[i] * yi[j];
            }
        }

        // Initial Hessian approximation: H0 = gamma * I
        let last = k - 1;
        let s_last = self.history.s(last);
        let y_last = self.history.y(last);
        let mut sy = 0.0;
        let mut yy = 0.0;
        for j in 0..n {
            sy += s_last[j] * y_last[j];
            yy += y_last[j] * y_last[j];
        }
        let gamma = if yy > 1e-30 { sy / yy } else { 1.0 };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Forward pass
        for i in 0..k {
            let yi = self.history.y(i);
            let mut dot = 0.0;
            for j in 0..n {
                dot += yi[j] * r[j];
            }
            let beta = self.history.rho(i) * dot;
            let si = self.history.s(i);
            for j in 0..n {
                r[j] += (alpha[i] - beta) * si[j];
            }
        }

        // Negate for descent direction
        for v in &mut r {
            *v = -*v;
        }
        r
    }

    /// Simple backtracking line search with Armijo condition.
    fn line_search(&self, grad: &[f64], direction: &[f64]) -> f64 {
        let n = self.x.len();

        // Directional derivative
        let mut dir_deriv = 0.0;
        for i in 0..n {
            dir_deriv += grad[i] * direction[i];
        }

        // If direction is not a descent direction, fall back to steepest descent.
        if dir_deriv >= 0.0 {
            let mut step: f64 = 1.0;
            for i in 0..n {
                if (self.bounds[i].1 - self.bounds[i].0).abs() <= 0.0 {
                    continue;
                }
                let span = (self.bounds[i].1 - self.bounds[i].0).abs();
                let guard = span.max(1.0) * 1e-12;
                let lo = self.bounds[i].0 + guard;
                let hi = self.bounds[i].1 - guard;
                if lo >= hi {
                    continue;
                }
                let d = -grad[i];
                if d > 0.0 {
                    let max_step = (hi - self.x[i]) / d;
                    if max_step.is_finite() && max_step > 0.0 {
                        step = step.min(max_step);
                    }
                } else if d < 0.0 {
                    let max_step = (lo - self.x[i]) / d;
                    if max_step.is_finite() && max_step > 0.0 {
                        step = step.min(max_step);
                    }
                }
            }
            return step.max(1e-20);
        }

        // Initial step size: 1.0 for quasi-Newton
        let mut step: f64 = 1.0;

        // Clamp step to stay within bounds
        for i in 0..n {
            if (self.bounds[i].1 - self.bounds[i].0).abs() <= 0.0 {
                continue;
            }
            let span = (self.bounds[i].1 - self.bounds[i].0).abs();
            let guard = span.max(1.0) * 1e-12;
            let lo = self.bounds[i].0 + guard;
            let hi = self.bounds[i].1 - guard;
            if lo >= hi {
                continue;
            }
            if direction[i] > 0.0 {
                let max_step = (hi - self.x[i]) / direction[i];
                if max_step.is_finite() && max_step > 0.0 {
                    step = step.min(max_step);
                }
            } else if direction[i] < 0.0 {
                let max_step = (lo - self.x[i]) / direction[i];
                if max_step.is_finite() && max_step > 0.0 {
                    step = step.min(max_step);
                }
            }
        }

        step.max(1e-20)
    }

    /// Projected gradient norm (gradient clamped at bounds).
    fn projected_gradient_norm(&self, grad: &[f64]) -> f64 {
        let n = self.x.len();
        let mut norm_sq = 0.0;
        for i in 0..n {
            let g = grad[i];
            let pg = if self.x[i] <= self.bounds[i].0 && g > 0.0 {
                0.0
            } else if self.x[i] >= self.bounds[i].1 && g < 0.0 {
                0.0
            } else {
                g
            };
            norm_sq += pg * pg;
        }
        norm_sq.sqrt()
    }

    /// Compute EDM (Estimated Distance to Minimum) = g^T H^{-1} g.
    ///
    /// Uses the L-BFGS two-loop recursion to compute H^{-1} g without forming
    /// the full inverse Hessian. Returns `NAN` if no history is available.
    pub fn compute_edm(&self, grad: &[f64]) -> f64 {
        if self.history.len() == 0 {
            return f64::NAN;
        }
        // lbfgs_direction returns -H^{-1} * g (descent direction).
        let neg_h_inv_g = self.lbfgs_direction(grad);
        // EDM = g^T * H^{-1} * g = g^T * (-neg_h_inv_g) = -sum(g_i * d_i)
        let mut edm = 0.0;
        for i in 0..grad.len() {
            edm -= grad[i] * neg_h_inv_g[i];
        }
        // EDM should be non-negative for a positive-definite H^{-1}.
        edm.max(0.0)
    }

    fn clamp_to_bounds(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        x.iter().zip(bounds.iter()).map(|(&v, &(lo, hi))| v.clamp(lo, hi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{LbfgsState, RingHistory};

    #[test]
    fn test_ring_history_empty() {
        let ring = RingHistory::new(3, 2);
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn test_ring_history_push_and_read() {
        let mut ring = RingHistory::new(3, 2);
        ring.push(&[1.0, 2.0], &[3.0, 4.0], 0.5);
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.s(0), &[1.0, 2.0]);
        assert_eq!(ring.y(0), &[3.0, 4.0]);
        assert_eq!(ring.rho(0), 0.5);
    }

    #[test]
    fn test_ring_history_partial_fill() {
        let mut ring = RingHistory::new(4, 2);
        for i in 0..3 {
            let v = i as f64;
            ring.push(&[v, v + 0.1], &[v + 10.0, v + 10.1], v + 100.0);
        }
        assert_eq!(ring.len(), 3);
        // Logical order: 0=oldest, 2=newest
        assert_eq!(ring.s(0), &[0.0, 0.1]);
        assert_eq!(ring.s(2), &[2.0, 2.1]);
        assert_eq!(ring.rho(1), 101.0);
    }

    #[test]
    fn test_ring_history_full_wraparound() {
        let mut ring = RingHistory::new(3, 2);
        // Push 5 entries into a ring of capacity 3
        for i in 0..5 {
            let v = i as f64;
            ring.push(&[v, v], &[v + 10.0, v + 10.0], v as f64);
        }
        assert_eq!(ring.len(), 3);
        // Should contain entries 2, 3, 4 (oldest to newest)
        assert_eq!(ring.s(0), &[2.0, 2.0]);
        assert_eq!(ring.s(1), &[3.0, 3.0]);
        assert_eq!(ring.s(2), &[4.0, 4.0]);
        assert_eq!(ring.rho(0), 2.0);
        assert_eq!(ring.rho(2), 4.0);
    }

    #[test]
    fn test_ring_history_m1() {
        let mut ring = RingHistory::new(1, 3);
        ring.push(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 0.1);
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.s(0), &[1.0, 2.0, 3.0]);

        ring.push(&[7.0, 8.0, 9.0], &[10.0, 11.0, 12.0], 0.2);
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.s(0), &[7.0, 8.0, 9.0]);
        assert_eq!(ring.rho(0), 0.2);
    }

    #[test]
    fn test_ring_history_matches_vec_reference() {
        // Push N entries through both Vec<Vec> and RingHistory, compare iteration.
        let m = 3;
        let n = 4;
        let total = 7;

        let mut ring = RingHistory::new(m, n);
        let mut s_vec: Vec<Vec<f64>> = Vec::new();
        let mut y_vec: Vec<Vec<f64>> = Vec::new();
        let mut rho_vec: Vec<f64> = Vec::new();

        for i in 0..total {
            let s: Vec<f64> = (0..n).map(|j| (i * n + j) as f64).collect();
            let y: Vec<f64> = (0..n).map(|j| (i * n + j + 100) as f64).collect();
            let rho = i as f64 * 0.01;

            ring.push(&s, &y, rho);

            if s_vec.len() >= m {
                s_vec.remove(0);
                y_vec.remove(0);
                rho_vec.remove(0);
            }
            s_vec.push(s);
            y_vec.push(y);
            rho_vec.push(rho);
        }

        assert_eq!(ring.len(), s_vec.len());
        for i in 0..ring.len() {
            assert_eq!(ring.s(i), s_vec[i].as_slice(), "s mismatch at {i}");
            assert_eq!(ring.y(i), y_vec[i].as_slice(), "y mismatch at {i}");
            assert_eq!(ring.rho(i), rho_vec[i], "rho mismatch at {i}");
        }
    }

    /// Micro-benchmark: RingHistory push+two-loop-read vs Vec<Vec> push+remove(0)+read.
    /// Prints timing for both at realistic dimensions (n=200, m=50, 500 iterations).
    #[test]
    #[ignore = "perf; run with `cargo test -p ns-inference --release ring_vs_vec_microbench -- --ignored --nocapture`"]
    fn ring_vs_vec_microbench() {
        let n = 200; // realistic param count (tchannel-like)
        let m = 50; // auto-m for large models
        let iters = 500;

        // Pre-generate data
        let entries: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..iters)
            .map(|i| {
                let s: Vec<f64> = (0..n).map(|j| (i * n + j) as f64 * 0.001).collect();
                let y: Vec<f64> = (0..n).map(|j| (i * n + j + 1000) as f64 * 0.001).collect();
                (s, y, 1.0 / (i as f64 + 1.0))
            })
            .collect();

        // --- Vec<Vec> baseline ---
        let t0 = std::time::Instant::now();
        let mut s_vec: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_vec: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_vec: Vec<f64> = Vec::with_capacity(m);
        let mut checksum_vec = 0.0f64;
        for (s, y, rho) in &entries {
            if s_vec.len() >= m {
                s_vec.remove(0);
                y_vec.remove(0);
                rho_vec.remove(0);
            }
            s_vec.push(s.clone());
            y_vec.push(y.clone());
            rho_vec.push(*rho);

            // Simulate two-loop read pattern
            let k = s_vec.len();
            for i in (0..k).rev() {
                for j in 0..n {
                    checksum_vec += s_vec[i][j] * rho_vec[i];
                }
            }
            for i in 0..k {
                for j in 0..n {
                    checksum_vec += y_vec[i][j] * rho_vec[i];
                }
            }
        }
        let dt_vec = t0.elapsed();

        // --- RingHistory ---
        let t0 = std::time::Instant::now();
        let mut ring = RingHistory::new(m, n);
        let mut checksum_ring = 0.0f64;
        for (s, y, rho) in &entries {
            ring.push(s, y, *rho);

            // Same two-loop read pattern
            let k = ring.len();
            for i in (0..k).rev() {
                let si = ring.s(i);
                let ri = ring.rho(i);
                for j in 0..n {
                    checksum_ring += si[j] * ri;
                }
            }
            for i in 0..k {
                let yi = ring.y(i);
                let ri = ring.rho(i);
                for j in 0..n {
                    checksum_ring += yi[j] * ri;
                }
            }
        }
        let dt_ring = t0.elapsed();

        // Verify correctness (checksums must match)
        assert!(
            (checksum_vec - checksum_ring).abs() < 1e-6,
            "checksums diverge: vec={checksum_vec}, ring={checksum_ring}"
        );

        let speedup = dt_vec.as_nanos() as f64 / dt_ring.as_nanos() as f64;
        eprintln!("=== RingHistory vs Vec<Vec> micro-benchmark (n={n}, m={m}, iters={iters}) ===");
        eprintln!("  Vec<Vec>:    {:>10.3} ms", dt_vec.as_secs_f64() * 1000.0);
        eprintln!("  RingHistory: {:>10.3} ms", dt_ring.as_secs_f64() * 1000.0);
        eprintln!("  Speedup:     {:>10.2}x", speedup);
        eprintln!("  Checksum:    {checksum_ring:.6}");
    }

    #[test]
    fn begin_iter_converges_on_small_relative_obj_change() {
        let mut st = LbfgsState::new(vec![0.0], vec![(-10.0, 10.0)], 5, 1e-6);

        let dir0 = st.begin_iter(1000.0, &[1.0]).expect("first iter should be active");
        let x1 = st.propose_x(&dir0, 1e-8);
        st.accept_x(x1);

        let dir1 = st.begin_iter(999.9995, &[1.0]);
        assert!(dir1.is_none(), "second iter should stop on relative objective criterion");
        assert!(st.converged, "state must be marked converged");
        assert!(!st.failed, "state must not be marked failed");
    }

    #[test]
    fn compute_edm_empty_history_is_nan() {
        let st = LbfgsState::new(vec![1.0, 2.0], vec![(-10.0, 10.0), (-10.0, 10.0)], 5, 1e-6);
        let edm = st.compute_edm(&[0.1, -0.2]);
        assert!(edm.is_nan(), "EDM should be NAN with empty history");
    }

    #[test]
    fn compute_edm_with_history() {
        // Manually push history into an LbfgsState and verify EDM computation.
        let mut st = LbfgsState::new(vec![1.0, 2.0], vec![(-10.0, 10.0), (-10.0, 10.0)], 5, 1e-12);

        // Simulate two iterations to build history:
        // iter 0: x=[5,3], g=[5,3], f=17
        st.x = vec![5.0, 3.0];
        st.prev_x = None;
        st.prev_grad = None;

        // iter 1: x=[1,1], g=[1,1], f=1 → s=[−4,−2], y=[−4,−2], sy=20
        // Push (s, y) manually
        st.history.push(&[-4.0, -2.0], &[-4.0, -2.0], 1.0 / 20.0);

        let grad = [1.0, 1.0];
        let edm = st.compute_edm(&grad);
        // With one history pair: gamma = sy/yy = 20/20 = 1.0, H0 = I
        // After L-BFGS update, H^{-1} should approximate I for this simple case.
        // EDM = g^T H^{-1} g ≈ |g|^2 = 2.0
        assert!(edm >= 0.0, "EDM must be non-negative, got {edm}");
        assert!(edm.is_finite(), "EDM must be finite, got {edm}");
        assert!((edm - 2.0).abs() < 0.5, "EDM should be close to |g|^2=2.0, got {edm}");
    }

    #[test]
    fn begin_iter_keeps_running_when_obj_change_is_not_small() {
        let mut st = LbfgsState::new(vec![0.0], vec![(-10.0, 10.0)], 5, 1e-6);

        let dir0 = st.begin_iter(1000.0, &[1.0]).expect("first iter should be active");
        let x1 = st.propose_x(&dir0, 1e-8);
        st.accept_x(x1);

        let dir1 = st.begin_iter(999.0, &[1.0]);
        assert!(dir1.is_some(), "optimizer should continue when relative objective drop is large");
        assert!(!st.converged, "state must remain active");
    }
}
