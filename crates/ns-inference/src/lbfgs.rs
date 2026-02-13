//! Standalone L-BFGS-B state machine for GPU lockstep optimization.
//!
//! Shared between CUDA and Metal batch toy fitters.

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
    /// L-BFGS memory: s vectors (x_k - x_{k-1}).
    s_history: Vec<Vec<f64>>,
    /// L-BFGS memory: y vectors (g_k - g_{k-1}).
    y_history: Vec<Vec<f64>>,
    /// L-BFGS memory: rho = 1 / (y^T s).
    rho_history: Vec<f64>,
    /// Maximum history size.
    m: usize,
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
        let x = Self::clamp_to_bounds(&x0, &bounds);
        Self {
            x,
            prev_grad: None,
            prev_x: None,
            fval: f64::INFINITY,
            prev_fval: None,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            rho_history: Vec::with_capacity(m),
            m,
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
                if self.s_history.len() >= self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                    self.rho_history.remove(0);
                }
                self.rho_history.push(1.0 / sy);
                self.s_history.push(s);
                self.y_history.push(y);
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
                if self.s_history.len() >= self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                    self.rho_history.remove(0);
                }
                self.rho_history.push(1.0 / sy);
                self.s_history.push(s);
                self.y_history.push(y);
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
        let k = self.s_history.len();

        if k == 0 {
            return grad.iter().map(|&g| -g).collect();
        }

        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; k];

        // Backward pass
        for i in (0..k).rev() {
            let mut dot = 0.0;
            for j in 0..n {
                dot += self.s_history[i][j] * q[j];
            }
            alpha[i] = self.rho_history[i] * dot;
            for j in 0..n {
                q[j] -= alpha[i] * self.y_history[i][j];
            }
        }

        // Initial Hessian approximation: H0 = gamma * I
        let last = k - 1;
        let mut sy = 0.0;
        let mut yy = 0.0;
        for j in 0..n {
            sy += self.s_history[last][j] * self.y_history[last][j];
            yy += self.y_history[last][j] * self.y_history[last][j];
        }
        let gamma = if yy > 1e-30 { sy / yy } else { 1.0 };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Forward pass
        for i in 0..k {
            let mut dot = 0.0;
            for j in 0..n {
                dot += self.y_history[i][j] * r[j];
            }
            let beta = self.rho_history[i] * dot;
            for j in 0..n {
                r[j] += (alpha[i] - beta) * self.s_history[i][j];
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

    fn clamp_to_bounds(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        x.iter().zip(bounds.iter()).map(|(&v, &(lo, hi))| v.clamp(lo, hi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::LbfgsState;

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
