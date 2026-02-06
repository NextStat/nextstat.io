//! Adaptation for NUTS: step size (dual averaging) and mass matrix (Welford variance).
//!
//! Implements the Stan warmup schedule: windowed adaptation with step size
//! tuning and diagonal mass matrix estimation.

/// Dual averaging for step size adaptation (Nesterov 2009, Stan variant).
///
/// Adapts `epsilon` to achieve a target average acceptance probability.
pub struct DualAveraging {
    target_accept: f64,
    log_eps: f64,
    log_eps_bar: f64,
    h_bar: f64,
    mu: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    step: usize,
}

impl DualAveraging {
    /// Create with target acceptance rate and initial step size.
    pub fn new(target_accept: f64, init_eps: f64) -> Self {
        // Initialize the smoothed step size to the same value as the current step size.
        // Starting from 1.0 (log_eps_bar=0) can badly distort short warmup runs and
        // makes multi-chain diagnostics (R-hat) flaky.
        let log_eps0 = init_eps.ln();
        Self {
            target_accept,
            log_eps: log_eps0,
            log_eps_bar: log_eps0,
            h_bar: 0.0,
            mu: (10.0 * init_eps).ln(),
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            step: 0,
        }
    }

    /// Update with observed acceptance probability from one transition.
    pub fn update(&mut self, accept_prob: f64) {
        self.step += 1;
        let m = self.step as f64;
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - accept_prob);

        self.log_eps = self.mu - (m.sqrt() / self.gamma) * self.h_bar;
        let m_kappa = m.powf(-self.kappa);
        self.log_eps_bar = m_kappa * self.log_eps + (1.0 - m_kappa) * self.log_eps_bar;
    }

    /// Current step size (during warmup).
    pub fn current_step_size(&self) -> f64 {
        self.log_eps.exp()
    }

    /// Final adapted step size (after warmup — use the smoothed version).
    pub fn adapted_step_size(&self) -> f64 {
        self.log_eps_bar.exp()
    }

    /// Reset internal state for a new adaptation window, keeping the current step size.
    pub fn reset(&mut self, init_eps: f64) {
        self.log_eps = init_eps.ln();
        self.log_eps_bar = init_eps.ln();
        self.h_bar = 0.0;
        self.mu = (10.0 * init_eps).ln();
        self.step = 0;
    }
}

/// Online Welford variance estimator (diagonal mass matrix).
pub struct WelfordVariance {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: usize,
}

impl WelfordVariance {
    /// Create for `dim`-dimensional parameter vector.
    pub fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0 }
    }

    /// Incorporate a new sample.
    pub fn update(&mut self, x: &[f64]) {
        self.count += 1;
        let n = self.count as f64;
        for i in 0..x.len() {
            let delta = x[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Current variance estimate. Returns `1.0` for each dimension if `count < 2`.
    pub fn variance(&self) -> Vec<f64> {
        if self.count < 2 {
            return vec![1.0; self.mean.len()];
        }
        let n = self.count as f64;
        self.m2.iter().map(|&m| (m / (n - 1.0)).max(1e-10)).collect()
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        for v in &mut self.mean {
            *v = 0.0;
        }
        for v in &mut self.m2 {
            *v = 0.0;
        }
        self.count = 0;
    }
}

/// Windowed adaptation combining step size + mass matrix tuning.
///
/// Follows a simplified Stan schedule:
/// ```text
/// n_warmup = 1000 (example):
///   Window 0: iters 0..75     — fast adapt (step size only)
///   Window 1: iters 75..175   — slow (step size + mass at end)
///   Window 2: iters 175..375  — slow (step size + mass at end)
///   Window 3: iters 375..775  — slow (step size + mass at end)
///   Window 4: iters 775..1000 — final (step size only, lock mass)
/// ```
pub struct WindowedAdaptation {
    dual_avg: DualAveraging,
    welford: WelfordVariance,
    windows: Vec<(usize, usize)>,
    current_window: usize,
    inv_mass_diag: Vec<f64>,
}

impl WindowedAdaptation {
    /// Create windowed adaptation for given dimension and warmup length.
    pub fn new(dim: usize, n_warmup: usize, target_accept: f64, init_eps: f64) -> Self {
        let windows = compute_windows(n_warmup);
        Self {
            dual_avg: DualAveraging::new(target_accept, init_eps),
            welford: WelfordVariance::new(dim),
            windows,
            current_window: 0,
            inv_mass_diag: vec![1.0; dim],
        }
    }

    /// Update adaptation with a new sample and its acceptance probability.
    ///
    /// Returns `true` if mass matrix was updated (at window boundary).
    pub fn update(&mut self, iter: usize, q: &[f64], accept_prob: f64) -> bool {
        // Always update step size
        self.dual_avg.update(accept_prob);

        let mut mass_updated = false;

        if self.current_window < self.windows.len() {
            let (_start, end) = self.windows[self.current_window];

            // Collect samples in slow windows (skip first and last)
            let is_slow_window =
                self.current_window > 0 && self.current_window < self.windows.len() - 1;
            if is_slow_window {
                self.welford.update(q);
            }

            // At window boundary, update mass and restart dual averaging
            if iter + 1 >= end {
                if is_slow_window {
                    let var = self.welford.variance();
                    self.inv_mass_diag = var.iter().map(|&v| 1.0 / v).collect();
                    self.welford.reset();
                    mass_updated = true;
                }

                // Restart dual averaging with current adapted step size
                let eps = self.dual_avg.adapted_step_size();
                self.dual_avg.reset(eps);
                self.current_window += 1;
            }
        }

        mass_updated
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.dual_avg.current_step_size()
    }

    /// Final adapted step size (smoothed).
    pub fn adapted_step_size(&self) -> f64 {
        self.dual_avg.adapted_step_size()
    }

    /// Current inverse mass diagonal.
    pub fn inv_mass_diag(&self) -> &[f64] {
        &self.inv_mass_diag
    }
}

/// Compute Stan-style adaptation windows.
fn compute_windows(n_warmup: usize) -> Vec<(usize, usize)> {
    if n_warmup < 20 {
        return vec![(0, n_warmup)];
    }

    let init_buffer = 75.min(n_warmup / 5);
    let term_buffer = 50.min(n_warmup / 5);
    let slow_size = n_warmup.saturating_sub(init_buffer + term_buffer);

    let mut windows = Vec::new();

    // Window 0: initial fast adaptation
    windows.push((0, init_buffer));

    if slow_size > 0 {
        // Doubling slow windows
        let mut start = init_buffer;
        let mut size = slow_size.min(25).max(1);
        while start + size < init_buffer + slow_size {
            let end = (start + size).min(init_buffer + slow_size);
            windows.push((start, end));
            start = end;
            size *= 2;
        }
        // Last slow window extends to start of term buffer
        if start < init_buffer + slow_size {
            windows.push((start, init_buffer + slow_size));
        }
    }

    // Final window: term buffer
    if term_buffer > 0 {
        windows.push((init_buffer + slow_size, n_warmup));
    }

    windows
}

/// Find a reasonable initial step size by testing energy error.
///
/// Doubles or halves `eps` until the acceptance probability crosses 0.5.
/// Follows Stan's algorithm (Hoffman & Gelman 2014, Algorithm 4).
pub fn find_reasonable_step_size(
    posterior: &crate::posterior::Posterior<'_, impl ns_core::traits::LogDensityModel + ?Sized>,
    q: &[f64],
    inv_mass_diag: &[f64],
) -> f64 {
    let integrator = crate::hmc::LeapfrogIntegrator::new(posterior, 1.0, inv_mass_diag.to_vec());

    let mut state = match integrator.init_state(q.to_vec()) {
        Ok(s) => s,
        Err(_) => return 0.01,
    };

    // Set unit momentum
    for p in &mut state.p {
        *p = 1.0;
    }
    let h0 = state.hamiltonian(inv_mass_diag);

    // Test a single leapfrog step at given eps
    let test_accept = |eps_test: f64| -> Option<f64> {
        let int_test =
            crate::hmc::LeapfrogIntegrator::new(posterior, eps_test, inv_mass_diag.to_vec());
        let mut s = state.clone();
        int_test.step(&mut s).ok()?;
        let h1 = s.hamiltonian(inv_mass_diag);
        let a = (h0 - h1).exp();
        if a.is_finite() { Some(a.min(1.0)) } else { None }
    };

    // Start from 0.1 and find where accept prob crosses 0.5
    let mut eps = 0.1;
    let accept0 = match test_accept(eps) {
        Some(a) => a,
        None => {
            // Try smaller
            eps = 0.001;
            match test_accept(eps) {
                Some(a) => a,
                None => return 0.001,
            }
        }
    };

    let direction: f64 = if accept0 > 0.5 { 1.0 } else { -1.0 };

    for _ in 0..50 {
        let new_eps = eps * 2.0_f64.powf(direction);
        if new_eps > 1e3 || new_eps < 1e-10 {
            break;
        }

        match test_accept(new_eps) {
            Some(a) => {
                if direction > 0.0 && a < 0.5 {
                    break;
                }
                if direction < 0.0 && a > 0.5 {
                    break;
                }
                eps = new_eps;
            }
            None => break,
        }
    }

    eps.clamp(1e-8, 1e3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_averaging_converges() {
        let mut da = DualAveraging::new(0.8, 1.0);
        // Feed constant accept_prob = 0.8 => should stabilize
        for _ in 0..100 {
            da.update(0.8);
        }
        let eps = da.adapted_step_size();
        assert!(eps > 0.0 && eps.is_finite(), "Step size should be positive finite: {}", eps);
    }

    #[test]
    fn test_dual_averaging_adapts_direction() {
        // If accept prob is always too high (> target), eps should increase
        let mut da_high = DualAveraging::new(0.8, 0.01);
        for _ in 0..200 {
            da_high.update(0.99);
        }
        let eps_high = da_high.adapted_step_size();

        // If accept prob is always too low, eps should decrease
        let mut da_low = DualAveraging::new(0.8, 1.0);
        for _ in 0..200 {
            da_low.update(0.1);
        }
        let eps_low = da_low.adapted_step_size();

        assert!(eps_high > eps_low, "High accept => larger step: {} vs {}", eps_high, eps_low);
    }

    #[test]
    fn test_welford_variance() {
        let mut w = WelfordVariance::new(2);
        // Known data: [1, 2, 3, 4, 5] and [10, 20, 30, 40, 50]
        let data = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]];
        for d in &data {
            w.update(d);
        }
        let var = w.variance();
        // Var([1,2,3,4,5]) = 2.5, Var([10,20,30,40,50]) = 250
        assert!((var[0] - 2.5).abs() < 1e-10, "Variance[0] should be 2.5: {}", var[0]);
        assert!((var[1] - 250.0).abs() < 1e-10, "Variance[1] should be 250: {}", var[1]);
    }

    #[test]
    fn test_welford_reset() {
        let mut w = WelfordVariance::new(2);
        w.update(&[1.0, 2.0]);
        w.update(&[3.0, 4.0]);
        w.reset();
        let var = w.variance();
        assert_eq!(var, vec![1.0, 1.0], "After reset, variance should default to 1.0");
    }

    #[test]
    fn test_compute_windows() {
        let windows = compute_windows(1000);
        // Should have at least 3 windows (fast, slow..., terminal)
        assert!(windows.len() >= 3, "Should have multiple windows: {:?}", windows);

        // First window starts at 0
        assert_eq!(windows[0].0, 0);

        // Last window ends at 1000
        assert_eq!(windows.last().unwrap().1, 1000);

        // Windows should be contiguous
        for i in 1..windows.len() {
            assert_eq!(windows[i].0, windows[i - 1].1, "Windows not contiguous at {}", i);
        }
    }

    #[test]
    fn test_compute_windows_small() {
        let windows = compute_windows(10);
        assert!(!windows.is_empty());
        assert_eq!(windows.last().unwrap().1, 10);
    }
}
