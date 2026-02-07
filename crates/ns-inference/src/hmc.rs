//! Hamiltonian Monte Carlo (HMC) leapfrog integrator and static sampler.
//!
//! This module provides the core building blocks for HMC: the leapfrog
//! integrator and a static-trajectory HMC sampler (for validation).
//! The NUTS sampler in [`crate::nuts`] builds on top of this.

use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;

/// Euclidean metric for HMC/NUTS.
///
/// We store the *inverse* mass matrix (a.k.a. precision) because that's what
/// leapfrog needs for the velocity `dq/dt = M^{-1} p` and for the kinetic
/// energy `K = 0.5 * p^T M^{-1} p`.
///
/// For the dense case we store the Cholesky factor `L` of the inverse mass
/// matrix: `M^{-1} = L L^T`, which lets us:
/// - multiply `M^{-1} p` efficiently via two triangular multiplies
/// - sample `p ~ N(0, M)` via solving `L^T p = z`, `z ~ N(0, I)`
#[derive(Debug, Clone)]
pub enum Metric {
    /// Diagonal inverse mass matrix.
    Diag(Vec<f64>),
    /// Dense inverse mass matrix stored as Cholesky factor `L` (lower-triangular),
    /// such that `inv_mass = L L^T`.
    DenseCholesky { dim: usize, l: Vec<f64> },
}

impl Metric {
    pub fn identity(dim: usize) -> Self {
        Self::Diag(vec![1.0; dim])
    }

    pub fn dim(&self) -> usize {
        match self {
            Metric::Diag(v) => v.len(),
            Metric::DenseCholesky { dim, .. } => *dim,
        }
    }

    #[inline]
    fn l_at(l: &[f64], dim: usize, i: usize, j: usize) -> f64 {
        l[i * dim + j]
    }

    /// Multiply by inverse mass: `v = M^{-1} p`.
    pub fn mul_inv_mass(&self, p: &[f64]) -> Vec<f64> {
        match self {
            Metric::Diag(inv_mass_diag) => inv_mass_diag.iter().zip(p.iter()).map(|(&m, &pi)| m * pi).collect(),
            Metric::DenseCholesky { dim, l } => {
                let n = *dim;
                debug_assert_eq!(p.len(), n);

                // t = L^T p  (upper triangular)
                let mut t = vec![0.0; n];
                for i in 0..n {
                    // t[i] = sum_{k=i..n-1} L[k,i] * p[k]
                    let mut acc = 0.0;
                    for k in i..n {
                        acc += Self::l_at(l, n, k, i) * p[k];
                    }
                    t[i] = acc;
                }

                // v = L t (lower triangular)
                let mut v = vec![0.0; n];
                for i in 0..n {
                    let mut acc = 0.0;
                    for k in 0..=i {
                        acc += Self::l_at(l, n, i, k) * t[k];
                    }
                    v[i] = acc;
                }
                v
            }
        }
    }

    pub fn kinetic_energy(&self, p: &[f64]) -> f64 {
        let v = self.mul_inv_mass(p);
        0.5 * p.iter().zip(v.iter()).map(|(&pi, &vi)| pi * vi).sum::<f64>()
    }

    pub fn sample_momentum(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        match self {
            Metric::Diag(inv_mass_diag) => {
                let mut p = vec![0.0; inv_mass_diag.len()];
                for i in 0..p.len() {
                    let inv_m = inv_mass_diag[i];
                    let sigma = if inv_m > 0.0 { (1.0 / inv_m).sqrt() } else { 1.0 };
                    p[i] = sigma * normal.sample(rng);
                }
                p
            }
            Metric::DenseCholesky { dim, l } => {
                let n = *dim;
                let mut z = vec![0.0; n];
                for i in 0..n {
                    z[i] = normal.sample(rng);
                }

                // Solve L^T p = z for p (back-substitution).
                let mut p = vec![0.0; n];
                for i_rev in 0..n {
                    let i = n - 1 - i_rev;
                    let mut rhs = z[i];
                    for k in (i + 1)..n {
                        rhs -= Self::l_at(l, n, k, i) * p[k];
                    }
                    let diag = Self::l_at(l, n, i, i);
                    // Defensive: if diag is non-finite or ~0, fall back to standard normal.
                    if !diag.is_finite() || diag.abs() < 1e-14 {
                        p[i] = z[i];
                    } else {
                        p[i] = rhs / diag;
                    }
                }
                p
            }
        }
    }

    /// Mass matrix diagonal (for reporting): `diag(M)`.
    pub fn mass_diag(&self) -> Vec<f64> {
        match self {
            Metric::Diag(inv_mass_diag) => inv_mass_diag.iter().map(|&q| if q > 0.0 { 1.0 / q } else { 1.0 }).collect(),
            Metric::DenseCholesky { dim, l } => {
                // We have Q = M^{-1} = L L^T. We need diag(M) = diag(Q^{-1}).
                // Compute diag(Q^{-1}) by solving Q x = e_i and taking x_i.
                let n = *dim;
                let mut out = vec![0.0; n];

                for i in 0..n {
                    // Forward solve: L y = e_i
                    let mut y = vec![0.0; n];
                    for r in 0..n {
                        let mut rhs = if r == i { 1.0 } else { 0.0 };
                        for c in 0..r {
                            rhs -= Self::l_at(l, n, r, c) * y[c];
                        }
                        let diag = Self::l_at(l, n, r, r);
                        y[r] = if !diag.is_finite() || diag.abs() < 1e-14 { 0.0 } else { rhs / diag };
                    }

                    // Backward solve: L^T x = y
                    let mut x = vec![0.0; n];
                    for r_rev in 0..n {
                        let r = n - 1 - r_rev;
                        let mut rhs = y[r];
                        for c in (r + 1)..n {
                            rhs -= Self::l_at(l, n, c, r) * x[c];
                        }
                        let diag = Self::l_at(l, n, r, r);
                        x[r] = if !diag.is_finite() || diag.abs() < 1e-14 { 0.0 } else { rhs / diag };
                    }

                    out[i] = x[i].max(1e-12);
                }

                out
            }
        }
    }
}

#[inline]
fn metropolis_accept(log_accept: f64, u: f64) -> bool {
    // Standard Metropolis criterion:
    // accept with prob min(1, exp(log_accept)).
    //
    // With `log_accept = H_current - H_proposal = -ΔH`, this rejects high-energy (lower-probability)
    // proposals and always accepts downhill moves (ΔH <= 0).
    debug_assert!(u > 0.0 && u < 1.0);
    u.ln() < log_accept
}

/// HMC phase-space state: position + momentum + cached potential/gradient.
#[derive(Debug, Clone)]
pub struct HmcState {
    /// Position in unconstrained space.
    pub q: Vec<f64>,
    /// Momentum.
    pub p: Vec<f64>,
    /// Potential energy: `-logpdf_unconstrained(q)`.
    pub potential: f64,
    /// Gradient of potential: `-grad_unconstrained(q)`.
    pub grad_potential: Vec<f64>,
}

impl HmcState {
    /// Kinetic energy: `0.5 * p^T * M^{-1} * p`.
    pub fn kinetic_energy(&self, metric: &Metric) -> f64 {
        metric.kinetic_energy(&self.p)
    }

    /// Total Hamiltonian: `H = U(q) + K(p)`.
    pub fn hamiltonian(&self, metric: &Metric) -> f64 {
        self.potential + self.kinetic_energy(metric)
    }
}

/// Leapfrog integrator for HMC.
pub struct LeapfrogIntegrator<'a, 'b, M: LogDensityModel + ?Sized> {
    posterior: &'a Posterior<'b, M>,
    step_size: f64,
    metric: Metric,
}

impl<'a, 'b, M: LogDensityModel + ?Sized> LeapfrogIntegrator<'a, 'b, M> {
    /// Create a new leapfrog integrator.
    pub fn new(posterior: &'a Posterior<'b, M>, step_size: f64, metric: Metric) -> Self {
        Self { posterior, step_size, metric }
    }

    /// Update step size (used during adaptation).
    pub fn set_step_size(&mut self, eps: f64) {
        self.step_size = eps;
    }

    /// Update inverse mass matrix diagonal (used during adaptation).
    pub fn set_metric(&mut self, metric: Metric) {
        self.metric = metric;
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Current inverse mass diagonal.
    pub fn metric(&self) -> &Metric {
        &self.metric
    }

    /// Initialize an HMC state at position `q`.
    pub fn init_state(&self, q: Vec<f64>) -> Result<HmcState> {
        let lp = self.posterior.logpdf_unconstrained(&q)?;
        let grad_lp = self.posterior.grad_unconstrained(&q)?;
        let potential = -lp;
        let grad_potential: Vec<f64> = grad_lp.iter().map(|&g| -g).collect();
        Ok(HmcState { q, p: vec![0.0; grad_potential.len()], potential, grad_potential })
    }

    /// Single leapfrog step: `(q, p, grad) -> (q', p', grad')`.
    ///
    /// Returns `Err` if gradient evaluation fails (e.g., NaN parameters).
    pub fn step(&self, state: &mut HmcState) -> Result<()> {
        self.step_with_eps(state, self.step_size)
    }

    /// Single leapfrog step with explicit step size (used for backward integration in NUTS).
    pub fn step_with_eps(&self, state: &mut HmcState, eps: f64) -> Result<()> {
        let n = state.q.len();

        // Half-step momentum
        for i in 0..n {
            state.p[i] -= 0.5 * eps * state.grad_potential[i];
        }

        // Full-step position
        let v = self.metric.mul_inv_mass(&state.p);
        for i in 0..n {
            state.q[i] += eps * v[i];
        }

        // Recompute potential and gradient at new position
        let lp = self.posterior.logpdf_unconstrained(&state.q)?;
        let grad_lp = self.posterior.grad_unconstrained(&state.q)?;
        state.potential = -lp;
        for i in 0..n {
            state.grad_potential[i] = -grad_lp[i];
        }

        // Half-step momentum
        for i in 0..n {
            state.p[i] -= 0.5 * eps * state.grad_potential[i];
        }

        Ok(())
    }

    /// Take one leapfrog step in the given direction (`+1` forward, `-1` backward).
    pub fn step_dir(&self, state: &mut HmcState, direction: i32) -> Result<()> {
        debug_assert!(direction == 1 || direction == -1);
        self.step_with_eps(state, self.step_size * (direction as f64))
    }

    /// Full trajectory: `n_steps` leapfrog steps.
    pub fn integrate(&self, mut state: HmcState, n_steps: usize) -> Result<HmcState> {
        for _ in 0..n_steps {
            self.step(&mut state)?;
        }
        Ok(state)
    }
}

/// Static HMC sampler (fixed trajectory length). Used for validation.
pub struct StaticHmcSampler<'a, 'b, M: LogDensityModel + ?Sized> {
    integrator: LeapfrogIntegrator<'a, 'b, M>,
    n_steps: usize,
}

impl<'a, 'b, M: LogDensityModel + ?Sized> StaticHmcSampler<'a, 'b, M> {
    /// Create a new static HMC sampler.
    pub fn new(
        posterior: &'a Posterior<'b, M>,
        step_size: f64,
        n_steps: usize,
        metric: Metric,
    ) -> Self {
        let integrator = LeapfrogIntegrator::new(posterior, step_size, metric);
        Self { integrator, n_steps }
    }

    /// Propose and accept/reject one HMC step.
    ///
    /// Returns `(new_state, accepted)`.
    pub fn step(&self, current: &HmcState, rng: &mut impl rand::Rng) -> Result<(HmcState, bool)> {
        let metric = self.integrator.metric();

        // Sample momentum ~ N(0, M)
        let mut proposal = current.clone();
        proposal.p = metric.sample_momentum(rng);

        let h_current = proposal.hamiltonian(metric);

        // Integrate
        let proposal = self.integrator.integrate(proposal, self.n_steps)?;
        let h_proposal = proposal.hamiltonian(metric);

        // Metropolis accept/reject
        let log_accept = h_current - h_proposal;
        let u: f64 = rng.random();
        let accepted = metropolis_accept(log_accept, u);

        if accepted { Ok((proposal, true)) } else { Ok((current.clone(), false)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::Posterior;
    use ns_translate::pyhf::{HistFactoryModel, Workspace};
    use rand::SeedableRng;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_leapfrog_energy_conservation() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let n = model.n_params();
        let metric = Metric::identity(n);
        let eps = 0.001; // very small step for good energy conservation

        let integrator = LeapfrogIntegrator::new(&posterior, eps, metric.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();

        let mut state = integrator.init_state(z_init).unwrap();
        // Set non-zero momentum
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        use rand_distr::Distribution;
        for i in 0..n {
            state.p[i] = normal.sample(&mut rng);
        }

        let h_initial = state.hamiltonian(&metric);

        // Take 100 leapfrog steps
        let state = integrator.integrate(state, 100).unwrap();
        let h_final = state.hamiltonian(&metric);

        let dh = (h_final - h_initial).abs();
        assert!(
            dh < 0.1,
            "Energy not conserved: H_init={}, H_final={}, dH={}",
            h_initial,
            h_final,
            dh
        );
    }

    #[test]
    fn test_static_hmc_deterministic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let n = model.n_params();
        let metric = Metric::identity(n);
        let sampler = StaticHmcSampler::new(&posterior, 0.1, 10, metric);

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let state = sampler.integrator.init_state(z_init).unwrap();

        // Run with same seed twice
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
        let (s1, a1) = sampler.step(&state, &mut rng1).unwrap();

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let (s2, a2) = sampler.step(&state, &mut rng2).unwrap();

        assert_eq!(a1, a2, "Acceptance should be deterministic");
        assert_eq!(s1.q, s2.q, "Samples should be deterministic");
    }

    #[test]
    fn test_static_hmc_samples_reasonable() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let n = model.n_params();
        let metric = Metric::identity(n);
        let sampler = StaticHmcSampler::new(&posterior, 0.05, 20, metric);

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let mut state = sampler.integrator.init_state(z_init).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n_samples = 200;
        let burn = 50;
        let mut accepted = 0;
        let mut poi_samples = Vec::new();

        for i in 0..n_samples {
            let (new_state, acc) = sampler.step(&state, &mut rng).unwrap();
            state = new_state;
            if acc {
                accepted += 1;
            }
            if i >= burn {
                let theta = posterior.to_constrained(&state.q).unwrap();
                poi_samples.push(theta[0]);
            }
        }

        let accept_rate = accepted as f64 / n_samples as f64;
        assert!(accept_rate > 0.1, "Acceptance rate too low: {}", accept_rate);

        // POI mean should be in reasonable range
        let poi_mean: f64 = poi_samples.iter().sum::<f64>() / poi_samples.len() as f64;
        assert!(poi_mean > 0.0 && poi_mean < 3.0, "POI mean out of range: {}", poi_mean);
    }

    #[test]
    fn test_metropolis_accept_contract() {
        // log_accept = -ΔH
        // - If ΔH <= 0 => log_accept >= 0 => always accept for u in (0,1)
        assert!(metropolis_accept(0.0, 0.5));
        assert!(metropolis_accept(1.0, 0.999999));

        // - If ΔH = 1 => log_accept = -1 => accept iff u < exp(-1) ~= 0.3679
        assert!(metropolis_accept(-1.0, 0.1));
        assert!(!metropolis_accept(-1.0, 0.5));
    }
}
