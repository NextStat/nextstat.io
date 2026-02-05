//! Hamiltonian Monte Carlo (HMC) leapfrog integrator and static sampler.
//!
//! This module provides the core building blocks for HMC: the leapfrog
//! integrator and a static-trajectory HMC sampler (for validation).
//! The NUTS sampler in [`crate::nuts`] builds on top of this.

use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;

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
    pub fn kinetic_energy(&self, inv_mass_diag: &[f64]) -> f64 {
        self.p.iter().zip(inv_mass_diag.iter()).map(|(&pi, &mi)| pi * pi * mi).sum::<f64>() * 0.5
    }

    /// Total Hamiltonian: `H = U(q) + K(p)`.
    pub fn hamiltonian(&self, inv_mass_diag: &[f64]) -> f64 {
        self.potential + self.kinetic_energy(inv_mass_diag)
    }
}

/// Leapfrog integrator for HMC.
pub struct LeapfrogIntegrator<'a, 'b, M: LogDensityModel + ?Sized> {
    posterior: &'a Posterior<'b, M>,
    step_size: f64,
    inv_mass_diag: Vec<f64>,
}

impl<'a, 'b, M: LogDensityModel + ?Sized> LeapfrogIntegrator<'a, 'b, M> {
    /// Create a new leapfrog integrator.
    pub fn new(posterior: &'a Posterior<'b, M>, step_size: f64, inv_mass_diag: Vec<f64>) -> Self {
        Self { posterior, step_size, inv_mass_diag }
    }

    /// Update step size (used during adaptation).
    pub fn set_step_size(&mut self, eps: f64) {
        self.step_size = eps;
    }

    /// Update inverse mass matrix diagonal (used during adaptation).
    pub fn set_inv_mass_diag(&mut self, inv_mass: Vec<f64>) {
        self.inv_mass_diag = inv_mass;
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Current inverse mass diagonal.
    pub fn inv_mass_diag(&self) -> &[f64] {
        &self.inv_mass_diag
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
        for i in 0..n {
            state.q[i] += eps * self.inv_mass_diag[i] * state.p[i];
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
        inv_mass_diag: Vec<f64>,
    ) -> Self {
        let integrator = LeapfrogIntegrator::new(posterior, step_size, inv_mass_diag);
        Self { integrator, n_steps }
    }

    /// Propose and accept/reject one HMC step.
    ///
    /// Returns `(new_state, accepted)`.
    pub fn step(&self, current: &HmcState, rng: &mut impl rand::Rng) -> Result<(HmcState, bool)> {
        use rand_distr::{Distribution, Normal};

        let n = current.q.len();
        let inv_mass = self.integrator.inv_mass_diag();

        // Sample momentum ~ N(0, M)
        let mut proposal = current.clone();
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            let sigma = (1.0 / inv_mass[i]).sqrt();
            proposal.p[i] = sigma * normal.sample(rng);
        }

        let h_current = proposal.hamiltonian(inv_mass);

        // Integrate
        let proposal = self.integrator.integrate(proposal, self.n_steps)?;
        let h_proposal = proposal.hamiltonian(inv_mass);

        // Metropolis accept/reject
        let log_accept = h_current - h_proposal;
        let u: f64 = rng.random();
        let accepted = u.ln() < log_accept;

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
        let inv_mass = vec![1.0; n];
        let eps = 0.001; // very small step for good energy conservation

        let integrator = LeapfrogIntegrator::new(&posterior, eps, inv_mass.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init);

        let mut state = integrator.init_state(z_init).unwrap();
        // Set non-zero momentum
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        use rand_distr::Distribution;
        for i in 0..n {
            state.p[i] = normal.sample(&mut rng);
        }

        let h_initial = state.hamiltonian(&inv_mass);

        // Take 100 leapfrog steps
        let state = integrator.integrate(state, 100).unwrap();
        let h_final = state.hamiltonian(&inv_mass);

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
        let inv_mass = vec![1.0; n];
        let sampler = StaticHmcSampler::new(&posterior, 0.1, 10, inv_mass.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init);
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
        let inv_mass = vec![1.0; n];
        let sampler = StaticHmcSampler::new(&posterior, 0.05, 20, inv_mass.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init);
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
                let theta = posterior.to_constrained(&state.q);
                poi_samples.push(theta[0]);
            }
        }

        let accept_rate = accepted as f64 / n_samples as f64;
        assert!(accept_rate > 0.1, "Acceptance rate too low: {}", accept_rate);

        // POI mean should be in reasonable range
        let poi_mean: f64 = poi_samples.iter().sum::<f64>() / poi_samples.len() as f64;
        assert!(poi_mean > 0.0 && poi_mean < 3.0, "POI mean out of range: {}", poi_mean);
    }
}
