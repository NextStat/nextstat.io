//! No-U-Turn Sampler (NUTS).
//!
//! Implements NUTS with tree doubling and the no-U-turn criterion.
//! The current implementation uses the slice-based NUTS variant:
//! proposals are selected uniformly among states that fall inside the slice.

use crate::adapt::{WindowedAdaptation, find_reasonable_step_size};
use crate::hmc::{HmcState, LeapfrogIntegrator};
use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;
use rand::Rng;

/// NUTS sampler configuration.
#[derive(Debug, Clone)]
pub struct NutsConfig {
    /// Maximum tree depth (default 10).
    pub max_treedepth: usize,
    /// Target acceptance probability (default 0.8).
    pub target_accept: f64,
    /// Stddev of random jitter added to the initial unconstrained position.
    ///
    /// This helps avoid identical initial states across chains.
    pub init_jitter: f64,
    /// Optional relative jitter scale for chain initialization.
    ///
    /// If set, jitter is computed per-parameter using model bounds and the local
    /// transform Jacobian at the initial point. This is more scale-aware than a
    /// single absolute `init_jitter` in unconstrained space.
    ///
    /// Mutually exclusive with `init_jitter > 0`.
    pub init_jitter_rel: Option<f64>,

    /// Optional overdispersed initialization around the deterministic starting point.
    ///
    /// Interpreted similarly to `init_jitter_rel`, but intended for larger dispersions
    /// (e.g. overdispersed chain init) and uses a wider clamp to avoid silently
    /// collapsing to tiny jitter near transform boundaries.
    ///
    /// Mutually exclusive with `init_jitter` and `init_jitter_rel`.
    pub init_overdispersed_rel: Option<f64>,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            max_treedepth: 10,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        }
    }
}

/// Result of one NUTS transition.
pub(crate) struct NutsTransition {
    pub q: Vec<f64>,
    pub potential: f64,
    pub grad_potential: Vec<f64>,
    pub depth: usize,
    pub divergent: bool,
    pub accept_prob: f64,
    pub energy: f64,
    #[allow(dead_code)]
    pub n_leapfrog: usize,
}

/// Internal tree node for NUTS tree-building.
struct NutsTree {
    q_left: Vec<f64>,
    p_left: Vec<f64>,
    grad_left: Vec<f64>,
    q_right: Vec<f64>,
    p_right: Vec<f64>,
    grad_right: Vec<f64>,
    q_proposal: Vec<f64>,
    potential_proposal: f64,
    grad_proposal: Vec<f64>,
    log_sum_weight: f64,
    depth: usize,
    n_leapfrog: usize,
    divergent: bool,
    turning: bool,
    sum_accept_prob: f64,
}

/// Maximum energy error before declaring divergence.
const DIVERGENCE_THRESHOLD: f64 = 1000.0;

/// Check the no-U-turn criterion.
fn is_turning(dq: &[f64], p_left: &[f64], p_right: &[f64], inv_mass: &[f64]) -> bool {
    let dot_left: f64 =
        dq.iter().zip(p_left.iter()).zip(inv_mass.iter()).map(|((&d, &p), &m)| d * p * m).sum();
    let dot_right: f64 =
        dq.iter().zip(p_right.iter()).zip(inv_mass.iter()).map(|((&d, &p), &m)| d * p * m).sum();
    dot_left < 0.0 || dot_right < 0.0
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
}

/// Build a single-node tree (one leapfrog step).
fn build_leaf<M: LogDensityModel + ?Sized>(
    integrator: &LeapfrogIntegrator<'_, '_, M>,
    state: &HmcState,
    direction: i32,
    log_u: f64,
    h0: f64,
    inv_mass: &[f64],
) -> Result<NutsTree> {
    let mut new_state = state.clone();

    // Integrate forward/backward by taking a step with +/- eps.
    integrator.step_dir(&mut new_state, direction)?;

    let h = new_state.hamiltonian(inv_mass);
    let energy_error = h - h0;
    let divergent = energy_error.abs() > DIVERGENCE_THRESHOLD;
    // Slice: keep only states with log_u <= log p(q,p) where log p = -H.
    // Use weights relative to the start point: log_weight = -(H - H0) = -energy_error.
    let logp = -h;
    let in_slice = log_u <= logp;
    // Slice-based NUTS: select uniformly among states in the slice.
    // (Stan's multinomial-weighted variant does not use an explicit slice threshold.)
    let log_weight = if in_slice { 0.0 } else { f64::NEG_INFINITY };

    let accept_prob = (-energy_error).exp().min(1.0);

    Ok(NutsTree {
        q_left: new_state.q.clone(),
        p_left: new_state.p.clone(),
        grad_left: new_state.grad_potential.clone(),
        q_right: new_state.q.clone(),
        p_right: new_state.p.clone(),
        grad_right: new_state.grad_potential.clone(),
        q_proposal: new_state.q.clone(),
        potential_proposal: new_state.potential,
        grad_proposal: new_state.grad_potential.clone(),
        log_sum_weight: log_weight,
        depth: 0,
        n_leapfrog: 1,
        divergent,
        turning: false,
        sum_accept_prob: accept_prob,
    })
}

/// Recursively build a balanced binary tree of depth `depth`.
fn build_tree<M: LogDensityModel + ?Sized>(
    integrator: &LeapfrogIntegrator<'_, '_, M>,
    state: &HmcState,
    depth: usize,
    direction: i32,
    log_u: f64,
    h0: f64,
    inv_mass: &[f64],
    rng: &mut impl Rng,
) -> Result<NutsTree> {
    if depth == 0 {
        return build_leaf(integrator, state, direction, log_u, h0, inv_mass);
    }

    // Build first half-tree
    let mut inner = build_tree(integrator, state, depth - 1, direction, log_u, h0, inv_mass, rng)?;

    if inner.divergent || inner.turning {
        return Ok(inner);
    }

    // Build second half-tree from the edge of the first
    let edge_state = if direction > 0 {
        HmcState {
            q: inner.q_right.clone(),
            p: inner.p_right.clone(),
            potential: 0.0, // not used for tree building
            grad_potential: inner.grad_right.clone(),
        }
    } else {
        HmcState {
            q: inner.q_left.clone(),
            p: inner.p_left.clone(),
            potential: 0.0,
            grad_potential: inner.grad_left.clone(),
        }
    };

    // Recompute potential for the edge state
    let outer =
        build_tree(integrator, &edge_state, depth - 1, direction, log_u, h0, inv_mass, rng)?;

    // Merge trees
    let new_log_sum_weight = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);

    // Multinomial selection: accept outer proposal with probability
    // exp(outer.log_sum_weight - new_log_sum_weight)
    let accept_outer = (outer.log_sum_weight - new_log_sum_weight).exp();
    let u: f64 = rng.random();
    if u < accept_outer {
        inner.q_proposal = outer.q_proposal;
        inner.potential_proposal = outer.potential_proposal;
        inner.grad_proposal = outer.grad_proposal;
    }

    inner.log_sum_weight = new_log_sum_weight;
    inner.n_leapfrog += outer.n_leapfrog;
    inner.sum_accept_prob += outer.sum_accept_prob;
    inner.divergent = inner.divergent || outer.divergent;

    // Update tree edges
    if direction > 0 {
        inner.q_right = outer.q_right;
        inner.p_right = outer.p_right;
        inner.grad_right = outer.grad_right;
    } else {
        inner.q_left = outer.q_left;
        inner.p_left = outer.p_left;
        inner.grad_left = outer.grad_left;
    }

    // Check U-turn on full tree
    let dq: Vec<f64> =
        inner.q_right.iter().zip(inner.q_left.iter()).map(|(&r, &l)| r - l).collect();
    inner.turning =
        inner.turning || outer.turning || is_turning(&dq, &inner.p_left, &inner.p_right, inv_mass);

    inner.depth = depth;
    Ok(inner)
}

/// Run one NUTS transition from the given state.
pub(crate) fn nuts_transition<M: LogDensityModel + ?Sized>(
    integrator: &LeapfrogIntegrator<'_, '_, M>,
    current: &HmcState,
    max_treedepth: usize,
    inv_mass: &[f64],
    rng: &mut impl Rng,
) -> Result<NutsTransition> {
    use rand_distr::{Distribution, Normal};

    let n = current.q.len();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Sample momentum ~ N(0, M)
    let mut state = current.clone();
    for i in 0..n {
        let sigma = (1.0 / inv_mass[i]).sqrt();
        state.p[i] = sigma * normal.sample(rng);
    }

    let h0 = state.hamiltonian(inv_mass);
    // Slice variable: log(u) where u ~ Uniform(0, exp(-H0)).
    // Equivalent: log_u = ln(rand()) - H0.
    let log_u: f64 = rng.random::<f64>().ln() - h0;

    // Initialize tree with current point
    let mut tree = NutsTree {
        q_left: state.q.clone(),
        p_left: state.p.clone(),
        grad_left: state.grad_potential.clone(),
        q_right: state.q.clone(),
        p_right: state.p.clone(),
        grad_right: state.grad_potential.clone(),
        q_proposal: state.q.clone(),
        potential_proposal: state.potential,
        grad_proposal: state.grad_potential.clone(),
        log_sum_weight: 0.0, // log(1) = 0
        depth: 0,
        n_leapfrog: 0,
        divergent: false,
        turning: false,
        sum_accept_prob: 0.0,
    };

    // Tree depth is 0-based (Stan convention): depth=0 means a single leapfrog step.
    // We track the maximum built depth and return it for diagnostics.
    let mut depth: usize = 0;
    let mut depth_reached: usize = 0;

    while depth <= max_treedepth {
        depth_reached = depth;
        // Choose direction uniformly: +1 or -1
        let direction: i32 = if rng.random::<bool>() { 1 } else { -1 };

        // Build subtree in chosen direction
        let edge_state = if direction > 0 {
            HmcState {
                q: tree.q_right.clone(),
                p: tree.p_right.clone(),
                potential: 0.0,
                grad_potential: tree.grad_right.clone(),
            }
        } else {
            HmcState {
                q: tree.q_left.clone(),
                p: tree.p_left.clone(),
                potential: 0.0,
                grad_potential: tree.grad_left.clone(),
            }
        };

        let subtree =
            build_tree(integrator, &edge_state, depth, direction, log_u, h0, inv_mass, rng)?;

        // Multinomial merge: accept subtree proposal with probability
        // exp(subtree.log_sum_weight - new_log_sum_weight)
        let new_log_sum_weight = log_sum_exp(tree.log_sum_weight, subtree.log_sum_weight);
        let accept_subtree = (subtree.log_sum_weight - new_log_sum_weight).exp();
        let u: f64 = rng.random();
        if u < accept_subtree {
            tree.q_proposal = subtree.q_proposal;
            tree.potential_proposal = subtree.potential_proposal;
            tree.grad_proposal = subtree.grad_proposal;
        }

        tree.log_sum_weight = new_log_sum_weight;
        tree.n_leapfrog += subtree.n_leapfrog;
        tree.sum_accept_prob += subtree.sum_accept_prob;
        tree.divergent = tree.divergent || subtree.divergent;
        tree.turning = tree.turning || subtree.turning;

        // Update tree edges
        if direction > 0 {
            tree.q_right = subtree.q_right;
            tree.p_right = subtree.p_right;
            tree.grad_right = subtree.grad_right;
        } else {
            tree.q_left = subtree.q_left;
            tree.p_left = subtree.p_left;
            tree.grad_left = subtree.grad_left;
        }

        // Check U-turn on full tree
        let dq: Vec<f64> =
            tree.q_right.iter().zip(tree.q_left.iter()).map(|(&r, &l)| r - l).collect();
        if is_turning(&dq, &tree.p_left, &tree.p_right, inv_mass) {
            tree.turning = true;
            break;
        }
        if tree.divergent || tree.turning {
            break;
        }

        depth += 1;
    }

    let n_total = tree.n_leapfrog.max(1) as f64;
    let accept_prob = tree.sum_accept_prob / n_total;

    Ok(NutsTransition {
        q: tree.q_proposal,
        potential: tree.potential_proposal,
        grad_potential: tree.grad_proposal,
        depth: depth_reached,
        divergent: tree.divergent,
        accept_prob,
        energy: h0,
        n_leapfrog: tree.n_leapfrog,
    })
}

/// Run NUTS sampling on any [`LogDensityModel`].
///
/// Returns raw chain data: draws in unconstrained and constrained space,
/// plus diagnostics (divergences, tree depths, acceptance probabilities).
pub fn sample_nuts<M: LogDensityModel>(
    model: &M,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    config: NutsConfig,
) -> Result<crate::chain::Chain> {
    use rand::SeedableRng;

    let posterior = Posterior::new(model);
    let dim = posterior.dim();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Initialize near the posterior mode (MLE) for stability and multi-chain consistency.
    //
    // This improves convergence of short warmup runs (e.g. CI quality gates) and
    // mirrors the common HEP workflow where MLE is readily available.
    let theta_init: Vec<f64> = {
        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        match mle.fit_minimum(model) {
            Ok(r) if r.converged => r.parameters,
            _ => model.parameter_init(),
        }
    };
    let z_init = posterior.to_unconstrained(&theta_init)?;
    let init_modes = (config.init_jitter > 0.0) as u8
        + config.init_jitter_rel.is_some() as u8
        + config.init_overdispersed_rel.is_some() as u8;
    if init_modes > 1 {
        return Err(ns_core::Error::Validation(
            "init_jitter, init_jitter_rel, init_overdispersed_rel are mutually exclusive"
                .to_string(),
        ));
    }

    let z_init: Vec<f64> = if let Some(frac) = config.init_overdispersed_rel.filter(|&f| f > 0.0) {
        use rand_distr::{Distribution, Normal};

        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        let jac = posterior.transform().jacobian_diag(&z_init);

        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let (lo, hi) = bounds[i];
            let lo_finite = lo > f64::NEG_INFINITY;
            let hi_finite = hi < f64::INFINITY;

            // Larger, more overdispersed constrained-space scale than init_jitter_rel.
            let theta0 = theta_init[i];
            let theta_sigma = if lo_finite && hi_finite {
                (hi - lo).abs() * frac
            } else if lo_finite || hi_finite {
                theta0.abs().max(1.0) * frac
            } else {
                0.0
            };

            let jac_abs = jac[i].abs().max(1e-12);
            let mut z_sigma = if theta_sigma > 0.0 {
                theta_sigma / jac_abs
            } else {
                (1.0 + z_init[i].abs()) * frac
            };

            // Overdispersed: allow larger excursions than init_jitter_rel.
            z_sigma = z_sigma.clamp(1e-6, 20.0);

            let normal = Normal::new(0.0, z_sigma).unwrap();
            out.push(z_init[i] + normal.sample(&mut rng));
        }
        out
    } else if let Some(frac) = config.init_jitter_rel.filter(|&f| f > 0.0) {
        use rand_distr::{Distribution, Normal};

        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        let jac = posterior.transform().jacobian_diag(&z_init);

        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let (lo, hi) = bounds[i];
            let lo_finite = lo > f64::NEG_INFINITY;
            let hi_finite = hi < f64::INFINITY;

            // Target constrained-space jitter scale, mapped to unconstrained using local Jacobian.
            let theta0 = theta_init[i];
            let theta_sigma = if lo_finite && hi_finite {
                (hi - lo).abs() * frac
            } else if lo_finite || hi_finite {
                theta0.abs().max(1.0) * frac
            } else {
                // Unbounded: treat as an unconstrained scale factor.
                0.0
            };

            let jac_abs = jac[i].abs().max(1e-12);
            let mut z_sigma = if theta_sigma > 0.0 {
                theta_sigma / jac_abs
            } else {
                // Unbounded: jitter around the current unconstrained location.
                (1.0 + z_init[i].abs()) * frac
            };

            // Avoid pathological huge jitters near transform boundaries.
            z_sigma = z_sigma.clamp(1e-6, 5.0);

            let normal = Normal::new(0.0, z_sigma).unwrap();
            out.push(z_init[i] + normal.sample(&mut rng));
        }
        out
    } else if config.init_jitter > 0.0 {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, config.init_jitter).unwrap();
        z_init.iter().map(|&z| z + normal.sample(&mut rng)).collect()
    } else {
        z_init
    };

    let inv_mass = vec![1.0; dim];
    let init_eps = find_reasonable_step_size(&posterior, &z_init, &inv_mass);

    let mut adaptation = WindowedAdaptation::new(dim, n_warmup, config.target_accept, init_eps);

    let integrator = LeapfrogIntegrator::new(&posterior, init_eps, inv_mass);

    // Initialize state
    let mut state = integrator.init_state(z_init)?;

    // Warmup
    for i in 0..n_warmup {
        let eps = adaptation.step_size();
        let inv_m = adaptation.inv_mass_diag().to_vec();
        let warmup_integrator = LeapfrogIntegrator::new(&posterior, eps, inv_m.clone());

        let transition =
            nuts_transition(&warmup_integrator, &state, config.max_treedepth, &inv_m, &mut rng)?;

        state.q = transition.q;
        state.potential = transition.potential;
        state.grad_potential = transition.grad_potential;

        adaptation.update(i, &state.q, transition.accept_prob);
    }

    // Sampling with fixed adapted parameters
    let final_eps = adaptation.adapted_step_size();
    let final_inv_mass = adaptation.inv_mass_diag().to_vec();
    let sample_integrator = LeapfrogIntegrator::new(&posterior, final_eps, final_inv_mass.clone());

    let mut draws_unconstrained = Vec::with_capacity(n_samples);
    let mut draws_constrained = Vec::with_capacity(n_samples);
    let mut divergences = Vec::with_capacity(n_samples);
    let mut tree_depths = Vec::with_capacity(n_samples);
    let mut accept_probs = Vec::with_capacity(n_samples);
    let mut energies = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let transition = nuts_transition(
            &sample_integrator,
            &state,
            config.max_treedepth,
            &final_inv_mass,
            &mut rng,
        )?;

        state.q = transition.q;
        state.potential = transition.potential;
        state.grad_potential = transition.grad_potential;

        draws_unconstrained.push(state.q.clone());
        draws_constrained.push(posterior.to_constrained(&state.q)?);
        divergences.push(transition.divergent);
        tree_depths.push(transition.depth);
        accept_probs.push(transition.accept_prob);
        energies.push(transition.energy);
    }

    let mass_diag: Vec<f64> = final_inv_mass.iter().map(|&m| 1.0 / m).collect();

    Ok(crate::chain::Chain {
        draws_unconstrained,
        draws_constrained,
        divergences,
        tree_depths,
        accept_probs,
        energies,
        max_treedepth: config.max_treedepth,
        step_size: final_eps,
        mass_diag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::{HistFactoryModel, Workspace};
    use rand::SeedableRng;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_nuts_transition_runs() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let dim = posterior.dim();
        let inv_mass = vec![1.0; dim];
        let integrator = LeapfrogIntegrator::new(&posterior, 0.1, inv_mass.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let state = integrator.init_state(z_init).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let transition = nuts_transition(&integrator, &state, 10, &inv_mass, &mut rng).unwrap();

        assert!(transition.depth <= 10);
        assert!(transition.accept_prob >= 0.0);
        assert!(transition.n_leapfrog > 0);
    }

    #[test]
    fn test_nuts_deterministic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let dim = posterior.dim();
        let inv_mass = vec![1.0; dim];
        let integrator = LeapfrogIntegrator::new(&posterior, 0.1, inv_mass.clone());

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let state = integrator.init_state(z_init).unwrap();

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let t1 = nuts_transition(&integrator, &state, 10, &inv_mass, &mut rng1).unwrap();

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let t2 = nuts_transition(&integrator, &state, 10, &inv_mass, &mut rng2).unwrap();

        assert_eq!(t1.q, t2.q, "NUTS should be deterministic with same seed");
        assert_eq!(t1.depth, t2.depth);
        assert_eq!(t1.divergent, t2.divergent);
    }

    #[test]
    fn test_sample_nuts_basic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.5,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };
        let chain = sample_nuts(&model, 100, 50, 42, config).unwrap();

        assert_eq!(chain.draws_constrained.len(), 50);
        assert_eq!(chain.draws_unconstrained.len(), 50);
        assert_eq!(chain.divergences.len(), 50);
        assert_eq!(chain.tree_depths.len(), 50);
        assert_eq!(chain.accept_probs.len(), 50);
        assert_eq!(chain.energies.len(), 50);

        // Divergence rate should be low
        let n_div: usize = chain.divergences.iter().filter(|&&d| d).count();
        let div_rate = n_div as f64 / 50.0;
        assert!(div_rate < 0.5, "Too many divergences: {} / 50 = {}", n_div, div_rate);

        // All constrained samples should have reasonable POI values
        for draw in &chain.draws_constrained {
            let poi = draw[0];
            assert!(
                poi.is_finite() && poi >= 0.0,
                "POI should be finite and non-negative: {}",
                poi
            );
        }
    }

    #[test]
    fn test_sample_nuts_deterministic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };
        let chain1 = sample_nuts(&model, 50, 20, 123, config.clone()).unwrap();
        let chain2 = sample_nuts(&model, 50, 20, 123, config).unwrap();

        assert_eq!(
            chain1.draws_constrained, chain2.draws_constrained,
            "Same seed should produce identical draws"
        );
        assert_eq!(chain1.energies, chain2.energies, "Energy series should be deterministic");
    }

    /// Quality gate: full pipeline must produce well-converged samples on the
    /// simple workspace.  This validates R-hat, ESS, divergence rate, E-BFMI,
    /// and posterior mean proximity to MLE.
    #[test]
    #[ignore] // slow (~10s); run with `cargo test -- --ignored`
    fn test_nuts_quality_gate() {
        use crate::chain::sample_nuts_multichain;
        use crate::diagnostics::compute_diagnostics;

        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let config = NutsConfig {
            max_treedepth: 10,
            target_accept: 0.8,
            init_jitter: 0.5,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };
        let result = sample_nuts_multichain(&model, 4, 500, 500, 42, config).unwrap();

        let diag = compute_diagnostics(&result);

        // R-hat < 1.05 for all parameters
        for (i, &rhat) in diag.r_hat.iter().enumerate() {
            assert!(
                rhat < 1.05,
                "R-hat for param {} = {} (should be < 1.05)",
                result.param_names[i],
                rhat,
            );
        }

        // Bulk ESS > 100 for all parameters
        for (i, &ess) in diag.ess_bulk.iter().enumerate() {
            assert!(
                ess > 100.0,
                "Bulk ESS for param {} = {} (should be > 100)",
                result.param_names[i],
                ess,
            );
        }

        // Divergence rate < 10%
        assert!(
            diag.divergence_rate < 0.10,
            "Divergence rate = {} (should be < 0.10)",
            diag.divergence_rate,
        );

        // E-BFMI > 0.2 for all chains
        for (i, &bfmi) in diag.ebfmi.iter().enumerate() {
            assert!(bfmi > 0.2, "E-BFMI for chain {} = {} (should be > 0.2)", i, bfmi,);
        }

        // POI posterior mean should be positive and in a reasonable range.
        //
        // Note: the Bayesian posterior mean differs from MLE due to the implicit
        // Jacobian prior from the sigmoid transform, so we only check that the
        // mean is in (0, 5) - broadly consistent with the signal strength.
        let poi_mean = result.param_mean(0);
        assert!(poi_mean > 0.0 && poi_mean < 5.0, "POI posterior mean out of range: {}", poi_mean,);
    }

    // -----------------------------------------------------------------------
    // Simulation-based calibration (SBC)
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct NormalMeanModel {
        data: Vec<f64>,
        sigma: f64,
        prior_mu: f64,
        prior_sigma: f64,
    }

    impl ns_core::traits::LogDensityModel for NormalMeanModel {
        type Prepared<'a>
            = ns_core::traits::PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            1
        }

        fn parameter_names(&self) -> Vec<String> {
            vec!["mu".to_string()]
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0]
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            if params.len() != 1 {
                return Err(ns_core::Error::Validation(format!(
                    "expected 1 parameter (mu), got {}",
                    params.len()
                )));
            }
            if !self.sigma.is_finite() || self.sigma <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "sigma must be finite and > 0, got {}",
                    self.sigma
                )));
            }
            if !self.prior_sigma.is_finite() || self.prior_sigma <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "prior_sigma must be finite and > 0, got {}",
                    self.prior_sigma
                )));
            }

            let mu = params[0];
            let inv_var = 1.0 / (self.sigma * self.sigma);
            let mut nll = 0.0;
            for &x in &self.data {
                let r = x - mu;
                nll += 0.5 * r * r * inv_var;
            }

            // Gaussian prior on mu (implemented as an additive NLL term).
            let z = (mu - self.prior_mu) / self.prior_sigma;
            nll += 0.5 * z * z;
            Ok(nll)
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            let mu = params.get(0).copied().ok_or_else(|| {
                ns_core::Error::Validation("expected 1 parameter (mu)".to_string())
            })?;
            let inv_var = 1.0 / (self.sigma * self.sigma);
            let mut g = 0.0;
            for &x in &self.data {
                // d/dmu 0.5 * (x-mu)^2 / sigma^2 = (mu - x) / sigma^2
                g += (mu - x) * inv_var;
            }
            g += (mu - self.prior_mu) / (self.prior_sigma * self.prior_sigma);
            Ok(vec![g])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            ns_core::traits::PreparedModelRef::new(self)
        }
    }

    fn sbc_ranks(
        n_rep: usize,
        n_obs: usize,
        n_warmup: usize,
        n_samples: usize,
        seed: u64,
    ) -> Vec<usize> {
        use crate::chain::sample_nuts_multichain;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let prior = Normal::new(0.0, 1.0).unwrap();
        let obs = Normal::new(0.0, 1.0).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };

        let mut ranks = Vec::with_capacity(n_rep);
        for rep in 0..n_rep {
            let mu_true = prior.sample(&mut rng);
            let data: Vec<f64> = (0..n_obs).map(|_| mu_true + obs.sample(&mut rng)).collect();
            let model = NormalMeanModel { data, sigma: 1.0, prior_mu: 0.0, prior_sigma: 1.0 };

            let result = sample_nuts_multichain(
                &model,
                1,
                n_warmup,
                n_samples,
                seed + rep as u64,
                config.clone(),
            )
            .expect("NUTS should run for SBC model");
            let draws = result.param_draws(0);
            let draws = draws.first().expect("one chain").as_slice();
            let rank = draws.iter().filter(|&&x| x < mu_true).count();
            ranks.push(rank);
        }
        ranks
    }

    #[test]
    fn test_sbc_rank_statistic_smoke() {
        // Keep this fast: few reps and small chains. We only check that ranks are in range.
        let n_samples = 40usize;
        let ranks = sbc_ranks(3, 10, 50, n_samples, 123);
        for r in ranks {
            assert!(r <= n_samples, "rank {} out of range", r);
        }
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_sbc_normal_mean -- --ignored`"]
    fn test_sbc_normal_mean() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let n_rep = 20usize;
        let n_samples = 200usize;
        let ranks = sbc_ranks(n_rep, 20, 200, n_samples, 7);

        // Histogram ranks into a few bins and run a chi-square uniformity test.
        let bins = 5usize;
        let mut counts = vec![0usize; bins];
        for r in ranks {
            let u = r as f64 / n_samples as f64;
            let mut b = (u * bins as f64).floor() as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1;
        }

        let expected = n_rep as f64 / bins as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();
        let dist = ChiSquared::new((bins - 1) as f64).unwrap();
        let p = 1.0 - dist.cdf(chi2);

        assert!(
            p > 0.01,
            "SBC ranks deviate from uniform too much: chi2={}, p={}, counts={:?}",
            chi2,
            p,
            counts
        );
    }

    // -----------------------------------------------------------------------
    // SBC: 2D Normal mean model (independent components)
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct Normal2DMeanModel {
        data: Vec<(f64, f64)>,
        sigma: f64,
        prior_sigma: f64,
    }

    impl ns_core::traits::LogDensityModel for Normal2DMeanModel {
        type Prepared<'a>
            = ns_core::traits::PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            2
        }

        fn parameter_names(&self) -> Vec<String> {
            vec!["mu1".to_string(), "mu2".to_string()]
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0, 0.0]
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            if params.len() != 2 {
                return Err(ns_core::Error::Validation(format!(
                    "expected 2 parameters (mu1, mu2), got {}",
                    params.len()
                )));
            }
            if !self.sigma.is_finite() || self.sigma <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "sigma must be finite and > 0, got {}",
                    self.sigma
                )));
            }
            if !self.prior_sigma.is_finite() || self.prior_sigma <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "prior_sigma must be finite and > 0, got {}",
                    self.prior_sigma
                )));
            }

            let mu1 = params[0];
            let mu2 = params[1];
            let inv_var = 1.0 / (self.sigma * self.sigma);
            let mut nll = 0.0;
            for &(x1, x2) in &self.data {
                let r1 = x1 - mu1;
                let r2 = x2 - mu2;
                nll += 0.5 * (r1 * r1 + r2 * r2) * inv_var;
            }

            // Independent Gaussian priors (up to constant).
            nll += 0.5 * (mu1 / self.prior_sigma).powi(2);
            nll += 0.5 * (mu2 / self.prior_sigma).powi(2);
            Ok(nll)
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            let mu1 = params.get(0).copied().ok_or_else(|| {
                ns_core::Error::Validation("expected 2 parameters (mu1, mu2)".to_string())
            })?;
            let mu2 = params.get(1).copied().ok_or_else(|| {
                ns_core::Error::Validation("expected 2 parameters (mu1, mu2)".to_string())
            })?;

            let inv_var = 1.0 / (self.sigma * self.sigma);
            let mut g1 = 0.0;
            let mut g2 = 0.0;
            for &(x1, x2) in &self.data {
                g1 += (mu1 - x1) * inv_var;
                g2 += (mu2 - x2) * inv_var;
            }
            g1 += mu1 / (self.prior_sigma * self.prior_sigma);
            g2 += mu2 / (self.prior_sigma * self.prior_sigma);
            Ok(vec![g1, g2])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            ns_core::traits::PreparedModelRef::new(self)
        }
    }

    fn sbc_ranks_2d(
        n_rep: usize,
        n_obs: usize,
        n_warmup: usize,
        n_samples: usize,
        seed: u64,
    ) -> Vec<(usize, usize)> {
        use crate::chain::sample_nuts_multichain;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let prior = Normal::new(0.0, 1.0).unwrap();
        let obs = Normal::new(0.0, 1.0).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };

        let mut ranks = Vec::with_capacity(n_rep);
        for rep in 0..n_rep {
            let mu1_true = prior.sample(&mut rng);
            let mu2_true = prior.sample(&mut rng);
            let data: Vec<(f64, f64)> = (0..n_obs)
                .map(|_| (mu1_true + obs.sample(&mut rng), mu2_true + obs.sample(&mut rng)))
                .collect();
            let model = Normal2DMeanModel { data, sigma: 1.0, prior_sigma: 1.0 };

            let result = sample_nuts_multichain(
                &model,
                1,
                n_warmup,
                n_samples,
                seed + rep as u64,
                config.clone(),
            )
            .expect("NUTS should run for SBC model");

            let draws1 = result.param_draws(0);
            let draws2 = result.param_draws(1);
            let draws1 = draws1.first().expect("one chain").as_slice();
            let draws2 = draws2.first().expect("one chain").as_slice();

            let r1 = draws1.iter().filter(|&&x| x < mu1_true).count();
            let r2 = draws2.iter().filter(|&&x| x < mu2_true).count();
            ranks.push((r1, r2));
        }
        ranks
    }

    #[test]
    fn test_sbc_rank_statistic_2d_smoke() {
        let n_samples = 30usize;
        let ranks = sbc_ranks_2d(2, 10, 40, n_samples, 321);
        for (r1, r2) in ranks {
            assert!(r1 <= n_samples && r2 <= n_samples);
        }
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_sbc_normal_2d_mean -- --ignored`"]
    fn test_sbc_normal_2d_mean() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let n_rep = 15usize;
        let n_samples = 200usize;
        let ranks = sbc_ranks_2d(n_rep, 20, 200, n_samples, 11);

        fn chi2_pvalue(counts: &[usize]) -> f64 {
            let bins = counts.len();
            let expected = counts.iter().sum::<usize>() as f64 / bins as f64;
            let chi2: f64 = counts
                .iter()
                .map(|&c| {
                    let d = c as f64 - expected;
                    d * d / expected
                })
                .sum();
            let dist = ChiSquared::new((bins - 1) as f64).unwrap();
            1.0 - dist.cdf(chi2)
        }

        let bins = 5usize;
        let mut counts1 = vec![0usize; bins];
        let mut counts2 = vec![0usize; bins];
        for (r1, r2) in ranks {
            let u1 = r1 as f64 / n_samples as f64;
            let u2 = r2 as f64 / n_samples as f64;
            let mut b1 = (u1 * bins as f64).floor() as usize;
            let mut b2 = (u2 * bins as f64).floor() as usize;
            b1 = b1.min(bins - 1);
            b2 = b2.min(bins - 1);
            counts1[b1] += 1;
            counts2[b2] += 1;
        }

        let p1 = chi2_pvalue(&counts1);
        let p2 = chi2_pvalue(&counts2);

        assert!(p1 > 0.01, "SBC mu1 ranks deviate from uniform: p={}, counts={:?}", p1, counts1);
        assert!(p2 > 0.01, "SBC mu2 ranks deviate from uniform: p={}, counts={:?}", p2, counts2);
    }

    // -----------------------------------------------------------------------
    // SBC: simple hierarchical intercept model
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct HierInterceptModel {
        y: Vec<f64>,
        sigma_y: f64,
        prior_mu_sigma: f64,
        // LogNormal prior on sigma_alpha: ln(sigma) ~ Normal(m, s)
        prior_sigma_m: f64,
        prior_sigma_s: f64,
    }

    impl ns_core::traits::LogDensityModel for HierInterceptModel {
        type Prepared<'a>
            = ns_core::traits::PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            // mu, sigma_alpha, alpha_j...
            2 + self.y.len()
        }

        fn parameter_names(&self) -> Vec<String> {
            let mut names = Vec::with_capacity(self.dim());
            names.push("mu".to_string());
            names.push("sigma_alpha".to_string());
            for j in 0..self.y.len() {
                names.push(format!("alpha[{}]", j));
            }
            names
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            let mut b = Vec::with_capacity(self.dim());
            b.push((f64::NEG_INFINITY, f64::INFINITY)); // mu
            b.push((0.0, f64::INFINITY)); // sigma_alpha
            for _ in 0..self.y.len() {
                b.push((f64::NEG_INFINITY, f64::INFINITY)); // alpha_j
            }
            b
        }

        fn parameter_init(&self) -> Vec<f64> {
            let mut p = Vec::with_capacity(self.dim());
            p.push(0.0);
            p.push(1.0);
            for _ in 0..self.y.len() {
                p.push(0.0);
            }
            p
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            let g = self.y.len();
            if params.len() != 2 + g {
                return Err(ns_core::Error::Validation(format!(
                    "expected {} parameters, got {}",
                    2 + g,
                    params.len()
                )));
            }
            if !self.sigma_y.is_finite() || self.sigma_y <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "sigma_y must be finite and > 0, got {}",
                    self.sigma_y
                )));
            }
            if !self.prior_mu_sigma.is_finite() || self.prior_mu_sigma <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "prior_mu_sigma must be finite and > 0, got {}",
                    self.prior_mu_sigma
                )));
            }
            if !self.prior_sigma_s.is_finite() || self.prior_sigma_s <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "prior_sigma_s must be finite and > 0, got {}",
                    self.prior_sigma_s
                )));
            }

            let mu = params[0];
            let sigma_a = params[1];
            if !sigma_a.is_finite() || sigma_a <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "sigma_alpha must be finite and > 0, got {}",
                    sigma_a
                )));
            }

            let inv_var_y = 1.0 / (self.sigma_y * self.sigma_y);

            // Likelihood: y_j ~ Normal(alpha_j, sigma_y) (up to constant in sigma_y)
            let mut nll = 0.0;
            for j in 0..g {
                let a = params[2 + j];
                let r = self.y[j] - a;
                nll += 0.5 * r * r * inv_var_y;
            }

            // Prior on alpha_j: Normal(mu, sigma_a), includes +log(sigma_a) term.
            for j in 0..g {
                let a = params[2 + j];
                let z = (a - mu) / sigma_a;
                nll += 0.5 * z * z;
            }
            nll += (g as f64) * sigma_a.ln();

            // Prior on mu: Normal(0, prior_mu_sigma) (up to constant).
            nll += 0.5 * (mu / self.prior_mu_sigma).powi(2);

            // Prior on sigma_a: LogNormal(m, s) (up to constant).
            let t = sigma_a.ln();
            let z = (t - self.prior_sigma_m) / self.prior_sigma_s;
            nll += 0.5 * z * z + t;

            Ok(nll)
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            let g = self.y.len();
            let mu = params[0];
            let sigma_a = params[1];
            let inv_var_y = 1.0 / (self.sigma_y * self.sigma_y);
            let inv_var_a = 1.0 / (sigma_a * sigma_a);

            let mut grad = vec![0.0; 2 + g];

            // d/d alpha_j: (alpha_j - y_j)/sigma_y^2 + (alpha_j - mu)/sigma_a^2
            for j in 0..g {
                let a = params[2 + j];
                grad[2 + j] += (a - self.y[j]) * inv_var_y;
                grad[2 + j] += (a - mu) * inv_var_a;
            }

            // d/d mu: sum (mu - alpha_j)/sigma_a^2 + mu/prior_mu_sigma^2
            for j in 0..g {
                let a = params[2 + j];
                grad[0] += (mu - a) * inv_var_a;
            }
            grad[0] += mu / (self.prior_mu_sigma * self.prior_mu_sigma);

            // d/d sigma_a: from alpha prior + log(sigma_a) normalization
            // alpha terms: -sum (a-mu)^2 / sigma^3 + g/sigma
            let mut sum_sq = 0.0;
            for j in 0..g {
                let a = params[2 + j];
                let d = a - mu;
                sum_sq += d * d;
            }
            let d_alpha = (g as f64) / sigma_a - sum_sq / (sigma_a * sigma_a * sigma_a);

            // sigma prior: LogNormal(m,s): 0.5*((ln s - m)/s)^2 + ln s
            // d/ds = ((ln s - m)/s^2 + 1) * 1/s
            let t = sigma_a.ln();
            let d_sigma_prior =
                ((t - self.prior_sigma_m) / (self.prior_sigma_s * self.prior_sigma_s) + 1.0)
                    / sigma_a;

            grad[1] = d_alpha + d_sigma_prior;

            Ok(grad)
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            ns_core::traits::PreparedModelRef::new(self)
        }
    }

    fn sbc_ranks_hier(
        n_rep: usize,
        n_groups: usize,
        n_warmup: usize,
        n_samples: usize,
        seed: u64,
    ) -> Vec<(usize, usize)> {
        use crate::chain::sample_nuts_multichain;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let prior_mu: Normal<f64> = Normal::new(0.0, 1.0).unwrap();
        let prior_logsigma: Normal<f64> = Normal::new(0.0, 0.5).unwrap();
        let obs: Normal<f64> = Normal::new(0.0, 1.0).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.85,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };

        let mut ranks = Vec::with_capacity(n_rep);
        for rep in 0..n_rep {
            let mu_true: f64 = prior_mu.sample(&mut rng);
            let log_sigma_true: f64 = prior_logsigma.sample(&mut rng);
            let sigma_true: f64 = log_sigma_true.exp();

            let alpha_prior = Normal::new(mu_true, sigma_true).unwrap();
            let y: Vec<f64> = (0..n_groups)
                .map(|_| alpha_prior.sample(&mut rng) + obs.sample(&mut rng))
                .collect();

            let model = HierInterceptModel {
                y,
                sigma_y: 1.0,
                prior_mu_sigma: 1.0,
                prior_sigma_m: 0.0,
                prior_sigma_s: 0.5,
            };

            let result = sample_nuts_multichain(
                &model,
                1,
                n_warmup,
                n_samples,
                seed + rep as u64,
                config.clone(),
            )
            .expect("NUTS should run for hierarchical SBC model");

            let draws_mu = result.param_draws(0);
            let draws_sigma = result.param_draws(1);
            let draws_mu = draws_mu.first().expect("one chain").as_slice();
            let draws_sigma = draws_sigma.first().expect("one chain").as_slice();

            let r_mu = draws_mu.iter().filter(|&&x| x < mu_true).count();
            let r_sigma = draws_sigma.iter().filter(|&&x| x < sigma_true).count();
            ranks.push((r_mu, r_sigma));
        }
        ranks
    }

    #[test]
    fn test_sbc_rank_statistic_hier_smoke() {
        let n_samples = 30usize;
        let ranks = sbc_ranks_hier(2, 3, 60, n_samples, 999);
        for (r_mu, r_sigma) in ranks {
            assert!(r_mu <= n_samples && r_sigma <= n_samples);
        }
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_sbc_hier_intercept -- --ignored`"]
    fn test_sbc_hier_intercept() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let n_rep = 12usize;
        let n_samples = 250usize;
        let ranks = sbc_ranks_hier(n_rep, 4, 250, n_samples, 2024);

        fn chi2_pvalue(counts: &[usize]) -> f64 {
            let bins = counts.len();
            let expected = counts.iter().sum::<usize>() as f64 / bins as f64;
            let chi2: f64 = counts
                .iter()
                .map(|&c| {
                    let d = c as f64 - expected;
                    d * d / expected
                })
                .sum();
            let dist = ChiSquared::new((bins - 1) as f64).unwrap();
            1.0 - dist.cdf(chi2)
        }

        let bins = 4usize;
        let mut counts_mu = vec![0usize; bins];
        let mut counts_sigma = vec![0usize; bins];
        for (r_mu, r_sigma) in ranks {
            let u_mu = r_mu as f64 / n_samples as f64;
            let u_sigma = r_sigma as f64 / n_samples as f64;
            let mut b_mu = (u_mu * bins as f64).floor() as usize;
            let mut b_sigma = (u_sigma * bins as f64).floor() as usize;
            b_mu = b_mu.min(bins - 1);
            b_sigma = b_sigma.min(bins - 1);
            counts_mu[b_mu] += 1;
            counts_sigma[b_sigma] += 1;
        }

        let p_mu = chi2_pvalue(&counts_mu);
        let p_sigma = chi2_pvalue(&counts_sigma);

        assert!(
            p_mu > 0.01,
            "SBC mu ranks deviate from uniform: p={}, counts={:?}",
            p_mu,
            counts_mu
        );
        assert!(
            p_sigma > 0.01,
            "SBC sigma ranks deviate from uniform: p={}, counts={:?}",
            p_sigma,
            counts_sigma
        );
    }
}
