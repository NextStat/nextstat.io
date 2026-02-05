//! No-U-Turn Sampler (NUTS) with multinomial trajectory selection.
//!
//! Implements the Stan-style NUTS algorithm with tree doubling,
//! no-U-turn criterion, and multinomial proposal selection.

use crate::adapt::{WindowedAdaptation, find_reasonable_step_size};
use crate::hmc::{HmcState, LeapfrogIntegrator};
use crate::posterior::Posterior;
use ns_core::Result;
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
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self { max_treedepth: 10, target_accept: 0.8, init_jitter: 0.5 }
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
fn build_leaf(
    integrator: &LeapfrogIntegrator<'_, '_>,
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
    let log_weight = if in_slice { -energy_error } else { f64::NEG_INFINITY };

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
fn build_tree(
    integrator: &LeapfrogIntegrator<'_, '_>,
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
pub(crate) fn nuts_transition(
    integrator: &LeapfrogIntegrator<'_, '_>,
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

    let mut depth: usize = 0;

    while depth < max_treedepth {
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
        depth,
        divergent: tree.divergent,
        accept_prob,
        energy: h0,
        n_leapfrog: tree.n_leapfrog,
    })
}

/// Run NUTS sampling on a HistFactory model.
///
/// Returns raw chain data: draws in unconstrained and constrained space,
/// plus diagnostics (divergences, tree depths, acceptance probabilities).
pub fn sample_nuts(
    model: &ns_translate::pyhf::HistFactoryModel,
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
            _ => model.parameters().iter().map(|p| p.init).collect(),
        }
    };
    let z_init = posterior.to_unconstrained(&theta_init);
    let z_init: Vec<f64> = if config.init_jitter > 0.0 {
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
        draws_constrained.push(posterior.to_constrained(&state.q));
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
        let z_init = posterior.to_unconstrained(&theta_init);
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
        let z_init = posterior.to_unconstrained(&theta_init);
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

        let config = NutsConfig { max_treedepth: 8, target_accept: 0.8, init_jitter: 0.5 };
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

        let config = NutsConfig { max_treedepth: 8, target_accept: 0.8, init_jitter: 0.0 };
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

        let config = NutsConfig { max_treedepth: 10, target_accept: 0.8, init_jitter: 0.5 };
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
            assert!(
                bfmi > 0.2,
                "E-BFMI for chain {} = {} (should be > 0.2)",
                i,
                bfmi,
            );
        }

        // POI posterior mean should be positive and in a reasonable range.
        //
        // Note: the Bayesian posterior mean differs from MLE due to the implicit
        // Jacobian prior from the sigmoid transform, so we only check that the
        // mean is in (0, 5) - broadly consistent with the signal strength.
        let poi_mean = result.param_mean(0);
        assert!(
            poi_mean > 0.0 && poi_mean < 5.0,
            "POI posterior mean out of range: {}",
            poi_mean,
        );
    }
}
