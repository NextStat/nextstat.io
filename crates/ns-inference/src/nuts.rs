//! No-U-Turn Sampler (NUTS).
//!
//! Implements NUTS with multinomial sampling and the generalized no-U-turn
//! criterion (Betancourt 2017).
//!
//! Proposal selection matches Stan:
//! - **Within subtrees**: multinomial sampling proportional to `exp(-energy_error)`
//! - **Top-level subtree joins**: Stan-style *progressive* sampling biased away
//!   from the initial point (uses `W_subtree / W_existing`, clamped to 1)
//!
//! The U-turn check uses the momentum sum (rho) instead of position difference.

use crate::adapt::{WindowedAdaptation, find_reasonable_step_size};
use crate::hmc::{HmcState, LeapfrogIntegrator};
use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;
use rand::Rng;

/// Chain initialization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum InitStrategy {
    /// Random initialization: each chain starts from `Uniform(-2, 2)` in unconstrained
    /// space (same as Stan/CmdStan). Best for general Bayesian models, especially
    /// hierarchical and funnel geometries.
    Random,
    /// MLE initialization: run a quick L-BFGS-B fit, then start near the mode.
    /// Best for HistFactory/HEP models where the mode is well-defined and the
    /// posterior is nearly Gaussian.
    Mle,
    /// Pathfinder initialization: run full L-BFGS fit with Hessian, then use
    /// the diagonal of the covariance as initial inverse mass matrix.
    /// This gives warmup a head-start on the preconditioner, allowing shorter
    /// warmup (e.g. 200 instead of 1000).
    Pathfinder,
}

/// Euclidean metric type for mass matrix adaptation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    /// Diagonal mass matrix (CmdStan default). Each parameter has an independent
    /// scale; cross-correlations are ignored.
    Diagonal,
    /// Dense mass matrix with off-diagonal elements. Better for correlated posteriors
    /// but requires more warmup samples for stable estimation.
    Dense,
    /// Automatic selection: dense for dim <= 32, diagonal otherwise.
    Auto,
}

/// NUTS sampler configuration.
#[derive(Debug, Clone)]
pub struct NutsConfig {
    /// Maximum tree depth (default 10).
    pub max_treedepth: usize,
    /// Target acceptance probability (default 0.8).
    pub target_accept: f64,
    /// Chain initialization strategy (default: Random).
    pub init_strategy: InitStrategy,
    /// Euclidean metric type (default: Diagonal, matching CmdStan).
    pub metric_type: MetricType,
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

    /// Fractional jitter applied to the step size during sampling.
    ///
    /// After warmup, each transition uses `eps * (1 + jitter * U(-1,1))` where
    /// `U(-1,1)` is uniform noise.  This breaks autocorrelation patterns that
    /// arise from a fixed step size.  Stan supports this (`stepsize_jitter`
    /// parameter, default 0).
    pub stepsize_jitter: f64,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            max_treedepth: 10,
            target_accept: 0.8,
            init_strategy: InitStrategy::Random,
            metric_type: MetricType::Diagonal,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
            stepsize_jitter: 0.0,
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
    /// Sum of momenta across all leaves in this sub-tree (generalized U-turn criterion).
    p_sum: Vec<f64>,
    depth: usize,
    n_leapfrog: usize,
    divergent: bool,
    turning: bool,
    sum_accept_prob: f64,
}

/// Maximum energy error before declaring divergence.
const DIVERGENCE_THRESHOLD: f64 = 1000.0;

/// Check the generalized no-U-turn criterion (Betancourt 2017).
///
/// `rho` is the sum of all momenta in the sub-tree.  The criterion checks
/// whether the trajectory is still making progress by testing
/// `rho · M^{-1} p_left >= 0` and `rho · M^{-1} p_right >= 0`.
fn is_turning(rho: &[f64], p_left: &[f64], p_right: &[f64], metric: &crate::hmc::Metric) -> bool {
    let v_left = metric.mul_inv_mass(p_left);
    let v_right = metric.mul_inv_mass(p_right);
    let dot_left: f64 = rho.iter().zip(v_left.iter()).map(|(&r, &v)| r * v).sum();
    let dot_right: f64 = rho.iter().zip(v_right.iter()).map(|(&r, &v)| r * v).sum();
    if !dot_left.is_finite() || !dot_right.is_finite() {
        return true;
    }
    dot_left < 0.0 || dot_right < 0.0
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    // Defensive: treat NaNs as "missing weight" (-inf) to avoid propagating NaNs
    // into selection probabilities and diagnostics.
    //
    // +inf is a valid (though unexpected) sentinel for overwhelming weight and
    // should dominate the sum.
    let a = if a.is_nan() { f64::NEG_INFINITY } else { a };
    let b = if b.is_nan() { f64::NEG_INFINITY } else { b };
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
}

/// Stable `P(select outer)` for multinomial subtree selection.
///
/// Returns `exp(logw_outer) / (exp(logw_inner) + exp(logw_outer))` with
/// protections for +/-inf and NaNs.
fn prob_select_outer(logw_inner: f64, logw_outer: f64) -> f64 {
    let a = if logw_inner.is_nan() { f64::NEG_INFINITY } else { logw_inner };
    let b = if logw_outer.is_nan() { f64::NEG_INFINITY } else { logw_outer };

    if b == f64::NEG_INFINITY {
        return 0.0;
    }
    if a == f64::NEG_INFINITY {
        return 1.0;
    }
    if b == f64::INFINITY {
        return if a == f64::INFINITY { 0.5 } else { 1.0 };
    }
    if a == f64::INFINITY {
        return 0.0;
    }

    // p = 1 / (1 + exp(a - b))
    let d = a - b;
    if !d.is_finite() {
        return 0.0;
    }
    if d > 0.0 {
        let e = (-d).exp(); // exp(b-a)
        e / (1.0 + e)
    } else {
        let e = d.exp(); // exp(a-b) in (0, 1]
        1.0 / (1.0 + e)
    }
}

/// Stan-style *progressive* sampling when joining a new subtree at top-level.
///
/// This is intentionally biased away from the initial point: it uses the ratio
/// `W_subtree / W_existing` (before updating the total weight), clamped to 1.
///
/// Note: this differs from the within-subtree multinomial selection, which uses
/// `W_outer / (W_inner + W_outer)`.
fn prob_select_outer_progressive(logw_existing: f64, logw_subtree: f64) -> f64 {
    let a = if logw_existing.is_nan() { f64::NEG_INFINITY } else { logw_existing };
    let b = if logw_subtree.is_nan() { f64::NEG_INFINITY } else { logw_subtree };

    if b == f64::NEG_INFINITY {
        return 0.0;
    }
    if a == f64::NEG_INFINITY {
        return 1.0;
    }
    if b == f64::INFINITY {
        return 1.0;
    }
    if a == f64::INFINITY {
        return 0.0;
    }

    let d = b - a; // log(W_sub / W_exist)
    if !d.is_finite() {
        return 0.0;
    }
    if d >= 0.0 { 1.0 } else { d.exp().clamp(0.0, 1.0) }
}

/// Build a single-node tree (one leapfrog step).
///
/// Multinomial NUTS: the leaf weight is `exp(-energy_error)` rather than a
/// binary in/out-of-slice indicator.
fn build_leaf<M: LogDensityModel + ?Sized>(
    integrator: &LeapfrogIntegrator<'_, '_, M>,
    state: &HmcState,
    direction: i32,
    h0: f64,
    metric: &crate::hmc::Metric,
) -> Result<NutsTree> {
    let mut new_state = state.clone();

    // Integrate forward/backward by taking a step with +/- eps.
    if integrator.step_dir(&mut new_state, direction).is_err() {
        // If the leapfrog step fails (e.g. non-finite logpdf/grad, or q blows up),
        // treat it as an immediate divergence with zero weight. This mirrors Stan's
        // behavior: invalid proposals should be rejected and drive step size down,
        // not abort the entire sampling run.
        let dim = state.q.len();
        return Ok(NutsTree {
            q_left: state.q.clone(),
            p_left: state.p.clone(),
            grad_left: state.grad_potential.clone(),
            q_right: state.q.clone(),
            p_right: state.p.clone(),
            grad_right: state.grad_potential.clone(),
            q_proposal: state.q.clone(),
            potential_proposal: state.potential,
            grad_proposal: state.grad_potential.clone(),
            log_sum_weight: f64::NEG_INFINITY,
            p_sum: vec![0.0; dim],
            depth: 0,
            n_leapfrog: 1,
            divergent: true,
            turning: true,
            sum_accept_prob: 0.0,
        });
    }

    let h = new_state.hamiltonian(metric);
    let energy_error = h - h0;
    let divergent =
        !h.is_finite() || !energy_error.is_finite() || energy_error.abs() > DIVERGENCE_THRESHOLD;
    // Multinomial NUTS: weight each leaf by exp(-energy_error).
    // log_weight = -energy_error for valid states, NEG_INFINITY for divergent.
    let log_weight = if divergent { f64::NEG_INFINITY } else { -energy_error };

    let accept_prob = if !energy_error.is_finite() { 0.0 } else { (-energy_error).exp().min(1.0) };

    let p_sum = new_state.p.clone();

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
        p_sum,
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
    h0: f64,
    metric: &crate::hmc::Metric,
    rng: &mut impl Rng,
) -> Result<NutsTree> {
    if depth == 0 {
        return build_leaf(integrator, state, direction, h0, metric);
    }

    // Build first half-tree (init subtree)
    let mut inner = build_tree(integrator, state, depth - 1, direction, h0, metric, rng)?;

    if inner.divergent || inner.turning {
        return Ok(inner);
    }

    // Save init subtree's momentum sum and junction momentum before merge
    // (needed for Stan-style cross-checks between subtrees).
    let rho_init = inner.p_sum.clone();
    let p_init_junction = if direction > 0 { inner.p_right.clone() } else { inner.p_left.clone() };

    // Build second half-tree (final subtree) from the edge of the first
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

    let outer = build_tree(integrator, &edge_state, depth - 1, direction, h0, metric, rng)?;

    // Save final subtree's junction momentum and momentum sum
    let p_final_junction = if direction > 0 { outer.p_left.clone() } else { outer.p_right.clone() };
    let rho_final = outer.p_sum.clone();

    // Merge trees
    let new_log_sum_weight = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);

    // Multinomial selection: accept outer proposal with probability proportional
    // to subtree weights. Use a stable logistic form to avoid inf - inf and
    // other numerical edge cases.
    //
    // Divergent leaves already have log_weight = -inf, so they contribute zero
    // selection probability. Turning subtrees contain valid leaves and should
    // participate in multinomial sampling (turning is a stopping criterion, not
    // a validity criterion).
    let accept_outer =
        prob_select_outer(inner.log_sum_weight, outer.log_sum_weight).clamp(0.0, 1.0);
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

    // Merge p_sum (generalized U-turn criterion)
    for (ps, os) in inner.p_sum.iter_mut().zip(outer.p_sum.iter()) {
        *ps += *os;
    }

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

    // Stan-style generalized U-turn check (3 criteria, Betancourt 2017).
    //
    // Check 1: Full merged tree — standard rho · v check on overall endpoints.
    // Check 2: Init-to-junction — catches U-turns at the boundary between subtrees
    //          using rho = rho_init + p_final_junction.
    // Check 3: Junction-to-final — symmetric check from the other side,
    //          using rho = rho_final + p_init_junction.
    let turning1 = is_turning(&inner.p_sum, &inner.p_left, &inner.p_right, metric);

    let rho_cross2: Vec<f64> =
        rho_init.iter().zip(p_final_junction.iter()).map(|(&a, &b)| a + b).collect();
    let p_start = if direction > 0 { &inner.p_left } else { &inner.p_right };
    let turning2 = is_turning(&rho_cross2, p_start, &p_final_junction, metric);

    let rho_cross3: Vec<f64> =
        rho_final.iter().zip(p_init_junction.iter()).map(|(&a, &b)| a + b).collect();
    let p_end = if direction > 0 { &inner.p_right } else { &inner.p_left };
    let turning3 = is_turning(&rho_cross3, &p_init_junction, p_end, metric);

    inner.turning = inner.turning || outer.turning || turning1 || turning2 || turning3;

    inner.depth = depth;
    Ok(inner)
}

/// Pre-allocated scratch buffers for the `nuts_transition()` main loop.
///
/// Eliminates ~9 `Vec<f64>` allocations per tree-doubling iteration
/// (up to ~90 per transition at max_treedepth=10).
struct NutsTransitionScratch {
    rho_existing: Vec<f64>,
    p_existing_junction: Vec<f64>,
    edge_state: HmcState,
    p_subtree_junction: Vec<f64>,
    rho_subtree: Vec<f64>,
    rho_cross: Vec<f64>,
}

impl NutsTransitionScratch {
    fn new(dim: usize) -> Self {
        Self {
            rho_existing: vec![0.0; dim],
            p_existing_junction: vec![0.0; dim],
            edge_state: HmcState {
                q: vec![0.0; dim],
                p: vec![0.0; dim],
                potential: 0.0,
                grad_potential: vec![0.0; dim],
            },
            p_subtree_junction: vec![0.0; dim],
            rho_subtree: vec![0.0; dim],
            rho_cross: vec![0.0; dim],
        }
    }
}

/// Run one NUTS transition from the given state.
pub(crate) fn nuts_transition<M: LogDensityModel + ?Sized>(
    integrator: &LeapfrogIntegrator<'_, '_, M>,
    current: &HmcState,
    max_treedepth: usize,
    rng: &mut impl Rng,
) -> Result<NutsTransition> {
    let metric = integrator.metric();

    // Sample momentum ~ N(0, M)
    let mut state = current.clone();
    state.p = metric.sample_momentum(rng);

    let h0 = state.hamiltonian(metric);
    if !h0.is_finite() {
        return Err(ns_core::Error::Validation(
            "non-finite initial Hamiltonian in NUTS transition".to_string(),
        ));
    }

    let dim = state.q.len();

    // Initialize tree with current point (multinomial: log_weight = 0 = log(1))
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
        p_sum: state.p.clone(),
        depth: 0,
        n_leapfrog: 0,
        divergent: false,
        turning: false,
        sum_accept_prob: 0.0,
    };

    // Pre-allocate scratch buffers for the tree-doubling loop.
    let mut scratch = NutsTransitionScratch::new(dim);

    // Tree doubling (Stan convention): `depth` counts completed doublings.
    // At depth d, the tree has 2^d leaves (2^d leapfrog steps total).
    // `while depth < max_treedepth` ensures at most 2^max_treedepth leaves
    // (e.g., 1024 for max_treedepth=10).  The previous `<=` was an off-by-one
    // that doubled the maximum trajectory length vs Stan.
    let mut depth: usize = 0;

    while depth < max_treedepth {
        // Choose direction uniformly: +1 or -1
        let direction: i32 = if rng.random::<bool>() { 1 } else { -1 };

        // Save existing tree's momentum sum and junction momentum before merge
        // (needed for Stan-style cross-checks between subtrees).
        scratch.rho_existing.copy_from_slice(&tree.p_sum);
        if direction > 0 {
            scratch.p_existing_junction.copy_from_slice(&tree.p_right);
        } else {
            scratch.p_existing_junction.copy_from_slice(&tree.p_left);
        }

        // Build subtree in chosen direction
        if direction > 0 {
            scratch.edge_state.q.copy_from_slice(&tree.q_right);
            scratch.edge_state.p.copy_from_slice(&tree.p_right);
            scratch.edge_state.potential = 0.0;
            scratch.edge_state.grad_potential.copy_from_slice(&tree.grad_right);
        } else {
            scratch.edge_state.q.copy_from_slice(&tree.q_left);
            scratch.edge_state.p.copy_from_slice(&tree.p_left);
            scratch.edge_state.potential = 0.0;
            scratch.edge_state.grad_potential.copy_from_slice(&tree.grad_left);
        }

        let subtree =
            build_tree(integrator, &scratch.edge_state, depth, direction, h0, metric, rng)?;

        // Save subtree's junction momentum and momentum sum
        if direction > 0 {
            scratch.p_subtree_junction.copy_from_slice(&subtree.p_left);
        } else {
            scratch.p_subtree_junction.copy_from_slice(&subtree.p_right);
        }
        scratch.rho_subtree.copy_from_slice(&subtree.p_sum);

        // Multinomial merge: accept subtree proposal with probability
        // exp(subtree.log_sum_weight - log_sum_weight_existing) (Stan-style progressive sampling).
        let accept_subtree =
            prob_select_outer_progressive(tree.log_sum_weight, subtree.log_sum_weight)
                .clamp(0.0, 1.0);
        let new_log_sum_weight = log_sum_exp(tree.log_sum_weight, subtree.log_sum_weight);
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

        // Merge p_sum (generalized U-turn criterion)
        for (ps, ss) in tree.p_sum.iter_mut().zip(subtree.p_sum.iter()) {
            *ps += *ss;
        }

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

        // Increment depth BEFORE checking U-turn (matches Stan: depth counts
        // completed doublings, so the reported value includes this iteration).
        depth += 1;

        // Stan-style generalized U-turn check (3 criteria).
        //
        // After merging, the existing tree and subtree form left/right halves
        // (depending on direction). We check:
        // 1. Full merged trajectory (rho_total against overall endpoints)
        // 2. Left start to junction (rho_left + p_right_junction)
        // 3. Junction to right end (rho_right + p_left_junction)
        let turning1 = is_turning(&tree.p_sum, &tree.p_left, &tree.p_right, metric);

        // Map existing/subtree to absolute left/right based on direction
        let (rho_left, rho_right, p_left_junction, p_right_junction) = if direction > 0 {
            // existing = left, subtree = right
            (
                &scratch.rho_existing,
                &scratch.rho_subtree,
                &scratch.p_existing_junction,
                &scratch.p_subtree_junction,
            )
        } else {
            // subtree = left, existing = right
            (
                &scratch.rho_subtree,
                &scratch.rho_existing,
                &scratch.p_subtree_junction,
                &scratch.p_existing_junction,
            )
        };

        for j in 0..dim {
            scratch.rho_cross[j] = rho_left[j] + p_right_junction[j];
        }
        let turning2 = is_turning(&scratch.rho_cross, &tree.p_left, p_right_junction, metric);

        for j in 0..dim {
            scratch.rho_cross[j] = rho_right[j] + p_left_junction[j];
        }
        let turning3 = is_turning(&scratch.rho_cross, p_left_junction, &tree.p_right, metric);

        if turning1 || turning2 || turning3 {
            tree.turning = true;
            break;
        }
        if tree.divergent || tree.turning {
            break;
        }
    }

    let n_total = tree.n_leapfrog.max(1) as f64;
    let mut accept_prob = tree.sum_accept_prob / n_total;
    if !accept_prob.is_finite() {
        accept_prob = 0.0;
    }
    accept_prob = accept_prob.clamp(0.0, 1.0);

    // Defensive: a non-finite proposal position would break downstream transforms
    // (e.g. mapping unconstrained -> constrained) and should be treated as a hard divergence.
    if tree.q_proposal.iter().any(|v| !v.is_finite()) {
        return Ok(NutsTransition {
            q: current.q.clone(),
            potential: current.potential,
            grad_potential: current.grad_potential.clone(),
            depth,
            divergent: true,
            accept_prob: 0.0,
            energy: h0,
            n_leapfrog: tree.n_leapfrog,
        });
    }

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

/// Clamp non-finite values in unconstrained coordinates to bounded values.
fn clamp_non_finite(z: &mut [f64]) {
    const Z_CLAMP: f64 = 20.0;
    for zi in z.iter_mut() {
        if zi.is_finite() {
            continue;
        }
        *zi = if zi.is_nan() {
            0.0
        } else if *zi == f64::NEG_INFINITY {
            -Z_CLAMP
        } else if *zi == f64::INFINITY {
            Z_CLAMP
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod nuts_numerics_tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_handles_infinities() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(f64::INFINITY, 0.0), f64::INFINITY);
        assert_eq!(log_sum_exp(0.0, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_prob_select_outer_basic() {
        // Equal weights -> 0.5
        let p = prob_select_outer(0.0, 0.0);
        assert!((p - 0.5).abs() < 1e-12);

        // Outer dominates -> ~1
        let p = prob_select_outer(-100.0, 0.0);
        assert!(p > 0.999);

        // Inner dominates -> ~0
        let p = prob_select_outer(0.0, -100.0);
        assert!(p < 0.001);
    }

    #[test]
    fn test_prob_select_outer_progressive_basic() {
        // Equal weights -> always accept (ratio 1).
        let p = prob_select_outer_progressive(0.0, 0.0);
        assert!((p - 1.0).abs() < 1e-12);

        // Subtree bigger -> always accept.
        let p = prob_select_outer_progressive(0.0, 1.0);
        assert!((p - 1.0).abs() < 1e-12);

        // Subtree smaller -> ratio.
        let p = prob_select_outer_progressive(0.0, -2.0);
        assert!((p - (-2.0f64).exp()).abs() < 1e-12);
    }
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

    // ---------- Initialization strategy ----------
    let mut pathfinder_inv_mass: Option<Vec<f64>> = None;

    let z_init: Vec<f64> = match config.init_strategy {
        InitStrategy::Random => {
            // Stan-style: Uniform(-2, 2) in unconstrained space, independent per chain.
            // Validates that the initial point has finite log-density; retries up to
            // 100 times with fresh draws (matching CmdStan behavior).
            let mut z = vec![0.0; dim];
            let mut ok = false;
            for _ in 0..100 {
                for zi in z.iter_mut() {
                    *zi = rng.random::<f64>() * 4.0 - 2.0; // Uniform(-2, 2)
                }
                // Validate: posterior must be finite at this point
                let theta = match posterior.to_constrained(&z) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                match model.nll(&theta) {
                    Ok(v) if v.is_finite() => {
                        ok = true;
                        break;
                    }
                    _ => continue,
                }
            }
            if !ok {
                // Fall back to model's default init (e.g. parameter_init)
                let theta_init = model.parameter_init();
                let mut zf = posterior.to_unconstrained(&theta_init)?;
                clamp_non_finite(&mut zf);
                zf
            } else {
                z
            }
        }
        InitStrategy::Mle => {
            // MLE initialization: find the mode, then optionally jitter.
            let theta_init: Vec<f64> = {
                let mle = crate::mle::MaximumLikelihoodEstimator::new();
                match mle.fit_minimum(model) {
                    Ok(r) if r.converged => r.parameters,
                    _ => model.parameter_init(),
                }
            };
            let mut z = posterior.to_unconstrained(&theta_init)?;
            clamp_non_finite(&mut z);

            // Apply jitter if requested
            let init_modes = (config.init_jitter > 0.0) as u8
                + config.init_jitter_rel.is_some() as u8
                + config.init_overdispersed_rel.is_some() as u8;
            if init_modes > 1 {
                return Err(ns_core::Error::Validation(
                    "init_jitter, init_jitter_rel, init_overdispersed_rel are mutually exclusive"
                        .to_string(),
                ));
            }

            if let Some(frac) = config.init_overdispersed_rel.filter(|&f| f > 0.0) {
                use rand_distr::{Distribution, Normal};
                let bounds = model.parameter_bounds();
                let jac = posterior.transform().jacobian_diag(&z);
                let mut out = Vec::with_capacity(dim);
                for i in 0..dim {
                    let (lo, hi) = bounds[i];
                    let lo_finite = lo > f64::NEG_INFINITY;
                    let hi_finite = hi < f64::INFINITY;
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
                        (1.0 + z[i].abs()) * frac
                    };
                    z_sigma = z_sigma.clamp(1e-6, 20.0);
                    let normal = Normal::new(0.0, z_sigma).unwrap();
                    out.push(z[i] + normal.sample(&mut rng));
                }
                out
            } else if let Some(frac) = config.init_jitter_rel.filter(|&f| f > 0.0) {
                use rand_distr::{Distribution, Normal};
                let bounds = model.parameter_bounds();
                let jac = posterior.transform().jacobian_diag(&z);
                let mut out = Vec::with_capacity(dim);
                for i in 0..dim {
                    let (lo, hi) = bounds[i];
                    let lo_finite = lo > f64::NEG_INFINITY;
                    let hi_finite = hi < f64::INFINITY;
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
                        (1.0 + z[i].abs()) * frac
                    };
                    z_sigma = z_sigma.clamp(1e-6, 5.0);
                    let normal = Normal::new(0.0, z_sigma).unwrap();
                    out.push(z[i] + normal.sample(&mut rng));
                }
                out
            } else if config.init_jitter > 0.0 {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(0.0, config.init_jitter).unwrap();
                z.iter().map(|&zi| zi + normal.sample(&mut rng)).collect()
            } else {
                z
            }
        }
        InitStrategy::Pathfinder => {
            // Pathfinder: full MLE fit with Hessian → position + diagonal inv_mass.
            // Falls back to random init on failure.
            match crate::mams::pathfinder_init_nuts(model, &posterior, dim) {
                Ok((z, mass)) => {
                    pathfinder_inv_mass = Some(mass);
                    z
                }
                Err(_) => {
                    let mut z = vec![0.0; dim];
                    let mut ok = false;
                    for _ in 0..100 {
                        for zi in z.iter_mut() {
                            *zi = rng.random::<f64>() * 4.0 - 2.0;
                        }
                        let theta = match posterior.to_constrained(&z) {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        match model.nll(&theta) {
                            Ok(v) if v.is_finite() => {
                                ok = true;
                                break;
                            }
                            _ => continue,
                        }
                    }
                    if !ok {
                        let theta_init = model.parameter_init();
                        let mut zf = posterior.to_unconstrained(&theta_init)?;
                        clamp_non_finite(&mut zf);
                        zf
                    } else {
                        z
                    }
                }
            }
        }
    };

    let metric = if let Some(inv_mass) = pathfinder_inv_mass {
        crate::hmc::Metric::Diag(inv_mass)
    } else {
        crate::hmc::Metric::identity(dim)
    };
    let init_eps = find_reasonable_step_size(&posterior, &z_init, &metric, &mut rng);

    let mut adaptation =
        WindowedAdaptation::new(dim, n_warmup, config.target_accept, init_eps, config.metric_type);
    adaptation.set_metric(metric.clone());

    let integrator = LeapfrogIntegrator::new(&posterior, init_eps, metric.clone());

    // Initialize state
    let mut state = integrator
        .init_state(z_init)
        .map_err(|e| ns_core::Error::Validation(format!("NUTS init_state failed: {e}")))?;
    let mut last_good_q = state.q.clone();
    let mut last_good_potential = state.potential;
    let mut last_good_grad = state.grad_potential.clone();

    // Warmup
    for i in 0..n_warmup {
        let eps = adaptation.step_size();
        let metric = adaptation.metric().clone();
        let warmup_integrator = LeapfrogIntegrator::new(&posterior, eps, metric);

        let transition =
            nuts_transition(&warmup_integrator, &state, config.max_treedepth, &mut rng)?;

        state.q = transition.q;
        state.potential = transition.potential;
        state.grad_potential = transition.grad_potential;

        let mut accept_prob = transition.accept_prob;
        if state.q.iter().any(|v| !v.is_finite()) {
            // Defensive: never allow non-finite unconstrained positions to escape warmup.
            // Treat as a hard divergence and keep the previous valid state.
            state.q = last_good_q.clone();
            state.potential = last_good_potential;
            state.grad_potential = last_good_grad.clone();
            accept_prob = 0.0;
        } else {
            last_good_q.clone_from(&state.q);
            last_good_potential = state.potential;
            last_good_grad.clone_from(&state.grad_potential);
        }

        let mass_updated = adaptation.update(i, &state.q, accept_prob);

        // Stan-exact: re-search for a reasonable step size after every metric
        // update (matches `init_stepsize()` + `set_mu()` + `restart()` in
        // Stan's `adapt_diag_e_nuts::transition()`).
        if mass_updated {
            let new_eps =
                find_reasonable_step_size(&posterior, &state.q, adaptation.metric(), &mut rng);
            adaptation.reinit_stepsize(new_eps);
        }
    }

    // Sampling with fixed adapted parameters
    let final_eps = adaptation.adapted_step_size();
    let final_metric = adaptation.metric().clone();
    let jitter = config.stepsize_jitter.clamp(0.0, 1.0);
    let use_jitter = jitter > 0.0;

    // Pre-build integrator for the common no-jitter case (avoids per-iteration clone).
    let fixed_integrator = if !use_jitter {
        Some(LeapfrogIntegrator::new(&posterior, final_eps, final_metric.clone()))
    } else {
        None
    };

    let mut draws_unconstrained = Vec::with_capacity(n_samples);
    let mut draws_constrained = Vec::with_capacity(n_samples);
    let mut divergences = Vec::with_capacity(n_samples);
    let mut tree_depths = Vec::with_capacity(n_samples);
    let mut accept_probs = Vec::with_capacity(n_samples);
    let mut energies = Vec::with_capacity(n_samples);
    let mut leapfrog_counts = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Optional step-size jittering (Stan-compatible: eps * (1 + j * U(-1,1)))
        let jittered_integrator;
        let integrator_ref = if let Some(ref fi) = fixed_integrator {
            fi
        } else {
            let u: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let eps = final_eps * (1.0 + jitter * u);
            jittered_integrator = LeapfrogIntegrator::new(&posterior, eps, final_metric.clone());
            &jittered_integrator
        };

        let transition = nuts_transition(integrator_ref, &state, config.max_treedepth, &mut rng)?;

        let mut divergent = transition.divergent;
        let mut accept_prob = transition.accept_prob;
        let depth = transition.depth;
        let energy = transition.energy;

        state.q = transition.q;
        state.potential = transition.potential;
        state.grad_potential = transition.grad_potential;

        if state.q.iter().any(|v| !v.is_finite()) {
            // Same defensive policy as warmup: reject the transition and keep last good state.
            state.q = last_good_q.clone();
            state.potential = last_good_potential;
            state.grad_potential = last_good_grad.clone();
            divergent = true;
            accept_prob = 0.0;
        } else {
            last_good_q.clone_from(&state.q);
            last_good_potential = state.potential;
            last_good_grad.clone_from(&state.grad_potential);
        }

        // `state.q` should be finite due to the guard above, but keep this code path robust:
        // if any non-finite slips through (or `last_good_q` is accidentally non-finite),
        // do not hard-fail the entire sampler.
        let constrained = match posterior.to_constrained(&state.q) {
            Ok(theta) => theta,
            Err(ns_core::Error::Validation(msg))
                if msg.contains("must contain only finite values") =>
            {
                // Treat as divergence and keep the last known-good state.
                divergent = true;
                accept_prob = 0.0;
                state.q = last_good_q.clone();
                state.potential = last_good_potential;
                state.grad_potential = last_good_grad.clone();
                posterior
                    .to_constrained(&state.q)
                    .unwrap_or_else(|_| vec![f64::NAN; posterior.dim()])
            }
            Err(e) => {
                return Err(ns_core::Error::Validation(format!("NUTS to_constrained failed: {e}")));
            }
        };

        draws_unconstrained.push(state.q.clone());
        draws_constrained.push(constrained);
        divergences.push(divergent);
        tree_depths.push(depth);
        accept_probs.push(accept_prob);
        energies.push(energy);
        leapfrog_counts.push(transition.n_leapfrog);
    }

    let mass_diag: Vec<f64> = final_metric.mass_diag();

    Ok(crate::chain::Chain {
        draws_unconstrained,
        draws_constrained,
        divergences,
        tree_depths,
        accept_probs,
        energies,
        n_leapfrog: leapfrog_counts,
        max_treedepth: config.max_treedepth,
        step_size: final_eps,
        mass_diag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_core::traits::{LogDensityModel, PreparedModelRef};
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
        let metric = crate::hmc::Metric::identity(dim);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.1, metric);

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let state = integrator.init_state(z_init).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let transition = nuts_transition(&integrator, &state, 10, &mut rng).unwrap();

        assert!(transition.depth <= 10);
        assert!(
            transition.accept_prob.is_finite()
                && transition.accept_prob >= 0.0
                && transition.accept_prob <= 1.0
        );
        assert!(transition.n_leapfrog > 0);
    }

    #[test]
    fn test_log_sum_exp_handles_neg_inf_and_nan() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(f64::NAN, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, f64::NAN), f64::NEG_INFINITY);
        let v = log_sum_exp(0.0, 0.0);
        assert!(v.is_finite() && (v - (2.0_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_nuts_deterministic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let dim = posterior.dim();
        let metric = crate::hmc::Metric::identity(dim);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.1, metric);

        let theta_init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let z_init = posterior.to_unconstrained(&theta_init).unwrap();
        let state = integrator.init_state(z_init).unwrap();

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let t1 = nuts_transition(&integrator, &state, 10, &mut rng1).unwrap();

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let t2 = nuts_transition(&integrator, &state, 10, &mut rng2).unwrap();

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
            ..Default::default()
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
            ..Default::default()
        };
        let chain1 = sample_nuts(&model, 50, 20, 123, config.clone()).unwrap();
        let chain2 = sample_nuts(&model, 50, 20, 123, config).unwrap();

        assert_eq!(
            chain1.draws_constrained, chain2.draws_constrained,
            "Same seed should produce identical draws"
        );
        assert_eq!(chain1.energies, chain2.energies, "Energy series should be deterministic");
    }

    #[test]
    fn test_sample_nuts_clamps_non_finite_z_init() {
        // A minimal bounded model whose optimum is exactly at the lower bound.
        // The inverse transform produces -inf in unconstrained space; sampling
        // should clamp to a large finite negative value and proceed.
        struct BoundaryModel;

        impl LogDensityModel for BoundaryModel {
            type Prepared<'a>
                = PreparedModelRef<'a, Self>
            where
                Self: 'a;

            fn dim(&self) -> usize {
                1
            }

            fn parameter_names(&self) -> Vec<String> {
                vec!["x".to_string()]
            }

            fn parameter_bounds(&self) -> Vec<(f64, f64)> {
                vec![(0.0, 10.0)]
            }

            fn parameter_init(&self) -> Vec<f64> {
                vec![0.0]
            }

            fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
                let x = params[0];
                Ok(x * x)
            }

            fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
                let x = params[0];
                Ok(vec![2.0 * x])
            }

            fn prepared(&self) -> Self::Prepared<'_> {
                PreparedModelRef::new(self)
            }
        }

        let model = BoundaryModel;
        let config = NutsConfig {
            max_treedepth: 6,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
            ..Default::default()
        };

        let chain = sample_nuts(&model, 10, 5, 123, config).unwrap();
        assert_eq!(chain.draws_unconstrained.len(), 5);

        for draw in &chain.draws_unconstrained {
            assert!(draw[0].is_finite(), "unconstrained draw must be finite");
        }
        for draw in &chain.draws_constrained {
            assert!(
                draw[0].is_finite() && draw[0] >= 0.0 && draw[0] <= 10.0,
                "constrained draw should stay within bounds: {}",
                draw[0]
            );
        }
    }

    /// Quality gate: full pipeline must produce well-converged samples on the
    /// simple workspace.  This validates R-hat, ESS, divergence rate, E-BFMI,
    /// and posterior mean proximity to MLE.
    #[test]
    #[ignore = "slow (~10s); run with `cargo test -p ns-inference test_nuts_quality_gate -- --ignored`"]
    fn test_nuts_quality_gate() {
        use crate::chain::sample_nuts_multichain;
        use crate::diagnostics::compute_diagnostics;

        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let config = NutsConfig {
            max_treedepth: 12,
            target_accept: 0.9,
            init_jitter: 0.5,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
            ..Default::default()
        };
        let result = sample_nuts_multichain(&model, 4, 1000, 1000, 42, config).unwrap();

        let diag = compute_diagnostics(&result);

        // HistFactory models can require longer runs to reach tight standards-like thresholds.
        // Keep this gate meaningful but realistic for a slow regression test.
        //
        // Note: Apex2 uses an even looser default for HistFactory in strict mode; see
        // `tests/apex2_nuts_quality_report.py`.
        //
        // R-hat < 1.5 for all parameters
        for (i, &rhat) in diag.r_hat.iter().enumerate() {
            assert!(
                rhat < 1.5,
                "R-hat for param {} = {} (should be < 1.5)",
                result.param_names[i],
                rhat,
            );
        }

        // Bulk ESS > 10 for all parameters
        for (i, &ess) in diag.ess_bulk.iter().enumerate() {
            assert!(
                ess > 10.0,
                "Bulk ESS for param {} = {} (should be > 10)",
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

    /// Stress gate: Neal's funnel (pathological geometry) should not produce
    /// exploding energy or catastrophic divergence/treedepth behavior.
    ///
    /// This is intentionally ignored by default (slow-ish) and is meant as a
    /// manual/nightly regression check.
    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_nuts_funnel_stress_gate -- --ignored`"]
    fn test_nuts_funnel_stress_gate() {
        use crate::chain::sample_nuts_multichain;
        use crate::diagnostics::compute_diagnostics;

        #[derive(Clone, Copy)]
        struct NealsFunnel2D;

        impl LogDensityModel for NealsFunnel2D {
            type Prepared<'a>
                = PreparedModelRef<'a, Self>
            where
                Self: 'a;

            fn dim(&self) -> usize {
                2
            }

            fn parameter_names(&self) -> Vec<String> {
                vec!["y".to_string(), "x".to_string()]
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
                        "NealsFunnel2D expects 2 params, got {}",
                        params.len()
                    )));
                }
                let y = params[0];
                let x = params[1];
                if !(y.is_finite() && x.is_finite()) {
                    return Err(ns_core::Error::Validation(
                        "NealsFunnel2D params must be finite".to_string(),
                    ));
                }

                // y ~ Normal(0, 3)
                // x | y ~ Normal(0, exp(y/2))
                //
                // nll = 0.5*(y/3)^2 + ln(3*sqrt(2pi))
                //     + 0.5*(x/exp(y/2))^2 + ln(exp(y/2)*sqrt(2pi))
                let ln2pi = (2.0 * std::f64::consts::PI).ln();
                let nll_y = 0.5 * (y * y) / 9.0 + 3.0_f64.ln() + 0.5 * ln2pi;
                let nll_x = 0.5 * x * x * (-y).exp() + 0.5 * y + 0.5 * ln2pi;
                Ok(nll_y + nll_x)
            }

            fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
                if params.len() != 2 {
                    return Err(ns_core::Error::Validation(format!(
                        "NealsFunnel2D expects 2 params, got {}",
                        params.len()
                    )));
                }
                let y = params[0];
                let x = params[1];
                if !(y.is_finite() && x.is_finite()) {
                    return Err(ns_core::Error::Validation(
                        "NealsFunnel2D params must be finite".to_string(),
                    ));
                }

                let exp_neg_y = (-y).exp();
                let dy = y / 9.0 - 0.5 * x * x * exp_neg_y + 0.5;
                let dx = x * exp_neg_y;
                Ok(vec![dy, dx])
            }

            fn prepared(&self) -> Self::Prepared<'_> {
                PreparedModelRef::new(self)
            }
        }

        let model = NealsFunnel2D;
        let config = NutsConfig {
            max_treedepth: 10,
            target_accept: 0.9,
            init_jitter: 1.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
            ..Default::default()
        };

        let result = sample_nuts_multichain(&model, 4, 600, 600, 123, config).unwrap();
        let diag = compute_diagnostics(&result);

        // No NaNs/infs in diagnostics or per-draw energies.
        assert!(diag.divergence_rate.is_finite(), "divergence_rate must be finite");
        for (chain_id, c) in result.chains.iter().enumerate() {
            for (i, &e) in c.energies.iter().enumerate() {
                assert!(
                    e.is_finite(),
                    "energy must be finite (chain={}, draw={}): {}",
                    chain_id,
                    i,
                    e
                );
            }
            for (i, &d) in c.tree_depths.iter().enumerate() {
                assert!(
                    d <= c.max_treedepth,
                    "treedepth exceeds max (chain={}, draw={}): {} > {}",
                    chain_id,
                    i,
                    d,
                    c.max_treedepth
                );
            }
        }

        // This is a stress model; we don't demand zero divergences, but it should not
        // be catastrophic in the default config.
        assert!(
            diag.divergence_rate < 0.30,
            "divergence_rate too high for funnel stress gate: {}",
            diag.divergence_rate
        );
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
            let mu = params.first().copied().ok_or_else(|| {
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
            ..Default::default()
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
            let mu1 = params.first().copied().ok_or_else(|| {
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
            ..Default::default()
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
            ..Default::default()
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
