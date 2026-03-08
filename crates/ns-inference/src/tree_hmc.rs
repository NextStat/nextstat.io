//! Shared tree-based HMC transition machinery.
//!
//! This module holds the multinomial tree-building core currently used by
//! NUTS. It is separated from `nuts.rs` so future tree-based samplers
//! (including WALNUTS) can share the same proposal accumulation and
//! generalized U-turn bookkeeping without cloning the implementation.

use crate::hmc::{HamiltonianStepper, HmcState, Metric};
use ns_core::Result;
use rand::Rng;

/// Result of one tree-based HMC transition.
#[derive(Debug, Clone)]
pub(crate) struct TreeTransition {
    pub q: Vec<f64>,
    pub potential: f64,
    pub grad_potential: Vec<f64>,
    pub depth: usize,
    pub divergent: bool,
    pub accept_prob: f64,
    pub energy: f64,
    pub n_leapfrog: usize,
}

/// Internal tree node for balanced binary tree-building.
struct TreeNode {
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
    /// Sum of momenta across all leaves in this sub-tree.
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
fn is_turning(rho: &[f64], p_left: &[f64], p_right: &[f64], metric: &Metric) -> bool {
    match metric {
        Metric::Diag(inv_mass) => {
            let mut dot_left = 0.0_f64;
            let mut dot_right = 0.0_f64;
            for i in 0..rho.len() {
                let w = rho[i] * inv_mass[i];
                dot_left += w * p_left[i];
                dot_right += w * p_right[i];
            }
            if !dot_left.is_finite() || !dot_right.is_finite() {
                return true;
            }
            dot_left < 0.0 || dot_right < 0.0
        }
        _ => {
            let v_left = metric.mul_inv_mass(p_left);
            let v_right = metric.mul_inv_mass(p_right);
            let dot_left: f64 = rho.iter().zip(v_left.iter()).map(|(&r, &v)| r * v).sum();
            let dot_right: f64 = rho.iter().zip(v_right.iter()).map(|(&r, &v)| r * v).sum();
            if !dot_left.is_finite() || !dot_right.is_finite() {
                return true;
            }
            dot_left < 0.0 || dot_right < 0.0
        }
    }
}

/// U-turn criterion for `rho = rho_a + rho_b` without materializing a
/// temporary `Vec` on the hot diagonal-metric path.
#[inline]
fn is_turning_sum(
    rho_a: &[f64],
    rho_b: &[f64],
    p_left: &[f64],
    p_right: &[f64],
    metric: &Metric,
) -> bool {
    match metric {
        Metric::Diag(inv_mass) => {
            let mut dot_left = 0.0_f64;
            let mut dot_right = 0.0_f64;
            for i in 0..rho_a.len() {
                let rho = rho_a[i] + rho_b[i];
                let w = rho * inv_mass[i];
                dot_left += w * p_left[i];
                dot_right += w * p_right[i];
            }
            if !dot_left.is_finite() || !dot_right.is_finite() {
                return true;
            }
            dot_left < 0.0 || dot_right < 0.0
        }
        _ => {
            let mut rho = vec![0.0; rho_a.len()];
            for i in 0..rho_a.len() {
                rho[i] = rho_a[i] + rho_b[i];
            }
            is_turning(&rho, p_left, p_right, metric)
        }
    }
}

pub(crate) fn log_sum_exp(a: f64, b: f64) -> f64 {
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
pub(crate) fn prob_select_outer(logw_inner: f64, logw_outer: f64) -> f64 {
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

    let d = a - b;
    if !d.is_finite() {
        return 0.0;
    }
    if d > 0.0 {
        let e = (-d).exp();
        e / (1.0 + e)
    } else {
        let e = d.exp();
        1.0 / (1.0 + e)
    }
}

/// Stan-style progressive sampling when joining a new subtree at top-level.
pub(crate) fn prob_select_outer_progressive(logw_existing: f64, logw_subtree: f64) -> f64 {
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

    let d = b - a;
    if !d.is_finite() {
        return 0.0;
    }
    if d >= 0.0 { 1.0 } else { d.exp().clamp(0.0, 1.0) }
}

/// Pre-allocated scratch buffers for the top-level tree-doubling loop.
struct TreeTransitionScratch {
    rho_existing: Vec<f64>,
    p_existing_junction: Vec<f64>,
    edge_state: HmcState,
    p_subtree_junction: Vec<f64>,
    rho_subtree: Vec<f64>,
    rho_cross: Vec<f64>,
}

impl TreeTransitionScratch {
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

fn build_leaf<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    state: &HmcState,
    direction: i32,
    h0: f64,
    metric: &Metric,
) -> Result<TreeNode> {
    let mut new_state = state.clone();

    if integrator.step_dir(&mut new_state, direction).is_err() {
        let dim = state.q.len();
        return Ok(TreeNode {
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
    let log_weight = if divergent { f64::NEG_INFINITY } else { -energy_error };
    let accept_prob = if !energy_error.is_finite() { 0.0 } else { (-energy_error).exp().min(1.0) };

    Ok(TreeNode {
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
        p_sum: new_state.p.clone(),
        depth: 0,
        n_leapfrog: 1,
        divergent,
        turning: false,
        sum_accept_prob: accept_prob,
    })
}

fn build_tree<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    state: &HmcState,
    depth: usize,
    direction: i32,
    h0: f64,
    metric: &Metric,
    rng: &mut impl Rng,
) -> Result<TreeNode> {
    if depth == 0 {
        return build_leaf(integrator, state, direction, h0, metric);
    }

    let mut inner = build_tree(integrator, state, depth - 1, direction, h0, metric, rng)?;

    if inner.divergent || inner.turning {
        return Ok(inner);
    }

    let edge_state = if direction > 0 {
        HmcState {
            q: inner.q_right.clone(),
            p: inner.p_right.clone(),
            potential: 0.0,
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
    let new_log_sum_weight = log_sum_exp(inner.log_sum_weight, outer.log_sum_weight);

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

    let (p_left_merged, p_right_merged, p_start, p_end, p_init_junction, p_final_junction) =
        if direction > 0 {
            (
                &inner.p_left,
                &outer.p_right,
                &inner.p_left,
                &outer.p_right,
                &inner.p_right,
                &outer.p_left,
            )
        } else {
            (
                &outer.p_left,
                &inner.p_right,
                &inner.p_right,
                &outer.p_left,
                &inner.p_left,
                &outer.p_right,
            )
        };

    let turning1 =
        is_turning_sum(&inner.p_sum, &outer.p_sum, p_left_merged, p_right_merged, metric);
    let turning2 =
        is_turning_sum(&inner.p_sum, p_final_junction, p_start, p_final_junction, metric);
    let turning3 = is_turning_sum(&outer.p_sum, p_init_junction, p_init_junction, p_end, metric);

    for (ps, os) in inner.p_sum.iter_mut().zip(outer.p_sum.iter()) {
        *ps += *os;
    }

    if direction > 0 {
        inner.q_right = outer.q_right;
        inner.p_right = outer.p_right;
        inner.grad_right = outer.grad_right;
    } else {
        inner.q_left = outer.q_left;
        inner.p_left = outer.p_left;
        inner.grad_left = outer.grad_left;
    }

    inner.turning = inner.turning || outer.turning || turning1 || turning2 || turning3;
    inner.depth = depth;
    Ok(inner)
}

/// Run one balanced tree-based HMC transition with multinomial proposal
/// selection and generalized U-turn termination.
pub(crate) fn tree_transition<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    current: &HmcState,
    max_treedepth: usize,
    rng: &mut impl Rng,
) -> Result<TreeTransition> {
    let metric = integrator.metric();

    let mut state = current.clone();
    state.p = metric.sample_momentum(rng);

    let h0 = state.hamiltonian(metric);
    if !h0.is_finite() {
        return Err(ns_core::Error::Validation(
            "non-finite initial Hamiltonian in tree-based HMC transition".to_string(),
        ));
    }

    let dim = state.q.len();
    let mut tree = TreeNode {
        q_left: state.q.clone(),
        p_left: state.p.clone(),
        grad_left: state.grad_potential.clone(),
        q_right: state.q.clone(),
        p_right: state.p.clone(),
        grad_right: state.grad_potential.clone(),
        q_proposal: state.q.clone(),
        potential_proposal: state.potential,
        grad_proposal: state.grad_potential.clone(),
        log_sum_weight: 0.0,
        p_sum: state.p.clone(),
        depth: 0,
        n_leapfrog: 0,
        divergent: false,
        turning: false,
        sum_accept_prob: 0.0,
    };

    let mut scratch = TreeTransitionScratch::new(dim);
    let mut depth: usize = 0;

    while depth < max_treedepth {
        let direction: i32 = if rng.random::<bool>() { 1 } else { -1 };

        scratch.rho_existing.copy_from_slice(&tree.p_sum);
        if direction > 0 {
            scratch.p_existing_junction.copy_from_slice(&tree.p_right);
        } else {
            scratch.p_existing_junction.copy_from_slice(&tree.p_left);
        }

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

        if direction > 0 {
            scratch.p_subtree_junction.copy_from_slice(&subtree.p_left);
        } else {
            scratch.p_subtree_junction.copy_from_slice(&subtree.p_right);
        }
        scratch.rho_subtree.copy_from_slice(&subtree.p_sum);

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

        for (ps, ss) in tree.p_sum.iter_mut().zip(subtree.p_sum.iter()) {
            *ps += *ss;
        }

        if direction > 0 {
            tree.q_right = subtree.q_right;
            tree.p_right = subtree.p_right;
            tree.grad_right = subtree.grad_right;
        } else {
            tree.q_left = subtree.q_left;
            tree.p_left = subtree.p_left;
            tree.grad_left = subtree.grad_left;
        }

        depth += 1;

        let turning1 = is_turning(&tree.p_sum, &tree.p_left, &tree.p_right, metric);

        let (rho_left, rho_right, p_left_junction, p_right_junction) = if direction > 0 {
            (
                &scratch.rho_existing,
                &scratch.rho_subtree,
                &scratch.p_existing_junction,
                &scratch.p_subtree_junction,
            )
        } else {
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

    if tree.q_proposal.iter().any(|v| !v.is_finite()) {
        return Ok(TreeTransition {
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

    Ok(TreeTransition {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_handles_infinities() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(f64::INFINITY, 0.0), f64::INFINITY);
        assert_eq!(log_sum_exp(0.0, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_prob_select_outer_basic() {
        let p = prob_select_outer(0.0, 0.0);
        assert!((p - 0.5).abs() < 1e-12);

        let p = prob_select_outer(-100.0, 0.0);
        assert!(p > 0.999);

        let p = prob_select_outer(0.0, -100.0);
        assert!(p < 0.001);
    }

    #[test]
    fn test_prob_select_outer_progressive_basic() {
        let p = prob_select_outer_progressive(0.0, 0.0);
        assert!((p - 1.0).abs() < 1e-12);

        let p = prob_select_outer_progressive(0.0, 1.0);
        assert!((p - 1.0).abs() < 1e-12);

        let p = prob_select_outer_progressive(0.0, -2.0);
        assert!((p - (-2.0f64).exp()).abs() < 1e-12);
    }
}
