//! High-Dimensional Fixed Effects (HDFE) solver via Method of Alternating
//! Projections (MAP).
//!
//! Provides exact multi-way fixed-effects absorption for both balanced and
//! unbalanced panels. Replaces single-pass iterative demeaning with a
//! convergent MAP algorithm that iterates until all group means are below
//! a configurable tolerance.
//!
//! # References
//!
//! - Correia (2017), "Linear Models with High-Dimensional Fixed Effects:
//!   An Efficient and Feasible Estimator." Working paper.
//! - Gaure (2013), "OLS with multiple high dimensional category variables."
//!   *Computational Statistics & Data Analysis*.
//! - Guimarães & Portugal (2010), "A simple feasible procedure to fit models
//!   with high-dimensional fixed effects." *Stata Journal*.

use ns_core::{Error, Result};
use std::collections::HashSet;

/// Default convergence tolerance for MAP iterations (L∞ of group means).
const DEFAULT_TOL: f64 = 1e-8;

/// Maximum MAP iterations (safety bound).
const DEFAULT_MAX_ITER: usize = 10_000;

/// Solver for absorbing high-dimensional fixed effects via the Method of
/// Alternating Projections (MAP).
///
/// Supports an arbitrary number of FE dimensions (entity, time, industry, …).
/// Each dimension is specified as a `Vec<usize>` mapping observation index to
/// a 0-based group level.
///
/// # Algorithm
///
/// For a single FE dimension, one demeaning pass is exact. For ≥ 2 dimensions,
/// the solver alternates between projections (subtracting group means) until
/// the maximum absolute group mean across all dimensions falls below `tol`.
///
/// # Degrees of freedom
///
/// For 2-way FE, the absorbed df is computed exactly via Union-Find on the
/// bipartite (entity, time) graph:
/// `df_absorbed = n_entity + n_time − n_connected_components`.
#[derive(Debug, Clone)]
pub struct FixedEffectsSolver {
    /// Number of observations.
    n: usize,
    /// For each FE dimension: group_of\[i\] = group index for observation i.
    group_of: Vec<Vec<usize>>,
    /// For each FE dimension: number of distinct groups.
    n_levels: Vec<usize>,
    /// For each FE dimension d, for each group g: list of observation indices.
    group_indices: Vec<Vec<Vec<usize>>>,
    /// Convergence tolerance (L∞ norm of group means).
    tol: f64,
    /// Maximum number of MAP iterations.
    max_iter: usize,
}

impl FixedEffectsSolver {
    /// Create a new HDFE solver.
    ///
    /// `groups` — one entry per FE dimension, each a `Vec<usize>` of length `n`
    /// mapping observations to 0-based group indices.
    pub fn new(groups: Vec<Vec<usize>>) -> Result<Self> {
        if groups.is_empty() {
            return Err(Error::Validation("at least one FE dimension required".into()));
        }
        let n = groups[0].len();
        if n == 0 {
            return Err(Error::Validation("n must be > 0".into()));
        }
        for (d, g) in groups.iter().enumerate() {
            if g.len() != n {
                return Err(Error::Validation(format!(
                    "FE dimension {} has length {}, expected {}",
                    d,
                    g.len(),
                    n
                )));
            }
        }

        let mut n_levels = Vec::with_capacity(groups.len());
        let mut group_indices = Vec::with_capacity(groups.len());
        for g in &groups {
            let max_g = g.iter().copied().max().unwrap_or(0);
            let nl = max_g + 1;
            n_levels.push(nl);
            let mut idx: Vec<Vec<usize>> = vec![Vec::new(); nl];
            for (i, &gi) in g.iter().enumerate() {
                idx[gi].push(i);
            }
            group_indices.push(idx);
        }

        Ok(Self {
            n,
            group_of: groups,
            n_levels,
            group_indices,
            tol: DEFAULT_TOL,
            max_iter: DEFAULT_MAX_ITER,
        })
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Number of FE dimensions.
    pub fn n_dimensions(&self) -> usize {
        self.group_of.len()
    }

    /// Total number of observations.
    pub fn n_obs(&self) -> usize {
        self.n
    }

    /// Number of levels in each FE dimension.
    pub fn levels(&self) -> &[usize] {
        &self.n_levels
    }

    /// Partial out (absorb) all fixed effects from a single vector.
    ///
    /// Returns the residual vector after removing group means iteratively
    /// until convergence (MAP algorithm).
    pub fn partial_out(&self, v: &[f64]) -> Result<Vec<f64>> {
        if v.len() != self.n {
            return Err(Error::Validation(format!("v length ({}) != n ({})", v.len(), self.n)));
        }

        let mut resid = v.to_vec();

        // Single FE dimension: one pass is exact.
        if self.group_of.len() == 1 {
            self.demean_dim(&mut resid, 0);
            return Ok(resid);
        }

        // Multiple FE dimensions: MAP with Aitken Δ² acceleration.
        //
        // Every 3 MAP sweeps we store three consecutive iterates (r0, r1, r2)
        // and apply the element-wise Aitken extrapolation:
        //   r_acc[i] = r0[i] − (r1[i]−r0[i])² / (r2[i] − 2·r1[i] + r0[i])
        // This converts linear convergence into superlinear, cutting iteration
        // count roughly in half for moderately unbalanced panels.

        let n = self.n;
        let mut r0 = vec![0.0_f64; n];
        let mut r1 = vec![0.0_f64; n];
        let mut phase = 0u8; // 0,1,2 → accumulating r0/r1/r2 for Aitken

        for _iter in 0..self.max_iter {
            match phase {
                0 => {
                    r0.copy_from_slice(&resid);
                    phase = 1;
                }
                1 => {
                    r1.copy_from_slice(&resid);
                    phase = 2;
                }
                2 => {
                    // resid is r2 — apply Aitken Δ² element-wise.
                    for i in 0..n {
                        let denom = resid[i] - 2.0 * r1[i] + r0[i];
                        if denom.abs() > 1e-30 {
                            let delta = r1[i] - r0[i];
                            resid[i] = r0[i] - delta * delta / denom;
                        }
                    }
                    phase = 0;
                }
                _ => unreachable!(),
            }

            // MAP sweep: project onto each FE dimension.
            for d in 0..self.group_of.len() {
                self.demean_dim(&mut resid, d);
            }

            // Convergence check: max |group mean| across all dimensions.
            let max_abs_mean = self.max_group_mean_abs(&resid);
            if max_abs_mean < self.tol {
                return Ok(resid);
            }
        }

        // Did not converge within max_iter — return best approximation.
        Ok(resid)
    }

    /// Partial out fixed effects from multiple vectors simultaneously.
    pub fn partial_out_many(&self, cols: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
        cols.iter().map(|c| self.partial_out(c)).collect()
    }

    /// Compute the number of degrees of freedom absorbed by the fixed effects.
    ///
    /// - 1-way: `n_levels − 1`.
    /// - 2-way: `n_entity + n_time − n_connected_components` (exact via
    ///   Union-Find).
    /// - k-way (k > 2): `Σ n_levels_d − 1` (conservative; assumes 1 component).
    pub fn degrees_of_freedom_absorbed(&self) -> usize {
        let k = self.group_of.len();
        if k == 1 {
            return self.n_levels[0].saturating_sub(1);
        }

        let n_components = if k == 2 {
            self.count_connected_components_2way()
        } else {
            1 // conservative for k > 2
        };

        let total_levels: usize = self.n_levels.iter().sum();
        total_levels.saturating_sub(n_components)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Single-pass demeaning for one FE dimension (in-place).
    fn demean_dim(&self, v: &mut [f64], d: usize) {
        for group_obs in &self.group_indices[d] {
            if group_obs.is_empty() {
                continue;
            }
            let ni = group_obs.len() as f64;
            let mut sum = 0.0;
            for &i in group_obs {
                sum += v[i];
            }
            let mean = sum / ni;
            for &i in group_obs {
                v[i] -= mean;
            }
        }
    }

    /// Maximum absolute group mean across all FE dimensions.
    fn max_group_mean_abs(&self, v: &[f64]) -> f64 {
        let mut max_val = 0.0_f64;
        for d in 0..self.group_of.len() {
            for group_obs in &self.group_indices[d] {
                if group_obs.is_empty() {
                    continue;
                }
                let ni = group_obs.len() as f64;
                let mut sum = 0.0;
                for &i in group_obs {
                    sum += v[i];
                }
                let abs_mean = (sum / ni).abs();
                if abs_mean > max_val {
                    max_val = abs_mean;
                }
            }
        }
        max_val
    }

    /// Count connected components for 2-way FE via Union-Find on the
    /// bipartite graph (dim0, dim1).
    fn count_connected_components_2way(&self) -> usize {
        let n0 = self.n_levels[0];
        let n1 = self.n_levels[1];
        let total = n0 + n1;

        let mut parent: Vec<usize> = (0..total).collect();
        let mut rank = vec![0u8; total];

        // Each observation connects group_of[0][i] with (n0 + group_of[1][i]).
        for i in 0..self.n {
            let a = self.group_of[0][i];
            let b = n0 + self.group_of[1][i];
            uf_union(&mut parent, &mut rank, a, b);
        }

        // Count distinct roots among *used* nodes only.
        let mut used = vec![false; total];
        for i in 0..self.n {
            used[self.group_of[0][i]] = true;
            used[n0 + self.group_of[1][i]] = true;
        }

        let mut roots = HashSet::new();
        for node in 0..total {
            if used[node] {
                roots.insert(uf_find(&mut parent, node));
            }
        }
        roots.len()
    }
}

// ------------------------------------------------------------------
// Union-Find helpers (module-private)
// ------------------------------------------------------------------

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    if rank[ra] < rank[rb] {
        parent[ra] = rb;
    } else if rank[ra] > rank[rb] {
        parent[rb] = ra;
    } else {
        parent[rb] = ra;
        rank[ra] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_dim_exact_one_pass() {
        // 6 obs, 2 groups: [0,0,0,1,1,1]
        let solver = FixedEffectsSolver::new(vec![vec![0, 0, 0, 1, 1, 1]]).unwrap();
        let v = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let r = solver.partial_out(&v).unwrap();
        // Group 0 mean = 2, group 1 mean = 20
        assert!((r[0] - (-1.0)).abs() < 1e-12);
        assert!((r[1] - 0.0).abs() < 1e-12);
        assert!((r[2] - 1.0).abs() < 1e-12);
        assert!((r[3] - (-10.0)).abs() < 1e-12);
        assert!((r[4] - 0.0).abs() < 1e-12);
        assert!((r[5] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn two_way_balanced_converges() {
        // 2 entities × 3 time periods (balanced panel)
        // entity: [0,0,0,1,1,1], time: [0,1,2,0,1,2]
        let entity = vec![0, 0, 0, 1, 1, 1];
        let time = vec![0, 1, 2, 0, 1, 2];
        let solver = FixedEffectsSolver::new(vec![entity, time]).unwrap();

        // y = entity_fe + time_fe + noise
        // entity_fe: [5, 10], time_fe: [1, 2, 3]
        let y = vec![6.0, 7.0, 8.0, 11.0, 12.0, 13.0];
        let r = solver.partial_out(&y).unwrap();

        // After absorbing entity + time FE, residuals should be ~0
        for (i, &ri) in r.iter().enumerate() {
            assert!(ri.abs() < 1e-7, "resid[{}] = {} (expected ~0)", i, ri);
        }
    }

    #[test]
    fn two_way_unbalanced_converges() {
        // Unbalanced: entity 0 has 3 obs, entity 1 has 2 obs
        // entity: [0,0,0,1,1], time: [0,1,2,1,2]
        let entity = vec![0, 0, 0, 1, 1];
        let time = vec![0, 1, 2, 1, 2];
        let solver = FixedEffectsSolver::new(vec![entity, time]).unwrap();

        let y = vec![10.0, 20.0, 30.0, 25.0, 35.0];
        let r = solver.partial_out(&y).unwrap();

        // All entity means and time means of residuals should be ~0
        let entity_mean_0 = (r[0] + r[1] + r[2]) / 3.0;
        let entity_mean_1 = (r[3] + r[4]) / 2.0;
        let time_mean_1 = (r[1] + r[3]) / 2.0;
        let time_mean_2 = (r[2] + r[4]) / 2.0;
        assert!(entity_mean_0.abs() < 1e-8);
        assert!(entity_mean_1.abs() < 1e-8);
        assert!(time_mean_1.abs() < 1e-8);
        assert!(time_mean_2.abs() < 1e-8);
        // time 0 only has entity 0, so r[0] should be 0 (singleton)
        assert!(r[0].abs() < 1e-8);
    }

    #[test]
    fn degrees_of_freedom_one_way() {
        let solver = FixedEffectsSolver::new(vec![vec![0, 0, 1, 1, 2, 2]]).unwrap();
        // 3 groups → 2 absorbed df
        assert_eq!(solver.degrees_of_freedom_absorbed(), 2);
    }

    #[test]
    fn degrees_of_freedom_two_way_connected() {
        // 3 entities × 4 time periods, fully connected
        let entity = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let time = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let solver = FixedEffectsSolver::new(vec![entity, time]).unwrap();
        // 3 + 4 - 1 = 6 (1 connected component)
        assert_eq!(solver.degrees_of_freedom_absorbed(), 6);
    }

    #[test]
    fn degrees_of_freedom_two_way_disconnected() {
        // Two disconnected components:
        // Component 1: entity 0 × time {0,1}
        // Component 2: entity 1 × time {2,3}
        let entity = vec![0, 0, 1, 1];
        let time = vec![0, 1, 2, 3];
        let solver = FixedEffectsSolver::new(vec![entity, time]).unwrap();
        // 2 entities + 4 times - 2 components = 4
        assert_eq!(solver.degrees_of_freedom_absorbed(), 4);
    }

    #[test]
    fn partial_out_many_consistency() {
        let entity = vec![0, 0, 0, 1, 1, 1];
        let time = vec![0, 1, 2, 0, 1, 2];
        let solver = FixedEffectsSolver::new(vec![entity, time]).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let results = solver.partial_out_many(&[&a, &b]).unwrap();
        let r_a = solver.partial_out(&a).unwrap();
        let r_b = solver.partial_out(&b).unwrap();

        for i in 0..6 {
            assert!((results[0][i] - r_a[i]).abs() < 1e-15);
            assert!((results[1][i] - r_b[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn validation_errors() {
        assert!(FixedEffectsSolver::new(vec![]).is_err());
        assert!(FixedEffectsSolver::new(vec![vec![]]).is_err());
        assert!(FixedEffectsSolver::new(vec![vec![0, 1], vec![0]]).is_err());

        let solver = FixedEffectsSolver::new(vec![vec![0, 0, 1, 1]]).unwrap();
        assert!(solver.partial_out(&[1.0, 2.0]).is_err()); // wrong length
    }
}
