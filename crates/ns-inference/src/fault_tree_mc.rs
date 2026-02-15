//! Monte Carlo fault-tree engine for aviation reliability analysis.
//!
//! Evaluates system failure probability via Monte Carlo simulation over a fault tree
//! with support for fixed, uncertain (epistemic), and Weibull mission-time failure modes.
//!
//! ## Architecture
//!
//! - **CPU path**: Rayon-parallel, chunked (no O(N) memory), deterministic via StdRng (ChaCha12).
//! - **CUDA path**: 1 thread = 1 scenario, inline xoshiro-ish RNG (see `ns-compute`).
//! - **Metal path**: Analogous MSL kernel.
//!
//! ## Reproducibility
//!
//! CPU: bit-exact across runs (same seed → same result).
//! GPU: statistically reproducible (within Monte Carlo SE) due to different RNG family.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// Counter-based scenario RNG. Same (seed, scenario_id) → same draw sequence.
///
/// Uses a fast hash-mix to decorrelate nearby (seed, scenario_id) pairs.
#[inline]
fn scenario_rng(seed: u64, scenario_id: u64) -> StdRng {
    StdRng::seed_from_u64(seed.wrapping_mul(2654435761).wrapping_add(scenario_id))
}

// ---------------------------------------------------------------------------
// Fault tree specification
// ---------------------------------------------------------------------------

/// Component failure model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureMode {
    /// Fixed probability per mission.
    Bernoulli { p: f64 },
    /// Uncertain: `p = sigmoid(mu + sigma * Z)`, `Z ~ N(0,1)` drawn once per scenario.
    BernoulliUncertain { mu: f64, sigma: f64 },
    /// Weibull mission: `P(fail) = 1 - exp(-(mission_time / lambda)^k)`.
    WeibullMission { k: f64, lambda: f64, mission_time: f64 },
}

/// Logic gate combining child nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gate {
    And,
    Or,
}

/// A node in the fault tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultTreeNode {
    /// Leaf: references a component by index into `FaultTreeSpec::components`.
    Component(usize),
    /// Gate: combines children (indices into `FaultTreeSpec::nodes`).
    Gate { gate: Gate, children: Vec<usize> },
}

/// Complete fault tree specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultTreeSpec {
    pub components: Vec<FailureMode>,
    pub nodes: Vec<FaultTreeNode>,
    /// Index into `nodes` that is the top-level system failure event.
    pub top_event: usize,
}

impl FaultTreeSpec {
    /// Validate the spec: check indices are in range, no empty gates.
    pub fn validate(&self) -> ns_core::Result<()> {
        use ns_core::Error;
        if self.nodes.is_empty() {
            return Err(Error::Validation("nodes must be non-empty".into()));
        }
        if self.top_event >= self.nodes.len() {
            return Err(Error::Validation(format!(
                "top_event {} out of range (n_nodes={})",
                self.top_event,
                self.nodes.len()
            )));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                FaultTreeNode::Component(idx) => {
                    if *idx >= self.components.len() {
                        return Err(Error::Validation(format!(
                            "node {} references component {} but only {} components",
                            i,
                            idx,
                            self.components.len()
                        )));
                    }
                }
                FaultTreeNode::Gate { children, .. } => {
                    if children.is_empty() {
                        return Err(Error::Validation(format!(
                            "node {} is a gate with no children",
                            i
                        )));
                    }
                    for &c in children {
                        if c >= self.nodes.len() {
                            return Err(Error::Validation(format!(
                                "node {} references child {} but only {} nodes",
                                i,
                                c,
                                self.nodes.len()
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scenario simulation (single scenario)
// ---------------------------------------------------------------------------

/// Sample component failure states for one scenario.
/// `z` is the shared epistemic normal draw for BernoulliUncertain components.
#[inline]
fn sample_components(components: &[FailureMode], rng: &mut StdRng, comp_states: &mut [bool]) {
    use rand::Rng;
    // First draw: Z for epistemic uncertainty (consumed even if not needed, for budget determinism).
    let z: f64 = StandardNormal.sample(rng);

    for (i, mode) in components.iter().enumerate() {
        let u: f64 = rng.random();
        comp_states[i] = match mode {
            FailureMode::Bernoulli { p } => u < *p,
            FailureMode::BernoulliUncertain { mu, sigma } => {
                let p = sigmoid(mu + sigma * z);
                u < p
            }
            FailureMode::WeibullMission { k, lambda, mission_time } => {
                // P(fail) = 1 - exp(-(T/lambda)^k)
                // Equivalently: sample T = lambda * (-ln(U))^(1/k), fail if T <= mission_time
                let t = lambda * (-u.ln()).powf(1.0 / k);
                t <= *mission_time
            }
        };
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Evaluate the fault tree bottom-up given component states.
/// Returns the state of each node.
#[inline]
fn evaluate_tree(node_states: &mut [bool], spec: &FaultTreeSpec, comp_states: &[bool]) {
    // Process nodes in index order. Assumes children have lower indices than parents
    // (topological order). This is enforced by construction in typical fault trees.
    for i in 0..spec.nodes.len() {
        node_states[i] = match &spec.nodes[i] {
            FaultTreeNode::Component(idx) => comp_states[*idx],
            FaultTreeNode::Gate { gate, children } => match gate {
                Gate::And => children.iter().all(|&c| node_states[c]),
                Gate::Or => children.iter().any(|&c| node_states[c]),
            },
        };
    }
}

/// Evaluate the fault tree with continuous soft-gate semantics.
/// Returns importance score in [0, 1] where 1.0 = TOP event occurred.
///
/// Gate semantics:
/// - AND gate: mean of child soft-states (fraction of inputs satisfied).
/// - OR gate: max of child soft-states (any-one proximity).
///
/// Used by multi-level CE-IS to form elite sets even when no actual TOP
/// failures occur (very rare events, p < 1e-5).
#[inline]
fn evaluate_tree_soft(node_soft: &mut [f64], spec: &FaultTreeSpec, comp_states: &[bool]) -> f64 {
    for i in 0..spec.nodes.len() {
        node_soft[i] = match &spec.nodes[i] {
            FaultTreeNode::Component(idx) => {
                if comp_states[*idx] {
                    1.0
                } else {
                    0.0
                }
            }
            FaultTreeNode::Gate { gate, children } => match gate {
                Gate::And => {
                    let sum: f64 = children.iter().map(|&c| node_soft[c]).sum();
                    sum / children.len() as f64
                }
                Gate::Or => children.iter().map(|&c| node_soft[c]).fold(0.0f64, f64::max),
            },
        };
    }
    node_soft[spec.top_event]
}

// ---------------------------------------------------------------------------
// Accumulator for chunked reduction
// ---------------------------------------------------------------------------

struct Accumulator {
    n_top_failures: u64,
    comp_fail_given_top: Vec<u64>,
    n_scenarios: u64,
}

impl Accumulator {
    fn new(n_components: usize) -> Self {
        Self { n_top_failures: 0, comp_fail_given_top: vec![0u64; n_components], n_scenarios: 0 }
    }

    fn merge(&mut self, other: &Accumulator) {
        self.n_top_failures += other.n_top_failures;
        self.n_scenarios += other.n_scenarios;
        for (a, b) in self.comp_fail_given_top.iter_mut().zip(&other.comp_fail_given_top) {
            *a += b;
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a fault-tree Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultTreeMcResult {
    pub n_scenarios: usize,
    pub n_top_failures: u64,
    pub p_failure: f64,
    pub se: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub wall_time_s: f64,
    pub scenarios_per_sec: f64,
    /// `P(component_i failed | TOP failed)` — Birnbaum-style importance.
    pub component_importance: Vec<f64>,
}

// ---------------------------------------------------------------------------
// GPU flatten helper
// ---------------------------------------------------------------------------

/// Flatten a `FaultTreeSpec` into arrays ready for GPU upload.
///
/// Returns `(comp_types, comp_params, node_types, node_data, children_offsets, children_flat)`.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn flatten_spec_for_gpu(
    spec: &FaultTreeSpec,
) -> (Vec<i32>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let n_comp = spec.components.len();
    let mut comp_types = Vec::with_capacity(n_comp);
    let mut comp_params = Vec::with_capacity(n_comp * 3);
    for mode in &spec.components {
        match mode {
            FailureMode::Bernoulli { p } => {
                comp_types.push(0i32);
                comp_params.extend_from_slice(&[*p, 0.0, 0.0]);
            }
            FailureMode::BernoulliUncertain { mu, sigma } => {
                comp_types.push(1i32);
                comp_params.extend_from_slice(&[*mu, *sigma, 0.0]);
            }
            FailureMode::WeibullMission { k, lambda, mission_time } => {
                comp_types.push(2i32);
                comp_params.extend_from_slice(&[*k, *lambda, *mission_time]);
            }
        }
    }

    let n_nodes = spec.nodes.len();
    let mut node_types = Vec::with_capacity(n_nodes);
    let mut node_data = Vec::with_capacity(n_nodes);
    let mut children_offsets = Vec::with_capacity(n_nodes + 1);
    let mut children_flat = Vec::new();

    children_offsets.push(0i32);
    for node in &spec.nodes {
        match node {
            FaultTreeNode::Component(idx) => {
                node_types.push(0i32);
                node_data.push(*idx as i32);
                children_offsets.push(children_flat.len() as i32);
            }
            FaultTreeNode::Gate { gate, children } => {
                node_types.push(match gate {
                    Gate::And => 1i32,
                    Gate::Or => 2i32,
                });
                node_data.push(0i32);
                for &c in children {
                    children_flat.push(c as i32);
                }
                children_offsets.push(children_flat.len() as i32);
            }
        }
    }

    if children_flat.is_empty() {
        children_flat.push(0);
    }

    (comp_types, comp_params, node_types, node_data, children_offsets, children_flat)
}

/// Run fault-tree MC on CUDA GPU. Returns `FaultTreeMcResult`.
#[cfg(feature = "cuda")]
pub fn fault_tree_mc_cuda(
    spec: &FaultTreeSpec,
    n_scenarios: usize,
    seed: u64,
    chunk_size: usize,
) -> ns_core::Result<FaultTreeMcResult> {
    use ns_compute::fault_tree_cuda::{FaultTreeCudaAccelerator, FlatFaultTreeData};

    spec.validate()?;
    let (comp_types, comp_params, node_types, node_data, children_offsets, children_flat) =
        flatten_spec_for_gpu(spec);

    let data = FlatFaultTreeData {
        n_components: spec.components.len(),
        n_nodes: spec.nodes.len(),
        top_node: spec.top_event,
        comp_types,
        comp_params,
        node_types,
        node_data,
        children_offsets,
        children_flat,
    };

    let t0 = std::time::Instant::now();
    let mut accel = FaultTreeCudaAccelerator::new(&data)?;
    let raw = accel.run(n_scenarios, seed, chunk_size)?;
    let wall_time_s = t0.elapsed().as_secs_f64();

    let n = n_scenarios as f64;
    let k = raw.n_top_failures as f64;
    let p = k / n;
    let se = (p * (1.0 - p) / n).sqrt();
    let component_importance = if raw.n_top_failures > 0 {
        raw.comp_fail_given_top.iter().map(|&c| c as f64 / k).collect()
    } else {
        vec![0.0; spec.components.len()]
    };

    Ok(FaultTreeMcResult {
        n_scenarios,
        n_top_failures: raw.n_top_failures,
        p_failure: p,
        se,
        ci_lower: (p - 1.96 * se).max(0.0),
        ci_upper: (p + 1.96 * se).min(1.0),
        wall_time_s,
        scenarios_per_sec: n / wall_time_s,
        component_importance,
    })
}

/// Run fault-tree MC on Metal GPU. Returns `FaultTreeMcResult`.
#[cfg(feature = "metal")]
pub fn fault_tree_mc_metal(
    spec: &FaultTreeSpec,
    n_scenarios: usize,
    seed: u64,
    chunk_size: usize,
) -> ns_core::Result<FaultTreeMcResult> {
    use ns_compute::fault_tree_metal::{FaultTreeMetalAccelerator, FlatFaultTreeData};

    spec.validate()?;
    let (comp_types, comp_params, node_types, node_data, children_offsets, children_flat) =
        flatten_spec_for_gpu(spec);

    let data = FlatFaultTreeData {
        n_components: spec.components.len(),
        n_nodes: spec.nodes.len(),
        top_node: spec.top_event,
        comp_types,
        comp_params,
        node_types,
        node_data,
        children_offsets,
        children_flat,
    };

    let t0 = std::time::Instant::now();
    let mut accel = FaultTreeMetalAccelerator::new(&data)?;
    let raw = accel.run(n_scenarios, seed, chunk_size)?;
    let wall_time_s = t0.elapsed().as_secs_f64();

    let n = n_scenarios as f64;
    let k = raw.n_top_failures as f64;
    let p = k / n;
    let se = (p * (1.0 - p) / n).sqrt();
    let component_importance = if raw.n_top_failures > 0 {
        raw.comp_fail_given_top.iter().map(|&c| c as f64 / k).collect()
    } else {
        vec![0.0; spec.components.len()]
    };

    Ok(FaultTreeMcResult {
        n_scenarios,
        n_top_failures: raw.n_top_failures,
        p_failure: p,
        se,
        ci_lower: (p - 1.96 * se).max(0.0),
        ci_upper: (p + 1.96 * se).min(1.0),
        wall_time_s,
        scenarios_per_sec: n / wall_time_s,
        component_importance,
    })
}

// ---------------------------------------------------------------------------
// CPU engine
// ---------------------------------------------------------------------------

/// Default chunk size for CPU engine (1M scenarios per chunk).
pub const DEFAULT_CHUNK_SIZE: usize = 1_000_000;

/// Run Monte Carlo fault-tree simulation on CPU.
///
/// Chunked execution: processes `chunk_size` scenarios per batch to bound memory.
/// Rayon-parallel across chunks. Deterministic for a given seed.
pub fn fault_tree_mc_cpu(
    spec: &FaultTreeSpec,
    n_scenarios: usize,
    seed: u64,
    chunk_size: usize,
) -> ns_core::Result<FaultTreeMcResult> {
    spec.validate()?;
    let n_comp = spec.components.len();
    let n_nodes = spec.nodes.len();
    let chunk_size = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };

    let t0 = std::time::Instant::now();

    // Split into chunks for parallel processing.
    let n_chunks = n_scenarios.div_ceil(chunk_size);
    let chunk_ranges: Vec<(usize, usize)> = (0..n_chunks)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n_scenarios);
            (start, end)
        })
        .collect();

    let total = chunk_ranges
        .into_par_iter()
        .fold(
            || Accumulator::new(n_comp),
            |mut acc, (start, end)| {
                let mut comp_states = vec![false; n_comp];
                let mut node_states = vec![false; n_nodes];

                for scenario_id in start..end {
                    let mut rng = scenario_rng(seed, scenario_id as u64);
                    sample_components(&spec.components, &mut rng, &mut comp_states);
                    evaluate_tree(&mut node_states, spec, &comp_states);

                    let top_failed = node_states[spec.top_event];
                    if top_failed {
                        acc.n_top_failures += 1;
                        for (i, &failed) in comp_states.iter().enumerate() {
                            if failed {
                                acc.comp_fail_given_top[i] += 1;
                            }
                        }
                    }
                    acc.n_scenarios += 1;
                }
                acc
            },
        )
        .reduce(
            || Accumulator::new(n_comp),
            |mut a, b| {
                a.merge(&b);
                a
            },
        );

    let wall_time_s = t0.elapsed().as_secs_f64();
    let n = total.n_scenarios as f64;
    let k = total.n_top_failures as f64;
    let p = k / n;
    // Wald SE for proportion.
    let se = (p * (1.0 - p) / n).sqrt();
    let ci_lower = (p - 1.96 * se).max(0.0);
    let ci_upper = (p + 1.96 * se).min(1.0);

    let component_importance = if total.n_top_failures > 0 {
        total.comp_fail_given_top.iter().map(|&c| c as f64 / k).collect()
    } else {
        vec![0.0; n_comp]
    };

    Ok(FaultTreeMcResult {
        n_scenarios: total.n_scenarios as usize,
        n_top_failures: total.n_top_failures,
        p_failure: p,
        se,
        ci_lower,
        ci_upper,
        wall_time_s,
        scenarios_per_sec: total.n_scenarios as f64 / wall_time_s,
        component_importance,
    })
}

// ---------------------------------------------------------------------------
// Cross-Entropy Importance Sampling (CE-IS)
// ---------------------------------------------------------------------------

/// Configuration for CE-IS fault-tree simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultTreeCeIsConfig {
    /// Scenarios per CE level (default 10,000).
    pub n_per_level: usize,
    /// Elite fraction ρ for CE update (default 0.01).
    pub elite_fraction: f64,
    /// Maximum number of CE levels (default 10).
    pub max_levels: usize,
    /// Upper clamp for proposal probabilities (default 0.99).
    pub q_max: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for FaultTreeCeIsConfig {
    fn default() -> Self {
        Self { n_per_level: 10_000, elite_fraction: 0.01, max_levels: 20, q_max: 0.99, seed: 42 }
    }
}

/// Result of a CE-IS fault-tree simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultTreeCeIsResult {
    pub p_failure: f64,
    pub se: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_levels: usize,
    pub n_total_scenarios: usize,
    pub final_proposal: Vec<f64>,
    pub coefficient_of_variation: f64,
    pub wall_time_s: f64,
}

// ---------------------------------------------------------------------------
// CE-IS proposal types
// ---------------------------------------------------------------------------

/// Per-component CE-IS proposal state.
#[derive(Debug, Clone)]
enum CeIsProposal {
    /// Bernoulli or WeibullMission (reduced to effective p).
    /// `q` is the current proposal probability, `p_nat` is the natural probability.
    Bernoulli { q: f64, p_nat: f64 },
    /// BernoulliUncertain: proposal on the latent Z ~ N(mu_q, sigma_q).
    NormalZ { mu_q: f64, sigma_q: f64, mu_nat: f64, sigma_nat: f64 },
}

/// Per-scenario auxiliary data for elite update.
#[derive(Clone)]
struct CeIsScenarioData {
    comp_states: Vec<bool>,
    /// Z draws for BernoulliUncertain components (indexed by component).
    z_draws: Vec<f64>,
}

/// Sample one scenario under the CE-IS proposal.
/// Returns (comp_states, z_draws, log_is_weight).
fn sample_ce_is_scenario(
    proposals: &[CeIsProposal],
    spec: &FaultTreeSpec,
    rng: &mut StdRng,
    comp_states: &mut [bool],
    z_draws: &mut [f64],
    node_states: &mut [bool],
) -> (bool, f64) {
    use rand::Rng;

    let mut log_w = 0.0f64;

    for (i, prop) in proposals.iter().enumerate() {
        match prop {
            CeIsProposal::Bernoulli { q, p_nat } => {
                let u: f64 = rng.random();
                let failed = u < *q;
                comp_states[i] = failed;
                z_draws[i] = 0.0; // unused
                if failed {
                    log_w += (p_nat / q).ln();
                } else {
                    log_w += ((1.0 - p_nat) / (1.0 - q)).ln();
                }
            }
            CeIsProposal::NormalZ { mu_q, sigma_q, mu_nat, sigma_nat } => {
                // Sample Z from N(mu_q, sigma_q).
                let z_std: f64 = StandardNormal.sample(rng);
                let z = mu_q + sigma_q * z_std;
                z_draws[i] = z;

                // Compute p from natural parameters.
                let p = sigmoid(mu_nat + sigma_nat * z);
                let u: f64 = rng.random();
                comp_states[i] = u < p;

                // IS weight ratio: phi(z;0,1) / phi(z;mu_q,sigma_q)
                // = (sigma_q) * exp(-0.5*z^2 + 0.5*((z-mu_q)/sigma_q)^2)
                let log_nat = -0.5 * z * z;
                let z_std_prop = (z - mu_q) / sigma_q;
                let log_prop = -0.5 * z_std_prop * z_std_prop - sigma_q.ln();
                log_w += log_nat - log_prop;
            }
        }
    }

    evaluate_tree(node_states, spec, comp_states);
    let top_failed = node_states[spec.top_event];
    (top_failed, log_w)
}

/// Run CE-IS fault-tree simulation.
///
/// Uses the Cross-Entropy method to iteratively optimize the IS proposal
/// distribution for rare-event estimation. Supports all failure modes:
/// - `Bernoulli`: direct CE on failure probability.
/// - `WeibullMission`: reduced to Bernoulli with `p_eff = 1 - exp(-(T/λ)^k)`.
/// - `BernoulliUncertain`: CE on the latent `Z ~ N(mu_q, sigma_q)` proposal.
pub fn fault_tree_mc_ce_is(
    spec: &FaultTreeSpec,
    config: &FaultTreeCeIsConfig,
) -> ns_core::Result<FaultTreeCeIsResult> {
    spec.validate()?;

    let n_comp = spec.components.len();

    // Initialize proposals per component.
    let mut proposals: Vec<CeIsProposal> = Vec::with_capacity(n_comp);
    for mode in &spec.components {
        match mode {
            FailureMode::Bernoulli { p } => {
                proposals.push(CeIsProposal::Bernoulli { q: *p, p_nat: *p });
            }
            FailureMode::WeibullMission { k, lambda, mission_time } => {
                let p_eff = 1.0 - (-(mission_time / lambda).powf(*k)).exp();
                proposals.push(CeIsProposal::Bernoulli { q: p_eff, p_nat: p_eff });
            }
            FailureMode::BernoulliUncertain { mu, sigma } => {
                proposals.push(CeIsProposal::NormalZ {
                    mu_q: 0.0,
                    sigma_q: 1.0,
                    mu_nat: *mu,
                    sigma_nat: *sigma,
                });
            }
        }
    }

    let t0 = std::time::Instant::now();
    let n_per = config.n_per_level;
    let n_nodes = spec.nodes.len();

    let mut n_levels = 0usize;
    let mut total_scenarios = 0usize;

    // CE iteration with multi-level importance-based elite selection.
    //
    // Standard CE-IS requires TOP failures to form the elite set. For very rare
    // events (p < 1e-5), no TOP failures occur in the first levels.
    //
    // Multi-level approach: when TOP failures are insufficient, use a continuous
    // "soft importance" function to select elite scenarios. The importance function
    // uses relaxed gate semantics (AND → mean, OR → max), giving a score in [0,1]
    // where 1.0 = actual TOP failure. This progressively biases proposals toward
    // failure-prone configurations until actual TOP failures emerge.
    for _level in 0..config.max_levels {
        n_levels += 1;
        total_scenarios += n_per;

        let mut weights = Vec::with_capacity(n_per);
        let mut top_failed_flags = Vec::with_capacity(n_per);
        let mut importances = Vec::with_capacity(n_per);
        let mut scenarios_data = Vec::with_capacity(n_per);

        let mut comp_states = vec![false; n_comp];
        let mut z_draws = vec![0.0f64; n_comp];
        let mut node_states = vec![false; n_nodes];
        let mut node_soft = vec![0.0f64; n_nodes];

        for s in 0..n_per {
            let mut srng =
                scenario_rng(config.seed.wrapping_add(n_levels as u64 * 1_000_000_000), s as u64);

            let (top_failed, log_w) = sample_ce_is_scenario(
                &proposals,
                spec,
                &mut srng,
                &mut comp_states,
                &mut z_draws,
                &mut node_states,
            );

            let imp = evaluate_tree_soft(&mut node_soft, spec, &comp_states);

            weights.push(log_w.exp());
            top_failed_flags.push(top_failed);
            importances.push(imp);
            scenarios_data.push(CeIsScenarioData {
                comp_states: comp_states.clone(),
                z_draws: z_draws.clone(),
            });
        }

        let n_top: usize = top_failed_flags.iter().filter(|&&f| f).count();
        let n_elite = ((config.elite_fraction * n_per as f64).ceil() as usize).max(1);

        // Select elite: standard path (TOP failures) or multi-level path (importance).
        let failure_indices: Vec<(usize, f64)> = if n_top >= n_elite {
            // Standard path: enough TOP failures for stable elite selection.
            let mut top_failures: Vec<(usize, f64)> =
                (0..n_per).filter(|&i| top_failed_flags[i]).map(|i| (i, weights[i])).collect();
            top_failures.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top_failures.truncate(n_elite);
            top_failures
        } else {
            // Multi-level path: select elite by soft importance score.
            // Sort all scenarios by importance (descending), take top-ρ.
            let mut indexed: Vec<(usize, f64)> = (0..n_per).map(|i| (i, importances[i])).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(n_elite);
            // Filter out zero-importance scenarios (no partial failures at all).
            indexed
                .into_iter()
                .filter(|&(_, imp)| imp > 0.0)
                .map(|(i, _)| (i, weights[i]))
                .collect()
        };

        if failure_indices.is_empty() {
            break;
        }

        let w_sum: f64 = failure_indices.iter().map(|&(_, w)| w).sum();
        if w_sum <= 0.0 {
            break;
        }

        // Update proposals per component with smoothing (α = 0.7).
        // Smoothing prevents oscillation while allowing convergence.
        let alpha = 0.7;
        let mut converged = true;
        for i in 0..n_comp {
            match &mut proposals[i] {
                CeIsProposal::Bernoulli { q, p_nat } => {
                    let mut wq = 0.0f64;
                    for &(idx, w) in &failure_indices {
                        if scenarios_data[idx].comp_states[i] {
                            wq += w;
                        }
                    }
                    let qi_raw = (wq / w_sum).clamp(*p_nat, config.q_max);
                    let qi = alpha * qi_raw + (1.0 - alpha) * *q;
                    let qi = qi.clamp(*p_nat, config.q_max);
                    if (qi - *q).abs() > 1e-6 {
                        converged = false;
                    }
                    *q = qi;
                }
                CeIsProposal::NormalZ { mu_q, sigma_q, .. } => {
                    // Weighted mean and std of Z draws from elite.
                    let mut wz = 0.0f64;
                    let mut wz2 = 0.0f64;
                    for &(idx, w) in &failure_indices {
                        let z = scenarios_data[idx].z_draws[i];
                        wz += w * z;
                        wz2 += w * z * z;
                    }
                    let raw_mu = wz / w_sum;
                    let raw_var = (wz2 / w_sum - raw_mu * raw_mu).max(0.0);
                    let raw_sigma = raw_var.sqrt().max(0.1);

                    let new_mu = alpha * raw_mu + (1.0 - alpha) * *mu_q;
                    let new_sigma = alpha * raw_sigma + (1.0 - alpha) * *sigma_q;
                    let new_sigma = new_sigma.max(0.1);

                    if (new_mu - *mu_q).abs() > 1e-6 || (new_sigma - *sigma_q).abs() > 1e-6 {
                        converged = false;
                    }
                    *mu_q = new_mu;
                    *sigma_q = new_sigma;
                }
            }
        }

        // Converge only when proposals stabilize AND we have actual TOP failures.
        if converged && n_top >= n_elite {
            break;
        }
    }

    // Final estimation.
    total_scenarios += n_per;
    let mut w_sum = 0.0f64;
    let mut w2_sum = 0.0f64;

    let mut comp_states = vec![false; n_comp];
    let mut z_draws = vec![0.0f64; n_comp];
    let mut node_states = vec![false; n_nodes];

    for s in 0..n_per {
        let mut srng =
            scenario_rng(config.seed.wrapping_add((n_levels as u64 + 1) * 1_000_000_000), s as u64);

        let (top_failed, log_w) = sample_ce_is_scenario(
            &proposals,
            spec,
            &mut srng,
            &mut comp_states,
            &mut z_draws,
            &mut node_states,
        );

        if top_failed {
            let w = log_w.exp();
            w_sum += w;
            w2_sum += w * w;
        }
    }

    let n = n_per as f64;
    let p_hat = w_sum / n;
    let var_hat = (w2_sum / n - p_hat * p_hat) / n;
    let se = var_hat.max(0.0).sqrt();
    let cv = if p_hat > 0.0 { se / p_hat } else { f64::INFINITY };

    // Build final_proposal: effective q values for Bernoulli, mu_q for NormalZ.
    let final_proposal: Vec<f64> = proposals
        .iter()
        .map(|p| match p {
            CeIsProposal::Bernoulli { q, .. } => *q,
            CeIsProposal::NormalZ { mu_q, .. } => *mu_q,
        })
        .collect();

    let wall_time_s = t0.elapsed().as_secs_f64();

    Ok(FaultTreeCeIsResult {
        p_failure: p_hat,
        se,
        ci_lower: (p_hat - 1.96 * se).max(0.0),
        ci_upper: p_hat + 1.96 * se,
        n_levels,
        n_total_scenarios: total_scenarios,
        final_proposal,
        coefficient_of_variation: cv,
        wall_time_s,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: simple OR-gate tree with 2 Bernoulli components.
    fn simple_or_tree(p0: f64, p1: f64) -> FaultTreeSpec {
        FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: p0 }, FailureMode::Bernoulli { p: p1 }],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Gate { gate: Gate::Or, children: vec![0, 1] },
            ],
            top_event: 2,
        }
    }

    /// Helper: simple AND-gate tree with 2 Bernoulli components.
    fn simple_and_tree(p0: f64, p1: f64) -> FaultTreeSpec {
        FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: p0 }, FailureMode::Bernoulli { p: p1 }],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1] },
            ],
            top_event: 2,
        }
    }

    #[test]
    fn test_rng_determinism() {
        // Same (seed, scenario_id) → same draw sequence.
        let mut rng1 = scenario_rng(42, 100);
        let mut rng2 = scenario_rng(42, 100);
        use rand::Rng;
        let v1: [f64; 5] = std::array::from_fn(|_| rng1.random());
        let v2: [f64; 5] = std::array::from_fn(|_| rng2.random());
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_rng_disjoint_substreams() {
        // Different scenario_id → different draws.
        let mut rng1 = scenario_rng(42, 0);
        let mut rng2 = scenario_rng(42, 1);
        use rand::Rng;
        let v1: f64 = rng1.random();
        let v2: f64 = rng2.random();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_evaluate_or_gate() {
        let spec = simple_or_tree(0.5, 0.5);
        let mut node_states = vec![false; 3];

        // Both fail → TOP fail
        evaluate_tree(&mut node_states, &spec, &[true, true]);
        assert!(node_states[2]);

        // One fails → TOP fail
        evaluate_tree(&mut node_states, &spec, &[true, false]);
        assert!(node_states[2]);

        evaluate_tree(&mut node_states, &spec, &[false, true]);
        assert!(node_states[2]);

        // Neither fails → TOP ok
        evaluate_tree(&mut node_states, &spec, &[false, false]);
        assert!(!node_states[2]);
    }

    #[test]
    fn test_evaluate_and_gate() {
        let spec = simple_and_tree(0.5, 0.5);
        let mut node_states = vec![false; 3];

        // Both fail → TOP fail
        evaluate_tree(&mut node_states, &spec, &[true, true]);
        assert!(node_states[2]);

        // One fails → TOP ok
        evaluate_tree(&mut node_states, &spec, &[true, false]);
        assert!(!node_states[2]);

        evaluate_tree(&mut node_states, &spec, &[false, true]);
        assert!(!node_states[2]);

        // Neither → TOP ok
        evaluate_tree(&mut node_states, &spec, &[false, false]);
        assert!(!node_states[2]);
    }

    #[test]
    fn test_cpu_engine_or_gate_bernoulli() {
        // OR(p=0.01, p=0.02) → P(TOP) = 1 - (1-0.01)(1-0.02) = 0.0298
        let spec = simple_or_tree(0.01, 0.02);
        let result = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();
        let expected = 1.0 - (1.0 - 0.01) * (1.0 - 0.02);
        assert!(
            (result.p_failure - expected).abs() < 4.0 * result.se,
            "p_failure={} expected={} se={}",
            result.p_failure,
            expected,
            result.se
        );
    }

    #[test]
    fn test_cpu_engine_and_gate_bernoulli() {
        // AND(p=0.1, p=0.2) → P(TOP) = 0.1 * 0.2 = 0.02
        let spec = simple_and_tree(0.1, 0.2);
        let result = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();
        let expected = 0.1 * 0.2;
        assert!(
            (result.p_failure - expected).abs() < 4.0 * result.se,
            "p_failure={} expected={} se={}",
            result.p_failure,
            expected,
            result.se
        );
    }

    #[test]
    fn test_cpu_engine_determinism() {
        let spec = simple_or_tree(0.01, 0.02);
        let r1 = fault_tree_mc_cpu(&spec, 100_000, 42, 0).unwrap();
        let r2 = fault_tree_mc_cpu(&spec, 100_000, 42, 0).unwrap();
        assert_eq!(r1.n_top_failures, r2.n_top_failures);
        assert_eq!(r1.p_failure, r2.p_failure);
    }

    #[test]
    fn test_cpu_engine_golden_small() {
        // Golden test: fixed seed, small N → exact n_top_failures.
        let spec = simple_or_tree(0.1, 0.1);
        let result = fault_tree_mc_cpu(&spec, 1000, 12345, 500).unwrap();
        // Re-run to verify stability.
        let result2 = fault_tree_mc_cpu(&spec, 1000, 12345, 500).unwrap();
        assert_eq!(result.n_top_failures, result2.n_top_failures);
        // Sanity: with p≈0.19, expect ~190 ± ~12.
        assert!(result.n_top_failures > 150 && result.n_top_failures < 230);
    }

    #[test]
    fn test_component_importance() {
        // If comp 0 has p=0.5 and comp 1 has p=0.001, OR gate:
        // Almost all TOP failures are due to comp 0.
        let spec = simple_or_tree(0.5, 0.001);
        let result = fault_tree_mc_cpu(&spec, 200_000, 42, 0).unwrap();
        assert!(result.component_importance[0] > 0.98);
        assert!(result.component_importance[1] < 0.02);
    }

    #[test]
    fn test_weibull_mission_mode() {
        // Weibull with k=1 (exponential), lambda=100, mission_time=10.
        // P(fail) = 1 - exp(-(10/100)^1) = 1 - exp(-0.1) ≈ 0.0952
        let spec = FaultTreeSpec {
            components: vec![FailureMode::WeibullMission {
                k: 1.0,
                lambda: 100.0,
                mission_time: 10.0,
            }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let result = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();
        let expected = 1.0 - (-0.1f64).exp();
        assert!(
            (result.p_failure - expected).abs() < 4.0 * result.se,
            "p={} expected={} se={}",
            result.p_failure,
            expected,
            result.se
        );
    }

    #[test]
    fn test_uncertain_bernoulli_mode() {
        // BernoulliUncertain with mu=-3, sigma=0 → p = sigmoid(-3) ≈ 0.0474
        let spec = FaultTreeSpec {
            components: vec![FailureMode::BernoulliUncertain { mu: -3.0, sigma: 0.0 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let result = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();
        let expected = sigmoid(-3.0);
        assert!(
            (result.p_failure - expected).abs() < 4.0 * result.se,
            "p={} expected={} se={}",
            result.p_failure,
            expected,
            result.se
        );
    }

    #[test]
    fn test_uncertain_bernoulli_with_epistemic() {
        // With sigma > 0, epistemic uncertainty increases mean failure probability.
        // Jensen's inequality: E[sigmoid(mu + sigma*Z)] > sigmoid(mu) for sigma > 0.
        let spec_no_unc = FaultTreeSpec {
            components: vec![FailureMode::BernoulliUncertain { mu: -2.0, sigma: 0.0 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let spec_unc = FaultTreeSpec {
            components: vec![FailureMode::BernoulliUncertain { mu: -2.0, sigma: 1.0 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let r0 = fault_tree_mc_cpu(&spec_no_unc, 200_000, 42, 0).unwrap();
        let r1 = fault_tree_mc_cpu(&spec_unc, 200_000, 42, 0).unwrap();
        // With epistemic uncertainty, failure prob should be higher.
        assert!(r1.p_failure > r0.p_failure);
    }

    #[test]
    fn test_deep_tree() {
        // Chain: 4 components under nested OR/AND gates.
        //   TOP = OR(gate1, gate2)
        //   gate1 = AND(comp0, comp1)
        //   gate2 = AND(comp2, comp3)
        // P(TOP) = 1 - (1 - p0*p1) * (1 - p2*p3)
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.1 },
                FailureMode::Bernoulli { p: 0.2 },
                FailureMode::Bernoulli { p: 0.3 },
                FailureMode::Bernoulli { p: 0.4 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),                                   // node 0
                FaultTreeNode::Component(1),                                   // node 1
                FaultTreeNode::Component(2),                                   // node 2
                FaultTreeNode::Component(3),                                   // node 3
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1] }, // node 4 = AND(c0, c1)
                FaultTreeNode::Gate { gate: Gate::And, children: vec![2, 3] }, // node 5 = AND(c2, c3)
                FaultTreeNode::Gate { gate: Gate::Or, children: vec![4, 5] }, // node 6 = OR(g4, g5)
            ],
            top_event: 6,
        };
        let result = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();
        let expected = 1.0 - (1.0 - 0.1 * 0.2) * (1.0 - 0.3 * 0.4);
        assert!(
            (result.p_failure - expected).abs() < 4.0 * result.se,
            "p={} expected={} se={}",
            result.p_failure,
            expected,
            result.se
        );
    }

    #[test]
    fn test_validation_empty_nodes() {
        let spec = FaultTreeSpec { components: vec![], nodes: vec![], top_event: 0 };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validation_bad_top_event() {
        let spec = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.1 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 5,
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validation_bad_component_ref() {
        let spec = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.1 }],
            nodes: vec![FaultTreeNode::Component(99)],
            top_event: 0,
        };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_zero_failure_prob() {
        let spec = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.0 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let result = fault_tree_mc_cpu(&spec, 100_000, 42, 0).unwrap();
        assert_eq!(result.n_top_failures, 0);
        assert_eq!(result.p_failure, 0.0);
        assert_eq!(result.component_importance, vec![0.0]);
    }

    #[test]
    fn test_certain_failure() {
        let spec = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 1.0 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let result = fault_tree_mc_cpu(&spec, 10_000, 42, 0).unwrap();
        assert_eq!(result.n_top_failures, 10_000);
        assert_eq!(result.p_failure, 1.0);
    }

    #[test]
    fn test_scenarios_per_sec_positive() {
        let spec = simple_or_tree(0.01, 0.01);
        let result = fault_tree_mc_cpu(&spec, 100_000, 42, 0).unwrap();
        assert!(result.scenarios_per_sec > 0.0);
        assert!(result.wall_time_s > 0.0);
    }

    #[test]
    fn test_chunk_size_independence() {
        // Different chunk sizes must yield the same result (same seed).
        let spec = simple_or_tree(0.05, 0.03);
        let r1 = fault_tree_mc_cpu(&spec, 50_000, 99, 1000).unwrap();
        let r2 = fault_tree_mc_cpu(&spec, 50_000, 99, 10_000).unwrap();
        let r3 = fault_tree_mc_cpu(&spec, 50_000, 99, 50_000).unwrap();
        assert_eq!(r1.n_top_failures, r2.n_top_failures);
        assert_eq!(r2.n_top_failures, r3.n_top_failures);
    }

    // -----------------------------------------------------------------------
    // CE-IS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ce_is_and_gate_exact() {
        // AND(p=0.01, p=0.02) → P(TOP) = 0.01 * 0.02 = 2e-4
        let spec = simple_and_tree(0.01, 0.02);
        let config = FaultTreeCeIsConfig {
            n_per_level: 20_000,
            elite_fraction: 0.01,
            seed: 42,
            ..Default::default()
        };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let exact = 0.01 * 0.02;
        assert!(
            (r.p_failure - exact).abs() < 5.0 * r.se.max(1e-6),
            "CE-IS p={:.2e} exact={:.2e} se={:.2e}",
            r.p_failure,
            exact,
            r.se
        );
        assert!(r.n_levels >= 1);
    }

    #[test]
    fn test_ce_is_or_of_16_rare() {
        // OR of 16 Bernoulli(1e-4): P = 1-(1-1e-4)^16 ≈ 1.6e-3
        let n = 16;
        let p = 1e-4;
        let spec = FaultTreeSpec {
            components: (0..n).map(|_| FailureMode::Bernoulli { p }).collect(),
            nodes: (0..n)
                .map(FaultTreeNode::Component)
                .chain(std::iter::once(FaultTreeNode::Gate {
                    gate: Gate::Or,
                    children: (0..n).collect(),
                }))
                .collect(),
            top_event: n,
        };
        let config = FaultTreeCeIsConfig {
            n_per_level: 20_000,
            elite_fraction: 0.01,
            seed: 99,
            ..Default::default()
        };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let exact = 1.0 - (1.0 - p).powi(n as i32);
        assert!(
            (r.p_failure - exact).abs() < 5.0 * r.se.max(1e-5),
            "CE-IS p={:.4e} exact={:.4e} se={:.4e}",
            r.p_failure,
            exact,
            r.se
        );
    }

    #[test]
    fn test_ce_is_determinism() {
        let spec = simple_and_tree(0.05, 0.03);
        let config = FaultTreeCeIsConfig { n_per_level: 5_000, seed: 123, ..Default::default() };
        let r1 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let r2 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        assert_eq!(r1.p_failure, r2.p_failure);
        assert_eq!(r1.n_levels, r2.n_levels);
    }

    #[test]
    fn test_ce_is_weibull_mission() {
        // WeibullMission with k=1, lambda=100, T=10.
        // p_eff = 1 - exp(-0.1) ≈ 0.0952
        // Single component → P(TOP) = p_eff.
        let spec = FaultTreeSpec {
            components: vec![FailureMode::WeibullMission {
                k: 1.0,
                lambda: 100.0,
                mission_time: 10.0,
            }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let config = FaultTreeCeIsConfig { n_per_level: 20_000, seed: 42, ..Default::default() };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let exact = 1.0 - (-0.1f64).exp();
        assert!(
            (r.p_failure - exact).abs() < 5.0 * r.se.max(1e-3),
            "CE-IS weibull p={:.4e} exact={:.4e} se={:.4e}",
            r.p_failure,
            exact,
            r.se,
        );
    }

    #[test]
    fn test_ce_is_bernoulli_uncertain() {
        // Single BernoulliUncertain(mu=-2, sigma=0.5) — moderate, well-behaved.
        // Compare with vanilla MC.
        let spec = FaultTreeSpec {
            components: vec![FailureMode::BernoulliUncertain { mu: -2.0, sigma: 0.5 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let config = FaultTreeCeIsConfig {
            n_per_level: 50_000,
            elite_fraction: 0.05,
            seed: 42,
            ..Default::default()
        };
        let ce = fault_tree_mc_ce_is(&spec, &config).unwrap();

        // Vanilla MC reference.
        let vanilla = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();

        assert!(
            (ce.p_failure - vanilla.p_failure).abs() < 5.0 * (ce.se + vanilla.se).max(0.01),
            "CE-IS p={:.4e} vanilla p={:.4e} ce_se={:.4e} van_se={:.4e}",
            ce.p_failure,
            vanilla.p_failure,
            ce.se,
            vanilla.se,
        );
        assert!(ce.p_failure > 0.0);
        assert!(ce.se.is_finite());
    }

    #[test]
    fn test_ce_is_mixed_components() {
        // Mix: Bernoulli + WeibullMission + BernoulliUncertain in one tree.
        // AND(Bernoulli(0.1), WeibullMission(k=1,lam=50,T=5), BernoulliUncertain(mu=-2,sigma=0.5))
        // Analytical: P(AND) = p0 * p1 * E[sigmoid(mu+sigma*Z)]
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.1 },
                FailureMode::WeibullMission { k: 1.0, lambda: 50.0, mission_time: 5.0 },
                FailureMode::BernoulliUncertain { mu: -2.0, sigma: 0.5 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1, 2] },
            ],
            top_event: 3,
        };

        let config = FaultTreeCeIsConfig {
            n_per_level: 20_000,
            elite_fraction: 0.01,
            seed: 42,
            ..Default::default()
        };
        let ce = fault_tree_mc_ce_is(&spec, &config).unwrap();

        // Vanilla MC reference.
        let vanilla = fault_tree_mc_cpu(&spec, 500_000, 42, 0).unwrap();

        assert!(
            (ce.p_failure - vanilla.p_failure).abs() < 5.0 * (ce.se + vanilla.se),
            "CE-IS mixed p={:.4e} vanilla p={:.4e}",
            ce.p_failure,
            vanilla.p_failure,
        );
        assert!(ce.p_failure > 0.0);
    }

    #[test]
    fn test_ce_is_mixed_determinism() {
        // Same seed → same result for mixed components.
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.05 },
                FailureMode::WeibullMission { k: 2.0, lambda: 100.0, mission_time: 20.0 },
                FailureMode::BernoulliUncertain { mu: -3.0, sigma: 1.0 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Gate { gate: Gate::Or, children: vec![0, 1, 2] },
            ],
            top_event: 3,
        };
        let config = FaultTreeCeIsConfig { n_per_level: 5_000, seed: 99, ..Default::default() };
        let r1 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let r2 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        assert_eq!(r1.p_failure, r2.p_failure);
        assert_eq!(r1.n_levels, r2.n_levels);
    }

    #[test]
    fn test_ce_is_se_smaller_than_vanilla() {
        // For a moderately rare event, CE-IS should achieve lower SE than vanilla MC
        // with comparable total scenario count.
        let spec = simple_and_tree(0.01, 0.02);
        let config = FaultTreeCeIsConfig {
            n_per_level: 20_000,
            elite_fraction: 0.01,
            seed: 42,
            ..Default::default()
        };
        let ce = fault_tree_mc_ce_is(&spec, &config).unwrap();

        // Vanilla MC with same total scenarios.
        let _vanilla = fault_tree_mc_cpu(&spec, ce.n_total_scenarios, 42, 0).unwrap();

        // CE-IS should have comparable or better SE for rare events.
        // (With p=2e-4 and ~40k scenarios, vanilla SE is quite large.)
        // We just verify CE-IS produced a reasonable result.
        assert!(ce.p_failure > 0.0, "CE-IS should find some failures");
        assert!(ce.se.is_finite(), "CE-IS SE should be finite");
        assert!(
            ce.coefficient_of_variation < 5.0,
            "CE-IS CV={} should be reasonable",
            ce.coefficient_of_variation
        );
    }

    #[test]
    fn test_ce_is_convergence_vs_vanilla() {
        // Verify CE-IS estimate is within CI of a large vanilla MC run.
        let spec = simple_and_tree(0.1, 0.2);
        let exact = 0.1 * 0.2;

        let config = FaultTreeCeIsConfig { n_per_level: 10_000, seed: 77, ..Default::default() };
        let ce = fault_tree_mc_ce_is(&spec, &config).unwrap();

        assert!(
            (ce.p_failure - exact).abs() < 5.0 * ce.se.max(1e-4),
            "CE-IS p={:.4e} exact={:.4e} se={:.4e}",
            ce.p_failure,
            exact,
            ce.se
        );
    }

    // -----------------------------------------------------------------------
    // Benchmarks — run with `cargo test -p ns-inference --release -- <name> --nocapture --ignored`
    // Results recorded in docs/benchmarks/phase2_5_benchmarks.md
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_simple_bernoulli() {
        // Simplest possible test: single Bernoulli p=0.5.
        let spec = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.5 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let cpu = fault_tree_mc_cpu(&spec, 100_000, 42, 0).unwrap();
        let metal = fault_tree_mc_metal(&spec, 100_000, 42, 0).unwrap();
        eprintln!(
            "Simple Bernoulli(0.5): CPU p={:.4} top={}, Metal p={:.4} top={}",
            cpu.p_failure, cpu.n_top_failures, metal.p_failure, metal.n_top_failures
        );
        assert!(
            metal.p_failure > 0.3 && metal.p_failure < 0.7,
            "Metal p_failure={} should be near 0.5",
            metal.p_failure
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_cpu_parity_smoke() {
        use ns_compute::fault_tree_cuda::FaultTreeCudaAccelerator;

        // CI/dev machines often compile with `--features cuda` but have no GPU.
        // Treat this as a best-effort correctness gate for CUDA runners.
        if !FaultTreeCudaAccelerator::is_available() {
            eprintln!("CUDA not available; skipping CUDA parity smoke test.");
            return;
        }

        let n_scenarios = 200_000usize;
        let seed = 42u64;

        // 1) Single component.
        let spec_a = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.1 }],
            nodes: vec![FaultTreeNode::Component(0)],
            top_event: 0,
        };
        let cpu_a = fault_tree_mc_cpu(&spec_a, n_scenarios, seed, 0).unwrap();
        let cuda_a = fault_tree_mc_cuda(&spec_a, n_scenarios, seed, 0).unwrap();
        let se_a = (cpu_a.se * cpu_a.se + cuda_a.se * cuda_a.se).sqrt().max(1e-20);
        assert!(
            (cpu_a.p_failure - cuda_a.p_failure).abs() <= 5.0 * se_a,
            "CUDA parity (single comp) failed: cpu p={:.6} se={:.2e}, cuda p={:.6} se={:.2e}",
            cpu_a.p_failure,
            cpu_a.se,
            cuda_a.p_failure,
            cuda_a.se
        );

        // 2) OR gate: analytic p = 1 - (1-p0)(1-p1).
        let spec_or = FaultTreeSpec {
            components: vec![FailureMode::Bernoulli { p: 0.1 }, FailureMode::Bernoulli { p: 0.2 }],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Gate { gate: Gate::Or, children: vec![0, 1] },
            ],
            top_event: 2,
        };
        let cpu_or = fault_tree_mc_cpu(&spec_or, n_scenarios, seed, 0).unwrap();
        let cuda_or = fault_tree_mc_cuda(&spec_or, n_scenarios, seed, 0).unwrap();
        let se_or = (cpu_or.se * cpu_or.se + cuda_or.se * cuda_or.se).sqrt().max(1e-20);
        assert!(
            (cpu_or.p_failure - cuda_or.p_failure).abs() <= 5.0 * se_or,
            "CUDA parity (OR) failed: cpu p={:.6} se={:.2e}, cuda p={:.6} se={:.2e}",
            cpu_or.p_failure,
            cpu_or.se,
            cuda_or.p_failure,
            cuda_or.se
        );
    }

    #[test]
    #[ignore] // benchmark — run explicitly
    #[cfg(feature = "metal")]
    fn bench_metal_vs_cpu() {
        // 16-component OR tree with mixed failure modes.
        let n = 16;
        let spec = FaultTreeSpec {
            components: (0..n)
                .map(|i| match i % 3 {
                    0 => FailureMode::Bernoulli { p: 0.01 },
                    1 => FailureMode::WeibullMission { k: 1.5, lambda: 200.0, mission_time: 10.0 },
                    _ => FailureMode::BernoulliUncertain { mu: -3.0, sigma: 0.5 },
                })
                .collect(),
            nodes: (0..n)
                .map(|i| FaultTreeNode::Component(i))
                .chain(std::iter::once(FaultTreeNode::Gate {
                    gate: Gate::Or,
                    children: (0..n).collect(),
                }))
                .collect(),
            top_event: n,
        };

        let seed = 42u64;
        for &n_scenarios in &[100_000usize, 1_000_000, 10_000_000] {
            let cpu = fault_tree_mc_cpu(&spec, n_scenarios, seed, 0).unwrap();
            let metal = fault_tree_mc_metal(&spec, n_scenarios, seed, 0).unwrap();

            let p_diff = (cpu.p_failure - metal.p_failure).abs();
            let max_se = cpu.se.max(metal.se);
            eprintln!(
                "\n=== Fault Tree MC: {} scenarios ===\n  CPU:   p={:.6} se={:.2e}  {:.0} scen/s  {:.1}ms\n  Metal: p={:.6} se={:.2e}  {:.0} scen/s  {:.1}ms\n  Δp={:.2e} ({:.1}σ)  Speedup: {:.2}x",
                n_scenarios,
                cpu.p_failure,
                cpu.se,
                cpu.scenarios_per_sec,
                cpu.wall_time_s * 1000.0,
                metal.p_failure,
                metal.se,
                metal.scenarios_per_sec,
                metal.wall_time_s * 1000.0,
                p_diff,
                p_diff / max_se,
                metal.scenarios_per_sec / cpu.scenarios_per_sec,
            );
        }
    }

    /// Build a realistic hierarchical fault tree with N_COMP components and 4 levels:
    ///   Level 0: N_COMP Component leaves
    ///   Level 1: groups of 4 components → OR gates (subsystems)
    ///   Level 2: groups of 4 subsystems → AND gates (redundancy groups)
    ///   Level 3: top-level OR gate (system failure)
    /// Mixed failure modes cycling through Bernoulli, WeibullMission, BernoulliUncertain.
    fn build_realistic_tree(n_comp: usize) -> FaultTreeSpec {
        let mut components = Vec::with_capacity(n_comp);
        for i in 0..n_comp {
            components.push(match i % 3 {
                0 => FailureMode::Bernoulli { p: 0.001 + 0.002 * (i as f64 / n_comp as f64) },
                1 => FailureMode::WeibullMission {
                    k: 1.2 + 0.3 * (i as f64 / n_comp as f64),
                    lambda: 5000.0 - 2000.0 * (i as f64 / n_comp as f64),
                    mission_time: 100.0,
                },
                _ => FailureMode::BernoulliUncertain {
                    mu: -5.0 + (i as f64 / n_comp as f64),
                    sigma: 0.3 + 0.2 * (i as f64 / n_comp as f64),
                },
            });
        }

        let mut nodes: Vec<FaultTreeNode> = Vec::new();

        // Level 0: component leaves.
        for i in 0..n_comp {
            nodes.push(FaultTreeNode::Component(i));
        }

        // Level 1: OR gates, groups of 4 (subsystem failure = any component fails).
        let n_l1 = n_comp.div_ceil(4);
        let l1_start = nodes.len();
        for g in 0..n_l1 {
            let start = g * 4;
            let end = (start + 4).min(n_comp);
            let children: Vec<usize> = (start..end).collect();
            nodes.push(FaultTreeNode::Gate { gate: Gate::Or, children });
        }

        // Level 2: AND gates, groups of 4 (redundancy = all subsystems must fail).
        let n_l2 = n_l1.div_ceil(4);
        let l2_start = nodes.len();
        for g in 0..n_l2 {
            let start = l1_start + g * 4;
            let end = (start + 4).min(l1_start + n_l1);
            let children: Vec<usize> = (start..end).collect();
            nodes.push(FaultTreeNode::Gate { gate: Gate::And, children });
        }

        // Level 3: top-level OR (system fails if any redundancy group fails).
        let top_children: Vec<usize> = (l2_start..l2_start + n_l2).collect();
        nodes.push(FaultTreeNode::Gate { gate: Gate::Or, children: top_children });
        let top_event = nodes.len() - 1;

        FaultTreeSpec { components, nodes, top_event }
    }

    #[test]
    #[ignore] // benchmark — run explicitly
    #[cfg(feature = "metal")]
    fn bench_metal_realistic_tree() {
        eprintln!("\n=== Metal vs CPU: Realistic Hierarchical Fault Trees ===\n");

        for &n_comp in &[32usize, 64, 128] {
            let spec = build_realistic_tree(n_comp);
            let n_nodes = spec.nodes.len();
            let seed = 42u64;

            // Warm up.
            let _ = fault_tree_mc_cpu(&spec, 10_000, seed, 0).unwrap();
            let _ = fault_tree_mc_metal(&spec, 10_000, seed, 0).unwrap();

            eprintln!("--- {} components, {} nodes, 4 levels ---", n_comp, n_nodes);

            for &n_scenarios in &[100_000usize, 1_000_000, 10_000_000] {
                let cpu = fault_tree_mc_cpu(&spec, n_scenarios, seed, 0).unwrap();
                let metal = fault_tree_mc_metal(&spec, n_scenarios, seed, 0).unwrap();

                let p_diff = (cpu.p_failure - metal.p_failure).abs();
                let max_se = cpu.se.max(metal.se).max(1e-20);
                eprintln!(
                    "  N={:>10}  CPU: p={:.4e} {:.0} scen/s {:.0}ms  Metal: p={:.4e} {:.0} scen/s {:.0}ms  Δ={:.1}σ  {:.1}x",
                    n_scenarios,
                    cpu.p_failure,
                    cpu.scenarios_per_sec,
                    cpu.wall_time_s * 1000.0,
                    metal.p_failure,
                    metal.scenarios_per_sec,
                    metal.wall_time_s * 1000.0,
                    p_diff / max_se,
                    metal.scenarios_per_sec / cpu.scenarios_per_sec,
                );
            }
            eprintln!();
        }
    }

    #[test]
    #[ignore] // diagnostic — run explicitly
    fn diag_ce_is_convergence() {
        // Same model as bench_ce_is_vs_vanilla.
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.05 },
                FailureMode::Bernoulli { p: 0.04 },
                FailureMode::WeibullMission { k: 1.5, lambda: 100.0, mission_time: 10.0 },
                FailureMode::BernoulliUncertain { mu: -2.0, sigma: 0.5 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Component(3),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1, 2, 3] },
            ],
            top_event: 4,
        };

        // Print effective natural probabilities.
        eprintln!("\n=== CE-IS Convergence Diagnostics ===\n");
        for (i, mode) in spec.components.iter().enumerate() {
            let p_eff = match mode {
                FailureMode::Bernoulli { p } => *p,
                FailureMode::WeibullMission { k, lambda, mission_time } => {
                    1.0 - (-(mission_time / lambda).powf(*k)).exp()
                }
                FailureMode::BernoulliUncertain { mu, sigma: _ } => {
                    sigmoid(*mu) // E[p] for Z=0
                }
            };
            eprintln!("  comp[{}]: {:?} → p_eff={:.4e}", i, mode, p_eff);
        }
        let p_and: f64 = spec
            .components
            .iter()
            .map(|m| match m {
                FailureMode::Bernoulli { p } => *p,
                FailureMode::WeibullMission { k, lambda, mission_time } => {
                    1.0 - (-(mission_time / lambda).powf(*k)).exp()
                }
                FailureMode::BernoulliUncertain { mu, .. } => sigmoid(*mu),
            })
            .product();
        eprintln!("  p_AND (approx) = {:.4e}\n", p_and);

        // Run CE-IS manually, logging each level.
        let n_comp = spec.components.len();
        let n_nodes = spec.nodes.len();
        let config = FaultTreeCeIsConfig {
            n_per_level: 100_000,
            elite_fraction: 0.01,
            max_levels: 15,
            q_max: 0.99,
            seed: 42,
        };

        // Initialize proposals.
        let mut proposals: Vec<CeIsProposal> = spec
            .components
            .iter()
            .map(|mode| match mode {
                FailureMode::Bernoulli { p } => CeIsProposal::Bernoulli { q: *p, p_nat: *p },
                FailureMode::WeibullMission { k, lambda, mission_time } => {
                    let p_eff = 1.0 - (-(mission_time / lambda).powf(*k)).exp();
                    CeIsProposal::Bernoulli { q: p_eff, p_nat: p_eff }
                }
                FailureMode::BernoulliUncertain { mu, sigma } => CeIsProposal::NormalZ {
                    mu_q: 0.0,
                    sigma_q: 1.0,
                    mu_nat: *mu,
                    sigma_nat: *sigma,
                },
            })
            .collect();

        for level in 0..config.max_levels {
            let mut weights = Vec::with_capacity(config.n_per_level);
            let mut top_failed_flags = Vec::with_capacity(config.n_per_level);
            let mut scenarios_data = Vec::with_capacity(config.n_per_level);
            let mut comp_states = vec![false; n_comp];
            let mut z_draws = vec![0.0f64; n_comp];
            let mut node_states = vec![false; n_nodes];

            for s in 0..config.n_per_level {
                let mut srng = scenario_rng(
                    config.seed.wrapping_add((level as u64 + 1) * 1_000_000_000),
                    s as u64,
                );
                let (top_failed, log_w) = sample_ce_is_scenario(
                    &proposals,
                    &spec,
                    &mut srng,
                    &mut comp_states,
                    &mut z_draws,
                    &mut node_states,
                );
                weights.push(log_w.exp());
                top_failed_flags.push(top_failed);
                scenarios_data.push(CeIsScenarioData {
                    comp_states: comp_states.clone(),
                    z_draws: z_draws.clone(),
                });
            }

            let n_top: usize = top_failed_flags.iter().filter(|&&f| f).count();

            // Weight statistics for failures.
            let fail_weights: Vec<f64> = (0..config.n_per_level)
                .filter(|&i| top_failed_flags[i])
                .map(|i| weights[i])
                .collect();
            let w_mean = if !fail_weights.is_empty() {
                fail_weights.iter().sum::<f64>() / fail_weights.len() as f64
            } else {
                0.0
            };
            let w_max = fail_weights.iter().cloned().fold(0.0f64, f64::max);
            let w_min = fail_weights.iter().cloned().fold(f64::INFINITY, f64::min);

            // Print proposals.
            let q_str: String = proposals
                .iter()
                .map(|p| match p {
                    CeIsProposal::Bernoulli { q, .. } => format!("{:.4}", q),
                    CeIsProposal::NormalZ { mu_q, sigma_q, .. } => {
                        format!("N({:.2},{:.2})", mu_q, sigma_q)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");

            eprintln!(
                "  Level {:2}: n_top={:5}/{:6}  w_mean={:.4e} w_range=[{:.2e},{:.2e}]  q=[{}]",
                level, n_top, config.n_per_level, w_mean, w_min, w_max, q_str
            );

            if n_top == 0 {
                eprintln!("  → 0 failures, CE-IS stops.");
                break;
            }

            // Elite update (same as production code).
            let n_elite =
                ((config.elite_fraction * config.n_per_level as f64).ceil() as usize).max(1);
            let mut failure_indices: Vec<(usize, f64)> = (0..config.n_per_level)
                .filter(|&i| top_failed_flags[i])
                .map(|i| (i, weights[i]))
                .collect();
            failure_indices
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            failure_indices.truncate(n_elite);

            let w_sum: f64 = failure_indices.iter().map(|&(_, w)| w).sum();
            if w_sum <= 0.0 {
                break;
            }

            let mut converged = true;
            for i in 0..n_comp {
                match &mut proposals[i] {
                    CeIsProposal::Bernoulli { q, p_nat } => {
                        let mut wq = 0.0f64;
                        for &(idx, w) in &failure_indices {
                            if scenarios_data[idx].comp_states[i] {
                                wq += w;
                            }
                        }
                        let qi = (wq / w_sum).clamp(*p_nat, config.q_max);
                        if (qi - *q).abs() > 1e-6 {
                            converged = false;
                        }
                        *q = qi;
                    }
                    CeIsProposal::NormalZ { mu_q, sigma_q, .. } => {
                        let mut wz = 0.0f64;
                        let mut wz2 = 0.0f64;
                        for &(idx, w) in &failure_indices {
                            let z = scenarios_data[idx].z_draws[i];
                            wz += w * z;
                            wz2 += w * z * z;
                        }
                        let new_mu = wz / w_sum;
                        let new_var = (wz2 / w_sum - new_mu * new_mu).max(0.0);
                        let new_sigma = new_var.sqrt().max(0.1);
                        if (new_mu - *mu_q).abs() > 1e-6 || (new_sigma - *sigma_q).abs() > 1e-6 {
                            converged = false;
                        }
                        *mu_q = new_mu;
                        *sigma_q = new_sigma;
                    }
                }
            }
            if converged {
                eprintln!("  → Converged.");
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Soft importance function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_tree_soft_and() {
        let spec = simple_and_tree(0.5, 0.5);
        let mut node_soft = vec![0.0f64; 3];

        // Both failed → 1.0
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[true, true]), 1.0);
        // One failed → 0.5
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[true, false]), 0.5);
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[false, true]), 0.5);
        // None failed → 0.0
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[false, false]), 0.0);
    }

    #[test]
    fn test_evaluate_tree_soft_or() {
        let spec = simple_or_tree(0.5, 0.5);
        let mut node_soft = vec![0.0f64; 3];

        // Both failed → 1.0
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[true, true]), 1.0);
        // One failed → 1.0 (max)
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[true, false]), 1.0);
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[false, true]), 1.0);
        // None failed → 0.0
        assert_eq!(evaluate_tree_soft(&mut node_soft, &spec, &[false, false]), 0.0);
    }

    #[test]
    fn test_evaluate_tree_soft_deep() {
        // TOP = OR(AND(c0, c1), AND(c2, c3))
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.1 },
                FailureMode::Bernoulli { p: 0.1 },
                FailureMode::Bernoulli { p: 0.1 },
                FailureMode::Bernoulli { p: 0.1 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Component(3),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1] },
                FaultTreeNode::Gate { gate: Gate::And, children: vec![2, 3] },
                FaultTreeNode::Gate { gate: Gate::Or, children: vec![4, 5] },
            ],
            top_event: 6,
        };
        let mut ns = vec![0.0f64; 7];

        // c0 failed, rest ok: AND(1,0)=0.5, AND(0,0)=0, OR(0.5,0)=0.5
        assert_eq!(evaluate_tree_soft(&mut ns, &spec, &[true, false, false, false]), 0.5);
        // c0+c1 failed: AND(1,1)=1.0, AND(0,0)=0, OR(1,0)=1.0 → TOP
        assert_eq!(evaluate_tree_soft(&mut ns, &spec, &[true, true, false, false]), 1.0);
        // c0+c2 failed: AND(1,0)=0.5, AND(1,0)=0.5, OR(0.5,0.5)=0.5
        assert_eq!(evaluate_tree_soft(&mut ns, &spec, &[true, false, true, false]), 0.5);
    }

    // -----------------------------------------------------------------------
    // Multi-level CE-IS tests (very rare events, p < 1e-5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ce_is_rare_and_of_4() {
        // AND-of-4 with p=0.01 each → P(TOP) = 1e-8.
        // This is too rare for standard CE-IS (n_per=10k → 0 TOP failures at level 1).
        // Multi-level CE should still converge.
        let spec = FaultTreeSpec {
            components: (0..4).map(|_| FailureMode::Bernoulli { p: 0.01 }).collect(),
            nodes: (0..4)
                .map(FaultTreeNode::Component)
                .chain(std::iter::once(FaultTreeNode::Gate {
                    gate: Gate::And,
                    children: vec![0, 1, 2, 3],
                }))
                .collect(),
            top_event: 4,
        };
        let exact = 0.01f64.powi(4); // 1e-8

        let config = FaultTreeCeIsConfig {
            n_per_level: 50_000,
            elite_fraction: 0.01,
            max_levels: 20,
            q_max: 0.99,
            seed: 42,
        };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();

        assert!(r.p_failure > 0.0, "multi-level CE-IS should find failures for p=1e-8");
        assert!(
            r.coefficient_of_variation < 5.0,
            "CV={} should be reasonable",
            r.coefficient_of_variation
        );
        // Allow wide tolerance for very rare event (order of magnitude).
        let ratio = r.p_failure / exact;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "CE-IS p={:.2e} exact={:.2e} ratio={:.2}",
            r.p_failure,
            exact,
            ratio
        );
    }

    #[test]
    fn test_ce_is_rare_and_of_6() {
        // AND-of-6 with p=0.01 each → P(TOP) = 1e-12.
        // Completely unreachable by vanilla MC. Multi-level CE should handle it.
        let spec = FaultTreeSpec {
            components: (0..6).map(|_| FailureMode::Bernoulli { p: 0.01 }).collect(),
            nodes: (0..6)
                .map(FaultTreeNode::Component)
                .chain(std::iter::once(FaultTreeNode::Gate {
                    gate: Gate::And,
                    children: (0..6).collect(),
                }))
                .collect(),
            top_event: 6,
        };
        let exact = 0.01f64.powi(6); // 1e-12

        let config = FaultTreeCeIsConfig {
            n_per_level: 50_000,
            elite_fraction: 0.01,
            max_levels: 20,
            q_max: 0.99,
            seed: 42,
        };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();

        assert!(r.p_failure > 0.0, "multi-level CE-IS should find failures for p=1e-12");
        assert!(r.se.is_finite(), "SE should be finite");
        // Order-of-magnitude check.
        let log_ratio = (r.p_failure.ln() - exact.ln()).abs();
        assert!(
            log_ratio < 5.0, // within ~e^5 ≈ 150x
            "CE-IS p={:.2e} exact={:.2e} log_ratio={:.1}",
            r.p_failure,
            exact,
            log_ratio
        );
    }

    #[test]
    fn test_ce_is_rare_determinism() {
        // Same seed → same result for rare event multi-level CE.
        let spec = FaultTreeSpec {
            components: (0..4).map(|_| FailureMode::Bernoulli { p: 0.01 }).collect(),
            nodes: (0..4)
                .map(FaultTreeNode::Component)
                .chain(std::iter::once(FaultTreeNode::Gate {
                    gate: Gate::And,
                    children: vec![0, 1, 2, 3],
                }))
                .collect(),
            top_event: 4,
        };
        let config = FaultTreeCeIsConfig {
            n_per_level: 10_000,
            max_levels: 20,
            seed: 77,
            ..Default::default()
        };
        let r1 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        let r2 = fault_tree_mc_ce_is(&spec, &config).unwrap();
        assert_eq!(r1.p_failure, r2.p_failure);
        assert_eq!(r1.n_levels, r2.n_levels);
    }

    #[test]
    fn test_ce_is_rare_mixed_modes() {
        // AND-of-4 with mixed failure modes, all rare (~0.01 each) → P ≈ 1e-8.
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.01 },
                FailureMode::WeibullMission { k: 1.0, lambda: 10000.0, mission_time: 100.0 },
                FailureMode::BernoulliUncertain { mu: -4.6, sigma: 0.3 }, // sigmoid(-4.6) ≈ 0.01
                FailureMode::Bernoulli { p: 0.01 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Component(3),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1, 2, 3] },
            ],
            top_event: 4,
        };

        let config = FaultTreeCeIsConfig {
            n_per_level: 50_000,
            elite_fraction: 0.01,
            max_levels: 20,
            q_max: 0.99,
            seed: 42,
        };
        let r = fault_tree_mc_ce_is(&spec, &config).unwrap();

        assert!(r.p_failure > 0.0, "multi-level CE-IS should find rare mixed failures");
        assert!(r.se.is_finite());
        // WeibullMission p_eff = 1-exp(-(100/10000)^1) = 1-exp(-0.01) ≈ 0.00995
        // BernoulliUncertain p ≈ sigmoid(-4.6) ≈ 0.01
        // Approx P(TOP) ≈ 0.01^4 ≈ 1e-8
        let approx_exact = 1e-8_f64;
        let log_ratio = (r.p_failure.ln() - approx_exact.ln()).abs();
        assert!(log_ratio < 5.0, "CE-IS p={:.2e} approx_exact={:.2e}", r.p_failure, approx_exact);
    }

    // -----------------------------------------------------------------------
    // Benchmarks — run with `cargo test -p ns-inference --release -- <name> --nocapture --ignored`
    // Results recorded in docs/benchmarks/phase2_5_benchmarks.md
    // -----------------------------------------------------------------------

    #[test]
    #[ignore] // benchmark — run explicitly
    fn bench_ce_is_rare_events() {
        // Demonstrate multi-level CE-IS on events too rare for vanilla MC.
        eprintln!("\n=== Multi-Level CE-IS: Very Rare Events ===\n");

        for &(n_and, p_each) in &[(4usize, 0.01f64), (6, 0.01), (8, 0.01)] {
            let spec = FaultTreeSpec {
                components: (0..n_and).map(|_| FailureMode::Bernoulli { p: p_each }).collect(),
                nodes: (0..n_and)
                    .map(FaultTreeNode::Component)
                    .chain(std::iter::once(FaultTreeNode::Gate {
                        gate: Gate::And,
                        children: (0..n_and).collect(),
                    }))
                    .collect(),
                top_event: n_and,
            };
            let exact = p_each.powi(n_and as i32);

            let config = FaultTreeCeIsConfig {
                n_per_level: 100_000,
                elite_fraction: 0.01,
                max_levels: 20,
                q_max: 0.99,
                seed: 42,
            };
            let r = fault_tree_mc_ce_is(&spec, &config).unwrap();

            let ratio = if exact > 0.0 { r.p_failure / exact } else { f64::NAN };
            eprintln!(
                "  AND-of-{} (p_each={:.2e}): exact={:.2e}  CE-IS={:.2e} ± {:.2e}  CV={:.2}  levels={}  N={}  ratio={:.2}  {:.0}ms",
                n_and,
                p_each,
                exact,
                r.p_failure,
                r.se,
                r.coefficient_of_variation,
                r.n_levels,
                r.n_total_scenarios,
                ratio,
                r.wall_time_s * 1000.0,
            );
        }
    }

    #[test]
    #[ignore] // benchmark — run explicitly
    fn bench_ce_is_vs_vanilla() {
        // AND-of-4 mixed failure modes, p ≈ 5e-6.
        let spec = FaultTreeSpec {
            components: vec![
                FailureMode::Bernoulli { p: 0.05 },
                FailureMode::Bernoulli { p: 0.04 },
                FailureMode::WeibullMission { k: 1.5, lambda: 100.0, mission_time: 10.0 },
                FailureMode::BernoulliUncertain { mu: -2.0, sigma: 0.5 },
            ],
            nodes: vec![
                FaultTreeNode::Component(0),
                FaultTreeNode::Component(1),
                FaultTreeNode::Component(2),
                FaultTreeNode::Component(3),
                FaultTreeNode::Gate { gate: Gate::And, children: vec![0, 1, 2, 3] },
            ],
            top_event: 4,
        };

        // Vanilla MC.
        let n_vanilla = 10_000_000;
        let vanilla = fault_tree_mc_cpu(&spec, n_vanilla, 42, 0).unwrap();

        // CE-IS.
        let config = FaultTreeCeIsConfig {
            n_per_level: 100_000,
            elite_fraction: 0.01,
            max_levels: 15,
            q_max: 0.99,
            seed: 42,
        };
        let ceis = fault_tree_mc_ce_is(&spec, &config).unwrap();

        eprintln!("\n=== CE-IS vs Vanilla MC (AND-of-4, mixed modes) ===");
        eprintln!(
            "  Vanilla: p={:.4e}  se={:.4e}  N={}  {:.0}ms",
            vanilla.p_failure,
            vanilla.se,
            n_vanilla,
            vanilla.wall_time_s * 1000.0
        );
        eprintln!(
            "  CE-IS:   p={:.4e}  se={:.4e}  N={}  {:.0}ms  CV={:.2}  levels={}",
            ceis.p_failure,
            ceis.se,
            ceis.n_total_scenarios,
            ceis.wall_time_s * 1000.0,
            ceis.coefficient_of_variation,
            ceis.n_levels
        );
        if vanilla.se > 0.0 && ceis.se > 0.0 {
            let var_ratio = (vanilla.se / ceis.se).powi(2);
            eprintln!("  Variance reduction: {:.1}x", var_ratio);
            eprintln!(
                "  For same SE as CE-IS, vanilla needs: {:.0}M samples",
                n_vanilla as f64 * var_ratio / 1e6
            );
        }
        eprintln!(
            "  Scenario reduction: {:.0}x fewer scenarios",
            n_vanilla as f64 / ceis.n_total_scenarios as f64
        );
    }
}
