//! WALNUTS sampler and trajectory kernel.
//!
//! This module exposes two layers:
//! - a low-level fixed-tuning WALNUTS kernel for deterministic parity tests
//! - a higher-level adaptive warmup API that returns the standard [`Chain`](crate::chain::Chain)
//!   / [`SamplerResult`](crate::chain::SamplerResult) contract used by the rest of
//!   the Bayesian surface

use crate::adapt::{WindowedAdaptation, find_reasonable_step_size};
use crate::hmc::{HamiltonianStepper, HmcState, LeapfrogIntegrator, Metric};
use crate::nuts::{InitStrategy, MetricType};
use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;
use rand::Rng;

/// Low-level WALNUTS trajectory configuration.
///
/// This controls only the structural trajectory-building rules. Adapted values
/// such as step size, inverse mass matrix, and `min_micro_steps` live in
/// [`WalnutsTuning`].
#[derive(Debug, Clone)]
pub struct WalnutsConfig {
    /// Maximum number of top-level NUTS doublings.
    pub max_treedepth: usize,
    /// Maximum number of dyadic halvings attempted when searching for a
    /// reversible macro step.
    pub max_step_halvings: usize,
    /// Initial minimum number of micro steps per macro step.
    ///
    /// For adaptive warmup this is the starting value; the adapted sampler may
    /// learn a different value and materialize it in [`WalnutsTuning`].
    pub min_micro_steps: usize,
    /// Maximum allowed Hamiltonian error at accepted macro steps.
    pub max_energy_error: f64,
}

impl Default for WalnutsConfig {
    fn default() -> Self {
        Self { max_treedepth: 10, max_step_halvings: 4, min_micro_steps: 1, max_energy_error: 2.0 }
    }
}

impl WalnutsConfig {
    fn validate(&self) -> Result<()> {
        if self.max_treedepth == 0 {
            return Err(ns_core::Error::Validation("max_treedepth must be >= 1".to_string()));
        }
        if self.min_micro_steps == 0 {
            return Err(ns_core::Error::Validation("min_micro_steps must be >= 1".to_string()));
        }
        if !self.max_energy_error.is_finite() || self.max_energy_error <= 0.0 {
            return Err(ns_core::Error::Validation(
                "max_energy_error must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fixed tuning values for WALNUTS.
#[derive(Debug, Clone)]
pub struct WalnutsTuning {
    /// Base micro step size.
    pub step_size: f64,
    /// Diagonal inverse mass matrix.
    ///
    /// This low-level fixed-tuning surface remains diagonal and is mainly used
    /// for deterministic kernel tests. The adaptive product surface
    /// [`sample_walnuts`] supports both diagonal and dense metrics.
    pub inv_mass_diag: Vec<f64>,
    /// Minimum number of micro steps per macro step.
    pub min_micro_steps: usize,
}

impl WalnutsTuning {
    fn validate(&self, dim: usize) -> Result<()> {
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(ns_core::Error::Validation(
                "WALNUTS step_size must be finite and > 0".to_string(),
            ));
        }
        if self.inv_mass_diag.len() != dim {
            return Err(ns_core::Error::Validation(format!(
                "WALNUTS inv_mass_diag dimension mismatch: expected {dim}, got {}",
                self.inv_mass_diag.len()
            )));
        }
        if self.inv_mass_diag.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ns_core::Error::Validation(
                "WALNUTS inv_mass_diag entries must be finite and > 0".to_string(),
            ));
        }
        if self.min_micro_steps == 0 {
            return Err(ns_core::Error::Validation(
                "WALNUTS min_micro_steps must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RuntimeWalnutsTuning {
    step_size: f64,
    metric: Metric,
    min_micro_steps: usize,
}

impl RuntimeWalnutsTuning {
    fn validate(&self, dim: usize) -> Result<()> {
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(ns_core::Error::Validation(
                "WALNUTS step_size must be finite and > 0".to_string(),
            ));
        }
        if self.metric.dim() != dim {
            return Err(ns_core::Error::Validation(format!(
                "WALNUTS metric dimension mismatch: expected {dim}, got {}",
                self.metric.dim()
            )));
        }
        match &self.metric {
            Metric::Diag(diag) => {
                if diag.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(ns_core::Error::Validation(
                        "WALNUTS diagonal inverse mass entries must be finite and > 0".to_string(),
                    ));
                }
            }
            Metric::DenseCholesky { dim, l } => {
                if l.len() != dim * dim {
                    return Err(ns_core::Error::Validation(format!(
                        "WALNUTS dense metric storage mismatch: expected {}, got {}",
                        dim * dim,
                        l.len()
                    )));
                }
                if l.iter().any(|v| !v.is_finite()) {
                    return Err(ns_core::Error::Validation(
                        "WALNUTS dense metric entries must be finite".to_string(),
                    ));
                }
                for i in 0..*dim {
                    let diag = l[i * dim + i];
                    if !diag.is_finite() || diag <= 0.0 {
                        return Err(ns_core::Error::Validation(
                            "WALNUTS dense metric Cholesky diagonal must be finite and > 0"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        if self.min_micro_steps == 0 {
            return Err(ns_core::Error::Validation(
                "WALNUTS min_micro_steps must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// High-level adaptive WALNUTS configuration.
///
/// This is the intended user-facing Rust API. The embedded [`WalnutsConfig`]
/// controls trajectory structure; the remaining fields govern initialization,
/// warmup, and post-warmup sampling behavior.
#[derive(Debug, Clone)]
pub struct AdaptiveWalnutsConfig {
    /// Structural WALNUTS kernel settings.
    pub kernel: WalnutsConfig,
    /// Target macro-step acceptance probability for Adam step-size adaptation.
    pub target_accept: f64,
    /// Target average tree depth used to adapt `min_micro_steps`.
    pub target_tree_depth: f64,
    /// Chain initialization strategy.
    pub init_strategy: InitStrategy,
    /// Absolute jitter in unconstrained space.
    pub init_jitter: f64,
    /// Relative jitter in constrained space, converted locally via the Jacobian.
    pub init_jitter_rel: Option<f64>,
    /// Larger relative jitter intended for overdispersed starts.
    pub init_overdispersed_rel: Option<f64>,
    /// Euclidean metric type used for warmup and post-warmup sampling.
    pub metric_type: MetricType,
    /// Optional post-warmup step-size jitter.
    pub stepsize_jitter: f64,
}

impl Default for AdaptiveWalnutsConfig {
    fn default() -> Self {
        Self {
            kernel: WalnutsConfig::default(),
            target_accept: 0.8,
            target_tree_depth: 4.0,
            init_strategy: InitStrategy::Random,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
            metric_type: MetricType::Diagonal,
            stepsize_jitter: 0.0,
        }
    }
}

impl AdaptiveWalnutsConfig {
    fn validate(&self) -> Result<()> {
        self.kernel.validate()?;
        if !self.target_accept.is_finite() || !(0.0..1.0).contains(&self.target_accept) {
            return Err(ns_core::Error::Validation(
                "target_accept must be finite and in (0, 1)".to_string(),
            ));
        }
        if !self.target_tree_depth.is_finite() || self.target_tree_depth <= 0.0 {
            return Err(ns_core::Error::Validation(
                "target_tree_depth must be finite and > 0".to_string(),
            ));
        }
        if !self.stepsize_jitter.is_finite() || !(0.0..=1.0).contains(&self.stepsize_jitter) {
            return Err(ns_core::Error::Validation(
                "stepsize_jitter must be finite and in [0, 1]".to_string(),
            ));
        }
        let init_modes = (self.init_jitter > 0.0) as u8
            + self.init_jitter_rel.is_some() as u8
            + self.init_overdispersed_rel.is_some() as u8;
        if init_modes > 1 {
            return Err(ns_core::Error::Validation(
                "init_jitter, init_jitter_rel, init_overdispersed_rel are mutually exclusive"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of one WALNUTS transition.
#[derive(Debug, Clone)]
pub struct WalnutsTransition {
    /// Selected position in unconstrained space.
    pub q: Vec<f64>,
    /// Potential at the selected state.
    pub potential: f64,
    /// Gradient of the potential at the selected state.
    pub grad_potential: Vec<f64>,
    /// Top-level tree depth reached by the transition.
    pub depth: usize,
    /// Whether the transition hit a macro-step failure / divergence.
    pub divergent: bool,
    /// Mean first-attempt local acceptance lower bound across macro steps.
    pub accept_prob: f64,
    /// Initial Hamiltonian after momentum resampling.
    pub energy: f64,
    /// Total number of micro leapfrog steps actually executed.
    pub n_leapfrog: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Backward,
    Forward,
}

#[derive(Debug, Clone, Copy)]
enum UpdateRule {
    Barker,
    Metropolis,
}

#[derive(Debug, Clone)]
struct ProposalW {
    q: Vec<f64>,
    potential: f64,
    grad_potential: Vec<f64>,
}

impl ProposalW {
    fn from_state(state: &HmcState) -> Self {
        Self {
            q: state.q.clone(),
            potential: state.potential,
            grad_potential: state.grad_potential.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct SpanW {
    state_bk: HmcState,
    log_joint_bk: f64,
    state_fw: HmcState,
    log_joint_fw: f64,
    selected: ProposalW,
    log_sum_weight: f64,
}

impl SpanW {
    fn from_state(state: HmcState, log_joint: f64) -> Self {
        let selected = ProposalW::from_state(&state);
        Self {
            state_bk: state.clone(),
            log_joint_bk: log_joint,
            state_fw: state,
            log_joint_fw: log_joint,
            selected,
            log_sum_weight: log_joint,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct TransitionStats {
    n_leapfrog: usize,
    n_macro_steps: usize,
    sum_accept_prob: f64,
}

impl TransitionStats {
    fn record_accept(&mut self, accept_prob: f64) {
        let accept_prob = if accept_prob.is_finite() { accept_prob.clamp(0.0, 1.0) } else { 0.0 };
        self.n_macro_steps += 1;
        self.sum_accept_prob += accept_prob;
    }

    fn mean_accept_prob(&self) -> f64 {
        if self.n_macro_steps == 0 {
            0.0
        } else {
            (self.sum_accept_prob / self.n_macro_steps as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanBuildFailure {
    Divergent,
    Uturn,
}

type SpanResult = std::result::Result<SpanW, SpanBuildFailure>;

trait LeafPolicy {
    fn build_leaf<I: HamiltonianStepper + ?Sized>(
        &mut self,
        integrator: &I,
        direction: Direction,
        span: &SpanW,
        stats: &mut TransitionStats,
    ) -> SpanResult;
}

struct SpanNutsPolicy;

impl LeafPolicy for SpanNutsPolicy {
    fn build_leaf<I: HamiltonianStepper + ?Sized>(
        &mut self,
        integrator: &I,
        direction: Direction,
        span: &SpanW,
        stats: &mut TransitionStats,
    ) -> SpanResult {
        span_leaf(integrator, direction, span, stats)
    }
}

struct WalnutsPolicy {
    config: WalnutsConfig,
}

impl LeafPolicy for WalnutsPolicy {
    fn build_leaf<I: HamiltonianStepper + ?Sized>(
        &mut self,
        integrator: &I,
        direction: Direction,
        span: &SpanW,
        stats: &mut TransitionStats,
    ) -> SpanResult {
        walnuts_leaf(integrator, direction, span, &self.config, stats)
    }
}

fn dense_identity_metric(dim: usize) -> Metric {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        l[i * dim + i] = 1.0;
    }
    Metric::DenseCholesky { dim, l }
}

fn initial_metric(metric_type: MetricType, dim: usize) -> Metric {
    match metric_type {
        MetricType::Diagonal => Metric::identity(dim),
        MetricType::Dense => dense_identity_metric(dim),
        MetricType::Auto => {
            if dim <= 32 {
                dense_identity_metric(dim)
            } else {
                Metric::identity(dim)
            }
        }
    }
}

#[inline]
fn log_joint(state: &HmcState, metric: &Metric) -> f64 {
    -state.hamiltonian(metric)
}

#[inline]
fn accept_lower_bound(log_joint_start: f64, log_joint_end: f64) -> f64 {
    if !log_joint_start.is_finite() || !log_joint_end.is_finite() {
        return 0.0;
    }
    (-(log_joint_end - log_joint_start).abs()).exp().clamp(0.0, 1.0)
}

fn is_uturn(direction: Direction, span1: &SpanW, span2: &SpanW, metric: &Metric) -> bool {
    let (span_bk, span_fw) = match direction {
        Direction::Forward => (span1, span2),
        Direction::Backward => (span2, span1),
    };

    let delta_q: Vec<f64> = span_fw
        .state_fw
        .q
        .iter()
        .zip(span_bk.state_bk.q.iter())
        .map(|(&q_fw, &q_bk)| q_fw - q_bk)
        .collect();
    let scaled_diff = metric.mul_inv_mass(&delta_q);
    let dot_fw =
        span_fw.state_fw.p.iter().zip(scaled_diff.iter()).map(|(&p, &dq)| p * dq).sum::<f64>();
    let dot_bk =
        span_bk.state_bk.p.iter().zip(scaled_diff.iter()).map(|(&p, &dq)| p * dq).sum::<f64>();

    !dot_fw.is_finite() || !dot_bk.is_finite() || dot_fw < 0.0 || dot_bk < 0.0
}

fn combine_spans(
    rng: &mut impl Rng,
    update_rule: UpdateRule,
    direction: Direction,
    span_old: SpanW,
    span_new: SpanW,
) -> SpanW {
    let SpanW {
        state_bk: old_state_bk,
        log_joint_bk: old_log_joint_bk,
        state_fw: old_state_fw,
        log_joint_fw: old_log_joint_fw,
        selected: old_selected,
        log_sum_weight: old_log_sum_weight,
    } = span_old;
    let SpanW {
        state_bk: new_state_bk,
        log_joint_bk: new_log_joint_bk,
        state_fw: new_state_fw,
        log_joint_fw: new_log_joint_fw,
        selected: new_selected,
        log_sum_weight: new_log_sum_weight,
    } = span_new;

    let log_sum_weight = crate::tree_hmc::log_sum_exp(old_log_sum_weight, new_log_sum_weight);

    let log_denom = match update_rule {
        UpdateRule::Metropolis => old_log_sum_weight,
        UpdateRule::Barker => log_sum_weight,
    };
    let update_log_prob = new_log_sum_weight - log_denom;
    let update_selected = update_log_prob.is_finite() && rng.random::<f64>().ln() < update_log_prob;
    let selected = if update_selected { new_selected } else { old_selected };

    match direction {
        Direction::Forward => SpanW {
            state_bk: old_state_bk,
            log_joint_bk: old_log_joint_bk,
            state_fw: new_state_fw,
            log_joint_fw: new_log_joint_fw,
            selected,
            log_sum_weight,
        },
        Direction::Backward => SpanW {
            state_bk: new_state_bk,
            log_joint_bk: new_log_joint_bk,
            state_fw: old_state_fw,
            log_joint_fw: old_log_joint_fw,
            selected,
            log_sum_weight,
        },
    }
}

fn within_tolerance<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    initial: &HmcState,
    step_size: f64,
    num_steps: usize,
    max_energy_error: f64,
    log_joint_initial: f64,
    stats: &mut TransitionStats,
) -> bool {
    let outcome = integrator.probe_log_joint_with_eps(initial, step_size, num_steps);
    stats.n_leapfrog += outcome.attempted_steps();
    let log_joint_final = match outcome.into_result() {
        Ok(log_joint_final) => log_joint_final,
        Err(_) => return false,
    };
    if !log_joint_final.is_finite() {
        return false;
    }
    (log_joint_final - log_joint_initial).abs() <= max_energy_error
}

fn reversible<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    end_state: &HmcState,
    log_joint_end: f64,
    step_size: f64,
    num_steps: usize,
    min_micro_steps: usize,
    max_energy_error: f64,
    stats: &mut TransitionStats,
) -> bool {
    if num_steps <= min_micro_steps {
        return true;
    }

    let mut coarser_steps = num_steps;
    let mut coarser_step_size = step_size;
    while coarser_steps > 2 * min_micro_steps {
        coarser_steps /= 2;
        coarser_step_size *= 2.0;
        if within_tolerance(
            integrator,
            end_state,
            -coarser_step_size,
            coarser_steps,
            max_energy_error,
            log_joint_end,
            stats,
        ) {
            return false;
        }
    }
    true
}

fn span_leaf<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    direction: Direction,
    span: &SpanW,
    stats: &mut TransitionStats,
) -> SpanResult {
    let metric = integrator.metric();
    let (start_state, log_joint_start, signed_step) = match direction {
        Direction::Forward => (&span.state_fw, span.log_joint_fw, integrator.step_size()),
        Direction::Backward => (&span.state_bk, span.log_joint_bk, -integrator.step_size()),
    };

    stats.n_leapfrog += 1;
    let mut next_state = start_state.clone();
    if integrator.step_with_eps(&mut next_state, signed_step).is_err() {
        stats.record_accept(0.0);
        return Err(SpanBuildFailure::Divergent);
    }

    let next_log_joint = log_joint(&next_state, metric);
    stats.record_accept(accept_lower_bound(log_joint_start, next_log_joint));
    if !next_log_joint.is_finite() {
        return Err(SpanBuildFailure::Divergent);
    }

    Ok(SpanW::from_state(next_state, next_log_joint))
}

fn macro_step<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    direction: Direction,
    span: &SpanW,
    config: &WalnutsConfig,
    stats: &mut TransitionStats,
) -> Option<(HmcState, f64)> {
    let metric = integrator.metric();
    let (start_state, log_joint_start, base_step) = match direction {
        Direction::Forward => (&span.state_fw, span.log_joint_fw, integrator.step_size()),
        Direction::Backward => (&span.state_bk, span.log_joint_bk, -integrator.step_size()),
    };

    for halving in 0..=config.max_step_halvings {
        let num_steps = config.min_micro_steps << halving;
        let step_size = base_step * 0.5_f64.powi(halving as i32);
        let mut next_state = start_state.clone();
        let outcome = integrator.step_many_with_eps(&mut next_state, step_size, num_steps);
        stats.n_leapfrog += outcome.attempted_steps();
        let ok = outcome.into_result().is_ok();

        let next_log_joint = if ok { log_joint(&next_state, metric) } else { f64::NEG_INFINITY };
        if halving == 0 {
            stats.record_accept(accept_lower_bound(log_joint_start, next_log_joint));
        }
        if !ok || !next_log_joint.is_finite() {
            continue;
        }

        if (next_log_joint - log_joint_start).abs() <= config.max_energy_error
            && reversible(
                integrator,
                &next_state,
                next_log_joint,
                step_size,
                num_steps,
                config.min_micro_steps,
                config.max_energy_error,
                stats,
            )
        {
            return Some((next_state, next_log_joint));
        }
    }

    None
}

fn walnuts_leaf<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    direction: Direction,
    span: &SpanW,
    config: &WalnutsConfig,
    stats: &mut TransitionStats,
) -> SpanResult {
    macro_step(integrator, direction, span, config, stats)
        .map(|(state, log_joint)| SpanW::from_state(state, log_joint))
        .ok_or(SpanBuildFailure::Divergent)
}

fn build_span_with_policy<I: HamiltonianStepper + ?Sized, P: LeafPolicy>(
    integrator: &I,
    rng: &mut impl Rng,
    depth: usize,
    direction: Direction,
    last_span: &SpanW,
    metric: &Metric,
    stats: &mut TransitionStats,
    policy: &mut P,
) -> SpanResult {
    if depth == 0 {
        return policy.build_leaf(integrator, direction, last_span, stats);
    }

    let subspan1 = build_span_with_policy(
        integrator,
        rng,
        depth - 1,
        direction,
        last_span,
        metric,
        stats,
        policy,
    )?;
    let subspan2 = build_span_with_policy(
        integrator,
        rng,
        depth - 1,
        direction,
        &subspan1,
        metric,
        stats,
        policy,
    )?;

    if is_uturn(direction, &subspan1, &subspan2, metric) {
        return Err(SpanBuildFailure::Uturn);
    }

    Ok(combine_spans(rng, UpdateRule::Barker, direction, subspan1, subspan2))
}

fn transition_with_policy<I: HamiltonianStepper + ?Sized, P: LeafPolicy>(
    integrator: &I,
    current: &HmcState,
    max_treedepth: usize,
    rng: &mut impl Rng,
    stats: &mut TransitionStats,
    policy: &mut P,
) -> Result<WalnutsTransition> {
    let metric = integrator.metric();

    let mut root = current.clone();
    root.p = metric.sample_momentum(rng);

    let h0 = root.hamiltonian(metric);
    if !h0.is_finite() {
        return Err(ns_core::Error::Validation(
            "non-finite initial Hamiltonian in WALNUTS transition".to_string(),
        ));
    }

    let mut span_accum = SpanW::from_state(root, -h0);
    let mut depth = 0usize;
    let mut divergent = false;

    for depth_idx in 0..max_treedepth {
        let direction = if rng.random::<bool>() { Direction::Forward } else { Direction::Backward };
        depth = depth_idx + 1;

        match build_span_with_policy(
            integrator,
            rng,
            depth_idx,
            direction,
            &span_accum,
            metric,
            stats,
            policy,
        ) {
            Ok(next_span) => {
                let combined_uturn = is_uturn(direction, &span_accum, &next_span, metric);
                span_accum =
                    combine_spans(rng, UpdateRule::Metropolis, direction, span_accum, next_span);
                if combined_uturn {
                    break;
                }
            }
            Err(SpanBuildFailure::Uturn) => break,
            Err(SpanBuildFailure::Divergent) => {
                divergent = true;
                break;
            }
        }
    }

    Ok(WalnutsTransition {
        q: span_accum.selected.q,
        potential: span_accum.selected.potential,
        grad_potential: span_accum.selected.grad_potential,
        depth,
        divergent,
        accept_prob: stats.mean_accept_prob(),
        energy: h0,
        n_leapfrog: stats.n_leapfrog,
    })
}

pub(crate) fn span_nuts_transition<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    current: &HmcState,
    max_treedepth: usize,
    rng: &mut impl Rng,
) -> Result<WalnutsTransition> {
    let mut stats = TransitionStats::default();
    let mut policy = SpanNutsPolicy;
    transition_with_policy(integrator, current, max_treedepth, rng, &mut stats, &mut policy)
}

/// Run one WALNUTS transition from the current HMC state.
pub fn walnuts_transition<I: HamiltonianStepper + ?Sized>(
    integrator: &I,
    current: &HmcState,
    config: &WalnutsConfig,
    rng: &mut impl Rng,
) -> Result<WalnutsTransition> {
    config.validate()?;

    let mut stats = TransitionStats::default();
    let mut policy = WalnutsPolicy { config: config.clone() };
    transition_with_policy(integrator, current, config.max_treedepth, rng, &mut stats, &mut policy)
}

#[derive(Debug, Clone)]
struct MinMicroStepsAdapter {
    target_tree_depth: f64,
    total_tree_depth: f64,
    count: f64,
}

impl MinMicroStepsAdapter {
    fn new(target_tree_depth: f64, initial_min_micro_steps: usize) -> Self {
        Self {
            target_tree_depth,
            total_tree_depth: target_tree_depth * initial_min_micro_steps.max(1) as f64,
            count: 1.0,
        }
    }

    fn observe(&mut self, tree_depth: usize) {
        self.total_tree_depth += tree_depth as f64;
        self.count += 1.0;
    }

    fn reset(&mut self, initial_min_micro_steps: usize) {
        self.total_tree_depth = self.target_tree_depth * initial_min_micro_steps.max(1) as f64;
        self.count = 1.0;
    }

    fn min_micro_steps(&self) -> usize {
        let mean_depth = self.total_tree_depth / self.count.max(1.0);
        let estimate = (mean_depth / self.target_tree_depth).floor().max(1.0);
        estimate.round() as usize
    }
}

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

fn sample_random_valid_position<M: LogDensityModel>(
    model: &M,
    posterior: &Posterior<'_, M>,
    dim: usize,
    rng: &mut impl Rng,
) -> Result<Vec<f64>> {
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
        Ok(zf)
    } else {
        Ok(z)
    }
}

fn apply_init_jitter<M: LogDensityModel>(
    model: &M,
    posterior: &Posterior<'_, M>,
    theta_init: &[f64],
    z_init: Vec<f64>,
    config: &AdaptiveWalnutsConfig,
    rng: &mut impl Rng,
) -> Result<Vec<f64>> {
    let dim = z_init.len();
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
        let jac = posterior.transform().jacobian_diag(&z_init);
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
                (1.0 + z_init[i].abs()) * frac
            };
            z_sigma = z_sigma.clamp(1e-6, 20.0);
            let normal = Normal::new(0.0, z_sigma).unwrap();
            out.push(z_init[i] + normal.sample(rng));
        }
        Ok(out)
    } else if let Some(frac) = config.init_jitter_rel.filter(|&f| f > 0.0) {
        use rand_distr::{Distribution, Normal};
        let bounds = model.parameter_bounds();
        let jac = posterior.transform().jacobian_diag(&z_init);
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
                (1.0 + z_init[i].abs()) * frac
            };
            z_sigma = z_sigma.clamp(1e-6, 5.0);
            let normal = Normal::new(0.0, z_sigma).unwrap();
            out.push(z_init[i] + normal.sample(rng));
        }
        Ok(out)
    } else if config.init_jitter > 0.0 {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, config.init_jitter).unwrap();
        Ok(z_init.iter().map(|&zi| zi + normal.sample(rng)).collect())
    } else {
        Ok(z_init)
    }
}

fn initialize_walnuts_position<M: LogDensityModel>(
    model: &M,
    posterior: &Posterior<'_, M>,
    rng: &mut impl Rng,
    config: &AdaptiveWalnutsConfig,
) -> Result<(Vec<f64>, Metric)> {
    let dim = posterior.dim();
    let default_metric = initial_metric(config.metric_type, dim);

    match config.init_strategy {
        InitStrategy::Random => {
            let z = sample_random_valid_position(model, posterior, dim, rng)?;
            Ok((z, default_metric))
        }
        InitStrategy::Mle => {
            let theta_init: Vec<f64> = {
                let mle = crate::mle::MaximumLikelihoodEstimator::new();
                match mle.fit_minimum(model) {
                    Ok(r) if r.converged => r.parameters,
                    _ => model.parameter_init(),
                }
            };
            let mut z = posterior.to_unconstrained(&theta_init)?;
            clamp_non_finite(&mut z);
            let z = apply_init_jitter(model, posterior, &theta_init, z, config, rng)?;
            Ok((z, default_metric))
        }
        InitStrategy::Pathfinder => {
            match crate::mams::pathfinder_init_nuts(model, posterior, dim, config.metric_type) {
                Ok((z, metric)) => Ok((z, metric)),
                Err(_) => {
                    let z = sample_random_valid_position(model, posterior, dim, rng)?;
                    Ok((z, default_metric))
                }
            }
        }
    }
}

fn recover_state_after_transition(
    state: &mut HmcState,
    transition: WalnutsTransition,
    last_good_q: &mut Vec<f64>,
    last_good_potential: &mut f64,
    last_good_grad: &mut Vec<f64>,
) -> (bool, f64, usize, f64) {
    let mut divergent = transition.divergent;
    let mut accept_prob = transition.accept_prob;
    let depth = transition.depth;
    let energy = transition.energy;

    state.q = transition.q;
    state.potential = transition.potential;
    state.grad_potential = transition.grad_potential;

    if state.q.iter().any(|v| !v.is_finite()) {
        state.q = last_good_q.clone();
        state.potential = *last_good_potential;
        state.grad_potential = last_good_grad.clone();
        divergent = true;
        accept_prob = 0.0;
    } else {
        last_good_q.clone_from(&state.q);
        *last_good_potential = state.potential;
        last_good_grad.clone_from(&state.grad_potential);
    }

    (divergent, accept_prob, depth, energy)
}

fn collect_walnuts_samples_with_transition_counts<M: LogDensityModel + ?Sized>(
    posterior: &Posterior<'_, M>,
    mut state: HmcState,
    rng: &mut impl Rng,
    n_samples: usize,
    kernel_config: &WalnutsConfig,
    tuning: &RuntimeWalnutsTuning,
    stepsize_jitter: f64,
    n_leapfrog_warmup_total: usize,
) -> Result<crate::chain::Chain> {
    let jitter = stepsize_jitter.clamp(0.0, 1.0);
    let use_jitter = jitter > 0.0;

    let base_config =
        WalnutsConfig { min_micro_steps: tuning.min_micro_steps, ..kernel_config.clone() };
    let final_metric = tuning.metric.clone();
    let fixed_integrator = if !use_jitter {
        Some(LeapfrogIntegrator::new(posterior, tuning.step_size, final_metric.clone()))
    } else {
        None
    };

    let mut last_good_q = state.q.clone();
    let mut last_good_potential = state.potential;
    let mut last_good_grad = state.grad_potential.clone();

    let mut draws_unconstrained = Vec::with_capacity(n_samples);
    let mut draws_constrained = Vec::with_capacity(n_samples);
    let mut divergences = Vec::with_capacity(n_samples);
    let mut tree_depths = Vec::with_capacity(n_samples);
    let mut accept_probs = Vec::with_capacity(n_samples);
    let mut energies = Vec::with_capacity(n_samples);
    let mut leapfrog_counts = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let jittered_integrator;
        let integrator_ref = if let Some(ref fi) = fixed_integrator {
            fi
        } else {
            let u: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let eps = tuning.step_size * (1.0 + jitter * u);
            jittered_integrator = LeapfrogIntegrator::new(posterior, eps, final_metric.clone());
            &jittered_integrator
        };

        let transition = walnuts_transition(integrator_ref, &state, &base_config, rng)?;
        let n_leapfrog = transition.n_leapfrog;
        let (mut divergent, mut accept_prob, depth, energy) = recover_state_after_transition(
            &mut state,
            transition,
            &mut last_good_q,
            &mut last_good_potential,
            &mut last_good_grad,
        );

        let constrained = match posterior.to_constrained(&state.q) {
            Ok(theta) => theta,
            Err(ns_core::Error::Validation(msg))
                if msg.contains("must contain only finite values") =>
            {
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
                return Err(ns_core::Error::Validation(format!(
                    "WALNUTS to_constrained failed: {e}"
                )));
            }
        };

        draws_unconstrained.push(state.q.clone());
        draws_constrained.push(constrained);
        divergences.push(divergent);
        tree_depths.push(depth);
        accept_probs.push(accept_prob);
        energies.push(energy);
        leapfrog_counts.push(n_leapfrog);
    }

    let mass_diag = final_metric.mass_diag();
    let inv_mass_matrix = final_metric.inv_mass_matrix();
    let metric_type_name = final_metric.metric_type_name().to_string();

    Ok(crate::chain::Chain {
        draws_unconstrained,
        draws_constrained,
        divergences,
        tree_depths,
        accept_probs,
        energies,
        n_leapfrog: leapfrog_counts,
        n_leapfrog_warmup_total,
        max_treedepth: kernel_config.max_treedepth,
        step_size: tuning.step_size,
        mass_diag,
        inv_mass_matrix,
        metric_type_name,
    })
}

fn adaptive_walnuts_warmup<M: LogDensityModel>(
    model: &M,
    posterior: &Posterior<'_, M>,
    n_warmup: usize,
    rng: &mut impl Rng,
    config: &AdaptiveWalnutsConfig,
) -> Result<(HmcState, RuntimeWalnutsTuning, usize)> {
    let (z_init, metric) = initialize_walnuts_position(model, posterior, rng, config)?;
    let init_eps = find_reasonable_step_size(posterior, &z_init, &metric, rng);
    let mut adaptation = WindowedAdaptation::new(
        posterior.dim(),
        n_warmup,
        config.target_accept,
        init_eps,
        config.metric_type,
    );
    adaptation.set_metric(metric.clone());
    let init_integrator = LeapfrogIntegrator::new(posterior, init_eps, metric);

    let mut state = init_integrator
        .init_state(z_init)
        .map_err(|e| ns_core::Error::Validation(format!("WALNUTS init_state failed: {e}")))?;
    let mut last_good_q = state.q.clone();
    let mut last_good_potential = state.potential;
    let mut last_good_grad = state.grad_potential.clone();

    let mut min_micro_adapter =
        MinMicroStepsAdapter::new(config.target_tree_depth, config.kernel.min_micro_steps);
    let mut n_leapfrog_warmup_total = 0usize;

    for iter in 0..n_warmup {
        let current_metric = adaptation.metric().clone();
        let tuning = RuntimeWalnutsTuning {
            step_size: adaptation.step_size(),
            metric: current_metric.clone(),
            min_micro_steps: min_micro_adapter.min_micro_steps(),
        };
        let integrator = LeapfrogIntegrator::new(posterior, tuning.step_size, current_metric);
        let transition_config =
            WalnutsConfig { min_micro_steps: tuning.min_micro_steps, ..config.kernel.clone() };

        let transition = walnuts_transition(&integrator, &state, &transition_config, rng)?;
        n_leapfrog_warmup_total += transition.n_leapfrog;
        let (divergent, accept_prob, depth, _energy) = recover_state_after_transition(
            &mut state,
            transition,
            &mut last_good_q,
            &mut last_good_potential,
            &mut last_good_grad,
        );

        let mass_updated = adaptation.update(iter, &state.q, accept_prob);
        if !divergent {
            min_micro_adapter.observe(depth);
        }
        if mass_updated {
            let new_eps = find_reasonable_step_size(posterior, &state.q, adaptation.metric(), rng);
            adaptation.reinit_stepsize(new_eps);
            // A new mass matrix defines a new local integration regime; discard
            // tree-depth history collected under the old metric so warmup does
            // not keep paying for stale `min_micro_steps` inflation.
            min_micro_adapter.reset(config.kernel.min_micro_steps);
        }
    }

    let tuning = RuntimeWalnutsTuning {
        step_size: adaptation.adapted_step_size(),
        metric: adaptation.metric().clone(),
        min_micro_steps: min_micro_adapter.min_micro_steps(),
    };
    Ok((state, tuning, n_leapfrog_warmup_total))
}

/// Run fixed-tuning WALNUTS sampling from `model.parameter_init()`.
pub fn sample_walnuts_fixed<M: LogDensityModel>(
    model: &M,
    n_samples: usize,
    seed: u64,
    tuning: WalnutsTuning,
    config: WalnutsConfig,
) -> Result<crate::chain::Chain> {
    use rand::SeedableRng;

    config.validate()?;
    let posterior = Posterior::new(model);
    tuning.validate(posterior.dim())?;

    let q_init = posterior.to_unconstrained(&model.parameter_init())?;
    let runtime_tuning = RuntimeWalnutsTuning {
        step_size: tuning.step_size,
        metric: Metric::Diag(tuning.inv_mass_diag.clone()),
        min_micro_steps: tuning.min_micro_steps,
    };
    runtime_tuning.validate(posterior.dim())?;
    let metric = runtime_tuning.metric.clone();
    let integrator = LeapfrogIntegrator::new(&posterior, tuning.step_size, metric);
    let state = integrator
        .init_state(q_init)
        .map_err(|e| ns_core::Error::Validation(format!("WALNUTS init_state failed: {e}")))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    collect_walnuts_samples_with_transition_counts(
        &posterior,
        state,
        &mut rng,
        n_samples,
        &config,
        &runtime_tuning,
        0.0,
        0,
    )
}

/// Run adaptive WALNUTS sampling on a single chain.
pub fn sample_walnuts<M: LogDensityModel>(
    model: &M,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    config: AdaptiveWalnutsConfig,
) -> Result<crate::chain::Chain> {
    use rand::SeedableRng;

    config.validate()?;

    let posterior = Posterior::new(model);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (state, tuning, n_leapfrog_warmup_total) =
        adaptive_walnuts_warmup(model, &posterior, n_warmup, &mut rng, &config)?;
    tuning.validate(posterior.dim())?;
    collect_walnuts_samples_with_transition_counts(
        &posterior,
        state,
        &mut rng,
        n_samples,
        &config.kernel,
        &tuning,
        config.stepsize_jitter,
        n_leapfrog_warmup_total,
    )
}

/// Run adaptive WALNUTS sampling on multiple chains in parallel.
pub fn sample_walnuts_multichain(
    model: &impl LogDensityModel,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    config: AdaptiveWalnutsConfig,
) -> Result<crate::chain::SamplerResult> {
    use rayon::prelude::*;

    let chains: Vec<Result<crate::chain::Chain>> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let chain_seed = seed.wrapping_add(chain_id as u64);
            sample_walnuts(model, n_warmup, n_samples, chain_seed, config.clone())
        })
        .collect();

    let chains: Vec<crate::chain::Chain> = chains.into_iter().collect::<Result<Vec<_>>>()?;
    Ok(crate::chain::SamplerResult {
        chains,
        param_names: model.parameter_names(),
        n_warmup,
        n_samples,
        diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::compute_diagnostics;
    use ns_core::traits::{LogDensityModel, PreparedModelRef};
    use rand::SeedableRng;

    struct StdNormal1D;
    struct NarrowNormal1D;
    #[derive(Clone, Copy)]
    struct Correlated2D;

    impl LogDensityModel for StdNormal1D {
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
            vec![(f64::NEG_INFINITY, f64::INFINITY)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.35]
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            Ok(0.5 * params[0] * params[0])
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            Ok(vec![params[0]])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    impl LogDensityModel for NarrowNormal1D {
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
            vec![(f64::NEG_INFINITY, f64::INFINITY)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.25]
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            Ok(50.0 * params[0] * params[0])
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            Ok(vec![100.0 * params[0]])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    impl LogDensityModel for Correlated2D {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            2
        }

        fn parameter_names(&self) -> Vec<String> {
            vec!["x".to_string(), "y".to_string()]
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0, 0.0]
        }

        fn nll(&self, p: &[f64]) -> Result<f64> {
            let rho = 0.95;
            let det_factor = 1.0 / (1.0 - rho * rho);
            let x = p[0];
            let y = p[1];
            let q = det_factor * (x * x - 2.0 * rho * x * y + y * y);
            let ln2pi = (2.0 * std::f64::consts::PI).ln();
            let ln_det = (1.0 - rho * rho).ln();
            Ok(0.5 * q + 0.5 * ln_det + ln2pi)
        }

        fn grad_nll(&self, p: &[f64]) -> Result<Vec<f64>> {
            let rho = 0.95;
            let det_factor = 1.0 / (1.0 - rho * rho);
            let x = p[0];
            let y = p[1];
            Ok(vec![det_factor * (x - rho * y), det_factor * (y - rho * x)])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    fn std_normal_setup(q0: f64, step: f64) -> (Posterior<'static, StdNormal1D>, Metric, HmcState) {
        let model = Box::leak(Box::new(StdNormal1D));
        let posterior = Posterior::new(model);
        let metric = Metric::identity(1);
        let integrator = LeapfrogIntegrator::new(&posterior, step, metric.clone());
        let state = integrator.init_state(vec![q0]).unwrap();
        (posterior, metric, state)
    }

    #[test]
    fn test_walnuts_matches_span_nuts_without_halving() {
        let (posterior, metric, state) = std_normal_setup(0.35, 0.25);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.25, metric);

        let mut rng_a = rand::rngs::StdRng::seed_from_u64(12345);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(12345);

        let span = span_nuts_transition(&integrator, &state, 5, &mut rng_a).unwrap();
        let walnuts = walnuts_transition(
            &integrator,
            &state,
            &WalnutsConfig {
                max_treedepth: 5,
                max_step_halvings: 0,
                min_micro_steps: 1,
                max_energy_error: 1e6,
            },
            &mut rng_b,
        )
        .unwrap();

        assert_eq!(span.q, walnuts.q);
        assert_eq!(span.potential, walnuts.potential);
        assert_eq!(span.grad_potential, walnuts.grad_potential);
        assert_eq!(span.depth, walnuts.depth);
        assert_eq!(span.n_leapfrog, walnuts.n_leapfrog);
        assert_eq!(span.divergent, walnuts.divergent);
    }

    #[test]
    fn test_walnuts_transition_is_deterministic() {
        let (posterior, metric, state) = std_normal_setup(-0.8, 0.35);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.35, metric);
        let config = WalnutsConfig {
            max_treedepth: 6,
            max_step_halvings: 2,
            min_micro_steps: 1,
            max_energy_error: 0.3,
        };

        let mut rng_a = rand::rngs::StdRng::seed_from_u64(77);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(77);

        let a = walnuts_transition(&integrator, &state, &config, &mut rng_a).unwrap();
        let b = walnuts_transition(&integrator, &state, &config, &mut rng_b).unwrap();

        assert_eq!(a.q, b.q);
        assert_eq!(a.potential, b.potential);
        assert_eq!(a.grad_potential, b.grad_potential);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.n_leapfrog, b.n_leapfrog);
        assert_eq!(a.divergent, b.divergent);
    }

    #[test]
    fn test_walnuts_transition_accepts_dense_metric() {
        let (posterior, _metric, state) = std_normal_setup(0.4, 0.25);
        let metric = Metric::DenseCholesky { dim: 1, l: vec![1.0] };
        let integrator = LeapfrogIntegrator::new(&posterior, 0.25, metric);
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);

        let transition = walnuts_transition(
            &integrator,
            &state,
            &WalnutsConfig {
                max_treedepth: 4,
                max_step_halvings: 1,
                min_micro_steps: 1,
                max_energy_error: 0.3,
            },
            &mut rng,
        )
        .unwrap();

        assert_eq!(transition.q.len(), 1);
        assert!(transition.accept_prob.is_finite());
    }

    #[test]
    fn test_min_micro_steps_adapter_tracks_target_depth() {
        let mut adapt = MinMicroStepsAdapter::new(4.0, 1);
        assert_eq!(adapt.min_micro_steps(), 1);
        adapt.observe(8);
        adapt.observe(8);
        assert!(adapt.min_micro_steps() >= 1);
    }

    #[test]
    fn test_min_micro_steps_adapter_reset_discards_stale_history() {
        let mut adapt = MinMicroStepsAdapter::new(4.0, 1);
        adapt.observe(12);
        adapt.observe(12);
        assert!(adapt.min_micro_steps() > 1);
        adapt.reset(1);
        assert_eq!(adapt.min_micro_steps(), 1);
    }

    #[test]
    fn test_walnuts_default_config_matches_product_surface() {
        let config = WalnutsConfig::default();
        assert_eq!(config.max_treedepth, 10);
        assert_eq!(config.max_step_halvings, 4);
        assert_eq!(config.min_micro_steps, 1);
        assert_eq!(config.max_energy_error, 2.0);
    }

    #[test]
    fn test_span_from_state_preserves_endpoints_and_selected_proposal() {
        let (_posterior, _metric, state) = std_normal_setup(0.35, 0.25);
        let log_joint = -1.2345;

        let span = SpanW::from_state(state, log_joint);

        assert_eq!(span.log_joint_bk, log_joint);
        assert_eq!(span.log_joint_fw, log_joint);
        assert_eq!(span.log_sum_weight, log_joint);
        assert_eq!(span.state_bk.q, span.state_fw.q);
        assert_eq!(span.state_bk.p, span.state_fw.p);
        assert_eq!(span.state_bk.grad_potential, span.state_fw.grad_potential);
        assert_eq!(span.selected.q, span.state_fw.q);
        assert_eq!(span.selected.potential, span.state_fw.potential);
        assert_eq!(span.selected.grad_potential, span.state_fw.grad_potential);
    }

    #[test]
    fn test_reversibility_probes_count_toward_n_leapfrog() {
        let (posterior, metric, initial_state) = std_normal_setup(0.35, 0.25);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.25, metric);

        let mut end_state = initial_state.clone();
        for _ in 0..4 {
            integrator.step_with_eps(&mut end_state, 0.25 / 4.0).unwrap();
        }
        let log_joint_end = log_joint(&end_state, integrator.metric());

        let mut stats = TransitionStats::default();
        let _ =
            reversible(&integrator, &end_state, log_joint_end, 0.25 / 4.0, 4, 1, 2.0, &mut stats);

        assert_eq!(stats.n_leapfrog, 2);
    }

    #[test]
    fn test_negative_eps_probe_matches_momentum_flip_probe() {
        let (posterior, metric, initial_state) = std_normal_setup(0.35, 0.25);
        let integrator = LeapfrogIntegrator::new(&posterior, 0.25, metric);

        let mut end_state = initial_state.clone();
        for _ in 0..4 {
            integrator.step_with_eps(&mut end_state, 0.25 / 4.0).unwrap();
        }
        let log_joint_end = log_joint(&end_state, integrator.metric());

        let mut flipped = end_state.clone();
        for p in &mut flipped.p {
            *p = -*p;
        }

        let mut flipped_stats = TransitionStats::default();
        let mut negative_eps_stats = TransitionStats::default();
        let flipped_ok = within_tolerance(
            &integrator,
            &flipped,
            0.25 / 2.0,
            2,
            2.0,
            log_joint_end,
            &mut flipped_stats,
        );
        let negative_eps_ok = within_tolerance(
            &integrator,
            &end_state,
            -(0.25 / 2.0),
            2,
            2.0,
            log_joint_end,
            &mut negative_eps_stats,
        );

        assert_eq!(flipped_ok, negative_eps_ok);
        assert_eq!(flipped_stats.n_leapfrog, negative_eps_stats.n_leapfrog);
    }

    #[test]
    fn test_sample_walnuts_adapts_narrow_posterior_to_large_mass_diag() {
        let model = NarrowNormal1D;
        let chain = sample_walnuts(&model, 200, 20, 123, AdaptiveWalnutsConfig::default()).unwrap();
        assert!(chain.mass_diag[0].is_finite());
        assert!(
            chain.mass_diag[0] > 5.0,
            "expected large mass_diag for concentrated posterior, got {}",
            chain.mass_diag[0]
        );
    }

    #[test]
    fn test_sample_walnuts_fixed_smoke() {
        let model = StdNormal1D;
        let chain = sample_walnuts_fixed(
            &model,
            40,
            42,
            WalnutsTuning { step_size: 0.25, inv_mass_diag: vec![1.0], min_micro_steps: 1 },
            WalnutsConfig {
                max_treedepth: 5,
                max_step_halvings: 1,
                min_micro_steps: 1,
                max_energy_error: 0.5,
            },
        )
        .unwrap();

        assert_eq!(chain.draws_constrained.len(), 40);
        assert_eq!(chain.draws_unconstrained.len(), 40);
        assert_eq!(chain.divergences.len(), 40);
        assert_eq!(chain.n_leapfrog.len(), 40);
        assert_eq!(chain.n_leapfrog_warmup_total, 0);
        assert!(chain.step_size > 0.0);
    }

    #[test]
    fn test_sample_walnuts_dense_metric_correlated() {
        let model = Correlated2D;
        let config = AdaptiveWalnutsConfig { metric_type: MetricType::Dense, ..Default::default() };
        let result = sample_walnuts_multichain(&model, 2, 500, 100, 42, config).unwrap();
        let diag = compute_diagnostics(&result);

        for (i, &rhat) in diag.r_hat.iter().enumerate() {
            assert!(
                rhat.is_finite(),
                "R-hat for param {} should be finite with dense WALNUTS",
                result.param_names[i],
            );
        }

        assert_eq!(result.chains[0].metric_type_name, "dense");
        assert!(result.chains[0].inv_mass_matrix.is_some());
        let inv_mass = result.chains[0].inv_mass_matrix.as_ref().unwrap();
        assert_eq!(inv_mass.len(), 4);
        assert!(
            inv_mass[1].abs() > 0.1,
            "off-diagonal of dense WALNUTS inv_mass should be substantial: {}",
            inv_mass[1]
        );
    }

    #[test]
    fn test_sample_walnuts_auto_metric_uses_dense_for_small_dim() {
        let model = Correlated2D;
        let config = AdaptiveWalnutsConfig { metric_type: MetricType::Auto, ..Default::default() };
        let chain = sample_walnuts(&model, 50, 20, 42, config).unwrap();

        assert_eq!(chain.metric_type_name, "dense");
        assert!(chain.inv_mass_matrix.is_some());
    }

    #[test]
    fn test_sample_walnuts_smoke() {
        let model = StdNormal1D;
        let chain = sample_walnuts(&model, 20, 30, 123, AdaptiveWalnutsConfig::default()).unwrap();
        assert_eq!(chain.draws_constrained.len(), 30);
        assert_eq!(chain.draws_unconstrained.len(), 30);
        assert_eq!(chain.divergences.len(), 30);
        assert_eq!(chain.n_leapfrog.len(), 30);
        assert!(chain.n_leapfrog_warmup_total > 0);
        assert!(chain.step_size.is_finite() && chain.step_size > 0.0);
    }

    #[test]
    fn test_sample_walnuts_is_deterministic() {
        let model = StdNormal1D;
        let c1 = sample_walnuts(&model, 20, 20, 77, AdaptiveWalnutsConfig::default()).unwrap();
        let c2 = sample_walnuts(&model, 20, 20, 77, AdaptiveWalnutsConfig::default()).unwrap();
        assert_eq!(c1.draws_constrained, c2.draws_constrained);
        assert_eq!(c1.tree_depths, c2.tree_depths);
    }

    #[test]
    fn test_sample_walnuts_multichain_basic() {
        let model = StdNormal1D;
        let result =
            sample_walnuts_multichain(&model, 2, 20, 15, 42, AdaptiveWalnutsConfig::default())
                .unwrap();
        assert_eq!(result.chains.len(), 2);
        assert_eq!(result.n_warmup, 20);
        assert_eq!(result.n_samples, 15);
    }
}
