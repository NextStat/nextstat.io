//! Metropolis-Adjusted Microcanonical Sampler (MAMS).
//!
//! Rust implementation of MAMS, based on arXiv:2503.01707 (Robnik et al., 2025).
//! The paper reports 2-7x ESS/gradient improvements over NUTS on their benchmarks
//! and improved stability on multiscale/funnel geometries.
//!
//! Key differences from NUTS/HMC:
//! - **Unit velocity**: `u` lives on the unit sphere `|u| = 1`. The Euclidean norm
//!   is preserved exactly by the projected integrator.
//! - **Isokinetic leapfrog**: velocity update uses projected gradient
//!   `u̇ = -(I - uu^T)∇ℒ(x)/(d-1)` followed by renormalization to `|u|=1`.
//! - **Partial refreshment**: velocity is mixed with Gaussian noise and
//!   renormalized: `u ← normalize(c₁·u + c₂·z/√d)` with `c₁ = exp(-ε/L)`.
//! - **Fixed trajectory length**: each transition uses `L/ε` leapfrog steps
//!   (no tree building), making per-transition cost predictable.
//! - **MH correction** with microcanonical kinetic energy error:
//!   `ΔK = (d-1)·[δ − ln2 + ln((1−e·u) + ζ(1+e·u))]` where `ζ = exp(−2δ)`.
//!   This log-sum-exp form avoids `cosh`/`sinh` overflow for large gradients.

use crate::adapt::{DualAveraging, WelfordVariance};
use crate::nuts::{InitStrategy, MetricType};
use crate::posterior::Posterior;
use ns_core::Result;
use ns_core::traits::LogDensityModel;
use rand::Rng;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// MAMS sampler configuration.
#[derive(Debug, Clone)]
pub struct MamsConfig {
    /// Number of warmup iterations (default: 1000).
    pub n_warmup: usize,
    /// Number of post-warmup samples (default: 1000).
    pub n_samples: usize,
    /// Target MH acceptance rate (default: 0.9).
    pub target_accept: f64,
    /// Initial step size; 0 = auto-detect (default: 0).
    pub init_step_size: f64,
    /// Initial decoherence length L; 0 = auto-tune (default: 0).
    pub init_l: f64,
    /// Maximum leapfrog steps per trajectory (default: 32768).
    pub max_leapfrog: usize,
    /// Use diagonal preconditioning (default: true).
    pub diagonal_precond: bool,
    /// Chain initialization strategy (default: Random).
    pub init_strategy: InitStrategy,
    /// Euclidean metric type (default: Diagonal).
    pub metric_type: MetricType,
}

impl Default for MamsConfig {
    fn default() -> Self {
        Self {
            n_warmup: 1000,
            n_samples: 1000,
            target_accept: 0.9,
            init_step_size: 0.0,
            init_l: 0.0,
            max_leapfrog: 32768,
            diagonal_precond: true,
            init_strategy: InitStrategy::Random,
            metric_type: MetricType::Diagonal,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-space state
// ---------------------------------------------------------------------------

/// Phase-space state for microcanonical dynamics.
#[derive(Debug, Clone)]
struct MicrocanonicalState {
    /// Position in unconstrained space.
    x: Vec<f64>,
    /// Unit velocity on S^{d-1}: |u| = 1.
    u: Vec<f64>,
    /// Potential energy U(x) = -logpdf(x) = ℒ(x).
    potential: f64,
    /// Gradient of potential ∇U(x) = -∇logpdf(x) = ∇ℒ(x).
    grad_potential: Vec<f64>,
}

/// Result of one MAMS transition.
struct MamsTransitionResult {
    /// New position.
    x: Vec<f64>,
    /// New unit velocity.
    u: Vec<f64>,
    /// New potential energy.
    potential: f64,
    /// New gradient.
    grad_potential: Vec<f64>,
    /// Whether the MH proposal was accepted.
    accepted: bool,
    /// MH log-weight (energy error ΔV + ΔK).
    energy_error: f64,
    /// Number of leapfrog steps taken.
    n_leapfrog: usize,
}

// ---------------------------------------------------------------------------
// Unit-sphere helpers
// ---------------------------------------------------------------------------

/// Normalize a vector to unit length. Returns the original norm.
fn normalize(v: &mut [f64]) -> f64 {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-30 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
    norm
}

/// Sample a random unit vector uniformly on S^{d-1}.
fn sample_unit_vector(dim: usize, rng: &mut impl Rng) -> Vec<f64> {
    use rand_distr::{Distribution, StandardNormal};
    let mut u: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
    normalize(&mut u);
    u
}

// ---------------------------------------------------------------------------
// Isokinetic leapfrog integrator (MAMS, arXiv:2503.01707)
// ---------------------------------------------------------------------------

/// One full isokinetic leapfrog step: B(ε/2) · A(ε) · B(ε/2).
///
/// The velocity `u` lives on the unit sphere |u| = 1. The update uses
/// the projected gradient: `u̇ = -(I - uu^T)∇ℒ(x)/(d-1)`.
///
/// **B-step** (half-step velocity):
///   δ = (ε/2)|g|/(d-1), e = g/|g|
///   e_dot_u = e · u
///   u_new = u·cosh(δ) - e·sinh(δ), then renormalize
///   ΔK += (d-1)·ln(cosh(δ) + e_dot_u·sinh(δ))
///
/// **A-step** (full-step position):
///   x += ε·u
///   recompute gradient
///
/// Returns ΔK (total kinetic energy error from both B-steps).
fn isokinetic_leapfrog_step(
    state: &mut MicrocanonicalState,
    eps: f64,
    posterior: &Posterior<'_, impl LogDensityModel + ?Sized>,
    inv_mass: &[f64],
) -> Result<f64> {
    let dim = state.x.len();
    let d = dim as f64;
    let dm1 = d - 1.0;
    if dm1 < 0.5 {
        // 1D case: projected gradient vanishes, just update position
        for i in 0..dim {
            state.x[i] += eps * state.u[i];
        }
        let new_potential = match posterior.logpdf_unconstrained(&state.x) {
            Ok(lp) if lp.is_finite() => -lp,
            _ => return Err(ns_core::Error::Validation("non-finite potential in leapfrog".into())),
        };
        let new_grad = posterior.grad_unconstrained(&state.x)?;
        state.potential = new_potential;
        state.grad_potential = new_grad.iter().map(|g| -g).collect();
        return Ok(0.0);
    }

    let mut delta_k = 0.0;

    // --- First B-step (ε/2) ---
    delta_k += b_step(state, eps * 0.5, inv_mass, dm1);

    // --- A-step (ε) ---
    // Position update in whitened space: q += ε·u.
    // Back to x-space: x_i += ε · √(inv_mass_i) · u_i.
    // This matches the B-step which uses g̃_i = √(inv_mass_i) · ∇U_i.
    for i in 0..dim {
        state.x[i] += eps * inv_mass[i].sqrt() * state.u[i];
    }

    // Recompute potential and gradient at new position
    let new_potential = match posterior.logpdf_unconstrained(&state.x) {
        Ok(lp) if lp.is_finite() => -lp,
        _ => return Err(ns_core::Error::Validation("non-finite potential in leapfrog".into())),
    };
    let new_grad = posterior.grad_unconstrained(&state.x)?;
    state.potential = new_potential;
    state.grad_potential = new_grad.iter().map(|g| -g).collect();

    // --- Second B-step (ε/2) ---
    delta_k += b_step(state, eps * 0.5, inv_mass, dm1);

    Ok(delta_k)
}

/// B-step: half-step velocity update on the unit sphere.
///
/// Computes the preconditioned gradient: g_tilde_i = sqrt(inv_mass_i) * grad_i
/// Then projects onto tangent plane of u and exponential-maps.
///
/// Uses `exp_m1`/`ln_1p` for full mantissa precision at both extremes:
/// - Large δ (funnel bottom): `ζ_m1 ≈ -1`, same stability as old ζ formulation.
/// - Small δ (well-behaved): `exp_m1` avoids catastrophic cancellation in `1 − exp(−2δ)`.
///
/// Returns ΔK contribution from this half-step.
fn b_step(state: &mut MicrocanonicalState, half_eps: f64, inv_mass: &[f64], dm1: f64) -> f64 {
    let dim = state.x.len();

    // Preconditioned gradient: g̃_i = √(inv_mass_i) · grad_potential_i.
    let mut g_norm_sq = 0.0;
    for i in 0..dim {
        let gi = inv_mass[i].sqrt() * state.grad_potential[i];
        g_norm_sq += gi * gi;
    }
    let g_norm = g_norm_sq.sqrt();

    if g_norm < 1e-30 {
        return 0.0;
    }

    // e = g̃/|g̃|  (unit gradient direction — UP the potential).
    let mut e_dot_u = 0.0;
    for i in 0..dim {
        let ei = inv_mass[i].sqrt() * state.grad_potential[i] / g_norm;
        e_dot_u += ei * state.u[i];
    }
    let e_dot_u = e_dot_u.clamp(-1.0, 1.0);

    // δ = half_eps · |g̃| / (d−1).
    let delta = half_eps * g_norm / dm1;

    // --- Full-precision formulation via ζ_m1 = exp(−2δ) − 1 ---
    //
    // ζ = exp(−2δ),  ζ_m1 = ζ − 1 = exp_m1(−2δ).
    // cosh δ ∝ (2 + ζ_m1),  sinh δ ∝ (−ζ_m1).
    //
    // Small δ: exp_m1(−2δ) ≈ −2δ + 2δ²/2 − ...  (no cancellation in 1−exp).
    // Large δ: ζ_m1 → −1.
    let zeta_m1 = (-2.0 * delta).exp_m1(); // = exp(−2δ) − 1

    let c_u = 2.0 + zeta_m1; // = 1 + ζ, proportional to cosh δ
    let c_e = -zeta_m1; // = 1 − ζ, proportional to sinh δ

    // u_new ∝ u·cosh δ − e·sinh δ  ∝  u·c_u − e·c_e
    let mut u_new_norm_sq = 0.0;
    for i in 0..dim {
        let ei = inv_mass[i].sqrt() * state.grad_potential[i] / g_norm;
        state.u[i] = state.u[i] * c_u - ei * c_e;
        u_new_norm_sq += state.u[i] * state.u[i];
    }
    let u_new_norm = u_new_norm_sq.sqrt();
    if u_new_norm > 1e-12 {
        let inv = 1.0 / u_new_norm;
        for ui in state.u.iter_mut() {
            *ui *= inv;
        }
    } else {
        // Degenerate: velocity collapsed to zero → point along −e (down the potential)
        for i in 0..dim {
            state.u[i] = -inv_mass[i].sqrt() * state.grad_potential[i] / g_norm;
        }
    }

    // ΔK = (d−1) · [δ + ln(1 + 0.5·ζ_m1·(1 + e·u))]
    //
    // Derivation: (d−1)·ln(cosh δ − (e·u)·sinh δ)
    //   = (d−1)·[δ − ln2 + ln((1−e·u) + ζ(1+e·u))]
    //   = (d−1)·[δ − ln2 + ln(2 + ζ_m1·(1+e·u))]
    //   = (d−1)·[δ + ln(1 + 0.5·ζ_m1·(1+e·u))]
    //
    // ln_1p avoids cancellation when δ is small (arg ≈ 0).
    let arg = 0.5 * zeta_m1 * (1.0 + e_dot_u);
    dm1 * (delta + arg.max(-1.0 + 1e-50).ln_1p())
}

// ---------------------------------------------------------------------------
// Partial velocity refresh on the unit sphere
// ---------------------------------------------------------------------------

/// Partial velocity refresh via exact spherical rotation (Gram-Schmidt).
///
/// Rotates `u` by angle `θ = ε/L` towards a random direction orthogonal to `u`:
///   1. Sample z ~ N(0, I)
///   2. Gram-Schmidt: z_perp = z - (u·z)u, then normalize to unit
///   3. u_new = u·cos(θ) + z_perp_hat·sin(θ)
///
/// This preserves |u| = 1 exactly (up to floating point) and avoids the
/// `normalize(c₁u + c₂z/√d)` formulation which distorts the rotation angle
/// in low dimensions.
fn partial_velocity_refresh_sphere(u: &mut [f64], eps: f64, l: f64, rng: &mut impl Rng) {
    use rand_distr::{Distribution, StandardNormal};

    let dim = u.len();
    let angle = eps / l;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Sample random direction and project out u-component (Gram-Schmidt)
    let mut z = vec![0.0; dim];
    let mut u_dot_z = 0.0;
    for i in 0..dim {
        z[i] = StandardNormal.sample(rng);
        u_dot_z += u[i] * z[i];
    }

    let mut z_perp_norm_sq = 0.0;
    for i in 0..dim {
        z[i] -= u_dot_z * u[i];
        z_perp_norm_sq += z[i] * z[i];
    }
    let z_perp_norm = z_perp_norm_sq.sqrt();

    if z_perp_norm > 1e-12 {
        let inv_norm = 1.0 / z_perp_norm;
        let mut u_new_norm_sq = 0.0;
        for i in 0..dim {
            u[i] = u[i] * cos_a + z[i] * inv_norm * sin_a;
            u_new_norm_sq += u[i] * u[i];
        }
        // Renormalize to compensate for floating-point drift
        let u_new_norm = u_new_norm_sq.sqrt();
        let inv = 1.0 / u_new_norm;
        for ui in u.iter_mut() {
            *ui *= inv;
        }
    }
    // If z_perp_norm ≈ 0, z was parallel to u — skip rotation (no fresh direction).
}

// ---------------------------------------------------------------------------
// MAMS transition
// ---------------------------------------------------------------------------

/// Run one MAMS transition: partial refresh → leapfrog trajectory → MH accept/reject.
///
/// The MH criterion uses the total energy error: ΔH = ΔV + ΔK,
/// where ΔV = U(x') - U(x) is the potential energy change, and
/// ΔK is the accumulated kinetic energy error from the isokinetic integrator.
fn mams_transition(
    state: &MicrocanonicalState,
    eps: f64,
    l: f64,
    n_steps: usize,
    posterior: &Posterior<'_, impl LogDensityModel + ?Sized>,
    inv_mass: &[f64],
    rng: &mut impl Rng,
) -> Result<MamsTransitionResult> {
    // Clone state for proposal
    let mut proposal = state.clone();

    // 1. Partial velocity refresh on the sphere
    partial_velocity_refresh_sphere(&mut proposal.u, eps, l, rng);

    // Save state after refresh (for MH comparison and rejection)
    let x_old = proposal.x.clone();
    let u_old = proposal.u.clone();
    let potential_old = proposal.potential;
    let grad_old = proposal.grad_potential.clone();

    // 2. Isokinetic leapfrog integration with early termination.
    //
    // Check total energy error (ΔV + ΔK) after each step — not just ΔV.
    // This saves gradient evaluations on trajectories that are already doomed
    // (e.g. in the narrow neck of a funnel).
    let mut n_leapfrog = 0;
    let mut divergent = false;
    let mut total_delta_k = 0.0;
    for _ in 0..n_steps {
        match isokinetic_leapfrog_step(&mut proposal, eps, posterior, inv_mass) {
            Ok(dk) => {
                total_delta_k += dk;
                n_leapfrog += 1;
            }
            Err(_) => {
                divergent = true;
                break;
            }
        }
        // Early termination: total energy error ΔV + ΔK
        let current_w = (proposal.potential - potential_old) + total_delta_k;
        if !current_w.is_finite() || current_w > 1000.0 {
            divergent = true;
            break;
        }
    }

    if divergent || proposal.x.iter().any(|v| !v.is_finite()) {
        // Reject: return to pre-refresh state with negated velocity
        let u_neg: Vec<f64> = u_old.iter().map(|ui| -ui).collect();
        return Ok(MamsTransitionResult {
            x: x_old,
            u: u_neg,
            potential: potential_old,
            grad_potential: grad_old,
            accepted: false,
            energy_error: f64::INFINITY,
            n_leapfrog,
        });
    }

    // 3. Metropolis-Hastings acceptance
    //
    // ΔH = ΔV + ΔK
    // ΔV = U(x') - U(x)
    // ΔK = accumulated from isokinetic leapfrog B-steps
    let delta_v = proposal.potential - potential_old;
    let w = delta_v + total_delta_k;

    let accept = w.is_finite() && (w <= 0.0 || rng.random::<f64>() < (-w).exp());

    if accept {
        Ok(MamsTransitionResult {
            x: proposal.x,
            u: proposal.u,
            potential: proposal.potential,
            grad_potential: proposal.grad_potential,
            accepted: true,
            energy_error: w,
            n_leapfrog,
        })
    } else {
        // Reject: return to pre-refresh position with negated velocity
        let u_neg: Vec<f64> = u_old.iter().map(|ui| -ui).collect();
        Ok(MamsTransitionResult {
            x: x_old,
            u: u_neg,
            potential: potential_old,
            grad_potential: grad_old,
            accepted: false,
            energy_error: w,
            n_leapfrog,
        })
    }
}

// ---------------------------------------------------------------------------
// Auto-tuning (warmup)
// ---------------------------------------------------------------------------

/// Find a reasonable initial step size for MAMS by testing short trajectories.
///
/// Binary search: if acceptance > 0.95 → double ε, if < 0.5 → halve ε.
fn find_mams_step_size(
    state: &MicrocanonicalState,
    l: f64,
    posterior: &Posterior<'_, impl LogDensityModel + ?Sized>,
    inv_mass: &[f64],
    rng: &mut impl Rng,
) -> f64 {
    // Use the actual trajectory length L (not relative to ε).
    // Start with ε such that we get ~10 leapfrog steps.
    let mut eps = (l / 10.0).max(0.01);

    // Binary search (~10 iterations)
    for _ in 0..10 {
        let n_steps = ((l / eps).round() as usize).clamp(1, 1024);
        let n_test = 5;
        let mut n_accepted = 0;
        for _ in 0..n_test {
            match mams_transition(state, eps, l, n_steps, posterior, inv_mass, rng) {
                Ok(r) if r.accepted => n_accepted += 1,
                _ => {}
            }
        }
        let acc = n_accepted as f64 / n_test as f64;
        if acc > 0.95 {
            eps *= 1.5; // grow more cautiously
        } else if acc < 0.5 {
            eps *= 0.5;
        } else {
            break;
        }
        // Keep ε within reasonable range: at least 1e-4, at most L/2
        eps = eps.clamp(1e-4, l * 0.5);
    }

    eps.clamp(1e-4, l * 0.5)
}

// estimate_ess_simple moved to crate::adapt::estimate_ess_simple
use crate::adapt::estimate_ess_simple;

/// Tune the decoherence length L by trying several candidates and picking
/// the one with best ESS per gradient evaluation.
fn tune_decoherence_length(
    state: &MicrocanonicalState,
    eps: f64,
    posterior: &Posterior<'_, impl LogDensityModel + ?Sized>,
    inv_mass: &[f64],
    rng: &mut impl Rng,
) -> f64 {
    let n_trials = 40; // transitions per candidate L

    // Minimum L: π/2 (absolute floor, independent of ε).
    // This ensures trajectories are always long enough for decorrelation.
    let l_min = (std::f64::consts::FRAC_PI_2).max(eps * 4.0);
    let mut best_l = l_min;
    let mut best_ess_per_grad = 0.0_f64;

    // Try L ∈ {l_min, 2·l_min, 4·l_min, ..., 64·l_min}
    let mut l_candidate = l_min;
    for _ in 0..7 {
        let n_steps = ((l_candidate / eps).round() as usize).clamp(1, 1024);
        let mut draws = Vec::with_capacity(n_trials);
        let mut total_leapfrog = 0usize;
        let mut current = state.clone();

        for _ in 0..n_trials {
            match mams_transition(&current, eps, l_candidate, n_steps, posterior, inv_mass, rng) {
                Ok(r) => {
                    total_leapfrog += r.n_leapfrog;
                    current.x = r.x;
                    current.u = r.u;
                    current.potential = r.potential;
                    current.grad_potential = r.grad_potential;
                    // Use potential energy as global ESS proxy
                    draws.push(current.potential);
                }
                Err(_) => {
                    total_leapfrog += 1;
                    draws.push(current.potential);
                }
            }
        }

        if total_leapfrog > 0 && draws.len() >= 4 {
            let ess = estimate_ess_simple(&draws);
            let ess_per_grad = ess / total_leapfrog as f64;
            if ess_per_grad > best_ess_per_grad {
                best_ess_per_grad = ess_per_grad;
                best_l = l_candidate;
            }
        }

        l_candidate *= 2.0;
    }

    best_l
}

// ---------------------------------------------------------------------------
// Clamp non-finite unconstrained values (shared with nuts.rs)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pathfinder initialization: MLE mode + diagonal inverse Hessian → inv_mass
// ---------------------------------------------------------------------------

/// Run Pathfinder initialization: L-BFGS to mode + diagonal inverse Hessian.
/// Returns `(z_init, inv_mass_diag)` in unconstrained space.
///
/// Used by both MAMS and NUTS init strategies.
pub(crate) fn pathfinder_init_nuts<M: LogDensityModel>(
    model: &M,
    posterior: &Posterior<M>,
    dim: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mle = crate::mle::MaximumLikelihoodEstimator::new();
    let fr = mle.fit(model)?; // Full fit with Hessian → uncertainties

    // Position: constrained → unconstrained
    let mut z = posterior.to_unconstrained(&fr.parameters)?;
    clamp_non_finite(&mut z);

    // inv_mass from uncertainties: var[i] = uncertainty[i]^2
    // This is a constrained-space approximation; Welford Phase 2 will refine it.
    let mut inv_mass = vec![1.0; dim];
    for i in 0..dim.min(fr.uncertainties.len()) {
        let var = fr.uncertainties[i] * fr.uncertainties[i];
        if var.is_finite() && var > 1e-10 {
            inv_mass[i] = var;
        }
    }

    Ok((z, inv_mass))
}

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/// Run MAMS sampling on any [`LogDensityModel`].
///
/// Returns a raw [`Chain`](crate::chain::Chain) with draws in unconstrained
/// and constrained space, plus MAMS-specific diagnostics.
pub fn sample_mams<M: LogDensityModel>(
    model: &M,
    config: MamsConfig,
    seed: u64,
) -> Result<crate::chain::Chain> {
    use rand::SeedableRng;

    let posterior = Posterior::new(model);
    let dim = posterior.dim();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // ---------- Initialization ----------
    let mut inv_mass = vec![1.0; dim];
    let mut pathfinder_metric = false;

    let z_init: Vec<f64> = match config.init_strategy {
        InitStrategy::Random => {
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
            z
        }
        InitStrategy::Pathfinder => {
            match pathfinder_init_nuts(model, &posterior, dim) {
                Ok((z, mass)) => {
                    inv_mass = mass;
                    pathfinder_metric = true;
                    z
                }
                Err(_) => {
                    // Fallback to random init on Pathfinder failure
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

    // Compute initial potential + gradient
    let init_logpdf = posterior.logpdf_unconstrained(&z_init)?;
    let init_potential = if init_logpdf.is_finite() { -init_logpdf } else { 1e6 };
    let init_grad_lp = posterior.grad_unconstrained(&z_init)?;
    let init_grad: Vec<f64> = init_grad_lp.iter().map(|g| -g).collect();

    // Initialize velocity: random unit vector on S^{d-1}
    let u_init = sample_unit_vector(dim, &mut rng);

    let mut state = MicrocanonicalState {
        x: z_init,
        u: u_init,
        potential: init_potential,
        grad_potential: init_grad,
    };

    // ---------- Trajectory length L ----------
    // L is an absolute trajectory length, not relative to ε.
    // Default: π·√d — the standard quarter-period of a d-dimensional oscillator.
    let mut l = if config.init_l > 0.0 {
        config.init_l
    } else {
        std::f64::consts::PI * (dim as f64).sqrt()
    };

    // ---------- Stan-style 4-phase warmup with DualAveraging ----------
    //
    // Phase 1: Fast DA — adapt step size only.
    // Phase 2: DA + Welford — adapt step size, collect mass matrix.
    // Phase 3: DA with new metric — re-adapt step size after mass update.
    // Phase 4: Tune L + equilibrate — no adaptation.
    //
    // When Pathfinder provided a Hessian-derived inverse mass matrix, Phase 2
    // (Welford variance collection) can be shortened since the diagonal metric
    // is already a good approximation.  The freed budget goes to Phase 4
    // (equilibration), letting the chain settle with the refined metric.
    //
    //   Default:    15% / 40% / 15% / 30%
    //   Pathfinder: 10% / 15% / 10% / 65%
    //
    // DualAveraging is essential for multi-scale geometry (funnels) where the
    // optimal ε varies as the chain explores different curvature regions.
    // Binary search alone picks ε from the startup position and gets stuck.
    let (p1, p2, p3) = if pathfinder_metric { (0.10, 0.15, 0.10) } else { (0.15, 0.40, 0.15) };
    let phase1_iters = (config.n_warmup as f64 * p1) as usize;
    let phase2_iters = (config.n_warmup as f64 * p2) as usize;
    let phase3_iters = (config.n_warmup as f64 * p3) as usize;
    let phase_final_iters =
        config.n_warmup.saturating_sub(phase1_iters + phase2_iters + phase3_iters);

    let mut eps = if config.init_step_size > 0.0 {
        config.init_step_size
    } else {
        find_mams_step_size(&state, l, &posterior, &inv_mass, &mut rng)
    };

    let mut da = DualAveraging::new(config.target_accept, eps);

    // --- Phase 1: Fast DA — adapt ε only ---
    for _ in 0..phase1_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        if let Ok(r) = mams_transition(&state, eps, l, n_steps, &posterior, &inv_mass, &mut rng) {
            let ap =
                if r.energy_error.is_finite() { (-r.energy_error).exp().min(1.0) } else { 0.0 };
            da.update(ap);
            eps = da.current_step_size();
            state.x = r.x;
            state.u = r.u;
            state.potential = r.potential;
            state.grad_potential = r.grad_potential;
        }
    }

    // --- Phase 2: DA + Welford — adapt ε, collect mass matrix ---
    let mut welford = WelfordVariance::new(dim);
    for _ in 0..phase2_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        if let Ok(r) = mams_transition(&state, eps, l, n_steps, &posterior, &inv_mass, &mut rng) {
            let ap =
                if r.energy_error.is_finite() { (-r.energy_error).exp().min(1.0) } else { 0.0 };
            da.update(ap);
            eps = da.current_step_size();
            state.x = r.x;
            state.u = r.u;
            state.potential = r.potential;
            state.grad_potential = r.grad_potential;
            if config.diagonal_precond {
                welford.update(&state.x);
            }
        }
    }

    // Update inv_mass from Welford variance with Stan-style regularization.
    if config.diagonal_precond && welford.count() >= 10 {
        let var = welford.variance();
        let count = welford.count() as f64;
        let alpha = count / (count + 5.0);
        for i in 0..dim {
            inv_mass[i] = (alpha * var[i] + 1e-3 * (1.0 - alpha)).max(1e-10);
        }
        // Reset DA with new step size for new metric
        eps = find_mams_step_size(&state, l, &posterior, &inv_mass, &mut rng);
        da = DualAveraging::new(config.target_accept, eps);
    }

    // --- Phase 3: DA with new metric — re-adapt ε ---
    for _ in 0..phase3_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        if let Ok(r) = mams_transition(&state, eps, l, n_steps, &posterior, &inv_mass, &mut rng) {
            let ap =
                if r.energy_error.is_finite() { (-r.energy_error).exp().min(1.0) } else { 0.0 };
            da.update(ap);
            eps = da.current_step_size();
            state.x = r.x;
            state.u = r.u;
            state.potential = r.potential;
            state.grad_potential = r.grad_potential;
        }
    }
    eps = da.adapted_step_size();

    // --- Phase 4: Tune L + equilibrate ---
    if config.init_l <= 0.0 {
        l = tune_decoherence_length(&state, eps, &posterior, &inv_mass, &mut rng);
    }
    for _ in 0..phase_final_iters {
        let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);
        if let Ok(r) = mams_transition(&state, eps, l, n_steps, &posterior, &inv_mass, &mut rng) {
            state.x = r.x;
            state.u = r.u;
            state.potential = r.potential;
            state.grad_potential = r.grad_potential;
        }
    }

    // ---------- Sampling ----------
    let n_steps = ((l / eps).round() as usize).clamp(1, config.max_leapfrog);

    let mut draws_unconstrained = Vec::with_capacity(config.n_samples);
    let mut draws_constrained = Vec::with_capacity(config.n_samples);
    let mut divergences = Vec::with_capacity(config.n_samples);
    let mut accept_probs = Vec::with_capacity(config.n_samples);
    let mut energies = Vec::with_capacity(config.n_samples);
    let mut leapfrog_counts = Vec::with_capacity(config.n_samples);

    for _ in 0..config.n_samples {
        match mams_transition(&state, eps, l, n_steps, &posterior, &inv_mass, &mut rng) {
            Ok(r) => {
                state.x = r.x;
                state.u = r.u;
                state.potential = r.potential;
                state.grad_potential = r.grad_potential;

                let constrained =
                    posterior.to_constrained(&state.x).unwrap_or_else(|_| vec![f64::NAN; dim]);

                draws_unconstrained.push(state.x.clone());
                draws_constrained.push(constrained);
                divergences.push(false);
                accept_probs.push(if r.accepted { 1.0 } else { 0.0 });
                energies.push(state.potential);
                leapfrog_counts.push(r.n_leapfrog);
            }
            Err(_) => {
                // Keep current state, mark as divergence
                let constrained =
                    posterior.to_constrained(&state.x).unwrap_or_else(|_| vec![f64::NAN; dim]);

                draws_unconstrained.push(state.x.clone());
                draws_constrained.push(constrained);
                divergences.push(true);
                accept_probs.push(0.0);
                energies.push(state.potential);
                leapfrog_counts.push(0);
            }
        }
    }

    // mass_diag = 1/inv_mass (for reporting, matches Chain convention)
    let mass_diag: Vec<f64> = inv_mass.iter().map(|&im| 1.0 / im.max(1e-30)).collect();

    Ok(crate::chain::Chain {
        draws_unconstrained,
        draws_constrained,
        divergences,
        tree_depths: vec![0; config.n_samples], // MAMS has no tree; fill with 0
        accept_probs,
        energies,
        n_leapfrog: leapfrog_counts,
        max_treedepth: usize::MAX, // MAMS has no tree — prevent false treedepth-rate failures
        step_size: eps,
        mass_diag,
    })
}

/// Run MAMS sampling on multiple chains in parallel via Rayon.
///
/// Each chain gets seed `seed + chain_id`.
pub fn sample_mams_multichain(
    model: &impl LogDensityModel,
    n_chains: usize,
    seed: u64,
    config: MamsConfig,
) -> Result<crate::chain::SamplerResult> {
    use rayon::prelude::*;

    let n_warmup = config.n_warmup;
    let n_samples = config.n_samples;

    let chains: Vec<Result<crate::chain::Chain>> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let chain_seed = seed.wrapping_add(chain_id as u64);
            sample_mams(model, config.clone(), chain_seed)
        })
        .collect();

    let chains: Vec<crate::chain::Chain> = chains.into_iter().collect::<Result<Vec<_>>>()?;
    let param_names: Vec<String> = model.parameter_names();

    Ok(crate::chain::SamplerResult { chains, param_names, n_warmup, n_samples, diagnostics: None })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ns_core::traits::{LogDensityModel, PreparedModelRef};

    // -----------------------------------------------------------------------
    // Test models
    // -----------------------------------------------------------------------

    /// Simple 1D standard normal: -logpdf(x) = x²/2
    #[derive(Clone, Copy)]
    struct Normal1D;

    impl LogDensityModel for Normal1D {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            1
        }
        fn parameter_names(&self) -> Vec<String> {
            vec!["x".into()]
        }
        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY)]
        }
        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0]
        }
        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            Ok(0.5 * params[0] * params[0])
        }
        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            Ok(vec![params[0]])
        }
        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    /// 4D diagonal MVN: x_i ~ N(mu_i, sigma_i²)
    #[derive(Clone)]
    struct DiagMVN {
        mu: Vec<f64>,
        inv_var: Vec<f64>,
    }

    impl DiagMVN {
        fn new(mu: Vec<f64>, sigma: Vec<f64>) -> Self {
            let inv_var = sigma.iter().map(|s| 1.0 / (s * s)).collect();
            Self { mu, inv_var }
        }
    }

    impl LogDensityModel for DiagMVN {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            self.mu.len()
        }
        fn parameter_names(&self) -> Vec<String> {
            (0..self.mu.len()).map(|i| format!("x[{}]", i)).collect()
        }
        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY); self.mu.len()]
        }
        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0; self.mu.len()]
        }
        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            Ok(params
                .iter()
                .zip(self.mu.iter())
                .zip(self.inv_var.iter())
                .map(|((x, m), iv)| 0.5 * (x - m) * (x - m) * iv)
                .sum())
        }
        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            Ok(params
                .iter()
                .zip(self.mu.iter())
                .zip(self.inv_var.iter())
                .map(|((x, m), iv)| (x - m) * iv)
                .collect())
        }
        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_isokinetic_leapfrog_energy_conservation() {
        // For the isokinetic integrator on a Gaussian, the ΔK from the B-steps
        // should approximately cancel the ΔV, so total energy error is small.
        let model = DiagMVN::new(vec![0.0; 4], vec![1.0; 4]);
        let posterior = Posterior::new(&model);
        let inv_mass = vec![1.0; 4];
        let eps = 0.05;

        let lp = posterior.logpdf_unconstrained(&[0.5, -0.3, 0.8, -0.2]).unwrap();
        let g = posterior.grad_unconstrained(&[0.5, -0.3, 0.8, -0.2]).unwrap();
        let mut u = vec![0.5, 0.5, 0.5, 0.5];
        normalize(&mut u);

        let mut state = MicrocanonicalState {
            x: vec![0.5, -0.3, 0.8, -0.2],
            u,
            potential: -lp,
            grad_potential: g.iter().map(|gi| -gi).collect(),
        };

        let initial_potential = state.potential;

        // 100 leapfrog steps, accumulate ΔK
        let mut total_dk = 0.0;
        for _ in 0..100 {
            let dk = isokinetic_leapfrog_step(&mut state, eps, &posterior, &inv_mass).unwrap();
            total_dk += dk;
        }

        let delta_v = state.potential - initial_potential;
        let total_energy_error = (delta_v + total_dk).abs();

        // Symplectic integrator: energy error should be bounded, O(ε²)
        assert!(
            total_energy_error < 5.0,
            "Total energy error too large: {} (ΔV={}, ΔK={})",
            total_energy_error,
            delta_v,
            total_dk
        );

        // Velocity should still be unit
        let u_norm: f64 = state.u.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((u_norm - 1.0).abs() < 1e-10, "|u| should be 1, got {}", u_norm);
    }

    #[test]
    fn test_partial_refresh_preserves_unit_norm() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut u = vec![1.0, 0.0, 0.0, 0.0];

        for _ in 0..100 {
            partial_velocity_refresh_sphere(&mut u, 0.5, 1.0, &mut rng);
            let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "|u| should be 1, got {}", norm);
        }

        // After many refreshes, u should have moved away from initial
        assert!(u[0].abs() < 0.99, "u should have mixed away from initial direction");
    }

    #[test]
    fn test_mams_accept_reject() {
        use rand::SeedableRng;
        let model = DiagMVN::new(vec![0.0; 4], vec![1.0; 4]);
        let posterior = Posterior::new(&model);
        let inv_mass = vec![1.0; 4];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let lp = posterior.logpdf_unconstrained(&[0.5, -0.3, 0.8, -0.2]).unwrap();
        let g = posterior.grad_unconstrained(&[0.5, -0.3, 0.8, -0.2]).unwrap();
        let u = sample_unit_vector(4, &mut rng);
        let state = MicrocanonicalState {
            x: vec![0.5, -0.3, 0.8, -0.2],
            u,
            potential: -lp,
            grad_potential: g.iter().map(|gi| -gi).collect(),
        };

        // Run several transitions, check energy_error is finite
        let mut n_accepted = 0;
        let n_total = 50;
        for _ in 0..n_total {
            let r = mams_transition(&state, 0.1, 0.5, 5, &posterior, &inv_mass, &mut rng).unwrap();
            assert!(r.energy_error.is_finite() || !r.accepted);
            if r.accepted {
                n_accepted += 1;
            }
        }
        // Should have some accepts
        assert!(n_accepted > 0, "No accepts in {} transitions", n_total);
    }

    // -----------------------------------------------------------------------
    // Distributional tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mams_1d_normal() {
        let model = Normal1D;
        let config =
            MamsConfig { n_warmup: 500, n_samples: 500, target_accept: 0.9, ..Default::default() };
        let chain = sample_mams(&model, config, 42).unwrap();

        assert_eq!(chain.draws_constrained.len(), 500);

        let draws: Vec<f64> = chain.draws_constrained.iter().map(|d| d[0]).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var =
            draws.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (draws.len() as f64 - 1.0);
        let std = var.sqrt();

        assert!(mean.abs() < 0.5, "Mean should be near 0: {}", mean);
        assert!(std > 0.3 && std < 2.0, "Std should be near 1: {}", std);
    }

    #[test]
    fn test_mams_mvn_diagonal() {
        let model = DiagMVN::new(vec![1.0, -2.0, 0.5, 3.0], vec![1.0, 0.5, 2.0, 1.5]);
        let config = MamsConfig {
            n_warmup: 1000,
            n_samples: 1000,
            target_accept: 0.9,
            ..Default::default()
        };
        let chain = sample_mams(&model, config, 123).unwrap();

        assert_eq!(chain.draws_constrained.len(), 1000);

        // Check each marginal mean
        let expected_mu = [1.0, -2.0, 0.5, 3.0];
        let expected_sigma = [1.0, 0.5, 2.0, 1.5];
        for d in 0..4 {
            let draws: Vec<f64> = chain.draws_constrained.iter().map(|draw| draw[d]).collect();
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            assert!(
                (mean - expected_mu[d]).abs()
                    < 3.0 * expected_sigma[d] / (draws.len() as f64).sqrt() + 1.0,
                "Marginal mean[{}] should be near {}: {}",
                d,
                expected_mu[d],
                mean
            );
        }
    }

    #[test]
    fn test_mams_8schools() {
        use crate::diagnostics::compute_diagnostics;
        use crate::eight_schools::EightSchoolsModel;

        let y = vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
        let sigma = vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
        let model = EightSchoolsModel::new(y, sigma, 5.0, 5.0).unwrap();

        let config =
            MamsConfig { n_warmup: 1000, n_samples: 500, target_accept: 0.9, ..Default::default() };
        let result = sample_mams_multichain(&model, 2, 42, config).unwrap();

        assert_eq!(result.chains.len(), 2);
        assert_eq!(result.total_draws(), 1000);

        // Check R-hat is reasonable (< 2.5 for short runs on hierarchical model)
        let diag = compute_diagnostics(&result);
        for (i, &rhat) in diag.r_hat.iter().enumerate() {
            assert!(
                rhat < 2.5,
                "R-hat for param {} = {} (should be < 2.5)",
                result.param_names[i],
                rhat,
            );
        }
    }

    #[test]
    fn test_mams_deterministic() {
        let model = Normal1D;
        let config = MamsConfig { n_warmup: 100, n_samples: 50, ..Default::default() };
        let c1 = sample_mams(&model, config.clone(), 42).unwrap();
        let c2 = sample_mams(&model, config, 42).unwrap();

        assert_eq!(c1.draws_constrained, c2.draws_constrained, "MAMS should be deterministic");
    }

    #[test]
    fn test_mams_multichain_basic() {
        let model = Normal1D;
        let config = MamsConfig { n_warmup: 200, n_samples: 100, ..Default::default() };
        let result = sample_mams_multichain(&model, 2, 42, config).unwrap();

        assert_eq!(result.chains.len(), 2);
        assert_eq!(result.n_warmup, 200);
        assert_eq!(result.n_samples, 100);
        assert_eq!(result.total_draws(), 200);
    }

    #[test]
    fn test_mams_acceptance_rate() {
        let model = Normal1D;
        let config =
            MamsConfig { n_warmup: 300, n_samples: 200, target_accept: 0.9, ..Default::default() };
        let chain = sample_mams(&model, config, 42).unwrap();

        let accept_rate: f64 =
            chain.accept_probs.iter().sum::<f64>() / chain.accept_probs.len() as f64;
        // With tuned ε, acceptance should be reasonable (> 0.3 at minimum)
        assert!(accept_rate > 0.3, "Acceptance rate too low: {}", accept_rate);
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_mams_vs_nuts_ess -- --ignored`"]
    fn test_mams_vs_nuts_ess() {
        use crate::diagnostics::compute_diagnostics;

        let model = DiagMVN::new(vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);

        // MAMS
        let mams_config = MamsConfig { n_warmup: 1000, n_samples: 1000, ..Default::default() };
        let mams_result = sample_mams_multichain(&model, 2, 42, mams_config).unwrap();
        let mams_diag = compute_diagnostics(&mams_result);
        let mams_min_ess: f64 = mams_diag.ess_bulk.iter().cloned().fold(f64::INFINITY, f64::min);

        // NUTS
        let nuts_config =
            crate::nuts::NutsConfig { max_treedepth: 10, target_accept: 0.8, ..Default::default() };
        let nuts_result =
            crate::chain::sample_nuts_multichain(&model, 2, 1000, 1000, 42, nuts_config).unwrap();
        let nuts_diag = compute_diagnostics(&nuts_result);
        let nuts_min_ess: f64 = nuts_diag.ess_bulk.iter().cloned().fold(f64::INFINITY, f64::min);

        // MAMS ESS should be at least 70% of NUTS (sanity check)
        assert!(
            mams_min_ess > 0.7 * nuts_min_ess,
            "MAMS min ESS ({:.0}) should be at least 70% of NUTS min ESS ({:.0})",
            mams_min_ess,
            nuts_min_ess
        );
    }
}
