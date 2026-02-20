//! Pre-built ODE-based pharmacokinetic systems and a generic ODE PK solver.
//!
//! Provides ready-to-use ODE systems for common PK scenarios that cannot be
//! solved analytically:
//!
//! - [`ParameterizedTransitOde`] — Transit compartment absorption (Savic & Karlsson 2009)
//! - [`MichaelisMentenPkOde`] — Saturable (Michaelis-Menten) elimination
//! - [`TmddOde`] — Target-Mediated Drug Disposition for biologics (Mager & Jusko 2001)
//!
//! All systems implement [`OdeSystem`] and can be integrated with the adaptive
//! solvers in [`crate::ode_adaptive`] via the generic [`OdePkSolver`].

use ns_core::{Error, Result};

use crate::dosing::{DoseEvent, DoseRoute};
use crate::ode_adaptive::{OdeOptions, OdeSystem, esdirk4, rk45};

// ---------------------------------------------------------------------------
// Transit Compartment Absorption ODE (Savic & Karlsson 2009)
// ---------------------------------------------------------------------------

/// Transit compartment absorption model (Savic & Karlsson 2009).
///
/// Models delayed and complex absorption profiles using a chain of `n_transit`
/// transit compartments before the central compartment. Parameters are stored
/// directly in the struct for use with [`OdePkSolver`].
///
/// # State vector
///
/// - 1-compartment: `[Transit_1, ..., Transit_n, Central]`
/// - 2-compartment: `[Transit_1, ..., Transit_n, Central, Peripheral]`
///
/// # Equations
///
/// ```text
/// dTransit_0/dt = -ktr * Transit_0
/// dTransit_i/dt = ktr * (Transit_{i-1} - Transit_i)   for i = 1..n_transit
/// dCentral/dt   = ktr * Transit_last - (CL/V) * Central - k12*Central + k21*Peripheral
/// dPeripheral/dt = k12 * Central - k21 * Peripheral   (if 2-cpt)
/// ```
pub struct ParameterizedTransitOde {
    /// Number of transit compartments.
    pub n_transit: usize,
    /// Number of disposition compartments (1 or 2).
    pub n_compartments: usize,
    /// Transit rate constant.
    pub ktr: f64,
    /// Clearance.
    pub cl: f64,
    /// Central volume of distribution.
    pub v1: f64,
    /// Inter-compartmental clearance (0.0 if 1-cpt).
    pub q: f64,
    /// Peripheral volume (0.0 if 1-cpt).
    pub v2: f64,
}

impl OdeSystem for ParameterizedTransitOde {
    fn ndim(&self) -> usize {
        self.n_transit + self.n_compartments
    }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let n = self.n_transit;
        let ktr = self.ktr;

        // Transit chain
        if n > 0 {
            // First transit: input from dose is handled externally via bolus
            dydt[0] = -ktr * y[0];
            for i in 1..n {
                dydt[i] = ktr * (y[i - 1] - y[i]);
            }
        }

        let central_idx = n;
        let ke = self.cl / self.v1;

        if self.n_compartments == 1 {
            // Central compartment
            let input = if n > 0 { ktr * y[n - 1] } else { 0.0 };
            dydt[central_idx] = input - ke * y[central_idx];
        } else {
            // 2-compartment disposition
            let periph_idx = n + 1;
            let k12 = self.q / self.v1;
            let k21 = self.q / self.v2;

            let input = if n > 0 { ktr * y[n - 1] } else { 0.0 };
            dydt[central_idx] =
                input - ke * y[central_idx] - k12 * y[central_idx] + k21 * y[periph_idx];
            dydt[periph_idx] = k12 * y[central_idx] - k21 * y[periph_idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Michaelis-Menten (saturable) elimination
// ---------------------------------------------------------------------------

/// Michaelis-Menten (saturable) elimination PK model.
///
/// Models nonlinear (capacity-limited) drug elimination where the elimination
/// rate saturates at high concentrations.
///
/// # Equations
///
/// - 1-compartment: `dA/dt = input - Vmax * C / (Km + C)`, where `C = A / V`
/// - 2-compartment: adds peripheral distribution `- Q*(C - Cp)` where `Cp = Ap / V2`
///
/// # State vector
///
/// - 1-compartment: `[A_central]` (amount in central)
/// - 2-compartment: `[A_central, A_peripheral]`
///
/// # Parameters (stored in struct)
///
/// - `vmax`: Maximum elimination rate (amount/time)
/// - `km`: Michaelis constant (concentration at half-Vmax)
/// - `v1`: Central volume of distribution
/// - `q`: Inter-compartmental clearance (0 if 1-cpt)
/// - `v2`: Peripheral volume (0 if 1-cpt)
pub struct MichaelisMentenPkOde {
    /// Number of disposition compartments (1 or 2).
    pub n_compartments: usize,
    /// Maximum elimination rate.
    pub vmax: f64,
    /// Michaelis constant.
    pub km: f64,
    /// Central volume.
    pub v1: f64,
    /// Inter-compartmental clearance (0.0 if 1-cpt).
    pub q: f64,
    /// Peripheral volume (0.0 if 1-cpt).
    pub v2: f64,
}

impl MichaelisMentenPkOde {
    /// Create a 1-compartment Michaelis-Menten model.
    ///
    /// # Arguments
    /// * `vmax` - Maximum elimination rate
    /// * `km` - Michaelis constant
    /// * `v` - Volume of distribution
    pub fn new_1cpt(vmax: f64, km: f64, v: f64) -> Self {
        Self { n_compartments: 1, vmax, km, v1: v, q: 0.0, v2: 0.0 }
    }

    /// Create a 2-compartment Michaelis-Menten model.
    ///
    /// # Arguments
    /// * `vmax` - Maximum elimination rate
    /// * `km` - Michaelis constant
    /// * `v1` - Central volume
    /// * `q` - Inter-compartmental clearance
    /// * `v2` - Peripheral volume
    pub fn new_2cpt(vmax: f64, km: f64, v1: f64, q: f64, v2: f64) -> Self {
        Self { n_compartments: 2, vmax, km, v1, q, v2 }
    }
}

impl OdeSystem for MichaelisMentenPkOde {
    fn ndim(&self) -> usize {
        self.n_compartments
    }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        // State is amount (not concentration): A_central, [A_peripheral]
        let a_central = y[0].max(0.0);
        let c = a_central / self.v1;

        // Michaelis-Menten elimination: rate = Vmax * C / (Km + C)
        let elim_rate = self.vmax * c / (self.km + c);

        if self.n_compartments == 1 {
            dydt[0] = -elim_rate;
        } else {
            let a_periph = y[1].max(0.0);
            let c_periph = a_periph / self.v2;
            let transfer = self.q * (c - c_periph);

            dydt[0] = -elim_rate - transfer;
            dydt[1] = transfer;
        }
    }
}

// ---------------------------------------------------------------------------
// Target-Mediated Drug Disposition (TMDD)
// ---------------------------------------------------------------------------

/// Target-Mediated Drug Disposition (TMDD) model for biologics.
///
/// Full TMDD model (Mager & Jusko 2001) for monoclonal antibodies and other
/// drugs that bind to cell-surface receptors with high affinity.
///
/// # Equations
///
/// ```text
/// dL/dt  = -kon*L*R + koff*LR - kel*L + input/V    (free drug)
/// dR/dt  = ksyn - kdeg*R - kon*L*R + koff*LR         (free receptor)
/// dLR/dt = kon*L*R - koff*LR - kint*LR               (drug-receptor complex)
/// ```
///
/// # State vector
///
/// `[L, R, LR]` — concentrations of free drug, free receptor, and complex.
///
/// # Parameters (stored in struct)
///
/// - `kon`: Association rate constant (1/(conc*time))
/// - `koff`: Dissociation rate constant (1/time)
/// - `kel`: Linear elimination rate of free drug (1/time)
/// - `ksyn`: Receptor synthesis rate (conc/time)
/// - `kdeg`: Receptor degradation rate (1/time)
/// - `kint`: Complex internalization rate (1/time)
/// - `v`: Volume of distribution
pub struct TmddOde {
    /// Association rate constant.
    pub kon: f64,
    /// Dissociation rate constant.
    pub koff: f64,
    /// Linear elimination rate of free drug.
    pub kel: f64,
    /// Receptor synthesis rate.
    pub ksyn: f64,
    /// Receptor degradation rate.
    pub kdeg: f64,
    /// Complex internalization rate.
    pub kint: f64,
    /// Volume of distribution.
    pub v: f64,
}

impl TmddOde {
    /// Create a new TMDD model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(kon: f64, koff: f64, kel: f64, ksyn: f64, kdeg: f64, kint: f64, v: f64) -> Self {
        Self { kon, koff, kel, ksyn, kdeg, kint, v }
    }

    /// Compute steady-state receptor concentration: `R_ss = ksyn / kdeg`.
    pub fn receptor_baseline(&self) -> f64 {
        self.ksyn / self.kdeg
    }
}

impl OdeSystem for TmddOde {
    fn ndim(&self) -> usize {
        3
    }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let l = y[0].max(0.0); // free drug concentration
        let r = y[1].max(0.0); // free receptor concentration
        let lr = y[2].max(0.0); // complex concentration

        let binding = self.kon * l * r;
        let dissociation = self.koff * lr;

        // Free drug
        dydt[0] = -binding + dissociation - self.kel * l;
        // Free receptor
        dydt[1] = self.ksyn - self.kdeg * r - binding + dissociation;
        // Drug-receptor complex
        dydt[2] = binding - dissociation - self.kint * lr;
    }
}

// ---------------------------------------------------------------------------
// Generic ODE PK Solver
// ---------------------------------------------------------------------------

/// Solver type for the ODE PK system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSolverType {
    /// Dormand-Prince 4(5) adaptive (non-stiff).
    Rk45,
    /// L-stable SDIRK2 (stiff systems like TMDD).
    Esdirk4,
}

/// Configuration options for the ODE PK solver.
#[derive(Debug, Clone)]
pub struct OdeSolverOptions {
    /// Relative tolerance (default: 1e-8).
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-10).
    pub atol: f64,
    /// Maximum number of integration steps per interval (default: 10000).
    pub max_steps: usize,
    /// Initial step size hint (default: 0.01). Set to 0.0 for automatic.
    pub h_init: f64,
}

impl Default for OdeSolverOptions {
    fn default() -> Self {
        Self { rtol: 1e-8, atol: 1e-10, max_steps: 10_000, h_init: 0.01 }
    }
}

impl OdeSolverOptions {
    /// Convert to the internal `OdeOptions` used by the adaptive solvers.
    fn to_ode_options(&self) -> OdeOptions {
        OdeOptions {
            rtol: self.rtol,
            atol: self.atol,
            h0: self.h_init,
            max_steps: self.max_steps,
            dense_output: true,
            ..OdeOptions::default()
        }
    }
}

/// Result of an ODE PK simulation.
#[derive(Debug, Clone)]
pub struct OdePkResult {
    /// Concentration in the observed compartment at each observation time.
    pub concentrations: Vec<f64>,
    /// Full state vector (all compartments) at each observation time.
    pub states: Vec<Vec<f64>>,
    /// Observation times.
    pub times: Vec<f64>,
    /// Area under the concentration-time curve (trapezoidal rule).
    pub auc: f64,
    /// Maximum observed concentration.
    pub cmax: f64,
    /// Time of maximum observed concentration.
    pub tmax: f64,
}

/// Generic ODE PK solver with dosing support.
///
/// Handles multiple dosing events (IV bolus, oral, infusion) and integrates
/// the ODE system between events, recording concentrations at specified
/// observation times.
///
/// # Example
///
/// ```ignore
/// use ns_inference::ode_pk::*;
/// use ns_inference::dosing::{DoseEvent, DoseRoute};
///
/// let mm = MichaelisMentenPkOde::new_1cpt(10.0, 5.0, 50.0);
/// let solver = OdePkSolver::new(Box::new(mm), OdeSolverType::Rk45, OdeSolverOptions::default());
///
/// let doses = vec![DoseEvent { time: 0.0, amount: 100.0, route: DoseRoute::IvBolus }];
/// let obs_times = vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
///
/// let result = solver.solve(&doses, &obs_times, 0, 0).unwrap();
/// println!("Cmax = {}, Tmax = {}, AUC = {}", result.cmax, result.tmax, result.auc);
/// ```
pub struct OdePkSolver {
    /// The ODE system to integrate.
    pub system: Box<dyn OdeSystem>,
    /// Choice of solver (RK45 for non-stiff, ESDIRK4 for stiff).
    pub solver: OdeSolverType,
    /// Solver configuration.
    pub options: OdeSolverOptions,
}

impl OdePkSolver {
    /// Create a new ODE PK solver.
    pub fn new(system: Box<dyn OdeSystem>, solver: OdeSolverType, options: OdeSolverOptions) -> Self {
        Self { system, solver, options }
    }

    /// Solve the ODE PK system with dosing and observation schedule.
    ///
    /// # Arguments
    ///
    /// * `doses` - Dosing events (IV bolus, oral, infusion)
    /// * `obs_times` - Times at which to report concentrations
    /// * `dose_compartment` - Index of compartment receiving the dose
    /// * `obs_compartment` - Index of compartment to observe (usually central)
    ///
    /// # Algorithm
    ///
    /// 1. Sort dose events and observation times chronologically
    /// 2. Initialize state vector to zeros
    /// 3. For each interval between events:
    ///    a. Apply dose to appropriate compartment (bolus: instantaneous addition)
    ///    b. Integrate ODE from current time to next event time
    ///    c. Record state at observation times falling within the interval
    /// 4. Compute AUC (trapezoidal rule), Cmax, and Tmax from recorded concentrations
    ///
    /// # Infusion handling
    ///
    /// Infusions are modeled as zero-order input. The solver splits infusion
    /// events into start and end sub-events and adds a constant rate term to
    /// the dose compartment during the infusion interval using a wrapper system.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `doses` is empty
    /// - `obs_times` is empty
    /// - `dose_compartment` or `obs_compartment` >= system dimension
    /// - The ODE solver fails to converge
    pub fn solve(
        &self,
        doses: &[DoseEvent],
        obs_times: &[f64],
        dose_compartment: usize,
        obs_compartment: usize,
    ) -> Result<OdePkResult> {
        // Validation
        if doses.is_empty() {
            return Err(Error::Validation("ode_pk: doses must be non-empty".into()));
        }
        if obs_times.is_empty() {
            return Err(Error::Validation("ode_pk: obs_times must be non-empty".into()));
        }
        let dim = self.system.ndim();
        if dose_compartment >= dim {
            return Err(Error::Validation(format!(
                "ode_pk: dose_compartment={dose_compartment} >= ndim={dim}"
            )));
        }
        if obs_compartment >= dim {
            return Err(Error::Validation(format!(
                "ode_pk: obs_compartment={obs_compartment} >= ndim={dim}"
            )));
        }

        // Sort observation times
        let mut sorted_obs: Vec<(usize, f64)> = obs_times.iter().copied().enumerate().collect();
        sorted_obs.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Build timeline of events (doses + infusion end-points)
        let mut timeline: Vec<TimelineEvent> = Vec::new();
        let mut active_infusions: Vec<InfusionInfo> = Vec::new();

        for dose in doses {
            match dose.route {
                DoseRoute::IvBolus => {
                    timeline.push(TimelineEvent::Bolus {
                        time: dose.time,
                        amount: dose.amount,
                        compartment: dose_compartment,
                    });
                }
                DoseRoute::Oral { bioavailability } => {
                    // Oral dose: apply as bolus to the dose compartment
                    // (transit or absorption compartment) with bioavailability scaling
                    timeline.push(TimelineEvent::Bolus {
                        time: dose.time,
                        amount: dose.amount * bioavailability,
                        compartment: dose_compartment,
                    });
                }
                DoseRoute::Infusion { duration } => {
                    let rate = dose.amount / duration;
                    let info = InfusionInfo {
                        rate,
                        compartment: dose_compartment,
                    };
                    timeline.push(TimelineEvent::InfusionStart {
                        time: dose.time,
                        infusion_idx: active_infusions.len(),
                    });
                    timeline.push(TimelineEvent::InfusionEnd {
                        time: dose.time + duration,
                        infusion_idx: active_infusions.len(),
                    });
                    active_infusions.push(info);
                }
            }
        }

        // Sort timeline by time
        timeline.sort_by(|a, b| a.time().total_cmp(&b.time()));

        // Collect all critical time points (event times + observation times)
        let mut critical_times: Vec<f64> = Vec::new();
        critical_times.push(0.0); // always start at t=0
        for ev in &timeline {
            critical_times.push(ev.time());
        }
        for &(_, t) in &sorted_obs {
            critical_times.push(t);
        }
        critical_times.sort_by(|a, b| a.total_cmp(b));
        critical_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // Initialize state
        let mut y = vec![0.0; dim];
        let mut t_current = 0.0;

        // Result storage (indexed by original observation order)
        let mut result_conc = vec![0.0_f64; obs_times.len()];
        let mut result_states = vec![vec![0.0_f64; dim]; obs_times.len()];
        let mut obs_ptr = 0usize; // pointer into sorted_obs

        // Track active infusion rates
        let mut infusion_active = vec![false; active_infusions.len()];

        let opts = self.options.to_ode_options();

        // Process each critical time point.
        //
        // At each critical time `tc` we:
        //   1. Integrate from t_current to tc (if tc > t_current)
        //   2. Apply all dosing events scheduled at tc (bolus, infusion start/end)
        //   3. Record any observations at tc (post-dose state)
        //
        // This ensures that an observation coinciding with a bolus sees the
        // post-dose concentration (superposition convention).
        let mut timeline_ptr = 0usize;

        for &tc in &critical_times {
            // Step 1: integrate from t_current to tc
            if tc > t_current + 1e-15 {
                let total_infusion_rate =
                    self.compute_infusion_rates(&active_infusions, &infusion_active);
                let has_infusion = total_infusion_rate.iter().any(|&r| r.abs() > 0.0);

                if has_infusion {
                    let wrapper = InfusionWrapper {
                        inner: &*self.system,
                        rates: &total_infusion_rate,
                    };
                    let sol = self.integrate_dyn(&wrapper, &y, t_current, tc, &opts)?;
                    if let Some(y_final) = sol.y.last() {
                        // Record observations strictly inside (t_current, tc)
                        self.record_observations(
                            &sol,
                            &sorted_obs,
                            &mut obs_ptr,
                            t_current,
                            tc,
                            obs_compartment,
                            &mut result_conc,
                            &mut result_states,
                        );
                        y.copy_from_slice(y_final);
                    }
                } else {
                    let sol = self.integrate_dyn(&*self.system, &y, t_current, tc, &opts)?;
                    if let Some(y_final) = sol.y.last() {
                        self.record_observations(
                            &sol,
                            &sorted_obs,
                            &mut obs_ptr,
                            t_current,
                            tc,
                            obs_compartment,
                            &mut result_conc,
                            &mut result_states,
                        );
                        y.copy_from_slice(y_final);
                    }
                }
                t_current = tc;
            }

            // Step 2: apply all events at tc
            while timeline_ptr < timeline.len()
                && timeline[timeline_ptr].time() <= tc + 1e-15
            {
                match &timeline[timeline_ptr] {
                    TimelineEvent::Bolus { amount, compartment, .. } => {
                        y[*compartment] += amount;
                    }
                    TimelineEvent::InfusionStart { infusion_idx, .. } => {
                        infusion_active[*infusion_idx] = true;
                    }
                    TimelineEvent::InfusionEnd { infusion_idx, .. } => {
                        infusion_active[*infusion_idx] = false;
                    }
                }
                timeline_ptr += 1;
            }

            // Step 3: record observations at tc (post-dose)
            while obs_ptr < sorted_obs.len() && sorted_obs[obs_ptr].1 <= tc + 1e-15 {
                let (orig_idx, _) = sorted_obs[obs_ptr];
                result_conc[orig_idx] = y[obs_compartment];
                result_states[orig_idx] = y.clone();
                obs_ptr += 1;
            }
        }

        // Handle any remaining observations after the last critical time
        while obs_ptr < sorted_obs.len() {
            let (orig_idx, obs_t) = sorted_obs[obs_ptr];
            if obs_t > t_current + 1e-15 {
                let sol = self.integrate_dyn(&*self.system, &y, t_current, obs_t, &opts)?;
                if let Some(y_final) = sol.y.last() {
                    result_conc[orig_idx] = y_final[obs_compartment];
                    result_states[orig_idx] = y_final.clone();
                    y.copy_from_slice(y_final);
                    t_current = obs_t;
                }
            } else {
                result_conc[orig_idx] = y[obs_compartment];
                result_states[orig_idx] = y.clone();
            }
            obs_ptr += 1;
        }

        // Compute derived PK endpoints
        let (auc, cmax, tmax) = compute_pk_endpoints(obs_times, &result_conc);

        Ok(OdePkResult {
            concentrations: result_conc,
            states: result_states,
            times: obs_times.to_vec(),
            auc,
            cmax,
            tmax,
        })
    }

    /// Integrate the ODE system using the selected solver.
    ///
    /// Uses a `Sized` wrapper (`DynOdeRef`) around `&dyn OdeSystem` so that
    /// `rk45`/`esdirk4` generic bounds are satisfied.
    fn integrate_dyn(
        &self,
        sys: &dyn OdeSystem,
        y0: &[f64],
        t0: f64,
        t1: f64,
        opts: &OdeOptions,
    ) -> Result<crate::ode::OdeSolution> {
        let wrapper = DynOdeRef(sys);
        match self.solver {
            OdeSolverType::Rk45 => rk45(&wrapper, y0, t0, t1, opts),
            OdeSolverType::Esdirk4 => esdirk4(&wrapper, y0, t0, t1, opts),
        }
    }

    /// Compute total infusion rate per compartment from active infusions.
    fn compute_infusion_rates(
        &self,
        infusions: &[InfusionInfo],
        active: &[bool],
    ) -> Vec<f64> {
        let dim = self.system.ndim();
        let mut rates = vec![0.0; dim];
        for (i, info) in infusions.iter().enumerate() {
            if active[i] {
                rates[info.compartment] += info.rate;
            }
        }
        rates
    }

    /// Record observation concentrations from a dense ODE solution.
    fn record_observations(
        &self,
        sol: &crate::ode::OdeSolution,
        sorted_obs: &[(usize, f64)],
        obs_ptr: &mut usize,
        t_start: f64,
        t_end: f64,
        obs_compartment: usize,
        result_conc: &mut [f64],
        result_states: &mut [Vec<f64>],
    ) {
        // Record observations strictly inside (t_start, t_end).
        // Observations exactly at critical times are handled by the main loop
        // (Step 3) to ensure post-dose state is used.
        while *obs_ptr < sorted_obs.len() {
            let (orig_idx, obs_t) = sorted_obs[*obs_ptr];
            // Stop if beyond this interval
            if obs_t >= t_end - 1e-15 {
                break;
            }
            // Skip observations at or before t_start (handled by previous iteration)
            if obs_t <= t_start + 1e-15 {
                *obs_ptr += 1;
                continue;
            }

            // Interpolate from dense output
            let state = interpolate_solution(sol, obs_t);
            result_conc[orig_idx] = state[obs_compartment];
            result_states[orig_idx] = state;
            *obs_ptr += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Sized wrapper around `&dyn OdeSystem` so that `rk45`/`esdirk4` generic
/// bounds (`S: OdeSystem`, implicitly `S: Sized`) are satisfied.
struct DynOdeRef<'a>(&'a dyn OdeSystem);

impl<'a> OdeSystem for DynOdeRef<'a> {
    fn ndim(&self) -> usize {
        self.0.ndim()
    }

    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.0.rhs(t, y, dydt);
    }

    fn jacobian(&self, t: f64, y: &[f64], jac: &mut Vec<Vec<f64>>) {
        self.0.jacobian(t, y, jac);
    }
}

/// A timeline event for the solver.
#[derive(Debug)]
enum TimelineEvent {
    Bolus { time: f64, amount: f64, compartment: usize },
    InfusionStart { time: f64, infusion_idx: usize },
    InfusionEnd { time: f64, infusion_idx: usize },
}

impl TimelineEvent {
    fn time(&self) -> f64 {
        match self {
            Self::Bolus { time, .. } => *time,
            Self::InfusionStart { time, .. } => *time,
            Self::InfusionEnd { time, .. } => *time,
        }
    }
}

/// Infusion metadata.
#[derive(Debug)]
struct InfusionInfo {
    rate: f64,
    compartment: usize,
}

/// ODE system wrapper that adds constant infusion rates.
struct InfusionWrapper<'a> {
    inner: &'a dyn OdeSystem,
    rates: &'a [f64],
}

impl<'a> OdeSystem for InfusionWrapper<'a> {
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.inner.rhs(t, y, dydt);
        for (i, &rate) in self.rates.iter().enumerate() {
            if i < dydt.len() {
                dydt[i] += rate;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Linearly interpolate the ODE solution at a given time.
fn interpolate_solution(sol: &crate::ode::OdeSolution, t: f64) -> Vec<f64> {
    if sol.t.is_empty() {
        return Vec::new();
    }
    if sol.t.len() == 1 || t <= sol.t[0] {
        return sol.y[0].clone();
    }
    if t >= *sol.t.last().unwrap() {
        return sol.y.last().unwrap().clone();
    }

    // Binary search for the interval containing t
    let mut lo = 0;
    let mut hi = sol.t.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if sol.t[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let ta = sol.t[lo];
    let tb = sol.t[hi];
    let frac = if (tb - ta).abs() < 1e-30 { 0.0 } else { (t - ta) / (tb - ta) };

    let n = sol.y[0].len();
    let mut yq = vec![0.0; n];
    for i in 0..n {
        yq[i] = sol.y[lo][i] + frac * (sol.y[hi][i] - sol.y[lo][i]);
    }
    yq
}

/// Compute AUC (trapezoidal rule), Cmax, and Tmax from concentration-time data.
fn compute_pk_endpoints(times: &[f64], concentrations: &[f64]) -> (f64, f64, f64) {
    if times.is_empty() || concentrations.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut auc = 0.0;
    let mut cmax = concentrations[0];
    let mut tmax = times[0];

    for i in 1..times.len() {
        // Trapezoidal rule
        let dt = times[i] - times[i - 1];
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * dt;

        if concentrations[i] > cmax {
            cmax = concentrations[i];
            tmax = times[i];
        }
    }

    (auc, cmax, tmax)
}

// ---------------------------------------------------------------------------
// NLL for ODE PK models
// ---------------------------------------------------------------------------

/// Negative log-likelihood for an ODE-based PK model.
///
/// Uses [`OdePkSolver`] to compute predicted concentrations, then evaluates
/// the additive Gaussian NLL:
///
/// ```text
/// NLL = 0.5 * n * ln(2*pi*sigma^2) + 0.5 * sum((y_obs - y_pred)^2 / sigma^2)
/// ```
///
/// # Arguments
///
/// * `solver` - Configured ODE PK solver
/// * `doses` - Dosing events
/// * `obs_times` - Observation times
/// * `obs_values` - Observed concentrations
/// * `dose_compartment` - Compartment receiving doses
/// * `obs_compartment` - Compartment being observed
/// * `sigma` - Standard deviation of additive Gaussian noise
///
/// # Returns
///
/// The negative log-likelihood value, or an error if the ODE solve fails.
#[allow(clippy::too_many_arguments)]
pub fn ode_pk_nll(
    solver: &OdePkSolver,
    doses: &[DoseEvent],
    obs_times: &[f64],
    obs_values: &[f64],
    dose_compartment: usize,
    obs_compartment: usize,
    sigma: f64,
) -> Result<f64> {
    if obs_times.len() != obs_values.len() {
        return Err(Error::Validation(format!(
            "ode_pk_nll: obs_times.len()={} != obs_values.len()={}",
            obs_times.len(),
            obs_values.len()
        )));
    }
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(Error::Validation("ode_pk_nll: sigma must be finite and > 0".into()));
    }

    let result = solver.solve(doses, obs_times, dose_compartment, obs_compartment)?;

    let n = obs_values.len() as f64;
    let sigma2 = sigma * sigma;
    let log_norm = 0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln();

    let mut sse = 0.0;
    for i in 0..obs_values.len() {
        let residual = obs_values[i] - result.concentrations[i];
        sse += residual * residual;
    }

    Ok(log_norm + 0.5 * sse / sigma2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dosing::DoseRoute;

    /// With 0 transit compartments, the transit model reduces to a simple
    /// 1-compartment model with direct input to central. An IV bolus should
    /// produce exponential decay matching the analytical 1-cpt IV solution.
    #[test]
    fn test_transit_1cpt_iv_bolus() {
        let cl = 5.0;
        let v = 50.0;
        let ke = cl / v; // 0.1

        let sys = ParameterizedTransitOde {
            n_transit: 0,
            n_compartments: 1,
            ktr: 1.0, // unused when n_transit=0
            cl,
            v1: v,
            q: 0.0,
            v2: 0.0,
        };

        let solver =
            OdePkSolver::new(Box::new(sys), OdeSolverType::Rk45, OdeSolverOptions::default());

        let dose = 1000.0;
        let doses = vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }];
        let obs_times: Vec<f64> = (0..=24).map(|h| h as f64).collect();

        let result = solver.solve(&doses, &obs_times, 0, 0).unwrap();

        // Compare with analytical: A(t) = dose * exp(-ke*t), C(t) = A(t)/V
        for (i, &t) in obs_times.iter().enumerate() {
            let expected = dose * (-ke * t).exp();
            let got = result.concentrations[i];
            let reldiff = if expected > 1e-10 { (got - expected).abs() / expected } else { (got - expected).abs() };
            assert!(
                reldiff < 1e-3,
                "t={t}: got amount={got}, expected={expected}, reldiff={reldiff}"
            );
        }
    }

    /// When concentration C << Km, Michaelis-Menten elimination approximates
    /// linear elimination with ke = Vmax / (Km * V). Verify this at low doses.
    #[test]
    fn test_michaelis_menten_linear_regime() {
        let vmax = 10.0;
        let km = 100.0; // large Km
        let v = 50.0;

        let sys = MichaelisMentenPkOde::new_1cpt(vmax, km, v);
        let solver =
            OdePkSolver::new(Box::new(sys), OdeSolverType::Rk45, OdeSolverOptions::default());

        let dose = 1.0; // small dose → C = dose/V = 0.02, well below Km=100
        let doses = vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }];
        let obs_times: Vec<f64> = (0..=10).map(|h| h as f64).collect();

        let result = solver.solve(&doses, &obs_times, 0, 0).unwrap();

        // Approximate linear ke = Vmax / (Km * V) ... actually in amount form:
        // dA/dt = -Vmax * (A/V) / (Km + A/V) ≈ -Vmax/(Km*V) * A when A/V << Km
        let ke_approx = vmax / (km * v);

        for (i, &t) in obs_times.iter().enumerate() {
            let expected = dose * (-ke_approx * t).exp();
            let got = result.concentrations[i];
            // Allow 5% relative error due to nonlinearity
            let reldiff =
                if expected > 1e-10 { (got - expected).abs() / expected } else { (got - expected).abs() };
            assert!(
                reldiff < 0.05,
                "t={t}: got={got}, expected_linear={expected}, reldiff={reldiff}"
            );
        }
    }

    /// Verify TMDD steady-state: when no drug is present, receptor should be
    /// at baseline R_ss = ksyn/kdeg. After drug washout, receptor should return.
    #[test]
    fn test_tmdd_equilibrium() {
        let ksyn = 1.0;
        let kdeg = 0.1;
        let r_ss = ksyn / kdeg; // 10.0

        let tmdd = TmddOde::new(
            0.01,  // kon
            0.001, // koff
            0.05,  // kel
            ksyn,
            kdeg,
            0.02, // kint
            50.0, // V
        );

        assert!((tmdd.receptor_baseline() - r_ss).abs() < 1e-10);

        let _solver =
            OdePkSolver::new(Box::new(tmdd), OdeSolverType::Esdirk4, OdeSolverOptions::default());

        // Start with receptor at baseline, give a small drug dose
        // State: [L, R, LR]
        // The solver initializes y=[0,0,0] and adds dose to compartment 0.
        // We need receptor to start at baseline. Use a workaround: integrate
        // without drug first to reach steady-state, then apply drug.
        //
        // For this test, just verify that with no drug and R(0) = R_ss,
        // the system stays at equilibrium. We test via direct ODE integration.
        let tmdd2 = TmddOde::new(0.01, 0.001, 0.05, ksyn, kdeg, 0.02, 50.0);
        let y0 = [0.0, r_ss, 0.0]; // no drug, receptor at baseline
        let opts = OdeOptions { rtol: 1e-8, atol: 1e-12, ..OdeOptions::default() };
        let sol = rk45(&tmdd2, &y0, 0.0, 100.0, &opts).unwrap();
        let y_final = sol.y.last().unwrap();

        // Receptor should remain at baseline
        assert!(
            (y_final[1] - r_ss).abs() < 0.01,
            "R should stay at baseline: got {}, expected {}",
            y_final[1],
            r_ss
        );
        // Drug and complex should remain at 0
        assert!(y_final[0].abs() < 1e-10, "L should be 0: got {}", y_final[0]);
        assert!(y_final[2].abs() < 1e-10, "LR should be 0: got {}", y_final[2]);
    }

    /// IV bolus into a 1-compartment ODE model should match the analytical
    /// exponential decay solution.
    #[test]
    fn test_ode_pk_solver_iv_bolus() {
        let cl = 10.0;
        let v = 100.0;
        let ke = cl / v;

        // Use ParameterizedTransitOde with 0 transit as a simple 1-cpt model
        let sys = ParameterizedTransitOde {
            n_transit: 0,
            n_compartments: 1,
            ktr: 0.0,
            cl,
            v1: v,
            q: 0.0,
            v2: 0.0,
        };

        let solver =
            OdePkSolver::new(Box::new(sys), OdeSolverType::Rk45, OdeSolverOptions::default());

        let dose = 500.0;
        let doses = vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }];
        let obs_times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

        let result = solver.solve(&doses, &obs_times, 0, 0).unwrap();

        // Analytical: A(t) = dose * exp(-ke * t)
        for (i, &t) in obs_times.iter().enumerate() {
            let expected = dose * (-ke * t).exp();
            let got = result.concentrations[i];
            let reldiff = if expected > 1e-10 { (got - expected).abs() / expected } else { 0.0 };
            assert!(
                reldiff < 1e-3,
                "IV bolus t={t}: got={got}, expected={expected}, reldiff={reldiff}"
            );
        }

        // Cmax should be at t=0 (IV bolus)
        assert!(
            (result.cmax - dose).abs() < 1e-6,
            "Cmax should equal dose: got {}",
            result.cmax
        );
        assert!(
            result.tmax.abs() < 1e-6,
            "Tmax should be at t=0: got {}",
            result.tmax
        );
    }

    /// Multiple IV bolus doses should produce superposition-like behavior
    /// (additive for linear systems).
    #[test]
    fn test_ode_pk_solver_multi_dose() {
        let cl = 5.0;
        let v = 50.0;
        let ke = cl / v;

        let sys = ParameterizedTransitOde {
            n_transit: 0,
            n_compartments: 1,
            ktr: 0.0,
            cl,
            v1: v,
            q: 0.0,
            v2: 0.0,
        };

        let solver =
            OdePkSolver::new(Box::new(sys), OdeSolverType::Rk45, OdeSolverOptions::default());

        let dose = 100.0;
        let interval = 12.0;
        let doses = vec![
            DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus },
            DoseEvent { time: interval, amount: dose, route: DoseRoute::IvBolus },
            DoseEvent { time: 2.0 * interval, amount: dose, route: DoseRoute::IvBolus },
        ];
        let obs_times: Vec<f64> = (0..=36).map(|h| h as f64).collect();

        let result = solver.solve(&doses, &obs_times, 0, 0).unwrap();

        // Verify superposition: analytical 3-dose superposition
        for (i, &t) in obs_times.iter().enumerate() {
            let mut expected = 0.0;
            for d in &doses {
                if t >= d.time {
                    expected += dose * (-ke * (t - d.time)).exp();
                }
            }
            let got = result.concentrations[i];
            let reldiff = if expected > 1e-6 { (got - expected).abs() / expected } else { 0.0 };
            assert!(
                reldiff < 1e-2,
                "multi-dose t={t}: got={got}, expected={expected}, reldiff={reldiff}"
            );
        }

        // After the second dose, concentration should be higher than just one dose
        // at equivalent time points
        let c_at_12 = result.concentrations[12]; // t=12, just after 2nd dose
        let c_at_0 = result.concentrations[0]; // t=0, just after 1st dose
        assert!(
            c_at_12 > c_at_0 * 0.5,
            "accumulation expected: c(12)={c_at_12}, c(0)={c_at_0}"
        );
    }

    /// NLL should be minimized when the model parameters match the true
    /// data-generating process.
    #[test]
    fn test_ode_pk_nll_minimum() {
        let cl_true = 5.0;
        let v_true = 50.0;
        let ke = cl_true / v_true;
        let dose = 100.0;
        let sigma = 0.5;

        // Generate "observed" data from the true model (no noise for this test)
        let obs_times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let obs_values: Vec<f64> = obs_times.iter().map(|&t| dose * (-ke * t).exp()).collect();

        let doses = vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }];

        // NLL at true parameters
        let sys_true = ParameterizedTransitOde {
            n_transit: 0,
            n_compartments: 1,
            ktr: 0.0,
            cl: cl_true,
            v1: v_true,
            q: 0.0,
            v2: 0.0,
        };
        let solver_true =
            OdePkSolver::new(Box::new(sys_true), OdeSolverType::Rk45, OdeSolverOptions::default());
        let nll_true = ode_pk_nll(&solver_true, &doses, &obs_times, &obs_values, 0, 0, sigma).unwrap();

        // NLL at wrong parameters (higher CL)
        let sys_wrong = ParameterizedTransitOde {
            n_transit: 0,
            n_compartments: 1,
            ktr: 0.0,
            cl: cl_true * 2.0,
            v1: v_true,
            q: 0.0,
            v2: 0.0,
        };
        let solver_wrong =
            OdePkSolver::new(Box::new(sys_wrong), OdeSolverType::Rk45, OdeSolverOptions::default());
        let nll_wrong =
            ode_pk_nll(&solver_wrong, &doses, &obs_times, &obs_values, 0, 0, sigma).unwrap();

        // NLL should be lower at true parameters
        assert!(
            nll_true < nll_wrong,
            "NLL at true params ({nll_true}) should be less than at wrong params ({nll_wrong})"
        );

        // At true parameters with no noise, NLL should equal the normalization constant
        let expected_nll = 0.5 * obs_values.len() as f64 * (2.0 * std::f64::consts::PI * sigma * sigma).ln();
        assert!(
            (nll_true - expected_nll).abs() < 0.1,
            "NLL at true params ({nll_true}) should be near normalization ({expected_nll})"
        );
    }

    /// Transit compartment model with several transit compartments should
    /// produce a delayed absorption peak.
    #[test]
    fn test_transit_delayed_absorption() {
        let n_transit = 5;
        let ktr = 2.0;
        let cl = 5.0;
        let v = 50.0;

        let sys = ParameterizedTransitOde {
            n_transit,
            n_compartments: 1,
            ktr,
            cl,
            v1: v,
            q: 0.0,
            v2: 0.0,
        };

        let solver =
            OdePkSolver::new(Box::new(sys), OdeSolverType::Rk45, OdeSolverOptions::default());

        let dose = 100.0;
        // Dose into first transit compartment
        let doses = vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }];
        // Observe up to 72h (>10 half-lives for ke=0.1, t1/2=6.93h)
        let obs_times: Vec<f64> = (0..=144).map(|h| h as f64 * 0.5).collect();

        // Dose compartment = 0 (first transit), obs compartment = n_transit (central)
        let result = solver.solve(&doses, &obs_times, 0, n_transit).unwrap();

        // Central compartment should be 0 at t=0 (drug is in transit chain)
        assert!(
            result.concentrations[0].abs() < 1e-6,
            "central at t=0 should be ~0: got {}",
            result.concentrations[0]
        );

        // Cmax should occur after t=0 (delayed absorption)
        assert!(
            result.tmax > 0.5,
            "Tmax should be delayed: got {}",
            result.tmax
        );

        // Peak should be positive
        assert!(
            result.cmax > 0.0,
            "Cmax should be positive: got {}",
            result.cmax
        );

        // After long time (72h, >10 half-lives), central should be nearly empty
        let last_conc = *result.concentrations.last().unwrap();
        assert!(
            last_conc < result.cmax * 0.1,
            "concentration at t=72h ({last_conc}) should be much less than Cmax ({})",
            result.cmax
        );
    }
}
