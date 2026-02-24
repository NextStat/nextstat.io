//! Dosing regimen abstraction for pharmacometric models.
//!
//! Supports:
//! - **IV bolus**: instantaneous injection into the central compartment.
//! - **Oral**: first-order absorption with bioavailability.
//! - **IV infusion**: zero-order input over a specified duration.
//!
//! Concentrations are computed via superposition of analytical solutions
//! (valid for linear PK models: 1-cpt and 2-cpt).

use ns_core::{Error, Result};

/// Route of administration for a single dose event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoseRoute {
    /// Instantaneous IV bolus into central compartment.
    IvBolus,
    /// Oral dose with first-order absorption.
    Oral {
        /// Fraction absorbed (0, 1].
        bioavailability: f64,
    },
    /// Zero-order IV infusion over `duration` time units.
    Infusion {
        /// Infusion duration (> 0).
        duration: f64,
    },
}

/// A single dosing event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoseEvent {
    /// Time of dose administration.
    pub time: f64,
    /// Dose amount (> 0).
    pub amount: f64,
    /// Route of administration.
    pub route: DoseRoute,
}

/// A dosing regimen: ordered sequence of dose events.
///
/// Used with 1-compartment and 2-compartment PK models to compute
/// concentration-time profiles under multi-dose schedules.
#[derive(Debug, Clone)]
pub struct DosingRegimen {
    events: Vec<DoseEvent>,
}

/// Solve the depressed cubic x³ + ax² + bx + c = 0 via the trigonometric method.
///
/// Returns three real eigenvalues sorted descending (alpha > beta > gamma).
/// Used for 3-compartment PK models where all roots are positive and real.
fn solve_cubic_eigenvalues(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let q = (a * a - 3.0 * b) / 9.0;
    let r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;
    let q3 = q * q * q;
    let disc = r * r - q3;
    if disc < 0.0 {
        // Three real roots (typical for PK)
        let theta = (r / q3.sqrt()).clamp(-1.0, 1.0).acos();
        let sq = -2.0 * q.sqrt();
        let mut r1 = sq * (theta / 3.0).cos() - a / 3.0;
        let mut r2 = sq * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0;
        let mut r3 = sq * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0;
        // Sort descending: alpha > beta > gamma
        if r1 < r2 {
            std::mem::swap(&mut r1, &mut r2);
        }
        if r1 < r3 {
            std::mem::swap(&mut r1, &mut r3);
        }
        if r2 < r3 {
            std::mem::swap(&mut r2, &mut r3);
        }
        (r1, r2, r3)
    } else {
        // One real root + two complex (shouldn't happen for PK)
        let aa = -(r.signum()) * (r.abs() + disc.sqrt()).cbrt();
        let bb = if aa.abs() > 1e-30 { q / aa } else { 0.0 };
        let root = aa + bb - a / 3.0;
        (root, root * 0.99, root * 0.98) // fallback
    }
}

impl DosingRegimen {
    /// Create a regimen from a list of dose events.
    ///
    /// Events are sorted by time. Validates all fields.
    pub fn from_events(mut events: Vec<DoseEvent>) -> Result<Self> {
        if events.is_empty() {
            return Err(Error::Validation("dosing regimen must have ≥1 event".to_string()));
        }
        for (i, ev) in events.iter().enumerate() {
            if !ev.time.is_finite() || ev.time < 0.0 {
                return Err(Error::Validation(format!("event {i}: time must be finite and >= 0")));
            }
            if !ev.amount.is_finite() || ev.amount <= 0.0 {
                return Err(Error::Validation(format!("event {i}: amount must be finite and > 0")));
            }
            match ev.route {
                DoseRoute::IvBolus => {}
                DoseRoute::Oral { bioavailability } => {
                    if !bioavailability.is_finite()
                        || bioavailability <= 0.0
                        || bioavailability > 1.0
                    {
                        return Err(Error::Validation(format!(
                            "event {i}: bioavailability must be in (0, 1]"
                        )));
                    }
                }
                DoseRoute::Infusion { duration } => {
                    if !duration.is_finite() || duration <= 0.0 {
                        return Err(Error::Validation(format!(
                            "event {i}: infusion duration must be finite and > 0"
                        )));
                    }
                }
            }
        }
        events.sort_by(|a, b| a.time.total_cmp(&b.time));
        Ok(Self { events })
    }

    /// Single IV bolus at time 0.
    pub fn single_iv_bolus(dose: f64) -> Result<Self> {
        Self::from_events(vec![DoseEvent { time: 0.0, amount: dose, route: DoseRoute::IvBolus }])
    }

    /// Single oral dose at time 0.
    pub fn single_oral(dose: f64, bioavailability: f64) -> Result<Self> {
        Self::from_events(vec![DoseEvent {
            time: 0.0,
            amount: dose,
            route: DoseRoute::Oral { bioavailability },
        }])
    }

    /// Single IV infusion at time 0.
    pub fn single_infusion(dose: f64, duration: f64) -> Result<Self> {
        Self::from_events(vec![DoseEvent {
            time: 0.0,
            amount: dose,
            route: DoseRoute::Infusion { duration },
        }])
    }

    /// Repeated doses at regular intervals.
    pub fn repeated(dose: f64, interval: f64, n_doses: usize, route: DoseRoute) -> Result<Self> {
        if n_doses == 0 {
            return Err(Error::Validation("n_doses must be > 0".to_string()));
        }
        if !interval.is_finite() || interval <= 0.0 {
            return Err(Error::Validation("interval must be finite and > 0".to_string()));
        }
        let events: Vec<DoseEvent> = (0..n_doses)
            .map(|i| DoseEvent { time: i as f64 * interval, amount: dose, route })
            .collect();
        Self::from_events(events)
    }

    /// Access the dose events (sorted by time).
    pub fn events(&self) -> &[DoseEvent] {
        &self.events
    }

    /// Number of dose events.
    pub fn n_doses(&self) -> usize {
        self.events.len()
    }

    /// Total amount administered.
    pub fn total_amount(&self) -> f64 {
        self.events.iter().map(|e| e.amount).sum()
    }

    /// Concentration at time `t` for a 1-compartment model via superposition.
    ///
    /// Parameters: `cl` (clearance), `v` (volume), `ka` (absorption rate for oral).
    /// For IV bolus: `C(t) = (D/V) · exp(−ke·Δt)`, `ke = cl/v`.
    /// For oral: uses the 1-cpt oral analytical solution.
    /// For infusion: uses the 1-cpt IV infusion analytical solution.
    pub fn conc_1cpt(&self, cl: f64, v: f64, ka: f64, t: f64) -> f64 {
        let ke = cl / v;
        let mut c = 0.0;
        for ev in &self.events {
            let dt = t - ev.time;
            if dt < 0.0 {
                continue;
            }
            match ev.route {
                DoseRoute::IvBolus => {
                    c += (ev.amount / v) * (-ke * dt).exp();
                }
                DoseRoute::Oral { bioavailability } => {
                    if (ka - ke).abs() < 1e-12 {
                        let d_amt = ev.amount * bioavailability;
                        c += (d_amt * ka / v) * dt * (-ke * dt).exp();
                    } else {
                        let d_amt = ev.amount * bioavailability;
                        let frac = ka / (ka - ke);
                        c += (d_amt / v) * frac * ((-ke * dt).exp() - (-ka * dt).exp());
                    }
                }
                DoseRoute::Infusion { duration } => {
                    let rate = ev.amount / duration;
                    if dt <= duration {
                        c += (rate / cl) * (1.0 - (-ke * dt).exp());
                    } else {
                        let c_end = (rate / cl) * (1.0 - (-ke * duration).exp());
                        c += c_end * (-ke * (dt - duration)).exp();
                    }
                }
            }
        }
        c
    }

    /// Concentration at time `t` for a 2-compartment IV model via superposition.
    ///
    /// Parameters: `cl`, `v1`, `v2`, `q`.
    /// Only IV bolus and infusion routes are valid (oral routes are ignored with a warning-free skip).
    pub fn conc_2cpt_iv(&self, cl: f64, v1: f64, v2: f64, q: f64, t: f64) -> f64 {
        let k10 = cl / v1;
        let k12 = q / v1;
        let k21 = q / v2;
        let sum = k10 + k12 + k21;
        let prod = k10 * k21;
        let disc = (sum * sum - 4.0 * prod).max(0.0);
        let sqrt_d = disc.sqrt();
        let alpha = 0.5 * (sum + sqrt_d);
        let beta = 0.5 * (sum - sqrt_d);
        let ab = alpha - beta;

        let mut c = 0.0;
        for ev in &self.events {
            let dt = t - ev.time;
            if dt < 0.0 {
                continue;
            }
            match ev.route {
                DoseRoute::IvBolus => {
                    if ab.abs() < 1e-12 {
                        let k = 0.5 * (alpha + beta);
                        c += (ev.amount / v1) * (-k * dt).exp();
                    } else {
                        let ca = (alpha - k21) / ab;
                        let cb = (k21 - beta) / ab;
                        c +=
                            (ev.amount / v1) * (ca * (-alpha * dt).exp() + cb * (-beta * dt).exp());
                    }
                }
                DoseRoute::Infusion { duration } => {
                    let rate = ev.amount / duration;
                    if ab.abs() < 1e-12 {
                        let k = 0.5 * (alpha + beta);
                        let inv_k = 1.0 / k;
                        if dt <= duration {
                            c += (rate / v1) * inv_k * (1.0 - (-k * dt).exp());
                        } else {
                            let c_end = (rate / v1) * inv_k * (1.0 - (-k * duration).exp());
                            c += c_end * (-k * (dt - duration)).exp();
                        }
                    } else {
                        let ca = (alpha - k21) / ab;
                        let cb = (k21 - beta) / ab;
                        let inv_a = 1.0 / alpha;
                        let inv_b = 1.0 / beta;
                        if dt <= duration {
                            let ia = ca * inv_a * (1.0 - (-alpha * dt).exp());
                            let ib = cb * inv_b * (1.0 - (-beta * dt).exp());
                            c += (rate / v1) * (ia + ib);
                        } else {
                            let ia_end = ca * inv_a * (1.0 - (-alpha * duration).exp());
                            let ib_end = cb * inv_b * (1.0 - (-beta * duration).exp());
                            let tail_a = (-alpha * (dt - duration)).exp();
                            let tail_b = (-beta * (dt - duration)).exp();
                            c += (rate / v1) * (ia_end * tail_a + ib_end * tail_b);
                        }
                    }
                }
                DoseRoute::Oral { .. } => {
                    // Oral route not valid for IV-only 2-cpt model; skip silently.
                }
            }
        }
        c
    }

    /// Concentration at time `t` for a 2-compartment oral model via superposition.
    ///
    /// Parameters: `cl`, `v1`, `v2`, `q`, `ka`.
    /// Supports all three dose routes.
    pub fn conc_2cpt_oral(&self, cl: f64, v1: f64, v2: f64, q: f64, ka: f64, t: f64) -> f64 {
        let k10 = cl / v1;
        let k12 = q / v1;
        let k21 = q / v2;
        let sum = k10 + k12 + k21;
        let prod = k10 * k21;
        let disc = (sum * sum - 4.0 * prod).max(0.0);
        let sqrt_d = disc.sqrt();
        let alpha = 0.5 * (sum + sqrt_d);
        let beta = 0.5 * (sum - sqrt_d);
        let ab = alpha - beta;

        let mut c = 0.0;
        for ev in &self.events {
            let dt = t - ev.time;
            if dt < 0.0 {
                continue;
            }
            match ev.route {
                DoseRoute::IvBolus => {
                    if ab.abs() < 1e-12 {
                        let k = 0.5 * (alpha + beta);
                        c += (ev.amount / v1) * (-k * dt).exp();
                    } else {
                        let ca = (alpha - k21) / ab;
                        let cb = (k21 - beta) / ab;
                        c +=
                            (ev.amount / v1) * (ca * (-alpha * dt).exp() + cb * (-beta * dt).exp());
                    }
                }
                DoseRoute::Oral { bioavailability } => {
                    let pref = ka * bioavailability * ev.amount / v1;
                    let denom_a = (ka - alpha) * (beta - alpha);
                    let denom_b = (ka - beta) * (alpha - beta);
                    let denom_c = (alpha - ka) * (beta - ka);

                    let (da, db, dc, ka_eff) = if denom_a.abs() < 1e-12
                        || denom_b.abs() < 1e-12
                        || denom_c.abs() < 1e-12
                    {
                        let ka_p = ka * (1.0 + 1e-8);
                        (
                            (ka_p - alpha) * (beta - alpha),
                            (ka_p - beta) * (alpha - beta),
                            (alpha - ka_p) * (beta - ka_p),
                            ka_p,
                        )
                    } else {
                        (denom_a, denom_b, denom_c, ka)
                    };
                    let ta = (k21 - alpha) / da * (-alpha * dt).exp();
                    let tb = (k21 - beta) / db * (-beta * dt).exp();
                    let tc = (k21 - ka_eff) / dc * (-ka_eff * dt).exp();
                    c += pref * (ta + tb + tc);
                }
                DoseRoute::Infusion { duration } => {
                    let rate = ev.amount / duration;
                    if ab.abs() < 1e-12 {
                        let k = 0.5 * (alpha + beta);
                        let inv_k = 1.0 / k;
                        if dt <= duration {
                            c += (rate / v1) * inv_k * (1.0 - (-k * dt).exp());
                        } else {
                            let c_end = (rate / v1) * inv_k * (1.0 - (-k * duration).exp());
                            c += c_end * (-k * (dt - duration)).exp();
                        }
                    } else {
                        let ca = (alpha - k21) / ab;
                        let cb = (k21 - beta) / ab;
                        let inv_a = 1.0 / alpha;
                        let inv_b = 1.0 / beta;
                        if dt <= duration {
                            let ia = ca * inv_a * (1.0 - (-alpha * dt).exp());
                            let ib = cb * inv_b * (1.0 - (-beta * dt).exp());
                            c += (rate / v1) * (ia + ib);
                        } else {
                            let ia_end = ca * inv_a * (1.0 - (-alpha * duration).exp());
                            let ib_end = cb * inv_b * (1.0 - (-beta * duration).exp());
                            let tail_a = (-alpha * (dt - duration)).exp();
                            let tail_b = (-beta * (dt - duration)).exp();
                            c += (rate / v1) * (ia_end * tail_a + ib_end * tail_b);
                        }
                    }
                }
            }
        }
        c
    }

    /// Concentration at time `t` for a 3-compartment IV model via superposition.
    ///
    /// Parameters: `cl`, `v1`, `v2`, `q2`, `v3`, `q3`.
    /// Only IV bolus and infusion routes are valid (oral routes are skipped).
    pub fn conc_3cpt_iv(
        &self,
        cl: f64,
        v1: f64,
        v2: f64,
        q2: f64,
        v3: f64,
        q3: f64,
        t: f64,
    ) -> f64 {
        let k10 = cl / v1;
        let k12 = q2 / v1;
        let k21 = q2 / v2;
        let k13 = q3 / v1;
        let k31 = q3 / v3;

        // Characteristic cubic: x³ + a*x² + b*x + c = 0
        let a = -(k10 + k12 + k21 + k13 + k31);
        let b = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
        let c_coeff = -(k10 * k21 * k31);

        let (alpha, beta, gamma) = solve_cubic_eigenvalues(a, b, c_coeff);

        let mut c = 0.0;
        for ev in &self.events {
            let dt = t - ev.time;
            if dt < 0.0 {
                continue;
            }
            match ev.route {
                DoseRoute::IvBolus => {
                    // C(t) = (D/V1) * [A*exp(-alpha*t) + B*exp(-beta*t) + G*exp(-gamma*t)]
                    let mut dab = (alpha - beta) * (alpha - gamma);
                    let mut dbb = (beta - alpha) * (beta - gamma);
                    let mut dgb = (gamma - alpha) * (gamma - beta);
                    // Handle degenerate eigenvalues
                    if dab.abs() < 1e-12 {
                        dab = if dab >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dbb.abs() < 1e-12 {
                        dbb = if dbb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dgb.abs() < 1e-12 {
                        dgb = if dgb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    let coeff_a = (alpha - k21) * (alpha - k31) / dab;
                    let coeff_b = (beta - k21) * (beta - k31) / dbb;
                    let coeff_g = (gamma - k21) * (gamma - k31) / dgb;
                    c += (ev.amount / v1)
                        * (coeff_a * (-alpha * dt).exp()
                            + coeff_b * (-beta * dt).exp()
                            + coeff_g * (-gamma * dt).exp());
                }
                DoseRoute::Infusion { duration } => {
                    let rate = ev.amount / duration;
                    let mut dab = (alpha - beta) * (alpha - gamma);
                    let mut dbb = (beta - alpha) * (beta - gamma);
                    let mut dgb = (gamma - alpha) * (gamma - beta);
                    if dab.abs() < 1e-12 {
                        dab = if dab >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dbb.abs() < 1e-12 {
                        dbb = if dbb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dgb.abs() < 1e-12 {
                        dgb = if dgb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    let coeff_a = (alpha - k21) * (alpha - k31) / dab;
                    let coeff_b = (beta - k21) * (beta - k31) / dbb;
                    let coeff_g = (gamma - k21) * (gamma - k31) / dgb;
                    let inv_a = 1.0 / alpha;
                    let inv_b = 1.0 / beta;
                    let inv_g = 1.0 / gamma;

                    if dt <= duration {
                        let ia = coeff_a * inv_a * (1.0 - (-alpha * dt).exp());
                        let ib = coeff_b * inv_b * (1.0 - (-beta * dt).exp());
                        let ig = coeff_g * inv_g * (1.0 - (-gamma * dt).exp());
                        c += (rate / v1) * (ia + ib + ig);
                    } else {
                        let ia_end = coeff_a * inv_a * (1.0 - (-alpha * duration).exp());
                        let ib_end = coeff_b * inv_b * (1.0 - (-beta * duration).exp());
                        let ig_end = coeff_g * inv_g * (1.0 - (-gamma * duration).exp());
                        let tail_a = (-alpha * (dt - duration)).exp();
                        let tail_b = (-beta * (dt - duration)).exp();
                        let tail_g = (-gamma * (dt - duration)).exp();
                        c += (rate / v1) * (ia_end * tail_a + ib_end * tail_b + ig_end * tail_g);
                    }
                }
                DoseRoute::Oral { .. } => {
                    // Oral route not valid for IV-only 3-cpt model; skip silently.
                }
            }
        }
        c
    }

    /// Concentration at time `t` for a 3-compartment oral model via superposition.
    ///
    /// Parameters: `cl`, `v1`, `v2`, `q2`, `v3`, `q3`, `ka`.
    /// Supports all three dose routes.
    pub fn conc_3cpt_oral(
        &self,
        cl: f64,
        v1: f64,
        v2: f64,
        q2: f64,
        v3: f64,
        q3: f64,
        ka: f64,
        t: f64,
    ) -> f64 {
        let k10 = cl / v1;
        let k12 = q2 / v1;
        let k21 = q2 / v2;
        let k13 = q3 / v1;
        let k31 = q3 / v3;

        // Characteristic cubic: x³ + a*x² + b*x + c = 0
        let a = -(k10 + k12 + k21 + k13 + k31);
        let b = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
        let c_coeff = -(k10 * k21 * k31);

        let (alpha, beta, gamma) = solve_cubic_eigenvalues(a, b, c_coeff);

        let mut c = 0.0;
        for ev in &self.events {
            let dt = t - ev.time;
            if dt < 0.0 {
                continue;
            }
            match ev.route {
                DoseRoute::IvBolus => {
                    // Same as conc_3cpt_iv bolus path
                    let mut dab = (alpha - beta) * (alpha - gamma);
                    let mut dbb = (beta - alpha) * (beta - gamma);
                    let mut dgb = (gamma - alpha) * (gamma - beta);
                    if dab.abs() < 1e-12 {
                        dab = if dab >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dbb.abs() < 1e-12 {
                        dbb = if dbb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dgb.abs() < 1e-12 {
                        dgb = if dgb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    let coeff_a = (alpha - k21) * (alpha - k31) / dab;
                    let coeff_b = (beta - k21) * (beta - k31) / dbb;
                    let coeff_g = (gamma - k21) * (gamma - k31) / dgb;
                    c += (ev.amount / v1)
                        * (coeff_a * (-alpha * dt).exp()
                            + coeff_b * (-beta * dt).exp()
                            + coeff_g * (-gamma * dt).exp());
                }
                DoseRoute::Oral { bioavailability } => {
                    // 4-exponential: ka adds a 4th rate constant
                    // C(t) = (ka*F*D/V1) * sum_i [ (ei-k21)*(ei-k31) / prod_{j!=i}(ei-ej) * exp(-ei*t) ]
                    // where eigenvalues are {alpha, beta, gamma, ka}
                    let pref = ka * bioavailability * ev.amount / v1;

                    // Use ka or perturbed ka to avoid degenerate denominators
                    let eigenvals = [alpha, beta, gamma];
                    let mut ka_eff = ka;
                    for &ev_val in &eigenvals {
                        if (ka_eff - ev_val).abs() < 1e-8 {
                            ka_eff = ev_val * (1.0 + 1e-8);
                        }
                    }

                    // Four terms: residue at s=-ei for (s+k21)(s+k31)/[(s+a)(s+b)(s+g)(s+ka)]
                    // Residue_i = (k21-ei)(k31-ei) / prod_{j!=i}(ej-ei)
                    let rates = [alpha, beta, gamma, ka_eff];
                    let mut contrib = 0.0;
                    for i in 0..4 {
                        let ei = rates[i];
                        let num = (k21 - ei) * (k31 - ei);
                        let mut denom = 1.0;
                        for j in 0..4 {
                            if j != i {
                                let d = rates[j] - ei;
                                if d.abs() < 1e-12 {
                                    denom *= if d >= 0.0 { 1e-12 } else { -1e-12 };
                                } else {
                                    denom *= d;
                                }
                            }
                        }
                        contrib += (num / denom) * (-ei * dt).exp();
                    }
                    c += pref * contrib;
                }
                DoseRoute::Infusion { duration } => {
                    // Same as conc_3cpt_iv infusion path
                    let rate = ev.amount / duration;
                    let mut dab = (alpha - beta) * (alpha - gamma);
                    let mut dbb = (beta - alpha) * (beta - gamma);
                    let mut dgb = (gamma - alpha) * (gamma - beta);
                    if dab.abs() < 1e-12 {
                        dab = if dab >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dbb.abs() < 1e-12 {
                        dbb = if dbb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    if dgb.abs() < 1e-12 {
                        dgb = if dgb >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    let coeff_a = (alpha - k21) * (alpha - k31) / dab;
                    let coeff_b = (beta - k21) * (beta - k31) / dbb;
                    let coeff_g = (gamma - k21) * (gamma - k31) / dgb;
                    let inv_a = 1.0 / alpha;
                    let inv_b = 1.0 / beta;
                    let inv_g = 1.0 / gamma;

                    if dt <= duration {
                        let ia = coeff_a * inv_a * (1.0 - (-alpha * dt).exp());
                        let ib = coeff_b * inv_b * (1.0 - (-beta * dt).exp());
                        let ig = coeff_g * inv_g * (1.0 - (-gamma * dt).exp());
                        c += (rate / v1) * (ia + ib + ig);
                    } else {
                        let ia_end = coeff_a * inv_a * (1.0 - (-alpha * duration).exp());
                        let ib_end = coeff_b * inv_b * (1.0 - (-beta * duration).exp());
                        let ig_end = coeff_g * inv_g * (1.0 - (-gamma * duration).exp());
                        let tail_a = (-alpha * (dt - duration)).exp();
                        let tail_b = (-beta * (dt - duration)).exp();
                        let tail_g = (-gamma * (dt - duration)).exp();
                        c += (rate / v1) * (ia_end * tail_a + ib_end * tail_b + ig_end * tail_g);
                    }
                }
            }
        }
        c
    }

    /// Compute concentration-time profile at given observation times.
    ///
    /// Uses 1-compartment oral model via superposition.
    pub fn predict_1cpt(&self, cl: f64, v: f64, ka: f64, times: &[f64]) -> Vec<f64> {
        times.iter().map(|&t| self.conc_1cpt(cl, v, ka, t)).collect()
    }

    /// Compute concentration-time profile at given observation times.
    ///
    /// Uses 2-compartment IV model via superposition.
    pub fn predict_2cpt_iv(&self, cl: f64, v1: f64, v2: f64, q: f64, times: &[f64]) -> Vec<f64> {
        times.iter().map(|&t| self.conc_2cpt_iv(cl, v1, v2, q, t)).collect()
    }

    /// Compute concentration-time profile at given observation times.
    ///
    /// Uses 2-compartment oral model via superposition.
    pub fn predict_2cpt_oral(
        &self,
        cl: f64,
        v1: f64,
        v2: f64,
        q: f64,
        ka: f64,
        times: &[f64],
    ) -> Vec<f64> {
        times.iter().map(|&t| self.conc_2cpt_oral(cl, v1, v2, q, ka, t)).collect()
    }

    /// Compute concentration-time profile at given observation times.
    ///
    /// Uses 3-compartment IV model via superposition.
    pub fn predict_3cpt_iv(
        &self,
        cl: f64,
        v1: f64,
        v2: f64,
        q2: f64,
        v3: f64,
        q3: f64,
        times: &[f64],
    ) -> Vec<f64> {
        times.iter().map(|&t| self.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, t)).collect()
    }

    /// Compute concentration-time profile at given observation times.
    ///
    /// Uses 3-compartment oral model via superposition.
    pub fn predict_3cpt_oral(
        &self,
        cl: f64,
        v1: f64,
        v2: f64,
        q2: f64,
        v3: f64,
        q3: f64,
        ka: f64,
        times: &[f64],
    ) -> Vec<f64> {
        times.iter().map(|&t| self.conc_3cpt_oral(cl, v1, v2, q2, v3, q3, ka, t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_iv_bolus_basic() {
        let reg = DosingRegimen::single_iv_bolus(100.0).unwrap();
        assert_eq!(reg.n_doses(), 1);
        assert!((reg.total_amount() - 100.0).abs() < 1e-12);

        let cl = 1.0;
        let v = 10.0;
        let ke = cl / v;
        let c0 = reg.conc_1cpt(cl, v, 1.0, 0.0);
        assert!((c0 - 100.0 / v).abs() < 1e-10, "C(0) = D/V for IV bolus");

        let c1 = reg.conc_1cpt(cl, v, 1.0, 1.0);
        let expected = (100.0 / v) * (-ke * 1.0_f64).exp();
        assert!((c1 - expected).abs() < 1e-10);
    }

    #[test]
    fn single_oral_matches_pk_model() {
        let dose = 100.0;
        let bioav = 0.9;
        let cl = 1.2;
        let v = 15.0;
        let ka = 2.0;
        let ke = cl / v;

        let reg = DosingRegimen::single_oral(dose, bioav).unwrap();

        for t in [0.5, 1.0, 2.0, 4.0, 8.0] {
            let c_reg = reg.conc_1cpt(cl, v, ka, t);
            let d_amt = dose * bioav;
            let frac = ka / (ka - ke);
            let c_ref = (d_amt / v) * frac * ((-ke * t).exp() - (-ka * t).exp());
            assert!((c_reg - c_ref).abs() < 1e-10, "t={t}: regimen={c_reg}, ref={c_ref}");
        }
    }

    #[test]
    fn repeated_oral_superposition() {
        let dose = 100.0;
        let bioav = 1.0;
        let interval = 12.0;
        let n_doses = 5;
        let cl = 1.0;
        let v = 10.0;
        let ka = 2.0;

        let reg = DosingRegimen::repeated(
            dose,
            interval,
            n_doses,
            DoseRoute::Oral { bioavailability: bioav },
        )
        .unwrap();
        assert_eq!(reg.n_doses(), 5);
        assert!((reg.total_amount() - 500.0).abs() < 1e-12);

        let t = 48.5;
        let c = reg.conc_1cpt(cl, v, ka, t);

        let mut c_manual = 0.0;
        let ke = cl / v;
        for i in 0..n_doses {
            let dt = t - i as f64 * interval;
            if dt > 0.0 {
                let frac = ka / (ka - ke);
                c_manual += (dose * bioav / v) * frac * ((-ke * dt).exp() - (-ka * dt).exp());
            }
        }
        assert!((c - c_manual).abs() < 1e-10, "superposition mismatch: reg={c}, manual={c_manual}");
    }

    #[test]
    fn iv_infusion_basic() {
        let dose = 100.0;
        let dur = 2.0;
        let reg = DosingRegimen::single_infusion(dose, dur).unwrap();

        let cl = 1.0;
        let v = 10.0;
        let ke = cl / v;
        let rate = dose / dur;

        let c_during = reg.conc_1cpt(cl, v, 1.0, 1.0);
        let expected = (rate / cl) * (1.0 - (-ke * 1.0_f64).exp());
        assert!((c_during - expected).abs() < 1e-10, "during infusion: {c_during} vs {expected}");

        let c_after = reg.conc_1cpt(cl, v, 1.0, 5.0);
        let c_end = (rate / cl) * (1.0 - (-ke * dur).exp());
        let expected_after = c_end * (-ke * 3.0_f64).exp();
        assert!(
            (c_after - expected_after).abs() < 1e-10,
            "after infusion: {c_after} vs {expected_after}"
        );
    }

    #[test]
    fn mixed_regimen() {
        let events = vec![
            DoseEvent { time: 0.0, amount: 500.0, route: DoseRoute::Infusion { duration: 1.0 } },
            DoseEvent {
                time: 12.0,
                amount: 200.0,
                route: DoseRoute::Oral { bioavailability: 0.8 },
            },
            DoseEvent {
                time: 24.0,
                amount: 200.0,
                route: DoseRoute::Oral { bioavailability: 0.8 },
            },
        ];
        let reg = DosingRegimen::from_events(events).unwrap();
        assert_eq!(reg.n_doses(), 3);
        assert!((reg.total_amount() - 900.0).abs() < 1e-12);

        let c = reg.conc_1cpt(1.0, 10.0, 2.0, 36.0);
        assert!(c > 0.0, "concentration should be positive after all doses");
        assert!(c.is_finite());
    }

    #[test]
    fn validation_rejects_bad_events() {
        assert!(DosingRegimen::from_events(vec![]).is_err());
        assert!(
            DosingRegimen::from_events(vec![DoseEvent {
                time: -1.0,
                amount: 100.0,
                route: DoseRoute::IvBolus,
            }])
            .is_err()
        );
        assert!(
            DosingRegimen::from_events(vec![DoseEvent {
                time: 0.0,
                amount: -100.0,
                route: DoseRoute::IvBolus,
            }])
            .is_err()
        );
        assert!(
            DosingRegimen::from_events(vec![DoseEvent {
                time: 0.0,
                amount: 100.0,
                route: DoseRoute::Oral { bioavailability: 1.5 },
            }])
            .is_err()
        );
        assert!(
            DosingRegimen::from_events(vec![DoseEvent {
                time: 0.0,
                amount: 100.0,
                route: DoseRoute::Infusion { duration: 0.0 },
            }])
            .is_err()
        );
    }

    #[test]
    fn predict_1cpt_multi_time() {
        let reg = DosingRegimen::single_iv_bolus(100.0).unwrap();
        let times = vec![0.0, 1.0, 2.0, 4.0];
        let concs = reg.predict_1cpt(1.0, 10.0, 1.0, &times);
        assert_eq!(concs.len(), 4);
        assert!((concs[0] - 10.0).abs() < 1e-10);
        assert!(concs[1] < concs[0]);
        assert!(concs[2] < concs[1]);
        assert!(concs[3] < concs[2]);
    }

    #[test]
    fn two_cpt_iv_superposition() {
        let dose = 100.0;
        let reg = DosingRegimen::repeated(dose, 12.0, 3, DoseRoute::IvBolus).unwrap();

        let cl = 1.0;
        let v1 = 10.0;
        let v2 = 20.0;
        let q = 0.5;

        let c0 = reg.conc_2cpt_iv(cl, v1, v2, q, 0.0);
        assert!((c0 - dose / v1).abs() < 1e-10, "C(0) = D/V1");

        let c_mid = reg.conc_2cpt_iv(cl, v1, v2, q, 12.0);
        assert!(c_mid > reg.conc_2cpt_iv(cl, v1, v2, q, 11.99));

        let concs = reg.predict_2cpt_iv(cl, v1, v2, q, &[0.0, 6.0, 12.0, 18.0, 24.0, 30.0]);
        assert_eq!(concs.len(), 6);
        assert!(concs.iter().all(|c| c.is_finite() && *c > 0.0));
    }

    #[test]
    fn two_cpt_oral_superposition() {
        let reg = DosingRegimen::repeated(100.0, 12.0, 3, DoseRoute::Oral { bioavailability: 1.0 })
            .unwrap();

        let concs = reg.predict_2cpt_oral(1.0, 10.0, 20.0, 0.5, 2.0, &[0.0, 1.0, 12.0, 25.0, 48.0]);
        assert_eq!(concs.len(), 5);
        assert!(concs[0].abs() < 1e-10, "oral C(0) ≈ 0");
        assert!(concs[1] > 0.0);
        assert!(concs.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn two_cpt_iv_infusion() {
        let reg = DosingRegimen::single_infusion(100.0, 2.0).unwrap();
        let cl = 1.0;
        let v1 = 10.0;
        let v2 = 20.0;
        let q = 0.5;

        let c_start = reg.conc_2cpt_iv(cl, v1, v2, q, 0.0);
        assert!(c_start.abs() < 1e-10, "C(0) = 0 for infusion");

        let c_during = reg.conc_2cpt_iv(cl, v1, v2, q, 1.0);
        assert!(c_during > 0.0, "concentration rises during infusion");

        let c_end = reg.conc_2cpt_iv(cl, v1, v2, q, 2.0);
        let c_after = reg.conc_2cpt_iv(cl, v1, v2, q, 10.0);
        assert!(c_after < c_end, "concentration decays after infusion ends");
    }

    #[test]
    fn events_sorted_by_time() {
        let events = vec![
            DoseEvent { time: 24.0, amount: 100.0, route: DoseRoute::IvBolus },
            DoseEvent { time: 0.0, amount: 200.0, route: DoseRoute::IvBolus },
            DoseEvent { time: 12.0, amount: 150.0, route: DoseRoute::IvBolus },
        ];
        let reg = DosingRegimen::from_events(events).unwrap();
        let times: Vec<f64> = reg.events().iter().map(|e| e.time).collect();
        assert_eq!(times, vec![0.0, 12.0, 24.0]);
    }

    #[test]
    fn three_cpt_iv_superposition() {
        let dose = 100.0;
        let reg = DosingRegimen::repeated(dose, 12.0, 3, DoseRoute::IvBolus).unwrap();

        let cl = 1.0;
        let v1 = 10.0;
        let v2 = 20.0;
        let q2 = 0.5;
        let v3 = 30.0;
        let q3 = 0.3;

        // C(0) = D/V1 for first IV bolus
        let c0 = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 0.0);
        assert!((c0 - dose / v1).abs() < 1e-8, "C(0) = D/V1: got {c0}, expected {}", dose / v1);

        // Monotone decay between doses (no new dose added in [0.1, 11.9])
        let mut prev = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 0.1);
        for i in 1..20 {
            let ti = 0.1 + i as f64 * 0.5;
            if ti >= 12.0 {
                break;
            }
            let ci = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, ti);
            assert!(ci <= prev + 1e-12, "non-monotone decay at t={ti}: prev={prev}, curr={ci}");
            prev = ci;
        }

        // After second dose, concentration jumps
        let c_before_2 = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 11.99);
        let c_after_2 = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 12.0);
        assert!(c_after_2 > c_before_2, "concentration jumps at second dose");

        // predict_3cpt_iv returns correct length with finite positive values
        let concs =
            reg.predict_3cpt_iv(cl, v1, v2, q2, v3, q3, &[0.0, 6.0, 12.0, 18.0, 24.0, 30.0]);
        assert_eq!(concs.len(), 6);
        assert!(concs.iter().all(|c| c.is_finite() && *c > 0.0));
    }

    #[test]
    fn three_cpt_oral_superposition() {
        let dose = 100.0;
        let reg = DosingRegimen::repeated(dose, 12.0, 3, DoseRoute::Oral { bioavailability: 1.0 })
            .unwrap();

        let cl = 1.0;
        let v1 = 10.0;
        let v2 = 20.0;
        let q2 = 0.5;
        let v3 = 30.0;
        let q3 = 0.3;
        let ka = 2.0;

        // Oral C(0) ≈ 0
        let c0 = reg.conc_3cpt_oral(cl, v1, v2, q2, v3, q3, ka, 0.0);
        assert!(c0.abs() < 1e-10, "oral C(0) ≈ 0: got {c0}");

        // All concentrations positive at later times
        let concs = reg.predict_3cpt_oral(
            cl,
            v1,
            v2,
            q2,
            v3,
            q3,
            ka,
            &[0.0, 0.5, 1.0, 6.0, 12.0, 25.0, 48.0],
        );
        assert_eq!(concs.len(), 7);
        assert!(concs[0].abs() < 1e-10, "oral C(0) ≈ 0");
        assert!(concs[1] > 0.0, "absorption produces positive concentration");
        assert!(concs.iter().all(|c| c.is_finite()));
        assert!(concs.iter().skip(1).all(|c| *c > 0.0));
    }

    #[test]
    fn three_cpt_iv_infusion() {
        let reg = DosingRegimen::single_infusion(100.0, 2.0).unwrap();
        let cl = 1.0;
        let v1 = 10.0;
        let v2 = 20.0;
        let q2 = 0.5;
        let v3 = 30.0;
        let q3 = 0.3;

        // C(0) = 0 for infusion
        let c_start = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 0.0);
        assert!(c_start.abs() < 1e-10, "C(0) = 0 for infusion: got {c_start}");

        // Concentration rises during infusion
        let c_during = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 1.0);
        assert!(c_during > 0.0, "concentration rises during infusion: got {c_during}");

        // Concentration decays after infusion ends
        let c_end = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 2.0);
        let c_after = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 10.0);
        assert!(
            c_after < c_end,
            "concentration decays after infusion: c_end={c_end}, c_after={c_after}"
        );

        // Monotone rise during infusion
        let c_half = reg.conc_3cpt_iv(cl, v1, v2, q2, v3, q3, 0.5);
        assert!(
            c_during > c_half,
            "monotone rise during infusion: c(0.5)={c_half}, c(1.0)={c_during}"
        );
    }
}
