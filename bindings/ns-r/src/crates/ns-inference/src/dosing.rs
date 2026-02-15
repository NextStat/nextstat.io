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
}
