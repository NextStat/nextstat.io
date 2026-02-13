//! Competing risks analysis: cumulative incidence and Fine–Gray regression (Phase 9 Cross-Vertical).
//!
//! In standard survival analysis, subjects experience a single event type. With
//! **competing risks**, multiple event types can occur and experiencing one
//! precludes observation of the others.
//!
//! ## Methods
//!
//! - **Cumulative Incidence Function (CIF)**: non-parametric estimator of the
//!   probability of experiencing a specific event type by time *t*, accounting
//!   for competing events (Aalen–Johansen estimator for the two-state case).
//!
//! - **Fine–Gray regression**: semi-parametric model for the **subdistribution
//!   hazard** of a specific event type. Analogous to Cox PH but on the
//!   subdistribution hazard scale, enabling direct modelling of CIF.
//!
//! ## Vertical applications
//!
//! - **Pharma / clinical trials**: death from disease vs death from other causes
//! - **Insurance**: lapse vs death vs disability
//! - **Epidemiology**: cause-specific mortality
//!
//! ## References
//!
//! - Fine JP, Gray RJ (1999). A proportional hazards model for the subdistribution
//!   of a competing risk. *JASA* 94(446):496–509.
//! - Gray RJ (1988). A class of K-sample tests for comparing the cumulative
//!   incidence of a competing risk. *Ann Statist* 16(3):1141–1154.

use ns_core::{Error, Result};

// ---------------------------------------------------------------------------
// Cumulative Incidence Function (CIF)
// ---------------------------------------------------------------------------

/// A single step in the cumulative incidence curve.
#[derive(Debug, Clone)]
pub struct CifStep {
    /// Event time.
    pub time: f64,
    /// Cumulative incidence at this time (probability of event by time t).
    pub cif: f64,
    /// Pointwise standard error (Aalen–Johansen / delta method).
    pub se: f64,
    /// Lower confidence bound.
    pub ci_lower: f64,
    /// Upper confidence bound.
    pub ci_upper: f64,
}

/// Result of a cumulative incidence estimation.
#[derive(Debug, Clone)]
pub struct CifEstimate {
    /// The event type (cause) for which CIF was computed.
    pub cause: u32,
    /// Steps of the CIF curve (one per distinct event time of this cause).
    pub steps: Vec<CifStep>,
    /// Number at risk at start.
    pub n: usize,
    /// Number of events of the target cause.
    pub n_events: usize,
}

/// Compute the cumulative incidence function (CIF) for a specific cause.
///
/// This is the Aalen–Johansen estimator for the sub-density of cause `target_cause`.
///
/// # Arguments
/// - `times`: observed times (event or censoring), must be >= 0.
/// - `events`: event indicator per subject. `0` = censored, positive integer = cause.
/// - `target_cause`: the cause for which to compute CIF.
/// - `conf_level`: confidence level (e.g. 0.95).
///
/// # Returns
/// A `CifEstimate` with CIF steps at each distinct event time.
pub fn cumulative_incidence(
    times: &[f64],
    events: &[u32],
    target_cause: u32,
    conf_level: f64,
) -> Result<CifEstimate> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".to_string()));
    }
    if events.len() != n {
        return Err(Error::Validation(format!(
            "times/events length mismatch: {} vs {}",
            n,
            events.len()
        )));
    }
    if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(Error::Validation("times must be finite and >= 0".to_string()));
    }
    if target_cause == 0 {
        return Err(Error::Validation("target_cause must be > 0 (0 = censored)".to_string()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }

    // Sort by time ascending; within ties: events before censored.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a].total_cmp(&times[b]).then_with(|| {
            // Events first (events[i] > 0 sorts before events[i] == 0).
            let ea = if events[a] > 0 { 0u8 } else { 1u8 };
            let eb = if events[b] > 0 { 0u8 } else { 1u8 };
            ea.cmp(&eb)
        })
    });

    let z_alpha = normal_quantile(1.0 - (1.0 - conf_level) / 2.0);

    // Distinct event times for the target cause.
    let mut steps = Vec::new();
    let mut at_risk = n;
    let mut km_surv = 1.0_f64; // overall Kaplan-Meier survival at t-
    let mut cif = 0.0_f64;
    let mut var_cif = 0.0_f64;
    let mut n_events_target = 0usize;

    // Accumulator for the variance terms (sum of d_k / (n_k * (n_k - d_k))).
    let mut sum_hazard_var = 0.0_f64;

    let mut i = 0;
    while i < n {
        let idx = order[i];
        let t = times[idx];

        // Count events at this time for each cause + censored.
        let mut d_target = 0usize;
        let mut d_any = 0usize;
        let mut c = 0usize;
        let mut j = i;
        while j < n && times[order[j]] == t {
            if events[order[j]] == 0 {
                c += 1;
            } else {
                d_any += 1;
                if events[order[j]] == target_cause {
                    d_target += 1;
                }
            }
            j += 1;
        }

        let nk = at_risk as f64;

        if d_target > 0 {
            // CIF increment: S(t-) * (d_target / n_k)
            let h_target = d_target as f64 / nk;
            let delta_cif = km_surv * h_target;
            cif += delta_cif;

            // Variance via Aalen (1978) / Choudhury (2002) delta-method.
            // Var(F_k(t)) ≈ Σ over event times ≤ t of terms involving:
            //   - overall hazard contribution to KM variance
            //   - cause-specific hazard contribution
            let d_total = d_any as f64;
            if nk > d_total && nk > 0.0 {
                sum_hazard_var += d_total / (nk * (nk - d_total));
            }
            var_cif =
                cif * cif * sum_hazard_var + km_surv * km_surv * h_target * (1.0 - h_target) / nk;
            n_events_target += d_target;
        }

        // Update overall KM survival (all-cause hazard).
        if d_any > 0 && nk > 0.0 {
            km_surv *= 1.0 - (d_any as f64) / nk;
            if d_target == 0 && nk > d_any as f64 {
                sum_hazard_var += (d_any as f64) / (nk * (nk - d_any as f64));
            }
        }

        if d_target > 0 {
            let se = var_cif.sqrt();
            let lo = (cif - z_alpha * se).max(0.0);
            let hi = (cif + z_alpha * se).min(1.0);
            steps.push(CifStep { time: t, cif, se, ci_lower: lo, ci_upper: hi });
        }

        at_risk -= d_any + c;
        i = j;
    }

    Ok(CifEstimate { cause: target_cause, steps, n, n_events: n_events_target })
}

// ---------------------------------------------------------------------------
// Gray's test for comparing CIF across groups
// ---------------------------------------------------------------------------

/// Result of Gray's K-sample test comparing cumulative incidence across groups.
#[derive(Debug, Clone)]
pub struct GrayTestResult {
    /// Test statistic (chi-squared distributed under H₀).
    pub statistic: f64,
    /// Degrees of freedom (K - 1 for K groups).
    pub df: usize,
    /// p-value (upper tail of chi-squared).
    pub p_value: f64,
}

/// Perform Gray's test comparing CIF of `target_cause` across groups.
///
/// This is the competing-risks analogue of the log-rank test.
///
/// # Arguments
/// - `times`: observed times.
/// - `events`: event codes (0 = censored, positive = cause).
/// - `groups`: group indicator per subject (0-indexed).
/// - `target_cause`: the cause to compare.
pub fn gray_test(
    times: &[f64],
    events: &[u32],
    groups: &[usize],
    target_cause: u32,
) -> Result<GrayTestResult> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".to_string()));
    }
    if events.len() != n || groups.len() != n {
        return Err(Error::Validation("times/events/groups length mismatch".to_string()));
    }
    if target_cause == 0 {
        return Err(Error::Validation("target_cause must be > 0".to_string()));
    }

    let n_groups = groups.iter().max().map(|m| m + 1).unwrap_or(0);
    if n_groups < 2 {
        return Err(Error::Validation("need at least 2 groups".to_string()));
    }

    // Sort by time ascending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    // Weighted score approach (simplified): use the Aalen–Johansen CIF-based
    // weights. For a two-group comparison, this reduces to a weighted sum of
    // (O_k - E_k) per group.
    //
    // We implement the direct counting formulation: for each distinct event
    // time, compute observed and expected events of `target_cause` per group.

    let mut obs = vec![0.0_f64; n_groups];
    let mut exp = vec![0.0_f64; n_groups];
    let mut var_mat = vec![0.0_f64; n_groups * n_groups]; // row-major variance matrix

    // Track at-risk per group.
    let mut at_risk = vec![0usize; n_groups];
    for &g in groups {
        at_risk[g] += 1;
    }

    let mut i = 0;
    while i < n {
        let t = times[order[i]];

        let mut d_target_g = vec![0usize; n_groups];
        let mut d_any_g = vec![0usize; n_groups];
        let mut c_g = vec![0usize; n_groups];
        let mut j = i;
        while j < n && times[order[j]] == t {
            let idx = order[j];
            let g = groups[idx];
            if events[idx] == 0 {
                c_g[g] += 1;
            } else {
                d_any_g[g] += 1;
                if events[idx] == target_cause {
                    d_target_g[g] += 1;
                }
            }
            j += 1;
        }

        let total_at_risk: usize = at_risk.iter().sum();
        let d_target_total: usize = d_target_g.iter().sum();

        if d_target_total > 0 && total_at_risk > 0 {
            let n_total = total_at_risk as f64;
            let d_total = d_target_total as f64;
            let p = d_total / n_total;

            for g in 0..n_groups {
                let nk = at_risk[g] as f64;
                let expected = nk * p;
                obs[g] += d_target_g[g] as f64;
                exp[g] += expected;

                // Variance contribution (hypergeometric-like).
                for g2 in 0..n_groups {
                    let nk2 = at_risk[g2] as f64;
                    let cov = if g == g2 {
                        nk * p * (1.0 - p) * (n_total - nk) / (n_total - 1.0).max(1.0)
                    } else {
                        -nk * nk2 * p * (1.0 - p) / (n_total - 1.0).max(1.0)
                    };
                    var_mat[g * n_groups + g2] += cov;
                }
            }
        }

        // Remove from at-risk.
        for g in 0..n_groups {
            at_risk[g] -= d_any_g[g] + c_g[g];
        }
        i = j;
    }

    // Chi-squared statistic: (O - E)' * V^{-1} * (O - E) using first (K-1) groups.
    let df = n_groups - 1;
    let stat = if df == 1 {
        // Scalar case: chi2 = (O_0 - E_0)^2 / V_00
        let u = obs[0] - exp[0];
        let v = var_mat[0];
        if v > 0.0 { u * u / v } else { 0.0 }
    } else {
        // General case: invert (K-1) x (K-1) sub-matrix via Cholesky.
        let mut u = Vec::with_capacity(df);
        for g in 0..df {
            u.push(obs[g] - exp[g]);
        }
        let mut v_sub = vec![0.0_f64; df * df];
        for r in 0..df {
            for c in 0..df {
                v_sub[r * df + c] = var_mat[r * n_groups + c];
            }
        }
        quad_form_inv(&u, &v_sub, df)
    };

    let p_value = chi2_sf(stat, df);

    Ok(GrayTestResult { statistic: stat, df, p_value })
}

// ---------------------------------------------------------------------------
// Fine-Gray subdistribution hazard model
// ---------------------------------------------------------------------------

/// Result of a Fine-Gray regression fit.
#[derive(Debug, Clone)]
pub struct FineGrayResult {
    /// Coefficient estimates (one per covariate).
    pub coefficients: Vec<f64>,
    /// Standard errors of coefficients.
    pub se: Vec<f64>,
    /// z-statistics (coef / se).
    pub z: Vec<f64>,
    /// Two-sided p-values.
    pub p_values: Vec<f64>,
    /// Number of subjects.
    pub n: usize,
    /// Number of events of the target cause.
    pub n_events: usize,
    /// Partial log-likelihood at convergence.
    pub log_likelihood: f64,
}

/// Fit a Fine-Gray subdistribution hazard model.
///
/// The Fine-Gray model modifies the Cox PH risk set: subjects who experience a
/// **competing** event remain in the risk set (with decreasing weight from the
/// Kaplan-Meier censoring distribution), so the model targets the CIF directly.
///
/// # Arguments
/// - `times`: observed times (event or censoring).
/// - `events`: event codes (0 = censored, positive integer = cause).
/// - `x`: covariate matrix (one row per subject, each row has `p` elements).
/// - `target_cause`: the cause to model.
///
/// # Returns
/// `FineGrayResult` with coefficient estimates, SE, z, p-values.
pub fn fine_gray_fit(
    times: &[f64],
    events: &[u32],
    x: &[Vec<f64>],
    target_cause: u32,
) -> Result<FineGrayResult> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".to_string()));
    }
    if events.len() != n || x.len() != n {
        return Err(Error::Validation("times/events/x length mismatch".to_string()));
    }
    if target_cause == 0 {
        return Err(Error::Validation("target_cause must be > 0".to_string()));
    }
    let p = x.first().map(|r| r.len()).unwrap_or(0);
    if p == 0 {
        return Err(Error::Validation("covariates must have at least 1 column".to_string()));
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(Error::Validation(format!(
                "x row {} has {} columns, expected {}",
                i,
                row.len(),
                p
            )));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation(format!("x row {} has non-finite values", i)));
        }
    }
    if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(Error::Validation("times must be finite and >= 0".to_string()));
    }

    // Pack into flat arrays and center covariates.
    let mut x_flat = Vec::with_capacity(n * p);
    for row in x {
        x_flat.extend_from_slice(row);
    }
    let mut means = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p {
            means[j] += x_flat[i * p + j];
        }
    }
    let n_f = n as f64;
    for j in 0..p {
        means[j] /= n_f;
    }
    for i in 0..n {
        for j in 0..p {
            x_flat[i * p + j] -= means[j];
        }
    }

    // Sort by time ascending; within ties: target events first, then competing, then censored.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a].total_cmp(&times[b]).then_with(|| {
            let priority = |idx: usize| -> u8 {
                if events[idx] == target_cause {
                    0
                } else if events[idx] > 0 {
                    1
                } else {
                    2
                }
            };
            priority(a).cmp(&priority(b))
        })
    });

    // Compute Kaplan-Meier of the censoring distribution G(t) = P(C > t).
    // This weights competing-event subjects in the Fine-Gray risk set.
    let censor_km = compute_censoring_km(times, events, &order, n);

    // Count target events.
    let n_events = events.iter().filter(|&&e| e == target_cause).count();
    if n_events == 0 {
        return Err(Error::Validation("no events of target_cause observed".to_string()));
    }

    // Newton-Raphson for the weighted partial likelihood.
    let max_iter = 50;
    let tol = 1e-9;
    let mut beta = vec![0.0_f64; p];

    let mut log_lik = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        let (ll, grad, mut hess) =
            fg_partial_loglik(&order, times, events, &x_flat, p, &beta, target_cause, &censor_km);

        // Add ridge regularization to improve numerical stability.
        for j in 0..p {
            let diag = hess[j * p + j].abs();
            if diag < 1e-10 {
                hess[j * p + j] -= 1e-6;
            }
        }

        // Solve: H * delta = -grad  →  delta = -H^{-1} * grad.
        let delta = solve_sym_linear(&hess, &grad, p);

        // Check for NaN/Inf in delta and clamp step size.
        let mut max_step = 0.0_f64;
        for d in &delta {
            if !d.is_finite() {
                break;
            }
            max_step = max_step.max(d.abs());
        }
        let scale = if max_step > 5.0 { 5.0 / max_step } else { 1.0 };

        let mut any_nan = false;
        for j in 0..p {
            if !delta[j].is_finite() {
                any_nan = true;
                break;
            }
            beta[j] += scale * delta[j];
            // Cap coefficient magnitude to prevent divergence on separated data.
            beta[j] = beta[j].clamp(-20.0, 20.0);
        }
        if any_nan {
            break;
        }

        let change = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max) * scale;
        log_lik = ll;
        if change < tol {
            break;
        }
    }

    // Standard errors from observed information matrix: SE = sqrt(diag(-H^{-1})).
    let (_ll_final, _grad_final, mut hess_final) =
        fg_partial_loglik(&order, times, events, &x_flat, p, &beta, target_cause, &censor_km);
    // Ridge for SE computation.
    for j in 0..p {
        let diag = hess_final[j * p + j].abs();
        if diag < 1e-10 {
            hess_final[j * p + j] -= 1e-6;
        }
    }
    let inv_hess = invert_sym(&hess_final, p);
    let mut se = vec![0.0_f64; p];
    let mut z_vals = vec![0.0_f64; p];
    let mut p_values = vec![0.0_f64; p];
    for j in 0..p {
        let v = (-inv_hess[j * p + j]).max(0.0);
        se[j] = v.sqrt();
        if se[j] > 0.0 {
            z_vals[j] = beta[j] / se[j];
            p_values[j] = 2.0 * normal_sf(z_vals[j].abs());
        } else {
            z_vals[j] = 0.0;
            p_values[j] = 1.0;
        }
    }

    Ok(FineGrayResult {
        coefficients: beta,
        se,
        z: z_vals,
        p_values,
        n,
        n_events,
        log_likelihood: log_lik,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the Kaplan-Meier estimator of the censoring distribution G(t) = P(C > t).
/// Returns G(t) for each subject in the sorted order.
fn compute_censoring_km(times: &[f64], events: &[u32], order: &[usize], n: usize) -> Vec<f64> {
    // For the censoring distribution, "event" = censored (events[i] == 0),
    // "censored" = any event (events[i] > 0). This is the reverse.
    let mut g = vec![1.0_f64; n];
    let mut at_risk = n;
    let mut km = 1.0_f64;
    let mut i = 0;
    while i < n {
        let t = times[order[i]];
        let mut n_censored = 0usize; // "events" for the censoring KM
        let mut n_events = 0usize; // "censored" for the censoring KM
        let mut j = i;
        while j < n && times[order[j]] == t {
            if events[order[j]] == 0 {
                n_censored += 1;
            } else {
                n_events += 1;
            }
            j += 1;
        }
        // G(t-) for all subjects at this time.
        for k in i..j {
            g[k] = km;
        }
        // Update: KM step for the censoring distribution.
        if n_censored > 0 && at_risk > 0 {
            km *= 1.0 - (n_censored as f64 / at_risk as f64);
        }
        at_risk -= n_censored + n_events;
        i = j;
    }
    g
}

/// Compute Fine-Gray weighted partial log-likelihood, gradient, and Hessian.
///
/// In the Fine-Gray model, the risk set at time t includes:
/// 1. Subjects still at risk (not yet had any event or been censored).
/// 2. Subjects who had a competing event before t, with weight G(t)/G(T_i)
///    where T_i is their competing event time and G is the censoring KM.
fn fg_partial_loglik(
    order: &[usize],
    times: &[f64],
    events: &[u32],
    x_flat: &[f64],
    p: usize,
    beta: &[f64],
    target_cause: u32,
    censor_km: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let n = order.len();

    // Compute exp(x_i' * beta) for all subjects.
    let mut exp_xb = vec![0.0_f64; n];
    for i in 0..n {
        let idx = order[i];
        let xi = &x_flat[idx * p..(idx + 1) * p];
        let mut xb = 0.0;
        for j in 0..p {
            xb += xi[j] * beta[j];
        }
        exp_xb[i] = ns_prob::math::exp_clamped(xb);
    }

    // Compute the current censoring KM value at each event time for
    // re-weighting competing-event subjects.
    //
    // For efficiency, we sweep backwards (risk set accumulation).
    // But for Fine-Gray, we sweep forward and accumulate the risk set.
    //
    // At each event time t_k of the target cause:
    //   Risk set R(t_k) = {i: T_i >= t_k} ∪ {i: T_i < t_k and event_i = competing}
    //   with weight w_i = G(t_k) / G(T_i) for the competing-event subjects.
    //
    // For simplicity and correctness, we use a direct computation approach.

    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; p];
    let mut hess = vec![0.0_f64; p * p];

    // Identify event times of the target cause.
    let mut event_positions = Vec::new();
    for i in 0..n {
        if events[order[i]] == target_cause {
            event_positions.push(i);
        }
    }

    for &ev_pos in &event_positions {
        let ev_idx = order[ev_pos];
        let t_event = times[ev_idx];
        let g_at_event = censor_km[ev_pos]; // G(t_event-)

        // Build weighted risk set: sum of w_i * exp(x_i' * beta) and weighted x_i.
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut s2 = vec![0.0_f64; p * p];

        for i in 0..n {
            let idx = order[i];
            let ti = times[idx];

            let w = if ti >= t_event {
                // Still at risk at t_event (regardless of event type).
                1.0
            } else if events[idx] > 0 && events[idx] != target_cause {
                // Competing event before t_event: IPCW weight.
                let gi = censor_km[i]; // G(T_i-)
                if gi > 1e-15 { g_at_event / gi } else { 0.0 }
            } else {
                // Censored before t_event or target event before t_event: not in risk set.
                0.0
            };

            if w <= 0.0 {
                continue;
            }

            let w_exp = w * exp_xb[i];
            let xi = &x_flat[idx * p..(idx + 1) * p];

            s0 += w_exp;
            for j in 0..p {
                s1[j] += w_exp * xi[j];
            }
            for j in 0..p {
                for k in 0..p {
                    s2[j * p + k] += w_exp * xi[j] * xi[k];
                }
            }
        }

        if s0 <= 0.0 {
            continue;
        }

        // Partial log-likelihood contribution.
        let xi_ev = &x_flat[ev_idx * p..(ev_idx + 1) * p];
        for j in 0..p {
            ll += xi_ev[j] * beta[j];
        }
        ll -= s0.ln();

        let inv_s0 = 1.0 / s0;
        for j in 0..p {
            grad[j] += xi_ev[j] - s1[j] * inv_s0;
        }

        let inv_s0_sq = inv_s0 * inv_s0;
        for j in 0..p {
            for k in 0..p {
                hess[j * p + k] -= s2[j * p + k] * inv_s0 - s1[j] * s1[k] * inv_s0_sq;
            }
        }
    }

    // Negate gradient for the convention grad = ∂(-ll)/∂β (gradient of NLL).
    let neg_grad: Vec<f64> = grad.iter().map(|v| -v).collect();
    // Hessian is already ∂²(-ll)/∂β² (negative observed information).
    (ll, neg_grad, hess)
}

/// Solve symmetric linear system A*x = b using Cholesky-like approach.
/// Falls back to diagonal if factorization fails.
fn solve_sym_linear(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    // Simple implementation: LDL decomposition for small p.
    // For p typically < 20 in survival regression, this is fine.
    let inv = invert_sym(a, p);
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        for j in 0..p {
            x[i] += inv[i * p + j] * b[j];
        }
    }
    x
}

/// Invert a symmetric matrix via Gauss-Jordan elimination.
fn invert_sym(a: &[f64], p: usize) -> Vec<f64> {
    let mut aug = vec![0.0_f64; p * 2 * p];
    for i in 0..p {
        for j in 0..p {
            aug[i * 2 * p + j] = a[i * p + j];
        }
        aug[i * 2 * p + p + i] = 1.0;
    }
    let w = 2 * p;
    for col in 0..p {
        // Partial pivot.
        let mut max_row = col;
        let mut max_val = aug[col * w + col].abs();
        for row in (col + 1)..p {
            let v = aug[row * w + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            // Near-singular: add ridge.
            aug[col * w + col] += 1e-6;
        }
        if max_row != col {
            for k in 0..w {
                aug.swap(col * w + k, max_row * w + k);
            }
        }
        let pivot = aug[col * w + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        let inv_pivot = 1.0 / pivot;
        for k in 0..w {
            aug[col * w + k] *= inv_pivot;
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            for k in 0..w {
                aug[row * w + k] -= factor * aug[col * w + k];
            }
        }
    }
    let mut result = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            result[i * p + j] = aug[i * w + p + j];
        }
    }
    result
}

/// Compute u' * V^{-1} * u where V is p×p symmetric.
fn quad_form_inv(u: &[f64], v: &[f64], p: usize) -> f64 {
    let inv = invert_sym(v, p);
    let mut result = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            result += u[i] * inv[i * p + j] * u[j];
        }
    }
    result.max(0.0)
}

/// Standard normal quantile (inverse CDF) via rational approximation (Abramowitz & Stegun).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let (sign, pp) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - pp).ln()).sqrt();

    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    let num = C0 + t * (C1 + t * C2);
    let den = 1.0 + t * (D1 + t * (D2 + t * D3));
    sign * (t - num / den)
}

/// Standard normal survival function P(Z > z).
fn normal_sf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
}

/// Upper tail of chi-squared distribution P(X > x) with `df` degrees of freedom.
fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    // P(X > x) = 1 - gamma_lr(df/2, x/2) where gamma_lr is the regularized lower incomplete gamma.
    let a = df as f64 / 2.0;
    let b = x / 2.0;
    1.0 - statrs::function::gamma::gamma_lr(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cif_basic() {
        // 10 subjects, 2 causes.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![1, 2, 0, 1, 0, 2, 1, 0, 0, 1];
        let cif = cumulative_incidence(&times, &events, 1, 0.95).unwrap();
        assert_eq!(cif.cause, 1);
        assert_eq!(cif.n, 10);
        assert_eq!(cif.n_events, 4);
        assert!(!cif.steps.is_empty());
        // CIF should be monotonically increasing.
        for w in cif.steps.windows(2) {
            assert!(w[1].cif >= w[0].cif);
        }
        // CIF should be in [0, 1].
        for s in &cif.steps {
            assert!(s.cif >= 0.0 && s.cif <= 1.0);
            assert!(s.se >= 0.0);
            assert!(s.ci_lower <= s.cif);
            assert!(s.ci_upper >= s.cif);
        }
    }

    #[test]
    fn cif_single_cause_matches_km() {
        // With only one cause (no competing events), CIF should equal 1 - KM.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![1, 0, 1, 0, 1];
        let cif = cumulative_incidence(&times, &events, 1, 0.95).unwrap();
        assert_eq!(cif.n_events, 3);
        // At t=1: n=5, CIF = 1/5 = 0.2, KM = 4/5
        assert!((cif.steps[0].cif - 0.2).abs() < 1e-10);
        // At t=2: censored, at_risk 5->4 (event at 1) -> 3 (censored at 2).
        // At t=3: n_at_risk=3, km_surv=4/5, CIF += (4/5)*(1/3) ≈ 0.4667
        let expected_cif2 = 0.2 + (4.0 / 5.0) * (1.0 / 3.0);
        assert!(
            (cif.steps[1].cif - expected_cif2).abs() < 1e-10,
            "CIF at t=3: got {}, expected {}",
            cif.steps[1].cif,
            expected_cif2
        );
    }

    #[test]
    fn cif_all_censored() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![0, 0, 0];
        let cif = cumulative_incidence(&times, &events, 1, 0.95).unwrap();
        assert_eq!(cif.n_events, 0);
        assert!(cif.steps.is_empty());
    }

    #[test]
    fn cif_validation_errors() {
        assert!(cumulative_incidence(&[], &[], 1, 0.95).is_err());
        assert!(cumulative_incidence(&[1.0], &[1, 2], 1, 0.95).is_err());
        assert!(cumulative_incidence(&[1.0], &[1], 0, 0.95).is_err());
        assert!(cumulative_incidence(&[-1.0], &[1], 1, 0.95).is_err());
        assert!(cumulative_incidence(&[1.0], &[1], 1, 1.5).is_err());
    }

    #[test]
    fn gray_test_basic() {
        // Two groups, cause 1 events.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let events = vec![1, 0, 2, 1, 0, 1, 1, 1, 0, 2, 1, 0];
        let groups = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let result = gray_test(&times, &events, &groups, 1).unwrap();
        assert_eq!(result.df, 1);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn gray_test_validation_errors() {
        assert!(gray_test(&[], &[], &[], 1).is_err());
        assert!(gray_test(&[1.0], &[1], &[0], 0).is_err());
        // Only one group.
        assert!(gray_test(&[1.0, 2.0], &[1, 1], &[0, 0], 1).is_err());
    }

    #[test]
    fn fine_gray_smoke() {
        // Larger dataset with 2 causes and 1 continuous covariate for numerical stability.
        let n = 40;
        let times: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5).collect();
        let events: Vec<u32> = (0..n)
            .map(|i| match i % 5 {
                0 | 2 => 1, // target
                1 => 2,     // competing
                _ => 0,     // censored
            })
            .collect();
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64) / (n as f64) - 0.5]).collect();
        let result = fine_gray_fit(&times, &events, &x, 1).unwrap();
        assert_eq!(result.n, n);
        assert!(result.n_events > 0);
        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(result.se.len(), 1);
        assert!(result.se[0] > 0.0, "SE should be > 0, got {}", result.se[0]);
        assert!(result.log_likelihood.is_finite());
        assert!(result.coefficients[0].is_finite(), "coef should be finite");
    }

    #[test]
    fn fine_gray_validation_errors() {
        assert!(fine_gray_fit(&[], &[], &[], 1).is_err());
        assert!(fine_gray_fit(&[1.0], &[1], &[vec![1.0]], 0).is_err());
        // No target events.
        assert!(fine_gray_fit(&[1.0, 2.0], &[2, 0], &[vec![1.0], vec![2.0]], 1).is_err());
    }

    #[test]
    fn fine_gray_no_competing_matches_sign() {
        // With only 1 cause (no competing events), Fine-Gray ≈ Cox PH.
        // Use continuous covariate = event time + noise to get moderate, finite MLE.
        // Higher covariate → later event → lower hazard → negative coefficient.
        let n = 40;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let t = 1.0 + (i as f64) * 0.3;
            times.push(t);
            // ~60% event rate, interleaved.
            events.push(if i % 5 < 3 { 1u32 } else { 0u32 });
            // Covariate positively correlated with time (continuous, mean-centered).
            x.push(vec![(i as f64) / (n as f64) - 0.5]);
        }
        let result = fine_gray_fit(&times, &events, &x, 1).unwrap();
        assert!(
            result.coefficients[0].is_finite(),
            "coef should be finite, got {}",
            result.coefficients[0]
        );
        // Subjects with higher x have later event times, so negative coef.
        assert!(
            result.coefficients[0] < 0.0,
            "expected negative coef, got {}",
            result.coefficients[0]
        );
    }

    #[test]
    fn chi2_sf_basic() {
        // P(chi2(1) > 3.84) ≈ 0.05
        let p = chi2_sf(3.841, 1);
        assert!((p - 0.05).abs() < 0.01, "chi2_sf(3.841, 1) = {}", p);
        // P(chi2(1) > 0) = 1.
        assert!((chi2_sf(0.0, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normal_quantile_basic() {
        assert!((normal_quantile(0.5) - 0.0).abs() < 1e-6);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) + 1.96).abs() < 0.01);
    }
}
