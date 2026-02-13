//! Fused single-pass CPU kernels for common unbinned model topologies.
//!
//! Instead of evaluating each PDF in a separate pass and then combining with logsumexp,
//! fused kernels compute the full `log(Σ νp · fp(x))` per event in one loop.
//! This eliminates intermediate allocations and reduces memory bandwidth by ~3×.
//!
//! Topology detection is performed at evaluation time; unsupported topologies
//! fall back to the generic multi-pass path.

use crate::math::{log_diff_exp, standard_normal_cdf, standard_normal_logpdf};
use crate::model::{Process, UnbinnedChannel};
use ns_core::Result;
use rayon::prelude::*;
use wide::f64x4;

/// Result of a fused NLL (+optional gradient) evaluation.
pub(crate) struct FusedResult {
    /// Sum of `log f(x_i)` over events (NOT negated; caller subtracts from nll).
    pub sum_logf: f64,
    /// Per-process sum of `r_p / ν_p` over events (for yield gradient), length = n_proc.
    pub sum_r_over_nu: Vec<f64>,
    /// Packed per-shape-parameter gradient sums (for shape gradient), length = total_shape.
    pub sum_r_dlogp: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Topology detection
// ---------------------------------------------------------------------------

/// Identifies the fused-kernel topology of a channel, if supported.
pub(crate) enum FusedTopology {
    /// Two processes: Gaussian (signal) + Exponential (background), 1D observable.
    GaussExp {
        /// Column name for the observable.
        obs_name: String,
        /// Index of `mu` in the Gaussian's shape_param_indices.
        gauss_mu_idx: usize,
        /// Index of `sigma` in the Gaussian's shape_param_indices.
        gauss_sigma_idx: usize,
        /// Index of `lambda` in the Exponential's shape_param_indices.
        exp_lambda_idx: usize,
    },
    /// Two processes: CrystalBall (signal) + Exponential (background), 1D observable.
    CbExp {
        obs_name: String,
        /// Indices of [mu, sigma, alpha, n] in global params (via shape_param_indices).
        cb_mu_idx: usize,
        cb_sigma_idx: usize,
        cb_alpha_idx: usize,
        cb_n_idx: usize,
        /// Index of `lambda` in the Exponential's shape_param_indices.
        exp_lambda_idx: usize,
    },
}

/// Try to detect a fused topology for this channel.
/// Returns `None` if the channel doesn't match any known pattern.
pub(crate) fn detect_topology(ch: &UnbinnedChannel) -> Option<FusedTopology> {
    if ch.processes.len() != 2 {
        return None;
    }

    // Identify each process by tag.
    let tag0 = ch.processes[0].pdf.pdf_tag();
    let tag1 = ch.processes[1].pdf.pdf_tag();

    // Helper: find (signal_idx, exp_idx) where signal matches `sig_tag`.
    let find_pair = |sig_tag: &str| -> Option<(usize, usize)> {
        if tag0 == sig_tag && tag1 == "exponential" {
            Some((0, 1))
        } else if tag1 == sig_tag && tag0 == "exponential" {
            Some((1, 0))
        } else {
            None
        }
    };

    // --- Try GaussExp ---
    if let Some((gauss_idx, exp_idx)) = find_pair("gaussian") {
        let gp = &ch.processes[gauss_idx];
        let ep = &ch.processes[exp_idx];
        if gp.shape_param_indices.len() == 2
            && ep.shape_param_indices.len() == 1
            && gp.pdf.observables().len() == 1
            && ep.pdf.observables().len() == 1
            && gp.pdf.observables()[0] == ep.pdf.observables()[0]
        {
            return Some(FusedTopology::GaussExp {
                obs_name: gp.pdf.observables()[0].clone(),
                gauss_mu_idx: gp.shape_param_indices[0],
                gauss_sigma_idx: gp.shape_param_indices[1],
                exp_lambda_idx: ep.shape_param_indices[0],
            });
        }
    }

    // --- Try CbExp ---
    if let Some((cb_idx, exp_idx)) = find_pair("crystal_ball") {
        let cp = &ch.processes[cb_idx];
        let ep = &ch.processes[exp_idx];
        if cp.shape_param_indices.len() == 4
            && ep.shape_param_indices.len() == 1
            && cp.pdf.observables().len() == 1
            && ep.pdf.observables().len() == 1
            && cp.pdf.observables()[0] == ep.pdf.observables()[0]
        {
            return Some(FusedTopology::CbExp {
                obs_name: cp.pdf.observables()[0].clone(),
                cb_mu_idx: cp.shape_param_indices[0],
                cb_sigma_idx: cp.shape_param_indices[1],
                cb_alpha_idx: cp.shape_param_indices[2],
                cb_n_idx: cp.shape_param_indices[3],
                exp_lambda_idx: ep.shape_param_indices[0],
            });
        }
    }

    None
}

fn is_crystal_ball(proc: &Process) -> bool {
    proc.pdf.pdf_tag() == "crystal_ball"
}

fn is_gaussian(proc: &Process) -> bool {
    proc.pdf.pdf_tag() == "gaussian"
}

// ---------------------------------------------------------------------------
// Fused GaussExp kernel
// ---------------------------------------------------------------------------

/// Fused single-pass NLL kernel for Gaussian + Exponential mixture.
///
/// Computes `Σ_i w_i · log(ν_g · g(x_i) + ν_e · e(x_i))` in one loop.
/// Also computes gradient contributions if `want_grad` is true.
pub(crate) fn fused_gauss_exp_nll(
    ch: &UnbinnedChannel,
    topo: &FusedTopology,
    params: &[f64],
    yields: &[f64],
    want_grad: bool,
) -> Result<FusedResult> {
    let FusedTopology::GaussExp { ref obs_name, gauss_mu_idx, gauss_sigma_idx, exp_lambda_idx } =
        *topo
    else {
        return Err(ns_core::Error::Validation(
            "fused_gauss_exp_nll called with non-GaussExp topology".into(),
        ));
    };

    let xs = ch
        .data
        .column(obs_name)
        .ok_or_else(|| ns_core::Error::Validation(format!("missing column '{obs_name}'")))?;
    let (a, b) = ch
        .data
        .bounds(obs_name)
        .ok_or_else(|| ns_core::Error::Validation(format!("missing bounds for '{obs_name}'")))?;
    let weights = ch.data.weights();
    let n_events = ch.data.n_events();

    // Determine which process index is Gaussian and which is Exponential.
    let (gauss_pidx, exp_pidx) =
        if is_gaussian(&ch.processes[0]) { (0usize, 1usize) } else { (1usize, 0usize) };

    let nu_g = yields[gauss_pidx];
    let nu_e = yields[exp_pidx];

    // Gaussian parameters.
    let mu = params[gauss_mu_idx];
    let sigma = params[gauss_sigma_idx];
    let inv_sigma = 1.0 / sigma;

    // Gaussian normalization on [a, b].
    let z_a = (a - mu) * inv_sigma;
    let z_b = (b - mu) * inv_sigma;
    let mut cdf_range = standard_normal_cdf(z_b) - standard_normal_cdf(z_a);
    if !cdf_range.is_finite() || cdf_range <= 0.0 {
        cdf_range = f64::MIN_POSITIVE;
    }
    let log_gauss_norm = sigma.ln() + cdf_range.ln();

    // Exponential normalization on [a, b].
    let lambda = params[exp_lambda_idx];
    let log_exp_norm = logz_exp(lambda, a, b)?;

    // Precompute log-yields for logsumexp.
    let log_nu_g = if nu_g > 0.0 { nu_g.ln() } else { f64::NEG_INFINITY };
    let log_nu_e = if nu_e > 0.0 { nu_e.ln() } else { f64::NEG_INFINITY };

    // Gradient of Gaussian normalization (needed if want_grad).
    let (dlogz_g_dmu, dlogz_g_dsigma) = if want_grad {
        let phi_a = standard_normal_logpdf(z_a).exp();
        let phi_b = standard_normal_logpdf(z_b).exp();
        let inv_z = 1.0 / cdf_range;
        let dlz_dmu = (phi_a - phi_b) * inv_sigma * inv_z;
        let dlz_dsigma = (z_a * phi_a - z_b * phi_b) * inv_sigma * inv_z;
        (dlz_dmu, dlz_dsigma)
    } else {
        (0.0, 0.0)
    };

    // Gradient of Exponential normalization.
    let dlogz_e_dlambda = if want_grad { dlogz_exp_dlambda(lambda, a, b)? } else { 0.0 };

    // --- SIMD-vectorized single-pass event loop ---
    // Process 4 events at a time using f64x4 (AVX2 / NEON).
    // Sequential for small N; rayon-parallel chunks for large N.

    #[derive(Clone)]
    struct FusedAcc {
        sum_logf: f64,
        sum_r_g_over_nu: f64,
        sum_r_e_over_nu: f64,
        sum_r_g_dmu: f64,
        sum_r_g_dsigma: f64,
        sum_r_e_dlambda: f64,
    }

    let zero = || FusedAcc {
        sum_logf: 0.0,
        sum_r_g_over_nu: 0.0,
        sum_r_e_over_nu: 0.0,
        sum_r_g_dmu: 0.0,
        sum_r_g_dsigma: 0.0,
        sum_r_e_dlambda: 0.0,
    };

    // Broadcast scalar constants to f64x4.
    let v_mu = f64x4::splat(mu);
    let v_inv_sigma = f64x4::splat(inv_sigma);
    let v_lambda = f64x4::splat(lambda);
    let v_log_gauss_norm = f64x4::splat(log_gauss_norm);
    let v_log_exp_norm = f64x4::splat(log_exp_norm);
    let v_log_nu_g = f64x4::splat(log_nu_g);
    let v_log_nu_e = f64x4::splat(log_nu_e);
    let v_half = f64x4::splat(0.5);
    let v_half_ln2pi = f64x4::splat(0.5 * std::f64::consts::TAU.ln());
    let v_one = f64x4::splat(1.0);
    let v_neg_inf = f64x4::splat(f64::NEG_INFINITY);

    // Gradient-only constants (broadcast once).
    let v_inv_nu_g = f64x4::splat(if nu_g > 0.0 { 1.0 / nu_g } else { 0.0 });
    let v_inv_nu_e = f64x4::splat(if nu_e > 0.0 { 1.0 / nu_e } else { 0.0 });
    let v_dlogz_g_dmu = f64x4::splat(dlogz_g_dmu);
    let v_dlogz_g_dsigma = f64x4::splat(dlogz_g_dsigma);
    let v_dlogz_e_dlambda = f64x4::splat(dlogz_e_dlambda);

    /// Process a contiguous slice of events using f64x4 SIMD + scalar remainder.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn process_slice(
        xs_slice: &[f64],
        weights_slice: Option<&[f64]>,
        acc: &mut FusedAcc,
        want_grad: bool,
        v_mu: f64x4,
        v_inv_sigma: f64x4,
        v_lambda: f64x4,
        v_log_gauss_norm: f64x4,
        v_log_exp_norm: f64x4,
        v_log_nu_g: f64x4,
        v_log_nu_e: f64x4,
        v_half: f64x4,
        v_half_ln2pi: f64x4,
        v_one: f64x4,
        v_neg_inf: f64x4,
        v_inv_nu_g: f64x4,
        v_inv_nu_e: f64x4,
        v_dlogz_g_dmu: f64x4,
        v_dlogz_g_dsigma: f64x4,
        v_dlogz_e_dlambda: f64x4,
        mu: f64,
        inv_sigma: f64,
        lambda: f64,
        log_gauss_norm: f64,
        log_exp_norm: f64,
        log_nu_g: f64,
        log_nu_e: f64,
        dlogz_g_dmu: f64,
        dlogz_g_dsigma: f64,
        dlogz_e_dlambda: f64,
        nu_g: f64,
        nu_e: f64,
    ) {
        let n = xs_slice.len();
        let chunks = n / 4;
        let _remainder = n % 4;

        // --- f64x4 vectorized loop ---
        let mut v_sum_logf = f64x4::ZERO;
        let mut v_sum_rg_onu = f64x4::ZERO;
        let mut v_sum_re_onu = f64x4::ZERO;
        let mut v_sum_rg_dmu = f64x4::ZERO;
        let mut v_sum_rg_ds = f64x4::ZERO;
        let mut v_sum_re_dl = f64x4::ZERO;

        for c in 0..chunks {
            let base = c * 4;
            let v_x = f64x4::new([
                xs_slice[base],
                xs_slice[base + 1],
                xs_slice[base + 2],
                xs_slice[base + 3],
            ]);
            let v_w = if let Some(ws) = weights_slice {
                f64x4::new([ws[base], ws[base + 1], ws[base + 2], ws[base + 3]])
            } else {
                v_one
            };

            // z_x = (x - mu) * inv_sigma
            let v_z = (v_x - v_mu) * v_inv_sigma;

            // standard_normal_logpdf(z) = -0.5 * z^2 - 0.5 * ln(2π)
            let v_logpdf = v_z.mul_neg_add(v_z * v_half, -v_half_ln2pi);

            // log_gauss = logpdf - log_gauss_norm
            let v_log_g = v_logpdf - v_log_gauss_norm;

            // log_exp = lambda * x - log_exp_norm
            let v_log_e = v_x.mul_add(v_lambda, -v_log_exp_norm);

            // term_g/e = log_nu + log_pdf
            let v_tg = v_log_nu_g + v_log_g;
            let v_te = v_log_nu_e + v_log_e;

            // logsumexp(term_g, term_e)
            let v_max = v_tg.max(v_te);
            let v_finite_mask = v_max.is_finite();
            let v_sum_exp = (v_tg - v_max).exp() + (v_te - v_max).exp();
            let v_logf_finite = v_max + v_sum_exp.ln();
            let v_logf = v_finite_mask.blend(v_logf_finite, v_neg_inf);

            v_sum_logf += v_w * v_logf;

            if want_grad {
                let v_rg = (v_tg - v_logf).exp();
                let v_re = (v_te - v_logf).exp();
                let v_w_rg = v_w * v_rg;
                let v_w_re = v_w * v_re;

                v_sum_rg_onu += v_w_rg * v_inv_nu_g;
                v_sum_re_onu += v_w_re * v_inv_nu_e;

                // d/dmu log_gauss = z * inv_sigma - dlogz_g_dmu
                let v_dlg_dmu = v_z.mul_add(v_inv_sigma, -v_dlogz_g_dmu);
                // d/dsigma log_gauss = (z^2 - 1) * inv_sigma - dlogz_g_dsigma
                let v_dlg_ds = (v_z * v_z - v_one).mul_add(v_inv_sigma, -v_dlogz_g_dsigma);
                // d/dlambda log_exp = x - dlogz_e_dlambda
                let v_dle_dl = v_x - v_dlogz_e_dlambda;

                v_sum_rg_dmu += v_w_rg * v_dlg_dmu;
                v_sum_rg_ds += v_w_rg * v_dlg_ds;
                v_sum_re_dl += v_w_re * v_dle_dl;
            }
        }

        // Horizontal reduce f64x4 → scalar.
        acc.sum_logf += v_sum_logf.reduce_add();
        if want_grad {
            acc.sum_r_g_over_nu += v_sum_rg_onu.reduce_add();
            acc.sum_r_e_over_nu += v_sum_re_onu.reduce_add();
            acc.sum_r_g_dmu += v_sum_rg_dmu.reduce_add();
            acc.sum_r_g_dsigma += v_sum_rg_ds.reduce_add();
            acc.sum_r_e_dlambda += v_sum_re_dl.reduce_add();
        }

        // --- Scalar remainder (0..3 events) ---
        let rem_start = chunks * 4;
        for i in rem_start..n {
            let x = xs_slice[i];
            let evt_w = weights_slice.map(|ws| ws[i]).unwrap_or(1.0);
            if evt_w == 0.0 {
                continue;
            }

            let z_x = (x - mu) * inv_sigma;
            let log_gauss = standard_normal_logpdf(z_x) - log_gauss_norm;
            let log_exp = lambda * x - log_exp_norm;
            let term_g = log_nu_g + log_gauss;
            let term_e = log_nu_e + log_exp;
            let max_term = term_g.max(term_e);
            let logf = if max_term.is_finite() {
                max_term + ((term_g - max_term).exp() + (term_e - max_term).exp()).ln()
            } else {
                f64::NEG_INFINITY
            };
            acc.sum_logf += evt_w * logf;

            if want_grad {
                let r_g = (term_g - logf).exp();
                let r_e = (term_e - logf).exp();
                if nu_g > 0.0 {
                    acc.sum_r_g_over_nu += evt_w * (r_g / nu_g);
                }
                if nu_e > 0.0 {
                    acc.sum_r_e_over_nu += evt_w * (r_e / nu_e);
                }
                let dlog_g_dmu = z_x * inv_sigma - dlogz_g_dmu;
                let dlog_g_dsigma = (z_x * z_x - 1.0) * inv_sigma - dlogz_g_dsigma;
                let dlog_e_dlambda = x - dlogz_e_dlambda;
                acc.sum_r_g_dmu += evt_w * r_g * dlog_g_dmu;
                acc.sum_r_g_dsigma += evt_w * r_g * dlog_g_dsigma;
                acc.sum_r_e_dlambda += evt_w * r_e * dlog_e_dlambda;
            }
        }
    }

    const PAR_THRESHOLD: usize = 8_000;
    const PAR_CHUNK: usize = 1024;

    // Avoid nested Rayon parallelism (e.g. toys-parallel outer loop calling into an events-parallel
    // kernel): if we're already on a Rayon worker thread, stay sequential here.
    let can_par = rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1;

    let acc = if can_par && n_events >= PAR_THRESHOLD {
        let ws = weights;
        xs.par_chunks(PAR_CHUNK)
            .enumerate()
            .fold(zero, |mut acc, (chunk_idx, x_chunk)| {
                let w_chunk = ws.map(|w| &w[chunk_idx * PAR_CHUNK..][..x_chunk.len()]);
                process_slice(
                    x_chunk,
                    w_chunk,
                    &mut acc,
                    want_grad,
                    v_mu,
                    v_inv_sigma,
                    v_lambda,
                    v_log_gauss_norm,
                    v_log_exp_norm,
                    v_log_nu_g,
                    v_log_nu_e,
                    v_half,
                    v_half_ln2pi,
                    v_one,
                    v_neg_inf,
                    v_inv_nu_g,
                    v_inv_nu_e,
                    v_dlogz_g_dmu,
                    v_dlogz_g_dsigma,
                    v_dlogz_e_dlambda,
                    mu,
                    inv_sigma,
                    lambda,
                    log_gauss_norm,
                    log_exp_norm,
                    log_nu_g,
                    log_nu_e,
                    dlogz_g_dmu,
                    dlogz_g_dsigma,
                    dlogz_e_dlambda,
                    nu_g,
                    nu_e,
                );
                acc
            })
            .reduce(zero, |mut a, b| {
                a.sum_logf += b.sum_logf;
                a.sum_r_g_over_nu += b.sum_r_g_over_nu;
                a.sum_r_e_over_nu += b.sum_r_e_over_nu;
                a.sum_r_g_dmu += b.sum_r_g_dmu;
                a.sum_r_g_dsigma += b.sum_r_g_dsigma;
                a.sum_r_e_dlambda += b.sum_r_e_dlambda;
                a
            })
    } else {
        let mut acc = zero();
        let ws = weights;
        process_slice(
            xs,
            ws,
            &mut acc,
            want_grad,
            v_mu,
            v_inv_sigma,
            v_lambda,
            v_log_gauss_norm,
            v_log_exp_norm,
            v_log_nu_g,
            v_log_nu_e,
            v_half,
            v_half_ln2pi,
            v_one,
            v_neg_inf,
            v_inv_nu_g,
            v_inv_nu_e,
            v_dlogz_g_dmu,
            v_dlogz_g_dsigma,
            v_dlogz_e_dlambda,
            mu,
            inv_sigma,
            lambda,
            log_gauss_norm,
            log_exp_norm,
            log_nu_g,
            log_nu_e,
            dlogz_g_dmu,
            dlogz_g_dsigma,
            dlogz_e_dlambda,
            nu_g,
            nu_e,
        );
        acc
    };

    // Pack results into FusedResult.
    let n_proc = 2;
    let mut sum_r_over_nu = vec![0.0; n_proc];
    sum_r_over_nu[gauss_pidx] = acc.sum_r_g_over_nu;
    sum_r_over_nu[exp_pidx] = acc.sum_r_e_over_nu;

    // Shape gradients: Gaussian has 2 params, Exponential has 1 = total 3.
    // Layout matches the generic path: [gauss_param0, gauss_param1, exp_param0]
    // where the offsets follow process order.
    let total_shape = 3;
    let mut sum_r_dlogp = vec![0.0; total_shape];
    if want_grad {
        let gauss_off = if gauss_pidx == 0 { 0 } else { 1 };
        let exp_off = if exp_pidx == 0 { 0 } else { 2 };
        sum_r_dlogp[gauss_off] = acc.sum_r_g_dmu;
        sum_r_dlogp[gauss_off + 1] = acc.sum_r_g_dsigma;
        sum_r_dlogp[exp_off] = acc.sum_r_e_dlambda;
    }

    Ok(FusedResult { sum_logf: acc.sum_logf, sum_r_over_nu, sum_r_dlogp })
}

// ---------------------------------------------------------------------------
// Fused CrystalBall+Exp kernel
// ---------------------------------------------------------------------------

/// Fused single-pass NLL kernel for CrystalBall + Exponential mixture.
///
/// Computes `Σ_i w_i · log(ν_cb · cb(x_i) + ν_e · e(x_i))` in one loop.
/// The CB piece-wise logf (Gaussian core for t > -alpha, power-law tail otherwise)
/// is handled per-event with a branch; SIMD uses blend (mask-select).
pub(crate) fn fused_cb_exp_nll(
    ch: &UnbinnedChannel,
    topo: &FusedTopology,
    params: &[f64],
    yields: &[f64],
    want_grad: bool,
) -> Result<FusedResult> {
    let (obs_name, cb_mu_idx, cb_sigma_idx, cb_alpha_idx, cb_n_idx, exp_lambda_idx) = match *topo {
        FusedTopology::CbExp {
            ref obs_name,
            cb_mu_idx,
            cb_sigma_idx,
            cb_alpha_idx,
            cb_n_idx,
            exp_lambda_idx,
        } => (obs_name, cb_mu_idx, cb_sigma_idx, cb_alpha_idx, cb_n_idx, exp_lambda_idx),
        _ => unreachable!("fused_cb_exp_nll called with non-CbExp topology"),
    };

    let xs = ch
        .data
        .column(obs_name)
        .ok_or_else(|| ns_core::Error::Validation(format!("missing column '{obs_name}'")))?;
    let (a, b) = ch
        .data
        .bounds(obs_name)
        .ok_or_else(|| ns_core::Error::Validation(format!("missing bounds for '{obs_name}'")))?;
    let weights = ch.data.weights();
    let n_events = ch.data.n_events();

    let (cb_pidx, exp_pidx) =
        if is_crystal_ball(&ch.processes[0]) { (0usize, 1usize) } else { (1usize, 0usize) };

    let nu_cb = yields[cb_pidx];
    let nu_e = yields[exp_pidx];

    // --- CB parameters ---
    let mu_cb = params[cb_mu_idx];
    let sigma_cb = params[cb_sigma_idx];
    let alpha = params[cb_alpha_idx];
    let n_tail = params[cb_n_idx];
    let inv_sigma = 1.0 / sigma_cb;

    // Validate CB parameters.
    if !alpha.is_finite() || alpha <= 0.0 || !n_tail.is_finite() || n_tail <= 1.0 {
        return Err(ns_core::Error::Validation(format!(
            "CrystalBall requires alpha > 0 and n > 1, got alpha={alpha}, n={n_tail}"
        )));
    }

    // Tail constants: log_a = n*ln(n/alpha) - 0.5*alpha^2, b_tail = n/alpha - alpha.
    let log_a = n_tail * (n_tail / alpha).ln() - 0.5 * alpha * alpha;
    let b_tail = n_tail / alpha - alpha;
    let t0 = -alpha; // tail/core boundary in t-space

    // CB normalization on [a, b] in t-space.
    let t_a = (a - mu_cb) * inv_sigma;
    let t_b = (b - mu_cb) * inv_sigma;

    let cb_norm = {
        let tail_int = if t_a < t0 {
            let t_upper = t_b.min(t0);
            cb_tail_integral(log_a, n_tail, b_tail, t_a, t_upper)?
        } else {
            0.0
        };
        let core_lo = t_a.max(t0);
        let core_hi = t_b;
        let core_int = if core_hi > core_lo { gauss_integral_range(core_lo, core_hi) } else { 0.0 };
        tail_int + core_int
    };
    if !cb_norm.is_finite() || cb_norm <= 0.0 {
        return Err(ns_core::Error::Validation(format!(
            "CB normalization is not finite/positive: {cb_norm}"
        )));
    }
    let log_cb_norm = sigma_cb.ln() + cb_norm.ln();

    // Exponential parameters + normalization.
    let lambda = params[exp_lambda_idx];
    let log_exp_norm = logz_exp(lambda, a, b)?;

    // Log-yields for logsumexp.
    let log_nu_cb = if nu_cb > 0.0 { nu_cb.ln() } else { f64::NEG_INFINITY };
    let log_nu_e = if nu_e > 0.0 { nu_e.ln() } else { f64::NEG_INFINITY };

    // --- Gradient precomputation ---
    // CB normalization derivatives w.r.t. alpha, n (Leibniz + tail integral derivatives).
    let (dlognorm_dmu, dlognorm_dsigma, dlognorm_dalpha, dlognorm_dn) = if want_grad {
        // Endpoint derivatives via Leibniz rule (same as crystal_ball.rs).
        let logf_a = if t_a > t0 { -0.5 * t_a * t_a } else { log_a - n_tail * (b_tail - t_a).ln() };
        let logf_b = if t_b > t0 { -0.5 * t_b * t_b } else { log_a - n_tail * (b_tail - t_b).ln() };
        let f_a = logf_a.exp();
        let f_b = logf_b.exp();
        let d_i_dmu = (f_a - f_b) * inv_sigma;
        let d_i_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma;

        // Tail integral derivatives w.r.t. alpha, n.
        let (d_i_dalpha, d_i_dn) = if t_a < t0 {
            let t_upper = t_b.min(t0);
            cb_tail_integral_derivs(alpha, n_tail, log_a, b_tail, t_a, t_upper)?
        } else {
            (0.0, 0.0)
        };

        let inv_norm = 1.0 / cb_norm;
        (
            d_i_dmu * inv_norm,
            d_i_dsigma * inv_norm, // NOT including 1/sigma; that is subtracted separately per-event
            d_i_dalpha * inv_norm,
            d_i_dn * inv_norm,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let dlogz_e_dlambda = if want_grad { dlogz_exp_dlambda(lambda, a, b)? } else { 0.0 };

    // --- Single-pass event loop (scalar, with rayon for large N) ---
    // CB has a branch (tail vs core) per event — not ideal for SIMD blend due to
    // the `ln(b_tail - t)` in the tail which needs care for numerical stability.
    // We use scalar loop with rayon parallelism; still eliminates multi-pass allocation.

    #[derive(Clone)]
    struct CbAcc {
        sum_logf: f64,
        sum_r_cb_over_nu: f64,
        sum_r_e_over_nu: f64,
        // CB shape grads: [mu, sigma, alpha, n].
        sum_r_cb_dmu: f64,
        sum_r_cb_dsigma: f64,
        sum_r_cb_dalpha: f64,
        sum_r_cb_dn: f64,
        sum_r_e_dlambda: f64,
    }

    let zero = || CbAcc {
        sum_logf: 0.0,
        sum_r_cb_over_nu: 0.0,
        sum_r_e_over_nu: 0.0,
        sum_r_cb_dmu: 0.0,
        sum_r_cb_dsigma: 0.0,
        sum_r_cb_dalpha: 0.0,
        sum_r_cb_dn: 0.0,
        sum_r_e_dlambda: 0.0,
    };

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn process_cb_event(
        x: f64,
        w: f64,
        acc: &mut CbAcc,
        want_grad: bool,
        mu_cb: f64,
        inv_sigma: f64,
        t0: f64,
        log_a: f64,
        n_tail: f64,
        b_tail: f64,
        alpha: f64,
        log_cb_norm: f64,
        lambda: f64,
        log_exp_norm: f64,
        log_nu_cb: f64,
        log_nu_e: f64,
        nu_cb: f64,
        nu_e: f64,
        dlognorm_dmu: f64,
        dlognorm_dsigma: f64,
        dlognorm_dalpha: f64,
        dlognorm_dn: f64,
        dlogz_e_dlambda: f64,
    ) {
        if w == 0.0 {
            return;
        }

        let t = (x - mu_cb) * inv_sigma;
        let is_core = t > t0;

        // CB log-prob (unnormalized, before subtracting log_cb_norm).
        let logf_cb_raw = if is_core { -0.5 * t * t } else { log_a - n_tail * (b_tail - t).ln() };
        let log_cb = logf_cb_raw - log_cb_norm;

        // Exponential log-prob.
        let log_exp = lambda * x - log_exp_norm;

        // logsumexp(log_nu_cb + log_cb, log_nu_e + log_exp).
        let term_cb = log_nu_cb + log_cb;
        let term_e = log_nu_e + log_exp;
        let max_term = term_cb.max(term_e);
        let logf = if max_term.is_finite() {
            max_term + ((term_cb - max_term).exp() + (term_e - max_term).exp()).ln()
        } else {
            f64::NEG_INFINITY
        };

        acc.sum_logf += w * logf;

        if want_grad {
            let r_cb = (term_cb - logf).exp();
            let r_e = (term_e - logf).exp();

            if nu_cb > 0.0 {
                acc.sum_r_cb_over_nu += w * (r_cb / nu_cb);
            }
            if nu_e > 0.0 {
                acc.sum_r_e_over_nu += w * (r_e / nu_e);
            }

            // CB shape gradients: d/dθ log_cb = d/dθ logf_raw - d/dθ log_cb_norm.
            // dt/dmu = -inv_sigma, dt/dsigma = -t * inv_sigma.
            let (dlogf_dt, dlogf_dalpha, dlogf_dn) = if is_core {
                (-t, 0.0, 0.0)
            } else {
                let bt = b_tail - t;
                let dft = n_tail / bt; // d/dt [log_a - n*ln(b-t)] = n/(b-t)
                // d/dalpha: d(log_a)/dalpha + d/dalpha(-n*ln(b-t)) via db/dalpha
                let dln_a_da = -(n_tail / alpha + alpha);
                let db_da = -(n_tail / (alpha * alpha) + 1.0);
                let dfa = dln_a_da - n_tail * db_da / bt;
                // d/dn: d(log_a)/dn + d/dn(-n*ln(b-t)) via db/dn
                let dln_a_dn = 1.0 + (n_tail / alpha).ln();
                let db_dn = 1.0 / alpha;
                let dfn = dln_a_dn - (bt).ln() - n_tail * db_dn / bt;
                (dft, dfa, dfn)
            };

            let d_mu = -inv_sigma * dlogf_dt - dlognorm_dmu;
            let d_sigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlognorm_dsigma;
            let d_alpha = dlogf_dalpha - dlognorm_dalpha;
            let d_n = dlogf_dn - dlognorm_dn;
            let d_lambda = x - dlogz_e_dlambda;

            acc.sum_r_cb_dmu += w * r_cb * d_mu;
            acc.sum_r_cb_dsigma += w * r_cb * d_sigma;
            acc.sum_r_cb_dalpha += w * r_cb * d_alpha;
            acc.sum_r_cb_dn += w * r_cb * d_n;
            acc.sum_r_e_dlambda += w * r_e * d_lambda;
        }
    }

    const PAR_THRESHOLD: usize = 8_000;
    const PAR_CHUNK: usize = 1024;

    // Avoid nested Rayon parallelism; see GaussExp kernel above.
    let can_par = rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1;

    let acc = if can_par && n_events >= PAR_THRESHOLD {
        let ws = weights;
        xs.par_chunks(PAR_CHUNK)
            .enumerate()
            .fold(zero, |mut acc, (chunk_idx, x_chunk)| {
                let w_chunk = ws.map(|w| &w[chunk_idx * PAR_CHUNK..][..x_chunk.len()]);
                for (i, &x) in x_chunk.iter().enumerate() {
                    let w = w_chunk.map(|wc| wc[i]).unwrap_or(1.0);
                    process_cb_event(
                        x,
                        w,
                        &mut acc,
                        want_grad,
                        mu_cb,
                        inv_sigma,
                        t0,
                        log_a,
                        n_tail,
                        b_tail,
                        alpha,
                        log_cb_norm,
                        lambda,
                        log_exp_norm,
                        log_nu_cb,
                        log_nu_e,
                        nu_cb,
                        nu_e,
                        dlognorm_dmu,
                        dlognorm_dsigma,
                        dlognorm_dalpha,
                        dlognorm_dn,
                        dlogz_e_dlambda,
                    );
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                a.sum_logf += b.sum_logf;
                a.sum_r_cb_over_nu += b.sum_r_cb_over_nu;
                a.sum_r_e_over_nu += b.sum_r_e_over_nu;
                a.sum_r_cb_dmu += b.sum_r_cb_dmu;
                a.sum_r_cb_dsigma += b.sum_r_cb_dsigma;
                a.sum_r_cb_dalpha += b.sum_r_cb_dalpha;
                a.sum_r_cb_dn += b.sum_r_cb_dn;
                a.sum_r_e_dlambda += b.sum_r_e_dlambda;
                a
            })
    } else {
        let mut acc = zero();
        for (i, &x) in xs.iter().enumerate() {
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            process_cb_event(
                x,
                w,
                &mut acc,
                want_grad,
                mu_cb,
                inv_sigma,
                t0,
                log_a,
                n_tail,
                b_tail,
                alpha,
                log_cb_norm,
                lambda,
                log_exp_norm,
                log_nu_cb,
                log_nu_e,
                nu_cb,
                nu_e,
                dlognorm_dmu,
                dlognorm_dsigma,
                dlognorm_dalpha,
                dlognorm_dn,
                dlogz_e_dlambda,
            );
        }
        acc
    };

    // Pack results.
    let n_proc = 2;
    let mut sum_r_over_nu = vec![0.0; n_proc];
    sum_r_over_nu[cb_pidx] = acc.sum_r_cb_over_nu;
    sum_r_over_nu[exp_pidx] = acc.sum_r_e_over_nu;

    // Shape gradients: CB has 4 params, Exp has 1 = total 5.
    let total_shape = 5;
    let mut sum_r_dlogp = vec![0.0; total_shape];
    if want_grad {
        let cb_off = if cb_pidx == 0 { 0 } else { 1 };
        let exp_off = if exp_pidx == 0 { 0 } else { 4 };
        sum_r_dlogp[cb_off] = acc.sum_r_cb_dmu;
        sum_r_dlogp[cb_off + 1] = acc.sum_r_cb_dsigma;
        sum_r_dlogp[cb_off + 2] = acc.sum_r_cb_dalpha;
        sum_r_dlogp[cb_off + 3] = acc.sum_r_cb_dn;
        sum_r_dlogp[exp_off] = acc.sum_r_e_dlambda;
    }

    Ok(FusedResult { sum_logf: acc.sum_logf, sum_r_over_nu, sum_r_dlogp })
}

/// CB tail integral: ∫_{t1}^{t2} exp(log_a - n*ln(b-t)) dt, for t2 <= -alpha.
#[inline]
fn cb_tail_integral(log_a: f64, n: f64, b: f64, t1: f64, t2: f64) -> Result<f64> {
    let m = n - 1.0;
    let a_val = log_a.exp();
    let u2 = (b - t2).powf(-m);
    let u1 = (b - t1).powf(-m);
    let i = a_val / m * (u2 - u1);
    if !i.is_finite() || i <= 0.0 {
        return Err(ns_core::Error::Validation(format!(
            "CB tail integral not finite/positive: {i}"
        )));
    }
    Ok(i)
}

/// Gaussian (core) integral: ∫_{t1}^{t2} exp(-0.5*t^2) dt = sqrt(2π) * (Φ(t2) - Φ(t1)).
#[inline]
fn gauss_integral_range(t1: f64, t2: f64) -> f64 {
    let sqrt_2pi = std::f64::consts::TAU.sqrt();
    sqrt_2pi * (standard_normal_cdf(t2) - standard_normal_cdf(t1))
}

/// Derivatives of CB tail integral w.r.t. alpha and n.
#[inline]
fn cb_tail_integral_derivs(
    alpha: f64,
    n: f64,
    log_a: f64,
    b: f64,
    t1: f64,
    t2: f64,
) -> Result<(f64, f64)> {
    let m = n - 1.0;
    let a_val = log_a.exp();

    let b1 = b - t1;
    let b2 = b - t2;
    let u1 = b1.powf(-m);
    let u2 = b2.powf(-m);
    let i = a_val / m * (u2 - u1);

    let dln_a_dalpha = -(n / alpha + alpha);
    let db_dalpha = -(n / (alpha * alpha) + 1.0);
    let v1 = b1.powf(-n);
    let v2 = b2.powf(-n);
    let di_dalpha = i * dln_a_dalpha - a_val * db_dalpha * (v2 - v1);

    let dln_a_dn = 1.0 + (n / alpha).ln();
    let db_dn = 1.0 / alpha;
    let du1_dn = u1 * (-b1.ln() - m * db_dn / b1);
    let du2_dn = u2 * (-b2.ln() - m * db_dn / b2);
    let di_dn = i * dln_a_dn - i / m + (a_val / m) * (du2_dn - du1_dn);

    Ok((di_dalpha, di_dn))
}

// ---------------------------------------------------------------------------
// Helpers (inlined versions of exponential normalization)
// ---------------------------------------------------------------------------

/// `log ∫_a^b exp(λ x) dx` — same as `ExponentialPdf::logz_and_ex` but standalone.
#[inline]
fn logz_exp(lambda: f64, a: f64, b: f64) -> Result<f64> {
    if lambda.abs() < 1e-12 {
        let z = b - a;
        if !(z.is_finite() && z > 0.0) {
            return Err(ns_core::Error::Validation(format!(
                "invalid bounds for fused exp: ({a}, {b})"
            )));
        }
        return Ok(z.ln());
    }

    let t_a = lambda * a;
    let t_b = lambda * b;
    let (hi_t, lo_t) = if t_b >= t_a { (t_b, t_a) } else { (t_a, t_b) };
    let log_num = if hi_t == lo_t { f64::NEG_INFINITY } else { log_diff_exp(hi_t, lo_t) };
    Ok(log_num - lambda.abs().ln())
}

/// `d/dλ log Z` for exponential on `[a, b]`.
/// This equals `E[x]` under the exponential distribution.
#[inline]
fn dlogz_exp_dlambda(lambda: f64, a: f64, b: f64) -> Result<f64> {
    if lambda.abs() < 1e-12 {
        return Ok(0.5 * (a + b));
    }
    let t_a = lambda * a;
    let t_b = lambda * b;
    let (x_hi, x_lo, r) =
        if t_b >= t_a { (b, a, (t_a - t_b).exp()) } else { (a, b, (t_b - t_a).exp()) };
    let denom = 1.0 - r;
    if denom <= 0.0 {
        return Ok(0.5 * (a + b));
    }
    let ratio = (x_hi - x_lo * r) / denom;
    Ok(ratio - 1.0 / lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Parameter, Process, UnbinnedChannel, UnbinnedModel, YieldExpr};
    use crate::pdf::{ExponentialPdf, GaussianPdf, UnbinnedPdf};
    use crate::{EventStore, ObservableSpec};
    use ns_core::traits::LogDensityModel;
    use std::sync::Arc;

    fn make_test_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
        let obs: Vec<f64> = {
            let mut state = 42u64;
            (0..n_events)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state as f64) / (u64::MAX as f64);
                    100.0 + u * 80.0
                })
                .collect()
        };

        let store = EventStore::from_columns(
            vec![ObservableSpec::branch(String::from("x"), (100.0, 180.0))],
            vec![(String::from("x"), obs)],
            None,
        )
        .unwrap();

        let parameters = vec![
            Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
            Parameter {
                name: "mu_sig".into(),
                init: 130.0,
                bounds: (100.0, 180.0),
                constraint: None,
            },
            Parameter {
                name: "sigma_sig".into(),
                init: 5.0,
                bounds: (1.0, 30.0),
                constraint: None,
            },
            Parameter { name: "lambda".into(), init: -0.02, bounds: (-1.0, 0.0), constraint: None },
            Parameter {
                name: "n_bkg".into(),
                init: n_events as f64 * 0.9,
                bounds: (0.0, n_events as f64 * 5.0),
                constraint: None,
            },
        ];

        let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

        let signal = Process {
            name: "signal".into(),
            pdf: Arc::new(GaussianPdf::new("x")),
            shape_param_indices: vec![1, 2],
            yield_expr: YieldExpr::Scaled { base_yield: n_events as f64 * 0.1, scale_index: 0 },
        };

        let background = Process {
            name: "background".into(),
            pdf: Arc::new(ExponentialPdf::new("x")),
            shape_param_indices: vec![3],
            yield_expr: YieldExpr::Parameter { index: 4 },
        };

        let channel = UnbinnedChannel {
            name: "sr".into(),
            include_in_fit: true,
            data: Arc::new(store),
            processes: vec![signal, background],
        };

        let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();
        (model, init)
    }

    #[test]
    fn test_topology_detection() {
        let (model, _) = make_test_model(100);
        let ch = &model.channels()[0];
        let topo = detect_topology(ch);
        assert!(topo.is_some(), "should detect GaussExp topology");
        match topo.unwrap() {
            FusedTopology::GaussExp { obs_name, gauss_mu_idx, gauss_sigma_idx, exp_lambda_idx } => {
                assert_eq!(obs_name, "x");
                assert_eq!(gauss_mu_idx, 1);
                assert_eq!(gauss_sigma_idx, 2);
                assert_eq!(exp_lambda_idx, 3);
            }
            other => {
                panic!("expected GaussExp, got {:?}-like topology", std::mem::discriminant(&other))
            }
        }
    }

    /// Compute NLL+grad via the generic multi-pass path (reference implementation).
    /// Uses the PDF batch methods directly, computing yields from known model structure.
    fn reference_nll_and_grad(n_events: usize) -> (f64, Vec<f64>) {
        let obs: Vec<f64> = {
            let mut state = 42u64;
            (0..n_events)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state as f64) / (u64::MAX as f64);
                    100.0 + u * 80.0
                })
                .collect()
        };

        let store = EventStore::from_columns(
            vec![ObservableSpec::branch(String::from("x"), (100.0, 180.0))],
            vec![(String::from("x"), obs)],
            None,
        )
        .unwrap();

        // Same params as make_test_model
        let mu = 1.0f64; // poi (scale)
        let mu_sig = 130.0f64; // gaussian mu
        let sigma = 5.0f64; // gaussian sigma
        let lambda = -0.02f64; // exponential lambda
        let n_bkg = n_events as f64 * 0.9;

        // yields: signal = base_yield * mu = 0.1*N * 1.0, background = n_bkg
        let nu_sig = n_events as f64 * 0.1 * mu;
        let nu_bkg = n_bkg;

        // PDF evaluations via batch methods
        let gauss = GaussianPdf::new("x");
        let expo = ExponentialPdf::new("x");

        let mut logp_g = vec![0.0f64; n_events];
        let mut dlogp_g = vec![0.0f64; n_events * 2];
        gauss.log_prob_grad_batch(&store, &[mu_sig, sigma], &mut logp_g, &mut dlogp_g).unwrap();

        let mut logp_e = vec![0.0f64; n_events];
        let mut dlogp_e = vec![0.0f64; n_events];
        expo.log_prob_grad_batch(&store, &[lambda], &mut logp_e, &mut dlogp_e).unwrap();

        // NLL = nu_tot - Σ_i log(nu_sig * g(x_i) + nu_bkg * e(x_i))
        let nu_tot = nu_sig + nu_bkg;
        let mut nll = nu_tot;
        // grad: [d/d_mu, d/d_mu_sig, d/d_sigma, d/d_lambda, d/d_n_bkg]
        let mut grad = vec![0.0f64; 5];

        // Yield grad from +nu_tot:
        // d/d_mu(nu_sig) = base_yield = 0.1*N
        grad[0] += n_events as f64 * 0.1;
        // d/d_n_bkg(nu_bkg) = 1
        grad[4] += 1.0;

        let mut sum_logf = 0.0f64;
        let mut sum_r_sig_over_nu = 0.0f64;
        let mut sum_r_bkg_over_nu = 0.0f64;
        let mut sum_r_sig_dmu = 0.0f64;
        let mut sum_r_sig_dsigma = 0.0f64;
        let mut sum_r_bkg_dlambda = 0.0f64;

        for i in 0..n_events {
            let term_g = if nu_sig > 0.0 { nu_sig.ln() + logp_g[i] } else { f64::NEG_INFINITY };
            let term_e = if nu_bkg > 0.0 { nu_bkg.ln() + logp_e[i] } else { f64::NEG_INFINITY };
            let max_t = term_g.max(term_e);
            let logf = max_t + ((term_g - max_t).exp() + (term_e - max_t).exp()).ln();
            sum_logf += logf;

            let r_g = (term_g - logf).exp();
            let r_e = (term_e - logf).exp();

            if nu_sig > 0.0 {
                sum_r_sig_over_nu += r_g / nu_sig;
            }
            if nu_bkg > 0.0 {
                sum_r_bkg_over_nu += r_e / nu_bkg;
            }

            sum_r_sig_dmu += r_g * dlogp_g[i * 2];
            sum_r_sig_dsigma += r_g * dlogp_g[i * 2 + 1];
            sum_r_bkg_dlambda += r_e * dlogp_e[i];
        }

        nll -= sum_logf;

        // Yield grad from event sum
        grad[0] -= n_events as f64 * 0.1 * sum_r_sig_over_nu; // d/d_mu
        grad[4] -= 1.0 * sum_r_bkg_over_nu; // d/d_n_bkg

        // Shape grad from event sum
        grad[1] -= sum_r_sig_dmu; // d/d_mu_sig
        grad[2] -= sum_r_sig_dsigma; // d/d_sigma
        grad[3] -= sum_r_bkg_dlambda; // d/d_lambda

        (nll, grad)
    }

    #[test]
    fn test_fused_nll_matches_generic() {
        for &n in &[100, 1_000, 10_000] {
            let (model, params) = make_test_model(n);

            // Fused path (via model.nll / model.grad_nll which now use fused kernel).
            let fused_nll = model.nll(&params).unwrap();
            let fused_grad = model.grad_nll(&params).unwrap();

            // Reference generic path (manual multi-pass computation).
            let (ref_nll, ref_grad) = reference_nll_and_grad(n);

            let nll_reldiff = ((fused_nll - ref_nll) / ref_nll.abs().max(1e-15)).abs();
            assert!(
                nll_reldiff < 1e-10,
                "NLL mismatch at n={n}: fused={fused_nll}, ref={ref_nll}, reldiff={nll_reldiff}"
            );

            for (j, (&fg, &rg)) in fused_grad.iter().zip(ref_grad.iter()).enumerate() {
                let denom = rg.abs().max(1e-15);
                let reldiff = ((fg - rg) / denom).abs();
                assert!(
                    reldiff < 1e-8,
                    "Grad[{j}] mismatch at n={n}: fused={fg}, ref={rg}, reldiff={reldiff}"
                );
            }
        }
    }

    #[test]
    fn test_fused_nll_only_matches_generic() {
        let (model, params) = make_test_model(5_000);

        let fused_nll = model.nll(&params).unwrap();
        let (ref_nll, _) = reference_nll_and_grad(5_000);

        let reldiff = ((fused_nll - ref_nll) / ref_nll.abs().max(1e-15)).abs();
        assert!(
            reldiff < 1e-10,
            "NLL-only mismatch: fused={fused_nll}, ref={ref_nll}, reldiff={reldiff}"
        );
    }

    #[test]
    fn test_model_fused_and_generic_entrypoints_match() {
        for &n in &[100, 1_000, 10_000] {
            let (model, params) = make_test_model(n);

            let fused_nll = model.nll(&params).unwrap();
            let fused_grad = model.grad_nll(&params).unwrap();
            let generic_nll = model.nll_generic(&params).unwrap();
            let generic_grad = model.grad_nll_generic(&params).unwrap();

            let nll_reldiff = ((fused_nll - generic_nll) / generic_nll.abs().max(1e-15)).abs();
            assert!(
                nll_reldiff < 1e-10,
                "NLL mismatch at n={n}: fused={fused_nll}, generic={generic_nll}, reldiff={nll_reldiff}"
            );
            for (j, (&fg, &gg)) in fused_grad.iter().zip(generic_grad.iter()).enumerate() {
                let denom = gg.abs().max(1e-15);
                let reldiff = ((fg - gg) / denom).abs();
                assert!(
                    reldiff < 1e-8,
                    "Grad[{j}] mismatch at n={n}: fused={fg}, generic={gg}, reldiff={reldiff}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // CrystalBall + Exponential fused kernel tests
    // -----------------------------------------------------------------------

    use crate::pdf::CrystalBallPdf;

    /// Build a CB+Exp model with known parameters.
    /// CB params: mu_cb=130, sigma_cb=5, alpha=1.5, n=5.0 (left-tail CB).
    /// Exp param: lambda=-0.02.
    /// Data: uniform on [100, 180] (xorshift).
    fn make_cb_exp_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
        let obs: Vec<f64> = {
            let mut state = 42u64;
            (0..n_events)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state as f64) / (u64::MAX as f64);
                    100.0 + u * 80.0
                })
                .collect()
        };

        let store = EventStore::from_columns(
            vec![ObservableSpec::branch(String::from("x"), (100.0, 180.0))],
            vec![(String::from("x"), obs)],
            None,
        )
        .unwrap();

        let parameters = vec![
            Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
            Parameter {
                name: "mu_cb".into(),
                init: 130.0,
                bounds: (100.0, 180.0),
                constraint: None,
            },
            Parameter { name: "sigma_cb".into(), init: 5.0, bounds: (1.0, 30.0), constraint: None },
            Parameter { name: "alpha_cb".into(), init: 1.5, bounds: (0.1, 10.0), constraint: None },
            Parameter { name: "n_cb".into(), init: 5.0, bounds: (1.01, 50.0), constraint: None },
            Parameter { name: "lambda".into(), init: -0.02, bounds: (-1.0, 0.0), constraint: None },
            Parameter {
                name: "n_bkg".into(),
                init: n_events as f64 * 0.9,
                bounds: (0.0, n_events as f64 * 5.0),
                constraint: None,
            },
        ];

        let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

        let signal = Process {
            name: "signal".into(),
            pdf: Arc::new(CrystalBallPdf::new("x")),
            shape_param_indices: vec![1, 2, 3, 4], // mu_cb, sigma_cb, alpha_cb, n_cb
            yield_expr: YieldExpr::Scaled { base_yield: n_events as f64 * 0.1, scale_index: 0 },
        };

        let background = Process {
            name: "background".into(),
            pdf: Arc::new(ExponentialPdf::new("x")),
            shape_param_indices: vec![5],
            yield_expr: YieldExpr::Parameter { index: 6 },
        };

        let channel = UnbinnedChannel {
            name: "sr".into(),
            include_in_fit: true,
            data: Arc::new(store),
            processes: vec![signal, background],
        };

        let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();
        (model, init)
    }

    #[test]
    fn test_cb_exp_topology_detection() {
        let (model, _) = make_cb_exp_model(100);
        let ch = &model.channels()[0];
        let topo = detect_topology(ch);
        assert!(topo.is_some(), "should detect CbExp topology");
        match topo.unwrap() {
            FusedTopology::CbExp {
                obs_name,
                cb_mu_idx,
                cb_sigma_idx,
                cb_alpha_idx,
                cb_n_idx,
                exp_lambda_idx,
            } => {
                assert_eq!(obs_name, "x");
                assert_eq!(cb_mu_idx, 1);
                assert_eq!(cb_sigma_idx, 2);
                assert_eq!(cb_alpha_idx, 3);
                assert_eq!(cb_n_idx, 4);
                assert_eq!(exp_lambda_idx, 5);
            }
            other => {
                panic!("expected CbExp, got {:?}-like topology", std::mem::discriminant(&other))
            }
        }
    }

    #[test]
    fn test_cb_exp_fused_matches_generic() {
        for &n in &[100, 1_000, 10_000] {
            let (model, params) = make_cb_exp_model(n);

            let fused_nll = model.nll(&params).unwrap();
            let fused_grad = model.grad_nll(&params).unwrap();
            let generic_nll = model.nll_generic(&params).unwrap();
            let generic_grad = model.grad_nll_generic(&params).unwrap();

            let nll_reldiff = ((fused_nll - generic_nll) / generic_nll.abs().max(1e-15)).abs();
            assert!(
                nll_reldiff < 1e-10,
                "CB+Exp NLL mismatch at n={n}: fused={fused_nll}, generic={generic_nll}, reldiff={nll_reldiff}"
            );
            for (j, (&fg, &gg)) in fused_grad.iter().zip(generic_grad.iter()).enumerate() {
                let denom = gg.abs().max(1e-15);
                let reldiff = ((fg - gg) / denom).abs();
                assert!(
                    reldiff < 1e-8,
                    "CB+Exp Grad[{j}] mismatch at n={n}: fused={fg}, generic={gg}, reldiff={reldiff}"
                );
            }
        }
    }

    #[test]
    fn test_cb_exp_nll_only_matches_generic() {
        let (model, params) = make_cb_exp_model(5_000);
        let fused_nll = model.nll(&params).unwrap();
        let generic_nll = model.nll_generic(&params).unwrap();
        let reldiff = ((fused_nll - generic_nll) / generic_nll.abs().max(1e-15)).abs();
        assert!(
            reldiff < 1e-10,
            "CB+Exp NLL-only mismatch: fused={fused_nll}, generic={generic_nll}, reldiff={reldiff}"
        );
    }

    #[test]
    fn test_cb_exp_gradient_near_tail_junction() {
        // Place mu_cb so that many events land near the tail/core boundary (t ≈ -alpha).
        // With mu_cb=140, alpha=1.5, sigma=5: boundary at x = 140 - 1.5*5 = 132.5.
        // Data on [100, 180] → many events near the junction.
        let n = 2_000;
        let obs: Vec<f64> = {
            let mut state = 99u64;
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state as f64) / (u64::MAX as f64);
                    // Concentrate events around the junction: 125..145.
                    125.0 + u * 20.0
                })
                .collect()
        };

        let store = EventStore::from_columns(
            vec![ObservableSpec::branch(String::from("x"), (125.0, 145.0))],
            vec![(String::from("x"), obs)],
            None,
        )
        .unwrap();

        let parameters = vec![
            Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
            Parameter {
                name: "mu_cb".into(),
                init: 140.0,
                bounds: (125.0, 145.0),
                constraint: None,
            },
            Parameter { name: "sigma_cb".into(), init: 5.0, bounds: (1.0, 30.0), constraint: None },
            Parameter { name: "alpha_cb".into(), init: 1.5, bounds: (0.1, 10.0), constraint: None },
            Parameter { name: "n_cb".into(), init: 5.0, bounds: (1.01, 50.0), constraint: None },
            Parameter { name: "lambda".into(), init: -0.03, bounds: (-1.0, 0.0), constraint: None },
            Parameter {
                name: "n_bkg".into(),
                init: n as f64 * 0.8,
                bounds: (0.0, n as f64 * 5.0),
                constraint: None,
            },
        ];

        let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

        let signal = Process {
            name: "signal".into(),
            pdf: Arc::new(CrystalBallPdf::new("x")),
            shape_param_indices: vec![1, 2, 3, 4],
            yield_expr: YieldExpr::Scaled { base_yield: n as f64 * 0.2, scale_index: 0 },
        };
        let background = Process {
            name: "background".into(),
            pdf: Arc::new(ExponentialPdf::new("x")),
            shape_param_indices: vec![5],
            yield_expr: YieldExpr::Parameter { index: 6 },
        };
        let channel = UnbinnedChannel {
            name: "sr".into(),
            include_in_fit: true,
            data: Arc::new(store),
            processes: vec![signal, background],
        };
        let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();

        let fused_nll = model.nll(&init).unwrap();
        let fused_grad = model.grad_nll(&init).unwrap();
        let generic_nll = model.nll_generic(&init).unwrap();
        let generic_grad = model.grad_nll_generic(&init).unwrap();

        let nll_reldiff = ((fused_nll - generic_nll) / generic_nll.abs().max(1e-15)).abs();
        assert!(
            nll_reldiff < 1e-10,
            "CB+Exp junction NLL mismatch: fused={fused_nll}, generic={generic_nll}, reldiff={nll_reldiff}"
        );
        for (j, (&fg, &gg)) in fused_grad.iter().zip(generic_grad.iter()).enumerate() {
            let denom = gg.abs().max(1e-15);
            let reldiff = ((fg - gg) / denom).abs();
            assert!(
                reldiff < 1e-7,
                "CB+Exp junction Grad[{j}] mismatch: fused={fg}, generic={gg}, reldiff={reldiff}"
            );
        }
    }
}
