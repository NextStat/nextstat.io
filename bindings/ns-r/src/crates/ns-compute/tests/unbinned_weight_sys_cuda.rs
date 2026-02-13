#![cfg(feature = "cuda")]

use ns_compute::cuda_unbinned_weight_sys::CudaUnbinnedWeightSysKernel;

fn histosys_interp(alpha: f64, down: f64, nominal: f64, up: f64, code: u32) -> (f64, f64) {
    if !(alpha.is_finite() && down.is_finite() && nominal.is_finite() && up.is_finite()) {
        return (nominal, 0.0);
    }
    if code == 0 {
        if alpha >= 0.0 {
            let der = up - nominal;
            (nominal + der * alpha, der)
        } else {
            let der = nominal - down;
            (nominal + der * alpha, der)
        }
    } else {
        let delta_up = up - nominal;
        let delta_dn = nominal - down;
        if alpha > 1.0 {
            return (nominal + delta_up * alpha, delta_up);
        }
        if alpha < -1.0 {
            return (nominal + delta_dn * alpha, delta_dn);
        }
        let s = 0.5 * (delta_up + delta_dn);
        let a = 0.0625 * (delta_up - delta_dn);
        let a2 = alpha * alpha;
        let a3 = a2 * alpha;
        let a4 = a2 * a2;
        let a5 = a4 * alpha;
        let a6 = a3 * a3;
        let tmp3 = (3.0 * a6) - (10.0 * a4) + (15.0 * a2);
        let dtmp3 = (18.0 * a5) - (40.0 * a3) + (30.0 * alpha);
        let delta = alpha * s + tmp3 * a;
        let ddelta = s + dtmp3 * a;
        (nominal + delta, ddelta)
    }
}

#[test]
fn cuda_unbinned_weight_sys_apply_matches_cpu() {
    if !CudaUnbinnedWeightSysKernel::is_available() {
        return;
    }

    let n_params = 6usize;
    let n_events = 8usize;
    let alpha_idx = vec![1u32, 4u32];
    let interp = vec![0u32, 1u32]; // code0, code4p

    let mut k = CudaUnbinnedWeightSysKernel::new(n_params, n_events, &alpha_idx, &interp).unwrap();

    let params = vec![0.0, 0.25, 0.0, 0.0, -0.5, 0.0];
    let w_nom = vec![1.0, 2.0, 1.5, 0.75, 3.0, 1.25, 0.9, 1.1];

    let mut w_down = Vec::new();
    let mut w_up = Vec::new();
    // sys0
    w_down.extend_from_slice(&[0.9, 1.8, 1.2, 0.7, 2.8, 1.0, 0.85, 1.0]);
    w_up.extend_from_slice(&[1.1, 2.2, 1.7, 0.8, 3.2, 1.5, 0.95, 1.2]);
    // sys1
    w_down.extend_from_slice(&[0.95, 1.7, 1.4, 0.72, 2.7, 1.1, 0.8, 1.05]);
    w_up.extend_from_slice(&[1.05, 2.3, 1.6, 0.78, 3.3, 1.4, 1.0, 1.15]);

    let (gpu_w, gpu_dw) = k.apply(&params, &w_nom, &w_down, &w_up).unwrap();

    let mut cpu_w = vec![0.0f64; n_events];
    let mut cpu_dw = vec![0.0f64; alpha_idx.len() * n_events];
    for i in 0..n_events {
        let nom = w_nom[i];
        let mut w = nom;
        for (s, (&aidx, &code)) in alpha_idx.iter().zip(interp.iter()).enumerate() {
            let alpha = params[aidx as usize];
            let down = w_down[s * n_events + i];
            let up = w_up[s * n_events + i];
            let (val, der) = histosys_interp(alpha, down, nom, up, code);
            w += val - nom;
            cpu_dw[s * n_events + i] = der;
        }
        if !w.is_finite() || w <= 0.0 {
            w = f64::MIN_POSITIVE;
            for s in 0..alpha_idx.len() {
                cpu_dw[s * n_events + i] = 0.0;
            }
        }
        cpu_w[i] = w;
    }

    for i in 0..n_events {
        let diff = (gpu_w[i] - cpu_w[i]).abs();
        assert!(diff < 1e-10, "w[{i}] mismatch: gpu={} cpu={} diff={}", gpu_w[i], cpu_w[i], diff);
    }
    for j in 0..cpu_dw.len() {
        let diff = (gpu_dw[j] - cpu_dw[j]).abs();
        assert!(
            diff < 1e-10,
            "dw[{j}] mismatch: gpu={} cpu={} diff={}",
            gpu_dw[j],
            cpu_dw[j],
            diff
        );
    }
}
