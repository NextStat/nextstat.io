//! Tests for pyhf parser

use super::schema::*;

#[test]
fn test_parse_simple_workspace() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).expect("Failed to parse simple_workspace.json");

    // Check channels
    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "singlechannel");
    assert_eq!(ws.channels[0].samples.len(), 2);

    // Check signal sample
    let signal = &ws.channels[0].samples[0];
    assert_eq!(signal.name, "signal");
    assert_eq!(signal.data, vec![5.0, 10.0]);
    assert_eq!(signal.modifiers.len(), 1);

    // Check background sample
    let background = &ws.channels[0].samples[1];
    assert_eq!(background.name, "background");
    assert_eq!(background.data, vec![50.0, 60.0]);

    // Check observations
    assert_eq!(ws.observations.len(), 1);
    assert_eq!(ws.observations[0].name, "singlechannel");
    assert_eq!(ws.observations[0].data, vec![53.0, 65.0]);

    // Check measurements
    assert_eq!(ws.measurements.len(), 1);
    assert_eq!(ws.measurements[0].name, "GaussExample");
    assert_eq!(ws.measurements[0].config.poi, "mu");
}

#[test]
fn test_parse_complex_workspace() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).expect("Failed to parse complex_workspace.json");

    // Check channels
    assert_eq!(ws.channels.len(), 2);
    assert!(ws.channels.iter().any(|c| c.name == "SR"));
    assert!(ws.channels.iter().any(|c| c.name == "CR"));

    // Check observations match channels
    assert_eq!(ws.observations.len(), 2);
    assert!(ws.observations.iter().any(|o| o.name == "SR"));
    assert!(ws.observations.iter().any(|o| o.name == "CR"));
}

#[test]
fn test_parse_all_modifier_types() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let mut found_modifiers = std::collections::HashSet::new();

    for channel in &ws.channels {
        for sample in &channel.samples {
            for modifier in &sample.modifiers {
                let mod_type = match modifier {
                    Modifier::NormFactor { .. } => "normfactor",
                    Modifier::NormSys { .. } => "normsys",
                    Modifier::HistoSys { .. } => "histosys",
                    Modifier::ShapeSys { .. } => "shapesys",
                    Modifier::ShapeFactor { .. } => "shapefactor",
                    Modifier::StatError { .. } => "staterror",
                    Modifier::Lumi { .. } => "lumi",
                    Modifier::Unknown(_) => "unknown",
                };
                found_modifiers.insert(mod_type);
            }
        }
    }

    // Verify we have multiple modifier types
    assert!(found_modifiers.contains("normfactor"));
    assert!(found_modifiers.contains("lumi"));
    assert!(found_modifiers.contains("normsys"));
    assert!(found_modifiers.contains("histosys"));
    assert!(found_modifiers.contains("staterror"));
    assert!(found_modifiers.contains("shapefactor"));
}

#[test]
fn test_serde_roundtrip() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    // Serialize back to JSON
    let serialized = serde_json::to_string_pretty(&ws).unwrap();

    // Parse again
    let ws2: Workspace = serde_json::from_str(&serialized).unwrap();

    // Check basic properties are preserved
    assert_eq!(ws.channels.len(), ws2.channels.len());
    assert_eq!(ws.observations.len(), ws2.observations.len());
    assert_eq!(ws.measurements.len(), ws2.measurements.len());
}

#[test]
fn test_parameter_config() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    // Check lumi parameter config
    let params = &ws.measurements[0].config.parameters;
    assert!(!params.is_empty());

    let lumi_param = params.iter().find(|p| p.name == "lumi");
    assert!(lumi_param.is_some());

    let lumi = lumi_param.unwrap();
    assert_eq!(lumi.inits, vec![1.0]);
    assert_eq!(lumi.auxdata, vec![1.0]);
    assert_eq!(lumi.sigmas, vec![0.02]);
}

#[test]
fn test_parameter_fixed_is_parsed_and_applied_as_bounds_clamp() {
    use ns_core::traits::LogDensityModel;

    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "s",
          "data": [1.0],
          "modifiers": [
            {"name": "lumi", "type": "lumi", "data": null}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [1.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": [
          {
            "name": "lumi",
            "inits": [1.0],
            "bounds": [[0.5, 1.5]],
            "auxdata": [1.0],
            "sigmas": [0.1],
            "fixed": true
          }
        ]
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let cfg = ws
        .measurements
        .first()
        .and_then(|m| m.config.parameters.first())
        .expect("parameter config present");
    assert!(cfg.fixed, "expected fixed=true");

    let model = super::HistFactoryModel::from_workspace(&ws).expect("build model");
    let names = model.parameter_names();
    let bounds = model.parameter_bounds();
    let lumi_idx = names.iter().position(|n| n == "lumi").expect("lumi param exists");
    assert_eq!(bounds[lumi_idx], (1.0, 1.0));
}

#[test]
fn test_model_rejects_observation_length_mismatch() {
    let json = include_str!("../../../../tests/fixtures/bad_observations_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("observations") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_model_rejects_sample_length_mismatch() {
    let json = include_str!("../../../../tests/fixtures/bad_sample_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("sample") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_model_rejects_histosys_template_length_mismatch() {
    let json =
        include_str!("../../../../tests/fixtures/bad_histosys_template_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("histosys") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_constraintterm_lognormal_applies_root_alpha_transform_to_normsys() {
    use ns_core::traits::LogDensityModel;

    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "s",
          "data": [1.0],
          "modifiers": [
            {"name": "syst2", "type": "normsys", "data": {"hi": 1.05, "lo": 0.95}}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [1.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": [
          {"name": "syst2", "constraint": {"type": "LogNormal", "rel_uncertainty": 0.3}}
        ]
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let model = super::HistFactoryModel::from_workspace_with_settings(
        &ws,
        super::NormSysInterpCode::Code1,
        super::HistoSysInterpCode::Code0,
    )
    .expect("build model");

    let names = model.parameter_names();
    let mu_idx = names.iter().position(|n| n == "mu").expect("mu param");
    let syst_idx = names.iter().position(|n| n == "syst2").expect("syst2 param");

    let mut params = model.parameter_init();
    params[mu_idx] = 1.0;
    params[syst_idx] = 2.0; // alpha

    let exp = model.expected_data(&params).expect("expected_data");
    assert_eq!(exp.len(), 1);

    // ROOT alphaOfBeta transform for LogNormal:
    // alpha_eff = (1/rel) * ((1+rel)^alpha - 1)
    let rel = 0.3_f64;
    let alpha = 2.0_f64;
    let alpha_eff = (1.0 / rel) * ((1.0 + rel).powf(alpha) - 1.0);
    let want = 1.0_f64 * 1.05_f64.powf(alpha_eff);
    assert!((exp[0] - want).abs() < 1e-12, "got={} want={}", exp[0], want);

    // Smoke-check NLL is finite.
    let nll = model.nll(&params).expect("nll");
    assert!(nll.is_finite());
}

#[test]
fn test_constraintterm_gamma_adds_gamma_penalty_and_maps_beta_to_alpha_for_normsys() {
    use ns_core::traits::LogDensityModel;
    use statrs::function::gamma::ln_gamma;

    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "s",
          "data": [1.0],
          "modifiers": [
            {"name": "syst2", "type": "normsys", "data": {"hi": 1.05, "lo": 0.95}}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [1.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": [
          {"name": "syst2", "constraint": {"type": "Gamma", "rel_uncertainty": 0.3}}
        ]
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let model = super::HistFactoryModel::from_workspace_with_settings(
        &ws,
        super::NormSysInterpCode::Code1,
        super::HistoSysInterpCode::Code0,
    )
    .expect("build model");

    let names = model.parameter_names();
    let mu_idx = names.iter().position(|n| n == "mu").expect("mu param");
    let syst_idx = names.iter().position(|n| n == "syst2").expect("syst2 param");

    // Beta parameter; choose beta=1.3 so alpha=(beta-1)/rel == 1.0.
    let rel = 0.3_f64;
    let beta = 1.0 + rel;
    let mut params = model.parameter_init();
    params[mu_idx] = 1.0;
    params[syst_idx] = beta;

    let exp = model.expected_data(&params).expect("expected_data");
    let want = 1.0_f64 * 1.05_f64.powf(1.0);
    assert!((exp[0] - want).abs() < 1e-12, "got={} want={}", exp[0], want);

    // At beta=1.0 (alpha=0), main-bin Poisson NLL for obs=1, exp=1 is exactly 1.
    params[syst_idx] = 1.0;
    let nll = model.nll(&params).expect("nll");
    let gamma_term = nll - 1.0;

    // ROOT/HistFactory gamma constraint:
    // tau = 1/rel^2, k = tau + 1, theta = 1/tau (scale)
    // NLL = beta/theta - (k-1)*ln(beta) + k*ln(theta) + lnGamma(k)
    let tau = 1.0 / (rel * rel);
    let k = tau + 1.0;
    let theta = 1.0 / tau;
    let beta = 1.0_f64;
    let want_gamma = beta / theta - (k - 1.0) * beta.ln() + k * theta.ln() + ln_gamma(k);
    assert!(
        (gamma_term - want_gamma).abs() < 1e-10,
        "gamma_term={} want_gamma={}",
        gamma_term,
        want_gamma
    );
}

/// f32 vs f64 precision PoC for Metal GPU feasibility.
///
/// Tests whether f32 NLL computation produces results accurate enough
/// for L-BFGS-B optimization on real HistFactory models.
///
/// Run with: `cargo test -p ns-translate --release test_f32_precision_poc -- --nocapture`
#[test]
fn test_f32_precision_poc() {
    use ns_ad::scalar::Scalar;
    use ns_core::traits::LogDensityModel;

    println!("\n================================================================================");
    println!("f32 vs f64 Precision PoC -- Metal GPU Feasibility");
    println!("================================================================================");

    // ---- Test 1: Simple workspace (2 bins, 2 params) ----
    {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = super::HistFactoryModel::from_workspace(&ws).unwrap();
        let n = model.n_params();
        let init = model.parameter_init();

        println!("\n--- Simple workspace ({n} params) ---");

        let params_f64: Vec<f64> = init.clone();
        let params_f32: Vec<f32> = init.iter().map(|&x| x as f32).collect();

        let nll_f64: f64 = model.nll_generic(&params_f64).unwrap();
        let nll_f32: f32 = model.nll_generic(&params_f32).unwrap();

        let abs_diff = (nll_f64 - nll_f32.value()).abs();
        let rel_diff = abs_diff / nll_f64.abs();

        println!("  NLL f64:     {nll_f64:.15}");
        println!("  NLL f32:     {:.15}", nll_f32.value());
        println!("  Abs diff:    {abs_diff:.6e}");
        println!("  Rel diff:    {rel_diff:.6e}");

        // f32 gradient via finite differences
        let eps = 1e-4_f64;
        let mut grad_f32_fd = vec![0.0f64; n];
        for i in 0..n {
            let mut p_plus: Vec<f32> = params_f32.clone();
            p_plus[i] += eps as f32;
            let mut p_minus: Vec<f32> = params_f32.clone();
            p_minus[i] -= eps as f32;
            let nll_plus: f32 = model.nll_generic(&p_plus).unwrap();
            let nll_minus: f32 = model.nll_generic(&p_minus).unwrap();
            grad_f32_fd[i] = (nll_plus.value() - nll_minus.value()) / (2.0 * eps);
        }

        // f64 gradient via finite differences (reference)
        let eps_f64 = 1e-8_f64;
        let mut grad_f64_fd = vec![0.0f64; n];
        for i in 0..n {
            let mut p_plus = params_f64.clone();
            p_plus[i] += eps_f64;
            let mut p_minus = params_f64.clone();
            p_minus[i] -= eps_f64;
            let nll_plus: f64 = model.nll_generic(&p_plus).unwrap();
            let nll_minus: f64 = model.nll_generic(&p_minus).unwrap();
            grad_f64_fd[i] = (nll_plus - nll_minus) / (2.0 * eps_f64);
        }

        println!("  Gradient (f64 fd vs f32 fd):");
        let mut max_grad_diff = 0.0f64;
        for i in 0..n {
            let diff = (grad_f64_fd[i] - grad_f32_fd[i]).abs();
            let rel = if grad_f64_fd[i].abs() > 1e-10 { diff / grad_f64_fd[i].abs() } else { diff };
            max_grad_diff = max_grad_diff.max(rel);
            println!(
                "    param[{i}]: f64={:.8e}  f32={:.8e}  rel_diff={rel:.4e}",
                grad_f64_fd[i], grad_f32_fd[i]
            );
        }
        println!("  Max gradient rel diff: {max_grad_diff:.4e}");
    }

    // ---- Test 2: tHu workspace (184 params, NLL~179) ----
    {
        let json = include_str!("../../../../tests/fixtures/workspace_tHu.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = super::HistFactoryModel::from_workspace(&ws).unwrap();
        let n = model.n_params();
        let init = model.parameter_init();
        let bounds = model.parameter_bounds();

        println!("\n--- tHu workspace ({n} params) ---");

        // Test at init params
        let params_f64: Vec<f64> = init.clone();
        let params_f32: Vec<f32> = init.iter().map(|&x| x as f32).collect();

        let nll_f64_init: f64 = model.nll_generic(&params_f64).unwrap();
        let nll_f32_init: f32 = model.nll_generic(&params_f32).unwrap();

        let abs_diff_init = (nll_f64_init - nll_f32_init.value()).abs();
        let rel_diff_init = abs_diff_init / nll_f64_init.abs();

        println!("  At init params:");
        println!("    NLL f64:   {nll_f64_init:.15}");
        println!("    NLL f32:   {:.15}", nll_f32_init.value());
        println!("    Abs diff:  {abs_diff_init:.6e}");
        println!("    Rel diff:  {rel_diff_init:.6e}");

        // Test at perturbed params (simulating optimizer mid-iteration)
        let mut rng_state = 42u64;
        let mut perturbed = init.clone();
        for p in &mut perturbed {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 0.2 - 0.1;
            *p = (*p + noise).max(0.01);
        }

        let perturbed_f32: Vec<f32> = perturbed.iter().map(|&x| x as f32).collect();
        let nll_f64_pert: f64 = model.nll_generic(&perturbed).unwrap();
        let nll_f32_pert: f32 = model.nll_generic(&perturbed_f32).unwrap();

        let abs_diff_pert = (nll_f64_pert - nll_f32_pert.value()).abs();
        let rel_diff_pert = abs_diff_pert / nll_f64_pert.abs();

        println!("  At perturbed params:");
        println!("    NLL f64:   {nll_f64_pert:.15}");
        println!("    NLL f32:   {:.15}", nll_f32_pert.value());
        println!("    Abs diff:  {abs_diff_pert:.6e}");
        println!("    Rel diff:  {rel_diff_pert:.6e}");

        // Gradient comparison (finite differences, 10 evenly-spaced params)
        println!("  Gradient comparison (10 params):");
        let eps = 1e-4_f64;
        let eps_f64 = 1e-8_f64;
        let mut max_grad_rel_diff = 0.0f64;
        let mut grad_rel_diffs: Vec<f64> = Vec::new();

        let test_indices: Vec<usize> = (0..10).map(|i| i * n / 10).collect();
        for &idx in &test_indices {
            // f32 finite-diff gradient
            let mut p_plus_f32: Vec<f32> = perturbed_f32.clone();
            p_plus_f32[idx] += eps as f32;
            let mut p_minus_f32: Vec<f32> = perturbed_f32.clone();
            p_minus_f32[idx] -= eps as f32;
            let nll_p: f32 = model.nll_generic(&p_plus_f32).unwrap();
            let nll_m: f32 = model.nll_generic(&p_minus_f32).unwrap();
            let grad_f32_val = (nll_p.value() - nll_m.value()) / (2.0 * eps);

            // f64 finite-diff gradient (reference)
            let mut p_plus_f64 = perturbed.clone();
            p_plus_f64[idx] += eps_f64;
            let mut p_minus_f64 = perturbed.clone();
            p_minus_f64[idx] -= eps_f64;
            let nll_p64: f64 = model.nll_generic(&p_plus_f64).unwrap();
            let nll_m64: f64 = model.nll_generic(&p_minus_f64).unwrap();
            let grad_f64_val = (nll_p64 - nll_m64) / (2.0 * eps_f64);

            let rel = if grad_f64_val.abs() > 1e-10 {
                (grad_f64_val - grad_f32_val).abs() / grad_f64_val.abs()
            } else {
                (grad_f64_val - grad_f32_val).abs()
            };
            max_grad_rel_diff = max_grad_rel_diff.max(rel);
            grad_rel_diffs.push(rel);

            println!(
                "    param[{idx:3}]: f64={:+.8e}  f32={:+.8e}  rel_diff={rel:.4e}",
                grad_f64_val, grad_f32_val
            );
        }

        grad_rel_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_grad_diff = grad_rel_diffs[grad_rel_diffs.len() / 2];

        println!("  Max gradient rel diff:    {max_grad_rel_diff:.4e}");
        println!("  Median gradient rel diff: {median_grad_diff:.4e}");

        // ---- Mini optimization comparison: 5 steps of gradient descent ----
        // (Only 5 steps because 184-param FD gradient is slow in debug)
        println!("\n  Mini optimization (5 gradient descent steps, lr=0.01):");
        let lr = 0.01;
        let n_steps = 5;

        // f64 path
        let mut x_f64 = init.clone();
        for step in 0..n_steps {
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut p_plus = x_f64.clone();
                p_plus[i] += eps_f64;
                let mut p_minus = x_f64.clone();
                p_minus[i] -= eps_f64;
                let np: f64 = model.nll_generic(&p_plus).unwrap();
                let nm: f64 = model.nll_generic(&p_minus).unwrap();
                grad[i] = (np - nm) / (2.0 * eps_f64);
            }
            for i in 0..n {
                x_f64[i] -= lr * grad[i];
                x_f64[i] = x_f64[i].clamp(bounds[i].0, bounds[i].1);
            }
            let nll: f64 = model.nll_generic(&x_f64).unwrap();
            println!("    f64 step {step}: NLL = {nll:.10}");
        }

        // f32 path
        let mut x_f32: Vec<f32> = init.iter().map(|&x| x as f32).collect();
        for step in 0..n_steps {
            let mut grad = vec![0.0f32; n];
            for i in 0..n {
                let mut p_plus = x_f32.clone();
                p_plus[i] += eps as f32;
                let mut p_minus = x_f32.clone();
                p_minus[i] -= eps as f32;
                let np: f32 = model.nll_generic(&p_plus).unwrap();
                let nm: f32 = model.nll_generic(&p_minus).unwrap();
                grad[i] = (np - nm) / (2.0 * eps as f32);
            }
            for i in 0..n {
                x_f32[i] -= lr as f32 * grad[i];
                x_f32[i] = x_f32[i].clamp(bounds[i].0 as f32, bounds[i].1 as f32);
            }
            // Evaluate in f64 for fair comparison
            let x_as_f64: Vec<f64> = x_f32.iter().map(|&x| x as f64).collect();
            let nll: f64 = model.nll_generic(&x_as_f64).unwrap();
            println!("    f32 step {step}: NLL = {nll:.10} (eval in f64)");
        }

        // Compare final params
        let x_f32_as_f64: Vec<f64> = x_f32.iter().map(|&x| x as f64).collect();
        let nll_final_f64: f64 = model.nll_generic(&x_f64).unwrap();
        let nll_final_f32: f64 = model.nll_generic(&x_f32_as_f64).unwrap();
        let mut max_param_diff = 0.0f64;
        for i in 0..n {
            let diff = (x_f64[i] - x_f32[i] as f64).abs();
            max_param_diff = max_param_diff.max(diff);
        }

        println!("  Final comparison after {n_steps} steps:");
        println!("    NLL f64:          {nll_final_f64:.10}");
        println!("    NLL f32 (->f64):  {nll_final_f32:.10}");
        println!("    NLL diff:         {:.6e}", (nll_final_f64 - nll_final_f32).abs());
        println!("    Max param diff:   {max_param_diff:.6e}");
    }

    println!("\n================================================================================");
    println!("PoC complete. See numbers above for Metal f32 feasibility decision.");
    println!("================================================================================\n");
}

/// Dual32 analytical gradient vs Dual (f64) analytical gradient.
///
/// This is the critical Metal GPU feasibility test: Metal computes analytical
/// gradients (not finite differences), so what matters is Dual32 accuracy,
/// not f32 FD accuracy.
///
/// Run with: `cargo test -p ns-translate --release test_f32_analytical_gradient_poc -- --nocapture`
#[test]
fn test_f32_analytical_gradient_poc() {
    use ns_ad::dual::Dual;
    use ns_ad::dual32::Dual32;
    use ns_core::traits::LogDensityModel;

    println!("\n================================================================================");
    println!("Dual32 (f32 analytical) vs Dual (f64 analytical) Gradient PoC");
    println!("================================================================================");

    // ---- Test 1: Simple workspace (2 params) ----
    {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = super::HistFactoryModel::from_workspace(&ws).unwrap();
        let n = model.n_params();
        let init = model.parameter_init();

        println!("\n--- Simple workspace ({n} params) ---");

        // f64 analytical gradient (reference)
        let mut grad_f64 = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual: Vec<Dual> = init.iter().map(|&x| Dual::constant(x)).collect();
            params_dual[i] = Dual::var(init[i]);
            let nll: Dual = model.nll_generic(&params_dual).unwrap();
            grad_f64[i] = nll.dot;
        }

        // f32 analytical gradient (what Metal would compute)
        let mut grad_f32 = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual32: Vec<Dual32> =
                init.iter().map(|&x| Dual32::constant(x as f32)).collect();
            params_dual32[i] = Dual32::var(init[i] as f32);
            let nll: Dual32 = model.nll_generic(&params_dual32).unwrap();
            grad_f32[i] = nll.dot as f64;
        }

        println!("  Analytical gradient comparison:");
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let rel = if grad_f64[i].abs() > 1e-10 {
                (grad_f64[i] - grad_f32[i]).abs() / grad_f64[i].abs()
            } else {
                (grad_f64[i] - grad_f32[i]).abs()
            };
            max_rel = max_rel.max(rel);
            println!(
                "    param[{i}]: f64={:+.12e}  f32={:+.12e}  rel={rel:.6e}",
                grad_f64[i], grad_f32[i]
            );
        }
        println!("  Max analytical grad rel diff: {max_rel:.6e}");
    }

    // ---- Test 2: tHu workspace (184 params) ----
    {
        let json = include_str!("../../../../tests/fixtures/workspace_tHu.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = super::HistFactoryModel::from_workspace(&ws).unwrap();
        let n = model.n_params();
        let init = model.parameter_init();

        println!("\n--- tHu workspace ({n} params) ---");

        // NLL values first (for reference)
        {
            use ns_ad::scalar::Scalar;
            let nll_f64: f64 = model.nll_generic(&init).unwrap();
            let p32: Vec<f32> = init.iter().map(|&x| x as f32).collect();
            let nll_f32: f32 = model.nll_generic(&p32).unwrap();
            let nll_f32_value = nll_f32.value();
            println!("  NLL f64: {nll_f64:.15}");
            println!("  NLL f32: {nll_f32_value:.15}");
            println!("  NLL rel diff: {:.6e}", (nll_f64 - nll_f32_value).abs() / nll_f64.abs());
        }

        // Full gradient: all 184 params
        println!("\n  Computing full f64 analytical gradient ({n} params)...");
        let mut grad_f64 = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual: Vec<Dual> = init.iter().map(|&x| Dual::constant(x)).collect();
            params_dual[i] = Dual::var(init[i]);
            let nll: Dual = model.nll_generic(&params_dual).unwrap();
            grad_f64[i] = nll.dot;
        }

        println!("  Computing full f32 analytical gradient ({n} params)...");
        let mut grad_f32 = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual32: Vec<Dual32> =
                init.iter().map(|&x| Dual32::constant(x as f32)).collect();
            params_dual32[i] = Dual32::var(init[i] as f32);
            let nll: Dual32 = model.nll_generic(&params_dual32).unwrap();
            grad_f32[i] = nll.dot as f64;
        }

        // Collect statistics
        let mut rel_diffs: Vec<(usize, f64, f64, f64, f64)> = Vec::new(); // (idx, f64_grad, f32_grad, abs_diff, rel_diff)
        for i in 0..n {
            let abs_diff = (grad_f64[i] - grad_f32[i]).abs();
            let rel =
                if grad_f64[i].abs() > 1e-10 { abs_diff / grad_f64[i].abs() } else { abs_diff };
            rel_diffs.push((i, grad_f64[i], grad_f32[i], abs_diff, rel));
        }

        // Sort by rel diff descending to show worst cases
        rel_diffs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

        println!("\n  Top 20 worst analytical gradient components:");
        for &(idx, g64, g32, _abs, rel) in rel_diffs.iter().take(20) {
            let sign_match = if g64.signum() == g32.signum() { " " } else { "!" };
            println!(
                "    {sign_match} param[{idx:3}]: f64={g64:+.8e}  f32={g32:+.8e}  rel={rel:.6e}"
            );
        }

        // Overall stats
        let all_rels: Vec<f64> = rel_diffs.iter().map(|r| r.4).collect();
        let max_rel = all_rels.iter().cloned().fold(0.0f64, f64::max);
        let mean_rel = all_rels.iter().sum::<f64>() / all_rels.len() as f64;
        let mut sorted_rels = all_rels.clone();
        sorted_rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_rel = sorted_rels[sorted_rels.len() / 2];
        let p90_rel = sorted_rels[(sorted_rels.len() as f64 * 0.9) as usize];
        let p99_rel = sorted_rels[(sorted_rels.len() as f64 * 0.99) as usize];

        // Sign disagreements
        let sign_mismatches: Vec<_> =
            rel_diffs.iter().filter(|r| r.1.signum() != r.2.signum() && r.1.abs() > 1e-6).collect();

        println!("\n  Analytical gradient statistics (all {n} params):");
        println!("    Max rel diff:    {max_rel:.6e}");
        println!("    Mean rel diff:   {mean_rel:.6e}");
        println!("    Median rel diff: {median_rel:.6e}");
        println!("    P90 rel diff:    {p90_rel:.6e}");
        println!("    P99 rel diff:    {p99_rel:.6e}");
        println!("    Sign mismatches: {} / {n} (where |grad_f64| > 1e-6)", sign_mismatches.len());

        // ---- Test at perturbed point ----
        println!("\n  --- At perturbed point ---");
        let mut rng_state = 42u64;
        let mut perturbed = init.clone();
        for p in &mut perturbed {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 0.2 - 0.1;
            *p = (*p + noise).max(0.01);
        }

        let mut grad_f64_pert = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual: Vec<Dual> = perturbed.iter().map(|&x| Dual::constant(x)).collect();
            params_dual[i] = Dual::var(perturbed[i]);
            let nll: Dual = model.nll_generic(&params_dual).unwrap();
            grad_f64_pert[i] = nll.dot;
        }

        let mut grad_f32_pert = vec![0.0f64; n];
        for i in 0..n {
            let mut params_dual32: Vec<Dual32> =
                perturbed.iter().map(|&x| Dual32::constant(x as f32)).collect();
            params_dual32[i] = Dual32::var(perturbed[i] as f32);
            let nll: Dual32 = model.nll_generic(&params_dual32).unwrap();
            grad_f32_pert[i] = nll.dot as f64;
        }

        let mut rel_diffs_pert: Vec<f64> = Vec::new();
        let mut sign_mismatch_pert = 0usize;
        for i in 0..n {
            let abs_diff = (grad_f64_pert[i] - grad_f32_pert[i]).abs();
            let rel = if grad_f64_pert[i].abs() > 1e-10 {
                abs_diff / grad_f64_pert[i].abs()
            } else {
                abs_diff
            };
            rel_diffs_pert.push(rel);
            if grad_f64_pert[i].signum() != grad_f32_pert[i].signum()
                && grad_f64_pert[i].abs() > 1e-6
            {
                sign_mismatch_pert += 1;
            }
        }

        rel_diffs_pert.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_rel_pert = rel_diffs_pert.iter().cloned().fold(0.0f64, f64::max);
        let median_rel_pert = rel_diffs_pert[rel_diffs_pert.len() / 2];
        let p99_rel_pert = rel_diffs_pert[(rel_diffs_pert.len() as f64 * 0.99) as usize];

        println!("  Perturbed gradient statistics:");
        println!("    Max rel diff:    {max_rel_pert:.6e}");
        println!("    Median rel diff: {median_rel_pert:.6e}");
        println!("    P99 rel diff:    {p99_rel_pert:.6e}");
        println!("    Sign mismatches: {sign_mismatch_pert} / {n} (where |grad_f64| > 1e-6)");

        // Verdict
        println!("\n  ================================================================");
        if max_rel < 0.01 && sign_mismatches.is_empty() {
            println!("  VERDICT: f32 analytical gradients SUFFICIENT for Metal GPU");
            println!("  Max error < 1%, no sign flips. L-BFGS-B should converge.");
        } else if max_rel < 0.1 && sign_mismatches.len() <= 2 {
            println!("  VERDICT: f32 analytical gradients MARGINAL for Metal GPU");
            println!("  Some components have >1% error. May need Kahan or mixed precision.");
        } else {
            println!("  VERDICT: f32 analytical gradients INSUFFICIENT for Metal GPU");
            println!("  Large errors or sign flips. Need f64 accumulation or different approach.");
        }
        println!("  ================================================================");
    }

    println!("\n================================================================================");
    println!("Dual32 analytical gradient PoC complete.");
    println!("================================================================================\n");
}

#[test]
fn test_serialize_for_gpu_rejects_non_positive_normsys() {
    // Workspace with NormSys hi=-1.0 (non-positive) — GPU serialization must reject.
    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "sig",
          "data": [5.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null}
          ]
        },
        {
          "name": "bkg",
          "data": [50.0],
          "modifiers": [
            {"name": "syst", "type": "normsys", "data": {"hi": -1.0, "lo": 0.5}}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [55.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": []
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let model = super::HistFactoryModel::from_workspace(&ws).expect("build model");

    let err = model.serialize_for_gpu().unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("non-positive factor") && msg.contains("gpu"),
        "expected GPU non-positive factor error, got: {}",
        msg
    );
}

#[test]
fn test_serialize_for_gpu_rejects_zero_lo_normsys() {
    // lo=0.0 is also non-positive — must be rejected.
    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "sig",
          "data": [5.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null}
          ]
        },
        {
          "name": "bkg",
          "data": [50.0],
          "modifiers": [
            {"name": "syst", "type": "normsys", "data": {"hi": 1.2, "lo": 0.0}}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [55.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": []
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let model = super::HistFactoryModel::from_workspace(&ws).expect("build model");

    let err = model.serialize_for_gpu().unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("non-positive factor"),
        "expected non-positive factor error, got: {}",
        msg
    );
}

#[test]
fn test_normsys_cpu_fallback_non_positive_factors() {
    use ns_core::traits::LogDensityModel;

    // Workspace with non-positive NormSys factor: hi=-1.0, lo=0.5.
    // CPU path uses piecewise-linear fallback. Verify expected_data correctness.
    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "bkg",
          "data": [50.0],
          "modifiers": [
            {"name": "syst", "type": "normsys", "data": {"hi": -1.0, "lo": 0.5}}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [55.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": []
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let model = super::HistFactoryModel::from_workspace(&ws).expect("build model");

    let names = model.parameter_names();
    let syst_idx = names.iter().position(|n| n == "syst").expect("syst param");

    // Piecewise-linear fallback for code4 with non-positive factors:
    //   alpha >= 0: factor = 1 + alpha * (hi - 1) = 1 + alpha * (-2)
    //   alpha < 0:  factor = 1 - alpha * (1 - lo) = 1 + |alpha| * 0.5
    // nominal = 50.0, so expected = nominal * factor

    // alpha = 0.0 → factor = 1.0 → expected = 50.0
    let mut params = model.parameter_init();
    params[syst_idx] = 0.0;
    let exp = model.expected_data(&params).unwrap();
    assert!((exp[0] - 50.0).abs() < 1e-10, "alpha=0: expected 50.0, got {}", exp[0]);

    // alpha = 0.5 → factor = 1 + 0.5*(-2) = 0.0 → expected = 0.0
    params[syst_idx] = 0.5;
    let exp = model.expected_data(&params).unwrap();
    assert!((exp[0] - 0.0).abs() < 1e-10, "alpha=0.5: expected 0.0, got {}", exp[0]);

    // alpha = -0.5 → factor = 1 - (-0.5)*(1 - 0.5) = 1 + 0.25 = 1.25 → expected = 62.5
    params[syst_idx] = -0.5;
    let exp = model.expected_data(&params).unwrap();
    assert!((exp[0] - 62.5).abs() < 1e-10, "alpha=-0.5: expected 62.5, got {}", exp[0]);
}

// ---------------------------------------------------------------------------
// Workspace operations (G1–G5 pyhf parity)
// ---------------------------------------------------------------------------

fn make_two_channel_workspace() -> Workspace {
    serde_json::from_value(serde_json::json!({
        "channels": [
            {
                "name": "SR",
                "samples": [
                    { "name": "signal", "data": [5.0, 10.0], "modifiers": [
                        { "name": "mu", "type": "normfactor", "data": null }
                    ]},
                    { "name": "background", "data": [50.0, 60.0], "modifiers": [
                        { "name": "syst1", "type": "normsys", "data": { "hi": 1.1, "lo": 0.9 } }
                    ]}
                ]
            },
            {
                "name": "CR",
                "samples": [
                    { "name": "background", "data": [100.0], "modifiers": [
                        { "name": "syst1", "type": "normsys", "data": { "hi": 1.05, "lo": 0.95 } }
                    ]}
                ]
            }
        ],
        "observations": [
            { "name": "SR", "data": [53.0, 65.0] },
            { "name": "CR", "data": [102.0] }
        ],
        "measurements": [
            { "name": "meas1", "config": { "poi": "mu", "parameters": [] } }
        ]
    }))
    .unwrap()
}

#[test]
fn test_workspace_prune_channel() {
    let ws = make_two_channel_workspace();
    let pruned = ws.prune(&["CR"], &[], &[], &[]);

    assert_eq!(pruned.channels.len(), 1);
    assert_eq!(pruned.channels[0].name, "SR");
    assert_eq!(pruned.observations.len(), 1);
    assert_eq!(pruned.observations[0].name, "SR");
    assert_eq!(pruned.measurements.len(), 1);
}

#[test]
fn test_workspace_prune_sample() {
    let ws = make_two_channel_workspace();
    let pruned = ws.prune(&[], &["signal"], &[], &[]);

    assert_eq!(pruned.channels.len(), 2);
    let sr = pruned.channels.iter().find(|c| c.name == "SR").unwrap();
    assert_eq!(sr.samples.len(), 1);
    assert_eq!(sr.samples[0].name, "background");
}

#[test]
fn test_workspace_prune_modifier() {
    let ws = make_two_channel_workspace();
    let pruned = ws.prune(&[], &[], &["syst1"], &[]);

    for ch in &pruned.channels {
        for s in &ch.samples {
            assert!(
                s.modifiers.iter().all(|m| m.name() != "syst1"),
                "syst1 should be pruned from {}:{}",
                ch.name,
                s.name
            );
        }
    }
}

#[test]
fn test_workspace_prune_measurement() {
    let ws = make_two_channel_workspace();
    let pruned = ws.prune(&[], &[], &[], &["meas1"]);
    assert_eq!(pruned.measurements.len(), 0);
}

#[test]
fn test_workspace_sorted_is_idempotent() {
    let ws = make_two_channel_workspace();
    let sorted1 = ws.sorted();
    let sorted2 = sorted1.sorted();

    let j1 = serde_json::to_string(&sorted1).unwrap();
    let j2 = serde_json::to_string(&sorted2).unwrap();
    assert_eq!(j1, j2, "sorted() should be idempotent");
}

#[test]
fn test_workspace_sorted_orders_channels() {
    let ws = make_two_channel_workspace();
    let sorted = ws.sorted();

    assert_eq!(sorted.channels[0].name, "CR");
    assert_eq!(sorted.channels[1].name, "SR");
    assert_eq!(sorted.observations[0].name, "CR");
    assert_eq!(sorted.observations[1].name, "SR");
}

#[test]
fn test_workspace_digest_deterministic() {
    let ws = make_two_channel_workspace();
    let d1 = ws.digest();
    let d2 = ws.digest();
    assert_eq!(d1, d2);
    assert_eq!(d1.len(), 64, "SHA-256 hex should be 64 chars");
}

#[test]
fn test_workspace_digest_changes_on_modification() {
    let ws = make_two_channel_workspace();
    let d_orig = ws.digest();
    let pruned = ws.prune(&["CR"], &[], &[], &[]);
    let d_pruned = pruned.digest();
    assert_ne!(d_orig, d_pruned, "digest should change after prune");
}

#[test]
fn test_workspace_rename_channel() {
    use std::collections::HashMap;
    let ws = make_two_channel_workspace();

    let ch_map: HashMap<String, String> =
        [("SR".to_string(), "SignalRegion".to_string())].into_iter().collect();
    let empty: HashMap<String, String> = HashMap::new();

    let renamed = ws.rename(&ch_map, &empty, &empty, &empty);

    assert!(renamed.channels.iter().any(|c| c.name == "SignalRegion"));
    assert!(!renamed.channels.iter().any(|c| c.name == "SR"));
    assert!(renamed.observations.iter().any(|o| o.name == "SignalRegion"));
}

#[test]
fn test_workspace_rename_modifier() {
    use std::collections::HashMap;
    let ws = make_two_channel_workspace();

    let empty: HashMap<String, String> = HashMap::new();
    let mod_map: HashMap<String, String> =
        [("syst1".to_string(), "alpha_syst1".to_string())].into_iter().collect();

    let renamed = ws.rename(&empty, &empty, &mod_map, &empty);

    for ch in &renamed.channels {
        for s in &ch.samples {
            for m in &s.modifiers {
                if m.modifier_type() == "normsys" {
                    assert_eq!(m.name(), "alpha_syst1");
                }
            }
        }
    }
}

#[test]
fn test_workspace_combine_no_overlap() {
    let ws1 = make_two_channel_workspace();
    let ws2: Workspace = serde_json::from_value(serde_json::json!({
        "channels": [
            {
                "name": "VR",
                "samples": [
                    { "name": "background", "data": [200.0], "modifiers": [] }
                ]
            }
        ],
        "observations": [{ "name": "VR", "data": [198.0] }],
        "measurements": [
            { "name": "meas2", "config": { "poi": "mu", "parameters": [] } }
        ]
    }))
    .unwrap();

    let combined = ws1.combine(&ws2, CombineJoin::None).unwrap();
    assert_eq!(combined.channels.len(), 3);
    assert_eq!(combined.observations.len(), 3);
    assert_eq!(combined.measurements.len(), 2);
}

#[test]
fn test_workspace_combine_overlap_none_errors() {
    let ws = make_two_channel_workspace();
    let result = ws.combine(&ws, CombineJoin::None);
    assert!(result.is_err());
}

#[test]
fn test_workspace_combine_left_outer() {
    let ws1 = make_two_channel_workspace();
    let mut ws2 = make_two_channel_workspace();
    ws2.channels[0].samples[0].data = vec![999.0, 999.0];

    let combined = ws1.combine(&ws2, CombineJoin::LeftOuter).unwrap();
    let sr = combined.channels.iter().find(|c| c.name == "SR").unwrap();
    assert_eq!(sr.samples[0].data, vec![5.0, 10.0], "left outer should keep self's data");
}

#[test]
fn test_workspace_combine_right_outer() {
    let ws1 = make_two_channel_workspace();
    let mut ws2 = make_two_channel_workspace();
    ws2.channels[0].samples[0].data = vec![999.0, 999.0];

    let combined = ws1.combine(&ws2, CombineJoin::RightOuter).unwrap();
    let sr = combined.channels.iter().find(|c| c.name == "SR").unwrap();
    assert_eq!(sr.samples[0].data, vec![999.0, 999.0], "right outer should keep other's data");
}
