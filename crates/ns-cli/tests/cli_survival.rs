use std::path::PathBuf;
use std::process::{Command, Output};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn assert_monotone_non_decreasing(xs: &[f64]) {
    for i in 1..xs.len() {
        assert!(xs[i] >= xs[i - 1] - 1e-12, "sequence not monotone at i={i}");
    }
}

#[test]
fn survival_cox_ph_fit_contract_cluster_robust_default() {
    let input = fixture_path("survival_cox_small.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&["survival", "cox-ph-fit", "--input", input.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "survival cox-ph-fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("model").and_then(|x| x.as_str()).unwrap(), "cox_ph");
    assert_eq!(v.get("ties").and_then(|x| x.as_str()).unwrap(), "efron");

    let coef = v.get("coef").and_then(|x| x.as_array()).expect("coef should be array");
    assert_eq!(coef.len(), 2);
    for c in coef {
        assert!(c.as_f64().unwrap().is_finite());
    }

    let se = v.get("se").and_then(|x| x.as_array()).expect("se should be array");
    assert_eq!(se.len(), 2);
    for s in se {
        let s = s.as_f64().unwrap();
        assert!(s.is_finite() && s >= 0.0);
    }

    assert_eq!(v.get("robust_kind").and_then(|x| x.as_str()).unwrap(), "cluster");
    let rse = v.get("robust_se").and_then(|x| x.as_array()).expect("robust_se should be array");
    assert_eq!(rse.len(), 2);
    for s in rse {
        let s = s.as_f64().unwrap();
        assert!(s.is_finite() && s >= 0.0);
    }

    let meta =
        v.get("robust_meta").and_then(|x| x.as_object()).expect("robust_meta should be object");
    assert!(meta.get("enabled").and_then(|x| x.as_bool()).unwrap());
    assert_eq!(meta.get("kind").and_then(|x| x.as_str()).unwrap(), "cluster");
    assert_eq!(meta.get("n_groups").and_then(|x| x.as_u64()).unwrap(), 3);

    let bt =
        v.get("baseline_times").and_then(|x| x.as_array()).expect("baseline_times should be array");
    let bh = v
        .get("baseline_cumhaz")
        .and_then(|x| x.as_array())
        .expect("baseline_cumhaz should be array");
    assert_eq!(bt.len(), bh.len());
    assert!(bt.len() >= 2);

    let bt_f: Vec<f64> = bt.iter().map(|x| x.as_f64().unwrap()).collect();
    let bh_f: Vec<f64> = bh.iter().map(|x| x.as_f64().unwrap()).collect();
    assert_monotone_non_decreasing(&bt_f);
    assert_monotone_non_decreasing(&bh_f);
}

#[test]
fn survival_cox_ph_fit_allows_disabling_robust() {
    let input = fixture_path("survival_cox_small.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "survival",
        "cox-ph-fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--no-robust",
    ]);
    assert!(
        out.status.success(),
        "survival cox-ph-fit --no-robust should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("robust_kind").unwrap().is_null());
    assert!(v.get("robust_se").unwrap().is_null());
    let meta =
        v.get("robust_meta").and_then(|x| x.as_object()).expect("robust_meta should be object");
    assert!(!meta.get("enabled").and_then(|x| x.as_bool()).unwrap());
}

#[test]
fn survival_cox_ph_fit_supports_breslow_ties() {
    let input = fixture_path("survival_cox_small.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "survival",
        "cox-ph-fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--ties",
        "breslow",
    ]);
    assert!(
        out.status.success(),
        "survival cox-ph-fit --ties breslow should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("ties").and_then(|x| x.as_str()).unwrap(), "breslow");
    let bt =
        v.get("baseline_times").and_then(|x| x.as_array()).expect("baseline_times should be array");
    assert!(bt.len() >= 2);
}

// ---------------------------------------------------------------------------
// Kaplan-Meier CLI tests
// ---------------------------------------------------------------------------

#[test]
fn survival_km_basic_output_contract() {
    let input = fixture_path("km_input.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&["survival", "km", "--input", input.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "survival km should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");

    assert_eq!(v.get("n").and_then(|x| x.as_u64()).unwrap(), 16);
    assert_eq!(v.get("n_events").and_then(|x| x.as_u64()).unwrap(), 11);
    assert!((v.get("conf_level").and_then(|x| x.as_f64()).unwrap() - 0.95).abs() < 1e-10);

    let median = v.get("median").and_then(|x| x.as_f64()).unwrap();
    assert!((median - 8.0).abs() < 1e-10, "median should be 8.0, got {median}");

    let steps = v.get("steps").and_then(|x| x.as_array()).expect("steps should be array");
    assert!(steps.len() >= 5);

    // Survival should be monotone non-increasing.
    let surv: Vec<f64> =
        steps.iter().map(|s| s.get("survival").unwrap().as_f64().unwrap()).collect();
    for i in 1..surv.len() {
        assert!(surv[i] <= surv[i - 1] + 1e-12, "survival not non-increasing at i={i}");
    }

    // First step: S(1) = 0.875
    let s1 = &steps[0];
    assert!((s1.get("survival").unwrap().as_f64().unwrap() - 0.875).abs() < 1e-10);
    assert_eq!(s1.get("n_risk").unwrap().as_u64().unwrap(), 16);
    assert_eq!(s1.get("n_events").unwrap().as_u64().unwrap(), 2);
}

#[test]
fn survival_km_custom_conf_level() {
    let input = fixture_path("km_input.json");
    assert!(input.exists());

    let out = run(&[
        "survival",
        "km",
        "--input",
        input.to_string_lossy().as_ref(),
        "--conf-level",
        "0.90",
    ]);
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));

    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!((v.get("conf_level").unwrap().as_f64().unwrap() - 0.90).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Log-rank test CLI tests
// ---------------------------------------------------------------------------

#[test]
fn survival_log_rank_basic_output_contract() {
    let input = fixture_path("logrank_input.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&["survival", "log-rank-test", "--input", input.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "survival log-rank-test should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");

    assert_eq!(v.get("n").and_then(|x| x.as_u64()).unwrap(), 23);
    assert_eq!(v.get("df").and_then(|x| x.as_u64()).unwrap(), 1);

    let chi2 = v.get("chi_squared").and_then(|x| x.as_f64()).unwrap();
    assert!((chi2 - 3.4).abs() < 0.15, "chi_squared: got {chi2}, expected ~3.4");

    let p = v.get("p_value").and_then(|x| x.as_f64()).unwrap();
    assert!(p > 0.05 && p < 0.10, "p_value: got {p}, expected ~0.065");

    let summaries = v.get("group_summaries").and_then(|x| x.as_array()).expect("group_summaries");
    assert_eq!(summaries.len(), 2);

    let o1 = summaries[0].get("observed").unwrap().as_f64().unwrap();
    let o2 = summaries[1].get("observed").unwrap().as_f64().unwrap();
    assert!((o1 - 7.0).abs() < 1e-10);
    assert!((o2 - 11.0).abs() < 1e-10);
}
