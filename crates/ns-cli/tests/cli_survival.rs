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
