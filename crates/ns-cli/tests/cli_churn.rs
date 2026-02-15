use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn tmp_path(filename: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_churn_{}_{}_{}", std::process::id(), nanos, filename));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn churn_bootstrap_hr_bca_smoke() {
    let data = tmp_path("churn_data.json");
    let gen_out = run(&[
        "churn",
        "generate-data",
        "--n-customers",
        "180",
        "--seed",
        "7",
        "-o",
        data.to_string_lossy().as_ref(),
    ]);
    assert!(
        gen_out.status.success(),
        "churn generate-data failed, stderr={}",
        String::from_utf8_lossy(&gen_out.stderr)
    );

    let out = run(&[
        "churn",
        "bootstrap-hr",
        "-i",
        data.to_string_lossy().as_ref(),
        "--n-bootstrap",
        "20",
        "--seed",
        "11",
        "--ci-method",
        "bca",
        "--n-jackknife",
        "12",
    ]);
    assert!(
        out.status.success(),
        "churn bootstrap-hr --ci-method bca failed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("ci_method_requested").and_then(|x| x.as_str()), Some("bca"));
    assert_eq!(v.get("n_bootstrap").and_then(|x| x.as_u64()), Some(20));
    assert_eq!(v.get("n_jackknife_requested").and_then(|x| x.as_u64()), Some(12));
    assert!(v.get("n_converged").and_then(|x| x.as_u64()).unwrap_or(0) >= 2);

    let coefs =
        v.get("coefficients").and_then(|x| x.as_array()).expect("coefficients should be an array");
    assert!(!coefs.is_empty(), "coefficients should be non-empty");
    for c in coefs {
        assert!(c.get("name").and_then(|x| x.as_str()).is_some(), "missing coefficient name");
        let lower =
            c.get("hr_ci_lower").and_then(|x| x.as_f64()).expect("hr_ci_lower should be numeric");
        let upper =
            c.get("hr_ci_upper").and_then(|x| x.as_f64()).expect("hr_ci_upper should be numeric");
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
        assert!(
            matches!(c.get("ci_method").and_then(|x| x.as_str()), Some("bca" | "percentile")),
            "ci_method must be bca or percentile fallback"
        );
        let diag = c
            .get("ci_diagnostics")
            .and_then(|x| x.as_object())
            .expect("ci_diagnostics should be object");
        assert!(diag.get("requested_method").and_then(|x| x.as_str()).is_some());
        assert!(diag.get("effective_method").and_then(|x| x.as_str()).is_some());
    }

    let _ = std::fs::remove_file(&data);
}

#[test]
fn churn_bootstrap_hr_default_is_percentile() {
    let data = tmp_path("churn_data_default_method.json");
    let gen_out = run(&[
        "churn",
        "generate-data",
        "--n-customers",
        "120",
        "--seed",
        "17",
        "-o",
        data.to_string_lossy().as_ref(),
    ]);
    assert!(
        gen_out.status.success(),
        "churn generate-data failed, stderr={}",
        String::from_utf8_lossy(&gen_out.stderr)
    );

    let out = run(&[
        "churn",
        "bootstrap-hr",
        "-i",
        data.to_string_lossy().as_ref(),
        "--n-bootstrap",
        "16",
        "--seed",
        "13",
    ]);
    assert!(
        out.status.success(),
        "churn bootstrap-hr default failed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("ci_method_requested").and_then(|x| x.as_str()), Some("percentile"));

    let _ = std::fs::remove_file(&data);
}
