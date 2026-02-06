use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root_from_manifest_dir() -> PathBuf {
    // crates/ns-cli -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root should exist")
}

fn fixture_path() -> PathBuf {
    repo_root_from_manifest_dir().join("tests").join("fixtures").join("simple_workspace.json")
}

fn run_nextstat_json(args: &[&str]) -> serde_json::Value {
    let exe = env!("CARGO_BIN_EXE_nextstat");
    let out = Command::new(exe).args(args).output().expect("failed to run nextstat");

    if !out.status.success() {
        panic!(
            "nextstat failed: status={}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    serde_json::from_slice(&out.stdout).expect("nextstat should emit valid JSON to stdout")
}

fn as_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("expected '{}' as f64, got: {}", key, v))
}

fn as_usize(v: &serde_json::Value, key: &str) -> usize {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| panic!("expected '{}' as u64, got: {}", key, v))
}

fn as_array_len(v: &serde_json::Value, key: &str) -> usize {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| a.len())
        .unwrap_or_else(|| panic!("expected '{}' as array, got: {}", key, v))
}

#[test]
fn phase3_1_tutorial_smoke_cli_outputs_have_expected_shape() {
    let input = fixture_path();
    assert!(input.is_file(), "missing fixture: {}", input.display());
    let input = input.to_string_lossy().to_string();

    // 1) Fit (MLE)
    let fit = run_nextstat_json(&["fit", "--input", &input, "--threads", "1"]);
    let n_params = as_array_len(&fit, "bestfit");
    assert!(n_params > 0, "expected non-empty bestfit");
    assert_eq!(as_array_len(&fit, "uncertainties"), n_params);
    assert_eq!(as_array_len(&fit, "parameter_names"), n_params);
    assert_eq!(as_array_len(&fit, "covariance"), n_params * n_params);
    assert!(fit.get("converged").and_then(|x| x.as_bool()).unwrap_or(false));
    assert!(as_f64(&fit, "nll").is_finite());
    assert!(as_f64(&fit, "twice_nll").is_finite());
    assert!(as_usize(&fit, "n_evaluations") > 0);

    // 2) Hypotest (qtilde + expected set)
    let ht = run_nextstat_json(&[
        "hypotest",
        "--input",
        &input,
        "--mu",
        "1.0",
        "--expected-set",
        "--threads",
        "1",
    ]);
    let cls = as_f64(&ht, "cls");
    assert!((0.0..=1.0).contains(&cls));
    assert!((0.0..=1.0).contains(&as_f64(&ht, "clb")));
    assert!((0.0..=1.0).contains(&as_f64(&ht, "clsb")));
    assert!(as_f64(&ht, "mu_hat").is_finite());

    let exp =
        ht.get("expected_set").and_then(|x| x.as_object()).expect("expected expected_set object");
    let exp_cls =
        exp.get("cls").and_then(|x| x.as_array()).expect("expected expected_set.cls array");
    assert_eq!(exp_cls.len(), 5);
    for v in exp_cls {
        let x = v.as_f64().expect("expected expected_set.cls elements to be f64");
        assert!(x.is_finite() && (0.0..=1.0).contains(&x));
    }
    let nsigma = exp
        .get("nsigma_order")
        .and_then(|x| x.as_array())
        .expect("expected expected_set.nsigma_order array");
    assert_eq!(nsigma.len(), 5);

    // 3) Upper limit (bisection mode + expected band)
    let ul = run_nextstat_json(&[
        "upper-limit",
        "--input",
        &input,
        "--alpha",
        "0.05",
        "--expected",
        "--threads",
        "1",
    ]);
    assert!((0.0..=1.0).contains(&as_f64(&ul, "alpha")));
    assert!(as_f64(&ul, "obs_limit").is_finite());
    let exp_limits =
        ul.get("exp_limits").and_then(|x| x.as_array()).expect("expected exp_limits array");
    assert_eq!(exp_limits.len(), 5);

    // 4) Profile scan
    let scan = run_nextstat_json(&[
        "scan",
        "--input",
        &input,
        "--start",
        "0.0",
        "--stop",
        "2.0",
        "--points",
        "21",
        "--threads",
        "1",
    ]);
    assert!(as_f64(&scan, "mu_hat").is_finite());
    assert!(as_f64(&scan, "nll_hat").is_finite());
    let points = scan.get("points").and_then(|x| x.as_array()).expect("expected points array");
    assert_eq!(points.len(), 21);
    for p in points {
        let mu = p.get("mu").and_then(|x| x.as_f64()).expect("mu must be f64");
        let q = p.get("q_mu").and_then(|x| x.as_f64()).expect("q_mu must be f64");
        assert!(mu.is_finite() && q.is_finite());
        assert!(q >= 0.0, "q_mu should be >= 0, got {}", q);
    }

    // 5) Viz artifacts (schema only, keeps runtime predictable).
    let prof = run_nextstat_json(&[
        "viz",
        "profile",
        "--input",
        &input,
        "--start",
        "0.0",
        "--stop",
        "2.0",
        "--points",
        "11",
        "--threads",
        "1",
    ]);
    assert!(prof.get("points").and_then(|x| x.as_array()).unwrap().len() == 11);

    let cls_art = run_nextstat_json(&[
        "viz",
        "cls",
        "--input",
        &input,
        "--alpha",
        "0.05",
        "--scan-start",
        "0.0",
        "--scan-stop",
        "5.0",
        "--scan-points",
        "31",
        "--threads",
        "1",
    ]);
    let cls_points = cls_art
        .get("points")
        .and_then(|x| x.as_array())
        .expect("expected points array for cls artifact");
    assert_eq!(cls_points.len(), 31);
}
