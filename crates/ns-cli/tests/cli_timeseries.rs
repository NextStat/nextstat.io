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

#[test]
fn timeseries_kalman_filter_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
    let fm = v.get("filtered_means").and_then(|x| x.as_array()).expect("filtered_means should be array");
    assert_eq!(fm.len(), 4);
}

#[test]
fn timeseries_kalman_filter_local_level_contract() {
    let input = fixture_path("kalman_local_level.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_level) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_linear_trend_contract() {
    let input = fixture_path("kalman_local_linear_trend.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_linear_trend) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_ar1_contract() {
    let input = fixture_path("kalman_ar1.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (ar1) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_arma11_contract() {
    let input = fixture_path("kalman_arma11.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (arma11) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_level_seasonal_contract() {
    let input = fixture_path("kalman_local_level_seasonal.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_level_seasonal) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_linear_trend_seasonal_contract() {
    let input = fixture_path("kalman_local_linear_trend_seasonal.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_linear_trend_seasonal) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_allows_missing_null() {
    let input = fixture_path("kalman_1d_missing.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_partial_missing_2d_contract() {
    let input = fixture_path("kalman_2d_partial_missing.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());

    let fm = v
        .get("filtered_means")
        .and_then(|x| x.as_array())
        .expect("filtered_means should be array");
    assert_eq!(fm.len(), 4);
    for t in 0..fm.len() {
        let row = fm[t].as_array().expect("filtered_means[t] should be array");
        assert_eq!(row.len(), 2);
    }

    let fc = v
        .get("filtered_covs")
        .and_then(|x| x.as_array())
        .expect("filtered_covs should be array");
    assert_eq!(fc.len(), 4);
    for t in 0..fc.len() {
        let rows = fc[t].as_array().expect("filtered_covs[t] should be array");
        assert_eq!(rows.len(), 2);
        for r in rows {
            let row = r.as_array().expect("filtered_covs[t][i] should be array");
            assert_eq!(row.len(), 2);
        }
    }
}

#[test]
fn timeseries_kalman_smooth_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-smooth",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-smooth should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let sm = v.get("smoothed_means").and_then(|x| x.as_array()).expect("smoothed_means should be array");
    assert_eq!(sm.len(), 4);
}

#[test]
fn timeseries_kalman_em_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-em",
        "--input",
        input.to_string_lossy().as_ref(),
        "--max-iter",
        "5",
        "--tol",
        "1e-12",
        "--min-diag",
        "1e-9",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-em should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("loglik_trace").and_then(|x| x.as_array()).unwrap().len() >= 2);
    let q = v.get("q").and_then(|x| x.as_array()).expect("q should be array");
    assert_eq!(q.len(), 1);
    let f = v.get("f").and_then(|x| x.as_array()).expect("f should be array");
    assert_eq!(f.len(), 1);
    let h = v.get("h").and_then(|x| x.as_array()).expect("h should be array");
    assert_eq!(h.len(), 1);
}

#[test]
fn timeseries_kalman_fit_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--max-iter",
        "5",
        "--tol",
        "1e-12",
        "--min-diag",
        "1e-9",
        "--forecast-steps",
        "2",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let em = v.get("em").expect("em should exist");
    assert!(em.get("loglik_trace").and_then(|x| x.as_array()).unwrap().len() >= 2);

    let model = v.get("model").expect("model should exist");
    assert!(model.get("f").is_some());
    assert!(model.get("q").is_some());
    assert!(model.get("h").is_some());
    assert!(model.get("r").is_some());
    assert!(model.get("m0").is_some());
    assert!(model.get("p0").is_some());

    let smooth = v.get("smooth").expect("smooth should exist");
    let sm = smooth
        .get("smoothed_means")
        .and_then(|x| x.as_array())
        .expect("smoothed_means should be array");
    assert_eq!(sm.len(), 4);

    let forecast = v.get("forecast").expect("forecast should exist");
    assert_eq!(
        forecast
            .get("obs_means")
            .and_then(|x| x.as_array())
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn timeseries_kalman_viz_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-viz",
        "--input",
        input.to_string_lossy().as_ref(),
        "--max-iter",
        "5",
        "--tol",
        "1e-12",
        "--min-diag",
        "1e-9",
        "--level",
        "0.9",
        "--forecast-steps",
        "2",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-viz should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("level").and_then(|x| x.as_f64()).unwrap() > 0.0);
    assert!(v.get("z").and_then(|x| x.as_f64()).unwrap().is_finite());

    let t_obs = v.get("t_obs").and_then(|x| x.as_array()).expect("t_obs should be array");
    assert_eq!(t_obs.len(), 4);

    let ys = v.get("ys").and_then(|x| x.as_array()).expect("ys should be array");
    assert_eq!(ys.len(), 4);

    let state_labels = v
        .get("state_labels")
        .and_then(|x| x.as_array())
        .expect("state_labels should be array");
    assert_eq!(state_labels.len(), 1);
    let obs_labels = v
        .get("obs_labels")
        .and_then(|x| x.as_array())
        .expect("obs_labels should be array");
    assert_eq!(obs_labels.len(), 1);

    let smooth = v.get("smooth").expect("smooth should exist");
    let state_mean = smooth
        .get("state_mean")
        .and_then(|x| x.as_array())
        .expect("smooth.state_mean should be array");
    assert_eq!(state_mean.len(), 4);
    let obs_mean = smooth
        .get("obs_mean")
        .and_then(|x| x.as_array())
        .expect("smooth.obs_mean should be array");
    assert_eq!(obs_mean.len(), 4);

    let forecast = v.get("forecast").expect("forecast should exist");
    assert_eq!(
        forecast
            .get("t")
            .and_then(|x| x.as_array())
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn timeseries_kalman_forecast_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-forecast",
        "--input",
        input.to_string_lossy().as_ref(),
        "--steps",
        "3",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-forecast should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(
        v.get("obs_means").and_then(|x| x.as_array()).unwrap().len(),
        3
    );
}

#[test]
fn timeseries_kalman_forecast_intervals_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-forecast",
        "--input",
        input.to_string_lossy().as_ref(),
        "--steps",
        "3",
        "--alpha",
        "0.05",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-forecast should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(
        v.get("obs_lower").and_then(|x| x.as_array()).unwrap().len(),
        3
    );
    assert_eq!(
        v.get("obs_upper").and_then(|x| x.as_array()).unwrap().len(),
        3
    );
    assert!(v.get("z").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert_eq!(v.get("alpha").and_then(|x| x.as_f64()).unwrap(), 0.05);
}

#[test]
fn timeseries_kalman_simulate_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-simulate",
        "--input",
        input.to_string_lossy().as_ref(),
        "--t-max",
        "5",
        "--seed",
        "123",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-simulate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("xs").and_then(|x| x.as_array()).unwrap().len(), 5);
    assert_eq!(v.get("ys").and_then(|x| x.as_array()).unwrap().len(), 5);
}
