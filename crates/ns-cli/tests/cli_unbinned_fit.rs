use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    // crates/ns-cli -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn tmp_path(filename: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, filename));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn assert_json_contract(v: &serde_json::Value) {
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );

    let param_names = v
        .get("parameter_names")
        .and_then(|x| x.as_array())
        .expect("parameter_names should be an array");
    assert!(!param_names.is_empty(), "parameter_names should be non-empty");

    let bestfit = v.get("bestfit").and_then(|x| x.as_array()).expect("bestfit should be an array");
    let unc = v
        .get("uncertainties")
        .and_then(|x| x.as_array())
        .expect("uncertainties should be an array");
    assert_eq!(bestfit.len(), param_names.len(), "bestfit length must match parameter_names");
    assert_eq!(unc.len(), param_names.len(), "uncertainties length must match parameter_names");

    let nll = v.get("nll").and_then(|x| x.as_f64()).expect("nll should be a number");
    assert!(nll.is_finite(), "nll must be finite");

    assert!(v.get("converged").and_then(|x| x.as_bool()).is_some(), "missing converged");
    let n_iter = v.get("n_iter").and_then(|x| x.as_u64()).expect("n_iter should be an integer");
    assert!(n_iter > 0, "n_iter should be > 0");
}

fn assert_metrics_contract(v: &serde_json::Value, command: &str) {
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_metrics_v0"),
        "schema_version mismatch: {v}"
    );
    assert_eq!(v.get("tool").and_then(|x| x.as_str()), Some("nextstat"));
    assert_eq!(v.get("command").and_then(|x| x.as_str()), Some(command));
    assert!(v.get("created_unix_ms").is_some(), "missing created_unix_ms");
    let timing = v.get("timing").and_then(|x| x.as_object()).expect("timing must be object");
    let t = timing
        .get("wall_time_s")
        .and_then(|x| x.as_f64())
        .expect("timing.wall_time_s must be number");
    assert!(t.is_finite() && t >= 0.0);
    assert!(v.get("metrics").and_then(|x| x.as_object()).is_some(), "metrics must be object");
}

fn assert_unbinned_fit_toys_contract(v: &serde_json::Value, n_toys: usize) {
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );

    let param_names = v
        .get("parameter_names")
        .and_then(|x| x.as_array())
        .expect("parameter_names should be an array");
    assert!(!param_names.is_empty(), "parameter_names should be non-empty");

    let poi_idx = v.get("poi_index").and_then(|x| x.as_u64()).expect("missing poi_index") as usize;
    assert!(poi_idx < param_names.len(), "poi_index out of range");

    let gen_section = v.get("gen").and_then(|x| x.as_object()).expect("gen should be object");
    assert!(gen_section.get("point").and_then(|x| x.as_str()).is_some(), "missing gen.point");
    assert_eq!(
        gen_section.get("n_toys").and_then(|x| x.as_u64()),
        Some(n_toys as u64),
        "gen.n_toys mismatch"
    );

    let gen_params =
        gen_section.get("params").and_then(|x| x.as_array()).expect("gen.params should be array");
    assert_eq!(gen_params.len(), param_names.len(), "gen.params length mismatch");

    let results = v.get("results").and_then(|x| x.as_object()).expect("results should be object");
    assert_eq!(
        results.get("n_toys").and_then(|x| x.as_u64()),
        Some(n_toys as u64),
        "results.n_toys mismatch"
    );

    let n_error =
        results.get("n_error").and_then(|x| x.as_u64()).expect("missing n_error") as usize;
    let n_validation_error = results
        .get("n_validation_error")
        .and_then(|x| x.as_u64())
        .expect("missing n_validation_error") as usize;
    let n_computation_error = results
        .get("n_computation_error")
        .and_then(|x| x.as_u64())
        .expect("missing n_computation_error") as usize;
    let n_converged =
        results.get("n_converged").and_then(|x| x.as_u64()).expect("missing n_converged") as usize;
    let n_nonconverged =
        results.get("n_nonconverged").and_then(|x| x.as_u64()).expect("missing n_nonconverged")
            as usize;
    assert_eq!(
        n_error,
        n_validation_error + n_computation_error,
        "n_error must equal n_validation_error + n_computation_error"
    );
    assert_eq!(n_error + n_converged + n_nonconverged, n_toys, "toy counts must add up to n_toys");

    let poi_hat =
        results.get("poi_hat").and_then(|x| x.as_array()).expect("poi_hat should be array");
    assert_eq!(poi_hat.len(), n_toys, "poi_hat length mismatch");

    let poi_sigma =
        results.get("poi_sigma").and_then(|x| x.as_array()).expect("poi_sigma should be array");
    assert_eq!(poi_sigma.len(), n_toys, "poi_sigma length mismatch");

    let converged =
        results.get("converged").and_then(|x| x.as_array()).expect("converged should be array");
    assert_eq!(converged.len(), n_toys, "converged length mismatch");

    let nll = results.get("nll").and_then(|x| x.as_array()).expect("nll should be array");
    assert_eq!(nll.len(), n_toys, "nll length mismatch");

    let guardrails =
        v.get("guardrails").and_then(|x| x.as_object()).expect("guardrails should be an object");
    assert!(
        guardrails.get("passed").and_then(|x| x.as_bool()).is_some(),
        "guardrails.passed should be a boolean"
    );
    assert!(
        guardrails.get("failures").and_then(|x| x.as_array()).is_some(),
        "guardrails.failures should be an array"
    );
}

fn assert_unbinned_ranking_contract(v: &serde_json::Value) {
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );

    let poi_idx = v.get("poi_index").and_then(|x| x.as_u64()).expect("missing poi_index") as usize;
    let mu_hat = v.get("mu_hat").and_then(|x| x.as_f64()).expect("missing mu_hat");
    assert!(mu_hat.is_finite(), "mu_hat must be finite");
    let nll_hat = v.get("nll_hat").and_then(|x| x.as_f64()).expect("missing nll_hat");
    assert!(nll_hat.is_finite(), "nll_hat must be finite");

    let ranking = v.get("ranking").and_then(|x| x.as_object()).expect("ranking should be object");
    let names =
        ranking.get("names").and_then(|x| x.as_array()).expect("ranking.names should be an array");
    let d_up = ranking
        .get("delta_mu_up")
        .and_then(|x| x.as_array())
        .expect("ranking.delta_mu_up should be an array");
    let d_down = ranking
        .get("delta_mu_down")
        .and_then(|x| x.as_array())
        .expect("ranking.delta_mu_down should be an array");
    let pull =
        ranking.get("pull").and_then(|x| x.as_array()).expect("ranking.pull should be an array");
    let constraint = ranking
        .get("constraint")
        .and_then(|x| x.as_array())
        .expect("ranking.constraint should be an array");
    assert_eq!(names.len(), d_up.len(), "ranking array length mismatch");
    assert_eq!(names.len(), d_down.len(), "ranking array length mismatch");
    assert_eq!(names.len(), pull.len(), "ranking array length mismatch");
    assert_eq!(names.len(), constraint.len(), "ranking array length mismatch");

    // poi_idx should still be present even if ranking is empty.
    let _ = poi_idx;
}

fn assert_unbinned_hypotest_contract(v: &serde_json::Value) {
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );

    assert!(v.get("poi_index").and_then(|x| x.as_u64()).is_some(), "missing poi_index");
    assert!(v.get("mu_test").and_then(|x| x.as_f64()).is_some(), "missing mu_test");
    assert!(v.get("mu_hat").and_then(|x| x.as_f64()).is_some(), "missing mu_hat");
    assert!(v.get("nll_hat").and_then(|x| x.as_f64()).is_some(), "missing nll_hat");
    assert!(v.get("nll_mu").and_then(|x| x.as_f64()).is_some(), "missing nll_mu");
    let q_mu = v.get("q_mu").and_then(|x| x.as_f64()).expect("missing q_mu");
    assert!(q_mu.is_finite() && q_mu >= 0.0, "q_mu must be finite and >= 0");
}

fn assert_unbinned_hypotest_toys_contract(v: &serde_json::Value, n_toys: usize) {
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );

    assert!(v.get("poi_index").and_then(|x| x.as_u64()).is_some(), "missing poi_index");
    assert!(v.get("mu_test").and_then(|x| x.as_f64()).is_some(), "missing mu_test");

    let cls = v.get("cls").and_then(|x| x.as_f64()).expect("missing cls");
    assert!(cls.is_finite() && (0.0..=1.0).contains(&cls), "cls must be in [0,1]");
    let clsb = v.get("clsb").and_then(|x| x.as_f64()).expect("missing clsb");
    assert!(clsb.is_finite() && (0.0..=1.0).contains(&clsb), "clsb must be in [0,1]");
    let clb = v.get("clb").and_then(|x| x.as_f64()).expect("missing clb");
    assert!(clb.is_finite() && (0.0..=1.0).contains(&clb), "clb must be in [0,1]");

    let q_obs = v.get("q_obs").and_then(|x| x.as_f64()).expect("missing q_obs");
    assert!(q_obs.is_finite() && q_obs >= 0.0, "q_obs must be finite and >= 0");

    let mu_hat = v.get("mu_hat").and_then(|x| x.as_f64()).expect("missing mu_hat");
    assert!(mu_hat.is_finite(), "mu_hat must be finite");

    let n_toys_obj = v.get("n_toys").and_then(|x| x.as_object()).expect("missing n_toys");
    assert_eq!(n_toys_obj.get("b").and_then(|x| x.as_u64()), Some(n_toys as u64));
    assert_eq!(n_toys_obj.get("sb").and_then(|x| x.as_u64()), Some(n_toys as u64));

    let n_error_obj = v.get("n_error").and_then(|x| x.as_object()).expect("missing n_error");
    assert!(n_error_obj.get("b").and_then(|x| x.as_u64()).is_some());
    assert!(n_error_obj.get("sb").and_then(|x| x.as_u64()).is_some());

    let n_nonconv_obj =
        v.get("n_nonconverged").and_then(|x| x.as_object()).expect("missing n_nonconverged");
    assert!(n_nonconv_obj.get("b").and_then(|x| x.as_u64()).is_some());
    assert!(n_nonconv_obj.get("sb").and_then(|x| x.as_u64()).is_some());

    assert!(v.get("seed").and_then(|x| x.as_u64()).is_some(), "missing seed");

    // expected_set can be null or an object with a 5-point cls band.
    if let Some(es) = v.get("expected_set")
        && !es.is_null()
    {
        let obj = es.as_object().expect("expected_set should be object");
        let nsigma =
            obj.get("nsigma_order").and_then(|x| x.as_array()).expect("missing nsigma_order");
        assert_eq!(nsigma.len(), 5, "nsigma_order must have length 5");
        let cls_band = obj.get("cls").and_then(|x| x.as_array()).expect("missing expected_set.cls");
        assert_eq!(cls_band.len(), 5, "expected_set.cls must have length 5");
    }
}

#[allow(dead_code)]
fn assert_close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{name} mismatch: |{a} - {b}| = {} > {tol}", (a - b).abs());
}

#[test]
fn unbinned_fit_toys_gpu_sample_toys_requires_gpu_cuda_or_metal() {
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        "does_not_exist.json",
        "--n-toys",
        "10",
        "--gpu-sample-toys",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-sample-toys requires --gpu cuda|metal"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_hypotest_toys_gpu_sample_toys_requires_gpu_cuda_or_metal() {
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        "does_not_exist.json",
        "--mu",
        "1.0",
        "--n-toys",
        "10",
        "--gpu-sample-toys",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-sample-toys requires --gpu cuda|metal"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_fit_toys_gpu_devices_requires_gpu_cuda() {
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        "does_not_exist.json",
        "--n-toys",
        "10",
        "--gpu-devices",
        "0,1",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-devices requires --gpu cuda"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_hypotest_toys_gpu_devices_requires_gpu_cuda() {
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        "does_not_exist.json",
        "--mu",
        "1.0",
        "--n-toys",
        "10",
        "--gpu-devices",
        "0,1",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-devices requires --gpu cuda"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_fit_toys_gpu_shards_allows_cuda_host_path() {
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        "does_not_exist.json",
        "--n-toys",
        "10",
        "--gpu",
        "cuda",
        "--gpu-shards",
        "2",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("does_not_exist.json")
            || stderr.contains("--gpu cuda requires building with --features cuda"),
        "expected no --gpu-shards/--gpu-sample-toys validation error, stderr={stderr}"
    );
}

#[test]
fn unbinned_hypotest_toys_gpu_shards_allows_cuda_host_path() {
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        "does_not_exist.json",
        "--mu",
        "1.0",
        "--n-toys",
        "10",
        "--gpu",
        "cuda",
        "--gpu-shards",
        "2",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("does_not_exist.json")
            || stderr.contains("--gpu cuda requires building with --features cuda"),
        "expected no --gpu-shards/--gpu-sample-toys validation error, stderr={stderr}"
    );
}

#[test]
fn unbinned_fit_toys_gpu_shards_requires_gpu_cuda() {
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        "does_not_exist.json",
        "--n-toys",
        "10",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
        "--gpu-shards",
        "2",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-shards currently requires --gpu cuda")
            || stderr.contains("--gpu metal requires building with --features metal"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_hypotest_toys_gpu_shards_requires_gpu_cuda() {
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        "does_not_exist.json",
        "--mu",
        "1.0",
        "--n-toys",
        "10",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
        "--gpu-shards",
        "2",
    ]);
    assert!(
        !out.status.success(),
        "expected non-zero exit, stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--gpu-shards currently requires --gpu cuda")
            || stderr.contains("--gpu metal requires building with --features metal"),
        "stderr did not mention flag constraint: {stderr}"
    );
}

#[test]
fn unbinned_fit_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_smoke_on_fixture_tree_with_data_weights() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec_weighted.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events", "weight": "weight_mc" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (data.weight) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_smoke_on_fixture_tree_with_data_weights_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path("unbinned_spec_weighted_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events", "weight": "weight_mc" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (data.weight) --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_smoke_on_fixture_tree_with_data_weights_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_weighted_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events", "weight": "weight_mc" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (data.weight) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn convert_root_to_parquet_then_unbinned_fit_smoke() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let parquet = tmp_path("events.parquet");
    let out = run(&[
        "convert",
        "--input",
        root.to_string_lossy().as_ref(),
        "--tree",
        "events",
        "--output",
        parquet.to_string_lossy().as_ref(),
        "--observable",
        "mbb:0:500",
    ]);
    assert!(
        out.status.success(),
        "convert should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(parquet.exists(), "convert did not create output parquet file");

    let config = tmp_path("unbinned_spec_parquet.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed on parquet input, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
#[cfg(feature = "metal")]
fn convert_root_to_parquet_then_unbinned_fit_gpu_metal_smoke() {
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let parquet = tmp_path("events_gpu_metal.parquet");
    let out = run(&[
        "convert",
        "--input",
        root.to_string_lossy().as_ref(),
        "--tree",
        "events",
        "--output",
        parquet.to_string_lossy().as_ref(),
        "--observable",
        "mbb:0:500",
    ]);
    assert!(
        out.status.success(),
        "convert should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(parquet.exists(), "convert did not create output parquet file");

    let config = tmp_path("unbinned_spec_parquet_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed on parquet input with --gpu metal, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
#[cfg(feature = "cuda")]
fn convert_root_to_parquet_then_unbinned_fit_gpu_cuda_smoke() {
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let parquet = tmp_path("events_gpu_cuda.parquet");
    let out = run(&[
        "convert",
        "--input",
        root.to_string_lossy().as_ref(),
        "--tree",
        "events",
        "--output",
        parquet.to_string_lossy().as_ref(),
        "--observable",
        "mbb:0:500",
    ]);
    assert!(
        out.status.success(),
        "convert should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(parquet.exists(), "convert did not create output parquet file");

    let config = tmp_path("unbinned_spec_parquet_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed on parquet input with --gpu cuda, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
fn unbinned_parquet_channel_selection_compiles_expected_event_counts() {
    // Build a synthetic multi-channel Parquet file and ensure `channels[].data.channel` selects
    // the correct subset when compiling the unbinned spec.
    use ns_unbinned::event_parquet::write_event_parquet_multi_channel;
    use ns_unbinned::{EventStore, ObservableSpec};

    let parquet = tmp_path("events_multi_channel.parquet");

    let obs = vec![ObservableSpec::branch("mbb", (0.0, 500.0))];
    let sr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![100.0, 110.0, 120.0])],
        Some(vec![1.0, 2.0, 3.0]),
    )
    .unwrap();
    let cr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![200.0, 210.0])],
        Some(vec![4.0, 5.0]),
    )
    .unwrap();

    write_event_parquet_multi_channel(
        &[("SR".to_string(), sr.clone()), ("CR".to_string(), cr.clone())],
        &parquet,
    )
    .unwrap();

    let config = tmp_path("unbinned_spec_parquet_channel_selection.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet, "channel": "SR" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            },
            {
                "name": "CR",
                "include_in_fit": false,
                "data": { "file": parquet, "channel": "CR" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let spec2 = ns_unbinned::spec::read_unbinned_spec(&config).unwrap();
    let model = ns_unbinned::spec::compile_unbinned_model(&spec2, &config).unwrap();

    let sr_ch = model.channels().iter().find(|c| c.name == "SR").unwrap();
    assert_eq!(sr_ch.data.n_events(), 3);
    assert_eq!(sr_ch.data.column("mbb").unwrap(), sr.column("mbb").unwrap());
    assert_eq!(sr_ch.data.weights().unwrap(), sr.weights().unwrap());

    let cr_ch = model.channels().iter().find(|c| c.name == "CR").unwrap();
    assert_eq!(cr_ch.data.n_events(), 2);
    assert_eq!(cr_ch.data.column("mbb").unwrap(), cr.column("mbb").unwrap());
    assert_eq!(cr_ch.data.weights().unwrap(), cr.weights().unwrap());

    // Unknown channel should fail compilation with a clear error.
    let bad_config = tmp_path("unbinned_spec_parquet_bad_channel.json");
    let bad_spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": { "parameters": [ { "name": "p", "init": 0.0, "bounds": [-1.0, 1.0] } ] },
        "channels": [
            {
                "name": "SR",
                "data": { "file": parquet, "channel": "DOES_NOT_EXIST" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    { "name": "p0", "pdf": { "type": "histogram", "observable": "mbb", "bin_edges": [0.0, 500.0], "bin_content": [1.0] }, "yield": { "type": "fixed", "value": 1.0 } }
                ]
            }
        ]
    });
    std::fs::write(&bad_config, serde_json::to_string_pretty(&bad_spec).unwrap()).unwrap();
    let bad_spec2 = ns_unbinned::spec::read_unbinned_spec(&bad_config).unwrap();
    let msg = match ns_unbinned::spec::compile_unbinned_model(&bad_spec2, &bad_config) {
        Ok(_) => panic!("expected compile_unbinned_model to fail for unknown channel"),
        Err(e) => format!("{e:#}"),
    };
    assert!(msg.contains("available channels"), "unexpected error: {msg}");

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&bad_config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
#[cfg(feature = "metal")]
fn unbinned_fit_smoke_multi_channel_parquet_gpu_metal() {
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        return;
    }

    use ns_unbinned::event_parquet::write_event_parquet_multi_channel;
    use ns_unbinned::{EventStore, ObservableSpec};

    let parquet = tmp_path("events_multi_channel_gpu_metal.parquet");
    let obs = vec![ObservableSpec::branch("mbb", (0.0, 500.0))];
    let sr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![100.0, 110.0, 120.0])],
        Some(vec![1.0, 2.0, 3.0]),
    )
    .unwrap();
    let cr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![200.0, 210.0])],
        Some(vec![4.0, 5.0]),
    )
    .unwrap();
    write_event_parquet_multi_channel(&[("SR".to_string(), sr), ("CR".to_string(), cr)], &parquet)
        .unwrap();

    let config = tmp_path("unbinned_spec_multi_channel_parquet_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet, "channel": "SR" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed on multi-channel parquet with --gpu metal, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
#[cfg(feature = "cuda")]
fn unbinned_fit_smoke_multi_channel_parquet_gpu_cuda() {
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        return;
    }

    use ns_unbinned::event_parquet::write_event_parquet_multi_channel;
    use ns_unbinned::{EventStore, ObservableSpec};

    let parquet = tmp_path("events_multi_channel_gpu_cuda.parquet");
    let obs = vec![ObservableSpec::branch("mbb", (0.0, 500.0))];
    let sr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![100.0, 110.0, 120.0])],
        Some(vec![1.0, 2.0, 3.0]),
    )
    .unwrap();
    let cr = EventStore::from_columns(
        obs.clone(),
        vec![("mbb".to_string(), vec![200.0, 210.0])],
        Some(vec![4.0, 5.0]),
    )
    .unwrap();
    write_event_parquet_multi_channel(&[("SR".to_string(), sr), ("CR".to_string(), cr)], &parquet)
        .unwrap();

    let config = tmp_path("unbinned_spec_multi_channel_parquet_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet, "channel": "SR" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed on multi-channel parquet with --gpu cuda, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[test]
fn unbinned_scan_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec_scan.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-scan",
        "--config",
        config.to_string_lossy().as_ref(),
        "--start",
        "0.0",
        "--stop",
        "2.0",
        "--points",
        "5",
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-scan should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(
        v.get("input_schema_version").and_then(|x| x.as_str()),
        Some("nextstat_unbinned_spec_v0"),
        "input_schema_version mismatch: {v}"
    );
    assert!(v.get("poi_index").and_then(|x| x.as_u64()).is_some(), "missing poi_index");
    assert!(v.get("mu_hat").and_then(|x| x.as_f64()).is_some(), "missing mu_hat");
    assert!(v.get("nll_hat").and_then(|x| x.as_f64()).is_some(), "missing nll_hat");
    let pts = v.get("points").and_then(|x| x.as_array()).expect("points should be array");
    assert_eq!(pts.len(), 5, "points length mismatch");

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_ranking_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec_ranking.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "mc_shape",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p"
                                }
                            ]
                        },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-ranking", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-ranking should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_ranking_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_hypotest_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-hypotest",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_hypotest_toys_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 50usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_hypotest_toys_smoke_on_fixture_tree_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 10usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_hypotest_toys_smoke_on_fixture_tree_gpu_sample_toys_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys_gpu_sample_toys_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 10usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys --gpu metal --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_hypotest_toys_accepts_histogram_from_tree_yield_weightsys_gpu_sample_toys_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path(
        "unbinned_spec_hypotest_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_metal.json",
    );
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 8usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys (histogram_from_tree + yield-only WeightSys) --gpu metal --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_hypotest_toys_smoke_two_channel_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys_two_channel_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 900.0, "scale": "mu" }
                    }
                ]
            },
            {
                "name": "CR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 400.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 25usize;

    let out_cpu = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
    ]);
    assert!(
        out_cpu.status.success(),
        "unbinned-hypotest-toys (two channel) cpu should succeed, stderr={}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );

    let out_gpu = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out_gpu.status.success(),
        "unbinned-hypotest-toys (two channel) --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out_gpu.stderr)
    );

    let v_cpu: serde_json::Value =
        serde_json::from_slice(&out_cpu.stdout).expect("stdout should be valid JSON");
    let v_gpu: serde_json::Value =
        serde_json::from_slice(&out_gpu.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v_cpu, n_toys);
    assert_unbinned_hypotest_toys_contract(&v_gpu, n_toys);

    let cls_cpu = v_cpu.get("cls").and_then(|x| x.as_f64()).unwrap();
    let cls_gpu = v_gpu.get("cls").and_then(|x| x.as_f64()).unwrap();
    let clsb_cpu = v_cpu.get("clsb").and_then(|x| x.as_f64()).unwrap();
    let clsb_gpu = v_gpu.get("clsb").and_then(|x| x.as_f64()).unwrap();
    let clb_cpu = v_cpu.get("clb").and_then(|x| x.as_f64()).unwrap();
    let clb_gpu = v_gpu.get("clb").and_then(|x| x.as_f64()).unwrap();
    let qobs_cpu = v_cpu.get("q_obs").and_then(|x| x.as_f64()).unwrap();
    let qobs_gpu = v_gpu.get("q_obs").and_then(|x| x.as_f64()).unwrap();
    let muhat_cpu = v_cpu.get("mu_hat").and_then(|x| x.as_f64()).unwrap();
    let muhat_gpu = v_gpu.get("mu_hat").and_then(|x| x.as_f64()).unwrap();

    // CPU vs GPU toy fits can differ slightly due to optimizer tolerances; require close agreement.
    assert_close("cls", cls_cpu, cls_gpu, 0.05);
    assert_close("clsb", clsb_cpu, clsb_gpu, 0.05);
    assert_close("clb", clb_cpu, clb_gpu, 0.05);
    assert_close("q_obs", qobs_cpu, qobs_gpu, 1e-2);
    assert_close("mu_hat", muhat_cpu, muhat_gpu, 1e-2);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_histogram_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_hist.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "bin_content": [1.0, 1.0, 1.0, 1.0, 1.0],
                            "pseudo_count": 0.5
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram pdf) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_crystal_ball_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_cb.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "cb_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "cb_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "cb_alpha", "init": 1.5, "bounds": [1.5, 1.5] },
                { "name": "cb_n", "init": 3.0, "bounds": [3.0, 3.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "crystal_ball",
                            "observable": "mbb",
                            "params": ["cb_mu", "cb_sigma", "cb_alpha", "cb_n"]
                        },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (crystal_ball) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_chebyshev_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_cheb.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "c1", "init": 0.0, "bounds": [-0.5, 0.5] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "chebyshev", "observable": "mbb", "params": ["c1"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (chebyshev) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_histogram_from_tree_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_hist_from_tree.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events" }
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram_from_tree) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_histogram_from_tree_with_weight_systematics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_hist_from_tree_weightsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                { "param": "alpha_jes", "up": "weight_jes_up/weight_mc", "down": "weight_jes_down/weight_mc", "interp": "code4p" }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram_from_tree + weight_systematics) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_histogram_from_tree_with_horizontal_systematics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_hist_from_tree_horizsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "horizontal_systematics": [
                                { "param": "alpha_jes", "up": "mbb*1.02", "down": "mbb*0.98", "interp": "code4p" }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram_from_tree + horizontal_systematics) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_kde_from_tree_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_kde.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "kde_from_tree",
                            "observable": "mbb",
                            "bandwidth": 25.0,
                            "max_events": 250,
                            "source": { "file": root, "tree": "events" }
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (kde_from_tree) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_kde_from_tree_with_horizontal_systematics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_kde_horizsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "kde_from_tree",
                            "observable": "mbb",
                            "bandwidth": 25.0,
                            "max_events": 250,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "horizontal_systematics": [
                                { "param": "alpha_jes", "up": "mbb*1.02", "down": "mbb*0.98", "interp": "code4p" }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (kde_from_tree + horizontal_systematics) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_kde_from_tree_with_weight_systematics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_kde_weightsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "kde_from_tree",
                            "observable": "mbb",
                            "bandwidth": 25.0,
                            "max_events": 250,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                { "param": "alpha_jes", "up": "weight_jes_up/weight_mc", "down": "weight_jes_down/weight_mc", "interp": "code4p" }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (kde_from_tree + weight_systematics) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_normsys_rate_modifier() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_normsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu0", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_lumi", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "bin_content": [1.0, 1.0, 1.0, 1.0, 1.0],
                            "pseudo_count": 0.5
                        },
                        "yield": {
                            "type": "parameter",
                            "name": "nu0",
                            "modifiers": [
                                { "type": "normsys", "param": "alpha_lumi", "lo": 0.97, "hi": 1.03 }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (NormSys) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_weightsys_rate_modifier() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_weightsys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": {
                            "type": "scaled",
                            "base_yield": 1000.0,
                            "scale": "mu",
                            "modifiers": [
                                { "type": "weightsys", "param": "alpha_jes", "lo": 0.92, "hi": 1.08, "interp_code": "code4p" }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (WeightSys) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_argus_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_argus.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "argus_c", "init": -10.0, "bounds": [-100.0, 0.0] },
                { "name": "argus_p", "init": 0.5, "bounds": [0.0, 5.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "argus", "observable": "mbb", "params": ["argus_c", "argus_p"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (Argus) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_voigtian_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_voigtian.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "v_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "v_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "v_gamma", "init": 10.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "voigtian", "observable": "mbb", "params": ["v_mu", "v_sigma", "v_gamma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (Voigtian) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_spline_pdf() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_spline.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "spline",
                            "observable": "mbb",
                            "knots_x": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "knots_y": [1.0, 1.2, 1.1, 0.9, 0.8, 0.7]
                        },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (Spline) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_accepts_product_pdf_on_parquet_two_observables() {
    use ns_unbinned::event_parquet::write_event_parquet;
    use ns_unbinned::{EventStore, ObservableSpec};

    let parquet = tmp_path("events_product_2d.parquet");
    let obs =
        vec![ObservableSpec::branch("x", (0.0, 1.0)), ObservableSpec::branch("y", (0.0, 5.0))];
    let store = EventStore::from_columns(
        obs,
        vec![
            ("x".to_string(), vec![0.1, 0.2, 0.7, 0.9]),
            ("y".to_string(), vec![0.5, 1.0, 2.0, 3.5]),
        ],
        None,
    )
    .unwrap();
    write_event_parquet(&store, &parquet).unwrap();

    let config = tmp_path("unbinned_spec_product_parquet.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "x_mu", "init": 0.5, "bounds": [0.0, 1.0] },
                { "name": "x_sigma", "init": 0.2, "bounds": [0.01, 1.0] },
                { "name": "y_lambda", "init": -0.5, "bounds": [-10.0, 10.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": parquet },
                "observables": [
                    { "name": "x", "bounds": [0.0, 1.0] },
                    { "name": "y", "bounds": [0.0, 5.0] }
                ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "product",
                            "components": [
                                { "type": "gaussian", "observable": "x", "params": ["x_mu", "x_sigma"] },
                                { "type": "exponential", "observable": "y", "params": ["y_lambda"] }
                            ]
                        },
                        "yield": { "type": "fixed", "value": 100.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out =
        run(&["unbinned-fit", "--config", config.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "unbinned-fit (Product) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&parquet);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_accepts_histogram_from_tree_yield_weightsys_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path("unbinned_spec_hist_from_tree_yield_weightsys_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram_from_tree + yield-only WeightSys) --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_accepts_histogram_from_tree_yield_weightsys_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_hist_from_tree_yield_weightsys_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (histogram_from_tree + yield-only WeightSys) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_accepts_histogram_from_tree_yield_weightsys_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_fit_toys_hist_from_tree_yield_weightsys_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "4",
        "--seed",
        "11",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys (histogram_from_tree + yield-only WeightSys) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 4);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_accepts_histogram_from_tree_yield_weightsys_gpu_sample_toys_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config =
        tmp_path("unbinned_spec_fit_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "4",
        "--seed",
        "11",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys (histogram_from_tree + yield-only WeightSys) --gpu cuda --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 4);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_gpu_sample_toys_sharded_writes_sharded_cuda_metrics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path(
        "unbinned_spec_fit_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_sharded_cuda.json",
    );
    let metrics = tmp_path("unbinned_fit_toys_gpu_sample_toys_sharded_cuda_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 7usize;
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "11",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-sample-toys",
        "--gpu-devices",
        "0",
        "--gpu-shards",
        "3",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu cuda --gpu-sample-toys --gpu-shards should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result_json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&result_json, n_toys);

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let toys = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.as_object())
        .expect("timing.breakdown.toys must be present");
    assert_eq!(
        toys.get("pipeline").and_then(|x| x.as_str()),
        Some("cuda_device_sharded"),
        "expected sharded CUDA device pipeline"
    );
    let device_ids = toys.get("device_ids").and_then(|x| x.as_array()).expect("device_ids array");
    assert_eq!(device_ids.len(), 1, "expected a single selected CUDA device");
    assert_eq!(device_ids[0].as_u64(), Some(0), "expected CUDA device id 0");
    let shard_plan =
        toys.get("device_shard_plan").and_then(|x| x.as_array()).expect("device_shard_plan array");
    assert_eq!(shard_plan.len(), 3, "expected 3 logical shards in plan");
    assert!(
        shard_plan.iter().all(|v| v.as_u64() == Some(0)),
        "single-GPU shard plan should map all shards to device 0"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_gpu_host_sharded_writes_sharded_cuda_metrics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path(
        "unbinned_spec_fit_toys_hist_from_tree_yield_weightsys_gpu_host_sharded_cuda.json",
    );
    let metrics = tmp_path("unbinned_fit_toys_gpu_host_sharded_cuda_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 7usize;
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "11",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-devices",
        "0",
        "--gpu-shards",
        "3",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu cuda --gpu-shards should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result_json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&result_json, n_toys);

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let toys = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.as_object())
        .expect("timing.breakdown.toys must be present");
    assert_eq!(
        toys.get("pipeline").and_then(|x| x.as_str()),
        Some("cuda_host_sharded"),
        "expected sharded CUDA host pipeline"
    );
    let device_ids = toys.get("device_ids").and_then(|x| x.as_array()).expect("device_ids array");
    assert_eq!(device_ids.len(), 1, "expected a single selected CUDA device");
    assert_eq!(device_ids[0].as_u64(), Some(0), "expected CUDA device id 0");
    let shard_plan =
        toys.get("device_shard_plan").and_then(|x| x.as_array()).expect("device_shard_plan array");
    assert_eq!(shard_plan.len(), 3, "expected 3 logical shards in plan");
    assert!(
        shard_plan.iter().all(|v| v.as_u64() == Some(0)),
        "single-GPU shard plan should map all shards to device 0"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_accepts_weightsys_rate_modifier_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path("unbinned_spec_weightsys_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": {
                            "type": "scaled",
                            "base_yield": 1000.0,
                            "scale": "mu",
                            "modifiers": [
                                { "type": "weightsys", "param": "alpha_jes", "lo": 0.92, "hi": 1.08, "interp_code": "code4p" }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (WeightSys) --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_accepts_weightsys_rate_modifier_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_weightsys_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": {
                            "type": "scaled",
                            "base_yield": 1000.0,
                            "scale": "mu",
                            "modifiers": [
                                { "type": "weightsys", "param": "alpha_jes", "lo": 0.92, "hi": 1.08, "interp_code": "code4p" }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit (WeightSys) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_hypotest_toys_accepts_weightsys_rate_modifier_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys_weightsys_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": {
                            "type": "scaled",
                            "base_yield": 1000.0,
                            "scale": "mu",
                            "modifiers": [
                                { "type": "weightsys", "param": "alpha_jes", "lo": 0.92, "hi": 1.08, "interp_code": "code4p" }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 10usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys (WeightSys) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_hypotest_toys_accepts_histogram_from_tree_yield_weightsys_gpu_sample_toys_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path(
        "unbinned_spec_hypotest_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_cuda.json",
    );
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 8usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys (histogram_from_tree + yield-only WeightSys) --gpu cuda --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_hypotest_toys_gpu_sample_toys_sharded_writes_sharded_cuda_metrics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path(
        "unbinned_spec_hypotest_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_sharded_cuda.json",
    );
    let metrics = tmp_path("unbinned_hypotest_toys_gpu_sample_toys_sharded_cuda_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 9usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-sample-toys",
        "--gpu-devices",
        "0",
        "--gpu-shards",
        "4",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys --gpu cuda --gpu-sample-toys --gpu-shards should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result_json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&result_json, n_toys);

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_hypotest_toys");

    let toys = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.as_object())
        .expect("timing.breakdown.toys must be present");
    assert_eq!(
        toys.get("pipeline").and_then(|x| x.as_str()),
        Some("cuda_device_sharded"),
        "expected sharded CUDA device pipeline"
    );
    let device_ids = toys.get("device_ids").and_then(|x| x.as_array()).expect("device_ids array");
    assert_eq!(device_ids.len(), 1, "expected a single selected CUDA device");
    assert_eq!(device_ids[0].as_u64(), Some(0), "expected CUDA device id 0");
    let shard_plan =
        toys.get("device_shard_plan").and_then(|x| x.as_array()).expect("device_shard_plan array");
    assert_eq!(shard_plan.len(), 4, "expected 4 logical shards in plan");
    assert!(
        shard_plan.iter().all(|v| v.as_u64() == Some(0)),
        "single-GPU shard plan should map all shards to device 0"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_hypotest_toys_gpu_host_sharded_writes_sharded_cuda_metrics() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path(
        "unbinned_spec_hypotest_toys_hist_from_tree_yield_weightsys_gpu_host_sharded_cuda.json",
    );
    let metrics = tmp_path("unbinned_hypotest_toys_gpu_host_sharded_cuda_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 9usize;
    let out = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-devices",
        "0",
        "--gpu-shards",
        "4",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-hypotest-toys --gpu cuda --gpu-shards should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result_json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&result_json, n_toys);

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_hypotest_toys");

    let toys = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.as_object())
        .expect("timing.breakdown.toys must be present");
    assert_eq!(
        toys.get("pipeline").and_then(|x| x.as_str()),
        Some("cuda_host_sharded"),
        "expected sharded CUDA host pipeline"
    );
    let device_ids = toys.get("device_ids").and_then(|x| x.as_array()).expect("device_ids array");
    assert_eq!(device_ids.len(), 1, "expected a single selected CUDA device");
    assert_eq!(device_ids[0].as_u64(), Some(0), "expected CUDA device id 0");
    let shard_plan =
        toys.get("device_shard_plan").and_then(|x| x.as_array()).expect("device_shard_plan array");
    assert_eq!(shard_plan.len(), 4, "expected 4 logical shards in plan");
    assert!(
        shard_plan.iter().all(|v| v.as_u64() == Some(0)),
        "single-GPU shard plan should map all shards to device 0"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_accepts_weightsys_rate_modifier_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let config = tmp_path("unbinned_spec_fit_toys_weightsys_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": {
                            "type": "scaled",
                            "base_yield": 1000.0,
                            "scale": "mu",
                            "modifiers": [
                                { "type": "weightsys", "param": "alpha_jes", "lo": 0.92, "hi": 1.08, "interp_code": "code4p" }
                            ]
                        }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 10usize;
    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys (WeightSys) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, n_toys);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_hypotest_toys_smoke_two_channel_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_hypotest_toys_two_channel_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 900.0, "scale": "mu" }
                    }
                ]
            },
            {
                "name": "CR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 400.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 25usize;

    let out_cpu = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
    ]);
    assert!(
        out_cpu.status.success(),
        "unbinned-hypotest-toys (two channel) cpu should succeed, stderr={}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );

    let out_gpu = run(&[
        "unbinned-hypotest-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "42",
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out_gpu.status.success(),
        "unbinned-hypotest-toys (two channel) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out_gpu.stderr)
    );

    let v_cpu: serde_json::Value =
        serde_json::from_slice(&out_cpu.stdout).expect("stdout should be valid JSON");
    let v_gpu: serde_json::Value =
        serde_json::from_slice(&out_gpu.stdout).expect("stdout should be valid JSON");
    assert_unbinned_hypotest_toys_contract(&v_cpu, n_toys);
    assert_unbinned_hypotest_toys_contract(&v_gpu, n_toys);

    let cls_cpu = v_cpu.get("cls").and_then(|x| x.as_f64()).unwrap();
    let cls_gpu = v_gpu.get("cls").and_then(|x| x.as_f64()).unwrap();
    let clsb_cpu = v_cpu.get("clsb").and_then(|x| x.as_f64()).unwrap();
    let clsb_gpu = v_gpu.get("clsb").and_then(|x| x.as_f64()).unwrap();
    let clb_cpu = v_cpu.get("clb").and_then(|x| x.as_f64()).unwrap();
    let clb_gpu = v_gpu.get("clb").and_then(|x| x.as_f64()).unwrap();
    let qobs_cpu = v_cpu.get("q_obs").and_then(|x| x.as_f64()).unwrap();
    let qobs_gpu = v_gpu.get("q_obs").and_then(|x| x.as_f64()).unwrap();
    let muhat_cpu = v_cpu.get("mu_hat").and_then(|x| x.as_f64()).unwrap();
    let muhat_gpu = v_gpu.get("mu_hat").and_then(|x| x.as_f64()).unwrap();

    // CPU vs GPU toy fits can differ slightly due to optimizer tolerances; require close agreement.
    assert_close("cls", cls_cpu, cls_gpu, 0.05);
    assert_close("clsb", clsb_cpu, clsb_gpu, 0.05);
    assert_close("clb", clb_cpu, clb_gpu, 0.05);
    assert_close("q_obs", qobs_cpu, qobs_gpu, 1e-2);
    assert_close("mu_hat", muhat_cpu, muhat_gpu, 1e-2);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_smoke_two_channel_gpu_cuda() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_fit_toys_two_channel_gpu_cuda.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            },
            {
                "name": "CR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 300.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 5usize;
    let out_cpu = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "123",
        "--threads",
        "1",
    ]);
    assert!(
        out_cpu.status.success(),
        "unbinned-fit-toys (two channel) cpu should succeed, stderr={}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );

    let out_gpu = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "cuda",
    ]);
    assert!(
        out_gpu.status.success(),
        "unbinned-fit-toys (two channel) --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out_gpu.stderr)
    );

    let v_cpu: serde_json::Value =
        serde_json::from_slice(&out_cpu.stdout).expect("stdout should be valid JSON");
    let v_gpu: serde_json::Value =
        serde_json::from_slice(&out_gpu.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v_cpu, n_toys);
    assert_unbinned_fit_toys_contract(&v_gpu, n_toys);

    let poi_hat_cpu =
        v_cpu.get("results").and_then(|x| x.get("poi_hat")).and_then(|x| x.as_array()).unwrap();
    let poi_hat_gpu =
        v_gpu.get("results").and_then(|x| x.get("poi_hat")).and_then(|x| x.as_array()).unwrap();
    assert_eq!(poi_hat_cpu.len(), n_toys);
    assert_eq!(poi_hat_gpu.len(), n_toys);
    for (i, (a, b)) in poi_hat_cpu.iter().zip(poi_hat_gpu.iter()).enumerate() {
        let a = a.as_f64().unwrap_or(f64::NAN);
        let b = b.as_f64().unwrap_or(f64::NAN);
        assert!(
            a.is_finite() && b.is_finite(),
            "poi_hat[{i}] must be finite in both cpu and gpu outputs"
        );
        assert_close("poi_hat", a, b, 2e-2);
    }

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_cuda_default_does_not_auto_enable_gpu_native() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_cuda_default_no_native.json");
    let metrics = tmp_path("unbinned_fit_toys_cuda_default_no_native_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "3",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let pipeline = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.get("pipeline"))
        .and_then(|x| x.as_str())
        .expect("timing.breakdown.toys.pipeline must be present");
    assert_eq!(
        pipeline, "cuda_device",
        "default --gpu cuda path must not auto-enable --gpu-native"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_cuda_gpu_native_requires_explicit_flag() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_cuda_explicit_native.json");
    let metrics = tmp_path("unbinned_fit_toys_cuda_explicit_native_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "3",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-native",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu cuda --gpu-native should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let pipeline = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.get("pipeline"))
        .and_then(|x| x.as_str())
        .expect("timing.breakdown.toys.pipeline must be present");
    assert_eq!(
        pipeline, "cuda_gpu_native",
        "explicit --gpu-native must select cuda_gpu_native pipeline"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_cuda_dcb_default_route_is_host() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_cuda_dcb_default_route.json");
    let metrics = tmp_path("unbinned_fit_toys_cuda_dcb_default_route_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "dcb_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "dcb_sigma", "init": 30.0, "bounds": [30.0, 30.0] },
                { "name": "dcb_alpha_l", "init": 1.5, "bounds": [1.5, 1.5] },
                { "name": "dcb_n_l", "init": 2.0, "bounds": [2.0, 2.0] },
                { "name": "dcb_alpha_r", "init": 2.0, "bounds": [2.0, 2.0] },
                { "name": "dcb_n_r", "init": 3.0, "bounds": [3.0, 3.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "double_crystal_ball",
                            "observable": "mbb",
                            "params": [
                                "dcb_mu",
                                "dcb_sigma",
                                "dcb_alpha_l",
                                "dcb_n_l",
                                "dcb_alpha_r",
                                "dcb_n_r"
                            ]
                        },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "3",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys DCB --gpu cuda should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let pipeline = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.get("pipeline"))
        .and_then(|x| x.as_str())
        .expect("timing.breakdown.toys.pipeline must be present");
    assert_eq!(
        pipeline, "host",
        "DCB default --gpu cuda route should remain host unless --gpu-native is passed"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[cfg(feature = "cuda")]
#[test]
fn unbinned_fit_toys_cuda_dcb_gpu_native_route_is_explicit() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_cuda_dcb_native_route.json");
    let metrics = tmp_path("unbinned_fit_toys_cuda_dcb_native_route_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "dcb_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "dcb_sigma", "init": 30.0, "bounds": [30.0, 30.0] },
                { "name": "dcb_alpha_l", "init": 1.5, "bounds": [1.5, 1.5] },
                { "name": "dcb_n_l", "init": 2.0, "bounds": [2.0, 2.0] },
                { "name": "dcb_alpha_r", "init": 2.0, "bounds": [2.0, 2.0] },
                { "name": "dcb_n_r", "init": 3.0, "bounds": [3.0, 3.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "double_crystal_ball",
                            "observable": "mbb",
                            "params": [
                                "dcb_mu",
                                "dcb_sigma",
                                "dcb_alpha_l",
                                "dcb_n_l",
                                "dcb_alpha_r",
                                "dcb_n_r"
                            ]
                        },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "3",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "cuda",
        "--gpu-native",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys DCB --gpu cuda --gpu-native should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let metrics_json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&metrics_json, "unbinned_fit_toys");

    let pipeline = metrics_json
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.get("pipeline"))
        .and_then(|x| x.as_str())
        .expect("timing.breakdown.toys.pipeline must be present");
    assert_eq!(
        pipeline, "cuda_gpu_native",
        "DCB --gpu-native must select cuda_gpu_native pipeline"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[test]
fn unbinned_fit_writes_metrics_json_to_file() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec.json");
    let metrics = tmp_path("unbinned_fit_metrics.json");

    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "data_like",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "fixed", "value": 1000.0 }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit",
        "--config",
        config.to_string_lossy().as_ref(),
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&v, "unbinned_fit");

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[test]
fn unbinned_fit_toys_smoke_on_fixture_tree() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "5",
        "--seed",
        "123",
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 5);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_toys_smoke_on_fixture_tree_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path("unbinned_spec_toys_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "5",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 5);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_toys_smoke_on_fixture_tree_gpu_sample_toys_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_fit_toys_gpu_sample_toys_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [0.0, 500.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [0.1, 200.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "5",
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys --gpu metal --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 5);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_toys_accepts_histogram_from_tree_yield_weightsys_gpu_sample_toys_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    let config = tmp_path(
        "unbinned_spec_fit_toys_hist_from_tree_yield_weightsys_gpu_sample_toys_metal.json",
    );
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "nu",
            "parameters": [
                { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                            "pseudo_count": 0.5,
                            "max_events": 500,
                            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
                            "weight_systematics": [
                                {
                                    "param": "alpha_jes",
                                    "up": "weight_jes_up/weight_mc",
                                    "down": "weight_jes_down/weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": false,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "parameter", "name": "nu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "4",
        "--seed",
        "11",
        "--gen",
        "init",
        "--threads",
        "1",
        "--gpu",
        "metal",
        "--gpu-sample-toys",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys (histogram_from_tree + yield-only WeightSys) --gpu metal --gpu-sample-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v, 4);

    let _ = std::fs::remove_file(&config);
}

#[cfg(feature = "metal")]
#[test]
fn unbinned_fit_toys_smoke_two_channel_gpu_metal() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    let config = tmp_path("unbinned_spec_fit_toys_two_channel_gpu_metal.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 800.0, "scale": "mu" }
                    }
                ]
            },
            {
                "name": "CR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 300.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let n_toys = 5usize;
    let out_cpu = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "123",
        "--threads",
        "1",
    ]);
    assert!(
        out_cpu.status.success(),
        "unbinned-fit-toys (two channel) cpu should succeed, stderr={}",
        String::from_utf8_lossy(&out_cpu.stderr)
    );

    let out_gpu = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        &n_toys.to_string(),
        "--seed",
        "123",
        "--threads",
        "1",
        "--gpu",
        "metal",
    ]);
    assert!(
        out_gpu.status.success(),
        "unbinned-fit-toys (two channel) --gpu metal should succeed, stderr={}",
        String::from_utf8_lossy(&out_gpu.stderr)
    );

    let v_cpu: serde_json::Value =
        serde_json::from_slice(&out_cpu.stdout).expect("stdout should be valid JSON");
    let v_gpu: serde_json::Value =
        serde_json::from_slice(&out_gpu.stdout).expect("stdout should be valid JSON");
    assert_unbinned_fit_toys_contract(&v_cpu, n_toys);
    assert_unbinned_fit_toys_contract(&v_gpu, n_toys);

    let poi_hat_cpu =
        v_cpu.get("results").and_then(|x| x.get("poi_hat")).and_then(|x| x.as_array()).unwrap();
    let poi_hat_gpu =
        v_gpu.get("results").and_then(|x| x.get("poi_hat")).and_then(|x| x.as_array()).unwrap();
    assert_eq!(poi_hat_cpu.len(), n_toys);
    assert_eq!(poi_hat_gpu.len(), n_toys);
    for (i, (a, b)) in poi_hat_cpu.iter().zip(poi_hat_gpu.iter()).enumerate() {
        let a = a.as_f64().unwrap_or(f64::NAN);
        let b = b.as_f64().unwrap_or(f64::NAN);
        assert!(
            a.is_finite() && b.is_finite(),
            "poi_hat[{i}] must be finite in both cpu and gpu outputs"
        );
        assert_close("poi_hat", a, b, 2e-2);
    }

    let _ = std::fs::remove_file(&config);
}

#[test]
fn unbinned_fit_toys_writes_metrics_json_to_file() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_metrics.json");
    let metrics = tmp_path("unbinned_fit_toys_metrics.json");
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "3",
        "--seed",
        "42",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "unbinned-fit-toys should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&metrics).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("metrics file should be JSON");
    assert_metrics_contract(&v, "unbinned_fit_toys");
    let toys = v
        .get("timing")
        .and_then(|x| x.get("breakdown"))
        .and_then(|x| x.get("toys"))
        .and_then(|x| x.as_object())
        .expect("timing.breakdown.toys must be present for unbinned-fit-toys metrics");
    assert_eq!(
        toys.get("pipeline").and_then(|x| x.as_str()),
        Some("cpu_batch"),
        "expected cpu_batch pipeline in unbinned-fit-toys metrics breakdown"
    );
    assert!(
        toys.get("sample_s").and_then(|x| x.as_f64()).unwrap_or(f64::NAN).is_finite(),
        "timing.breakdown.toys.sample_s must be finite"
    );
    assert!(
        toys.get("fit_s").and_then(|x| x.as_f64()).unwrap_or(f64::NAN).is_finite(),
        "timing.breakdown.toys.fit_s must be finite"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&metrics);
}

#[test]
fn unbinned_merge_toys_smoke_from_cpu_shards() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_merge_smoke.json");
    let shard0 = tmp_path("unbinned_toys_shard0.json");
    let shard1 = tmp_path("unbinned_toys_shard1.json");
    let merged = tmp_path("unbinned_toys_merged.json");

    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out0 = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "6",
        "--seed",
        "42",
        "--threads",
        "1",
        "--shard",
        "0/2",
        "--output",
        shard0.to_string_lossy().as_ref(),
    ]);
    assert!(
        out0.status.success(),
        "shard0 run should succeed, stderr={}",
        String::from_utf8_lossy(&out0.stderr)
    );

    let out1 = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "6",
        "--seed",
        "42",
        "--threads",
        "1",
        "--shard",
        "1/2",
        "--output",
        shard1.to_string_lossy().as_ref(),
    ]);
    assert!(
        out1.status.success(),
        "shard1 run should succeed, stderr={}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let merged_out = run(&[
        "unbinned-merge-toys",
        shard0.to_string_lossy().as_ref(),
        shard1.to_string_lossy().as_ref(),
        "--output",
        merged.to_string_lossy().as_ref(),
    ]);
    assert!(
        merged_out.status.success(),
        "unbinned-merge-toys should succeed, stderr={}",
        String::from_utf8_lossy(&merged_out.stderr)
    );

    let merged_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&merged).unwrap())
        .expect("merged output should be JSON");
    let results =
        merged_json.get("results").and_then(|x| x.as_object()).expect("results should be object");
    assert_eq!(results.get("n_toys").and_then(|x| x.as_u64()), Some(6));
    let n_error = results.get("n_error").and_then(|x| x.as_u64()).expect("missing n_error");
    let n_validation_error = results
        .get("n_validation_error")
        .and_then(|x| x.as_u64())
        .expect("missing n_validation_error");
    let n_computation_error = results
        .get("n_computation_error")
        .and_then(|x| x.as_u64())
        .expect("missing n_computation_error");
    assert_eq!(n_error, n_validation_error + n_computation_error);
    assert!(
        merged_json.get("shards").and_then(|x| x.as_array()).map(|x| x.len()).unwrap_or(0) == 2,
        "expected 2 shard entries"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&shard0);
    let _ = std::fs::remove_file(&shard1);
    let _ = std::fs::remove_file(&merged);
}

#[test]
fn unbinned_merge_toys_rejects_mismatched_gen_config() {
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        return;
    }

    let config = tmp_path("unbinned_spec_toys_merge_bad_gen.json");
    let shard0 = tmp_path("unbinned_toys_badgen_shard0.json");
    let shard1 = tmp_path("unbinned_toys_badgen_shard1.json");

    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0] },
                { "name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0] }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": true,
                "data": { "file": root, "tree": "events" },
                "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": { "type": "gaussian", "observable": "mbb", "params": ["gauss_mu", "gauss_sigma"] },
                        "yield": { "type": "scaled", "base_yield": 1000.0, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&config, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let out0 = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "6",
        "--seed",
        "42",
        "--threads",
        "1",
        "--shard",
        "0/2",
        "--output",
        shard0.to_string_lossy().as_ref(),
    ]);
    assert!(out0.status.success(), "shard0 run should succeed");

    let out1 = run(&[
        "unbinned-fit-toys",
        "--config",
        config.to_string_lossy().as_ref(),
        "--n-toys",
        "6",
        "--seed",
        "42",
        "--threads",
        "1",
        "--shard",
        "1/2",
        "--output",
        shard1.to_string_lossy().as_ref(),
    ]);
    assert!(out1.status.success(), "shard1 run should succeed");

    // Corrupt non-seed gen field to ensure merge checks full gen config.
    let mut bad: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&shard1).unwrap()).expect("shard1 JSON");
    bad["gen"]["point"] = serde_json::json!("mle");
    std::fs::write(&shard1, serde_json::to_string_pretty(&bad).unwrap()).unwrap();

    let merged_out = run(&[
        "unbinned-merge-toys",
        shard0.to_string_lossy().as_ref(),
        shard1.to_string_lossy().as_ref(),
    ]);
    assert!(
        !merged_out.status.success(),
        "merge should fail for mismatched gen config, stdout={}, stderr={}",
        String::from_utf8_lossy(&merged_out.stdout),
        String::from_utf8_lossy(&merged_out.stderr)
    );
    let stderr = String::from_utf8_lossy(&merged_out.stderr);
    assert!(
        stderr.contains("different gen config"),
        "stderr should mention gen mismatch, got: {stderr}"
    );

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&shard0);
    let _ = std::fs::remove_file(&shard1);
}
