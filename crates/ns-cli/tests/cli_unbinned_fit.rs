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
