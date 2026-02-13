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

fn run_json(args: &[&str]) -> serde_json::Value {
    let out = run(args);
    if !out.status.success() {
        panic!(
            "nextstat failed: status={}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON")
}

fn must_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("expected '{key}' as f64, got: {v}"))
}

fn must_obj<'a>(
    v: &'a serde_json::Value,
    key: &str,
) -> &'a serde_json::Map<String, serde_json::Value> {
    v.get(key)
        .and_then(|x| x.as_object())
        .unwrap_or_else(|| panic!("expected '{key}' as object, got: {v}"))
}

fn must_array<'a>(v: &'a serde_json::Value, key: &str) -> &'a Vec<serde_json::Value> {
    v.get(key)
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("expected '{key}' as array, got: {v}"))
}

fn find_param(v: &serde_json::Value, name: &str) -> f64 {
    let names = must_array(v, "parameter_names");
    let bestfit = must_array(v, "bestfit");
    let idx = names
        .iter()
        .position(|n| n.as_str() == Some(name))
        .unwrap_or_else(|| panic!("missing parameter '{name}' in parameter_names: {v}"));
    bestfit[idx]
        .as_f64()
        .unwrap_or_else(|| panic!("bestfit[{idx}] should be f64, got: {bestfit:?}"))
}

fn ws_sample_sum_yield(workspace: &serde_json::Value, channel: &str, sample: &str) -> f64 {
    let channels = workspace
        .get("channels")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("workspace.channels should be an array, got: {workspace}"));

    let ch = channels
        .iter()
        .find(|c| c.get("name").and_then(|x| x.as_str()) == Some(channel))
        .unwrap_or_else(|| panic!("missing channel '{channel}' in workspace: {workspace}"));
    let samples = ch
        .get("samples")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("workspace.channels[].samples should be an array, got: {ch}"));
    let s = samples
        .iter()
        .find(|ss| ss.get("name").and_then(|x| x.as_str()) == Some(sample))
        .unwrap_or_else(|| panic!("missing sample '{sample}' in channel '{channel}': {ch}"));
    let data = s
        .get("data")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("sample.data should be an array, got: {s}"));

    data.iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("sample.data elements must be f64: {s}")))
        .sum()
}

#[test]
fn unbinned_matches_binned_baselines_on_histogram_from_tree() {
    // We reuse the committed ROOT fixture (no external Python/ROOT required).
    let root = fixture_path("simple_tree.root");
    if !root.exists() {
        eprintln!("Fixture not found: run `python3 tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    // 1) Build a binned pyhf workspace from a tiny TREx config.
    //    Key: nominal weights are unweighted counts (Weight=1.0), but the JES systematic uses
    //    ratio weights (w_up/w_nom) so both binned and unbinned can share the same definition.
    let cfg_path = tmp_path("unbinned_binned_baseline.config");
    let out_dir = tmp_path("unbinned_binned_baseline_out");
    let _ = std::fs::remove_file(&cfg_path);
    let _ = std::fs::remove_dir_all(&out_dir);

    let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: "SR"
Type: SIGNAL
Variable: mbb
Binning: 0; 50, 100;150;200;300
Selection: njet >= 4 && mbb > 0

Sample: "signal"
Type: SIGNAL
Title: "Signal sample"
File: tests/fixtures/simple_tree.root
Weight: 1.0
Regions: SR
NormFactor: mu
StatError: false

Systematic: "jes"
Type: weight
Samples: signal
Regions: SR
WeightUp: weight_jes_up / weight_mc
WeightDown: weight_jes_down / weight_mc
"#;
    std::fs::write(&cfg_path, cfg).unwrap();

    let repo = repo_root();
    let out = run(&[
        "build-hists",
        "--config",
        cfg_path.to_string_lossy().as_ref(),
        "--base-dir",
        repo.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "build-hists should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let ws_path = out_dir.join("workspace.json");
    assert!(ws_path.exists(), "missing workspace.json: {}", ws_path.display());
    let ws: serde_json::Value = serde_json::from_slice(&std::fs::read(&ws_path).unwrap())
        .expect("workspace.json must be JSON");

    // Signal base yield at mu=1, jes=0 is the sample's nominal total yield.
    let base_yield = ws_sample_sum_yield(&ws, "SR", "signal");
    assert!(base_yield.is_finite() && base_yield > 0.0, "invalid base_yield={base_yield}");

    // 2) Build an equivalent unbinned spec that reads the same event store and builds a
    //    histogram_from_tree PDF with the same binning + JES weight systematic.
    let spec_path = tmp_path("unbinned_binned_baseline_spec.json");
    let _ = std::fs::remove_file(&spec_path);

    let root_str = root.to_string_lossy().to_string();
    let selection = "njet >= 4 && mbb > 0";
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 10.0] },
                { "name": "jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [
            {
                "name": "SR",
                "data": { "file": root_str, "tree": "events", "selection": selection },
                "observables": [
                    { "name": "mbb", "bounds": [0.0, 300.0] }
                ],
                "processes": [
                    {
                        "name": "signal",
                        "pdf": {
                            "type": "histogram_from_tree",
                            "observable": "mbb",
                            "bin_edges": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
                            "source": { "file": root_str, "tree": "events", "selection": selection },
                            "weight_systematics": [
                                {
                                    "param": "jes",
                                    "up": "weight_jes_up / weight_mc",
                                    "down": "weight_jes_down / weight_mc",
                                    "interp": "code4p",
                                    "apply_to_shape": true,
                                    "apply_to_yield": true
                                }
                            ]
                        },
                        "yield": { "type": "scaled", "base_yield": base_yield, "scale": "mu" }
                    }
                ]
            }
        ]
    });
    std::fs::write(&spec_path, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    // 3) Fit parity: mu_hat should match (and be near 1 since data is Asimov=sum(samples)).
    let b_fit = run_json(&["fit", "--input", ws_path.to_string_lossy().as_ref(), "--threads", "1"]);
    let u_fit = run_json(&[
        "unbinned-fit",
        "--config",
        spec_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    let b_mu = find_param(&b_fit, "mu");
    let u_mu = find_param(&u_fit, "mu");
    assert!((b_mu - u_mu).abs() < 1e-6, "mu_hat mismatch: binned={b_mu} unbinned={u_mu}");
    assert!((u_mu - 1.0).abs() < 1e-3, "unexpected unbinned mu_hat={u_mu} (expected ~1.0)");

    // 4) Hypotest baseline: compare q_mu at a fixed mu_test.
    let mu_test = 2.0;
    let b_ht = run_json(&[
        "hypotest",
        "--input",
        ws_path.to_string_lossy().as_ref(),
        "--mu",
        &mu_test.to_string(),
        "--threads",
        "1",
    ]);
    let u_ht = run_json(&[
        "unbinned-hypotest",
        "--config",
        spec_path.to_string_lossy().as_ref(),
        "--mu",
        &mu_test.to_string(),
        "--threads",
        "1",
    ]);
    let b_q = must_f64(&b_ht, "q_mu");
    let u_q = must_f64(&u_ht, "q_mu");
    assert!(b_q.is_finite() && b_q >= 0.0, "invalid binned q_mu={b_q}");
    assert!(u_q.is_finite() && u_q >= 0.0, "invalid unbinned q_mu={u_q}");
    assert!(
        (b_q - u_q).abs() < 1e-6,
        "q_mu mismatch at mu_test={mu_test}: binned={b_q} unbinned={u_q}"
    );

    // 5) Ranking baseline: `jes` should be present and match in impact.
    let b_rank = run_json(&[
        "viz",
        "ranking",
        "--input",
        ws_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    let u_rank = run_json(&[
        "unbinned-ranking",
        "--config",
        spec_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);

    let b_names = must_array(&b_rank, "names");
    assert_eq!(b_names.len(), 1, "expected exactly one ranked nuisance, got: {b_rank}");
    assert_eq!(b_names[0].as_str(), Some("jes"), "unexpected binned nuisance name: {b_rank}");

    let u_ranking = must_obj(&u_rank, "ranking");
    let u_names = u_ranking
        .get("names")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("unbinned ranking.names should be array, got: {u_rank}"));
    assert_eq!(u_names.len(), 1, "expected exactly one ranked nuisance, got: {u_rank}");
    assert_eq!(u_names[0].as_str(), Some("jes"), "unexpected unbinned nuisance name: {u_rank}");

    let b_d_up = must_array(&b_rank, "delta_mu_up")[0].as_f64().unwrap();
    let b_d_dn = must_array(&b_rank, "delta_mu_down")[0].as_f64().unwrap();
    let u_d_up = u_ranking
        .get("delta_mu_up")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("unbinned ranking.delta_mu_up[0] missing: {u_rank}"));
    let u_d_dn = u_ranking
        .get("delta_mu_down")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("unbinned ranking.delta_mu_down[0] missing: {u_rank}"));

    assert!(
        (b_d_up - u_d_up).abs() < 1e-6,
        "delta_mu_up mismatch: binned={b_d_up} unbinned={u_d_up}"
    );
    assert!(
        (b_d_dn - u_d_dn).abs() < 1e-6,
        "delta_mu_down mismatch: binned={b_d_dn} unbinned={u_d_dn}"
    );

    // Cleanup best-effort.
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&cfg_path);
    let _ = std::fs::remove_file(&spec_path);
}
