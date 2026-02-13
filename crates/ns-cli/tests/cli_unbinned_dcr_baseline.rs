#![cfg(feature = "neural")]

use ns_unbinned::event_parquet::write_event_parquet;
use ns_unbinned::event_store::{EventStore, ObservableSpec};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
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

fn norm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

fn build_hist_counts(edges: &[f64], mu: f64, sigma: f64, total_yield: f64) -> Vec<f64> {
    let mut raw = Vec::with_capacity(edges.len().saturating_sub(1));
    for i in 0..edges.len() - 1 {
        let lo = edges[i];
        let hi = edges[i + 1];
        let w = hi - lo;
        let c = 0.5 * (lo + hi);
        raw.push(norm_pdf(c, mu, sigma) * w);
    }
    let sum = raw.iter().sum::<f64>().max(1e-30);
    raw.iter().map(|v| total_yield * (v / sum)).collect()
}

fn build_event_store_from_bin_centers(edges: &[f64], counts: &[f64]) -> EventStore {
    let mut xs = Vec::<f64>::new();
    for i in 0..edges.len() - 1 {
        let c = 0.5 * (edges[i] + edges[i + 1]);
        let n = counts[i].round().max(0.0) as usize;
        xs.extend(std::iter::repeat_n(c, n));
    }
    let obs = ObservableSpec::branch("mass", (edges[0], *edges.last().unwrap()));
    EventStore::from_columns(vec![obs], vec![("mass".to_string(), xs)], None).unwrap()
}

#[test]
fn dcr_surrogate_significance_matches_binned_histfactory_within_5pct() {
    let manifest = fixture_path("dcr_test/flow_manifest.json");
    if !manifest.exists() {
        eprintln!("SKIP: missing DCR fixtures under tests/fixtures/dcr_test");
        return;
    }

    // Build a small, deterministic dataset: bin-center "Asimov-like" events.
    // Background follows N(0,1) with histosys shift δ=0.5 at alpha=±1 (matches dcr_test fixture).
    // Signal follows N(2,1) and is scaled by POI mu.
    let n_bins = 60usize;
    let lo = -6.0;
    let hi = 6.0;
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges.push(lo + (hi - lo) * (i as f64) / (n_bins as f64));
    }

    let y_bkg = 2000.0;
    let y_sig = 400.0;
    let delta = 0.5;

    let bkg_nom = build_hist_counts(&edges, 0.0, 1.0, y_bkg);
    let bkg_hi = build_hist_counts(&edges, delta, 1.0, y_bkg);
    let bkg_lo = build_hist_counts(&edges, -delta, 1.0, y_bkg);
    let sig = build_hist_counts(&edges, 2.0, 1.0, y_sig);

    let obs: Vec<f64> = bkg_nom.iter().zip(sig.iter()).map(|(b, s)| b + s).collect();

    // Unbinned events: replicate bin centers according to observations.
    let store = build_event_store_from_bin_centers(&edges, &obs);
    let parquet = tmp_path("dcr_baseline_events.parquet");
    let _ = std::fs::remove_file(&parquet);
    write_event_parquet(&store, &parquet).unwrap();

    // Binned workspace (pyhf JSON).
    let ws_path = tmp_path("dcr_baseline_workspace.json");
    let _ = std::fs::remove_file(&ws_path);
    let ws = serde_json::json!({
        "channels": [{
            "name": "SR",
            "samples": [
                {
                    "name": "signal",
                    "data": sig,
                    "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }]
                },
                {
                    "name": "background",
                    "data": bkg_nom,
                    "modifiers": [{
                        "name": "alpha_syst",
                        "type": "histosys",
                        "data": { "hi_data": bkg_hi, "lo_data": bkg_lo }
                    }]
                }
            ]
        }],
        "observations": [{ "name": "SR", "data": obs }],
        "measurements": [{
            "name": "meas",
            "config": { "poi": "mu", "parameters": [] }
        }],
        "version": "1.0.0"
    });
    std::fs::write(&ws_path, serde_json::to_string_pretty(&ws).unwrap()).unwrap();

    // Unbinned spec using DCR surrogate for background.
    let spec_path = tmp_path("dcr_baseline_unbinned_spec.json");
    let _ = std::fs::remove_file(&spec_path);
    let spec = serde_json::json!({
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                { "name": "mu", "init": 1.0, "bounds": [0.0, 5.0] },
                { "name": "mu_sig", "init": 2.0, "bounds": [2.0, 2.0] },
                { "name": "sigma_sig", "init": 1.0, "bounds": [1.0, 1.0] },
                { "name": "alpha_syst", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
            ]
        },
        "channels": [{
            "name": "SR",
            "data": { "file": parquet },
            "observables": [{ "name": "mass", "bounds": [lo, hi] }],
            "processes": [
                {
                    "name": "signal",
                    "pdf": { "type": "gaussian", "observable": "mass", "params": ["mu_sig", "sigma_sig"] },
                    "yield": { "type": "scaled", "base_yield": y_sig, "scale": "mu" }
                },
                {
                    "name": "background",
                    "pdf": { "type": "dcr_surrogate", "manifest": manifest, "systematics": ["alpha_syst"] },
                    "yield": { "type": "fixed", "value": y_bkg }
                }
            ]
        }]
    });
    std::fs::write(&spec_path, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    // Fit parity (rough): mu_hat should be close to 1 for both.
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
    assert!((b_mu - 1.0).abs() < 0.05, "unexpected binned mu_hat={b_mu} (expected ~1.0)");
    assert!((u_mu - 1.0).abs() < 0.05, "unexpected unbinned mu_hat={u_mu} (expected ~1.0)");

    // Hypotest parity: compare significance proxy z = sqrt(q_mu) at mu_test.
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

    let b_z = b_q.sqrt();
    let u_z = u_q.sqrt();
    let rel = (b_z - u_z).abs() / b_z.max(1e-9);
    assert!(
        rel < 0.05,
        "DCR significance mismatch (z=sqrt(q_mu)) at mu_test={mu_test}: binned z={b_z} unbinned z={u_z} (rel={rel})"
    );
}
