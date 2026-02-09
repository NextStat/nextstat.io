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

fn sum_bins(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[test]
fn viz_distributions_smoke_histfactory_fixture() {
    let ws = fixture_path("histfactory/workspace.json");
    let xml = fixture_path("histfactory/combination.xml");
    assert!(ws.exists(), "missing fixture: {}", ws.display());
    assert!(xml.exists(), "missing fixture: {}", xml.display());

    // 1) Fit -> fit.json
    let fit_out = tmp_path("fit_histfactory.json");
    let out_fit = run(&[
        "fit",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--output",
        fit_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_fit.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out_fit.stderr)
    );

    // 2) viz distributions -> artifact.json
    let artifact_out = tmp_path("viz_distributions.json");
    let out_viz = run(&[
        "viz",
        "distributions",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--histfactory-xml",
        xml.to_string_lossy().as_ref(),
        "--fit",
        fit_out.to_string_lossy().as_ref(),
        "--output",
        artifact_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_viz.status.success(),
        "viz distributions should succeed, stderr={}",
        String::from_utf8_lossy(&out_viz.stderr)
    );

    let artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&artifact_out).unwrap()).unwrap();
    assert_eq!(
        artifact.get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_distributions_v0")
    );

    let channels =
        artifact.get("channels").and_then(|v| v.as_array()).expect("channels should be an array");
    assert_eq!(channels.len(), 1, "expected 1 channel in fixture");
    let ch = &channels[0];

    let edges =
        ch.get("bin_edges").and_then(|v| v.as_array()).expect("bin_edges should be an array");
    let data_y = ch.get("data_y").and_then(|v| v.as_array()).expect("data_y should be an array");
    assert_eq!(edges.len(), data_y.len() + 1, "edges length must be n_bins+1");

    let samples = ch.get("samples").and_then(|v| v.as_array()).expect("samples should be an array");
    assert_eq!(samples.len(), 2, "expected 2 samples in fixture");

    // Prefit should match nominal workspace values for this fixture.
    let mut prefit_by_name: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for s in samples {
        let name = s.get("name").and_then(|v| v.as_str()).unwrap().to_string();
        let prefit = s
            .get("prefit_y")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        prefit_by_name.insert(name, prefit);
    }
    let sig = prefit_by_name.get("signal").expect("missing signal");
    let bkg = prefit_by_name.get("background").expect("missing background");
    assert_eq!(sig, &vec![5.0, 10.0, 3.0]);
    assert_eq!(bkg, &vec![10.0, 18.0, 9.0]);

    let total_prefit = ch
        .get("total_prefit_y")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(total_prefit, sum_bins(sig, bkg));

    let _ = std::fs::remove_file(&fit_out);
    let _ = std::fs::remove_file(&artifact_out);
}
