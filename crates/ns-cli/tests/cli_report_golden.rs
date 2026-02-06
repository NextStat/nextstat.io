use serde_json::Value;
use std::path::{Path, PathBuf};
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

fn golden_dir() -> PathBuf {
    repo_root().join("tests/fixtures/trex_report_goldens/histfactory_v0")
}

fn tmp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, name));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap())
        .unwrap_or_else(|e| panic!("invalid JSON at {}: {}", path.display(), e))
}

fn write_pretty_json(path: &Path, v: &Value) {
    let bytes = serde_json::to_vec_pretty(v).expect("serialize JSON");
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {}", path.display(), e));
}

fn sanitize_artifact(mut v: Value) -> Value {
    // Remove nondeterministic and environment-dependent fields.
    if let Some(meta) = v.get_mut("meta").and_then(|m| m.as_object_mut()) {
        meta.remove("created_unix_ms");
        meta.remove("tool_version");
        if let Some(parity_mode) = meta.get_mut("parity_mode").and_then(|p| p.as_object_mut()) {
            // `threads` is part of parity mode; keep it, but ensure it is serialized as integer.
            if let Some(t) = parity_mode.get_mut("threads") {
                if let Some(n) = t.as_u64() {
                    *t = Value::from(n);
                }
            }
        }
    }
    v
}

fn assert_json_near(path: &str, got: &Value, want: &Value, atol: f64, rtol: f64) {
    match (got, want) {
        (Value::Null, Value::Null) => {}
        (Value::Bool(a), Value::Bool(b)) => assert_eq!(a, b, "{path}: bool mismatch"),
        (Value::String(a), Value::String(b)) => assert_eq!(a, b, "{path}: string mismatch"),
        (Value::Number(a), Value::Number(b)) => {
            let ga = a.as_f64().unwrap();
            let wb = b.as_f64().unwrap();
            let abs_diff = (ga - wb).abs();
            let allowed = atol + rtol * wb.abs();
            assert!(
                abs_diff <= allowed,
                "{path}: number mismatch got={ga:?} want={wb:?} abs_diff={abs_diff:e} allowed={allowed:e}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(
                a.len(),
                b.len(),
                "{path}: array length mismatch got={} want={}",
                a.len(),
                b.len()
            );
            for (i, (ga, wb)) in a.iter().zip(b.iter()).enumerate() {
                assert_json_near(&format!("{path}[{i}]"), ga, wb, atol, rtol);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut a_keys: Vec<&String> = a.keys().collect();
            let mut b_keys: Vec<&String> = b.keys().collect();
            a_keys.sort();
            b_keys.sort();
            assert_eq!(a_keys, b_keys, "{path}: object keys mismatch");
            for k in a_keys {
                assert_json_near(&format!("{path}.{k}"), &a[k], &b[k], atol, rtol);
            }
        }
        _ => panic!("{path}: JSON type mismatch got={:?} want={:?}", got, want),
    }
}

fn pseudo_fit_json_for_workspace(workspace_path: &Path) -> Value {
    let bytes = std::fs::read(workspace_path).expect("read workspace");
    let ws: ns_translate::pyhf::Workspace =
        serde_json::from_slice(&bytes).expect("parse workspace");
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&ws).expect("build model");

    let n = model.parameters().len();
    let mut names = Vec::with_capacity(n);
    let mut bestfit = Vec::with_capacity(n);
    let mut uncs = Vec::with_capacity(n);
    for p in model.parameters() {
        names.push(p.name.clone());
        bestfit.push(p.constraint_center.unwrap_or(p.init));
        uncs.push(p.constraint_width.unwrap_or(1.0).max(1e-12));
    }
    let mut cov = vec![0.0_f64; n * n];
    for i in 0..n {
        cov[i * n + i] = uncs[i] * uncs[i];
    }

    serde_json::json!({
        "parameter_names": names,
        "poi_index": model.poi_index(),
        "bestfit": bestfit,
        "uncertainties": uncs,
        "nll": 0.0,
        "twice_nll": 0.0,
        "converged": true,
        "n_iter": 0,
        "n_fev": 0,
        "n_gev": 0,
        "covariance": cov,
    })
}

#[test]
fn report_artifacts_match_goldens_histfactory_v0() {
    let ws = fixture_path("histfactory/workspace.json");
    let xml = fixture_path("histfactory/combination.xml");
    assert!(ws.exists(), "missing fixture: {}", ws.display());
    assert!(xml.exists(), "missing fixture: {}", xml.display());

    let work_dir = tmp_dir("report_golden");
    std::fs::create_dir_all(&work_dir).unwrap();
    let out_dir = work_dir.join("out");
    let fit_path = work_dir.join("pseudo_fit.json");

    let fit_json = pseudo_fit_json_for_workspace(&ws);
    write_pretty_json(&fit_path, &fit_json);

    let out = run(&[
        "report",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--histfactory-xml",
        xml.to_string_lossy().as_ref(),
        "--fit",
        fit_path.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--skip-uncertainty",
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "report should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let got_dist = sanitize_artifact(read_json(&out_dir.join("distributions.json")));
    let got_yields = sanitize_artifact(read_json(&out_dir.join("yields.json")));
    let got_pulls = sanitize_artifact(read_json(&out_dir.join("pulls.json")));
    let got_corr = sanitize_artifact(read_json(&out_dir.join("corr.json")));

    let record = std::env::var_os("NS_RECORD_GOLDENS").is_some();
    if record {
        std::fs::create_dir_all(golden_dir()).unwrap();
        write_pretty_json(&golden_dir().join("distributions.json"), &got_dist);
        write_pretty_json(&golden_dir().join("yields.json"), &got_yields);
        write_pretty_json(&golden_dir().join("pulls.json"), &got_pulls);
        write_pretty_json(&golden_dir().join("corr.json"), &got_corr);
        // NOTE: we intentionally do not record `fit.json` (this test supplies `--fit`).
        return;
    }

    let want_dist = read_json(&golden_dir().join("distributions.json"));
    let want_yields = read_json(&golden_dir().join("yields.json"));
    let want_pulls = read_json(&golden_dir().join("pulls.json"));
    let want_corr = read_json(&golden_dir().join("corr.json"));

    // Tight numeric tolerance: these artifacts are deterministic given the fixture + pseudo-fit.
    let atol = 1e-9;
    let rtol = 1e-9;
    assert_json_near("distributions", &got_dist, &want_dist, atol, rtol);
    assert_json_near("yields", &got_yields, &want_yields, atol, rtol);
    assert_json_near("pulls", &got_pulls, &want_pulls, atol, rtol);
    assert_json_near("corr", &got_corr, &want_corr, atol, rtol);

    let _ = std::fs::remove_dir_all(&work_dir);
}
