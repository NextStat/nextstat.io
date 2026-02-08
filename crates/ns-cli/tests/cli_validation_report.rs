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

fn tmp_file_path(suffix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "nextstat_validation_report_{}_{}_{}.json",
        std::process::id(),
        nanos,
        suffix
    ));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn validation_report_writes_json_artifact() {
    let apex2 = fixture_path("apex2_master_min.json");
    let workspace = fixture_path("simple_workspace.json");
    assert!(apex2.exists(), "missing fixture: {}", apex2.display());
    assert!(workspace.exists(), "missing fixture: {}", workspace.display());

    let out_path = tmp_file_path("out");
    let out = run(&[
        "validation-report",
        "--apex2",
        apex2.to_string_lossy().as_ref(),
        "--workspace",
        workspace.to_string_lossy().as_ref(),
        "--out",
        out_path.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        out.status.success(),
        "validation-report should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&out_path).expect("output JSON should exist");
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("validation_report_v1"));
    assert_eq!(v.get("deterministic").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(
        v.pointer("/apex2_summary/overall").and_then(|x| x.as_str()),
        Some("pass")
    );

    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn validation_report_deterministic_is_byte_identical_across_runs() {
    let apex2 = fixture_path("apex2_master_min_plus.json");
    let workspace = fixture_path("simple_workspace.json");
    assert!(apex2.exists(), "missing fixture: {}", apex2.display());
    assert!(workspace.exists(), "missing fixture: {}", workspace.display());

    let out1 = tmp_file_path("out1");
    let out2 = tmp_file_path("out2");

    for out_path in [&out1, &out2] {
        let out = run(&[
            "validation-report",
            "--apex2",
            apex2.to_string_lossy().as_ref(),
            "--workspace",
            workspace.to_string_lossy().as_ref(),
            "--out",
            out_path.to_string_lossy().as_ref(),
            "--deterministic",
        ]);
        assert!(
            out.status.success(),
            "validation-report should succeed, stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let bytes1 = std::fs::read(&out1).expect("first output JSON should exist");
    let bytes2 = std::fs::read(&out2).expect("second output JSON should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(v.get("deterministic").and_then(|x| x.as_bool()), Some(true));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));

    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn validation_report_includes_regulated_review_and_unknown_suite_counts() {
    let apex2 = fixture_path("apex2_master_min_plus.json");
    let workspace = fixture_path("simple_workspace.json");
    assert!(apex2.exists(), "missing fixture: {}", apex2.display());
    assert!(workspace.exists(), "missing fixture: {}", workspace.display());

    let out_path = tmp_file_path("out_plus");
    let out = run(&[
        "validation-report",
        "--apex2",
        apex2.to_string_lossy().as_ref(),
        "--workspace",
        workspace.to_string_lossy().as_ref(),
        "--out",
        out_path.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        out.status.success(),
        "validation-report should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&out_path).expect("output JSON should exist");
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("output should be JSON");

    // `regulated_review` must exist and provide minimum evidence for review workflows.
    assert_eq!(
        v.pointer("/regulated_review/contains_raw_data")
            .and_then(|x| x.as_bool()),
        Some(false)
    );
    assert!(
        v.pointer("/regulated_review/intended_use")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .len()
            > 0
    );
    assert!(
        v.pointer("/regulated_review/risk_based_assurance")
            .and_then(|x| x.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    );

    // Generic suite extraction should compute case counts for suites not hardcoded in Rust.
    assert_eq!(
        v.pointer("/apex2_summary/suites/timeseries/n_cases")
            .and_then(|x| x.as_u64()),
        Some(2)
    );
    assert_eq!(
        v.pointer("/apex2_summary/suites/timeseries/n_ok")
            .and_then(|x| x.as_u64()),
        Some(2)
    );
    assert_eq!(
        v.pointer("/apex2_summary/suites/pharma_reference/n_cases")
            .and_then(|x| x.as_u64()),
        Some(3)
    );
    assert_eq!(
        v.pointer("/apex2_summary/suites/pharma_reference/n_ok")
            .and_then(|x| x.as_u64()),
        Some(3)
    );

    // Evidence strings should include suite statuses when present in the Apex2 master report.
    let items = v
        .pointer("/regulated_review/risk_based_assurance")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let evidence_all: Vec<String> = items
        .iter()
        .flat_map(|item| {
            item.get("evidence")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|s| s.as_str().map(|x| x.to_string()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        evidence_all
            .iter()
            .any(|s| s == "apex2:suite:timeseries:status=ok"),
        "expected evidence to include timeseries status"
    );
    assert!(
        evidence_all
            .iter()
            .any(|s| s == "apex2:suite:pharma_reference:status=ok"),
        "expected evidence to include pharma_reference status"
    );

    let _ = std::fs::remove_file(&out_path);
}
