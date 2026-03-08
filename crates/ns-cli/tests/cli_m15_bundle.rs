use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn doc_spec_path(name: &str) -> PathBuf {
    repo_root().join("docs/specs").join(name)
}

fn tmp_file_path(suffix: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "nextstat_m15_bundle_{}_{}_{}_{}.{}",
        std::process::id(),
        nanos,
        seq,
        suffix,
        ext
    ));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn build_assessment_table() -> PathBuf {
    let config = doc_spec_path("m15_config_v1.example.json");
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("assessment_table", "json");
    let result = run(&[
        "m15",
        "assessment-table",
        "--config",
        config.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 assessment-table should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_map(assessment_table: &Path) -> PathBuf {
    let config = doc_spec_path("m15_config_v1.example.json");
    let out = tmp_file_path("map", "json");
    let result = run(&[
        "m15",
        "map",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 map should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_mar(map: &Path, assessment_table: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("mar", "json");
    let result = run(&[
        "m15",
        "mar",
        "--map",
        map.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 mar should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

#[test]
fn m15_bundle_writes_deterministic_json() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let assessment_table = build_assessment_table();
    let map = build_map(&assessment_table);
    let mar = build_mar(&map, &assessment_table);
    let out1 = tmp_file_path("one", "json");
    let out2 = tmp_file_path("two", "json");

    for out in [&out1, &out2] {
        let result = run(&[
            "m15",
            "bundle",
            "--config",
            config.to_string_lossy().as_ref(),
            "--assessment-table",
            assessment_table.to_string_lossy().as_ref(),
            "--map",
            map.to_string_lossy().as_ref(),
            "--mar",
            mar.to_string_lossy().as_ref(),
            "--validation-report",
            validation_report.to_string_lossy().as_ref(),
            "--pharma-validation",
            pharma_validation.to_string_lossy().as_ref(),
            "--output",
            out.to_string_lossy().as_ref(),
            "--deterministic",
        ]);
        assert!(
            result.status.success(),
            "m15 bundle should succeed, stderr={}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    let bytes1 = std::fs::read(&out1).expect("first JSON output should exist");
    let bytes2 = std::fs::read(&out2).expect("second JSON output should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("m15_bundle_manifest_v1"));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));
    assert_eq!(v.get("bundle_status").and_then(|x| x.as_str()), Some("complete"));
    assert_eq!(v.pointer("/integrity/all_hashes_present").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(
        v.pointer("/integrity/deterministic_re_render_verified").and_then(|x| x.as_bool()),
        Some(true)
    );
    assert_eq!(
        v.pointer("/integrity/missing_required_roles").and_then(|x| x.as_array()).map(|x| x.len()),
        Some(0)
    );
    assert_eq!(
        v.pointer("/integrity/signoff_roles_complete").and_then(|x| x.as_bool()),
        Some(true)
    );
    assert_eq!(
        v.pointer("/integrity/signoff_roles_distinct").and_then(|x| x.as_bool()),
        Some(true)
    );
    assert_eq!(
        v.pointer("/integrity/missing_signoff_roles").and_then(|x| x.as_array()).map(|x| x.len()),
        Some(0)
    );
    assert_eq!(
        v.pointer("/artifacts/assessment_table/path").and_then(|x| x.as_str()),
        Some("m15_assessment_table.json")
    );
    assert_eq!(v.pointer("/artifacts/map/path").and_then(|x| x.as_str()), Some("m15_map.json"));
    assert_eq!(v.pointer("/artifacts/mar/path").and_then(|x| x.as_str()), Some("m15_mar.json"));
    assert_eq!(
        v.pointer("/artifacts/validation_report/path").and_then(|x| x.as_str()),
        Some("validation_report.json")
    );
    assert_eq!(
        v.pointer("/artifacts/pharma_validation/path").and_then(|x| x.as_str()),
        Some("pharma_validation.json")
    );
    assert_eq!(v.pointer("/files/0/path").and_then(|x| x.as_str()), Some("m15_config.json"));

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&mar);
    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn m15_bundle_marks_draft_when_artifact_drift_is_detected() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let assessment_table = build_assessment_table();
    let map = build_map(&assessment_table);
    let mar = build_mar(&map, &assessment_table);
    let out = tmp_file_path("bundle_drift", "json");

    let mut mar_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mar).expect("mar output should exist"))
            .expect("mar output should be valid JSON");
    mar_json["document_status"] = serde_json::Value::String("draft".to_string());
    std::fs::write(
        &mar,
        serde_json::to_string_pretty(&mar_json).expect("mar json should serialize"),
    )
    .expect("tampered mar should be writable");

    let result = run(&[
        "m15",
        "bundle",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--map",
        map.to_string_lossy().as_ref(),
        "--mar",
        mar.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 bundle should succeed for drifted artifact, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).expect("bundle output should exist"))
            .expect("bundle output should be valid JSON");
    assert_eq!(v.get("bundle_status").and_then(|x| x.as_str()), Some("draft"));
    assert_eq!(
        v.pointer("/integrity/deterministic_re_render_verified").and_then(|x| x.as_bool()),
        Some(false)
    );
    assert_eq!(
        v.pointer("/integrity/missing_required_roles").and_then(|x| x.as_array()).map(|x| x.len()),
        Some(0)
    );

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&mar);
    let _ = std::fs::remove_file(&out);
}
