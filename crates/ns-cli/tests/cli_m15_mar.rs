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
    p.push(format!("nextstat_m15_mar_{}_{}_{}_{}.{}", std::process::id(), nanos, seq, suffix, ext));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn build_assessment_table() -> PathBuf {
    build_assessment_table_from_config(&doc_spec_path("m15_config_v1.example.json"))
}

fn build_assessment_table_from_config(config: &Path) -> PathBuf {
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
    build_map_from_config(&doc_spec_path("m15_config_v1.example.json"), assessment_table)
}

fn build_map_from_config(config: &Path, assessment_table: &Path) -> PathBuf {
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

#[test]
fn m15_mar_writes_deterministic_json() {
    let assessment_table = build_assessment_table();
    let map = build_map(&assessment_table);
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out1 = tmp_file_path("one", "json");
    let out2 = tmp_file_path("two", "json");

    for out in [&out1, &out2] {
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
    }

    let bytes1 = std::fs::read(&out1).expect("first JSON output should exist");
    let bytes2 = std::fs::read(&out2).expect("second JSON output should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("m15_mar_v1"));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));
    assert_eq!(v.get("document_status").and_then(|x| x.as_str()), Some("reviewed"));
    assert_eq!(
        v.pointer("/profile_requirements/profile_label").and_then(|x| x.as_str()),
        Some("ICH Core")
    );
    assert_eq!(
        v.pointer("/profile_requirements/framing_heading").and_then(|x| x.as_str()),
        Some("ICH Core Results Framing")
    );
    assert_eq!(v.pointer("/criterion_results/0/status").and_then(|x| x.as_str()), Some("met"));
    assert_eq!(v.pointer("/deviations").and_then(|x| x.as_array()).map(|x| x.len()), Some(0));
    assert_eq!(v.pointer("/based_on_map_ref").and_then(|x| x.as_str()), Some("m15_map.json"));
    assert_eq!(
        v.pointer("/linked_artifacts/assessment_table_ref").and_then(|x| x.as_str()),
        Some("m15_assessment_table.json")
    );
    assert_eq!(
        v.pointer("/linked_artifacts/validation_report_ref").and_then(|x| x.as_str()),
        Some("validation_report.json")
    );
    assert_eq!(
        v.pointer("/linked_artifacts/pharma_validation_ref").and_then(|x| x.as_str()),
        Some("pharma_validation.json")
    );
    assert_eq!(
        v.pointer("/questions/0/evidence_refs/0").and_then(|x| x.as_str()),
        Some("validation_report.json#/apex2_summary/overall")
    );
    assert_eq!(
        v.pointer("/questions/0/evidence_refs/1").and_then(|x| x.as_str()),
        Some("pharma_validation.json#/status")
    );
    assert_eq!(
        v.pointer("/criterion_results/0/notes").and_then(|x| x.as_str()),
        Some("evaluated pharma_validation.json/status")
    );
    assert_eq!(
        v.pointer("/criterion_results/1/notes").and_then(|x| x.as_str()),
        Some("evaluated validation_report.json/apex2_summary/overall")
    );
    assert_eq!(v.pointer("/governance/approvers/0").and_then(|x| x.as_str()), Some("Priya Nair"));
    assert_eq!(v.pointer("/signoff/approver").and_then(|x| x.as_str()), Some("Priya Nair"));
    assert_eq!(v.pointer("/signoff/status").and_then(|x| x.as_str()), Some("reviewed"));

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn m15_mar_can_render_markdown() {
    let assessment_table = build_assessment_table();
    let map = build_map(&assessment_table);
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("markdown", "md");

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
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 mar markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let md = std::fs::read_to_string(&out).expect("markdown output should exist");
    assert!(md.contains("# ICH M15 Model Analysis Report"));
    assert!(md.contains("Criterion Results"));
    assert!(md.contains("supported"));
    assert!(md.contains("Approvers: Priya Nair"));
    assert!(md.contains("status=reviewed"));

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn m15_mar_can_emit_approved_status_when_signoff_is_approved() {
    let mut config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(doc_spec_path("m15_config_v1.example.json")).unwrap(),
    )
    .expect("config example should parse");
    config["review_plan"]["status"] = serde_json::Value::String("approved".to_string());
    let config_path = tmp_file_path("approved_config", "json");
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config).expect("config should serialize"),
    )
    .expect("config temp file should be writable");

    let assessment_table = build_assessment_table_from_config(&config_path);
    let map = build_map_from_config(&config_path, &assessment_table);
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("approved", "json");

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
        "m15 mar approved path should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).expect("approved MAR output should exist"))
            .expect("approved MAR output should be JSON");
    assert_eq!(v.get("document_status").and_then(|x| x.as_str()), Some("approved"));
    assert_eq!(v.pointer("/signoff/status").and_then(|x| x.as_str()), Some("approved"));

    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&out);
}
