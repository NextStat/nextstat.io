use sha2::{Digest, Sha256};
use std::path::PathBuf;
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
    p.push(format!("nextstat_m15_map_{}_{}_{}_{}.{}", std::process::id(), nanos, seq, suffix, ext));
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

#[test]
fn m15_map_writes_deterministic_json() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let assessment_table = build_assessment_table();
    let out1 = tmp_file_path("one", "json");
    let out2 = tmp_file_path("two", "json");

    for out in [&out1, &out2] {
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
    }

    let bytes1 = std::fs::read(&out1).expect("first JSON output should exist");
    let bytes2 = std::fs::read(&out2).expect("second JSON output should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("m15_map_v1"));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));
    assert_eq!(v.get("document_status").and_then(|x| x.as_str()), Some("frozen"));
    assert_eq!(
        v.pointer("/profile_requirements/profile_label").and_then(|x| x.as_str()),
        Some("ICH Core")
    );
    assert_eq!(
        v.pointer("/profile_requirements/framing_heading").and_then(|x| x.as_str()),
        Some("ICH Core Planning Framing")
    );
    assert_eq!(v.pointer("/questions/0/question_id").and_then(|x| x.as_str()), Some("QOI-001"));
    assert_eq!(
        v.pointer("/planned_datasets/0/dataset_id").and_then(|x| x.as_str()),
        Some("DS-001")
    );
    assert_eq!(
        v.pointer("/linked_artifacts/assessment_table_ref").and_then(|x| x.as_str()),
        Some("m15_assessment_table.json")
    );
    assert_eq!(
        v.pointer("/linked_artifacts/validation_report_ref").and_then(|x| x.as_str()),
        Some("validation_report.json")
    );
    assert_eq!(v.pointer("/governance/approvers/0").and_then(|x| x.as_str()), Some("Priya Nair"));
    assert_eq!(v.pointer("/signoff/primary_author").and_then(|x| x.as_str()), Some("Elena Voss"));
    assert_eq!(v.pointer("/signoff/qa_reviewer").and_then(|x| x.as_str()), Some("Martin Hale"));
    assert_eq!(v.pointer("/signoff/approver").and_then(|x| x.as_str()), Some("Priya Nair"));
    assert_eq!(v.pointer("/signoff/status").and_then(|x| x.as_str()), Some("reviewed"));

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn m15_map_can_render_markdown() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let assessment_table = build_assessment_table();
    let out = tmp_file_path("markdown", "md");

    let result = run(&[
        "m15",
        "map",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 map markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let md = std::fs::read_to_string(&out).expect("markdown output should exist");
    assert!(md.contains("# ICH M15 Model Analysis Plan"));
    assert!(md.contains("QOI-001"));
    assert!(md.contains("Technical Acceptance Criteria"));
    assert!(md.contains("Approvers: Priya Nair"));
    assert!(md.contains("status=reviewed"));

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn m15_map_rejects_non_distinct_signoff_roles() {
    let mut config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(doc_spec_path("m15_config_v1.example.json")).unwrap(),
    )
    .expect("config example should parse");
    config["review_plan"]["approver"] = serde_json::Value::String("Martin Hale".to_string());
    let config_path = tmp_file_path("invalid_signoff", "json");
    let config_bytes = serde_json::to_vec_pretty(&config).expect("config should serialize");
    std::fs::write(&config_path, &config_bytes).expect("config temp file should be writable");

    let assessment_table = build_assessment_table();
    let mut assessment_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&assessment_table).unwrap())
            .expect("assessment table should parse");
    let config_sha = {
        let mut hasher = Sha256::new();
        hasher.update(&config_bytes);
        format!("{:x}", hasher.finalize())
    };
    assessment_json["source_config_sha256"] = serde_json::Value::String(config_sha);
    std::fs::write(
        &assessment_table,
        serde_json::to_string_pretty(&assessment_json).expect("assessment table should serialize"),
    )
    .expect("assessment table temp file should be writable");
    let out = tmp_file_path("invalid_signoff_out", "json");
    let result = run(&[
        "m15",
        "map",
        "--config",
        config_path.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);

    assert!(!result.status.success(), "m15 map should reject non-distinct signoff roles");
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("review_plan roles must be assigned to distinct people"),
        "stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&out);
}
