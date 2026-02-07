//! Offline expression corpus tests for TREx/ROOT compatibility.

use ns_root::CompiledExpr;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(serde::Deserialize)]
struct Corpus {
    schema_version: String,
    expressions: Vec<String>,
}

#[test]
fn trex_expr_corpora_compile() {
    let dir = repo_root().join("tests/fixtures/trex_expr_corpus");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read trex_expr_corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|s| s.to_str()).unwrap_or("").ends_with("_exprs.json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "expected at least one *_exprs.json under {}", dir.display());

    let mut failures: Vec<(String, String, String)> = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("read {}", path.display()));
        let corpus: Corpus =
            serde_json::from_str(&text).unwrap_or_else(|_| panic!("parse {}", path.display()));
        assert_eq!(
            corpus.schema_version,
            "trex_expr_corpus_v0",
            "unexpected schema_version in {}",
            path.display()
        );

        for expr in corpus.expressions {
            match CompiledExpr::compile(&expr) {
                Ok(_) => {}
                Err(e) => failures.push((path.display().to_string(), expr, e.to_string())),
            }
        }
    }

    if !failures.is_empty() {
        let mut msg = String::new();
        msg.push_str(&format!("{} expression(s) failed to compile:\n", failures.len()));
        for (path, expr, err) in failures {
            msg.push_str(&format!("- [{path}] {expr}\n  {err}\n"));
        }
        panic!("{msg}");
    }
}
