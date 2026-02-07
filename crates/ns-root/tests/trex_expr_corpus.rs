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
fn fccfitter_corpus_compiles() {
    let path = repo_root().join("tests/fixtures/trex_expr_corpus/fccfitter_exprs.json");
    let text = std::fs::read_to_string(&path).expect("read fccfitter_exprs.json");
    let corpus: Corpus = serde_json::from_str(&text).expect("parse fccfitter_exprs.json");
    assert_eq!(corpus.schema_version, "trex_expr_corpus_v0");

    let mut failures: Vec<(String, String)> = Vec::new();
    for expr in corpus.expressions {
        match CompiledExpr::compile(&expr) {
            Ok(_) => {}
            Err(e) => failures.push((expr, e.to_string())),
        }
    }

    if !failures.is_empty() {
        let mut msg = String::new();
        msg.push_str(&format!("{} expression(s) failed to compile:\n", failures.len()));
        for (expr, err) in failures {
            msg.push_str(&format!("- {expr}\n  {err}\n"));
        }
        panic!("{msg}");
    }
}

