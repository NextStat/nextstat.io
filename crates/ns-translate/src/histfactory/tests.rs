use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::Value;

use super::{bin_edges_by_channel_from_xml, from_xml};

fn fixture_combination_xml() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/histfactory/combination.xml")
}

#[test]
fn histfactory_from_xml_is_deterministic() {
    let xml = fixture_combination_xml();

    let ws1 = from_xml(&xml).expect("from_xml(ws1)");
    let ws2 = from_xml(&xml).expect("from_xml(ws2)");

    // Compare as JSON values so map key ordering cannot cause spurious diffs.
    let v1: Value = serde_json::to_value(ws1).expect("to_value(ws1)");
    let v2: Value = serde_json::to_value(ws2).expect("to_value(ws2)");
    assert_eq!(v1, v2);
}

#[test]
fn histfactory_bin_edges_by_channel_is_deterministic() {
    let xml = fixture_combination_xml();

    let a = bin_edges_by_channel_from_xml(&xml).expect("bin_edges (a)");
    let b = bin_edges_by_channel_from_xml(&xml).expect("bin_edges (b)");

    // HashMap order is non-deterministic; compare via JSON Value map equality.
    let v1: Value = serde_json::to_value(a).expect("to_value(a)");
    let v2: Value = serde_json::to_value(b).expect("to_value(b)");
    assert_eq!(v1, v2);
}

#[test]
fn histfactory_from_xml_is_fast_enough_smoke() {
    let xml = fixture_combination_xml();

    // Very loose bound to catch accidental O(N^2) / pathological IO regressions without flaking CI.
    let t0 = Instant::now();
    let _ = from_xml(&xml).expect("from_xml");
    let elapsed = t0.elapsed();
    assert!(elapsed < Duration::from_secs(10), "HistFactory ingest too slow: {:?}", elapsed);
}
