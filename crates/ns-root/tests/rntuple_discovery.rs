//! Integration tests: discover and extract RNTuple top-level payloads.

use ns_root::{
    RNTUPLE_ENVELOPE_TYPE_FOOTER, RNTUPLE_ENVELOPE_TYPE_HEADER, RNTUPLE_ENVELOPE_TYPE_PAGELIST,
    RNTupleFieldKind, RNTupleScalarType, RootError, RootFile,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

fn unique_tmp_path(stem: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}_{}.root", stem, std::process::id(), nanos))
}

fn write_corrupted_single_u64_offset_page_copy(
    src: &PathBuf,
    position: u64,
    nbytes_on_storage: u32,
    stem: &str,
) -> PathBuf {
    let mut bytes = fs::read(src).expect("failed to read fixture bytes");
    let off = position as usize;
    let nbytes = nbytes_on_storage as usize;
    let end = off + nbytes;
    assert_eq!(nbytes, 8, "expected one-u64 offset page for this fixture");
    assert!(end <= bytes.len(), "page out of bounds in copied bytes");
    bytes[off..off + 8].copy_from_slice(&3u64.to_le_bytes());
    let dst = unique_tmp_path(stem);
    fs::write(&dst, &bytes).expect("failed to write corrupted fixture");
    dst
}

#[test]
fn list_rntuples_and_read_payload_from_rntuple_fixture() {
    let path = fixture_path("rntuple_simple.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open rntuple fixture");
    assert!(f.has_rntuples().expect("has_rntuples failed"));

    let keys = f.list_rntuples().expect("list_rntuples failed");
    assert_eq!(keys.len(), 1, "expected exactly one rntuple key");
    assert_eq!(keys[0].name, "Events");
    assert!(keys[0].class_name.to_ascii_lowercase().contains("rntuple"));

    let payload = f.read_rntuple_payload("Events").expect("read_rntuple_payload failed");
    assert!(!payload.is_empty(), "rntuple payload should not be empty");

    // Values cross-checked against ROOT's own ROOT::RNTuple getters on this fixture.
    let anchor = f.read_rntuple_anchor("Events").expect("read_rntuple_anchor failed");
    assert_eq!(anchor.version_epoch, 1);
    assert_eq!(anchor.version_major, 0);
    assert_eq!(anchor.version_minor, 1);
    assert_eq!(anchor.version_patch, 0);
    assert_eq!(anchor.seek_header, 284);
    assert_eq!(anchor.nbytes_header, 143);
    assert_eq!(anchor.len_header, 243);
    assert_eq!(anchor.seek_footer, 727);
    assert_eq!(anchor.nbytes_footer, 82);
    assert_eq!(anchor.len_footer, 160);
    assert_eq!(anchor.max_key_size, 1_073_741_824);

    let env = f.read_rntuple_envelopes("Events").expect("read_rntuple_envelopes failed");
    assert_eq!(env.anchor, anchor);
    assert_eq!(env.header.len(), anchor.len_header as usize);
    assert_eq!(env.footer.len(), anchor.len_footer as usize);
    assert!(!env.header.is_empty());
    assert!(!env.footer.is_empty());

    let meta =
        f.read_rntuple_metadata_summary("Events").expect("read_rntuple_metadata_summary failed");
    assert_eq!(meta.anchor, anchor);
    assert_eq!(meta.header_envelope.envelope_type, RNTUPLE_ENVELOPE_TYPE_HEADER);
    assert_eq!(meta.footer_envelope.envelope_type, RNTUPLE_ENVELOPE_TYPE_FOOTER);
    assert_eq!(meta.header_envelope.envelope_len as usize, env.header.len());
    assert_eq!(meta.footer_envelope.envelope_len as usize, env.footer.len());
    assert_eq!(meta.header_summary.ntuple_name.as_deref(), Some("Events"));
    assert_eq!(meta.header_summary.writer.as_deref(), Some("ROOT v6.38.00"));
    assert!(meta.header_summary.strings.iter().any(|s| s == "pt"));
    assert!(meta.header_summary.strings.iter().any(|s| s == "n"));
    assert!(
        meta.header_summary
            .field_tokens
            .iter()
            .any(|f| f.name == "pt" && f.type_name.to_ascii_lowercase().contains("float"))
    );
    assert!(
        meta.header_summary
            .field_tokens
            .iter()
            .any(|f| f.name == "n" && f.type_name.to_ascii_lowercase().contains("int"))
    );

    let schema =
        f.read_rntuple_schema_summary("Events").expect("read_rntuple_schema_summary failed");
    assert_eq!(schema.ntuple_name.as_deref(), Some("Events"));
    assert!(schema.fields.iter().any(|f| {
        f.name == "pt"
            && f.scalar_type == Some(RNTupleScalarType::F32)
            && f.kind == RNTupleFieldKind::Primitive
            && f.element_scalar_type.is_none()
            && f.fixed_len.is_none()
    }));
    assert!(schema.fields.iter().any(|f| {
        f.name == "n"
            && f.scalar_type == Some(RNTupleScalarType::I32)
            && f.kind == RNTupleFieldKind::Primitive
            && f.element_scalar_type.is_none()
            && f.fixed_len.is_none()
    }));

    let footer =
        f.read_rntuple_footer_summary("Events").expect("read_rntuple_footer_summary failed");
    assert_eq!(footer.header_xxhash3_le, meta.header_envelope.xxhash3_le);
    assert_eq!(footer.cluster_groups.len(), 1);
    let cg = &footer.cluster_groups[0];
    assert_eq!(cg.min_entry, 0);
    assert_eq!(cg.entry_span, 8);
    assert_eq!(cg.n_clusters, 1);
    assert_eq!(cg.page_list_envelope_len, 164);
    assert_eq!(cg.page_list_locator.nbytes_on_storage, 94);
    assert_eq!(cg.page_list_locator.position, 591);

    let page_list = f
        .read_rntuple_pagelist_envelope("Events", 0)
        .expect("read_rntuple_pagelist_envelope failed");
    assert_eq!(page_list.cluster_group, *cg);
    assert_eq!(page_list.page_list.len(), cg.page_list_envelope_len as usize);
    let info =
        ns_root::parse_rntuple_envelope(&page_list.page_list, Some(RNTUPLE_ENVELOPE_TYPE_PAGELIST))
            .expect("pagelist envelope parse failed");
    assert_eq!(info.envelope_type, RNTUPLE_ENVELOPE_TYPE_PAGELIST);
    assert_eq!(info.envelope_len as usize, page_list.page_list.len());

    let pages =
        f.read_rntuple_pagelist_summary("Events", 0).expect("read_rntuple_pagelist_summary failed");
    assert_eq!(pages.pages.len(), 2);
    assert!(pages.footer_xxhash3_le != 0);
    assert_eq!(pages.pages[0].record_tag_raw, -40);
    assert_eq!(pages.pages[0].repetition_raw, 1);
    assert_eq!(pages.pages[0].element_count_raw, -8);
    assert_eq!(pages.pages[0].nbytes_on_storage, 32);
    assert_eq!(pages.pages[0].position, 469);
    assert_eq!(pages.pages[1].record_tag_raw, -40);
    assert_eq!(pages.pages[1].repetition_raw, 1);
    assert_eq!(pages.pages[1].element_count_raw, -8);
    assert_eq!(pages.pages[1].nbytes_on_storage, 32);
    assert_eq!(pages.pages[1].position, 509);

    let blob = f.read_rntuple_page_blob("Events", 0, 0).expect("read_rntuple_page_blob failed");
    assert_eq!(blob.page, pages.pages[0]);
    assert_eq!(blob.page_blob.len(), 32);
    assert!(!blob.page_blob.is_empty());

    let primitive_cols = f
        .read_rntuple_primitive_columns_f64("Events", 0)
        .expect("read_rntuple_primitive_columns_f64 failed");
    assert_eq!(primitive_cols.len(), 2);
    assert_eq!(primitive_cols[0].field_name, "pt");
    assert_eq!(primitive_cols[0].values, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    assert_eq!(primitive_cols[1].field_name, "n");
    assert_eq!(primitive_cols[1].values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert!(
        f.read_rntuple_fixed_array_columns_f64("Events", 0)
            .expect("read_rntuple_fixed_array_columns_f64 failed")
            .is_empty()
    );
    assert!(
        f.read_rntuple_variable_array_columns_f64("Events", 0)
            .expect("read_rntuple_variable_array_columns_f64 failed")
            .is_empty()
    );
    assert!(
        f.read_rntuple_pair_columns_f64("Events", 0)
            .expect("read_rntuple_pair_columns_f64 failed")
            .is_empty()
    );
    let decoded = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("read_rntuple_decoded_columns_f64 failed");
    assert_eq!(decoded.primitive, primitive_cols);
    assert!(decoded.fixed_arrays.is_empty());
    assert!(decoded.variable_arrays.is_empty());
    assert!(decoded.pairs.is_empty());
    assert!(decoded.pair_scalar_variable.is_empty());
    assert!(decoded.pair_variable_scalar.is_empty());
    assert!(decoded.pair_variable_variable.is_empty());
    let all_clusters = f
        .read_rntuple_decoded_columns_all_clusters_f64("Events")
        .expect("read_rntuple_decoded_columns_all_clusters_f64 failed");
    assert_eq!(all_clusters.len(), 1);
    assert_eq!(all_clusters[0].cluster_group_index, 0);
    assert_eq!(all_clusters[0].min_entry, 0);
    assert_eq!(all_clusters[0].entry_span, 8);
    assert_eq!(all_clusters[0].columns, decoded);

    let err = f
        .read_rntuple_page_blob("Events", 0, 2)
        .expect_err("expected out-of-bounds page_index error");
    assert!(matches!(err, RootError::Deserialization(_)));

    let err = f
        .read_rntuple_pagelist_envelope("Events", 1)
        .expect_err("expected out-of-bounds cluster_group_index error");
    assert!(matches!(err, RootError::Deserialization(_)));
}

#[test]
fn non_rntuple_fixture_reports_no_rntuples_and_rejects_payload_read() {
    let path = fixture_path("simple_tree.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open tree fixture");
    assert!(!f.has_rntuples().expect("has_rntuples failed"));
    assert!(f.list_rntuples().expect("list_rntuples failed").is_empty());

    let err = f.read_rntuple_payload("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f.read_rntuple_anchor("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f.read_rntuple_envelopes("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_metadata_summary("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_schema_summary("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_footer_summary("events").expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_pagelist_envelope("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_pagelist_summary("events", 0).expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_page_blob("events", 0, 0).expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_primitive_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_fixed_array_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_variable_array_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err =
        f.read_rntuple_pair_columns_f64("events", 0).expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_pair_scalar_variable_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_pair_variable_scalar_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_pair_variable_variable_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_decoded_columns_f64("events", 0)
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));

    let err = f
        .read_rntuple_decoded_columns_all_clusters_f64("events")
        .expect_err("expected unsupported class error");
    assert!(matches!(err, RootError::UnsupportedClass(_)));
}

#[test]
fn complex_rntuple_fixture_maps_primitive_array_and_nested_kinds() {
    let path = fixture_path("rntuple_complex.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open complex rntuple fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");

    assert!(schema.fields.iter().any(|fld| {
        fld.name == "pt"
            && fld.kind == RNTupleFieldKind::Primitive
            && fld.scalar_type == Some(RNTupleScalarType::F32)
    }));
    assert!(schema.fields.iter().any(|fld| {
        fld.name == "n"
            && fld.kind == RNTupleFieldKind::Primitive
            && fld.scalar_type == Some(RNTupleScalarType::I32)
    }));
    assert!(schema.fields.iter().any(|fld| {
        fld.name == "arr_fixed"
            && fld.kind == RNTupleFieldKind::FixedArray
            && fld.element_scalar_type == Some(RNTupleScalarType::F32)
            && fld.fixed_len == Some(3)
    }));
    assert!(schema.fields.iter().any(|fld| {
        fld.name == "arr_var"
            && fld.kind == RNTupleFieldKind::VariableArray
            && fld.element_scalar_type == Some(RNTupleScalarType::I32)
    }));
    assert!(
        schema
            .fields
            .iter()
            .any(|fld| fld.name == "pair_ff" && fld.kind == RNTupleFieldKind::Nested)
    );

    let footer = f.read_rntuple_footer_summary("Events").expect("footer summary failed");
    assert_eq!(footer.cluster_groups.len(), 1);
    let cg = &footer.cluster_groups[0];
    assert_eq!(cg.min_entry, 0);
    assert_eq!(cg.entry_span, 1);
    assert_eq!(cg.n_clusters, 1);
    assert_eq!(cg.page_list_envelope_len, 364);
    assert_eq!(cg.page_list_locator.nbytes_on_storage, 124);
    assert_eq!(cg.page_list_locator.position, 764);

    let page_list =
        f.read_rntuple_pagelist_envelope("Events", 0).expect("pagelist envelope failed");
    assert_eq!(page_list.page_list.len(), 364);
    let info =
        ns_root::parse_rntuple_envelope(&page_list.page_list, Some(RNTUPLE_ENVELOPE_TYPE_PAGELIST))
            .expect("pagelist envelope parse failed");
    assert_eq!(info.envelope_type, RNTUPLE_ENVELOPE_TYPE_PAGELIST);

    let pages =
        f.read_rntuple_pagelist_summary("Events", 0).expect("read_rntuple_pagelist_summary failed");
    assert!(pages.footer_xxhash3_le != 0);
    assert_eq!(pages.pages.len(), 7);
    let expected_sizes = [4u32, 4, 12, 8, 12, 4, 4];
    let expected_positions = [618u64, 630, 642, 662, 678, 698, 710];
    for (idx, page) in pages.pages.iter().enumerate() {
        assert_eq!(page.record_tag_raw, -40);
        assert_eq!(page.repetition_raw, 1);
        assert_eq!(page.nbytes_on_storage, expected_sizes[idx]);
        assert_eq!(page.position, expected_positions[idx]);
    }

    let blob = f.read_rntuple_page_blob("Events", 0, 0).expect("read_rntuple_page_blob failed");
    assert_eq!(blob.page, pages.pages[0]);
    assert_eq!(blob.page_blob.len(), 4);
    assert_eq!(
        blob.page_blob,
        42.5f32.to_le_bytes(),
        "first page should hold the pt scalar in this fixture"
    );

    let primitive_cols = f
        .read_rntuple_primitive_columns_f64("Events", 0)
        .expect("read_rntuple_primitive_columns_f64 failed");
    assert_eq!(primitive_cols.len(), 2);
    assert_eq!(primitive_cols[0].field_name, "pt");
    assert_eq!(primitive_cols[0].values, vec![42.5]);
    assert_eq!(primitive_cols[1].field_name, "n");
    assert_eq!(primitive_cols[1].values, vec![7.0]);

    let fixed_cols = f
        .read_rntuple_fixed_array_columns_f64("Events", 0)
        .expect("read_rntuple_fixed_array_columns_f64 failed");
    assert_eq!(fixed_cols.len(), 1);
    assert_eq!(fixed_cols[0].field_name, "arr_fixed");
    assert_eq!(fixed_cols[0].fixed_len, 3);
    assert_eq!(fixed_cols[0].values, vec![vec![1.0, 2.0, 3.0]]);

    let var_cols = f
        .read_rntuple_variable_array_columns_f64("Events", 0)
        .expect("read_rntuple_variable_array_columns_f64 failed");
    assert_eq!(var_cols.len(), 1);
    assert_eq!(var_cols[0].field_name, "arr_var");
    assert_eq!(var_cols[0].values, vec![vec![10.0, 20.0, 30.0]]);

    let pair_cols =
        f.read_rntuple_pair_columns_f64("Events", 0).expect("read_rntuple_pair_columns_f64 failed");
    assert_eq!(pair_cols.len(), 1);
    assert_eq!(pair_cols[0].field_name, "pair_ff");
    assert_eq!(pair_cols[0].values, vec![(4.0, 5.0)]);

    let decoded = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("read_rntuple_decoded_columns_f64 failed");
    assert_eq!(decoded.primitive, primitive_cols);
    assert_eq!(decoded.fixed_arrays, fixed_cols);
    assert_eq!(decoded.variable_arrays, var_cols);
    assert_eq!(decoded.pairs, pair_cols);
    assert!(decoded.pair_scalar_variable.is_empty());
    assert!(decoded.pair_variable_scalar.is_empty());
    assert!(decoded.pair_variable_variable.is_empty());

    let all_clusters = f
        .read_rntuple_decoded_columns_all_clusters_f64("Events")
        .expect("read_rntuple_decoded_columns_all_clusters_f64 failed");
    assert_eq!(all_clusters.len(), 1);
    assert_eq!(all_clusters[0].cluster_group_index, 0);
    assert_eq!(all_clusters[0].min_entry, 0);
    assert_eq!(all_clusters[0].entry_span, 1);
    assert_eq!(all_clusters[0].columns, decoded);
}

#[test]
fn corrupted_variable_array_offsets_fail_with_deserialization_error() {
    let src = fixture_path("rntuple_complex.root");
    assert!(src.exists(), "missing fixture: {}", src.display());

    let original = RootFile::open(&src).expect("failed to open complex fixture");
    let var_cols = original
        .read_rntuple_variable_array_columns_f64("Events", 0)
        .expect("variable-array decode should pass on original fixture");
    assert_eq!(var_cols.len(), 1);
    let offset_page_index = var_cols[0].offset_page_index;
    let page_list =
        original.read_rntuple_pagelist_summary("Events", 0).expect("pagelist summary should parse");
    let offset_page = &page_list.pages[offset_page_index];
    assert_eq!(
        offset_page.nbytes_on_storage, 8,
        "fixture expectation: offset page is one u64 value"
    );

    let mut bytes = fs::read(&src).expect("failed to read fixture bytes");
    let off = offset_page.position as usize;
    let end = off + offset_page.nbytes_on_storage as usize;
    assert!(end <= bytes.len(), "offset page out of bounds in copied bytes");
    bytes[off..off + 8].copy_from_slice(&2u64.to_le_bytes());

    let dst = unique_tmp_path("rntuple_complex_corrupt_offsets");
    fs::write(&dst, &bytes).expect("failed to write corrupted fixture");

    let corrupted = RootFile::open(&dst).expect("failed to open corrupted fixture");
    let err = corrupted
        .read_rntuple_variable_array_columns_f64("Events", 0)
        .expect_err("expected decode failure for corrupted offsets");
    assert!(matches!(err, RootError::Deserialization(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("data page not found") || msg.contains("offset page not found"),
        "unexpected error message for corrupted offsets: {}",
        msg
    );

    let _ = fs::remove_file(dst);
}

#[test]
fn multicluster_fixture_decodes_all_cluster_groups() {
    let path = fixture_path("rntuple_multicluster.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open multicluster rntuple fixture");
    let footer = f.read_rntuple_footer_summary("Events").expect("footer summary failed");
    assert!(footer.cluster_groups.len() >= 2, "expected multiple cluster groups");

    let all_clusters = f
        .read_rntuple_decoded_columns_all_clusters_f64("Events")
        .expect("read_rntuple_decoded_columns_all_clusters_f64 failed");
    assert_eq!(all_clusters.len(), footer.cluster_groups.len());

    let mut flattened_pt = Vec::new();
    let mut flattened_n = Vec::new();
    let mut total_entries = 0u64;

    for (idx, cluster) in all_clusters.iter().enumerate() {
        let cg = &footer.cluster_groups[idx];
        assert_eq!(cluster.cluster_group_index, idx);
        assert_eq!(cluster.min_entry, cg.min_entry);
        assert_eq!(cluster.entry_span, cg.entry_span);
        assert!(cluster.columns.fixed_arrays.is_empty());
        assert!(cluster.columns.variable_arrays.is_empty());
        assert!(cluster.columns.pairs.is_empty());
        assert!(cluster.columns.pair_scalar_variable.is_empty());
        assert!(cluster.columns.pair_variable_scalar.is_empty());
        assert!(cluster.columns.pair_variable_variable.is_empty());

        let pt = cluster
            .columns
            .primitive
            .iter()
            .find(|c| c.field_name == "pt")
            .expect("pt primitive column missing");
        let n = cluster
            .columns
            .primitive
            .iter()
            .find(|c| c.field_name == "n")
            .expect("n primitive column missing");

        assert_eq!(pt.values.len(), cluster.entry_span as usize);
        assert_eq!(n.values.len(), cluster.entry_span as usize);
        flattened_pt.extend_from_slice(&pt.values);
        flattened_n.extend_from_slice(&n.values);
        total_entries += cluster.entry_span;
    }

    assert_eq!(total_entries, 6);
    assert_eq!(flattened_pt, vec![0.25, 1.25, 2.25, 3.25, 4.25, 5.25]);
    assert_eq!(flattened_n, vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn large_mixed_layout_fixture_decodes_all_cluster_groups() {
    let path = fixture_path("rntuple_bench_large.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open large mixed-layout fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(
        schema.fields.iter().any(|fld| fld.kind == RNTupleFieldKind::Primitive),
        "expected at least one primitive field"
    );
    assert!(
        schema.fields.iter().any(|fld| fld.kind == RNTupleFieldKind::VariableArray),
        "expected at least one variable-array field"
    );
    assert!(
        schema.fields.iter().any(|fld| fld.kind == RNTupleFieldKind::Nested),
        "expected at least one nested field"
    );

    let footer = f.read_rntuple_footer_summary("Events").expect("footer summary failed");
    assert_eq!(footer.cluster_groups.len(), 20);

    let all_clusters = f
        .read_rntuple_decoded_columns_all_clusters_f64("Events")
        .expect("all-cluster decode failed on large mixed-layout fixture");
    assert_eq!(all_clusters.len(), footer.cluster_groups.len());

    let mut total_entries = 0u64;
    let mut saw_variable = false;
    let mut saw_nested_pair = false;
    for cluster in &all_clusters {
        let expected = &footer.cluster_groups[cluster.cluster_group_index];
        assert_eq!(cluster.min_entry, expected.min_entry);
        assert_eq!(cluster.entry_span, expected.entry_span);
        assert!(!cluster.columns.primitive.is_empty(), "primitive columns missing");

        total_entries += cluster.entry_span;
        saw_variable |= !cluster.columns.variable_arrays.is_empty();
        saw_nested_pair |= !cluster.columns.pairs.is_empty()
            || !cluster.columns.pair_scalar_variable.is_empty()
            || !cluster.columns.pair_variable_scalar.is_empty()
            || !cluster.columns.pair_variable_variable.is_empty();
    }

    assert_eq!(total_entries, 2_000_000);
    assert!(saw_variable, "expected decoded variable-array columns");
    assert!(saw_nested_pair, "expected decoded nested pair columns");
}

#[test]
fn pair_scalar_variable_fixture_decodes_supported_nested_pair() {
    let path = fixture_path("rntuple_pair_scalar_variable.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open pair-scalar-variable fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(
        schema
            .fields
            .iter()
            .any(|fld| fld.name == "pair_f_vec" && fld.kind == RNTupleFieldKind::Nested)
    );

    let decoded = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("decoded columns should succeed for pair<float,vector<int>>");
    assert_eq!(decoded.primitive.len(), 1);
    assert_eq!(decoded.primitive[0].field_name, "pt");
    assert_eq!(decoded.primitive[0].values, vec![11.5]);
    assert!(decoded.pairs.is_empty());
    assert_eq!(decoded.pair_scalar_variable.len(), 1);
    assert!(decoded.pair_variable_scalar.is_empty());
    assert!(decoded.pair_variable_variable.is_empty());
    assert_eq!(decoded.pair_scalar_variable[0].field_name, "pair_f_vec");
    assert_eq!(decoded.pair_scalar_variable[0].values, vec![(3.0, vec![10.0, 20.0])]);

    let only = f
        .read_rntuple_pair_scalar_variable_columns_f64("Events", 0)
        .expect("pair scalar-variable decode API should succeed");
    assert_eq!(only, decoded.pair_scalar_variable);
}

#[test]
fn pair_variable_scalar_fixture_decodes_supported_nested_pair() {
    let path = fixture_path("rntuple_pair_variable_scalar.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open pair-variable-scalar fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(
        schema
            .fields
            .iter()
            .any(|fld| fld.name == "pair_vec_f" && fld.kind == RNTupleFieldKind::Nested)
    );

    let decoded = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("decoded columns should succeed for pair<vector<int>,float>");
    assert_eq!(decoded.primitive.len(), 1);
    assert_eq!(decoded.primitive[0].field_name, "pt");
    assert_eq!(decoded.primitive[0].values, vec![12.5]);
    assert!(decoded.pairs.is_empty());
    assert!(decoded.pair_scalar_variable.is_empty());
    assert_eq!(decoded.pair_variable_scalar.len(), 1);
    assert!(decoded.pair_variable_variable.is_empty());
    assert_eq!(decoded.pair_variable_scalar[0].field_name, "pair_vec_f");
    assert_eq!(decoded.pair_variable_scalar[0].values, vec![(vec![10.0, 20.0], 6.5)]);

    let only = f
        .read_rntuple_pair_variable_scalar_columns_f64("Events", 0)
        .expect("pair variable-scalar decode API should succeed");
    assert_eq!(only, decoded.pair_variable_scalar);
}

#[test]
fn pair_variable_variable_fixture_decodes_supported_nested_pair() {
    let path = fixture_path("rntuple_pair_variable_variable.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open pair-variable-variable fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(
        schema
            .fields
            .iter()
            .any(|fld| fld.name == "pair_vec_vec" && fld.kind == RNTupleFieldKind::Nested)
    );

    let decoded = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("decoded columns should succeed for pair<vector<int>,vector<float>>");
    assert_eq!(decoded.primitive.len(), 1);
    assert_eq!(decoded.primitive[0].field_name, "pt");
    assert_eq!(decoded.primitive[0].values, vec![13.5]);
    assert!(decoded.pairs.is_empty());
    assert!(decoded.pair_scalar_variable.is_empty());
    assert!(decoded.pair_variable_scalar.is_empty());
    assert_eq!(decoded.pair_variable_variable.len(), 1);
    assert_eq!(decoded.pair_variable_variable[0].field_name, "pair_vec_vec");
    assert_eq!(decoded.pair_variable_variable[0].values, vec![(vec![1.0, 2.0], vec![30.0, 40.5])]);

    let only = f
        .read_rntuple_pair_variable_variable_columns_f64("Events", 0)
        .expect("pair variable-variable decode API should succeed");
    assert_eq!(only, decoded.pair_variable_variable);
}

#[test]
fn corrupted_pair_scalar_variable_offsets_fail_with_deserialization_error() {
    let src = fixture_path("rntuple_pair_scalar_variable.root");
    assert!(src.exists(), "missing fixture: {}", src.display());

    let original = RootFile::open(&src).expect("failed to open pair-scalar-variable fixture");
    let pair_cols = original
        .read_rntuple_pair_scalar_variable_columns_f64("Events", 0)
        .expect("pair scalar-variable decode should pass on original fixture");
    assert_eq!(pair_cols.len(), 1);
    let page_list =
        original.read_rntuple_pagelist_summary("Events", 0).expect("pagelist summary should parse");
    let offset_page = &page_list.pages[pair_cols[0].right_offset_page_index];
    assert_eq!(
        offset_page.nbytes_on_storage, 8,
        "fixture expectation: offset page is one u64 value"
    );

    let dst = write_corrupted_single_u64_offset_page_copy(
        &src,
        offset_page.position,
        offset_page.nbytes_on_storage,
        "rntuple_pair_scalar_variable_corrupt_offsets",
    );
    let corrupted = RootFile::open(&dst).expect("failed to open corrupted fixture");
    let err = corrupted
        .read_rntuple_pair_scalar_variable_columns_f64("Events", 0)
        .expect_err("expected decode failure for corrupted pair offsets");
    assert!(matches!(err, RootError::Deserialization(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("data page not found") || msg.contains("offset page not found"),
        "unexpected error message for corrupted pair offsets: {}",
        msg
    );

    let _ = fs::remove_file(dst);
}

#[test]
fn corrupted_pair_variable_scalar_offsets_fail_with_deserialization_error() {
    let src = fixture_path("rntuple_pair_variable_scalar.root");
    assert!(src.exists(), "missing fixture: {}", src.display());

    let original = RootFile::open(&src).expect("failed to open pair-variable-scalar fixture");
    let pair_cols = original
        .read_rntuple_pair_variable_scalar_columns_f64("Events", 0)
        .expect("pair variable-scalar decode should pass on original fixture");
    assert_eq!(pair_cols.len(), 1);
    let page_list =
        original.read_rntuple_pagelist_summary("Events", 0).expect("pagelist summary should parse");
    let offset_page = &page_list.pages[pair_cols[0].left_offset_page_index];
    assert_eq!(
        offset_page.nbytes_on_storage, 8,
        "fixture expectation: offset page is one u64 value"
    );

    let dst = write_corrupted_single_u64_offset_page_copy(
        &src,
        offset_page.position,
        offset_page.nbytes_on_storage,
        "rntuple_pair_variable_scalar_corrupt_offsets",
    );
    let corrupted = RootFile::open(&dst).expect("failed to open corrupted fixture");
    let err = corrupted
        .read_rntuple_pair_variable_scalar_columns_f64("Events", 0)
        .expect_err("expected decode failure for corrupted pair offsets");
    assert!(matches!(err, RootError::Deserialization(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("data page not found") || msg.contains("offset page not found"),
        "unexpected error message for corrupted pair offsets: {}",
        msg
    );

    let _ = fs::remove_file(dst);
}

#[test]
fn corrupted_pair_variable_variable_offsets_fail_with_deserialization_error() {
    let src = fixture_path("rntuple_pair_variable_variable.root");
    assert!(src.exists(), "missing fixture: {}", src.display());

    let original = RootFile::open(&src).expect("failed to open pair-variable-variable fixture");
    let pair_cols = original
        .read_rntuple_pair_variable_variable_columns_f64("Events", 0)
        .expect("pair variable-variable decode should pass on original fixture");
    assert_eq!(pair_cols.len(), 1);
    let page_list =
        original.read_rntuple_pagelist_summary("Events", 0).expect("pagelist summary should parse");
    let offset_page = &page_list.pages[pair_cols[0].right_offset_page_index];
    assert_eq!(
        offset_page.nbytes_on_storage, 8,
        "fixture expectation: offset page is one u64 value"
    );

    let dst = write_corrupted_single_u64_offset_page_copy(
        &src,
        offset_page.position,
        offset_page.nbytes_on_storage,
        "rntuple_pair_variable_variable_corrupt_offsets",
    );
    let corrupted = RootFile::open(&dst).expect("failed to open corrupted fixture");
    let err = corrupted
        .read_rntuple_pair_variable_variable_columns_f64("Events", 0)
        .expect_err("expected decode failure for corrupted pair offsets");
    assert!(matches!(err, RootError::Deserialization(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("data page not found") || msg.contains("offset page not found"),
        "unexpected error message for corrupted pair offsets: {}",
        msg
    );

    let _ = fs::remove_file(dst);
}

#[test]
fn unsupported_nested_fixture_returns_deterministic_unsupported_class() {
    let path = fixture_path("rntuple_unsupported_nested.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open unsupported nested fixture");
    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(
        schema
            .fields
            .iter()
            .any(|fld| fld.name == "pair_f_pair" && fld.kind == RNTupleFieldKind::Nested)
    );

    let err = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect_err("expected UnsupportedClass for unsupported nested field");
    assert!(matches!(err, RootError::UnsupportedClass(_)));
    let msg = err.to_string();
    assert!(msg.contains("pair_f_pair"), "unexpected error message: {}", msg);
    assert!(msg.contains("unsupported type"), "unexpected error message: {}", msg);
}

#[test]
fn schema_evolution_fixture_keeps_decoding_known_fields_across_cluster_groups() {
    let path = fixture_path("rntuple_schema_evolution.root");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let f = RootFile::open(&path).expect("failed to open schema evolution fixture");
    let footer = f.read_rntuple_footer_summary("Events").expect("footer summary failed");
    assert!(
        footer.cluster_groups.len() >= 2,
        "expected at least two cluster groups for schema evolution fixture"
    );

    let schema = f.read_rntuple_schema_summary("Events").expect("schema summary failed");
    assert!(schema.fields.iter().any(|fld| fld.name == "pt"));
    assert!(schema.fields.iter().any(|fld| fld.name == "n"));

    let g0 = f
        .read_rntuple_decoded_columns_f64("Events", 0)
        .expect("decode should succeed for first cluster group");
    let g1 = f
        .read_rntuple_decoded_columns_f64("Events", 1)
        .expect("decode should succeed for second cluster group");
    assert_eq!(g0.primitive.len(), 1);
    assert_eq!(g1.primitive.len(), 2);
    assert_eq!(g0.primitive[0].field_name, "pt");
    assert_eq!(g0.primitive[0].values, vec![1.0, 2.0]);
    assert!(g0.primitive.iter().all(|col| col.field_name != "n"));
    let g1_pt = g1
        .primitive
        .iter()
        .find(|col| col.field_name == "pt")
        .expect("pt missing in second cluster group");
    let g1_n = g1
        .primitive
        .iter()
        .find(|col| col.field_name == "n")
        .expect("n missing in second cluster group");
    assert_eq!(g1_pt.values, vec![10.0, 11.0]);
    assert_eq!(g1_n.values, vec![100.0, 101.0]);

    let all = f
        .read_rntuple_decoded_columns_all_clusters_f64("Events")
        .expect("all-cluster decode should succeed for schema-evolution fixture");
    assert_eq!(all.len(), footer.cluster_groups.len());
    assert_eq!(all[0].columns, g0);
    assert_eq!(all[1].columns, g1);
}
