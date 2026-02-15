//! Integration tests: discover and extract RNTuple top-level payloads.

use ns_root::{
    RNTUPLE_ENVELOPE_TYPE_FOOTER, RNTUPLE_ENVELOPE_TYPE_HEADER, RNTupleFieldKind,
    RNTupleScalarType, RootError, RootFile,
};
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
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
}
