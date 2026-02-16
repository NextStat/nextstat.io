//! Tests for Arrow ingest/export/parquet roundtrip.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Builder, LargeListBuilder, LargeStringBuilder, ListBuilder,
    StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::export::record_batch_to_ipc;
use super::ingest::{
    ArrowIngestConfig, from_arrow_ipc, from_record_batches, from_record_batches_with_modifiers,
};
use super::parquet::{read_parquet_bytes, write_parquet_bytes};
use crate::pyhf::HistFactoryModel;

/// Build a test RecordBatch with the histogram schema.
fn make_test_batch() -> RecordBatch {
    let mut channel_builder = StringBuilder::new();
    let mut sample_builder = StringBuilder::new();
    let mut yields_builder = ListBuilder::new(Float64Builder::new());
    let mut stat_error_builder = ListBuilder::new(Float64Builder::new());

    // SR / signal: [5.0, 10.0, 15.0]
    channel_builder.append_value("SR");
    sample_builder.append_value("signal");
    let yb = yields_builder.values();
    yb.append_value(5.0);
    yb.append_value(10.0);
    yb.append_value(15.0);
    yields_builder.append(true);
    let sb = stat_error_builder.values();
    sb.append_value(1.0);
    sb.append_value(2.0);
    sb.append_value(3.0);
    stat_error_builder.append(true);

    // SR / background: [100.0, 200.0, 150.0]
    channel_builder.append_value("SR");
    sample_builder.append_value("background");
    let yb = yields_builder.values();
    yb.append_value(100.0);
    yb.append_value(200.0);
    yb.append_value(150.0);
    yields_builder.append(true);
    let sb = stat_error_builder.values();
    sb.append_value(10.0);
    sb.append_value(14.0);
    sb.append_value(12.0);
    stat_error_builder.append(true);

    // CR / background: [500.0, 600.0]
    channel_builder.append_value("CR");
    sample_builder.append_value("background");
    let yb = yields_builder.values();
    yb.append_value(500.0);
    yb.append_value(600.0);
    yields_builder.append(true);
    stat_error_builder.append(false); // null stat_error

    let schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("sample", DataType::Utf8, false),
        Field::new(
            "yields",
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
        Field::new(
            "stat_error",
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(channel_builder.finish()) as ArrayRef,
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
            Arc::new(stat_error_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Build a test RecordBatch using large-offset Arrow types.
fn make_test_batch_large_offsets() -> RecordBatch {
    let mut channel_builder = LargeStringBuilder::new();
    let mut sample_builder = LargeStringBuilder::new();
    let mut yields_builder = LargeListBuilder::new(Float64Builder::new());
    let mut stat_error_builder = LargeListBuilder::new(Float64Builder::new());

    // SR / signal: [5.0, 10.0, 15.0]
    channel_builder.append_value("SR");
    sample_builder.append_value("signal");
    let yb = yields_builder.values();
    yb.append_value(5.0);
    yb.append_value(10.0);
    yb.append_value(15.0);
    yields_builder.append(true);
    let sb = stat_error_builder.values();
    sb.append_value(1.0);
    sb.append_value(2.0);
    sb.append_value(3.0);
    stat_error_builder.append(true);

    // SR / background: [100.0, 200.0, 150.0]
    channel_builder.append_value("SR");
    sample_builder.append_value("background");
    let yb = yields_builder.values();
    yb.append_value(100.0);
    yb.append_value(200.0);
    yb.append_value(150.0);
    yields_builder.append(true);
    let sb = stat_error_builder.values();
    sb.append_value(10.0);
    sb.append_value(14.0);
    sb.append_value(12.0);
    stat_error_builder.append(true);

    // CR / background: [500.0, 600.0]
    channel_builder.append_value("CR");
    sample_builder.append_value("background");
    let yb = yields_builder.values();
    yb.append_value(500.0);
    yb.append_value(600.0);
    yields_builder.append(true);
    stat_error_builder.append(false);

    let schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::LargeUtf8, false),
        Field::new("sample", DataType::LargeUtf8, false),
        Field::new(
            "yields",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
        Field::new(
            "stat_error",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(channel_builder.finish()) as ArrayRef,
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
            Arc::new(stat_error_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap()
}

fn make_modifiers_batches_large_offsets() -> (RecordBatch, RecordBatch) {
    // Yields table (large offsets)
    let mut channel_builder = LargeStringBuilder::new();
    let mut sample_builder = LargeStringBuilder::new();
    let mut yields_builder = LargeListBuilder::new(Float64Builder::new());
    let mut stat_error_builder = LargeListBuilder::new(Float64Builder::new());

    channel_builder.append_value("SR");
    sample_builder.append_value("signal");
    let yb = yields_builder.values();
    yb.append_value(5.0);
    yb.append_value(10.0);
    yields_builder.append(true);
    stat_error_builder.append(false);

    let yields_schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::LargeUtf8, false),
        Field::new("sample", DataType::LargeUtf8, false),
        Field::new(
            "yields",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
        Field::new(
            "stat_error",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
    ]));
    let yields_batch = RecordBatch::try_new(
        yields_schema,
        vec![
            Arc::new(channel_builder.finish()) as ArrayRef,
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
            Arc::new(stat_error_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap();

    // Modifiers table (large offsets)
    let mut mch_builder = LargeStringBuilder::new();
    let mut msa_builder = LargeStringBuilder::new();
    let mut mname_builder = LargeStringBuilder::new();
    let mut mtype_builder = LargeStringBuilder::new();
    let mut data_hi_builder = LargeListBuilder::new(Float64Builder::new());
    let mut data_lo_builder = LargeListBuilder::new(Float64Builder::new());
    let mut data_builder = LargeListBuilder::new(Float64Builder::new());

    mch_builder.append_value("SR");
    msa_builder.append_value("signal");
    mname_builder.append_value("sig_norm");
    mtype_builder.append_value("normsys");

    let hib = data_hi_builder.values();
    hib.append_value(1.1);
    data_hi_builder.append(true);

    let lob = data_lo_builder.values();
    lob.append_value(0.9);
    data_lo_builder.append(true);

    data_builder.append(false);

    let mods_schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::LargeUtf8, false),
        Field::new("sample", DataType::LargeUtf8, false),
        Field::new("modifier_name", DataType::LargeUtf8, false),
        Field::new("modifier_type", DataType::LargeUtf8, false),
        Field::new(
            "data_hi",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
        Field::new(
            "data_lo",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
        Field::new(
            "data",
            DataType::LargeList(Arc::new(Field::new_list_field(DataType::Float64, true))),
            true,
        ),
    ]));

    let mods_batch = RecordBatch::try_new(
        mods_schema,
        vec![
            Arc::new(mch_builder.finish()) as ArrayRef,
            Arc::new(msa_builder.finish()) as ArrayRef,
            Arc::new(mname_builder.finish()) as ArrayRef,
            Arc::new(mtype_builder.finish()) as ArrayRef,
            Arc::new(data_hi_builder.finish()) as ArrayRef,
            Arc::new(data_lo_builder.finish()) as ArrayRef,
            Arc::new(data_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap();

    (yields_batch, mods_batch)
}

#[test]
fn test_ingest_basic() {
    let batch = make_test_batch();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    assert_eq!(ws.channels.len(), 2);
    assert_eq!(ws.observations.len(), 2);
    assert_eq!(ws.measurements.len(), 1);
    assert_eq!(ws.measurements[0].config.poi, "mu");

    // Channels sorted alphabetically
    assert_eq!(ws.channels[0].name, "CR");
    assert_eq!(ws.channels[1].name, "SR");

    // CR has 1 sample (background), 2 bins
    assert_eq!(ws.channels[0].samples.len(), 1);
    assert_eq!(ws.channels[0].samples[0].name, "background");
    assert_eq!(ws.channels[0].samples[0].data, vec![500.0, 600.0]);

    // SR has 2 samples, 3 bins
    assert_eq!(ws.channels[1].samples.len(), 2);

    // SR observation = Asimov (sum of yields)
    let sr_obs = &ws.observations[1];
    assert_eq!(sr_obs.name, "SR");
    assert_eq!(sr_obs.data, vec![105.0, 210.0, 165.0]);
}

#[test]
fn test_ingest_large_offsets_basic() {
    let batch = make_test_batch_large_offsets();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    assert_eq!(ws.channels.len(), 2);
    assert_eq!(ws.observations.len(), 2);
    assert_eq!(ws.measurements[0].config.poi, "mu");

    let sr_obs = ws.observations.iter().find(|o| o.name == "SR").unwrap();
    assert_eq!(sr_obs.data, vec![105.0, 210.0, 165.0]);
}

#[test]
fn test_ingest_with_custom_observations() {
    let batch = make_test_batch();
    let mut obs = std::collections::HashMap::new();
    obs.insert("SR".to_string(), vec![110.0, 215.0, 170.0]);
    obs.insert("CR".to_string(), vec![510.0, 590.0]);

    let config = ArrowIngestConfig { observations: Some(obs), ..Default::default() };

    let ws = from_record_batches(&[batch], &config).unwrap();

    assert_eq!(ws.observations[0].data, vec![510.0, 590.0]); // CR
    assert_eq!(ws.observations[1].data, vec![110.0, 215.0, 170.0]); // SR
}

#[test]
fn test_ingest_signal_gets_normfactor() {
    let batch = make_test_batch();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    // Find the signal sample in SR channel
    let sr = &ws.channels[1]; // SR (alphabetical)
    let signal = sr.samples.iter().find(|s| s.name == "signal").unwrap();

    // Signal should have a normfactor modifier
    assert!(signal.modifiers.iter().any(|m| matches!(m,
        crate::pyhf::schema::Modifier::NormFactor { name, .. } if name == "mu"
    )));
}

#[test]
fn test_ingest_staterror_modifier() {
    let batch = make_test_batch();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    let sr = &ws.channels[1];
    let signal = sr.samples.iter().find(|s| s.name == "signal").unwrap();

    // Signal in SR should have staterror modifier
    let se = signal
        .modifiers
        .iter()
        .find(|m| matches!(m, crate::pyhf::schema::Modifier::StatError { .. }));
    assert!(se.is_some());
}

#[test]
fn test_ipc_roundtrip() {
    let batch = make_test_batch();

    // Serialize to IPC
    let ipc_bytes = record_batch_to_ipc(&batch).unwrap();
    assert!(!ipc_bytes.is_empty());

    // Deserialize back via ingest
    let config = ArrowIngestConfig::default();
    let ws = from_arrow_ipc(&ipc_bytes, &config).unwrap();

    assert_eq!(ws.channels.len(), 2);
}

#[test]
fn test_parquet_roundtrip() {
    let batch = make_test_batch();

    // Write to Parquet bytes
    let pq_bytes = write_parquet_bytes(std::slice::from_ref(&batch)).unwrap();
    assert!(!pq_bytes.is_empty());

    // Read back
    let batches = read_parquet_bytes(&pq_bytes).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);

    // Full roundtrip: Parquet bytes → Workspace
    let config = ArrowIngestConfig::default();
    let ws = super::parquet::from_parquet_bytes(&pq_bytes, &config).unwrap();
    assert_eq!(ws.channels.len(), 2);
}

#[test]
fn test_workspace_to_model_roundtrip() {
    let batch = make_test_batch();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    // Convert to JSON and build model
    let json = serde_json::to_string(&ws).unwrap();
    let ws2: crate::pyhf::schema::Workspace = serde_json::from_str(&json).unwrap();
    let model = HistFactoryModel::from_workspace(&ws2).unwrap();

    assert!(!model.parameters().is_empty());
    assert!(model.channel_names().len() == 2);
}

#[test]
fn test_missing_column_error() {
    // Build a batch without 'channel' column
    let mut sample_builder = StringBuilder::new();
    let mut yields_builder = ListBuilder::new(Float64Builder::new());

    sample_builder.append_value("signal");
    let yb = yields_builder.values();
    yb.append_value(1.0);
    yields_builder.append(true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("sample", DataType::Utf8, false),
        Field::new(
            "yields",
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap();

    let config = ArrowIngestConfig::default();
    let result = from_record_batches(&[batch], &config);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("channel"));
}

#[test]
fn test_inconsistent_bins_error() {
    let mut channel_builder = StringBuilder::new();
    let mut sample_builder = StringBuilder::new();
    let mut yields_builder = ListBuilder::new(Float64Builder::new());

    // SR / signal: 3 bins
    channel_builder.append_value("SR");
    sample_builder.append_value("signal");
    let yb = yields_builder.values();
    yb.append_value(1.0);
    yb.append_value(2.0);
    yb.append_value(3.0);
    yields_builder.append(true);

    // SR / background: 2 bins (inconsistent!)
    channel_builder.append_value("SR");
    sample_builder.append_value("background");
    let yb = yields_builder.values();
    yb.append_value(10.0);
    yb.append_value(20.0);
    yields_builder.append(true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("sample", DataType::Utf8, false),
        Field::new(
            "yields",
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(channel_builder.finish()) as ArrayRef,
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
        ],
    )
    .unwrap();

    let config = ArrowIngestConfig::default();
    let result = from_record_batches(&[batch], &config);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("inconsistent"));
}

#[test]
fn test_parquet_bytes_projected_column_subset() {
    let batch = make_test_batch();
    let pq_bytes = write_parquet_bytes(std::slice::from_ref(&batch)).unwrap();

    // Read only channel + sample columns (no yields, no stat_error)
    let batches =
        super::parquet::read_parquet_bytes_projected(&pq_bytes, &["channel", "sample"], None)
            .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 2);
    assert!(batches[0].schema().column_with_name("channel").is_some());
    assert!(batches[0].schema().column_with_name("sample").is_some());
    assert!(batches[0].schema().column_with_name("yields").is_none());
}

#[test]
fn test_parquet_bytes_projected_row_limit() {
    let batch = make_test_batch();
    let pq_bytes = write_parquet_bytes(std::slice::from_ref(&batch)).unwrap();

    // Read all columns but only first 2 rows
    let batches = super::parquet::read_parquet_bytes_projected(&pq_bytes, &[], Some(2)).unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
}

#[test]
fn test_parquet_bytes_projected_empty_columns_reads_all() {
    let batch = make_test_batch();
    let pq_bytes = write_parquet_bytes(std::slice::from_ref(&batch)).unwrap();

    // Empty columns list = read all
    let batches = super::parquet::read_parquet_bytes_projected(&pq_bytes, &[], None).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_columns(), 4); // channel, sample, yields, stat_error
    assert_eq!(batches[0].num_rows(), 3);
}

#[test]
fn test_yields_from_workspace_roundtrip() {
    // Build workspace from test batch
    let batch = make_test_batch();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&[batch], &config).unwrap();

    // Export yields back to RecordBatch
    let yields_batch = super::export::yields_from_workspace(&ws).unwrap();
    assert_eq!(yields_batch.num_rows(), 3); // 2 SR samples + 1 CR sample

    // Re-ingest and verify consistency
    let ws2 = from_record_batches(&[yields_batch], &config).unwrap();
    assert_eq!(ws2.channels.len(), ws.channels.len());

    for ch in &ws.channels {
        let ch2 = ws2.channels.iter().find(|c| c.name == ch.name).unwrap();
        assert_eq!(ch.samples.len(), ch2.samples.len());
        for s in &ch.samples {
            let s2 = ch2.samples.iter().find(|ss| ss.name == s.name).unwrap();
            assert_eq!(s.data, s2.data);
        }
    }
}

/// Build a test workspace with modifiers for roundtrip testing.
fn make_workspace_with_modifiers() -> crate::pyhf::schema::Workspace {
    use crate::pyhf::schema::*;

    Workspace {
        channels: vec![Channel {
            name: "SR".into(),
            samples: vec![
                Sample {
                    name: "signal".into(),
                    data: vec![5.0, 10.0, 15.0],
                    modifiers: vec![
                        Modifier::NormFactor { name: "mu".into(), data: None },
                        Modifier::HistoSys {
                            name: "jet_unc".into(),
                            data: HistoSysData {
                                hi_data: vec![6.0, 11.0, 16.0],
                                lo_data: vec![4.0, 9.0, 14.0],
                            },
                        },
                    ],
                },
                Sample {
                    name: "background".into(),
                    data: vec![100.0, 200.0, 150.0],
                    modifiers: vec![
                        Modifier::NormSys {
                            name: "lumi_unc".into(),
                            data: NormSysData { hi: 1.05, lo: 0.95 },
                        },
                        Modifier::StatError {
                            name: "staterror_SR".into(),
                            data: vec![10.0, 14.0, 12.0],
                        },
                    ],
                },
            ],
        }],
        observations: vec![Observation { name: "SR".into(), data: vec![110.0, 215.0, 170.0] }],
        measurements: vec![Measurement {
            name: "meas".into(),
            config: MeasurementConfig { poi: "mu".into(), parameters: vec![] },
        }],
        version: Some("1.0.0".into()),
    }
}

#[test]
fn test_modifiers_export_roundtrip() {
    let ws = make_workspace_with_modifiers();

    // Export modifiers to RecordBatch
    let mod_batch = super::export::modifiers_to_record_batch(&ws).unwrap();
    assert!(mod_batch.num_rows() > 0);

    // Verify expected modifier types are present
    let type_col = mod_batch
        .column(mod_batch.schema().index_of("modifier_type").unwrap())
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();
    let types: Vec<&str> = (0..type_col.len()).map(|i| type_col.value(i)).collect();
    assert!(types.contains(&"histosys"));
    assert!(types.contains(&"normsys"));
    assert!(types.contains(&"staterror"));
    assert!(types.contains(&"normfactor"));
}

#[test]
fn test_ingest_with_modifiers_large_offsets() {
    let (yields_batch, mods_batch) = make_modifiers_batches_large_offsets();
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches_with_modifiers(&[yields_batch], &[mods_batch], &config).unwrap();

    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "SR");
    assert_eq!(ws.channels[0].samples.len(), 1);

    let signal = &ws.channels[0].samples[0];
    assert_eq!(signal.name, "signal");
    assert!(
        signal.modifiers.iter().any(|m| matches!(
            m,
            crate::pyhf::schema::Modifier::NormSys { name, .. } if name == "sig_norm"
        )),
        "expected NormSys modifier parsed from LargeUtf8/LargeList batch"
    );
}

#[test]
fn test_workspace_parquet_bytes_roundtrip_with_modifiers() {
    let ws = make_workspace_with_modifiers();

    // Export yields + modifiers to Parquet bytes
    let yields_batch = super::export::yields_from_workspace(&ws).unwrap();
    let mod_batch = super::export::modifiers_to_record_batch(&ws).unwrap();
    let yields_bytes = write_parquet_bytes(&[yields_batch]).unwrap();
    let mod_bytes = write_parquet_bytes(&[mod_batch]).unwrap();

    // Read back and reconstruct workspace
    let yield_batches = read_parquet_bytes(&yields_bytes).unwrap();
    let mod_batches = read_parquet_bytes(&mod_bytes).unwrap();

    let config = ArrowIngestConfig::default();
    let ws2 =
        super::ingest::from_record_batches_with_modifiers(&yield_batches, &mod_batches, &config)
            .unwrap();

    // Verify structure
    assert_eq!(ws2.channels.len(), 1);
    assert_eq!(ws2.channels[0].name, "SR");
    assert_eq!(ws2.channels[0].samples.len(), 2);

    // Verify signal sample data preserved
    let signal = ws2.channels[0].samples.iter().find(|s| s.name == "signal").unwrap();
    assert_eq!(signal.data, vec![5.0, 10.0, 15.0]);

    // Verify signal has histosys modifier from Parquet roundtrip
    let has_histosys = signal.modifiers.iter().any(
        |m| matches!(m, crate::pyhf::schema::Modifier::HistoSys { name, .. } if name == "jet_unc"),
    );
    assert!(has_histosys, "signal should have histosys modifier after roundtrip");

    // Verify background has normsys
    let bkg = ws2.channels[0].samples.iter().find(|s| s.name == "background").unwrap();
    let has_normsys = bkg.modifiers.iter().any(
        |m| matches!(m, crate::pyhf::schema::Modifier::NormSys { name, .. } if name == "lumi_unc"),
    );
    assert!(has_normsys, "background should have normsys modifier after roundtrip");

    // Verify the reconstructed workspace can build a model
    let model = HistFactoryModel::from_workspace(&ws2).unwrap();
    assert!(!model.parameters().is_empty());
}

#[test]
fn test_parquet_mmap_basic() {
    let batch = make_test_batch();

    // Write to a temp file
    let dir = std::env::temp_dir().join("ns_test_mmap_basic");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test.parquet");
    super::parquet::write_parquet(&path, &[batch]).unwrap();

    // Read back via mmap
    let batches = super::parquet::read_parquet_mmap(&path).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 4);

    // Full roundtrip: mmap → Workspace
    let config = ArrowIngestConfig::default();
    let ws = from_record_batches(&batches, &config).unwrap();
    assert_eq!(ws.channels.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_parquet_mmap_projected() {
    let batch = make_test_batch();

    let dir = std::env::temp_dir().join("ns_test_mmap_proj");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test.parquet");
    super::parquet::write_parquet(&path, &[batch]).unwrap();

    // Column projection: only channel + yields
    let batches =
        super::parquet::read_parquet_mmap_projected(&path, &["channel", "yields"], None).unwrap();
    assert_eq!(batches[0].num_columns(), 2);
    assert!(batches[0].schema().column_with_name("channel").is_some());
    assert!(batches[0].schema().column_with_name("yields").is_some());
    assert!(batches[0].schema().column_with_name("sample").is_none());

    // Row limit
    let batches = super::parquet::read_parquet_mmap_projected(&path, &[], Some(1)).unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_parquet_mmap_filtered_row_group_pruning() {
    use arrow::array::Float64Array;

    // Build a batch with 100 rows: mass values from 0.0..100.0
    let mass_vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let pt_vals: Vec<f64> = (0..100).map(|i| (i * 2) as f64).collect();

    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("mass", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("pt", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(Float64Array::from(mass_vals)) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(pt_vals)) as ArrayRef,
        ],
    )
    .unwrap();

    // Write with very small row group size (10 rows) to get 10 row groups.
    let dir = std::env::temp_dir().join("ns_test_mmap_filtered");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("events.parquet");

    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_size(10)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();

    let file = std::fs::File::create(&path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Verify we have multiple row groups.
    {
        let file = std::fs::File::open(&path).unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(reader.metadata().num_row_groups(), 10);
    }

    // Filter: mass in [25.0, 34.0] — should match row groups 2 and 3 (rows 20-29, 30-39).
    let bounds = vec![super::parquet::ColumnBound { column: "mass".into(), lo: 25.0, hi: 34.0 }];

    let batches = super::parquet::read_parquet_mmap_filtered(&path, &[], &bounds, None).unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    // Row groups [20..30) and [30..40) have overlap → 20 rows read.
    assert_eq!(total_rows, 20, "should read exactly 2 row groups (20 rows)");

    // Verify all mass values are in [20, 39] (the row groups that overlap [25, 34]).
    let mass_col = batches[0].column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..mass_col.len() {
        let v = mass_col.value(i);
        assert!((20.0..40.0).contains(&v), "mass value {v} outside expected row group range");
    }

    // Filter: mass in [95.0, 200.0] — should match only row group 9 (rows 90-99).
    let bounds2 = vec![super::parquet::ColumnBound { column: "mass".into(), lo: 95.0, hi: 200.0 }];
    let batches2 = super::parquet::read_parquet_mmap_filtered(&path, &[], &bounds2, None).unwrap();
    let total2: usize = batches2.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total2, 10, "should read exactly 1 row group (10 rows)");

    // Filter: mass in [200.0, 300.0] — no overlap, should return 0 rows.
    let bounds3 = vec![super::parquet::ColumnBound { column: "mass".into(), lo: 200.0, hi: 300.0 }];
    let batches3 = super::parquet::read_parquet_mmap_filtered(&path, &[], &bounds3, None).unwrap();
    let total3: usize = batches3.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total3, 0, "no row groups should match [200, 300]");

    // Filter with projection: only read 'mass' column.
    let batches4 =
        super::parquet::read_parquet_mmap_filtered(&path, &["mass"], &bounds, None).unwrap();
    assert_eq!(batches4[0].num_columns(), 1);
    assert!(batches4[0].schema().column_with_name("mass").is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_parquet_mmap_parallel_decode() {
    use arrow::array::Float64Array;

    // 200 rows, 20 row groups of 10.
    let mass_vals: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("mass", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("weight", arrow::datatypes::DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(Float64Array::from(mass_vals.clone())) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(vec![1.0; 200])) as ArrayRef,
        ],
    )
    .unwrap();

    let dir = std::env::temp_dir().join("ns_test_mmap_parallel");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("events.parquet");

    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_size(10)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Parallel read — all rows, no filter.
    let par_batches = super::parquet::read_parquet_mmap_parallel(&path, &[], &[], None).unwrap();
    let par_total: usize = par_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(par_total, 200);

    // Parallel read with predicate: mass in [50, 79] → row groups 5,6,7 → 30 rows.
    let bounds = vec![super::parquet::ColumnBound { column: "mass".into(), lo: 50.0, hi: 79.0 }];
    let par_filtered =
        super::parquet::read_parquet_mmap_parallel(&path, &[], &bounds, None).unwrap();
    let par_f_total: usize = par_filtered.iter().map(|b| b.num_rows()).sum();
    assert_eq!(par_f_total, 30);

    // Compare with sequential filtered to ensure identical results.
    let seq_filtered =
        super::parquet::read_parquet_mmap_filtered(&path, &[], &bounds, None).unwrap();
    let seq_f_total: usize = seq_filtered.iter().map(|b| b.num_rows()).sum();
    assert_eq!(par_f_total, seq_f_total);

    // Parallel read with row limit.
    let par_limited =
        super::parquet::read_parquet_mmap_parallel(&path, &[], &[], Some(25)).unwrap();
    let par_l_total: usize = par_limited.iter().map(|b| b.num_rows()).sum();
    assert_eq!(par_l_total, 25);

    // Parallel read with projection.
    let par_proj = super::parquet::read_parquet_mmap_parallel(&path, &["mass"], &[], None).unwrap();
    assert_eq!(par_proj[0].num_columns(), 1);
    assert!(par_proj[0].schema().column_with_name("mass").is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_parquet_events_soa_gpu_ready() {
    use arrow::array::Float64Array;

    // 50 events, 5 row groups of 10.
    let n = 50usize;
    let mass: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
    let pt: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 2.0).collect();
    let wt: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();

    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("mass", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("pt", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("weight", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(Float64Array::from(mass.clone())) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(pt.clone())) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(wt.clone())) as ArrayRef,
        ],
    )
    .unwrap();

    let dir = std::env::temp_dir().join("ns_test_events_soa");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("events.parquet");

    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_size(10)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Read as SoA with weights.
    let data =
        super::parquet::read_parquet_events_soa(&path, &["mass", "pt"], &[], Some("weight"), None)
            .unwrap();

    assert_eq!(data.n_events, n);
    assert_eq!(data.n_obs, 2);
    assert_eq!(data.soa.len(), 2 * n);
    assert_eq!(data.column_names, vec!["mass", "pt"]);

    // Verify SoA layout: soa[obs_idx * n_events + event_idx].
    for i in 0..n {
        assert!((data.soa[i] - mass[i]).abs() < 1e-12, "mass SoA mismatch at event {i}");
        assert!((data.soa[n + i] - pt[i]).abs() < 1e-12, "pt SoA mismatch at event {i}");
    }

    // Verify weights.
    let w = data.weights.as_ref().unwrap();
    assert_eq!(w.len(), n);
    for i in 0..n {
        assert!((w[i] - wt[i]).abs() < 1e-12, "weight mismatch at event {i}");
    }

    // Read with predicate pushdown: mass in [120, 130] → row groups 2,3 → 20 events.
    let bounds = vec![super::parquet::ColumnBound { column: "mass".into(), lo: 120.0, hi: 130.0 }];
    let filtered =
        super::parquet::read_parquet_events_soa(&path, &["mass"], &bounds, None, None).unwrap();
    assert_eq!(filtered.n_events, 20);
    assert_eq!(filtered.n_obs, 1);
    assert_eq!(filtered.soa.len(), 20);
    assert!(filtered.weights.is_none());

    // Empty result.
    let bounds_empty =
        vec![super::parquet::ColumnBound { column: "mass".into(), lo: 999.0, hi: 1000.0 }];
    let empty =
        super::parquet::read_parquet_events_soa(&path, &["mass"], &bounds_empty, None, None)
            .unwrap();
    assert_eq!(empty.n_events, 0);
    assert_eq!(empty.soa.len(), 0);

    let _ = std::fs::remove_dir_all(&dir);
}
