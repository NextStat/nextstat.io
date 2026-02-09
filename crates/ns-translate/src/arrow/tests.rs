//! Tests for Arrow ingest/export/parquet roundtrip.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, ListBuilder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use super::export::record_batch_to_ipc;
use super::ingest::{ArrowIngestConfig, from_arrow_ipc, from_record_batches};
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
