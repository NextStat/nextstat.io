use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, UInt8Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use ns_inference::churn::{
    ChurnDataConfig, ChurnMappingConfig, CorrectionMethod, churn_diagnostics_report,
    churn_risk_model, churn_uplift, cohort_retention_matrix, generate_churn_dataset,
    ingest_churn_arrays, retention_analysis, segment_comparison_report, survival_uplift_report,
};

// ---------------------------------------------------------------------------
// generate-data
// ---------------------------------------------------------------------------

pub fn cmd_churn_generate_data(
    n_customers: usize,
    n_cohorts: usize,
    max_time: f64,
    treatment_fraction: f64,
    seed: u64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let config = ChurnDataConfig {
        n_customers,
        n_cohorts,
        max_time,
        treatment_fraction,
        seed,
        ..Default::default()
    };
    let ds = generate_churn_dataset(&config)?;

    let records_json: Vec<serde_json::Value> = ds
        .records
        .iter()
        .map(|r| {
            serde_json::json!({
                "customer_id": r.customer_id,
                "cohort": r.cohort,
                "plan": r.plan,
                "region": r.region,
                "usage_score": r.usage_score,
                "support_tickets": r.support_tickets,
                "treated": r.treated,
                "time": r.time,
                "event": r.event,
            })
        })
        .collect();

    let output_json = serde_json::json!({
        "n": ds.records.len(),
        "n_events": ds.events.iter().filter(|&&e| e).count(),
        "config": {
            "n_customers": config.n_customers,
            "n_cohorts": config.n_cohorts,
            "max_time": config.max_time,
            "treatment_fraction": config.treatment_fraction,
            "seed": config.seed,
        },
        "times": ds.times,
        "events": ds.events,
        "groups": ds.groups,
        "treated": ds.treated,
        "covariates": ds.covariates,
        "covariate_names": ["plan_basic", "plan_premium", "usage_score", "support_tickets"],
        "records": records_json,
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_generate_data",
            serde_json::json!({ "seed": seed, "n_customers": n_customers }),
            &PathBuf::from("<generated>"),
            &output_json,
            false,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// retention
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RetentionInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
}

pub fn cmd_churn_retention(
    input: &PathBuf,
    conf_level: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: RetentionInputJson = serde_json::from_str(&raw)?;

    let ra = retention_analysis(&data.times, &data.events, &data.groups, conf_level)?;

    let overall_steps: Vec<serde_json::Value> = ra
        .overall
        .steps
        .iter()
        .map(|s| {
            serde_json::json!({
                "time": s.time, "survival": s.survival,
                "ci_lower": s.ci_lower, "ci_upper": s.ci_upper,
                "n_risk": s.n_risk, "n_events": s.n_events,
            })
        })
        .collect();

    let by_group_json: Vec<serde_json::Value> = ra
        .by_group
        .iter()
        .map(|(g, km)| {
            let steps: Vec<serde_json::Value> = km
                .steps
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "time": s.time, "survival": s.survival,
                        "ci_lower": s.ci_lower, "ci_upper": s.ci_upper,
                    })
                })
                .collect();
            serde_json::json!({
                "group": g,
                "n": km.n,
                "n_events": km.n_events,
                "median": km.median,
                "steps": steps,
            })
        })
        .collect();

    let output_json = serde_json::json!({
        "overall": {
            "n": ra.overall.n,
            "n_events": ra.overall.n_events,
            "median": ra.overall.median,
            "steps": overall_steps,
        },
        "by_group": by_group_json,
        "log_rank": {
            "chi_squared": ra.log_rank.chi_squared,
            "df": ra.log_rank.df,
            "p_value": ra.log_rank.p_value,
        },
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_retention",
            serde_json::json!({ "conf_level": conf_level }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// risk-model
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RiskModelInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    covariates: Vec<Vec<f64>>,
    #[serde(default = "default_covariate_names")]
    covariate_names: Vec<String>,
}

fn default_covariate_names() -> Vec<String> {
    vec!["plan_basic".into(), "plan_premium".into(), "usage_score".into(), "support_tickets".into()]
}

pub fn cmd_churn_risk_model(
    input: &PathBuf,
    conf_level: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: RiskModelInputJson = serde_json::from_str(&raw)?;

    let result = churn_risk_model(
        &data.times,
        &data.events,
        &data.covariates,
        &data.covariate_names,
        conf_level,
    )?;

    let coefs_json: Vec<serde_json::Value> = (0..result.names.len())
        .map(|j| {
            serde_json::json!({
                "name": result.names[j],
                "coefficient": result.coefficients[j],
                "se": result.se[j],
                "hazard_ratio": result.hazard_ratios[j],
                "hr_ci_lower": result.hr_ci_lower[j],
                "hr_ci_upper": result.hr_ci_upper[j],
            })
        })
        .collect();

    let output_json = serde_json::json!({
        "n": result.n,
        "n_events": result.n_events,
        "nll": result.nll,
        "conf_level": conf_level,
        "coefficients": coefs_json,
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_risk_model",
            serde_json::json!({ "conf_level": conf_level }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// bootstrap-hr
// ---------------------------------------------------------------------------

pub fn cmd_churn_bootstrap_hr(
    input: &PathBuf,
    n_bootstrap: usize,
    seed: u64,
    conf_level: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: RiskModelInputJson = serde_json::from_str(&raw)?;

    let names_owned: Vec<String> = data.covariate_names.clone();
    let result = ns_inference::churn::bootstrap_hazard_ratios(
        &data.times,
        &data.events,
        &data.covariates,
        &names_owned,
        n_bootstrap,
        seed,
        conf_level,
    )?;

    let coefs_json: Vec<serde_json::Value> = (0..result.names.len())
        .map(|j| {
            serde_json::json!({
                "name": result.names[j],
                "hr_point": result.hr_point[j],
                "hr_ci_lower": result.hr_ci_lower[j],
                "hr_ci_upper": result.hr_ci_upper[j],
            })
        })
        .collect();

    let output_json = serde_json::json!({
        "n_bootstrap": result.n_bootstrap,
        "n_converged": result.n_converged,
        "elapsed_s": result.elapsed_s,
        "conf_level": conf_level,
        "coefficients": coefs_json,
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_bootstrap_hr",
            serde_json::json!({ "n_bootstrap": n_bootstrap, "seed": seed, "conf_level": conf_level }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// uplift
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct UpliftInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    treated: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}

pub fn cmd_churn_uplift(
    input: &PathBuf,
    horizon: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: UpliftInputJson = serde_json::from_str(&raw)?;

    let result = churn_uplift(&data.times, &data.events, &data.treated, &data.covariates, horizon)?;

    let output_json = serde_json::json!({
        "ate": result.ate,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_treated": result.n_treated,
        "n_control": result.n_control,
        "gamma_critical": result.gamma_critical,
        "horizon": horizon,
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_uplift",
            serde_json::json!({ "horizon": horizon }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ingest (Parquet / CSV → churn JSON)
// ---------------------------------------------------------------------------

/// Read a Parquet or CSV file into Arrow RecordBatches.
fn read_tabular_file(path: &PathBuf) -> Result<Vec<RecordBatch>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    match ext.as_str() {
        "parquet" | "pq" => {
            let batches = ns_translate::arrow::parquet::read_parquet_batches(path)
                .context("failed to read Parquet file")?;
            Ok(batches)
        }
        "csv" | "tsv" | "txt" => read_csv_to_batches(path, if ext == "tsv" { b'\t' } else { b',' }),
        _ => {
            anyhow::bail!("unsupported file extension '.{ext}' — expected .parquet, .csv, or .tsv")
        }
    }
}

/// Read a CSV/TSV file into a single Arrow RecordBatch using the standalone `csv` crate.
///
/// Strategy: read all rows as strings, then build typed Arrow arrays by attempting
/// numeric parsing per column. Columns that parse as f64 become Float64; others stay Utf8.
fn read_csv_to_batches(path: &PathBuf, delimiter: u8) -> Result<Vec<RecordBatch>> {
    use arrow::array::{ArrayRef, Float64Builder, StringBuilder};
    use arrow::datatypes::{Field, Schema};

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let headers: Vec<String> = rdr
        .headers()
        .context("failed to read CSV headers")?
        .iter()
        .map(|h| h.to_string())
        .collect();

    if headers.is_empty() {
        anyhow::bail!("CSV file has no columns");
    }

    let n_cols = headers.len();
    let mut columns: Vec<Vec<String>> = vec![Vec::new(); n_cols];

    for result in rdr.records() {
        let record = result.context("failed to read CSV row")?;
        for (j, field) in record.iter().enumerate() {
            if j < n_cols {
                columns[j].push(field.to_string());
            }
        }
    }

    if columns[0].is_empty() {
        anyhow::bail!("CSV file contains no data rows");
    }

    let n_rows = columns[0].len();
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_cols);

    for (j, col_data) in columns.iter().enumerate() {
        let all_numeric = col_data.iter().all(|s| {
            s.is_empty()
                || s.parse::<f64>().is_ok()
                || s.eq_ignore_ascii_case("true")
                || s.eq_ignore_ascii_case("false")
        });
        let all_bool = col_data.iter().all(|s| {
            s.is_empty()
                || s.eq_ignore_ascii_case("true")
                || s.eq_ignore_ascii_case("false")
                || s == "0"
                || s == "1"
        });

        if all_bool
            && col_data
                .iter()
                .any(|s| s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false"))
        {
            let mut builder = arrow::array::BooleanBuilder::with_capacity(n_rows);
            for s in col_data {
                if s.is_empty() {
                    builder.append_null();
                } else {
                    builder.append_value(s.eq_ignore_ascii_case("true") || s == "1");
                }
            }
            fields.push(Field::new(&headers[j], DataType::Boolean, true));
            arrays.push(std::sync::Arc::new(builder.finish()));
        } else if all_numeric {
            let mut builder = Float64Builder::with_capacity(n_rows);
            for s in col_data {
                if s.is_empty() {
                    builder.append_null();
                } else if s.eq_ignore_ascii_case("true") {
                    builder.append_value(1.0);
                } else if s.eq_ignore_ascii_case("false") {
                    builder.append_value(0.0);
                } else {
                    builder.append_value(s.parse::<f64>().unwrap());
                }
            }
            fields.push(Field::new(&headers[j], DataType::Float64, true));
            arrays.push(std::sync::Arc::new(builder.finish()));
        } else {
            let mut builder = StringBuilder::with_capacity(n_rows, n_rows * 16);
            for s in col_data {
                builder.append_value(s);
            }
            fields.push(Field::new(&headers[j], DataType::Utf8, true));
            arrays.push(std::sync::Arc::new(builder.finish()));
        }
    }

    let schema = std::sync::Arc::new(Schema::new(fields));
    let batch =
        RecordBatch::try_new(schema, arrays).context("failed to build RecordBatch from CSV")?;
    Ok(vec![batch])
}

/// Extract a Float64 column from RecordBatches, coercing from Int64 if needed.
fn extract_f64_column(batches: &[RecordBatch], col_name: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(col_name)
            .ok_or_else(|| anyhow::anyhow!("column '{}' not found", col_name))?;
        match col.data_type() {
            DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.unwrap_or(f64::NAN)));
            }
            DataType::Float32 => {
                let arr = col.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(f64::NAN, |x| x as f64)));
            }
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(f64::NAN, |x| x as f64)));
            }
            DataType::Int32 => {
                let arr = col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(f64::NAN, |x| x as f64)));
            }
            dt => anyhow::bail!(
                "column '{}' has unsupported type {:?} (expected numeric)",
                col_name,
                dt
            ),
        }
    }
    Ok(out)
}

/// Extract a boolean column from RecordBatches, coercing from Int/UInt if needed.
fn extract_bool_column(batches: &[RecordBatch], col_name: &str) -> Result<Vec<bool>> {
    let mut out = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(col_name)
            .ok_or_else(|| anyhow::anyhow!("column '{}' not found", col_name))?;
        match col.data_type() {
            DataType::Boolean => {
                let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                out.extend(arr.iter().map(|v| v.unwrap_or(false)));
            }
            DataType::UInt8 => {
                let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
                out.extend(arr.iter().map(|v| v.is_some_and(|x| x != 0)));
            }
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.is_some_and(|x| x != 0)));
            }
            DataType::Int32 => {
                let arr = col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                out.extend(arr.iter().map(|v| v.is_some_and(|x| x != 0)));
            }
            DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.is_some_and(|x| x != 0.0)));
            }
            dt => anyhow::bail!(
                "column '{}' has unsupported type {:?} (expected bool/int)",
                col_name,
                dt
            ),
        }
    }
    Ok(out)
}

/// Extract an Int64 column (for groups), coercing from Utf8 via hashing.
fn extract_group_column(batches: &[RecordBatch], col_name: &str) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(col_name)
            .ok_or_else(|| anyhow::anyhow!("column '{}' not found", col_name))?;
        match col.data_type() {
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.unwrap_or(0)));
            }
            DataType::Int32 => {
                let arr = col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(0i64, |x| x as i64)));
            }
            DataType::UInt8 => {
                let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(0i64, |x| x as i64)));
            }
            DataType::Utf8 => {
                let arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                let mut label_map: std::collections::HashMap<String, i64> =
                    std::collections::HashMap::new();
                let mut next_id = 0i64;
                for val in arr.iter() {
                    let s = val.unwrap_or("");
                    let id = *label_map.entry(s.to_string()).or_insert_with(|| {
                        let id = next_id;
                        next_id += 1;
                        id
                    });
                    out.push(id);
                }
            }
            dt => anyhow::bail!(
                "column '{}' has unsupported type {:?} (expected int or string)",
                col_name,
                dt
            ),
        }
    }
    Ok(out)
}

/// Extract a UInt8 treated column, coercing from various int/bool types.
fn extract_treated_column(batches: &[RecordBatch], col_name: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(col_name)
            .ok_or_else(|| anyhow::anyhow!("column '{}' not found", col_name))?;
        match col.data_type() {
            DataType::UInt8 => {
                let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
                out.extend(arr.iter().map(|v| v.unwrap_or(0)));
            }
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(0u8, |x| x as u8)));
            }
            DataType::Int32 => {
                let arr = col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(0u8, |x| x as u8)));
            }
            DataType::Boolean => {
                let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                out.extend(arr.iter().map(|v| if v.unwrap_or(false) { 1u8 } else { 0u8 }));
            }
            DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                out.extend(arr.iter().map(|v| v.map_or(0u8, |x| if x != 0.0 { 1 } else { 0 })));
            }
            dt => anyhow::bail!(
                "column '{}' has unsupported type {:?} (expected int/bool)",
                col_name,
                dt
            ),
        }
    }
    Ok(out)
}

pub fn cmd_churn_ingest(
    input: &PathBuf,
    mapping_path: &PathBuf,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    // 1. Read mapping YAML.
    let mapping_raw = std::fs::read_to_string(mapping_path)
        .with_context(|| format!("failed to read mapping file {}", mapping_path.display()))?;
    let mapping: ChurnMappingConfig = serde_yaml_ng::from_str(&mapping_raw)
        .with_context(|| format!("failed to parse mapping YAML {}", mapping_path.display()))?;

    // 2. Validate mapping config.
    if mapping.time_col.is_none() && mapping.signup_ts_col.is_none() {
        anyhow::bail!(
            "mapping must specify either 'time_col' or 'signup_ts_col' + 'observation_end'"
        );
    }

    // 3. Read tabular data.
    let batches = read_tabular_file(input)?;
    if batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0) {
        anyhow::bail!("input file contains no rows");
    }

    // 4. Extract columns per mapping.
    let times = if let Some(ref tc) = mapping.time_col {
        extract_f64_column(&batches, tc)?
    } else {
        anyhow::bail!(
            "timestamp-based duration computation (signup_ts_col + churn_ts_col) is not yet implemented; use time_col with pre-computed durations"
        );
    };

    let events = extract_bool_column(&batches, &mapping.event_col)?;

    let groups = if let Some(ref gc) = mapping.group_col {
        Some(extract_group_column(&batches, gc)?)
    } else {
        None
    };

    let treated = if let Some(ref tc) = mapping.treated_col {
        Some(extract_treated_column(&batches, tc)?)
    } else {
        None
    };

    let covariate_names: Vec<String> = mapping.covariate_cols.clone();
    let covariates: Vec<Vec<f64>> = if covariate_names.is_empty() {
        vec![]
    } else {
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for cname in &covariate_names {
            cols.push(extract_f64_column(&batches, cname)?);
        }
        let n = times.len();
        let p = covariate_names.len();
        let mut row_major: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(p);
            for col in &cols {
                row.push(col[i]);
            }
            row_major.push(row);
        }
        row_major
    };

    // 5. Call core ingestion.
    let result = ingest_churn_arrays(
        &times,
        &events,
        groups.as_deref(),
        treated.as_deref(),
        &covariates,
        &covariate_names,
        mapping.observation_end,
    )?;

    // 6. Print warnings.
    for w in &result.warnings {
        eprintln!("warning: {w}");
    }

    let ds = &result.dataset;

    // 7. Build output JSON compatible with existing churn commands.
    let output_json = serde_json::json!({
        "n": ds.times.len(),
        "n_events": ds.events.iter().filter(|&&e| e).count(),
        "n_dropped": result.n_dropped,
        "times": ds.times,
        "events": ds.events,
        "groups": ds.groups,
        "treated": ds.treated,
        "covariates": ds.covariates,
        "covariate_names": result.covariate_names,
        "mapping": {
            "source": input.display().to_string(),
            "time_col": mapping.time_col,
            "event_col": mapping.event_col,
            "group_col": mapping.group_col,
            "treated_col": mapping.treated_col,
            "covariate_cols": mapping.covariate_cols,
            "observation_end": mapping.observation_end,
            "time_unit": mapping.time_unit,
        },
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_ingest",
            serde_json::json!({
                "mapping": mapping_path.display().to_string(),
                "source": input.display().to_string(),
            }),
            input,
            &output_json,
            false,
        )?;
    }

    eprintln!(
        "Ingested {} rows ({} events, {} censored, {} dropped) from {}",
        ds.times.len(),
        ds.events.iter().filter(|&&e| e).count(),
        ds.events.iter().filter(|&&e| !e).count(),
        result.n_dropped,
        input.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// cohort-matrix
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CohortMatrixInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
}

pub fn cmd_churn_cohort_matrix(
    input: &PathBuf,
    periods_str: &str,
    out_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: CohortMatrixInputJson = serde_json::from_str(&raw)?;

    // Parse period boundaries from comma-separated string.
    let period_boundaries: Vec<f64> = periods_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().with_context(|| format!("invalid period boundary: '{s}'")))
        .collect::<Result<_>>()?;

    let matrix =
        cohort_retention_matrix(&data.times, &data.events, &data.groups, &period_boundaries)?;

    // Build JSON output.
    let output_json = serde_json::to_value(&matrix)?;

    crate::write_json(output, &output_json)?;

    // Write CSV + JSON to out_dir if specified.
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;

        // JSON artifact.
        let json_path = dir.join("cohort_retention_matrix.json");
        let json_str = serde_json::to_string_pretty(&output_json)?;
        std::fs::write(&json_path, json_str)?;

        // CSV artifact: flat table (cohort, period, n_at_risk, n_events, n_censored, retention_rate, cumulative_retention).
        let csv_path = dir.join("cohort_retention_matrix.csv");
        let mut wtr = csv::Writer::from_path(&csv_path)?;
        wtr.write_record([
            "cohort",
            "period_end",
            "n_at_risk",
            "n_events",
            "n_censored",
            "retention_rate",
            "cumulative_retention",
        ])?;

        let write_row = |wtr: &mut csv::Writer<std::fs::File>,
                         label: &str,
                         boundaries: &[f64],
                         row: &ns_inference::churn::CohortMatrixRow|
         -> Result<()> {
            for (j, cell) in row.periods.iter().enumerate() {
                wtr.write_record(&[
                    label.to_string(),
                    format!("{}", boundaries[j]),
                    cell.n_at_risk.to_string(),
                    cell.n_events.to_string(),
                    cell.n_censored.to_string(),
                    format!("{:.6}", cell.retention_rate),
                    format!("{:.6}", cell.cumulative_retention),
                ])?;
            }
            Ok(())
        };

        for cohort_row in &matrix.cohorts {
            write_row(&mut wtr, &cohort_row.cohort.to_string(), &period_boundaries, cohort_row)?;
        }
        write_row(&mut wtr, "overall", &period_boundaries, &matrix.overall)?;
        wtr.flush()?;

        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", csv_path.display());
    }

    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_cohort_matrix",
            serde_json::json!({ "periods": period_boundaries }),
            input,
            &output_json,
            false,
        )?;
    }

    eprintln!(
        "Cohort matrix: {} cohorts × {} periods",
        matrix.cohorts.len(),
        period_boundaries.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// compare (segment comparison report)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CompareInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
}

fn parse_correction_method(s: &str) -> Result<CorrectionMethod> {
    match s.to_lowercase().replace('-', "_").as_str() {
        "bonferroni" | "bonf" => Ok(CorrectionMethod::Bonferroni),
        "benjamini_hochberg" | "bh" | "fdr" => Ok(CorrectionMethod::BenjaminiHochberg),
        "none" => Ok(CorrectionMethod::None),
        _ => anyhow::bail!(
            "unknown correction method '{}' — expected bonferroni, benjamini_hochberg, or none",
            s
        ),
    }
}

pub fn cmd_churn_compare(
    input: &PathBuf,
    correction_str: &str,
    alpha: f64,
    conf_level: f64,
    out_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: CompareInputJson = serde_json::from_str(&raw)?;
    let correction = parse_correction_method(correction_str)?;

    let report = segment_comparison_report(
        &data.times,
        &data.events,
        &data.groups,
        conf_level,
        correction,
        alpha,
    )?;

    let output_json = serde_json::to_value(&report)?;

    crate::write_json(output, &output_json)?;

    // Write CSV + JSON to out_dir if specified.
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;

        // JSON artifact.
        let json_path = dir.join("segment_comparison.json");
        let json_str = serde_json::to_string_pretty(&output_json)?;
        std::fs::write(&json_path, json_str)?;

        // Segments summary CSV.
        let seg_path = dir.join("segment_summary.csv");
        {
            let mut wtr = csv::Writer::from_path(&seg_path)?;
            wtr.write_record(["group", "n", "n_events", "median", "observed", "expected"])?;
            for seg in &report.segments {
                wtr.write_record(&[
                    seg.group.to_string(),
                    seg.n.to_string(),
                    seg.n_events.to_string(),
                    seg.median.map_or("NA".to_string(), |m| format!("{:.4}", m)),
                    format!("{:.4}", seg.observed),
                    format!("{:.4}", seg.expected),
                ])?;
            }
            wtr.flush()?;
        }

        // Pairwise comparisons CSV.
        let pw_path = dir.join("pairwise_comparisons.csv");
        {
            let mut wtr = csv::Writer::from_path(&pw_path)?;
            wtr.write_record([
                "group_a",
                "group_b",
                "chi_squared",
                "p_value",
                "p_adjusted",
                "hazard_ratio_proxy",
                "median_diff",
                "significant",
            ])?;
            for pw in &report.pairwise {
                wtr.write_record(&[
                    pw.group_a.to_string(),
                    pw.group_b.to_string(),
                    format!("{:.6}", pw.chi_squared),
                    format!("{:.6}", pw.p_value),
                    format!("{:.6}", pw.p_adjusted),
                    format!("{:.4}", pw.hazard_ratio_proxy),
                    pw.median_diff.map_or("NA".to_string(), |d| format!("{:.4}", d)),
                    pw.significant.to_string(),
                ])?;
            }
            wtr.flush()?;
        }

        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", seg_path.display());
        eprintln!("Wrote {}", pw_path.display());
    }

    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_compare",
            serde_json::json!({
                "correction": correction_str,
                "alpha": alpha,
                "conf_level": conf_level,
            }),
            input,
            &output_json,
            false,
        )?;
    }

    // Print human-readable summary to stderr.
    eprintln!(
        "Segment comparison: {} segments, {} observations, {} events",
        report.segments.len(),
        report.n,
        report.n_events
    );
    eprintln!(
        "Overall log-rank: χ²={:.4}, df={}, p={:.6}",
        report.overall_chi_squared, report.overall_df, report.overall_p_value
    );
    let n_sig = report.pairwise.iter().filter(|pw| pw.significant).count();
    eprintln!(
        "Pairwise: {} comparisons, {} significant (α={}, correction={:?})",
        report.pairwise.len(),
        n_sig,
        alpha,
        report.correction_method
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// uplift-survival (survival-native causal uplift)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct UpliftSurvivalInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    treated: Vec<u8>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
}

pub fn cmd_churn_uplift_survival(
    input: &PathBuf,
    horizon: f64,
    eval_horizons_str: &str,
    trim: f64,
    out_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: UpliftSurvivalInputJson = serde_json::from_str(&raw)?;

    let eval_horizons: Vec<f64> = eval_horizons_str
        .split(',')
        .map(|s| s.trim().parse::<f64>().with_context(|| format!("invalid eval horizon: '{s}'")))
        .collect::<Result<_>>()?;

    // Covariates can be either row-major (n × p) or column-major (p arrays of length n).
    // Detect format: if covariates.len() == n and each inner has same len, it's row-major.
    // If covariates.len() != n but covariates[0].len() == n, it's column-major → transpose.
    let x: Vec<Vec<f64>> = if data.covariates.is_empty() {
        vec![]
    } else {
        let n = data.times.len();
        let outer = data.covariates.len();
        let inner = data.covariates[0].len();
        if outer == n {
            // Row-major: each element is one observation's covariates.
            data.covariates.clone()
        } else if inner == n {
            // Column-major: each element is one covariate's values → transpose.
            (0..n).map(|i| (0..outer).map(|j| data.covariates[j][i]).collect()).collect()
        } else {
            // Mismatch — skip covariates.
            eprintln!("Warning: covariates shape mismatch, skipping IPW");
            vec![]
        }
    };

    let report = survival_uplift_report(
        &data.times,
        &data.events,
        &data.treated,
        &x,
        horizon,
        &eval_horizons,
        trim,
    )?;

    let output_json = serde_json::to_value(&report)?;

    crate::write_json(output, &output_json)?;

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;

        let json_path = dir.join("survival_uplift.json");
        let json_str = serde_json::to_string_pretty(&output_json)?;
        std::fs::write(&json_path, json_str)?;

        // Arms summary CSV.
        let arms_path = dir.join("uplift_arms.csv");
        {
            let mut wtr = csv::Writer::from_path(&arms_path)?;
            wtr.write_record(["arm", "n", "n_events", "rmst", "median"])?;
            for arm in &report.arms {
                wtr.write_record(&[
                    arm.arm.clone(),
                    arm.n.to_string(),
                    arm.n_events.to_string(),
                    format!("{:.4}", arm.rmst),
                    arm.median.map_or("NA".to_string(), |m| format!("{:.4}", m)),
                ])?;
            }
            wtr.flush()?;
        }

        // ΔS(t) CSV.
        let ds_path = dir.join("uplift_delta_survival.csv");
        {
            let mut wtr = csv::Writer::from_path(&ds_path)?;
            wtr.write_record([
                "horizon",
                "survival_treated",
                "survival_control",
                "delta_survival",
            ])?;
            for d in &report.survival_diffs {
                wtr.write_record(&[
                    format!("{:.2}", d.horizon),
                    format!("{:.6}", d.survival_treated),
                    format!("{:.6}", d.survival_control),
                    format!("{:.6}", d.delta_survival),
                ])?;
            }
            wtr.flush()?;
        }

        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", arms_path.display());
        eprintln!("Wrote {}", ds_path.display());
    }

    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_uplift_survival",
            serde_json::json!({
                "horizon": horizon,
                "eval_horizons": eval_horizons,
                "trim": trim,
            }),
            input,
            &output_json,
            false,
        )?;
    }

    eprintln!(
        "Survival uplift: ΔRMST={:.4} (treated={:.4}, control={:.4}) at τ={}",
        report.delta_rmst, report.rmst_treated, report.rmst_control, horizon
    );
    eprintln!(
        "Overlap: {}/{} used (ESS treated={:.1}, control={:.1}), IPW={}",
        report.overlap.n_after_trim,
        report.overlap.n_total,
        report.overlap.ess_treated,
        report.overlap.ess_control,
        report.ipw_applied
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// diagnostics (guardrails report)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DiagnosticsInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    groups: Vec<i64>,
    #[serde(default)]
    treated: Vec<u8>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
}

pub fn cmd_churn_diagnostics(
    input: &PathBuf,
    trim: f64,
    covariate_names_str: Option<&str>,
    out_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let raw = std::fs::read_to_string(input)?;
    let data: DiagnosticsInputJson = serde_json::from_str(&raw)?;

    let covariate_names: Vec<String> = covariate_names_str
        .map(|s| s.split(',').map(|c| c.trim().to_string()).collect())
        .unwrap_or_default();

    // Detect covariate layout (row-major vs column-major), same as uplift-survival.
    let n = data.times.len();
    let x: Vec<Vec<f64>> = if data.covariates.is_empty() {
        vec![]
    } else {
        let outer = data.covariates.len();
        let inner = data.covariates[0].len();
        if outer == n {
            data.covariates.clone()
        } else if inner == n {
            (0..n).map(|i| (0..outer).map(|j| data.covariates[j][i]).collect()).collect()
        } else {
            eprintln!("Warning: covariates shape mismatch, skipping propensity/balance");
            vec![]
        }
    };

    let report = churn_diagnostics_report(
        &data.times,
        &data.events,
        &data.groups,
        &data.treated,
        &x,
        &covariate_names,
        trim,
    )?;

    let output_json = serde_json::to_value(&report)?;

    crate::write_json(output, &output_json)?;

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;

        let json_path = dir.join("diagnostics.json");
        let json_str = serde_json::to_string_pretty(&output_json)?;
        std::fs::write(&json_path, json_str)?;

        // Censoring CSV.
        let cens_path = dir.join("censoring_by_segment.csv");
        {
            let mut wtr = csv::Writer::from_path(&cens_path)?;
            wtr.write_record(["group", "n", "n_events", "n_censored", "frac_censored"])?;
            for seg in &report.censoring_by_segment {
                wtr.write_record(&[
                    seg.group.to_string(),
                    seg.n.to_string(),
                    seg.n_events.to_string(),
                    seg.n_censored.to_string(),
                    format!("{:.4}", seg.frac_censored),
                ])?;
            }
            wtr.flush()?;
        }

        // Covariate balance CSV.
        if !report.covariate_balance.is_empty() {
            let bal_path = dir.join("covariate_balance.csv");
            let mut wtr = csv::Writer::from_path(&bal_path)?;
            wtr.write_record(["name", "smd_raw", "mean_treated", "mean_control"])?;
            for cb in &report.covariate_balance {
                wtr.write_record(&[
                    cb.name.clone(),
                    format!("{:.4}", cb.smd_raw),
                    format!("{:.4}", cb.mean_treated),
                    format!("{:.4}", cb.mean_control),
                ])?;
            }
            wtr.flush()?;
            eprintln!("Wrote {}", bal_path.display());
        }

        eprintln!("Wrote {}", json_path.display());
        eprintln!("Wrote {}", cens_path.display());
    }

    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "churn_diagnostics",
            serde_json::json!({ "trim": trim }),
            input,
            &output_json,
            false,
        )?;
    }

    // Human-readable summary.
    eprintln!(
        "Diagnostics: {} observations, {} events, {:.0}% censored",
        report.n,
        report.n_events,
        report.overall_censoring_frac * 100.0
    );
    eprintln!("Trust gate: {}", if report.trust_gate_passed { "PASSED" } else { "FAILED" });
    if !report.warnings.is_empty() {
        eprintln!("Warnings ({}):", report.warnings.len());
        for w in &report.warnings {
            eprintln!("  [{}] {}: {}", w.severity, w.category, w.message);
        }
    }

    Ok(())
}
