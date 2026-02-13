//! `nextstat convert` — ROOT TTree → Parquet conversion.

use anyhow::{Context, Result};
use std::path::Path;

use ns_root::RootFile;
use ns_unbinned::event_parquet;
use ns_unbinned::event_store::{EventStore, ObservableSpec};

/// Parsed observable spec from CLI: `name:low:high` or `name:expr:low:high`.
pub struct ObsArg {
    pub name: String,
    pub expr: String,
    pub bounds: (f64, f64),
}

impl ObsArg {
    /// Parse `"mass:100:180"` or `"mass:m_inv:100:180"` (name:expr:lo:hi).
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts.len() {
            3 => {
                let name = parts[0].to_string();
                let lo: f64 = parts[1].parse().context(format!("bad low bound in '{s}'"))?;
                let hi: f64 = parts[2].parse().context(format!("bad high bound in '{s}'"))?;
                Ok(Self { expr: name.clone(), name, bounds: (lo, hi) })
            }
            4 => {
                let name = parts[0].to_string();
                let expr = parts[1].to_string();
                let lo: f64 = parts[2].parse().context(format!("bad low bound in '{s}'"))?;
                let hi: f64 = parts[3].parse().context(format!("bad high bound in '{s}'"))?;
                Ok(Self { name, expr, bounds: (lo, hi) })
            }
            _ => anyhow::bail!(
                "invalid observable spec '{s}': expected 'name:low:high' or 'name:expr:low:high'"
            ),
        }
    }

    pub fn to_observable_spec(&self) -> ObservableSpec {
        ObservableSpec::expression(&self.name, &self.expr, self.bounds)
    }
}

pub fn cmd_convert(
    input: &Path,
    tree: &str,
    output: &Path,
    observables: &[ObsArg],
    selection: Option<&str>,
    weight: Option<&str>,
    max_events: Option<usize>,
) -> Result<()> {
    if observables.is_empty() {
        anyhow::bail!("at least one --observable is required (format: name:low:high)");
    }

    let obs_specs: Vec<ObservableSpec> =
        observables.iter().map(|o| o.to_observable_spec()).collect();

    tracing::info!("opening ROOT file: {}", input.display());
    let root = RootFile::open(input)
        .with_context(|| format!("failed to open ROOT file {}", input.display()))?;

    tracing::info!("reading TTree '{tree}' with {} observables", obs_specs.len());
    let store = EventStore::from_tree(&root, tree, &obs_specs, selection, weight)
        .with_context(|| format!("failed to read TTree '{tree}'"))?;

    let n = store.n_events();
    tracing::info!("read {n} events");

    let store = if let Some(max) = max_events {
        if max < n {
            tracing::info!("truncating to {max} events (--max-events)");
            truncate_store(&store, &obs_specs, max)?
        } else {
            store
        }
    } else {
        store
    };

    tracing::info!("writing Parquet to {}", output.display());
    event_parquet::write_event_parquet(&store, output)
        .with_context(|| format!("failed to write Parquet {}", output.display()))?;

    let final_n = store.n_events();
    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    eprintln!(
        "Converted {final_n} events ({} observables{}) → {} ({:.1} KB)",
        obs_specs.len(),
        if store.weights().is_some() { " + weights" } else { "" },
        output.display(),
        file_size as f64 / 1024.0,
    );

    Ok(())
}

/// Truncate an EventStore to the first `max` events.
fn truncate_store(
    store: &EventStore,
    obs_specs: &[ObservableSpec],
    max: usize,
) -> Result<EventStore> {
    let columns: Vec<(String, Vec<f64>)> = store
        .column_names()
        .iter()
        .map(|name| {
            let col = store.column(name).unwrap();
            (name.clone(), col[..max].to_vec())
        })
        .collect();
    let weights = store.weights().map(|w| w[..max].to_vec());
    EventStore::from_columns(obs_specs.to_vec(), columns, weights)
        .map_err(|e| anyhow::anyhow!("truncate_store: {e}"))
}
