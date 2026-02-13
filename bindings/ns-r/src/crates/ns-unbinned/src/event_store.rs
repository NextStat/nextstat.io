//! Columnar event storage for unbinned likelihood evaluation.

use ns_core::{Error, Result};
use ns_root::{CompiledExpr, JaggedCol, RootFile};
use std::collections::{BTreeMap, HashMap};

/// Observable specification for ingesting event-level data.
#[derive(Debug, Clone)]
pub struct ObservableSpec {
    /// Column name in the resulting [`EventStore`].
    pub name: String,
    /// Expression over ROOT branches used to compute this observable.
    ///
    /// For simple cases this is just the branch name (e.g. `"mass"`).
    pub expr: String,
    /// Support bounds `(low, high)` for this observable in the selected region `Ω`.
    ///
    /// Bounds are used for PDF normalization. Use finite bounds whenever possible.
    pub bounds: (f64, f64),
}

impl ObservableSpec {
    /// Convenience constructor for an observable that is read directly from a branch.
    pub fn branch(name: impl Into<String>, bounds: (f64, f64)) -> Self {
        let name = name.into();
        Self { expr: name.clone(), name, bounds }
    }

    /// Constructor for an observable computed from an expression over branches.
    pub fn expression(
        name: impl Into<String>,
        expr: impl Into<String>,
        bounds: (f64, f64),
    ) -> Self {
        Self { name: name.into(), expr: expr.into(), bounds }
    }
}

/// Diagnostics for per-event weights in an [`EventStore`].
#[derive(Debug, Clone)]
pub struct WeightSummary {
    /// Number of events (raw count).
    pub n_events: usize,
    /// Sum of weights `Σ w_i`.
    pub sum_weights: f64,
    /// Effective sample size `(Σw)² / Σw²`.
    pub effective_sample_size: f64,
    /// Minimum weight.
    pub min_weight: f64,
    /// Maximum weight.
    pub max_weight: f64,
    /// Mean weight `Σw / N`.
    pub mean_weight: f64,
    /// Number of events with weight exactly 0.
    pub n_zero: usize,
}

/// Columnar event storage (Structure-of-Arrays / SoA).
#[derive(Debug, Clone)]
pub struct EventStore {
    n_events: usize,
    column_names: Vec<String>,
    columns: Vec<Vec<f64>>,
    name_to_index: HashMap<String, usize>,
    bounds: HashMap<String, (f64, f64)>,
    weights: Option<Vec<f64>>,
}

impl EventStore {
    /// Create an [`EventStore`] from already materialized columns.
    ///
    /// `observables` defines which columns are required and provides bounds used for PDF
    /// normalization. Extra columns are accepted but ignored.
    pub fn from_columns(
        observables: Vec<ObservableSpec>,
        columns: impl IntoIterator<Item = (String, Vec<f64>)>,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        if observables.is_empty() {
            return Err(Error::Validation("EventStore requires at least one observable".into()));
        }

        let mut by_name: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (name, col) in columns {
            by_name.insert(name, col);
        }

        let mut column_names = Vec::with_capacity(observables.len());
        let mut cols = Vec::with_capacity(observables.len());
        let mut bounds = HashMap::with_capacity(observables.len());

        let mut n_events: Option<usize> = None;
        for obs in &observables {
            let (lo, hi) = obs.bounds;
            if lo.is_nan() || hi.is_nan() || lo >= hi {
                return Err(Error::Validation(format!(
                    "invalid bounds for observable '{}': expected low < high, got ({lo}, {hi})",
                    obs.name
                )));
            }
            let col = by_name.remove(&obs.name).ok_or_else(|| {
                Error::Validation(format!("missing observable column '{}'", obs.name))
            })?;
            let n = col.len();
            if let Some(ne) = n_events {
                if n != ne {
                    return Err(Error::Validation(format!(
                        "column length mismatch for '{}': expected {}, got {}",
                        obs.name, ne, n
                    )));
                }
            } else {
                n_events = Some(n);
            }
            column_names.push(obs.name.clone());
            // Validate values.
            let check_bounds = lo.is_finite() && hi.is_finite();
            if col.iter().any(|x| !x.is_finite()) {
                return Err(Error::Validation(format!(
                    "observable '{}' contains non-finite values",
                    obs.name
                )));
            }
            if check_bounds && col.iter().any(|&x| x < lo || x > hi) {
                return Err(Error::Validation(format!(
                    "observable '{}' contains values outside bounds ({lo}, {hi})",
                    obs.name
                )));
            }
            cols.push(col);
            bounds.insert(obs.name.clone(), obs.bounds);
        }

        let n_events = n_events.unwrap_or(0);

        if let Some(w) = &weights {
            if w.len() != n_events {
                return Err(Error::Validation(format!(
                    "weights length mismatch: expected {}, got {}",
                    n_events,
                    w.len()
                )));
            }
            if w.iter().any(|x| !x.is_finite()) {
                return Err(Error::Validation("weights must be finite".into()));
            }
            if w.iter().any(|x| *x < 0.0) {
                return Err(Error::Validation(
                    "negative event weights are not supported: the unbinned extended NLL \
                     requires non-negative frequency weights (w_i >= 0)"
                        .into(),
                ));
            }
        }

        let name_to_index =
            column_names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect::<HashMap<_, _>>();

        Ok(Self { n_events, column_names, columns: cols, name_to_index, bounds, weights })
    }

    /// Read event columns from a ROOT TTree, optionally applying a selection and a weight expression.
    ///
    /// Selection and weight expressions use the `ns-root` expression language.
    pub fn from_tree(
        root_file: &RootFile,
        tree_name: &str,
        observables: &[ObservableSpec],
        selection: Option<&str>,
        weight_expr: Option<&str>,
    ) -> Result<Self> {
        let (store, _) = Self::from_tree_with_extra_weights(
            root_file,
            tree_name,
            observables,
            selection,
            weight_expr,
            &[],
        )?;
        Ok(store)
    }

    /// Read event columns from a ROOT TTree, with one "primary" weight expression plus additional
    /// extra weight expressions evaluated on the same selected events.
    ///
    /// This is useful for template morphing / weight systematics, where we need several per-event
    /// weight columns (nominal + variations) but want to read branches and evaluate the selection
    /// only once.
    pub fn from_tree_with_extra_weights(
        root_file: &RootFile,
        tree_name: &str,
        observables: &[ObservableSpec],
        selection: Option<&str>,
        weight_expr: Option<&str>,
        extra_weight_exprs: &[&str],
    ) -> Result<(Self, Vec<Vec<f64>>)> {
        if observables.is_empty() {
            return Err(Error::Validation(
                "EventStore.from_tree_with_extra_weights requires observables".into(),
            ));
        }

        let tree = root_file
            .get_tree(tree_name)
            .map_err(|e| Error::RootFile(format!("failed to load TTree '{}': {e}", tree_name)))?;

        // Compile all expressions up-front.
        let compiled_obs: Vec<(ObservableSpec, CompiledExpr)> = observables
            .iter()
            .cloned()
            .map(|o| {
                let c = CompiledExpr::compile(&o.expr).map_err(|e| {
                    Error::Validation(format!("failed to compile observable '{}': {e}", o.name))
                })?;
                Ok((o, c))
            })
            .collect::<Result<_>>()?;

        let compiled_sel = match selection {
            Some(s) => Some(CompiledExpr::compile(s).map_err(|e| {
                Error::Validation(format!("failed to compile selection '{s}': {e}"))
            })?),
            None => None,
        };

        let compiled_wt = match weight_expr {
            Some(s) => Some(CompiledExpr::compile(s).map_err(|e| {
                Error::Validation(format!("failed to compile weight expression '{s}': {e}"))
            })?),
            None => None,
        };

        let compiled_extra_wts: Vec<CompiledExpr> = extra_weight_exprs
            .iter()
            .map(|s| {
                CompiledExpr::compile(s).map_err(|e| {
                    Error::Validation(format!(
                        "failed to compile extra weight expression '{s}': {e}"
                    ))
                })
            })
            .collect::<Result<_>>()?;

        // Collect required branches (deduplicated) for vectorized evaluation.
        let mut required_branches: BTreeMap<String, ()> = BTreeMap::new();
        let mut required_jagged: BTreeMap<String, ()> = BTreeMap::new();

        for (_, c) in &compiled_obs {
            for b in &c.required_branches {
                required_branches.insert(b.clone(), ());
            }
            for b in &c.required_jagged_branches {
                required_jagged.insert(b.clone(), ());
            }
        }
        if let Some(c) = &compiled_sel {
            for b in &c.required_branches {
                required_branches.insert(b.clone(), ());
            }
            for b in &c.required_jagged_branches {
                required_jagged.insert(b.clone(), ());
            }
        }
        if let Some(c) = &compiled_wt {
            for b in &c.required_branches {
                required_branches.insert(b.clone(), ());
            }
            for b in &c.required_jagged_branches {
                required_jagged.insert(b.clone(), ());
            }
        }
        for c in &compiled_extra_wts {
            for b in &c.required_branches {
                required_branches.insert(b.clone(), ());
            }
            for b in &c.required_jagged_branches {
                required_jagged.insert(b.clone(), ());
            }
        }

        let branch_names: Vec<String> = required_branches.keys().cloned().collect();
        let jagged_names: Vec<String> = required_jagged.keys().cloned().collect();

        // Materialize required branches (deduped) into name→column maps.
        let mut branch_cols: HashMap<String, Vec<f64>> = HashMap::with_capacity(branch_names.len());
        for b in &branch_names {
            let col = root_file.branch_data(&tree, b).map_err(|e| {
                Error::RootFile(format!("failed to read branch '{b}' from '{tree_name}': {e}"))
            })?;
            branch_cols.insert(b.clone(), col);
        }

        let n_events =
            branch_names.first().and_then(|b| branch_cols.get(b)).map(|c| c.len()).unwrap_or(0);

        for b in &branch_names {
            let Some(c) = branch_cols.get(b) else { continue };
            if c.len() != n_events {
                return Err(Error::Validation(format!(
                    "branch length mismatch for '{b}': expected {}, got {}",
                    n_events,
                    c.len()
                )));
            }
        }

        let mut jagged_cols: HashMap<String, JaggedCol> =
            HashMap::with_capacity(jagged_names.len());
        for b in &jagged_names {
            let col = root_file.branch_data_jagged(&tree, b).map_err(|e| {
                Error::RootFile(format!(
                    "failed to read jagged branch '{b}' from '{tree_name}': {e}"
                ))
            })?;
            jagged_cols.insert(b.clone(), col);
        }

        // Helper: expand constant expression results to length n_events.
        fn expand_to_n(v: Vec<f64>, n: usize) -> Vec<f64> {
            if n == 0 {
                return Vec::new();
            }
            if v.len() == n {
                return v;
            }
            if v.len() == 1 {
                return vec![v[0]; n];
            }
            // Unexpected length; return as-is and let caller validate.
            v
        }

        // Evaluate one expression using the branch/jagged maps in the order required by the bytecode.
        let eval_expr = |c: &CompiledExpr| -> Result<Vec<f64>> {
            let cols = c
                .required_branches
                .iter()
                .map(|b| {
                    branch_cols.get(b).map(|v| v.as_slice()).ok_or_else(|| {
                        Error::Validation(format!("expression requires missing branch '{b}'"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            if c.required_jagged_branches.is_empty() {
                return Ok(expand_to_n(c.eval_bulk(&cols), n_events));
            }

            let jagged = c
                .required_jagged_branches
                .iter()
                .map(|b| {
                    jagged_cols.get(b).ok_or_else(|| {
                        Error::Validation(format!(
                            "expression requires missing jagged branch '{b}'"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(expand_to_n(c.eval_bulk_with_jagged(&cols, &jagged), n_events))
        };

        // Build selection mask.
        let sel_values = match &compiled_sel {
            Some(c) => eval_expr(c)?,
            None => vec![1.0; n_events],
        };

        if sel_values.len() != n_events {
            return Err(Error::Validation(format!(
                "selection evaluation returned length {}, expected {}",
                sel_values.len(),
                n_events
            )));
        }

        let mut selected_idx = Vec::with_capacity(n_events);
        for (i, &v) in sel_values.iter().enumerate() {
            if v != 0.0 && v.is_finite() {
                selected_idx.push(i);
            }
        }
        if selected_idx.is_empty() {
            return Err(Error::Validation("selection resulted in 0 events".into()));
        }

        // Evaluate observables and filter to selected rows.
        let mut out_columns: Vec<(String, Vec<f64>)> = Vec::with_capacity(compiled_obs.len());
        for (obs, c) in &compiled_obs {
            let values = eval_expr(c)?;
            if values.len() != n_events {
                return Err(Error::Validation(format!(
                    "observable '{}' evaluation returned length {}, expected {}",
                    obs.name,
                    values.len(),
                    n_events
                )));
            }
            let mut filtered = Vec::with_capacity(selected_idx.len());
            for &i in &selected_idx {
                filtered.push(values[i]);
            }
            out_columns.push((obs.name.clone(), filtered));
        }

        // Primary weights (optional) evaluated on the full tree, then filtered.
        let weights = if let Some(c) = &compiled_wt {
            let values = eval_expr(c)?;
            if values.len() != n_events {
                return Err(Error::Validation(format!(
                    "weight evaluation returned length {}, expected {}",
                    values.len(),
                    n_events
                )));
            }
            let mut filtered = Vec::with_capacity(selected_idx.len());
            for &i in &selected_idx {
                filtered.push(values[i]);
            }
            Some(filtered)
        } else {
            None
        };

        // Extra weights evaluated on the full tree, then filtered.
        let mut extra_weights: Vec<Vec<f64>> = Vec::with_capacity(compiled_extra_wts.len());
        for (expr_str, c) in extra_weight_exprs.iter().zip(&compiled_extra_wts) {
            let values = eval_expr(c)?;
            if values.len() != n_events {
                return Err(Error::Validation(format!(
                    "extra weight evaluation returned length {}, expected {} for expr '{expr_str}'",
                    values.len(),
                    n_events
                )));
            }
            let mut filtered = Vec::with_capacity(selected_idx.len());
            for &i in &selected_idx {
                filtered.push(values[i]);
            }
            if filtered.iter().any(|x| !x.is_finite()) {
                return Err(Error::Validation(format!(
                    "extra weights for expr '{expr_str}' must be finite"
                )));
            }
            if filtered.iter().any(|x| *x < 0.0) {
                return Err(Error::Validation(format!(
                    "extra weights for expr '{expr_str}' must be >= 0"
                )));
            }
            extra_weights.push(filtered);
        }

        let store = Self::from_columns(observables.to_vec(), out_columns, weights)?;
        Ok((store, extra_weights))
    }

    /// Number of events.
    pub fn n_events(&self) -> usize {
        self.n_events
    }

    /// Names of stored columns (stable order).
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        let idx = self.name_to_index.get(name).copied()?;
        self.columns.get(idx).map(|c| c.as_slice())
    }

    /// Bounds for an observable, if defined.
    pub fn bounds(&self, name: &str) -> Option<(f64, f64)> {
        self.bounds.get(name).copied()
    }

    /// Optional per-event weights.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Sum of event weights. Returns `n_events` as f64 if no weights are set.
    pub fn sum_weights(&self) -> f64 {
        match &self.weights {
            Some(w) => w.iter().sum(),
            None => self.n_events as f64,
        }
    }

    /// Effective sample size: `(Σw)² / Σw²`.
    ///
    /// For unweighted data this equals `n_events`. For weighted data it quantifies
    /// the equivalent unweighted sample size, accounting for weight variance.
    /// Returns 0.0 if sum of squared weights is zero.
    pub fn effective_sample_size(&self) -> f64 {
        match &self.weights {
            None => self.n_events as f64,
            Some(w) => {
                let sum_w: f64 = w.iter().sum();
                let sum_w2: f64 = w.iter().map(|&wi| wi * wi).sum();
                if sum_w2 > 0.0 { (sum_w * sum_w) / sum_w2 } else { 0.0 }
            }
        }
    }

    /// Weight diagnostics summary.
    ///
    /// Returns `None` if the store has no per-event weights (all unit-weight).
    pub fn weight_summary(&self) -> Option<WeightSummary> {
        let w = self.weights.as_ref()?;
        let n = w.len();
        if n == 0 {
            return Some(WeightSummary {
                n_events: 0,
                sum_weights: 0.0,
                effective_sample_size: 0.0,
                min_weight: 0.0,
                max_weight: 0.0,
                mean_weight: 0.0,
                n_zero: 0,
            });
        }
        let sum_w: f64 = w.iter().sum();
        let sum_w2: f64 = w.iter().map(|&wi| wi * wi).sum();
        let ess = if sum_w2 > 0.0 { (sum_w * sum_w) / sum_w2 } else { 0.0 };
        let min_w = w.iter().copied().fold(f64::INFINITY, f64::min);
        let max_w = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n_zero = w.iter().filter(|&&wi| wi == 0.0).count();
        Some(WeightSummary {
            n_events: n,
            sum_weights: sum_w,
            effective_sample_size: ess,
            min_weight: min_w,
            max_weight: max_w,
            mean_weight: sum_w / n as f64,
            n_zero,
        })
    }

    /// Build an [`EventStore`] from an Arrow [`RecordBatch`].
    ///
    /// Observable bounds are read from the schema's `nextstat.observables` key-value
    /// metadata if available.  Pass explicit `observables` to override.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn from_arrow(
        batch: &arrow::record_batch::RecordBatch,
        observables: Option<&[ObservableSpec]>,
    ) -> Result<Self> {
        crate::event_parquet::event_store_from_record_batch(batch, observables)
    }

    /// Read an [`EventStore`] from a Parquet file on disk.
    ///
    /// Observable bounds are read from Parquet key-value metadata if available.
    /// Pass explicit `observables` to override.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn from_parquet(
        path: &std::path::Path,
        observables: Option<&[ObservableSpec]>,
    ) -> Result<Self> {
        crate::event_parquet::read_event_parquet(path, observables)
    }

    /// Read an [`EventStore`] from a multi-channel Parquet file, selecting a given `_channel`.
    ///
    /// This requires the Parquet file to contain a `_channel` column and filters rows to
    /// `_channel == channel`.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn from_parquet_channel(
        path: &std::path::Path,
        observables: Option<&[ObservableSpec]>,
        channel: &str,
    ) -> Result<Self> {
        crate::event_parquet::read_event_parquet_channel(path, observables, channel)
    }

    /// Read an [`EventStore`] from in-memory Parquet bytes.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn from_parquet_bytes(data: &[u8], observables: Option<&[ObservableSpec]>) -> Result<Self> {
        crate::event_parquet::read_event_parquet_bytes(data, observables)
    }

    /// Read an [`EventStore`] from in-memory Parquet bytes, selecting a given `_channel`.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn from_parquet_bytes_channel(
        data: &[u8],
        observables: Option<&[ObservableSpec]>,
        channel: &str,
    ) -> Result<Self> {
        crate::event_parquet::read_event_parquet_bytes_channel(data, observables, channel)
    }

    /// Convert this [`EventStore`] to an Arrow [`RecordBatch`].
    ///
    /// Observable bounds are embedded in the schema's key-value metadata.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn to_arrow(&self) -> Result<arrow::record_batch::RecordBatch> {
        crate::event_parquet::event_store_to_record_batch(self)
    }

    /// Write this [`EventStore`] to a Parquet file.
    ///
    /// Uses Zstd compression if the `arrow-io-zstd` feature is enabled, Snappy otherwise.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn to_parquet(&self, path: &std::path::Path) -> Result<()> {
        crate::event_parquet::write_event_parquet(self, path)
    }

    /// Write this [`EventStore`] to in-memory Parquet bytes.
    ///
    /// Requires the `arrow-io` feature.
    #[cfg(feature = "arrow-io")]
    pub fn to_parquet_bytes(&self) -> Result<Vec<u8>> {
        crate::event_parquet::write_event_parquet_bytes(self)
    }
}
