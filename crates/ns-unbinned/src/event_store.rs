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
                    "negative event weights are not supported (see plan risk note)".into(),
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
        if observables.is_empty() {
            return Err(Error::Validation("EventStore.from_tree requires observables".into()));
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

        // Weights (optional) evaluated on the full tree, then filtered.
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

        Self::from_columns(observables.to_vec(), out_columns, weights)
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
}
