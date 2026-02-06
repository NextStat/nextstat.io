//! Single-pass histogram filling from column data with selections and weights.

use std::collections::HashMap;

use crate::error::{Result, RootError};
use crate::expr::CompiledExpr;
use crate::histogram::Histogram;

/// Specification for filling one histogram.
pub struct HistogramSpec {
    /// Histogram name.
    pub name: String,
    /// Expression for the variable to histogram.
    pub variable: CompiledExpr,
    /// Optional weight expression.
    pub weight: Option<CompiledExpr>,
    /// Optional selection expression (entries passing if > 0).
    pub selection: Option<CompiledExpr>,
    /// Bin edges (must be sorted, length = n_bins + 1).
    pub bin_edges: Vec<f64>,
}

/// Result of filling a histogram.
#[derive(Debug, Clone)]
pub struct FilledHistogram {
    /// Histogram name.
    pub name: String,
    /// Bin edges.
    pub bin_edges: Vec<f64>,
    /// Bin contents (sum of weights per bin).
    pub bin_content: Vec<f64>,
    /// Sum of weights squared per bin.
    pub sumw2: Vec<f64>,
    /// Total entries passing selection.
    pub entries: u64,
}

impl From<FilledHistogram> for Histogram {
    fn from(fh: FilledHistogram) -> Self {
        let n_bins = fh.bin_content.len();
        let x_min = fh.bin_edges[0];
        let x_max = fh.bin_edges[n_bins];
        Histogram {
            name: fh.name,
            title: String::new(),
            n_bins,
            x_min,
            x_max,
            bin_edges: fh.bin_edges,
            bin_content: fh.bin_content,
            sumw2: Some(fh.sumw2),
            entries: fh.entries as f64,
        }
    }
}

/// Fill multiple histograms in a single pass over the data.
///
/// `columns` maps branch names to their data arrays. All arrays must have
/// the same length.
pub fn fill_histograms(
    specs: &[HistogramSpec],
    columns: &HashMap<String, Vec<f64>>,
) -> Result<Vec<FilledHistogram>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }

    // Determine number of entries from any column
    let n_entries = columns.values().next().map(|v| v.len()).unwrap_or(0);

    // Pre-evaluate all expressions column-wise for efficiency
    let mut var_vals: Vec<Vec<f64>> = Vec::with_capacity(specs.len());
    let mut weight_vals: Vec<Option<Vec<f64>>> = Vec::with_capacity(specs.len());
    let mut sel_vals: Vec<Option<Vec<f64>>> = Vec::with_capacity(specs.len());

    for spec in specs {
        var_vals.push(eval_expr_columns(&spec.variable, columns, n_entries)?);
        weight_vals.push(match &spec.weight {
            Some(w) => Some(eval_expr_columns(w, columns, n_entries)?),
            None => None,
        });
        sel_vals.push(match &spec.selection {
            Some(s) => Some(eval_expr_columns(s, columns, n_entries)?),
            None => None,
        });
    }

    // Fill histograms
    let mut results: Vec<FilledHistogram> = specs
        .iter()
        .map(|spec| {
            let n_bins = spec.bin_edges.len() - 1;
            FilledHistogram {
                name: spec.name.clone(),
                bin_edges: spec.bin_edges.clone(),
                bin_content: vec![0.0; n_bins],
                sumw2: vec![0.0; n_bins],
                entries: 0,
            }
        })
        .collect();

    for entry in 0..n_entries {
        for (i, _spec) in specs.iter().enumerate() {
            // Check selection
            if let Some(ref sel) = sel_vals[i] {
                if sel[entry] <= 0.0 {
                    continue;
                }
            }

            let val = var_vals[i][entry];
            let weight = match &weight_vals[i] {
                Some(w) => w[entry],
                None => 1.0,
            };

            // Find bin (binary search)
            let bin = find_bin(&results[i].bin_edges, val);
            if let Some(b) = bin {
                results[i].bin_content[b] += weight;
                results[i].sumw2[b] += weight * weight;
                results[i].entries += 1;
            }
        }
    }

    Ok(results)
}

/// Find the bin index for a value given sorted bin edges.
///
/// Returns `None` for underflow/overflow.
fn find_bin(edges: &[f64], val: f64) -> Option<usize> {
    if val < edges[0] || val >= edges[edges.len() - 1] {
        return None;
    }
    match edges.binary_search_by(|e| e.partial_cmp(&val).unwrap()) {
        Ok(i) => {
            if i >= edges.len() - 1 {
                None
            } else {
                Some(i)
            }
        }
        Err(i) => {
            if i == 0 || i >= edges.len() {
                None
            } else {
                Some(i - 1)
            }
        }
    }
}

/// Evaluate a compiled expression using column data.
fn eval_expr_columns(
    expr: &CompiledExpr,
    columns: &HashMap<String, Vec<f64>>,
    n_entries: usize,
) -> Result<Vec<f64>> {
    let cols: Vec<&[f64]> = expr
        .required_branches
        .iter()
        .map(|name| {
            columns
                .get(name.as_str())
                .map(|v| v.as_slice())
                .ok_or_else(|| RootError::Expression(format!("missing column: '{}'", name)))
        })
        .collect::<Result<_>>()?;

    if cols.is_empty() {
        // Constant expression
        let val = expr.eval_row(&[]);
        return Ok(vec![val; n_entries]);
    }

    Ok(expr.eval_bulk(&cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_simple() {
        let spec = HistogramSpec {
            name: "h".into(),
            variable: CompiledExpr::compile("x").unwrap(),
            weight: None,
            selection: None,
            bin_edges: vec![0.0, 1.0, 2.0, 3.0],
        };

        let mut cols = HashMap::new();
        cols.insert("x".into(), vec![0.5, 1.5, 2.5, 0.5, -1.0, 3.5]);

        let result = fill_histograms(&[spec], &cols).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bin_content, vec![2.0, 1.0, 1.0]);
        assert_eq!(result[0].entries, 4);
    }

    #[test]
    fn fill_with_weight() {
        let spec = HistogramSpec {
            name: "h".into(),
            variable: CompiledExpr::compile("x").unwrap(),
            weight: Some(CompiledExpr::compile("w").unwrap()),
            selection: None,
            bin_edges: vec![0.0, 1.0, 2.0],
        };

        let mut cols = HashMap::new();
        cols.insert("x".into(), vec![0.5, 1.5, 0.5]);
        cols.insert("w".into(), vec![2.0, 3.0, 1.0]);

        let result = fill_histograms(&[spec], &cols).unwrap();
        assert_eq!(result[0].bin_content, vec![3.0, 3.0]);
        assert_eq!(result[0].sumw2, vec![5.0, 9.0]);
    }

    #[test]
    fn fill_with_selection() {
        let spec = HistogramSpec {
            name: "h".into(),
            variable: CompiledExpr::compile("x").unwrap(),
            weight: None,
            selection: Some(CompiledExpr::compile("x > 1.0").unwrap()),
            bin_edges: vec![0.0, 1.0, 2.0, 3.0],
        };

        let mut cols = HashMap::new();
        cols.insert("x".into(), vec![0.5, 1.5, 2.5, 0.3]);

        let result = fill_histograms(&[spec], &cols).unwrap();
        assert_eq!(result[0].bin_content, vec![0.0, 1.0, 1.0]);
        assert_eq!(result[0].entries, 2);
    }

    #[test]
    fn find_bin_edge_cases() {
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(find_bin(&edges, -0.5), None);
        assert_eq!(find_bin(&edges, 3.0), None);
        assert_eq!(find_bin(&edges, 0.0), Some(0));
        assert_eq!(find_bin(&edges, 1.0), Some(1));
        assert_eq!(find_bin(&edges, 2.99), Some(2));
    }

    #[test]
    fn filled_histogram_to_histogram() {
        let fh = FilledHistogram {
            name: "test".into(),
            bin_edges: vec![0.0, 1.0, 2.0],
            bin_content: vec![5.0, 3.0],
            sumw2: vec![25.0, 9.0],
            entries: 8,
        };
        let h: Histogram = fh.into();
        assert_eq!(h.n_bins, 2);
        assert_eq!(h.x_min, 0.0);
        assert_eq!(h.x_max, 2.0);
    }
}
