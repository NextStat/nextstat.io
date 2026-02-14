//! 2D likelihood contour artifact — two-POI profile likelihood contours.
//!
//! TRExFitter `GetContour` produces 2D Δ(2NLL) contour plots for fits with
//! two parameters of interest, showing e.g. 68% and 95% CL regions.
//! This module provides the plot-friendly JSON artifact.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// A single point in a 2D NLL grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourGridPoint {
    /// First POI value.
    pub x: f64,
    /// Second POI value.
    pub y: f64,
    /// 2 × (NLL(x,y) − NLL_min).
    pub twice_delta_nll: f64,
    /// Whether the conditional fit converged at this grid point.
    pub converged: bool,
}

/// A single extracted contour line at a given confidence level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourLine {
    /// Confidence level label (e.g. "68%" or "95%").
    pub cl_label: String,
    /// Δ(2NLL) threshold for this CL (1.0 for 68%, 3.84 for 95% in 1D;
    /// 2.30 for 68%, 5.99 for 95% in 2D).
    pub threshold: f64,
    /// X-coordinates of contour vertices (closed polygon).
    pub x: Vec<f64>,
    /// Y-coordinates of contour vertices (closed polygon).
    pub y: Vec<f64>,
}

/// Plot-friendly artifact for 2D likelihood contours.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourArtifact {
    /// Label for the first POI axis.
    pub x_label: String,
    /// Label for the second POI axis.
    pub y_label: String,
    /// Best-fit value for the first POI.
    pub x_hat: f64,
    /// Best-fit value for the second POI.
    pub y_hat: f64,
    /// Global minimum NLL.
    pub nll_min: f64,
    /// Grid x-values (unique, sorted).
    pub x_values: Vec<f64>,
    /// Grid y-values (unique, sorted).
    pub y_values: Vec<f64>,
    /// 2D grid of `2*(NLL - NLL_min)` values, row-major `[y_idx][x_idx]`.
    pub twice_delta_nll_grid: Vec<Vec<f64>>,
    /// Extracted contour lines at standard CL thresholds.
    pub contours: Vec<ContourLine>,
    /// Raw grid points (optional, for debugging).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grid_points: Vec<ContourGridPoint>,
}

/// Standard 2D χ² thresholds for Δ(2NLL).
/// For 2 degrees of freedom: 68.27% → 2.30, 95.45% → 6.18, 99.73% → 11.83.
pub const CL_2D_68: f64 = 2.30;
pub const CL_2D_95: f64 = 5.99;
pub const CL_2D_99: f64 = 11.83;

impl ContourArtifact {
    /// Build a contour artifact from a grid of pre-computed NLL values.
    ///
    /// # Arguments
    /// * `x_label`, `y_label` — axis labels
    /// * `x_hat`, `y_hat` — best-fit POI values
    /// * `nll_min` — global NLL minimum
    /// * `x_values` — sorted unique grid x-coordinates
    /// * `y_values` — sorted unique grid y-coordinates
    /// * `grid_points` — all evaluated grid points (must cover `x_values × y_values`)
    pub fn from_grid(
        x_label: &str,
        y_label: &str,
        x_hat: f64,
        y_hat: f64,
        nll_min: f64,
        x_values: Vec<f64>,
        y_values: Vec<f64>,
        grid_points: Vec<ContourGridPoint>,
    ) -> Result<Self> {
        let nx = x_values.len();
        let ny = y_values.len();
        if nx < 2 || ny < 2 {
            return Err(Error::Validation(
                "contour grid must have at least 2 points in each dimension".into(),
            ));
        }

        // Build the 2D grid (row-major: [y_idx][x_idx]).
        let mut grid = vec![vec![f64::NAN; nx]; ny];
        for p in &grid_points {
            let xi = match x_values.iter().position(|&v| (v - p.x).abs() < 1e-12) {
                Some(i) => i,
                None => continue,
            };
            let yi = match y_values.iter().position(|&v| (v - p.y).abs() < 1e-12) {
                Some(i) => i,
                None => continue,
            };
            grid[yi][xi] = p.twice_delta_nll;
        }

        // Extract contours via marching squares.
        let contours = vec![
            extract_contour_marching_squares(&x_values, &y_values, &grid, CL_2D_68, "68%"),
            extract_contour_marching_squares(&x_values, &y_values, &grid, CL_2D_95, "95%"),
        ];

        Ok(Self {
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            x_hat,
            y_hat,
            nll_min,
            x_values,
            y_values,
            twice_delta_nll_grid: grid,
            contours,
            grid_points,
        })
    }

    /// Build from a raw 2D NLL grid without individual grid points.
    pub fn from_raw_grid(
        x_label: &str,
        y_label: &str,
        x_hat: f64,
        y_hat: f64,
        nll_min: f64,
        x_values: Vec<f64>,
        y_values: Vec<f64>,
        twice_delta_nll_grid: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let nx = x_values.len();
        let ny = y_values.len();
        if nx < 2 || ny < 2 {
            return Err(Error::Validation(
                "contour grid must have at least 2 points in each dimension".into(),
            ));
        }
        if twice_delta_nll_grid.len() != ny {
            return Err(Error::Validation(format!(
                "grid row count {} does not match y_values length {}",
                twice_delta_nll_grid.len(),
                ny,
            )));
        }
        for (i, row) in twice_delta_nll_grid.iter().enumerate() {
            if row.len() != nx {
                return Err(Error::Validation(format!(
                    "grid row {} length {} does not match x_values length {}",
                    i,
                    row.len(),
                    nx,
                )));
            }
        }

        let contours = vec![
            extract_contour_marching_squares(
                &x_values,
                &y_values,
                &twice_delta_nll_grid,
                CL_2D_68,
                "68%",
            ),
            extract_contour_marching_squares(
                &x_values,
                &y_values,
                &twice_delta_nll_grid,
                CL_2D_95,
                "95%",
            ),
        ];

        Ok(Self {
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            x_hat,
            y_hat,
            nll_min,
            x_values,
            y_values,
            twice_delta_nll_grid,
            contours,
            grid_points: Vec::new(),
        })
    }
}

/// Simple marching-squares contour extraction on a regular grid.
///
/// Returns a `ContourLine` with vertices tracing the iso-line at `threshold`.
/// For simplicity, this produces an ordered but not necessarily closed polygon;
/// downstream renderers (matplotlib `contour`, plotly) can handle open polylines.
fn extract_contour_marching_squares(
    x_values: &[f64],
    y_values: &[f64],
    grid: &[Vec<f64>],
    threshold: f64,
    cl_label: &str,
) -> ContourLine {
    let nx = x_values.len();
    let ny = y_values.len();
    let mut cx = Vec::new();
    let mut cy = Vec::new();

    for yi in 0..ny.saturating_sub(1) {
        for xi in 0..nx.saturating_sub(1) {
            let v00 = grid[yi][xi];
            let v10 = grid[yi][xi + 1];
            let v01 = grid[yi + 1][xi];
            let v11 = grid[yi + 1][xi + 1];

            if v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan() {
                continue;
            }

            let x0 = x_values[xi];
            let x1 = x_values[xi + 1];
            let y0 = y_values[yi];
            let y1 = y_values[yi + 1];

            // Check each edge for a crossing.
            // Bottom edge (y=y0): v00 → v10
            if (v00 - threshold) * (v10 - threshold) < 0.0 {
                let t = (threshold - v00) / (v10 - v00);
                cx.push(x0 + t * (x1 - x0));
                cy.push(y0);
            }
            // Right edge (x=x1): v10 → v11
            if (v10 - threshold) * (v11 - threshold) < 0.0 {
                let t = (threshold - v10) / (v11 - v10);
                cx.push(x1);
                cy.push(y0 + t * (y1 - y0));
            }
            // Top edge (y=y1): v01 → v11
            if (v01 - threshold) * (v11 - threshold) < 0.0 {
                let t = (threshold - v01) / (v11 - v01);
                cx.push(x0 + t * (x1 - x0));
                cy.push(y1);
            }
            // Left edge (x=x0): v00 → v01
            if (v00 - threshold) * (v01 - threshold) < 0.0 {
                let t = (threshold - v00) / (v01 - v00);
                cx.push(x0);
                cy.push(y0 + t * (y1 - y0));
            }
        }
    }

    // Sort points by angle from centroid for a cleaner polygon.
    if cx.len() > 2 {
        let mean_x: f64 = cx.iter().sum::<f64>() / cx.len() as f64;
        let mean_y: f64 = cy.iter().sum::<f64>() / cy.len() as f64;
        let mut indices: Vec<usize> = (0..cx.len()).collect();
        indices.sort_by(|&a, &b| {
            let angle_a = (cy[a] - mean_y).atan2(cx[a] - mean_x);
            let angle_b = (cy[b] - mean_y).atan2(cx[b] - mean_x);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_x: Vec<f64> = indices.iter().map(|&i| cx[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| cy[i]).collect();
        cx = sorted_x;
        cy = sorted_y;
    }

    ContourLine { cl_label: cl_label.to_string(), threshold, x: cx, y: cy }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parabolic_grid() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        // Simple parabolic bowl: 2*(nll - nll_min) = x^2 + y^2
        let xs: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
        let ys: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
        let grid: Vec<Vec<f64>> =
            ys.iter().map(|&y| xs.iter().map(|&x| x * x + y * y).collect()).collect();
        (xs, ys, grid)
    }

    #[test]
    fn test_contour_from_raw_grid() {
        let (xs, ys, grid) = make_parabolic_grid();
        let art =
            ContourArtifact::from_raw_grid("mu_1", "mu_2", 0.0, 0.0, 0.0, xs, ys, grid).unwrap();
        assert_eq!(art.contours.len(), 2);
        // 68% contour at threshold=2.30 should have points (radius ~1.52)
        assert!(!art.contours[0].x.is_empty(), "68% contour should have vertices");
        // 95% contour at threshold=5.99 should have points (radius ~2.45)
        assert!(!art.contours[1].x.is_empty(), "95% contour should have vertices");
    }

    #[test]
    fn test_contour_from_grid_points() {
        let (xs, ys, grid_2d) = make_parabolic_grid();
        let mut pts = Vec::new();
        for (yi, &y) in ys.iter().enumerate() {
            for (xi, &x) in xs.iter().enumerate() {
                pts.push(ContourGridPoint {
                    x,
                    y,
                    twice_delta_nll: grid_2d[yi][xi],
                    converged: true,
                });
            }
        }
        let art = ContourArtifact::from_grid("mu_1", "mu_2", 0.0, 0.0, 0.0, xs, ys, pts).unwrap();
        assert!(!art.contours[0].x.is_empty());
    }

    #[test]
    fn test_contour_validation_too_small() {
        let r = ContourArtifact::from_raw_grid(
            "x",
            "y",
            0.0,
            0.0,
            0.0,
            vec![0.0],
            vec![0.0],
            vec![vec![0.0]],
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_contour_serialization() {
        let (xs, ys, grid) = make_parabolic_grid();
        let art =
            ContourArtifact::from_raw_grid("mu_1", "mu_2", 0.0, 0.0, 0.0, xs, ys, grid).unwrap();
        let json = serde_json::to_string(&art).unwrap();
        let _back: ContourArtifact = serde_json::from_str(&json).unwrap();
    }
}
