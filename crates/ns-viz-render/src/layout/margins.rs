use crate::canvas::Canvas;
use crate::config::VizConfig;
use crate::layout::axes::Axis;
use crate::primitives::TextStyle;

/// Rectangular plot area within the canvas.
#[derive(Debug, Clone, Copy)]
pub struct PlotArea {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

impl PlotArea {
    pub fn right(&self) -> f64 {
        self.left + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.top + self.height
    }

    /// Compute auto-margins from axis labels and config.
    pub fn auto(
        canvas: &Canvas,
        y_axis: Option<&Axis>,
        x_axis: Option<&Axis>,
        config: &VizConfig,
    ) -> Self {
        let tick_style = TextStyle { size: config.font.tick_size, ..Default::default() };
        let label_style = TextStyle { size: config.font.label_size, ..Default::default() };

        // Left margin: y-axis tick labels + axis label + padding
        let mut left = 15.0; // base padding
        if let Some(y) = y_axis {
            let max_tick_w = y
                .tick_labels
                .iter()
                .map(|l| canvas.measure_text(l, &tick_style).width)
                .fold(0.0_f64, f64::max);
            left += max_tick_w + 8.0; // tick label + gap
            if !y.label.is_empty() {
                left += label_style.size + 6.0; // axis label (rotated)
            }
        }

        // Bottom margin: x-axis tick labels + axis label + padding
        let mut bottom = 15.0;
        if let Some(x) = x_axis {
            bottom += tick_style.size + 6.0; // tick labels
            if !x.label.is_empty() {
                bottom += label_style.size + 6.0;
            }
        }

        // Top margin: header space
        let top = if config.experiment.name.is_empty() {
            12.0
        } else {
            config.font.label_size * 1.3 + 20.0
        };

        // Right margin
        let right = 15.0;

        let width = canvas.width - left - right;
        let height = canvas.height - top - bottom;

        Self { left, top, width: width.max(50.0), height: height.max(50.0) }
    }

    /// Manual margins (for multi-panel layouts).
    pub fn manual(left: f64, top: f64, width: f64, height: f64) -> Self {
        Self { left, top, width, height }
    }
}
