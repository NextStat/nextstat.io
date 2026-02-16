use ns_viz::unfolding::{ResponseMatrixArtifact, UnfoldedSpectrumArtifact};

use crate::canvas::Canvas;
use crate::color::{self, Color};
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

/// Render unfolded spectrum.
pub fn render(artifact: &UnfoldedSpectrumArtifact, config: &VizConfig) -> crate::Result<String> {
    let n = artifact.values.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let x_min = artifact.bin_edges.first().copied().unwrap_or(0.0);
    let x_max = artifact.bin_edges.last().copied().unwrap_or(1.0);
    let y_max = artifact
        .values
        .iter()
        .zip(artifact.errors_total.iter())
        .map(|(&v, &e)| v + e)
        .fold(0.0_f64, f64::max)
        * 1.3;

    let x_axis = Axis::auto_linear(x_min, x_max, 6).with_label(&artifact.observable_label);
    let y_axis = Axis::auto_linear(0.0, y_max, 5).with_label(&artifact.y_label);

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    // Truth values (if present) as histogram steps
    if !artifact.truth_values.is_empty() {
        for bi in 0..artifact.truth_values.len().min(n) {
            let x_lo = artifact.bin_edges[bi];
            let x_hi = artifact.bin_edges[bi + 1];
            let px_lo = x_axis.data_to_pixel(x_lo, area.left, area.right());
            let px_hi = x_axis.data_to_pixel(x_hi, area.left, area.right());
            let py_top = y_axis.data_to_pixel(artifact.truth_values[bi], area.bottom(), area.top);
            let py_bot = y_axis.data_to_pixel(0.0, area.bottom(), area.top);

            canvas.rect(
                px_lo,
                py_top,
                px_hi - px_lo,
                py_bot - py_top,
                &Style {
                    fill: Some(config.colors.expected.with_alpha(0.2)),
                    stroke: Some(config.colors.expected),
                    stroke_width: 1.0,
                    opacity: 1.0,
                },
            );
        }
    }

    // Unfolded data points
    for bi in 0..n {
        let bin = &artifact.bins[bi];
        let x_center = (bin.low + bin.high) / 2.0;
        let px = x_axis.data_to_pixel(x_center, area.left, area.right());
        let py = y_axis.data_to_pixel(bin.value, area.bottom(), area.top);
        let py_lo = y_axis.data_to_pixel(bin.value - bin.error_total, area.bottom(), area.top);
        let py_hi = y_axis.data_to_pixel(bin.value + bin.error_total, area.bottom(), area.top);

        canvas.error_bar(px, py_lo, py_hi, 3.0, &LineStyle::solid(config.colors.observed, 1.0));
        canvas.marker(
            px,
            py,
            &MarkerStyle {
                color: config.colors.observed,
                size: 2.5,
                fill: true,
                ..Default::default()
            },
        );
    }

    canvas.pop_clip();

    let mut entries = vec![LegendEntry {
        label: "Unfolded".into(),
        color: config.colors.observed,
        kind: LegendKind::Marker,
    }];
    if !artifact.truth_values.is_empty() {
        entries.push(LegendEntry {
            label: "Truth".into(),
            color: config.colors.expected,
            kind: LegendKind::FilledRect,
        });
    }
    legend::draw_legend(&mut canvas, &area, &entries, config.font.size, false);

    Ok(canvas.finish_svg())
}

/// Render response matrix as heatmap.
pub fn render_matrix(
    artifact: &ResponseMatrixArtifact,
    config: &VizConfig,
) -> crate::Result<String> {
    let n_reco = artifact.matrix.len();
    if n_reco == 0 {
        return Ok(empty_svg());
    }
    let n_truth = artifact.matrix[0].len();

    let cell_size = if n_reco.max(n_truth) <= 20 { 22.0 } else { 14.0 };
    let margin = 60.0;
    let fig_w = margin + cell_size * n_truth as f64 + 40.0;
    let fig_h = margin + cell_size * n_reco as f64 + 40.0;

    let mut canvas = Canvas::new(fig_w, fig_h)?;
    let area =
        PlotArea::manual(margin, 35.0, cell_size * n_truth as f64, cell_size * n_reco as f64);
    draw_experiment_header(&mut canvas, &area, config);

    let max_val = artifact.matrix.iter().flatten().copied().fold(0.0_f64, f64::max).max(1e-10);

    for row in 0..n_reco {
        for col in 0..n_truth {
            let val = artifact.matrix[row][col];
            let frac = val / max_val;
            let c = color::Color::lerp(Color::rgb(255, 255, 255), config.colors.expected, frac);
            let x = area.left + col as f64 * cell_size;
            let y = area.top + row as f64 * cell_size;
            canvas.rect(x, y, cell_size, cell_size, &Style::filled(c));

            if n_reco.max(n_truth) <= 15 {
                let text_color =
                    if frac > 0.5 { Color::rgb(255, 255, 255) } else { Color::rgb(0, 0, 0) };
                canvas.text(
                    x + cell_size / 2.0,
                    y + cell_size / 2.0,
                    &format!("{:.2}", val),
                    &TextStyle {
                        size: (cell_size * 0.35).min(7.0),
                        color: text_color,
                        anchor: TextAnchor::Middle,
                        baseline: TextBaseline::Central,
                        ..Default::default()
                    },
                );
            }
        }
    }

    // Labels
    canvas.text(
        area.left + area.width / 2.0,
        area.bottom() + 20.0,
        "Truth bin",
        &TextStyle {
            size: config.font.label_size,
            anchor: TextAnchor::Middle,
            ..Default::default()
        },
    );
    canvas.text_rotated(
        area.left - 30.0,
        area.top + area.height / 2.0,
        "Reco bin",
        &TextStyle {
            size: config.font.label_size,
            anchor: TextAnchor::Middle,
            ..Default::default()
        },
        -90.0,
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No unfolding data</text></svg>"#.into()
}
