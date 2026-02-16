use ns_viz::significance::SignificanceScanArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &SignificanceScanArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.scan_values.is_empty() {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let x_min = artifact.scan_values.first().copied().unwrap_or(0.0);
    let x_max = artifact.scan_values.last().copied().unwrap_or(1.0);
    let x_axis = Axis::auto_linear(x_min, x_max, 6).with_label(&artifact.scan_label);

    let z_max = artifact
        .z_obs_values
        .iter()
        .chain(artifact.z_exp_values.iter())
        .copied()
        .fold(0.0_f64, f64::max)
        .max(5.5)
        * 1.1;
    let y_axis = Axis::auto_linear(0.0, z_max, 5).with_label("Local significance (Z)");

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    // Threshold lines (3σ, 5σ)
    for (z, label) in [(3.0, "3\u{03C3}"), (5.0, "5\u{03C3}")] {
        if z < z_max {
            let py = y_axis.data_to_pixel(z, area.bottom(), area.top);
            canvas.line(
                area.left,
                py,
                area.right(),
                py,
                &LineStyle::dashed(Color::rgb(180, 80, 80), 0.7),
            );
            canvas.text(
                area.right() - 4.0,
                py - 4.0,
                label,
                &TextStyle {
                    size: config.font.tick_size * 0.8,
                    color: Color::rgb(180, 80, 80),
                    anchor: TextAnchor::End,
                    ..Default::default()
                },
            );
        }
    }

    // Z_obs line
    let obs_pts: Vec<(f64, f64)> = artifact
        .scan_values
        .iter()
        .zip(artifact.z_obs_values.iter())
        .map(|(&x, &z)| {
            (
                x_axis.data_to_pixel(x, area.left, area.right()),
                y_axis.data_to_pixel(z, area.bottom(), area.top),
            )
        })
        .collect();
    canvas.polyline(&obs_pts, &LineStyle::solid(config.colors.observed, 1.5));

    // Z_exp line (dashed)
    let exp_pts: Vec<(f64, f64)> = artifact
        .scan_values
        .iter()
        .zip(artifact.z_exp_values.iter())
        .map(|(&x, &z)| {
            (
                x_axis.data_to_pixel(x, area.left, area.right()),
                y_axis.data_to_pixel(z, area.bottom(), area.top),
            )
        })
        .collect();
    canvas.polyline(&exp_pts, &LineStyle::dashed(config.colors.expected, 1.2));

    canvas.pop_clip();

    legend::draw_legend(
        &mut canvas,
        &area,
        &[
            LegendEntry {
                label: "Z observed".into(),
                color: config.colors.observed,
                kind: LegendKind::Line(None),
            },
            LegendEntry {
                label: "Z expected".into(),
                color: config.colors.expected,
                kind: LegendKind::Line(Some("6 3".into())),
            },
        ],
        config.font.size,
        false,
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No significance data</text></svg>"#.into()
}
