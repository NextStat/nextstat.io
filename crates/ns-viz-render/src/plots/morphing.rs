use ns_viz::morphing::MorphingArtifact;

use crate::canvas::Canvas;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &MorphingArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.bin_edges.len() < 2 {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let x_min = artifact.bin_edges.first().copied().unwrap_or(0.0);
    let x_max = artifact.bin_edges.last().copied().unwrap_or(1.0);
    let y_max = artifact
        .templates
        .iter()
        .flat_map(|t| t.bin_contents.iter())
        .chain(artifact.interpolated.iter())
        .copied()
        .fold(0.0_f64, f64::max)
        * 1.2;

    let x_axis = Axis::auto_linear(x_min, x_max, 6).with_label(&artifact.observable_label);
    let y_axis = Axis::auto_linear(0.0, y_max, 5).with_label("Events");

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);
    let palette = config.palette_colors();
    let n_bins = artifact.bin_edges.len() - 1;

    let mut legend_entries = Vec::new();

    // Template lines (step histogram style)
    for (ti, tmpl) in artifact.templates.iter().enumerate() {
        let color = palette[ti % palette.len()];
        let pts: Vec<(f64, f64)> = (0..n_bins.min(tmpl.bin_contents.len()))
            .flat_map(|bi| {
                let x_lo = artifact.bin_edges[bi];
                let x_hi = artifact.bin_edges[bi + 1];
                let y = tmpl.bin_contents[bi];
                vec![
                    (
                        x_axis.data_to_pixel(x_lo, area.left, area.right()),
                        y_axis.data_to_pixel(y, area.bottom(), area.top),
                    ),
                    (
                        x_axis.data_to_pixel(x_hi, area.left, area.right()),
                        y_axis.data_to_pixel(y, area.bottom(), area.top),
                    ),
                ]
            })
            .collect();
        canvas.polyline(&pts, &LineStyle::solid(color, 1.0));
        legend_entries.push(LegendEntry {
            label: tmpl.label.clone(),
            color,
            kind: LegendKind::Line(None),
        });
    }

    // Interpolated (dashed, black)
    if !artifact.interpolated.is_empty() {
        let pts: Vec<(f64, f64)> = (0..n_bins.min(artifact.interpolated.len()))
            .flat_map(|bi| {
                let x_lo = artifact.bin_edges[bi];
                let x_hi = artifact.bin_edges[bi + 1];
                let y = artifact.interpolated[bi];
                vec![
                    (
                        x_axis.data_to_pixel(x_lo, area.left, area.right()),
                        y_axis.data_to_pixel(y, area.bottom(), area.top),
                    ),
                    (
                        x_axis.data_to_pixel(x_hi, area.left, area.right()),
                        y_axis.data_to_pixel(y, area.bottom(), area.top),
                    ),
                ]
            })
            .collect();
        canvas.polyline(&pts, &LineStyle::dashed(config.colors.observed, 1.5));
        legend_entries.push(LegendEntry {
            label: format!("Interp. ({} = {:.2})", artifact.parameter_label, artifact.target_value),
            color: config.colors.observed,
            kind: LegendKind::Line(Some("6 3".into())),
        });
    }

    canvas.pop_clip();
    legend::draw_legend(&mut canvas, &area, &legend_entries, config.font.size, false);

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No morphing data</text></svg>"#.into()
}
