use ns_viz::contour::ContourArtifact;

use crate::canvas::Canvas;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &ContourArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.contours.is_empty() && artifact.grid_points.is_empty() {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let x_min = artifact.x_values.first().copied().unwrap_or(0.0);
    let x_max = artifact.x_values.last().copied().unwrap_or(1.0);
    let y_min = artifact.y_values.first().copied().unwrap_or(0.0);
    let y_max = artifact.y_values.last().copied().unwrap_or(1.0);

    let x_axis = Axis::auto_linear(x_min, x_max, 6).with_label(&artifact.x_label);
    let y_axis = Axis::auto_linear(y_min, y_max, 6).with_label(&artifact.y_label);

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    let contour_colors =
        [config.colors.expected, config.colors.expected.with_alpha(0.6), config.colors.signal];

    // Draw contour lines
    for (ci, contour) in artifact.contours.iter().enumerate() {
        let color = contour_colors[ci % contour_colors.len()];
        let pts: Vec<(f64, f64)> = contour
            .x
            .iter()
            .zip(contour.y.iter())
            .map(|(&cx, &cy)| {
                (
                    x_axis.data_to_pixel(cx, area.left, area.right()),
                    y_axis.data_to_pixel(cy, area.bottom(), area.top),
                )
            })
            .collect();

        if pts.len() >= 2 {
            canvas.polyline(&pts, &LineStyle::solid(color, 1.5));
        }

        // Label
        if let Some(&(lx, ly)) = pts.first() {
            canvas.text(
                lx + 3.0,
                ly - 3.0,
                &contour.cl_label,
                &TextStyle { size: config.font.tick_size * 0.8, color, ..Default::default() },
            );
        }
    }

    // Best-fit marker
    let bf_x = x_axis.data_to_pixel(artifact.x_hat, area.left, area.right());
    let bf_y = y_axis.data_to_pixel(artifact.y_hat, area.bottom(), area.top);
    canvas.marker(
        bf_x,
        bf_y,
        &MarkerStyle {
            color: config.colors.observed,
            size: 4.0,
            fill: true,
            shape: MarkerShape::Diamond,
        },
    );

    canvas.pop_clip();
    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No contour data</text></svg>"#.into()
}
