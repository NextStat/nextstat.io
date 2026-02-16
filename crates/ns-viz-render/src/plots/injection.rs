use ns_viz::injection::InjectionArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &InjectionArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.points.is_empty() {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let mu_min = artifact.mu_injected_values.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = artifact.mu_injected_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (mu_max - mu_min).max(0.1) * 0.1;

    let x_axis = Axis::auto_linear(mu_min - pad, mu_max + pad, 6)
        .with_label(format!("{} (injected)", artifact.poi_label));
    let y_axis = Axis::auto_linear(mu_min - pad, mu_max + pad, 6)
        .with_label(format!("{} (fitted)", artifact.poi_label));

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    // Diagonal y=x reference line
    let diag_x0 = x_axis.data_to_pixel(x_axis.min, area.left, area.right());
    let diag_y0 = y_axis.data_to_pixel(x_axis.min, area.bottom(), area.top);
    let diag_x1 = x_axis.data_to_pixel(x_axis.max, area.left, area.right());
    let diag_y1 = y_axis.data_to_pixel(x_axis.max, area.bottom(), area.top);
    canvas.line(
        diag_x0,
        diag_y0,
        diag_x1,
        diag_y1,
        &LineStyle::dashed(Color::rgb(180, 180, 180), 0.8),
    );

    // Fit line (slope, intercept)
    let fit_y0 = artifact.linearity_slope * x_axis.min + artifact.linearity_intercept;
    let fit_y1 = artifact.linearity_slope * x_axis.max + artifact.linearity_intercept;
    let fit_px0 = x_axis.data_to_pixel(x_axis.min, area.left, area.right());
    let fit_py0 = y_axis.data_to_pixel(fit_y0, area.bottom(), area.top);
    let fit_px1 = x_axis.data_to_pixel(x_axis.max, area.left, area.right());
    let fit_py1 = y_axis.data_to_pixel(fit_y1, area.bottom(), area.top);
    canvas.line(fit_px0, fit_py0, fit_px1, fit_py1, &LineStyle::solid(config.colors.expected, 1.2));

    // Data points
    for pt in &artifact.points {
        let px = x_axis.data_to_pixel(pt.mu_injected, area.left, area.right());
        let py = y_axis.data_to_pixel(pt.mu_hat_mean, area.bottom(), area.top);
        let py_lo = y_axis.data_to_pixel(pt.mu_hat_mean - pt.mu_hat_std, area.bottom(), area.top);
        let py_hi = y_axis.data_to_pixel(pt.mu_hat_mean + pt.mu_hat_std, area.bottom(), area.top);

        canvas.error_bar(px, py_lo, py_hi, 3.0, &LineStyle::solid(config.colors.observed, 1.0));
        canvas.marker(
            px,
            py,
            &MarkerStyle {
                color: config.colors.observed,
                size: 3.0,
                fill: true,
                ..Default::default()
            },
        );
    }

    canvas.pop_clip();

    // Annotation: slope
    canvas.text(
        area.left + 5.0,
        area.top + 14.0,
        &format!(
            "slope = {:.4}, intercept = {:.4}",
            artifact.linearity_slope, artifact.linearity_intercept
        ),
        &TextStyle { size: config.font.tick_size * 0.9, ..Default::default() },
    );

    legend::draw_legend(
        &mut canvas,
        &area,
        &[
            LegendEntry {
                label: "Data".into(),
                color: config.colors.observed,
                kind: LegendKind::Marker,
            },
            LegendEntry {
                label: "Fit".into(),
                color: config.colors.expected,
                kind: LegendKind::Line(None),
            },
            LegendEntry {
                label: "y = x".into(),
                color: Color::rgb(180, 180, 180),
                kind: LegendKind::Line(Some("6 3".into())),
            },
        ],
        config.font.size,
        false,
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No injection data</text></svg>"#.into()
}
