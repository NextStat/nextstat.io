use ns_viz::profile::ProfileCurveArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &ProfileCurveArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.mu_values.is_empty() {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    // Axes
    let mu_min = artifact.mu_values.first().copied().unwrap_or(0.0);
    let mu_max = artifact.mu_values.last().copied().unwrap_or(5.0);
    let x_axis = Axis::auto_linear(mu_min, mu_max, 6).with_label("\u{03BC}");

    let y_max = artifact.twice_delta_nll.iter().copied().fold(0.0_f64, f64::max).clamp(4.0, 10.0);
    let y_axis = Axis::auto_linear(0.0, y_max, 5).with_label("2\u{0394}NLL");

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    // Clip to plot area
    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    // Profile curve
    let points: Vec<(f64, f64)> = artifact
        .mu_values
        .iter()
        .zip(artifact.twice_delta_nll.iter())
        .map(|(&mu, &dnll)| {
            let px = x_axis.data_to_pixel(mu, area.left, area.right());
            let py = y_axis.data_to_pixel(dnll, area.bottom(), area.top);
            (px, py)
        })
        .collect();

    canvas.polyline(&points, &LineStyle::solid(config.colors.observed, 1.5));

    // Horizontal threshold lines (1σ, 2σ)
    let sigma_thresholds = [(1.0, "1\u{03C3}"), (3.84, "2\u{03C3}")];
    for (thresh, label) in sigma_thresholds {
        if thresh <= y_max {
            let py = y_axis.data_to_pixel(thresh, area.bottom(), area.top);
            canvas.line(
                area.left,
                py,
                area.right(),
                py,
                &LineStyle::dashed(Color::rgb(180, 80, 80), 0.8),
            );
            let label_style = TextStyle {
                size: config.font.tick_size * 0.85,
                color: Color::rgb(180, 80, 80),
                anchor: TextAnchor::End,
                ..Default::default()
            };
            canvas.text(area.right() - 4.0, py - 3.0, label, &label_style);
        }
    }

    // Vertical line at mu_hat
    let mu_hat_px = x_axis.data_to_pixel(artifact.mu_hat, area.left, area.right());
    canvas.line(
        mu_hat_px,
        area.top,
        mu_hat_px,
        area.bottom(),
        &LineStyle::dashed(config.colors.expected, 0.8),
    );

    // Label mu_hat
    let hat_label_style = TextStyle {
        size: config.font.tick_size * 0.85,
        color: config.colors.expected,
        anchor: TextAnchor::Start,
        ..Default::default()
    };
    canvas.text(
        mu_hat_px + 3.0,
        area.top + 12.0,
        &format!("\u{03BC}\u{0302} = {:.3}", artifact.mu_hat),
        &hat_label_style,
    );

    canvas.pop_clip();

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No profile data</text></svg>"#.into()
}
