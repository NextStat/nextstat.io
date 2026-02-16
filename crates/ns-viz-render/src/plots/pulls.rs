use ns_viz::pulls::PullsArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_frame;
use crate::primitives::*;

/// Render pulls artifact to SVG string.
pub fn render(artifact: &PullsArtifact, config: &VizConfig) -> crate::Result<String> {
    let entries = &artifact.entries;
    let n = entries.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let row_h = 20.0;
    let fig_w = config.figure.width;
    let fig_h = (row_h * n as f64 + 100.0).max(230.0);

    let mut canvas = Canvas::new(fig_w, fig_h)?;

    // Layout: label area left, pull bars center, constraint bars right
    let label_w = {
        let style = TextStyle { size: config.font.tick_size, ..Default::default() };
        entries.iter().map(|e| canvas.measure_text(&e.name, &style).width).fold(0.0_f64, f64::max)
            + 10.0
    };

    let area = PlotArea::manual(label_w + 15.0, 35.0, fig_w - label_w - 30.0, row_h * n as f64);

    draw_experiment_header(&mut canvas, &area, config);

    // Background bands (±1σ, ±2σ)
    let x_center = area.left + area.width / 2.0;
    let px_per_sigma = area.width / 6.0; // ±3σ range

    // ±2σ band
    canvas.rect(
        x_center - 2.0 * px_per_sigma,
        area.top,
        4.0 * px_per_sigma,
        area.height,
        &Style { fill: Some(config.colors.band_2sigma.with_alpha(0.35)), ..Default::default() },
    );
    // ±1σ band
    canvas.rect(
        x_center - px_per_sigma,
        area.top,
        2.0 * px_per_sigma,
        area.height,
        &Style { fill: Some(config.colors.band_1sigma.with_alpha(0.35)), ..Default::default() },
    );

    // Center line
    canvas.line(
        x_center,
        area.top,
        x_center,
        area.bottom(),
        &LineStyle::dashed(Color::rgb(100, 100, 100), 0.6),
    );

    // Draw each entry
    for (i, entry) in entries.iter().enumerate() {
        let y = area.top + (i as f64 + 0.5) * row_h;

        // Label
        let label_style = TextStyle {
            size: config.font.tick_size,
            anchor: TextAnchor::End,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(area.left - 5.0, y, &entry.name, &label_style);

        // Pull bar
        let pull_px = x_center + entry.pull * px_per_sigma;
        let bar_color = if entry.pull >= 0.0 {
            config.colors.positive_pull
        } else {
            config.colors.negative_pull
        };
        let bar_w = (pull_px - x_center).abs();
        let bar_x = pull_px.min(x_center);
        canvas.rect(
            bar_x,
            y - row_h * 0.3,
            bar_w,
            row_h * 0.6,
            &Style::filled(bar_color.with_alpha(0.7)),
        );

        // Constraint error bar (θ̂ ± σ̂/σ₀)
        let err_lo = x_center + (entry.pull - entry.constraint) * px_per_sigma;
        let err_hi = x_center + (entry.pull + entry.constraint) * px_per_sigma;
        canvas.error_bar(pull_px, err_lo, err_hi, 3.0, &LineStyle::solid(Color::rgb(0, 0, 0), 1.0));

        // Marker at pull value
        canvas.marker(
            pull_px,
            y,
            &MarkerStyle {
                color: Color::rgb(0, 0, 0),
                size: 2.0,
                fill: true,
                ..Default::default()
            },
        );
    }

    // Sigma labels at bottom
    let sigma_y = area.bottom() + 14.0;
    let sigma_style = TextStyle {
        size: config.font.tick_size,
        anchor: TextAnchor::Middle,
        color: Color::rgb(80, 80, 80),
        ..Default::default()
    };
    for s in [-3, -2, -1, 0, 1, 2, 3] {
        let px = x_center + s as f64 * px_per_sigma;
        canvas.text(px, sigma_y, &format!("{s}"), &sigma_style);
    }

    // Pull label
    let pull_label_style = TextStyle {
        size: config.font.label_size,
        anchor: TextAnchor::Middle,
        ..Default::default()
    };
    canvas.text(
        x_center,
        area.bottom() + 28.0,
        "(\u{03B8}\u{0302} \u{2212} \u{03B8}\u{2080}) / \u{0394}\u{03B8}",
        &pull_label_style,
    );

    draw_frame(&mut canvas, &area);

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No pull entries</text></svg>"#.into()
}
