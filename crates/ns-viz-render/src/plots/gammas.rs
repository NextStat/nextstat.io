use ns_viz::gammas::GammasArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_frame;
use crate::primitives::*;

pub fn render(artifact: &GammasArtifact, config: &VizConfig) -> crate::Result<String> {
    let entries = &artifact.entries;
    let n = entries.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let row_h = 18.0;
    let fig_w = config.figure.width;
    let fig_h = (row_h * n as f64 + 100.0).max(230.0);
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let label_style = TextStyle { size: config.font.tick_size * 0.9, ..Default::default() };
    let label_w = entries
        .iter()
        .map(|e| canvas.measure_text(&e.name, &label_style).width)
        .fold(0.0_f64, f64::max)
        + 10.0;

    let area = PlotArea::manual(label_w + 15.0, 35.0, fig_w - label_w - 30.0, row_h * n as f64);
    draw_experiment_header(&mut canvas, &area, config);

    let x_center = area.left + area.width / 2.0;
    let x_range = 1.0; // ±0.5 around 1.0
    let px_per_unit = area.width / x_range;

    // ±1σ band around 1.0
    canvas.rect(
        x_center - 0.5 * px_per_unit * 0.5,
        area.top,
        px_per_unit * 0.5,
        area.height,
        &Style::filled(config.colors.band_1sigma.with_alpha(0.3)),
    );

    // Reference line at γ=1.0
    canvas.line(
        x_center,
        area.top,
        x_center,
        area.bottom(),
        &LineStyle::dashed(Color::rgb(150, 150, 150), 0.6),
    );

    for (i, entry) in entries.iter().enumerate() {
        let y = area.top + (i as f64 + 0.5) * row_h;

        // Label
        let name_style = TextStyle {
            size: config.font.tick_size * 0.9,
            anchor: TextAnchor::End,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(area.left - 4.0, y, &entry.name, &name_style);

        // Postfit point + error bar
        let val_px = x_center + (entry.postfit_value - 1.0) * px_per_unit;
        let lo_px = x_center + (entry.postfit_value - entry.postfit_sigma - 1.0) * px_per_unit;
        let hi_px = x_center + (entry.postfit_value + entry.postfit_sigma - 1.0) * px_per_unit;

        canvas.error_bar(
            val_px,
            lo_px.min(hi_px),
            lo_px.max(hi_px),
            0.0,
            &LineStyle::solid(config.colors.expected, 1.0),
        );
        // Using horizontal error bar: lo_px and hi_px are x coordinates, y is fixed
        canvas.error_bar_h(lo_px, hi_px, y, 3.0, &LineStyle::solid(config.colors.expected, 1.0));
        canvas.marker(
            val_px,
            y,
            &MarkerStyle {
                color: config.colors.expected,
                size: 2.0,
                fill: true,
                ..Default::default()
            },
        );
    }

    // Tick labels
    let tick_y = area.bottom() + 12.0;
    let tick_style =
        TextStyle { size: config.font.tick_size, anchor: TextAnchor::Middle, ..Default::default() };
    for val in [0.5, 0.75, 1.0, 1.25, 1.5] {
        let px = x_center + (val - 1.0) * px_per_unit;
        if px >= area.left && px <= area.right() {
            canvas.text(px, tick_y, &format!("{val:.2}"), &tick_style);
        }
    }

    canvas.text(
        x_center,
        area.bottom() + 26.0,
        "\u{03B3}",
        &TextStyle {
            size: config.font.label_size,
            anchor: TextAnchor::Middle,
            ..Default::default()
        },
    );

    draw_frame(&mut canvas, &area);
    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No gamma entries</text></svg>"#.into()
}
