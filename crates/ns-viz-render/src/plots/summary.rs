use ns_viz::summary::SummaryArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_frame;
use crate::primitives::*;

pub fn render(artifact: &SummaryArtifact, config: &VizConfig) -> crate::Result<String> {
    let entries = &artifact.entries;
    let n = entries.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let row_h = 28.0;
    let fig_w = config.figure.width;
    let fig_h = (row_h * n as f64 + 100.0).max(200.0);
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let label_style = TextStyle { size: config.font.tick_size, ..Default::default() };
    let label_w = entries
        .iter()
        .map(|e| canvas.measure_text(&e.label, &label_style).width)
        .fold(0.0_f64, f64::max)
        + 10.0;

    let area = PlotArea::manual(label_w + 15.0, 35.0, fig_w - label_w - 30.0, row_h * n as f64);
    draw_experiment_header(&mut canvas, &area, config);

    // Determine mu range
    let mu_min = entries.iter().map(|e| e.mu_hat - 2.0 * e.sigma).fold(f64::INFINITY, f64::min);
    let mu_max = entries.iter().map(|e| e.mu_hat + 2.0 * e.sigma).fold(f64::NEG_INFINITY, f64::max);
    let range = (mu_max - mu_min).max(0.1);
    let pad = range * 0.15;
    let x_min = mu_min - pad;
    let x_max = mu_max + pad;

    let palette = config.palette_colors();

    for (i, entry) in entries.iter().enumerate() {
        let y = area.top + (i as f64 + 0.5) * row_h;
        let color = palette[i % palette.len()];

        // Label
        let name_style = TextStyle {
            size: config.font.tick_size,
            anchor: TextAnchor::End,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(area.left - 5.0, y, &entry.label, &name_style);

        // Horizontal error bar
        let mu_px = area.left + (entry.mu_hat - x_min) / (x_max - x_min) * area.width;
        let lo_px = area.left + (entry.mu_hat - entry.sigma - x_min) / (x_max - x_min) * area.width;
        let hi_px = area.left + (entry.mu_hat + entry.sigma - x_min) / (x_max - x_min) * area.width;

        canvas.error_bar_h(lo_px, hi_px, y, 6.0, &LineStyle::solid(color, 1.5));
        canvas.marker(
            mu_px,
            y,
            &MarkerStyle { color, size: 3.5, fill: true, ..Default::default() },
        );

        // Value annotation
        let val_style = TextStyle {
            size: config.font.tick_size * 0.85,
            color: Color::rgb(100, 100, 100),
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(
            hi_px + 6.0,
            y,
            &format!("{:.3} \u{00B1} {:.3}", entry.mu_hat, entry.sigma),
            &val_style,
        );
    }

    // X-axis label
    let poi_name = &artifact.meta.poi_name;
    canvas.text(
        area.left + area.width / 2.0,
        area.bottom() + 20.0,
        poi_name,
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
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No summary data</text></svg>"#.into()
}
