use ns_viz::uncertainty::UncertaintyBreakdownArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_frame;
use crate::primitives::*;

pub fn render(
    artifact: &UncertaintyBreakdownArtifact,
    config: &VizConfig,
) -> crate::Result<String> {
    let groups = &artifact.groups;
    let n = groups.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let row_h = 22.0;
    let fig_w = config.figure.width;
    let fig_h = (row_h * (n + 1) as f64 + 100.0).max(230.0); // +1 for total
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let label_style = TextStyle { size: config.font.tick_size, ..Default::default() };
    let label_w = groups
        .iter()
        .map(|g| canvas.measure_text(&g.name, &label_style).width)
        .fold(0.0_f64, f64::max)
        .max(canvas.measure_text("Total", &label_style).width)
        + 10.0;

    let area =
        PlotArea::manual(label_w + 15.0, 35.0, fig_w - label_w - 30.0, row_h * (n + 1) as f64);
    draw_experiment_header(&mut canvas, &area, config);

    let max_impact =
        groups.iter().map(|g| g.impact.abs()).fold(artifact.total.abs(), f64::max).max(0.001);
    let px_per_unit = area.width / (max_impact * 1.2);
    let palette = config.palette_colors();

    for (i, group) in groups.iter().enumerate() {
        let y = area.top + (i as f64 + 0.5) * row_h;
        let color = palette[i % palette.len()];

        let name_style = TextStyle {
            size: config.font.tick_size,
            anchor: TextAnchor::End,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(area.left - 5.0, y, &group.name, &name_style);

        let bar_w = group.impact.abs() * px_per_unit;
        canvas.rect(
            area.left,
            y - row_h * 0.3,
            bar_w,
            row_h * 0.6,
            &Style::filled(color.with_alpha(0.8)),
        );

        // Value annotation
        let val_style = TextStyle {
            size: config.font.tick_size * 0.85,
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(area.left + bar_w + 4.0, y, &format!("{:.4}", group.impact), &val_style);
    }

    // Total bar
    let total_y = area.top + (n as f64 + 0.5) * row_h;
    let total_style = TextStyle {
        size: config.font.tick_size,
        anchor: TextAnchor::End,
        baseline: TextBaseline::Central,
        weight: FontWeight::Bold,
        ..Default::default()
    };
    canvas.text(area.left - 5.0, total_y, "Total", &total_style);
    let total_w = artifact.total.abs() * px_per_unit;
    canvas.rect(
        area.left,
        total_y - row_h * 0.3,
        total_w,
        row_h * 0.6,
        &Style::filled(Color::rgb(60, 60, 60)),
    );
    let val_style = TextStyle {
        size: config.font.tick_size * 0.85,
        anchor: TextAnchor::Start,
        baseline: TextBaseline::Central,
        weight: FontWeight::Bold,
        ..Default::default()
    };
    canvas.text(area.left + total_w + 4.0, total_y, &format!("{:.4}", artifact.total), &val_style);

    // Separator line above total
    let sep_y = area.top + n as f64 * row_h;
    canvas.line(
        area.left,
        sep_y,
        area.right(),
        sep_y,
        &LineStyle::solid(Color::rgb(200, 200, 200), 0.5),
    );

    draw_frame(&mut canvas, &area);
    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No uncertainty data</text></svg>"#.into()
}
