use ns_viz::corr::CorrArtifact;

use crate::canvas::Canvas;
use crate::color::{self, Color};
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

pub fn render(artifact: &CorrArtifact, config: &VizConfig) -> crate::Result<String> {
    let n = artifact.parameter_names.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let max_label_chars = artifact.parameter_names.iter().map(|s| s.len()).max().unwrap_or(5);

    let cell_size = if n <= 20 { 22.0 } else { 14.0 };
    let label_margin = (max_label_chars as f64 * config.font.tick_size * 0.5).min(180.0);
    let colorbar_w = 20.0;
    let colorbar_gap = 10.0;

    let matrix_w = cell_size * n as f64;
    let matrix_h = matrix_w;
    let fig_w = label_margin + matrix_w + colorbar_gap + colorbar_w + 40.0;
    let fig_h = 45.0 + matrix_h + label_margin + 10.0;

    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let area = PlotArea::manual(label_margin + 10.0, 35.0, matrix_w, matrix_h);
    draw_experiment_header(&mut canvas, &area, config);

    // Draw cells
    let annotate = n <= 20;
    for row in 0..n {
        for col in 0..n {
            let val = artifact.corr[row][col];
            let cell_color = color::rdbu_r(val);

            let x = area.left + col as f64 * cell_size;
            let y = area.top + row as f64 * cell_size;

            canvas.rect(x, y, cell_size, cell_size, &Style::filled(cell_color));

            if annotate {
                let text_color =
                    if val.abs() > 0.6 { Color::rgb(255, 255, 255) } else { Color::rgb(0, 0, 0) };
                let text_style = TextStyle {
                    size: (cell_size * 0.4).min(8.0),
                    color: text_color,
                    anchor: TextAnchor::Middle,
                    baseline: TextBaseline::Central,
                    ..Default::default()
                };
                let label = format!("{:.2}", val);
                canvas.text(x + cell_size / 2.0, y + cell_size / 2.0, &label, &text_style);
            }
        }
    }

    // Row labels (left)
    let row_label_style = TextStyle {
        size: config.font.tick_size * 0.85,
        anchor: TextAnchor::End,
        baseline: TextBaseline::Central,
        ..Default::default()
    };
    for (i, name) in artifact.parameter_names.iter().enumerate() {
        let y = area.top + (i as f64 + 0.5) * cell_size;
        canvas.text(area.left - 4.0, y, name, &row_label_style);
    }

    // Column labels (bottom, rotated)
    let col_label_style = TextStyle {
        size: config.font.tick_size * 0.85,
        anchor: TextAnchor::End,
        baseline: TextBaseline::Central,
        ..Default::default()
    };
    for (i, name) in artifact.parameter_names.iter().enumerate() {
        let x = area.left + (i as f64 + 0.5) * cell_size;
        let y = area.bottom() + 4.0;
        canvas.text_rotated(x, y, name, &col_label_style, 45.0);
    }

    // Colorbar
    let cb_x = area.right() + colorbar_gap;
    let cb_steps = 50;
    let cb_h = matrix_h / cb_steps as f64;
    for i in 0..cb_steps {
        let val = 1.0 - 2.0 * i as f64 / (cb_steps - 1) as f64; // +1 → -1
        let c = color::rdbu_r(val);
        let y = area.top + i as f64 * cb_h;
        canvas.rect(cb_x, y, colorbar_w, cb_h + 0.5, &Style::filled(c));
    }
    // Colorbar labels
    let cb_label_style = TextStyle {
        size: config.font.tick_size * 0.85,
        anchor: TextAnchor::Start,
        baseline: TextBaseline::Central,
        ..Default::default()
    };
    canvas.text(cb_x + colorbar_w + 3.0, area.top, "+1.0", &cb_label_style);
    canvas.text(cb_x + colorbar_w + 3.0, area.top + matrix_h / 2.0, "0.0", &cb_label_style);
    canvas.text(cb_x + colorbar_w + 3.0, area.bottom(), "\u{2212}1.0", &cb_label_style);

    // Frame
    let frame_style = LineStyle::solid(Color::rgb(0, 0, 0), 0.5);
    canvas.line(area.left, area.top, area.right(), area.top, &frame_style);
    canvas.line(area.left, area.bottom(), area.right(), area.bottom(), &frame_style);
    canvas.line(area.left, area.top, area.left, area.bottom(), &frame_style);
    canvas.line(area.right(), area.top, area.right(), area.bottom(), &frame_style);

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No correlation data</text></svg>"#.into()
}
