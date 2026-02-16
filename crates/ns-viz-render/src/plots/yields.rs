use ns_viz::yields::YieldsArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

pub fn render(artifact: &YieldsArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.channels.is_empty() {
        return Ok(empty_svg());
    }

    let channels = &artifact.channels;
    let all_samples: Vec<&str> = {
        let mut names: Vec<&str> = Vec::new();
        for ch in channels {
            for s in &ch.samples {
                if !names.contains(&s.name.as_str()) {
                    names.push(&s.name);
                }
            }
        }
        names
    };

    let col_w = 90.0;
    let row_h = 18.0;
    let label_col_w = 120.0;
    let n_rows = all_samples.len() + 3; // header + samples + total + data
    let n_cols = channels.len();

    let fig_w = (label_col_w + col_w * n_cols as f64 + 30.0).max(300.0);
    let fig_h = (row_h * n_rows as f64 + 80.0).max(200.0);
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let area = PlotArea::manual(15.0, 35.0, fig_w - 30.0, fig_h - 50.0);
    draw_experiment_header(&mut canvas, &area, config);

    let table_x = area.left;
    let table_y = area.top + 5.0;

    let header_style =
        TextStyle { size: config.font.tick_size, weight: FontWeight::Bold, ..Default::default() };
    let cell_style = TextStyle {
        size: config.font.tick_size * 0.9,
        anchor: TextAnchor::End,
        ..Default::default()
    };
    let label_cell_style = TextStyle { size: config.font.tick_size * 0.9, ..Default::default() };

    // Header row: sample names as row headers, channel names as column headers
    canvas.text(table_x, table_y, "Sample", &header_style);
    for (ci, ch) in channels.iter().enumerate() {
        let cx = table_x + label_col_w + ci as f64 * col_w + col_w / 2.0;
        canvas.text(
            cx,
            table_y,
            &ch.channel_name,
            &TextStyle { anchor: TextAnchor::Middle, ..header_style.clone() },
        );
    }

    // Separator line
    let sep_y = table_y + row_h * 0.6;
    canvas.line(
        table_x,
        sep_y,
        table_x + label_col_w + n_cols as f64 * col_w,
        sep_y,
        &LineStyle::solid(Color::rgb(200, 200, 200), 0.5),
    );

    // Sample rows
    for (si, sample_name) in all_samples.iter().enumerate() {
        let ry = table_y + (si as f64 + 1.0) * row_h;

        // Alternating row background
        if si % 2 == 0 {
            canvas.rect(
                table_x,
                ry - row_h * 0.3,
                label_col_w + n_cols as f64 * col_w,
                row_h,
                &Style::filled(Color::rgb(248, 248, 250)),
            );
        }

        canvas.text(table_x + 2.0, ry, sample_name, &label_cell_style);

        for (ci, ch) in channels.iter().enumerate() {
            let cx = table_x + label_col_w + ci as f64 * col_w + col_w - 4.0;
            if let Some(s) = ch.samples.iter().find(|s| s.name == *sample_name) {
                canvas.text(cx, ry, &format!("{:.1}", s.postfit), &cell_style);
            } else {
                canvas.text(cx, ry, "\u{2014}", &cell_style);
            }
        }
    }

    // Total row
    let total_row = all_samples.len() + 1;
    let ty = table_y + total_row as f64 * row_h;
    canvas.line(
        table_x,
        ty - row_h * 0.4,
        table_x + label_col_w + n_cols as f64 * col_w,
        ty - row_h * 0.4,
        &LineStyle::solid(Color::rgb(150, 150, 150), 0.5),
    );
    canvas.text(
        table_x + 2.0,
        ty,
        "Total MC",
        &TextStyle { weight: FontWeight::Bold, ..label_cell_style.clone() },
    );
    for (ci, ch) in channels.iter().enumerate() {
        let cx = table_x + label_col_w + ci as f64 * col_w + col_w - 4.0;
        canvas.text(
            cx,
            ty,
            &format!("{:.1}", ch.total_postfit),
            &TextStyle { weight: FontWeight::Bold, ..cell_style.clone() },
        );
    }

    // Data row
    let data_row = all_samples.len() + 2;
    let dy = table_y + data_row as f64 * row_h;
    canvas.text(
        table_x + 2.0,
        dy,
        "Data",
        &TextStyle { weight: FontWeight::Bold, ..label_cell_style.clone() },
    );
    for (ci, ch) in channels.iter().enumerate() {
        let cx = table_x + label_col_w + ci as f64 * col_w + col_w - 4.0;
        let blinded = ch.data_is_blinded.unwrap_or(false);
        if blinded {
            canvas.text(cx, dy, "BLIND", &cell_style);
        } else {
            canvas.text(
                cx,
                dy,
                &format!("{:.0}", ch.data),
                &TextStyle { weight: FontWeight::Bold, ..cell_style.clone() },
            );
        }
    }

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No yields data</text></svg>"#.into()
}
