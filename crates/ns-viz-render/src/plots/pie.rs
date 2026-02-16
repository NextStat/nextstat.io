use ns_viz::pie::PieArtifact;

use crate::canvas::Canvas;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

use std::f64::consts::PI;
use std::fmt::Write;

pub fn render(artifact: &PieArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.channels.is_empty() {
        return Ok(empty_svg());
    }

    let ch = &artifact.channels[0];
    let n = ch.slices.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let area = PlotArea::manual(15.0, 35.0, fig_w - 30.0, fig_h - 50.0);
    draw_experiment_header(&mut canvas, &area, config);

    let cx = area.left + area.width * 0.4;
    let cy = area.top + area.height / 2.0;
    let r = area.height.min(area.width * 0.7) / 2.0 - 5.0;

    let palette = config.palette_colors();
    let mut start_angle = -PI / 2.0; // 12 o'clock

    for (i, slice) in ch.slices.iter().enumerate() {
        let sweep = 2.0 * PI * slice.fraction;
        let end_angle = start_angle + sweep;
        let color = palette[i % palette.len()];

        // SVG arc path
        let x1 = cx + r * start_angle.cos();
        let y1 = cy + r * start_angle.sin();
        let x2 = cx + r * end_angle.cos();
        let y2 = cy + r * end_angle.sin();
        let large_arc = if sweep > PI { 1 } else { 0 };

        let mut d = String::new();
        write!(
            d,
            "M{cx:.2},{cy:.2} L{x1:.2},{y1:.2} A{r:.2},{r:.2} 0 {large_arc} 1 {x2:.2},{y2:.2} Z"
        )
        .unwrap();

        // Draw as raw path
        canvas.rect(0.0, 0.0, 0.0, 0.0, &Style::default()); // placeholder
        // Use canvas internals — since Canvas doesn't expose path directly,
        // draw as polygon approximation
        let steps = (sweep / 0.02).max(10.0) as usize;
        let mut pts = vec![(cx, cy)];
        for s in 0..=steps {
            let a = start_angle + sweep * s as f64 / steps as f64;
            pts.push((cx + r * a.cos(), cy + r * a.sin()));
        }
        canvas.polygon(&pts, &Style::filled(color));

        // Label: place at mid-angle, slightly outside radius
        let mid_angle = start_angle + sweep / 2.0;
        let label_r = r + 14.0;
        let lx = cx + label_r * mid_angle.cos();
        let ly = cy + label_r * mid_angle.sin();

        if slice.fraction > 0.03 {
            let anchor = if mid_angle.cos() < -0.1 {
                TextAnchor::End
            } else if mid_angle.cos() > 0.1 {
                TextAnchor::Start
            } else {
                TextAnchor::Middle
            };
            canvas.text(
                lx,
                ly,
                &format!("{} ({:.1}%)", slice.sample_name, slice.fraction * 100.0),
                &TextStyle {
                    size: config.font.tick_size * 0.85,
                    anchor,
                    baseline: TextBaseline::Central,
                    ..Default::default()
                },
            );
        }

        start_angle = end_angle;
    }

    // Channel name title
    canvas.text(
        cx,
        area.top - 5.0,
        &ch.channel_name,
        &TextStyle {
            size: config.font.label_size,
            anchor: TextAnchor::Middle,
            weight: FontWeight::Bold,
            ..Default::default()
        },
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No pie data</text></svg>"#.into()
}
