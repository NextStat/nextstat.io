use crate::canvas::Canvas;
use crate::color::Color;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

pub struct LegendEntry {
    pub label: String,
    pub color: Color,
    pub kind: LegendKind,
}

pub enum LegendKind {
    FilledRect,
    Line(Option<String>), // dash pattern
    Marker,
    HatchedRect,
}

/// Draw a legend in the plot area.
pub fn draw_legend(
    canvas: &mut Canvas,
    area: &PlotArea,
    entries: &[LegendEntry],
    config_font_size: f64,
    frame: bool,
) {
    if entries.is_empty() {
        return;
    }

    let row_height = config_font_size + 4.0;
    let swatch_w = 14.0;
    let swatch_h = config_font_size - 2.0;
    let gap = 6.0;
    let padding = 6.0;

    let text_style = TextStyle {
        size: config_font_size * 0.85,
        baseline: TextBaseline::Central,
        ..Default::default()
    };

    // Measure max label width
    let max_w = entries
        .iter()
        .map(|e| canvas.measure_text(&e.label, &text_style).width)
        .fold(0.0_f64, f64::max);

    let legend_w = padding + swatch_w + gap + max_w + padding;
    let legend_h = padding + entries.len() as f64 * row_height + padding;

    // Position: top-right of plot area
    let lx = area.right() - legend_w - 5.0;
    let ly = area.top + 5.0;

    // Background
    let bg_style = Style {
        fill: Some(Color::rgba(255, 255, 255, 0.9)),
        stroke: if frame { Some(Color::rgb(200, 200, 200)) } else { None },
        stroke_width: 0.5,
        opacity: 1.0,
    };
    canvas.rect(lx, ly, legend_w, legend_h, &bg_style);

    for (i, entry) in entries.iter().enumerate() {
        let ey = ly + padding + i as f64 * row_height + row_height / 2.0;
        let sx = lx + padding;

        match entry.kind {
            LegendKind::FilledRect => {
                canvas.rect(
                    sx,
                    ey - swatch_h / 2.0,
                    swatch_w,
                    swatch_h,
                    &Style::filled(entry.color),
                );
            }
            LegendKind::Line(ref dash) => {
                let ls = LineStyle { color: entry.color, width: 1.5, dash: dash.clone() };
                canvas.line(sx, ey, sx + swatch_w, ey, &ls);
            }
            LegendKind::Marker => {
                canvas.marker(
                    sx + swatch_w / 2.0,
                    ey,
                    &MarkerStyle { color: entry.color, size: 3.0, ..Default::default() },
                );
            }
            LegendKind::HatchedRect => {
                let pid = format!("legend_hatch_{i}");
                canvas.hatch_rect(
                    sx,
                    ey - swatch_h / 2.0,
                    swatch_w,
                    swatch_h,
                    &pid,
                    entry.color,
                    4.0,
                );
            }
        }

        canvas.text(sx + swatch_w + gap, ey, &entry.label, &text_style);
    }
}
