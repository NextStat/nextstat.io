use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::layout::axes::Axis;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

/// Draw a standard box frame with axes, ticks, grid, and labels.
pub fn draw_axes(
    canvas: &mut Canvas,
    area: &PlotArea,
    x_axis: &Axis,
    y_axis: &Axis,
    config: &VizConfig,
) {
    let frame_color = Color::rgb(0, 0, 0);
    let frame_style = LineStyle::solid(frame_color, 0.8);
    let tick_style_line = LineStyle::solid(frame_color, 0.6);
    let minor_tick_style = LineStyle::solid(frame_color, 0.4);

    let inward = config.axes.tick_direction == "in";
    let tl = config.axes.tick_length;
    let mtl = config.axes.minor_tick_length;

    // Frame rectangle
    canvas.line(area.left, area.top, area.right(), area.top, &frame_style);
    canvas.line(area.left, area.bottom(), area.right(), area.bottom(), &frame_style);
    canvas.line(area.left, area.top, area.left, area.bottom(), &frame_style);
    canvas.line(area.right(), area.top, area.right(), area.bottom(), &frame_style);

    let tick_label_style = TextStyle {
        size: config.font.tick_size,
        color: frame_color,
        anchor: TextAnchor::Middle,
        baseline: TextBaseline::Hanging,
        ..Default::default()
    };

    // --- X axis ticks ---
    for (i, &val) in x_axis.tick_positions.iter().enumerate() {
        let px = x_axis.data_to_pixel(val, area.left, area.right());
        if px < area.left - 0.5 || px > area.right() + 0.5 {
            continue;
        }

        // Grid
        if config.grid.show {
            let grid_style = LineStyle {
                color: config.grid.color.with_alpha(config.grid.alpha),
                width: 0.5,
                dash: Some("3 3".into()),
            };
            canvas.line(px, area.top, px, area.bottom(), &grid_style);
        }

        // Bottom tick
        if inward {
            canvas.line(px, area.bottom(), px, area.bottom() - tl, &tick_style_line);
        } else {
            canvas.line(px, area.bottom(), px, area.bottom() + tl, &tick_style_line);
        }
        // Top tick
        if config.axes.show_top_ticks {
            if inward {
                canvas.line(px, area.top, px, area.top + tl, &tick_style_line);
            } else {
                canvas.line(px, area.top, px, area.top - tl, &tick_style_line);
            }
        }

        // Tick label
        if let Some(label) = x_axis.tick_labels.get(i) {
            let label_y = if inward { area.bottom() + 3.0 } else { area.bottom() + tl + 3.0 };
            canvas.text(px, label_y, label, &tick_label_style);
        }
    }

    // X minor ticks
    for &val in &x_axis.minor_ticks {
        let px = x_axis.data_to_pixel(val, area.left, area.right());
        if px < area.left - 0.5 || px > area.right() + 0.5 {
            continue;
        }
        if inward {
            canvas.line(px, area.bottom(), px, area.bottom() - mtl, &minor_tick_style);
        } else {
            canvas.line(px, area.bottom(), px, area.bottom() + mtl, &minor_tick_style);
        }
    }

    // --- Y axis ticks ---
    let y_tick_label_style = TextStyle {
        size: config.font.tick_size,
        color: frame_color,
        anchor: TextAnchor::End,
        baseline: TextBaseline::Central,
        ..Default::default()
    };

    for (i, &val) in y_axis.tick_positions.iter().enumerate() {
        let py = y_axis.data_to_pixel(val, area.bottom(), area.top);
        if py < area.top - 0.5 || py > area.bottom() + 0.5 {
            continue;
        }

        // Grid
        if config.grid.show {
            let grid_style = LineStyle {
                color: config.grid.color.with_alpha(config.grid.alpha),
                width: 0.5,
                dash: Some("3 3".into()),
            };
            canvas.line(area.left, py, area.right(), py, &grid_style);
        }

        // Left tick
        if inward {
            canvas.line(area.left, py, area.left + tl, py, &tick_style_line);
        } else {
            canvas.line(area.left, py, area.left - tl, py, &tick_style_line);
        }
        // Right tick
        if config.axes.show_right_ticks {
            if inward {
                canvas.line(area.right(), py, area.right() - tl, py, &tick_style_line);
            } else {
                canvas.line(area.right(), py, area.right() + tl, py, &tick_style_line);
            }
        }

        // Tick label
        if let Some(label) = y_axis.tick_labels.get(i) {
            let label_x = if inward { area.left - 4.0 } else { area.left - tl - 4.0 };
            canvas.text(label_x, py, label, &y_tick_label_style);
        }
    }

    // Y minor ticks
    for &val in &y_axis.minor_ticks {
        let py = y_axis.data_to_pixel(val, area.bottom(), area.top);
        if py < area.top - 0.5 || py > area.bottom() + 0.5 {
            continue;
        }
        if inward {
            canvas.line(area.left, py, area.left + mtl, py, &minor_tick_style);
        } else {
            canvas.line(area.left, py, area.left - mtl, py, &minor_tick_style);
        }
    }

    // --- Axis labels ---
    let label_style = TextStyle {
        size: config.font.label_size,
        color: frame_color,
        anchor: TextAnchor::Middle,
        ..Default::default()
    };

    if !x_axis.label.is_empty() {
        let label_y = if inward {
            area.bottom() + config.font.tick_size + 14.0
        } else {
            area.bottom() + tl + config.font.tick_size + 14.0
        };
        canvas.text(area.left + area.width / 2.0, label_y, &x_axis.label, &label_style);
    }

    if !y_axis.label.is_empty() {
        let label_x = area.left - 40.0;
        let label_y = area.top + area.height / 2.0;
        canvas.text_rotated(label_x, label_y, &y_axis.label, &label_style, -90.0);
    }
}

/// Draw axes frame only (no ticks) — for pull/ranking panels with custom tick rendering.
pub fn draw_frame(canvas: &mut Canvas, area: &PlotArea) {
    let style = LineStyle::solid(Color::rgb(0, 0, 0), 0.8);
    canvas.line(area.left, area.top, area.right(), area.top, &style);
    canvas.line(area.left, area.bottom(), area.right(), area.bottom(), &style);
    canvas.line(area.left, area.top, area.left, area.bottom(), &style);
    canvas.line(area.right(), area.top, area.right(), area.bottom(), &style);
}
