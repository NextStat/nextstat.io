use ns_viz::separation::SeparationArtifact;

use crate::canvas::Canvas;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &SeparationArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.channels.is_empty() {
        return Ok(empty_svg());
    }

    let ch = &artifact.channels[0];
    let n = ch.signal_shape.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let x_min = ch.bin_edges.first().copied().unwrap_or(0.0);
    let x_max = ch.bin_edges.last().copied().unwrap_or(1.0);
    let y_max =
        ch.signal_shape.iter().chain(ch.background_shape.iter()).copied().fold(0.0_f64, f64::max)
            * 1.2;

    let x_axis = Axis::auto_linear(x_min, x_max, 6);
    let y_axis = Axis::auto_linear(0.0, y_max, 5).with_label("Normalised");

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    let sig_color = config.colors.expected;
    let bkg_color = config.colors.signal;

    // Signal filled histogram
    for bi in 0..n {
        let x_lo = ch.bin_edges[bi];
        let x_hi = ch.bin_edges[bi + 1];
        let px_lo = x_axis.data_to_pixel(x_lo, area.left, area.right());
        let px_hi = x_axis.data_to_pixel(x_hi, area.left, area.right());
        let py_top = y_axis.data_to_pixel(ch.signal_shape[bi], area.bottom(), area.top);
        let py_bot = y_axis.data_to_pixel(0.0, area.bottom(), area.top);

        canvas.rect(
            px_lo,
            py_top,
            px_hi - px_lo,
            py_bot - py_top,
            &Style {
                fill: Some(sig_color.with_alpha(0.3)),
                stroke: Some(sig_color),
                stroke_width: 1.0,
                opacity: 1.0,
            },
        );
    }

    // Background filled histogram
    for bi in 0..n {
        let x_lo = ch.bin_edges[bi];
        let x_hi = ch.bin_edges[bi + 1];
        let px_lo = x_axis.data_to_pixel(x_lo, area.left, area.right());
        let px_hi = x_axis.data_to_pixel(x_hi, area.left, area.right());
        let py_top = y_axis.data_to_pixel(ch.background_shape[bi], area.bottom(), area.top);
        let py_bot = y_axis.data_to_pixel(0.0, area.bottom(), area.top);

        canvas.rect(
            px_lo,
            py_top,
            px_hi - px_lo,
            py_bot - py_top,
            &Style {
                fill: Some(bkg_color.with_alpha(0.3)),
                stroke: Some(bkg_color),
                stroke_width: 1.0,
                opacity: 1.0,
            },
        );
    }

    canvas.pop_clip();

    // Separation label
    let sep_style =
        TextStyle { size: config.font.size, anchor: TextAnchor::Start, ..Default::default() };
    canvas.text(
        area.left + 5.0,
        area.top + 14.0,
        &format!("{}: sep = {:.4}", ch.channel_name, ch.separation),
        &sep_style,
    );

    legend::draw_legend(
        &mut canvas,
        &area,
        &[
            LegendEntry { label: "Signal".into(), color: sig_color, kind: LegendKind::FilledRect },
            LegendEntry {
                label: "Background".into(),
                color: bkg_color,
                kind: LegendKind::FilledRect,
            },
        ],
        config.font.size,
        false,
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No separation data</text></svg>"#.into()
}
