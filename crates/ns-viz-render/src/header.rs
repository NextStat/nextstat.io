use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::layout::margins::PlotArea;
use crate::primitives::*;

/// Draw the experiment header label (e.g., **ATLAS** *Internal*, √s = 13.6 TeV, 140 fb⁻¹).
pub fn draw_experiment_header(canvas: &mut Canvas, area: &PlotArea, config: &VizConfig) {
    if config.experiment.name.is_empty() {
        return;
    }

    let header_size = config.font.label_size * 1.3;
    let x = area.left + area.width * 0.02;
    let y = area.top - 6.0;

    // Experiment name (bold)
    let bold_style = TextStyle {
        size: header_size,
        color: Color::rgb(0, 0, 0),
        weight: FontWeight::Bold,
        anchor: TextAnchor::Start,
        baseline: TextBaseline::Alphabetic,
        ..Default::default()
    };
    canvas.text(x, y, &config.experiment.name, &bold_style);

    // Measure name width to position status
    let name_w = canvas.measure_text(&config.experiment.name, &bold_style).width;

    // Status (italic) — right after name with small gap
    if !config.experiment.status.is_empty() {
        let italic_style = TextStyle {
            size: header_size * 0.85,
            color: Color::rgb(0, 0, 0),
            style: FontStyle::Italic,
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Alphabetic,
            ..Default::default()
        };
        canvas.text(x + name_w + 5.0, y, &config.experiment.status, &italic_style);
    }

    // Energy + luminosity line (right-aligned, top of plot)
    let mut info_parts = Vec::new();
    if config.experiment.sqrt_s_tev > 0.0 {
        info_parts.push(format!("\u{221A}s = {} TeV", config.experiment.sqrt_s_tev));
    }
    if config.experiment.lumi_fb_inv > 0.0 {
        info_parts.push(format!("{} fb\u{207B}\u{00B9}", config.experiment.lumi_fb_inv));
    }

    if !info_parts.is_empty() {
        let info = info_parts.join(", ");
        let info_style = TextStyle {
            size: config.font.tick_size,
            color: Color::rgb(80, 80, 80),
            anchor: TextAnchor::End,
            baseline: TextBaseline::Alphabetic,
            ..Default::default()
        };
        canvas.text(area.right(), y, &info, &info_style);
    }
}
