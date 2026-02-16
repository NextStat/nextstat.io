use ns_viz::ranking::RankingArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::multi_panel::DualPanelLayout;
use crate::plots::axes_draw::draw_frame;
use crate::primitives::*;

pub fn render(artifact: &RankingArtifact, config: &VizConfig) -> crate::Result<String> {
    let n = artifact.names.len();
    if n == 0 {
        return Ok(empty_svg());
    }

    let top_n = config.ranking.top_n.unwrap_or(n).min(n);
    let row_h = 20.0;
    let fig_w = config.figure.width;
    let fig_h = (row_h * top_n as f64 + 110.0).max(280.0);

    let mut canvas = Canvas::new(fig_w, fig_h)?;

    // Label width
    let label_style = TextStyle { size: config.font.tick_size, ..Default::default() };
    let label_w = artifact.names[..top_n]
        .iter()
        .map(|n| canvas.measure_text(n, &label_style).width)
        .fold(0.0_f64, f64::max)
        + 10.0;

    let content_left = label_w + 15.0;
    let content_w = fig_w - content_left - 15.0;
    let content_top = 35.0;
    let content_h = row_h * top_n as f64;

    let layout = DualPanelLayout::new(content_left, content_top, content_w, content_h, 8.0, 0.65);

    draw_experiment_header(&mut canvas, &layout.left, config);

    // --- Impact panel (left) ---
    let impact_area = &layout.left;

    // Find max impact for scaling
    let max_impact = artifact.delta_mu_up[..top_n]
        .iter()
        .chain(artifact.delta_mu_down[..top_n].iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(0.01);

    // Symmetric x range
    let impact_x_center = impact_area.left + impact_area.width / 2.0;
    let impact_px_per_unit = impact_area.width / (2.0 * max_impact * 1.15);

    for i in 0..top_n {
        let y = impact_area.top + (i as f64 + 0.5) * row_h;

        // Parameter label (to the left of the panel)
        let name_style = TextStyle {
            size: config.font.tick_size,
            anchor: TextAnchor::End,
            baseline: TextBaseline::Central,
            ..Default::default()
        };
        canvas.text(impact_area.left - 5.0, y, &artifact.names[i], &name_style);

        // Impact up bar (blue-ish)
        let up = artifact.delta_mu_up[i];
        let bar_w = (up * impact_px_per_unit).abs();
        let bar_x = if up >= 0.0 { impact_x_center } else { impact_x_center - bar_w };
        canvas.rect(
            bar_x,
            y - row_h * 0.22,
            bar_w,
            row_h * 0.22,
            &Style::filled(config.colors.expected.with_alpha(0.7)),
        );

        // Impact down bar (red-ish)
        let down = artifact.delta_mu_down[i];
        let bar_w_d = (down * impact_px_per_unit).abs();
        let bar_x_d = if down >= 0.0 { impact_x_center } else { impact_x_center - bar_w_d };
        canvas.rect(
            bar_x_d,
            y,
            bar_w_d,
            row_h * 0.22,
            &Style::filled(config.colors.signal.with_alpha(0.7)),
        );
    }

    // Center line
    canvas.line(
        impact_x_center,
        impact_area.top,
        impact_x_center,
        impact_area.bottom(),
        &LineStyle::dashed(Color::rgb(100, 100, 100), 0.5),
    );

    draw_frame(&mut canvas, impact_area);

    // Impact axis label
    let impact_label_style =
        TextStyle { size: config.font.tick_size, anchor: TextAnchor::Middle, ..Default::default() };
    canvas.text(
        impact_x_center,
        impact_area.bottom() + 14.0,
        &format!("\u{0394}{}", config.ranking.poi_label),
        &impact_label_style,
    );

    // --- Pull panel (right) ---
    let pull_area = &layout.right;
    let pull_x_center = pull_area.left + pull_area.width / 2.0;
    let pull_px_per_sigma = pull_area.width / 4.0; // ±2σ range

    // ±1σ band
    canvas.rect(
        pull_x_center - pull_px_per_sigma,
        pull_area.top,
        2.0 * pull_px_per_sigma,
        pull_area.height,
        &Style { fill: Some(config.colors.band_1sigma.with_alpha(0.3)), ..Default::default() },
    );

    canvas.line(
        pull_x_center,
        pull_area.top,
        pull_x_center,
        pull_area.bottom(),
        &LineStyle::dashed(Color::rgb(100, 100, 100), 0.5),
    );

    for i in 0..top_n {
        let y = pull_area.top + (i as f64 + 0.5) * row_h;

        let pull_px = pull_x_center + artifact.pull[i] * pull_px_per_sigma;
        let err_lo =
            pull_x_center + (artifact.pull[i] - artifact.constraint[i]) * pull_px_per_sigma;
        let err_hi =
            pull_x_center + (artifact.pull[i] + artifact.constraint[i]) * pull_px_per_sigma;

        canvas.error_bar(pull_px, err_lo, err_hi, 3.0, &LineStyle::solid(Color::rgb(0, 0, 0), 1.0));
        canvas.marker(
            pull_px,
            y,
            &MarkerStyle {
                color: Color::rgb(0, 0, 0),
                size: 2.5,
                fill: true,
                ..Default::default()
            },
        );
    }

    draw_frame(&mut canvas, pull_area);

    // Pull axis label
    canvas.text(
        pull_x_center,
        pull_area.bottom() + 14.0,
        "(\u{03B8}\u{0302} \u{2212} \u{03B8}\u{2080}) / \u{0394}\u{03B8}",
        &impact_label_style,
    );

    // Legend
    let legend_y = fig_h - 20.0;
    let legend_style = TextStyle {
        size: config.font.tick_size * 0.9,
        baseline: TextBaseline::Central,
        ..Default::default()
    };
    let lx = impact_area.left;
    canvas.rect(
        lx,
        legend_y - 4.0,
        10.0,
        8.0,
        &Style::filled(config.colors.expected.with_alpha(0.7)),
    );
    canvas.text(lx + 14.0, legend_y, "+1\u{03C3} post-fit impact", &legend_style);
    let lx2 = lx + 130.0;
    canvas.rect(
        lx2,
        legend_y - 4.0,
        10.0,
        8.0,
        &Style::filled(config.colors.signal.with_alpha(0.7)),
    );
    canvas.text(lx2 + 14.0, legend_y, "\u{2212}1\u{03C3} post-fit impact", &legend_style);

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No ranking entries</text></svg>"#.into()
}
