use ns_viz::cls::ClsCurveArtifact;

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::margins::PlotArea;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &ClsCurveArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.mu_values.is_empty() {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height;
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let mu_min = artifact.mu_values.first().copied().unwrap_or(0.0);
    let mu_max = artifact.mu_values.last().copied().unwrap_or(5.0);
    let x_axis = Axis::auto_linear(mu_min, mu_max, 6).with_label("\u{03BC}");
    let y_axis = Axis::auto_log(1e-3, 1.0).with_label("CL\u{209B}");

    let area = PlotArea::auto(&canvas, Some(&y_axis), Some(&x_axis), config);
    draw_experiment_header(&mut canvas, &area, config);
    draw_axes(&mut canvas, &area, &x_axis, &y_axis, config);

    let _clip = canvas.push_clip(area.left, area.top, area.width, area.height);

    let mu = &artifact.mu_values;
    // Nsigma order: [+2σ, +1σ, median, -1σ, -2σ] → indices [0, 1, 2, 3, 4]

    // Helper: mu_values → pixel x, cls_vec → pixel y
    let to_px = |cls_vec: &[f64]| -> Vec<(f64, f64)> {
        mu.iter()
            .zip(cls_vec.iter())
            .map(|(&m, &c)| {
                let px = x_axis.data_to_pixel(m, area.left, area.right());
                let py = y_axis.data_to_pixel(c.max(1e-4), area.bottom(), area.top);
                (px, py)
            })
            .collect()
    };

    // ±2σ band (yellow)
    let band_2s_hi: Vec<f64> = artifact.cls_exp[0].clone(); // +2σ
    let band_2s_lo: Vec<f64> = artifact.cls_exp[4].clone(); // -2σ
    {
        let x_px: Vec<f64> =
            mu.iter().map(|&m| x_axis.data_to_pixel(m, area.left, area.right())).collect();
        let y_hi: Vec<f64> = band_2s_hi
            .iter()
            .map(|&c| y_axis.data_to_pixel(c.max(1e-4), area.bottom(), area.top))
            .collect();
        let y_lo: Vec<f64> = band_2s_lo
            .iter()
            .map(|&c| y_axis.data_to_pixel(c.max(1e-4), area.bottom(), area.top))
            .collect();
        canvas.fill_between(
            &x_px,
            &y_lo,
            &y_hi,
            &Style::filled(config.colors.band_2sigma.with_alpha(0.5)),
        );
    }

    // ±1σ band (green)
    let band_1s_hi: Vec<f64> = artifact.cls_exp[1].clone(); // +1σ
    let band_1s_lo: Vec<f64> = artifact.cls_exp[3].clone(); // -1σ
    {
        let x_px: Vec<f64> =
            mu.iter().map(|&m| x_axis.data_to_pixel(m, area.left, area.right())).collect();
        let y_hi: Vec<f64> = band_1s_hi
            .iter()
            .map(|&c| y_axis.data_to_pixel(c.max(1e-4), area.bottom(), area.top))
            .collect();
        let y_lo: Vec<f64> = band_1s_lo
            .iter()
            .map(|&c| y_axis.data_to_pixel(c.max(1e-4), area.bottom(), area.top))
            .collect();
        canvas.fill_between(
            &x_px,
            &y_lo,
            &y_hi,
            &Style::filled(config.colors.band_1sigma.with_alpha(0.5)),
        );
    }

    // Expected median (dashed)
    let exp_pts = to_px(&artifact.cls_exp[2]);
    canvas.polyline(&exp_pts, &LineStyle::dashed(config.colors.expected, 1.2));

    // Observed (solid)
    let obs_pts = to_px(&artifact.cls_obs);
    canvas.polyline(&obs_pts, &LineStyle::solid(config.colors.observed, 1.5));

    // Alpha line (horizontal at CL = alpha)
    let alpha_py = y_axis.data_to_pixel(artifact.alpha, area.bottom(), area.top);
    canvas.line(
        area.left,
        alpha_py,
        area.right(),
        alpha_py,
        &LineStyle { color: Color::rgb(200, 60, 60), width: 0.8, dash: Some("4 2".into()) },
    );
    let alpha_label_style = TextStyle {
        size: config.font.tick_size * 0.8,
        color: Color::rgb(200, 60, 60),
        anchor: TextAnchor::End,
        ..Default::default()
    };
    canvas.text(
        area.right() - 4.0,
        alpha_py - 4.0,
        &format!("\u{03B1} = {}", artifact.alpha),
        &alpha_label_style,
    );

    canvas.pop_clip();

    // Legend
    legend::draw_legend(
        &mut canvas,
        &area,
        &[
            LegendEntry {
                label: "Observed".into(),
                color: config.colors.observed,
                kind: LegendKind::Line(None),
            },
            LegendEntry {
                label: "Expected".into(),
                color: config.colors.expected,
                kind: LegendKind::Line(Some("6 3".into())),
            },
            LegendEntry {
                label: "\u{00B1}1\u{03C3}".into(),
                color: config.colors.band_1sigma,
                kind: LegendKind::FilledRect,
            },
            LegendEntry {
                label: "\u{00B1}2\u{03C3}".into(),
                color: config.colors.band_2sigma,
                kind: LegendKind::FilledRect,
            },
        ],
        config.font.size,
        false,
    );

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No CLs data</text></svg>"#.into()
}
