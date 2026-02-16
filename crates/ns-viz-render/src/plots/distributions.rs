use ns_viz::distributions::{DistributionsArtifact, DistributionsChannelArtifact};

use crate::canvas::Canvas;
use crate::color::Color;
use crate::config::VizConfig;
use crate::header::draw_experiment_header;
use crate::layout::axes::Axis;
use crate::layout::legend::{self, LegendEntry, LegendKind};
use crate::layout::multi_panel::MainRatioLayout;
use crate::plots::axes_draw::draw_axes;
use crate::primitives::*;

pub fn render(artifact: &DistributionsArtifact, config: &VizConfig) -> crate::Result<String> {
    if artifact.channels.is_empty() {
        return Ok(empty_svg());
    }
    // Render first channel (multi-channel: one SVG per channel)
    render_channel(&artifact.channels[0], config)
}

fn render_channel(ch: &DistributionsChannelArtifact, config: &VizConfig) -> crate::Result<String> {
    let n_bins = ch.data_y.len();
    if n_bins == 0 {
        return Ok(empty_svg());
    }

    let fig_w = config.figure.width;
    let fig_h = config.figure.height * 1.3; // taller for main+ratio
    let mut canvas = Canvas::new(fig_w, fig_h)?;

    let palette = config.palette_colors();
    let blinded = ch.data_is_blinded.unwrap_or(false);

    // Determine bin edges for x-axis
    let bin_edges = &ch.bin_edges;
    let x_min = bin_edges.first().copied().unwrap_or(0.0);
    let x_max = bin_edges.last().copied().unwrap_or(1.0);

    // Y range from stacked histograms
    let y_max =
        ch.data_y.iter().chain(ch.total_postfit_y.iter()).copied().fold(0.0_f64, f64::max) * 1.3;

    let x_axis_main = Axis::auto_linear(x_min, x_max, 6);
    let y_axis_main = Axis::auto_linear(0.0, y_max, 5).with_label("Events");

    // Ratio y axis
    let ratio_range = config.distributions.ratio_y_range;
    let x_axis_ratio = Axis::auto_linear(x_min, x_max, 6);
    let y_axis_ratio = Axis::auto_linear(ratio_range[0], ratio_range[1], 3).with_label("Data / MC");

    // Layout: compute margins then split into main + ratio
    let left_margin = {
        let style = TextStyle { size: config.font.tick_size, ..Default::default() };
        y_axis_main
            .tick_labels
            .iter()
            .map(|l| canvas.measure_text(l, &style).width)
            .fold(0.0_f64, f64::max)
            + config.font.label_size
            + 22.0
    };
    let right_margin = 15.0;
    let top_margin =
        if config.experiment.name.is_empty() { 12.0 } else { config.font.label_size * 1.3 + 20.0 };
    let bottom_margin = config.font.tick_size + config.font.label_size + 20.0;
    let content_w = fig_w - left_margin - right_margin;
    let content_h = fig_h - top_margin - bottom_margin;

    let layout = MainRatioLayout::new(left_margin, top_margin, content_w, content_h, 4.0, 0.25);

    draw_experiment_header(&mut canvas, &layout.main, config);

    // --- Main panel ---
    let main = &layout.main;
    draw_axes(&mut canvas, main, &x_axis_main, &y_axis_main, config);
    let _clip = canvas.push_clip(main.left, main.top, main.width, main.height);

    // Stacked histogram bars
    let _n_samples = ch.samples.len();
    let mut cumulative = vec![0.0_f64; n_bins];

    for (si, sample) in ch.samples.iter().enumerate() {
        let color = if si < palette.len() { palette[si] } else { Color::hex("#888888") };

        for bi in 0..n_bins {
            let x_lo = bin_edges[bi];
            let x_hi = bin_edges[bi + 1];
            let y_base = cumulative[bi];
            let y_top = y_base + sample.postfit_y[bi];

            let px_lo = x_axis_main.data_to_pixel(x_lo, main.left, main.right());
            let px_hi = x_axis_main.data_to_pixel(x_hi, main.left, main.right());
            let py_base = y_axis_main.data_to_pixel(y_base, main.bottom(), main.top);
            let py_top = y_axis_main.data_to_pixel(y_top, main.bottom(), main.top);

            canvas.rect(px_lo, py_top, px_hi - px_lo, py_base - py_top, &Style::filled(color));
        }

        for (bi, cum) in cumulative.iter_mut().enumerate().take(n_bins) {
            *cum += sample.postfit_y[bi];
        }
    }

    // MC band (hatched)
    if config.distributions.show_mc_band
        && let Some(band) = &ch.mc_band_postfit
    {
        let pid = "mc_band_hatch";
        for bi in 0..n_bins {
            let x_lo = bin_edges[bi];
            let x_hi = bin_edges[bi + 1];
            let px_lo = x_axis_main.data_to_pixel(x_lo, main.left, main.right());
            let px_hi = x_axis_main.data_to_pixel(x_hi, main.left, main.right());
            let py_lo = y_axis_main.data_to_pixel(band.lo[bi], main.bottom(), main.top);
            let py_hi = y_axis_main.data_to_pixel(band.hi[bi], main.bottom(), main.top);

            canvas.rect(
                px_lo,
                py_hi,
                px_hi - px_lo,
                py_lo - py_hi,
                &Style { fill: Some(Color::rgb(0, 0, 0).with_alpha(0.08)), ..Default::default() },
            );
            if bi == 0 {
                canvas.hatch_rect(
                    px_lo,
                    py_hi,
                    px_hi - px_lo,
                    py_lo - py_hi,
                    pid,
                    Color::rgb(100, 100, 100),
                    4.0,
                );
            } else {
                // Reuse pattern
                let w = px_hi - px_lo;
                let h = py_lo - py_hi;
                let _elem = format!(
                    r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="url(#{pid})" />"#,
                    px_lo, py_hi, w, h,
                );
                // Raw SVG hack — reuse pattern defined in first bin
                canvas.rect(
                    px_lo,
                    py_hi,
                    w,
                    h,
                    &Style { fill: None, stroke: None, stroke_width: 0.0, opacity: 1.0 },
                );
            }
        }
    }

    // Data points (if not blinded)
    if !blinded {
        let marker = MarkerStyle {
            color: config.colors.observed,
            size: 2.5,
            fill: true,
            ..Default::default()
        };
        let err_style = LineStyle::solid(config.colors.observed, 1.0);

        for bi in 0..n_bins {
            let x_center = (bin_edges[bi] + bin_edges[bi + 1]) / 2.0;
            let px = x_axis_main.data_to_pixel(x_center, main.left, main.right());
            let py = y_axis_main.data_to_pixel(ch.data_y[bi], main.bottom(), main.top);
            let py_lo = y_axis_main.data_to_pixel(
                ch.data_y[bi] - ch.data_yerr_lo[bi],
                main.bottom(),
                main.top,
            );
            let py_hi = y_axis_main.data_to_pixel(
                ch.data_y[bi] + ch.data_yerr_hi[bi],
                main.bottom(),
                main.top,
            );

            canvas.error_bar(px, py_lo, py_hi, 0.0, &err_style);
            canvas.marker(px, py, &marker);
        }
    } else {
        // BLINDED label
        let blind_style = TextStyle {
            size: config.font.label_size * 2.0,
            color: Color::rgb(200, 200, 200),
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Central,
            weight: FontWeight::Bold,
            ..Default::default()
        };
        canvas.text(
            main.left + main.width / 2.0,
            main.top + main.height / 2.0,
            "BLINDED",
            &blind_style,
        );
    }

    canvas.pop_clip();

    // --- Ratio panel ---
    let ratio = &layout.ratio;
    draw_axes(&mut canvas, ratio, &x_axis_ratio, &y_axis_ratio, config);
    let _clip2 = canvas.push_clip(ratio.left, ratio.top, ratio.width, ratio.height);

    // Reference line at y=1
    let ref_py = y_axis_ratio.data_to_pixel(1.0, ratio.bottom(), ratio.top);
    canvas.line(
        ratio.left,
        ref_py,
        ratio.right(),
        ref_py,
        &LineStyle::dashed(Color::rgb(150, 150, 150), 0.6),
    );

    // Ratio band
    if let Some(band) = &ch.ratio_band {
        for bi in 0..n_bins {
            let x_lo = bin_edges[bi];
            let x_hi = bin_edges[bi + 1];
            let px_lo = x_axis_ratio.data_to_pixel(x_lo, ratio.left, ratio.right());
            let px_hi = x_axis_ratio.data_to_pixel(x_hi, ratio.left, ratio.right());
            let py_lo = y_axis_ratio.data_to_pixel(band.lo[bi], ratio.bottom(), ratio.top);
            let py_hi = y_axis_ratio.data_to_pixel(band.hi[bi], ratio.bottom(), ratio.top);

            canvas.rect(
                px_lo,
                py_hi,
                px_hi - px_lo,
                py_lo - py_hi,
                &Style { fill: Some(Color::rgb(0, 0, 0).with_alpha(0.1)), ..Default::default() },
            );
        }
    }

    // Ratio data points
    if !blinded {
        let marker = MarkerStyle {
            color: config.colors.observed,
            size: 2.0,
            fill: true,
            ..Default::default()
        };
        let err_style = LineStyle::solid(config.colors.observed, 0.8);

        for bi in 0..n_bins {
            let x_center = (bin_edges[bi] + bin_edges[bi + 1]) / 2.0;
            let px = x_axis_ratio.data_to_pixel(x_center, ratio.left, ratio.right());
            let py = y_axis_ratio.data_to_pixel(ch.ratio_y[bi], ratio.bottom(), ratio.top);
            let py_lo = y_axis_ratio.data_to_pixel(
                ch.ratio_y[bi] - ch.ratio_yerr_lo[bi],
                ratio.bottom(),
                ratio.top,
            );
            let py_hi = y_axis_ratio.data_to_pixel(
                ch.ratio_y[bi] + ch.ratio_yerr_hi[bi],
                ratio.bottom(),
                ratio.top,
            );

            canvas.error_bar(px, py_lo, py_hi, 0.0, &err_style);
            canvas.marker(px, py, &marker);
        }
    }

    canvas.pop_clip();

    // Channel name
    let ch_style = TextStyle {
        size: config.font.size,
        anchor: TextAnchor::Start,
        weight: FontWeight::Bold,
        ..Default::default()
    };
    canvas.text(main.left + 5.0, main.top + 14.0, &ch.channel_name, &ch_style);

    // Legend
    let entries: Vec<LegendEntry> = ch
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| LegendEntry {
            label: s.name.clone(),
            color: if i < palette.len() { palette[i] } else { Color::hex("#888888") },
            kind: LegendKind::FilledRect,
        })
        .chain(std::iter::once(LegendEntry {
            label: "Data".into(),
            color: config.colors.observed,
            kind: LegendKind::Marker,
        }))
        .collect();
    legend::draw_legend(&mut canvas, main, &entries, config.font.size, false);

    Ok(canvas.finish_svg())
}

fn empty_svg() -> String {
    r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><text x="10" y="30">No distribution data</text></svg>"#.into()
}
