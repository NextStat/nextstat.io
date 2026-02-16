use std::fmt::Write as FmtWrite;

use crate::color::Color;
use crate::font::{FontHandle, svg_font_style};
use crate::primitives::*;
use crate::text::{TextMetrics, measure_text};

/// An SVG element stored for deferred rendering.
#[derive(Debug, Clone)]
enum SvgElement {
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        style: Style,
    },
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        style: LineStyle,
    },
    Polyline {
        points: Vec<(f64, f64)>,
        style: LineStyle,
        close: bool,
    },
    Polygon {
        points: Vec<(f64, f64)>,
        style: Style,
    },
    Text {
        x: f64,
        y: f64,
        content: String,
        style: TextStyle,
        rotate: Option<f64>,
    },
    Path {
        d: String,
        style: Style,
    },
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        style: Style,
    },
    #[allow(dead_code)]
    Group {
        transform: Option<String>,
        clip_id: Option<String>,
        children: Vec<SvgElement>,
    },
    Raw(String),
}

/// Immediate-mode SVG canvas. Coordinates in points (1pt = 1/72").
pub struct Canvas {
    pub width: f64,
    pub height: f64,
    elements: Vec<SvgElement>,
    defs: Vec<String>,
    clip_stack: Vec<String>,
    next_clip_id: usize,
    fonts: FontHandle,
}

impl Canvas {
    pub fn new(width: f64, height: f64) -> crate::Result<Self> {
        Ok(Self {
            width,
            height,
            elements: Vec::new(),
            defs: Vec::new(),
            clip_stack: Vec::new(),
            next_clip_id: 0,
            fonts: FontHandle::embedded()?,
        })
    }

    pub fn fonts(&self) -> &FontHandle {
        &self.fonts
    }

    // --- Drawing primitives ---

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, style: &Style) {
        self.push(SvgElement::Rect { x, y, w, h, style: style.clone() });
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &LineStyle) {
        self.push(SvgElement::Line { x1, y1, x2, y2, style: style.clone() });
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], style: &LineStyle) {
        self.push(SvgElement::Polyline {
            points: points.to_vec(),
            style: style.clone(),
            close: false,
        });
    }

    pub fn polygon(&mut self, points: &[(f64, f64)], style: &Style) {
        self.push(SvgElement::Polygon { points: points.to_vec(), style: style.clone() });
    }

    pub fn text(&mut self, x: f64, y: f64, content: &str, style: &TextStyle) {
        self.push(SvgElement::Text {
            x,
            y,
            content: content.to_string(),
            style: style.clone(),
            rotate: None,
        });
    }

    pub fn text_rotated(&mut self, x: f64, y: f64, content: &str, style: &TextStyle, angle: f64) {
        self.push(SvgElement::Text {
            x,
            y,
            content: content.to_string(),
            style: style.clone(),
            rotate: Some(angle),
        });
    }

    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, style: &Style) {
        self.push(SvgElement::Circle { cx, cy, r, style: style.clone() });
    }

    /// Fill between y_lo and y_hi at given x positions (for bands).
    pub fn fill_between(&mut self, x: &[f64], y_lo: &[f64], y_hi: &[f64], style: &Style) {
        if x.len() < 2 {
            return;
        }
        let mut d = String::new();
        // Forward along y_hi
        write!(d, "M{:.2},{:.2}", x[0], y_hi[0]).unwrap();
        for i in 1..x.len() {
            write!(d, " L{:.2},{:.2}", x[i], y_hi[i]).unwrap();
        }
        // Backward along y_lo
        for i in (0..x.len()).rev() {
            write!(d, " L{:.2},{:.2}", x[i], y_lo[i]).unwrap();
        }
        d.push('Z');
        self.push(SvgElement::Path { d, style: style.clone() });
    }

    /// Error bar: vertical line + optional horizontal caps.
    pub fn error_bar(&mut self, x: f64, y_lo: f64, y_hi: f64, cap_width: f64, style: &LineStyle) {
        self.line(x, y_lo, x, y_hi, style);
        if cap_width > 0.0 {
            let half = cap_width / 2.0;
            self.line(x - half, y_lo, x + half, y_lo, style);
            self.line(x - half, y_hi, x + half, y_hi, style);
        }
    }

    /// Horizontal error bar.
    pub fn error_bar_h(
        &mut self,
        x_lo: f64,
        x_hi: f64,
        y: f64,
        cap_height: f64,
        style: &LineStyle,
    ) {
        self.line(x_lo, y, x_hi, y, style);
        if cap_height > 0.0 {
            let half = cap_height / 2.0;
            self.line(x_lo, y - half, x_lo, y + half, style);
            self.line(x_hi, y - half, x_hi, y + half, style);
        }
    }

    /// Data marker (filled circle or other shape).
    pub fn marker(&mut self, x: f64, y: f64, marker: &MarkerStyle) {
        match marker.shape {
            MarkerShape::Circle => {
                let style = if marker.fill {
                    Style {
                        fill: Some(marker.color),
                        stroke: Some(marker.color),
                        stroke_width: 0.5,
                        opacity: 1.0,
                    }
                } else {
                    Style {
                        fill: Some(Color::rgb(255, 255, 255)),
                        stroke: Some(marker.color),
                        stroke_width: 1.0,
                        opacity: 1.0,
                    }
                };
                self.circle(x, y, marker.size, &style);
            }
            _ => {
                // Square/Triangle/Diamond — simple filled circle fallback
                let style = Style::filled(marker.color);
                self.circle(x, y, marker.size, &style);
            }
        }
    }

    /// Add a hatched rectangle definition + instance.
    #[allow(clippy::too_many_arguments)]
    pub fn hatch_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        pattern_id: &str,
        color: Color,
        spacing: f64,
    ) {
        // Define hatch pattern if not already defined
        let def = format!(
            r#"<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{sp}" height="{sp}" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="{sp}" stroke="{c}" stroke-width="0.8"/></pattern>"#,
            pid = pattern_id,
            sp = spacing,
            c = color.to_svg_fill(),
        );
        self.defs.push(def);
        self.push(SvgElement::Rect {
            x,
            y,
            w,
            h,
            style: Style { fill: None, stroke: None, stroke_width: 0.0, opacity: 1.0 },
        });
        // Add raw rect with pattern fill
        self.push(SvgElement::Raw(format!(
            r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" fill="url(#{pattern_id})" />"#
        )));
    }

    // --- Clip paths ---

    pub fn push_clip(&mut self, x: f64, y: f64, w: f64, h: f64) -> String {
        let id = format!("clip{}", self.next_clip_id);
        self.next_clip_id += 1;
        self.defs.push(format!(
            r#"<clipPath id="{id}"><rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" /></clipPath>"#
        ));
        self.clip_stack.push(id.clone());
        id
    }

    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    // --- Text measurement ---

    pub fn measure_text(&self, content: &str, style: &TextStyle) -> TextMetrics {
        let font = self.fonts.select(style.weight, style.style);
        measure_text(font, content, style.size)
    }

    // --- SVG output ---

    fn push(&mut self, elem: SvgElement) {
        self.elements.push(elem);
    }

    pub fn finish_svg(&self) -> String {
        let mut out = String::with_capacity(32 * 1024);
        writeln!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#,
            w = self.width,
            h = self.height,
        )
        .unwrap();

        // Embedded fonts
        out.push_str(&svg_font_style());
        out.push('\n');

        // Defs (clip paths, patterns)
        if !self.defs.is_empty() {
            out.push_str("<defs>\n");
            for d in &self.defs {
                out.push_str(d);
                out.push('\n');
            }
            out.push_str("</defs>\n");
        }

        // Background (white)
        writeln!(out, r#"<rect width="{}" height="{}" fill="white" />"#, self.width, self.height)
            .unwrap();

        // Elements
        for elem in &self.elements {
            render_element(&mut out, elem);
        }

        out.push_str("</svg>\n");
        out
    }
}

fn render_element(out: &mut String, elem: &SvgElement) {
    match elem {
        SvgElement::Rect { x, y, w, h, style } => {
            write!(out, r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}""#).unwrap();
            write_style_attrs(out, style);
            out.push_str(" />\n");
        }
        SvgElement::Line { x1, y1, x2, y2, style } => {
            write!(out, r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}""#).unwrap();
            write_line_attrs(out, style);
            out.push_str(" />\n");
        }
        SvgElement::Polyline { points, style, close } => {
            let tag = if *close { "polygon" } else { "polyline" };
            write!(out, r#"<{tag} points=""#).unwrap();
            for (i, (x, y)) in points.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                write!(out, "{x:.2},{y:.2}").unwrap();
            }
            out.push('"');
            if *close {
                write_style_attrs(out, &Style::stroked(style.color, style.width));
            } else {
                write!(out, r#" fill="none""#).unwrap();
                write_line_attrs(out, style);
            }
            out.push_str(" />\n");
        }
        SvgElement::Polygon { points, style } => {
            write!(out, r#"<polygon points=""#).unwrap();
            for (i, (x, y)) in points.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                write!(out, "{x:.2},{y:.2}").unwrap();
            }
            out.push('"');
            write_style_attrs(out, style);
            out.push_str(" />\n");
        }
        SvgElement::Text { x, y, content, style, rotate } => {
            write!(out, r#"<text x="{x:.2}" y="{y:.2}""#).unwrap();
            write!(out, r#" font-family="Inter, sans-serif" font-size="{:.1}""#, style.size)
                .unwrap();
            write!(out, r#" fill="{}""#, style.color.to_svg_fill()).unwrap();
            write!(out, r#" text-anchor="{}""#, style.anchor.as_str()).unwrap();
            write!(out, r#" dominant-baseline="{}""#, style.baseline.as_str()).unwrap();
            if style.weight == FontWeight::Bold {
                write!(out, r#" font-weight="bold""#).unwrap();
            }
            if style.style == FontStyle::Italic {
                write!(out, r#" font-style="italic""#).unwrap();
            }
            if let Some(angle) = rotate {
                write!(out, r#" transform="rotate({angle:.1},{x:.2},{y:.2})""#).unwrap();
            }
            out.push('>');
            // Escape XML
            for ch in content.chars() {
                match ch {
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    '&' => out.push_str("&amp;"),
                    '"' => out.push_str("&quot;"),
                    _ => out.push(ch),
                }
            }
            out.push_str("</text>\n");
        }
        SvgElement::Path { d, style } => {
            write!(out, r#"<path d="{d}""#).unwrap();
            write_style_attrs(out, style);
            out.push_str(" />\n");
        }
        SvgElement::Circle { cx, cy, r, style } => {
            write!(out, r#"<circle cx="{cx:.2}" cy="{cy:.2}" r="{r:.2}""#).unwrap();
            write_style_attrs(out, style);
            out.push_str(" />\n");
        }
        SvgElement::Group { transform, clip_id, children } => {
            out.push_str("<g");
            if let Some(t) = transform {
                write!(out, r#" transform="{t}""#).unwrap();
            }
            if let Some(id) = clip_id {
                write!(out, r#" clip-path="url(#{id})""#).unwrap();
            }
            out.push_str(">\n");
            for child in children {
                render_element(out, child);
            }
            out.push_str("</g>\n");
        }
        SvgElement::Raw(s) => {
            out.push_str(s);
            out.push('\n');
        }
    }
}

fn write_style_attrs(out: &mut String, style: &Style) {
    if let Some(fill) = &style.fill {
        write!(out, r#" fill="{}""#, fill.to_svg_fill()).unwrap();
    } else {
        write!(out, r#" fill="none""#).unwrap();
    }
    if let Some(stroke) = &style.stroke {
        write!(out, r#" stroke="{}""#, stroke.to_svg_fill()).unwrap();
        write!(out, r#" stroke-width="{:.2}""#, style.stroke_width).unwrap();
    }
    if (style.opacity - 1.0).abs() > 1e-4 {
        write!(out, r#" opacity="{:.3}""#, style.opacity).unwrap();
    }
}

fn write_line_attrs(out: &mut String, style: &LineStyle) {
    write!(out, r#" stroke="{}""#, style.color.to_svg_fill()).unwrap();
    write!(out, r#" stroke-width="{:.2}""#, style.width).unwrap();
    if let Some(dash) = &style.dash {
        write!(out, r#" stroke-dasharray="{dash}""#).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_canvas() {
        let c = Canvas::new(100.0, 50.0).unwrap();
        let svg = c.finish_svg();
        assert!(svg.contains("width=\"100\""));
        assert!(svg.contains("height=\"50\""));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn rect_rendering() {
        let mut c = Canvas::new(200.0, 100.0).unwrap();
        c.rect(10.0, 20.0, 50.0, 30.0, &Style::filled(Color::hex("#ff0000")));
        let svg = c.finish_svg();
        assert!(svg.contains(r##"fill="#ff0000""##));
        assert!(svg.contains("width=\"50.00\""));
    }

    #[test]
    fn text_rendering() {
        let mut c = Canvas::new(200.0, 100.0).unwrap();
        c.text(10.0, 20.0, "Hello World", &TextStyle::default());
        let svg = c.finish_svg();
        assert!(svg.contains("Hello World"));
        assert!(svg.contains("font-family=\"Inter, sans-serif\""));
    }
}
