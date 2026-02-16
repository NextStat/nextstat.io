use serde::Deserialize;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: f64,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn hex(s: &str) -> Self {
        let s = s.strip_prefix('#').unwrap_or(s);
        let r = u8::from_str_radix(&s[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&s[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&s[4..6], 16).unwrap_or(0);
        Self { r, g, b, a: 1.0 }
    }

    pub const fn with_alpha(mut self, a: f64) -> Self {
        self.a = a;
        self
    }

    pub fn to_svg_fill(&self) -> String {
        if (self.a - 1.0).abs() < 1e-6 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("rgba({},{},{},{:.3})", self.r, self.g, self.b, self.a)
        }
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Linear interpolation between two colors (for colormaps).
    pub fn lerp(a: Color, b: Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: (a.r as f64 * (1.0 - t) + b.r as f64 * t).round() as u8,
            g: (a.g as f64 * (1.0 - t) + b.g as f64 * t).round() as u8,
            b: (a.b as f64 * (1.0 - t) + b.b as f64 * t).round() as u8,
            a: a.a * (1.0 - t) + b.a * t,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_svg_fill())
    }
}

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Color::hex(&s))
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::rgb(0, 0, 0)
    }
}

// --- Palettes ---

pub const HEP2026: &[&str] = &[
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6",
    "#9D755D", "#BAB0AC",
];

pub const ATLAS_WONG: &[&str] =
    &["#0072b2", "#d55e00", "#56b4e9", "#e69f00", "#f0e442", "#009e73", "#cc79a7"];

pub const CMS_PETROFF6: &[&str] =
    &["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"];

pub const TABLEAU10: &[&str] = &[
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ab",
];

pub fn palette_colors(name: &str) -> Vec<Color> {
    let strs = match name {
        "hep2026" => HEP2026,
        "atlas_wong" => ATLAS_WONG,
        "cms_petroff6" => CMS_PETROFF6,
        "tableau10" => TABLEAU10,
        _ => HEP2026,
    };
    strs.iter().map(|s| Color::hex(s)).collect()
}

// --- Diverging colormap (RdBu_r) for correlation matrices ---

/// RdBu_r diverging colormap: -1.0 → blue, 0.0 → white, +1.0 → red
pub fn rdbu_r(val: f64) -> Color {
    let v = val.clamp(-1.0, 1.0);
    if v < 0.0 {
        // white → blue
        let t = -v;
        Color::lerp(Color::rgb(255, 255, 255), Color::hex("#2166ac"), t)
    } else {
        // white → red
        Color::lerp(Color::rgb(255, 255, 255), Color::hex("#b2182b"), v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parsing() {
        let c = Color::hex("#1D4ED8");
        assert_eq!(c.r, 0x1D);
        assert_eq!(c.g, 0x4E);
        assert_eq!(c.b, 0xD8);
        assert!((c.a - 1.0).abs() < 1e-9);
    }

    #[test]
    fn svg_fill_opaque() {
        let c = Color::rgb(29, 78, 216);
        assert_eq!(c.to_svg_fill(), "#1d4ed8");
    }

    #[test]
    fn svg_fill_alpha() {
        let c = Color::rgb(29, 78, 216).with_alpha(0.5);
        assert_eq!(c.to_svg_fill(), "rgba(29,78,216,0.500)");
    }

    #[test]
    fn palette_lookup() {
        assert_eq!(palette_colors("hep2026").len(), 10);
        assert_eq!(palette_colors("atlas_wong").len(), 7);
        assert_eq!(palette_colors("cms_petroff6").len(), 6);
    }

    #[test]
    fn rdbu_extremes() {
        let blue = rdbu_r(-1.0);
        let red = rdbu_r(1.0);
        let white = rdbu_r(0.0);
        assert_eq!(white.r, 255);
        assert!(blue.b > blue.r);
        assert!(red.r > red.b);
    }
}
