use crate::color::Color;
use serde::Deserialize;

/// Fill + stroke style for rectangles and polygons.
#[derive(Debug, Clone)]
pub struct Style {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub opacity: f64,
}

impl Default for Style {
    fn default() -> Self {
        Self { fill: None, stroke: None, stroke_width: 1.0, opacity: 1.0 }
    }
}

impl Style {
    pub fn filled(color: Color) -> Self {
        Self { fill: Some(color), ..Default::default() }
    }

    pub fn stroked(color: Color, width: f64) -> Self {
        Self { stroke: Some(color), stroke_width: width, ..Default::default() }
    }
}

/// Line style.
#[derive(Debug, Clone)]
pub struct LineStyle {
    pub color: Color,
    pub width: f64,
    pub dash: Option<String>,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self { color: Color::rgb(0, 0, 0), width: 1.0, dash: None }
    }
}

impl LineStyle {
    pub fn solid(color: Color, width: f64) -> Self {
        Self { color, width, dash: None }
    }

    pub fn dashed(color: Color, width: f64) -> Self {
        Self { color, width, dash: Some("6 3".into()) }
    }

    pub fn dotted(color: Color, width: f64) -> Self {
        Self { color, width, dash: Some("2 2".into()) }
    }
}

/// Text style.
#[derive(Debug, Clone)]
pub struct TextStyle {
    pub size: f64,
    pub color: Color,
    pub weight: FontWeight,
    pub style: FontStyle,
    pub anchor: TextAnchor,
    pub baseline: TextBaseline,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            size: 10.0,
            color: Color::rgb(0, 0, 0),
            weight: FontWeight::Regular,
            style: FontStyle::Normal,
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Alphabetic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum FontWeight {
    Regular,
    Bold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontStyle {
    Normal,
    Italic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchor {
    Start,
    Middle,
    End,
}

impl TextAnchor {
    pub fn as_str(&self) -> &str {
        match self {
            TextAnchor::Start => "start",
            TextAnchor::Middle => "middle",
            TextAnchor::End => "end",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextBaseline {
    Alphabetic,
    Central,
    Hanging,
}

impl TextBaseline {
    pub fn as_str(&self) -> &str {
        match self {
            TextBaseline::Alphabetic => "auto",
            TextBaseline::Central => "central",
            TextBaseline::Hanging => "hanging",
        }
    }
}

/// Marker style for data points.
#[derive(Debug, Clone)]
pub struct MarkerStyle {
    pub shape: MarkerShape,
    pub size: f64,
    pub color: Color,
    pub fill: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerShape {
    Circle,
    Square,
    Triangle,
    Diamond,
}

impl Default for MarkerStyle {
    fn default() -> Self {
        Self { shape: MarkerShape::Circle, size: 3.0, color: Color::rgb(0, 0, 0), fill: true }
    }
}
