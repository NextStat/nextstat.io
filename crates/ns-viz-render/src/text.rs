use ab_glyph::{Font, FontRef, ScaleFont};

use crate::font::FontHandle;
use crate::primitives::TextStyle;

#[derive(Debug, Clone, Copy)]
pub struct TextMetrics {
    pub width: f64,
    pub height: f64,
    pub ascent: f64,
}

/// Measure text width and height in points using ab_glyph.
pub fn measure_text(font: &FontRef<'_>, text: &str, size_pt: f64) -> TextMetrics {
    let scale = ab_glyph::PxScale::from(size_pt as f32);
    let scaled = font.as_scaled(scale);

    let mut width: f32 = 0.0;
    let mut prev_glyph_id = None;
    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        if let Some(prev) = prev_glyph_id {
            width += scaled.kern(prev, glyph_id);
        }
        width += scaled.h_advance(glyph_id);
        prev_glyph_id = Some(glyph_id);
    }

    let ascent = scaled.ascent();
    let descent = scaled.descent();
    let height = ascent - descent;

    TextMetrics { width: width as f64, height: height as f64, ascent: ascent as f64 }
}

/// Measure text with a TextStyle, selecting the correct font face.
pub fn measure_styled(fonts: &FontHandle, text: &str, style: &TextStyle) -> TextMetrics {
    let font = fonts.select(style.weight, style.style);
    measure_text(font, text, style.size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_hello() {
        let fonts = FontHandle::embedded().unwrap();
        let m = measure_text(&fonts.regular, "Hello", 12.0);
        assert!(m.width > 20.0);
        assert!(m.height > 8.0);
        assert!(m.ascent > 0.0);
    }

    #[test]
    fn bold_wider_than_regular() {
        let fonts = FontHandle::embedded().unwrap();
        let r = measure_text(&fonts.regular, "Test", 12.0);
        let b = measure_text(&fonts.bold, "Test", 12.0);
        // Bold is typically wider
        assert!(b.width >= r.width * 0.95);
    }
}
