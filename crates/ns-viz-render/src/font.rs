use ab_glyph::FontRef;
use base64::{Engine, engine::general_purpose::STANDARD};

use crate::primitives::{FontStyle, FontWeight};

static INTER_REGULAR: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
static INTER_BOLD: &[u8] = include_bytes!("../fonts/Inter-Bold.ttf");
static INTER_ITALIC: &[u8] = include_bytes!("../fonts/Inter-Italic.ttf");

pub struct FontHandle {
    pub regular: FontRef<'static>,
    pub bold: FontRef<'static>,
    pub italic: FontRef<'static>,
}

impl FontHandle {
    pub fn embedded() -> crate::Result<Self> {
        Ok(Self {
            regular: FontRef::try_from_slice(INTER_REGULAR)
                .map_err(|e| crate::RenderError::Font(e.to_string()))?,
            bold: FontRef::try_from_slice(INTER_BOLD)
                .map_err(|e| crate::RenderError::Font(e.to_string()))?,
            italic: FontRef::try_from_slice(INTER_ITALIC)
                .map_err(|e| crate::RenderError::Font(e.to_string()))?,
        })
    }

    pub fn select(&self, weight: FontWeight, style: FontStyle) -> &FontRef<'static> {
        match (weight, style) {
            (FontWeight::Bold, _) => &self.bold,
            (_, FontStyle::Italic) => &self.italic,
            _ => &self.regular,
        }
    }

    pub fn regular_bytes() -> &'static [u8] {
        INTER_REGULAR
    }

    pub fn bold_bytes() -> &'static [u8] {
        INTER_BOLD
    }

    pub fn italic_bytes() -> &'static [u8] {
        INTER_ITALIC
    }
}

/// Generate SVG `<style>` block with embedded @font-face declarations.
pub fn svg_font_style() -> String {
    let reg_b64 = STANDARD.encode(INTER_REGULAR);
    let bold_b64 = STANDARD.encode(INTER_BOLD);
    let italic_b64 = STANDARD.encode(INTER_ITALIC);
    format!(
        r#"<style>
@font-face {{
  font-family: 'Inter';
  font-weight: 400;
  font-style: normal;
  src: url('data:font/ttf;base64,{reg_b64}') format('truetype');
}}
@font-face {{
  font-family: 'Inter';
  font-weight: 700;
  font-style: normal;
  src: url('data:font/ttf;base64,{bold_b64}') format('truetype');
}}
@font-face {{
  font-family: 'Inter';
  font-weight: 400;
  font-style: italic;
  src: url('data:font/ttf;base64,{italic_b64}') format('truetype');
}}
</style>"#
    )
}
