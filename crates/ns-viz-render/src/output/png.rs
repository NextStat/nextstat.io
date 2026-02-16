use crate::{RenderError, font::FontHandle};

/// Convert SVG string to PNG bytes at the given DPI.
pub fn svg_to_png(svg: &str, dpi: u32) -> crate::Result<Vec<u8>> {
    let mut opt = usvg::Options::default();
    let fontdb = opt.fontdb_mut();
    fontdb.load_font_data(FontHandle::regular_bytes().to_vec());
    fontdb.load_font_data(FontHandle::bold_bytes().to_vec());
    fontdb.load_font_data(FontHandle::italic_bytes().to_vec());

    let tree = usvg::Tree::from_str(svg, &opt).map_err(|e| RenderError::Png(e.to_string()))?;

    let scale = dpi as f32 / 72.0;
    let size = tree.size();
    let w = (size.width() * scale) as u32;
    let h = (size.height() * scale) as u32;

    let mut pixmap = tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| RenderError::Png("failed to create pixmap".into()))?;

    // Fill white background
    pixmap.fill(tiny_skia::Color::WHITE);

    resvg::render(&tree, tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());

    pixmap.encode_png().map_err(|e| RenderError::Png(e.to_string()))
}
