use crate::{RenderError, font::FontHandle};

/// Convert SVG string to PDF bytes.
pub fn svg_to_pdf(svg: &str) -> crate::Result<Vec<u8>> {
    let mut opt = usvg::Options::default();
    let fontdb = opt.fontdb_mut();
    fontdb.load_font_data(FontHandle::regular_bytes().to_vec());
    fontdb.load_font_data(FontHandle::bold_bytes().to_vec());
    fontdb.load_font_data(FontHandle::italic_bytes().to_vec());

    let tree = usvg::Tree::from_str(svg, &opt).map_err(|e| RenderError::Pdf(e.to_string()))?;

    svg2pdf::to_pdf(&tree, svg2pdf::ConversionOptions::default(), svg2pdf::PageOptions::default())
        .map_err(|e| RenderError::Pdf(e.to_string()))
}
