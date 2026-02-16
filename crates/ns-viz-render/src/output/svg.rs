use std::path::Path;

/// Write SVG string to a file.
pub fn save_svg(svg: &str, path: &Path) -> crate::Result<()> {
    std::fs::write(path, svg)?;
    Ok(())
}
