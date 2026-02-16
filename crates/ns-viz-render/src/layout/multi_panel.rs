use crate::layout::margins::PlotArea;

/// Main + Ratio panel layout (e.g., distributions plot).
/// Main panel gets ~75% of available height, ratio panel gets ~25%.
#[derive(Debug, Clone)]
pub struct MainRatioLayout {
    pub main: PlotArea,
    pub ratio: PlotArea,
}

impl MainRatioLayout {
    pub fn new(
        left: f64,
        top: f64,
        width: f64,
        total_height: f64,
        gap: f64,
        ratio_frac: f64,
    ) -> Self {
        let ratio_h = total_height * ratio_frac;
        let main_h = total_height - ratio_h - gap;

        Self {
            main: PlotArea::manual(left, top, width, main_h),
            ratio: PlotArea::manual(left, top + main_h + gap, width, ratio_h),
        }
    }
}

/// Dual panel layout (e.g., ranking: impact panel + pull panel).
/// Left panel gets `left_frac` of width, right panel gets the rest.
#[derive(Debug, Clone)]
pub struct DualPanelLayout {
    pub left: PlotArea,
    pub right: PlotArea,
}

impl DualPanelLayout {
    pub fn new(x: f64, top: f64, total_width: f64, height: f64, gap: f64, left_frac: f64) -> Self {
        let left_w = total_width * left_frac;
        let right_w = total_width - left_w - gap;

        Self {
            left: PlotArea::manual(x, top, left_w, height),
            right: PlotArea::manual(x + left_w + gap, top, right_w, height),
        }
    }
}
