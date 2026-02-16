use serde::Deserialize;
use std::collections::HashMap;

use crate::color::Color;
use crate::theme::BuiltinTheme;

/// Top-level visualization configuration (YAML or programmatic).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VizConfig {
    pub theme: String,
    pub figure: FigureConfig,
    pub font: FontConfig,
    pub axes: AxesConfig,
    pub grid: GridConfig,
    pub experiment: ExperimentConfig,
    pub colors: ColorsConfig,
    pub palette: String,
    pub sample_colors: HashMap<String, Color>,
    pub channels: HashMap<String, ChannelConfig>,
    pub output: OutputConfig,
    pub distributions: DistributionsConfig,
    pub ranking: RankingConfig,
    pub pulls: PullsConfig,
    pub corr: CorrConfig,
}

impl Default for VizConfig {
    fn default() -> Self {
        BuiltinTheme::NextStat2026.base_config()
    }
}

impl VizConfig {
    pub fn palette_colors(&self) -> Vec<Color> {
        crate::color::palette_colors(&self.palette)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FigureConfig {
    pub width: f64,
    pub height: f64,
}

impl Default for FigureConfig {
    fn default() -> Self {
        Self {
            width: 518.4,  // 7.2" * 72
            height: 302.4, // 4.2" * 72
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FontConfig {
    pub size: f64,
    pub label_size: f64,
    pub tick_size: f64,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self { size: 10.0, label_size: 11.0, tick_size: 8.5 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AxesConfig {
    pub tick_direction: String,
    pub show_top_ticks: bool,
    pub show_right_ticks: bool,
    pub tick_length: f64,
    pub minor_tick_length: f64,
}

impl Default for AxesConfig {
    fn default() -> Self {
        Self {
            tick_direction: "in".into(),
            show_top_ticks: true,
            show_right_ticks: true,
            tick_length: 5.0,
            minor_tick_length: 3.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GridConfig {
    pub show: bool,
    pub color: Color,
    pub alpha: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self { show: true, color: Color::hex("#CBD5E1"), alpha: 0.55 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ExperimentConfig {
    pub name: String,
    pub status: String,
    pub sqrt_s_tev: f64,
    pub lumi_fb_inv: f64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "NEXTSTAT".into(),
            status: "Internal".into(),
            sqrt_s_tev: 13.6,
            lumi_fb_inv: 140.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ColorsConfig {
    pub observed: Color,
    pub expected: Color,
    pub band_1sigma: Color,
    pub band_2sigma: Color,
    pub signal: Color,
    pub positive_pull: Color,
    pub negative_pull: Color,
}

impl Default for ColorsConfig {
    fn default() -> Self {
        Self {
            observed: Color::hex("#111827"),
            expected: Color::hex("#1D4ED8"),
            band_1sigma: Color::hex("#7BD389"),
            band_2sigma: Color::hex("#F2D95C"),
            signal: Color::hex("#DC2626"),
            positive_pull: Color::hex("#1D4ED8"),
            negative_pull: Color::hex("#DC2626"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub format: String,
    pub dpi: u32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self { format: "svg".into(), dpi: 220 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DistributionsConfig {
    pub show_mc_band: bool,
    pub band_hatch: String,
    pub ratio_y_range: [f64; 2],
}

impl Default for DistributionsConfig {
    fn default() -> Self {
        Self { show_mc_band: true, band_hatch: "////".into(), ratio_y_range: [0.5, 1.5] }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RankingConfig {
    pub top_n: Option<usize>,
    pub poi_label: String,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            top_n: None,
            poi_label: "\u{03BC}".into(), // μ
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PullsConfig {
    pub max_nps: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CorrConfig {
    pub order: String,
    pub cmap: String,
}

impl Default for CorrConfig {
    fn default() -> Self {
        Self { order: "group_base".into(), cmap: "RdBu_r".into() }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ChannelConfig {
    pub blinded: bool,
}

/// Resolve a VizConfig from optional YAML string.
/// Priority: user YAML overrides → theme base config.
pub fn resolve_config(user_yaml: Option<&str>) -> crate::Result<VizConfig> {
    match user_yaml {
        None => Ok(VizConfig::default()),
        Some(yaml) => {
            let config: VizConfig = serde_yaml_ng::from_str(yaml)
                .map_err(|e| crate::RenderError::Config(e.to_string()))?;
            Ok(config)
        }
    }
}
