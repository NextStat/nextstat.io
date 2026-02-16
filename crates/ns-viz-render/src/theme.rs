use crate::color::Color;
use crate::config::*;

/// Built-in theme presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinTheme {
    NextStat2026,
    Atlas,
    Cms,
    Minimal,
}

impl BuiltinTheme {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "atlas" => Self::Atlas,
            "cms" => Self::Cms,
            "minimal" => Self::Minimal,
            _ => Self::NextStat2026,
        }
    }

    pub fn base_config(self) -> VizConfig {
        match self {
            Self::NextStat2026 => nextstat2026(),
            Self::Atlas => atlas(),
            Self::Cms => cms(),
            Self::Minimal => minimal(),
        }
    }
}

fn nextstat2026() -> VizConfig {
    VizConfig {
        theme: "nextstat2026".into(),
        figure: FigureConfig { width: 518.4, height: 302.4 },
        font: FontConfig { size: 10.0, label_size: 11.0, tick_size: 8.5 },
        axes: AxesConfig {
            tick_direction: "in".into(),
            show_top_ticks: true,
            show_right_ticks: true,
            tick_length: 5.0,
            minor_tick_length: 3.0,
        },
        grid: GridConfig { show: true, color: Color::hex("#CBD5E1"), alpha: 0.55 },
        experiment: ExperimentConfig {
            name: "NEXTSTAT".into(),
            status: "Internal".into(),
            sqrt_s_tev: 13.6,
            lumi_fb_inv: 140.0,
        },
        palette: "hep2026".into(),
        ..VizConfig {
            theme: String::new(),
            figure: FigureConfig::default(),
            font: FontConfig::default(),
            axes: AxesConfig::default(),
            grid: GridConfig::default(),
            experiment: ExperimentConfig::default(),
            colors: ColorsConfig::default(),
            palette: String::new(),
            sample_colors: Default::default(),
            channels: Default::default(),
            output: OutputConfig::default(),
            distributions: DistributionsConfig::default(),
            ranking: RankingConfig::default(),
            pulls: PullsConfig::default(),
            corr: CorrConfig::default(),
        }
    }
}

fn atlas() -> VizConfig {
    VizConfig {
        theme: "atlas".into(),
        figure: FigureConfig { width: 576.0, height: 432.0 },
        font: FontConfig { size: 11.0, label_size: 12.0, tick_size: 9.5 },
        axes: AxesConfig {
            tick_direction: "in".into(),
            show_top_ticks: true,
            show_right_ticks: true,
            tick_length: 6.0,
            minor_tick_length: 3.0,
        },
        grid: GridConfig { show: false, ..GridConfig::default() },
        experiment: ExperimentConfig {
            name: "ATLAS".into(),
            status: "Internal".into(),
            sqrt_s_tev: 13.6,
            lumi_fb_inv: 140.0,
        },
        palette: "atlas_wong".into(),
        ..nextstat2026()
    }
}

fn cms() -> VizConfig {
    VizConfig {
        theme: "cms".into(),
        figure: FigureConfig { width: 720.0, height: 720.0 },
        font: FontConfig { size: 10.0, label_size: 11.0, tick_size: 9.0 },
        axes: AxesConfig {
            tick_direction: "in".into(),
            show_top_ticks: true,
            show_right_ticks: true,
            tick_length: 5.0,
            minor_tick_length: 3.0,
        },
        grid: GridConfig { show: false, ..GridConfig::default() },
        experiment: ExperimentConfig {
            name: "CMS".into(),
            status: "Preliminary".into(),
            sqrt_s_tev: 13.6,
            lumi_fb_inv: 138.0,
        },
        palette: "cms_petroff6".into(),
        ..nextstat2026()
    }
}

fn minimal() -> VizConfig {
    VizConfig {
        theme: "minimal".into(),
        figure: FigureConfig { width: 432.0, height: 302.4 },
        font: FontConfig { size: 9.0, label_size: 10.0, tick_size: 8.0 },
        axes: AxesConfig {
            tick_direction: "out".into(),
            show_top_ticks: false,
            show_right_ticks: false,
            tick_length: 4.0,
            minor_tick_length: 2.0,
        },
        grid: GridConfig { show: false, ..GridConfig::default() },
        experiment: ExperimentConfig {
            name: String::new(),
            status: String::new(),
            sqrt_s_tev: 0.0,
            lumi_fb_inv: 0.0,
        },
        palette: "tableau10".into(),
        ..nextstat2026()
    }
}
