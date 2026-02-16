/// Axis configuration with tick generation and data→pixel mapping.
#[derive(Debug, Clone)]
pub struct Axis {
    pub min: f64,
    pub max: f64,
    pub log: bool,
    pub label: String,
    pub tick_positions: Vec<f64>,
    pub tick_labels: Vec<String>,
    pub minor_ticks: Vec<f64>,
}

impl Axis {
    /// Auto-scale linear axis with "nice number" ticks.
    pub fn auto_linear(data_min: f64, data_max: f64, target_ticks: usize) -> Self {
        let (nice_min, nice_max, step) = nice_range(data_min, data_max, target_ticks);
        let mut ticks = Vec::new();
        let mut labels = Vec::new();
        let mut v = nice_min;
        while v <= nice_max + step * 0.01 {
            ticks.push(v);
            labels.push(format_tick(v, step));
            v += step;
        }

        // Minor ticks: 5 subdivisions per major
        let minor_step = step / 5.0;
        let mut minor = Vec::new();
        let mut mv = nice_min;
        while mv <= nice_max + minor_step * 0.01 {
            if !ticks.iter().any(|t| (t - mv).abs() < minor_step * 0.01) {
                minor.push(mv);
            }
            mv += minor_step;
        }

        Self {
            min: nice_min,
            max: nice_max,
            log: false,
            label: String::new(),
            tick_positions: ticks,
            tick_labels: labels,
            minor_ticks: minor,
        }
    }

    /// Auto-scale logarithmic axis.
    pub fn auto_log(data_min: f64, data_max: f64) -> Self {
        let log_min = data_min.max(1e-20).log10().floor() as i32;
        let log_max = data_max.max(1e-20).log10().ceil() as i32;

        let mut ticks = Vec::new();
        let mut labels = Vec::new();
        let mut minor = Vec::new();

        for exp in log_min..=log_max {
            let v = 10.0_f64.powi(exp);
            ticks.push(v);
            labels.push(format!("10{}", superscript(exp)));
            // Minor ticks at 2..9
            for m in 2..=9 {
                let mv = m as f64 * 10.0_f64.powi(exp - 1);
                if mv > data_min * 0.5 && mv < data_max * 2.0 {
                    minor.push(mv);
                }
            }
        }

        Self {
            min: 10.0_f64.powi(log_min),
            max: 10.0_f64.powi(log_max),
            log: true,
            label: String::new(),
            tick_positions: ticks,
            tick_labels: labels,
            minor_ticks: minor,
        }
    }

    /// Fixed axis with explicit limits (no tick auto-generation).
    pub fn fixed(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            log: false,
            label: String::new(),
            tick_positions: Vec::new(),
            tick_labels: Vec::new(),
            minor_ticks: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Map a data value to pixel coordinate.
    pub fn data_to_pixel(&self, value: f64, px_min: f64, px_max: f64) -> f64 {
        if self.log {
            let log_val = value.max(1e-20).ln();
            let log_min = self.min.max(1e-20).ln();
            let log_max = self.max.max(1e-20).ln();
            let frac = (log_val - log_min) / (log_max - log_min);
            px_min + frac * (px_max - px_min)
        } else {
            let frac = (value - self.min) / (self.max - self.min);
            px_min + frac * (px_max - px_min)
        }
    }

    /// Map pixel coordinate to data value (inverse).
    pub fn pixel_to_data(&self, px: f64, px_min: f64, px_max: f64) -> f64 {
        let frac = (px - px_min) / (px_max - px_min);
        if self.log {
            let log_min = self.min.max(1e-20).ln();
            let log_max = self.max.max(1e-20).ln();
            (log_min + frac * (log_max - log_min)).exp()
        } else {
            self.min + frac * (self.max - self.min)
        }
    }
}

/// "Nice numbers" algorithm for pleasant tick spacing.
fn nice_range(data_min: f64, data_max: f64, target_ticks: usize) -> (f64, f64, f64) {
    if (data_max - data_min).abs() < 1e-15 {
        return (data_min - 1.0, data_max + 1.0, 1.0);
    }
    let range = data_max - data_min;
    let rough_step = range / (target_ticks.max(2) - 1) as f64;
    let step = nice_step(rough_step);
    let nice_min = (data_min / step).floor() * step;
    let nice_max = (data_max / step).ceil() * step;
    (nice_min, nice_max, step)
}

fn nice_step(rough: f64) -> f64 {
    let exp = rough.abs().log10().floor();
    let frac = rough / 10.0_f64.powf(exp);
    let nice_frac = if frac <= 1.5 {
        1.0
    } else if frac <= 3.5 {
        2.0
    } else if frac <= 7.5 {
        5.0
    } else {
        10.0
    };
    nice_frac * 10.0_f64.powf(exp)
}

fn format_tick(value: f64, step: f64) -> String {
    let decimals = if step >= 1.0 { 0 } else { (-step.log10().floor()) as usize };
    if decimals == 0 {
        // Avoid "-0"
        let v = if value.abs() < step * 0.01 { 0.0 } else { value };
        format!("{}", v as i64)
    } else {
        format!("{:.prec$}", value, prec = decimals)
    }
}

fn superscript(n: i32) -> String {
    let s = n.to_string();
    s.chars()
        .map(|c| match c {
            '-' => '\u{207B}',
            '0' => '\u{2070}',
            '1' => '\u{00B9}',
            '2' => '\u{00B2}',
            '3' => '\u{00B3}',
            '4' => '\u{2074}',
            '5' => '\u{2075}',
            '6' => '\u{2076}',
            '7' => '\u{2077}',
            '8' => '\u{2078}',
            '9' => '\u{2079}',
            _ => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_linear_basic() {
        let ax = Axis::auto_linear(0.0, 10.0, 6);
        assert!(!ax.tick_positions.is_empty());
        assert!(ax.min <= 0.0);
        assert!(ax.max >= 10.0);
    }

    #[test]
    fn data_to_pixel_linear() {
        let ax = Axis::auto_linear(0.0, 100.0, 5);
        let px = ax.data_to_pixel(50.0, 0.0, 500.0);
        assert!((px - 250.0).abs() < 1.0);
    }

    #[test]
    fn auto_log() {
        let ax = Axis::auto_log(0.01, 1000.0);
        assert!(ax.log);
        assert!(ax.min <= 0.01);
        assert!(ax.max >= 1000.0);
    }

    #[test]
    fn nice_step_values() {
        assert!((nice_step(3.2) - 2.0).abs() < 1e-9);
        assert!((nice_step(0.7) - 0.5).abs() < 1e-9);
        assert!((nice_step(15.0) - 10.0).abs() < 1e-9);
        assert!((nice_step(4.5) - 5.0).abs() < 1e-9);
        assert!((nice_step(1.2) - 1.0).abs() < 1e-9);
    }
}
