#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
    use ns_translate::pyhf::{HistFactoryModel, Workspace};
    use serde::{Deserialize, Serialize};
    use wasm_bindgen::prelude::*;

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct RunOptions {
        alpha: Option<f64>,
        scan_start: Option<f64>,
        scan_stop: Option<f64>,
        scan_points: Option<usize>,
    }

    impl Default for RunOptions {
        fn default() -> Self {
            Self {
                alpha: Some(0.05),
                scan_start: Some(0.0),
                scan_stop: Some(5.0),
                scan_points: Some(41),
            }
        }
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct PlaygroundResult {
        measurement_name: Option<String>,
        poi_name: String,
        n_params: usize,
        mu_hat: f64,
        free_data_nll: f64,
        alpha: f64,
        scan: Vec<f64>,
        observed_cls: Vec<f64>,
        expected_cls: Vec<[f64; 5]>,
        observed_limit: f64,
        expected_limits: [f64; 5],
        elapsed_ms: f64,
    }

    fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![start];
        }
        let step = (stop - start) / ((n - 1) as f64);
        (0..n).map(|i| start + (i as f64) * step).collect()
    }

    fn js_err(msg: impl Into<String>) -> JsValue {
        js_sys::Error::new(&msg.into()).into()
    }

    #[wasm_bindgen]
    pub fn run_asymptotic_upper_limits(
        workspace_json: &str,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();

        let opts: RunOptions = if options.is_null() || options.is_undefined() {
            RunOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|e| js_err(e.to_string()))?
        };

        let alpha = opts.alpha.unwrap_or(0.05);
        let scan_start = opts.scan_start.unwrap_or(0.0);
        let scan_stop = opts.scan_stop.unwrap_or(5.0);
        let scan_points = opts.scan_points.unwrap_or(41);

        if !(0.0 < alpha && alpha < 1.0) {
            return Err(js_err(format!("alpha must be in (0,1), got {alpha}")));
        }
        if scan_points < 2 {
            return Err(js_err(format!("scan_points must be >= 2, got {scan_points}")));
        }
        if !scan_start.is_finite() || !scan_stop.is_finite() {
            return Err(js_err("scan_start/scan_stop must be finite"));
        }

        let start_t = web_time::Instant::now();

        let ws: Workspace =
            serde_json::from_str(workspace_json).map_err(|e| js_err(e.to_string()))?;
        let measurement_name = ws.measurements.first().map(|m| m.name.clone());
        let model = HistFactoryModel::from_workspace(&ws).map_err(|e| js_err(e.to_string()))?;

        let mle = MaximumLikelihoodEstimator::new();
        let ctx = AsymptoticCLsContext::new(&mle, &model).map_err(|e| js_err(e.to_string()))?;

        let poi_idx = ctx.poi_index();
        let poi_name = model
            .parameters()
            .get(poi_idx)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "poi".to_string());

        let scan = linspace(scan_start, scan_stop, scan_points);
        let scan_res = ctx
            .upper_limits_qtilde_linear_scan_result(&mle, alpha, &scan)
            .map_err(|e| js_err(e.to_string()))?;

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = PlaygroundResult {
            measurement_name,
            poi_name,
            n_params: model.parameters().len(),
            mu_hat: ctx.mu_hat(),
            free_data_nll: ctx.free_data_nll(),
            alpha,
            scan: scan_res.scan,
            observed_cls: scan_res.observed_cls,
            expected_cls: scan_res.expected_cls,
            observed_limit: scan_res.observed_limit,
            expected_limits: scan_res.expected_limits,
            elapsed_ms,
        };

        serde_wasm_bindgen::to_value(&out).map_err(|e| js_err(e.to_string()))
    }
}
