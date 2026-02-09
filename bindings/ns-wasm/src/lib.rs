#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use ns_inference::profile_likelihood::scan_histfactory;
    use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
    use ns_translate::arrow::ingest::ArrowIngestConfig;
    use ns_translate::arrow::parquet as parquet_io;
    use ns_translate::pyhf::schema::{
        Channel, Measurement, MeasurementConfig, Modifier, Observation, Sample,
    };
    use ns_translate::pyhf::{HistFactoryModel, Workspace};
    use serde::{Deserialize, Serialize};
    use wasm_bindgen::prelude::*;

    // ── Memory safety constants ──────────────────────────────────────────
    const MAX_WORKSPACE_BYTES: usize = 5 * 1024 * 1024; // 5 MB JSON
    const MAX_PARAMS: usize = 500;
    const MAX_SCAN_POINTS: usize = 200;

    fn guard_workspace(json: &str) -> Result<(), JsValue> {
        if json.len() > MAX_WORKSPACE_BYTES {
            return Err(js_err(format!(
                "Workspace JSON too large: {} bytes (max {} MB). Use a smaller workspace or the native CLI.",
                json.len(),
                MAX_WORKSPACE_BYTES / (1024 * 1024)
            )));
        }
        Ok(())
    }

    fn guard_model(model: &HistFactoryModel) -> Result<(), JsValue> {
        let n = model.parameters().len();
        if n > MAX_PARAMS {
            return Err(js_err(format!(
                "Model has {} parameters (max {} in browser). Use the native CLI for large models.",
                n, MAX_PARAMS
            )));
        }
        Ok(())
    }

    fn parse_workspace(json: &str) -> Result<(Workspace, HistFactoryModel), JsValue> {
        guard_workspace(json)?;
        let ws: Workspace = serde_json::from_str(json).map_err(|e| js_err(e.to_string()))?;
        let model = HistFactoryModel::from_workspace(&ws).map_err(|e| js_err(e.to_string()))?;
        guard_model(&model)?;
        Ok((ws, model))
    }

    // ── Histogram-table ingestion (CSV/Parquet -> Workspace JSON) ─────────────

    #[derive(Debug, Clone, Deserialize)]
    struct HistogramRow {
        channel: String,
        sample: String,
        yields: Vec<f64>,
        #[serde(default)]
        stat_error: Option<Vec<f64>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct HistogramIngestOptions {
        poi: Option<String>,
        observations: Option<std::collections::HashMap<String, Vec<f64>>>,
        measurement_name: Option<String>,
    }

    fn ingest_config_from_options(options: JsValue) -> Result<ArrowIngestConfig, JsValue> {
        let opts: HistogramIngestOptions = if options.is_null() || options.is_undefined() {
            HistogramIngestOptions {
                poi: Some("mu".to_string()),
                observations: None,
                measurement_name: None,
            }
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|e| js_err(e.to_string()))?
        };

        let mut cfg = ArrowIngestConfig::default();
        if let Some(poi) = opts.poi {
            cfg.poi = poi;
        }
        cfg.observations = opts.observations;
        if let Some(m) = opts.measurement_name {
            cfg.measurement_name = m;
        }
        Ok(cfg)
    }

    fn asimov_from_samples(samples: &[Sample]) -> Vec<f64> {
        if samples.is_empty() {
            return vec![];
        }
        let n_bins = samples[0].data.len();
        let mut out = vec![0.0; n_bins];
        for s in samples {
            for (i, &v) in s.data.iter().enumerate() {
                if i < n_bins {
                    out[i] += v;
                }
            }
        }
        out
    }

    fn workspace_from_rows(
        rows: &[HistogramRow],
        cfg: &ArrowIngestConfig,
    ) -> Result<Workspace, JsValue> {
        use std::collections::HashMap;

        // Group rows by channel
        let mut channel_map: HashMap<&str, Vec<&HistogramRow>> = HashMap::new();
        for r in rows {
            channel_map.entry(r.channel.as_str()).or_default().push(r);
        }

        let mut channels: Vec<Channel> = Vec::new();
        let mut observations: Vec<Observation> = Vec::new();

        for (channel_name, channel_rows) in channel_map {
            let mut expected_bins: Option<usize> = None;
            let mut samples: Vec<Sample> = Vec::new();

            for row in channel_rows {
                if row.yields.is_empty() {
                    return Err(js_err(format!(
                        "Empty yields for channel={} sample={}",
                        row.channel, row.sample
                    )));
                }

                if let Some(ref se) = row.stat_error {
                    if se.len() != row.yields.len() {
                        return Err(js_err(format!(
                            "stat_error length mismatch for channel={} sample={}: yields={} stat_error={}",
                            row.channel,
                            row.sample,
                            row.yields.len(),
                            se.len()
                        )));
                    }
                }

                let n_bins = row.yields.len();
                if let Some(eb) = expected_bins {
                    if n_bins != eb {
                        return Err(js_err(format!(
                            "Inconsistent bin count in channel '{}': expected {}, got {} (sample '{}')",
                            channel_name, eb, n_bins, row.sample
                        )));
                    }
                } else {
                    expected_bins = Some(n_bins);
                }

                let mut modifiers: Vec<Modifier> = Vec::new();

                // Mirror ns-translate Arrow ingest semantics:
                // - attach POI normfactor if sample == poi OR sample == "signal"
                if row.sample == cfg.poi || row.sample == "signal" {
                    modifiers.push(Modifier::NormFactor { name: cfg.poi.clone(), data: None });
                }

                if let Some(ref se) = row.stat_error {
                    modifiers.push(Modifier::StatError {
                        name: format!("staterror_{}", channel_name),
                        data: se.clone(),
                    });
                }

                samples.push(Sample {
                    name: row.sample.clone(),
                    data: row.yields.clone(),
                    modifiers,
                });
            }

            // Deterministic ordering of samples (important for stable JSON + reproducibility).
            samples.sort_by(|a, b| a.name.cmp(&b.name));

            // Observation: user-provided or Asimov.
            let obs_data = if let Some(ref obs_map) = cfg.observations {
                obs_map.get(channel_name).cloned().unwrap_or_else(|| asimov_from_samples(&samples))
            } else {
                asimov_from_samples(&samples)
            };

            observations.push(Observation { name: channel_name.to_string(), data: obs_data });

            channels.push(Channel { name: channel_name.to_string(), samples });
        }

        channels.sort_by(|a, b| a.name.cmp(&b.name));
        observations.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Workspace {
            channels,
            observations,
            measurements: vec![Measurement {
                name: cfg.measurement_name.clone(),
                config: MeasurementConfig { poi: cfg.poi.clone(), parameters: vec![] },
            }],
            version: Some("1.0.0".to_string()),
        })
    }

    #[wasm_bindgen]
    pub fn workspace_from_histogram_rows_json(
        rows_json: &str,
        options: JsValue,
    ) -> Result<String, JsValue> {
        console_error_panic_hook::set_once();
        let cfg = ingest_config_from_options(options)?;
        let rows: Vec<HistogramRow> =
            serde_json::from_str(rows_json).map_err(|e| js_err(e.to_string()))?;
        let ws = workspace_from_rows(&rows, &cfg)?;
        serde_json::to_string(&ws).map_err(|e| js_err(e.to_string()))
    }

    #[wasm_bindgen]
    pub fn workspace_from_parquet_bytes(
        parquet_bytes: js_sys::Uint8Array,
        options: JsValue,
    ) -> Result<String, JsValue> {
        console_error_panic_hook::set_once();
        let cfg = ingest_config_from_options(options)?;

        let mut buf = vec![0u8; parquet_bytes.length() as usize];
        parquet_bytes.copy_to(&mut buf);

        let ws = parquet_io::from_parquet_bytes(&buf, &cfg).map_err(|e| js_err(e.to_string()))?;
        serde_json::to_string(&ws).map_err(|e| js_err(e.to_string()))
    }

    // ── Shared types ─────────────────────────────────────────────────────

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

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct FitResultJs {
        measurement_name: Option<String>,
        parameter_names: Vec<String>,
        parameters: Vec<f64>,
        uncertainties: Vec<f64>,
        nll: f64,
        converged: bool,
        n_iter: usize,
        elapsed_ms: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct ProfilePointJs {
        mu: f64,
        delta_nll: f64,
        q_mu: f64,
        nll_mu: f64,
        converged: bool,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct ProfileScanResultJs {
        measurement_name: Option<String>,
        poi_name: String,
        mu_hat: f64,
        nll_hat: f64,
        n_params: usize,
        points: Vec<ProfilePointJs>,
        elapsed_ms: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct HypotestResultJs {
        measurement_name: Option<String>,
        poi_name: String,
        mu_test: f64,
        cls: f64,
        clsb: f64,
        clb: f64,
        q_mu: f64,
        q_mu_a: f64,
        mu_hat: f64,
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

    fn poi_name(model: &HistFactoryModel) -> String {
        model
            .poi_index()
            .and_then(|i| model.parameters().get(i))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "poi".to_string())
    }

    // ── 1. Brazil Band (existing) ────────────────────────────────────────

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
        let scan_points = opts.scan_points.unwrap_or(41).min(MAX_SCAN_POINTS);

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

        let (ws, model) = parse_workspace(workspace_json)?;
        let measurement_name = ws.measurements.first().map(|m| m.name.clone());

        let mle = MaximumLikelihoodEstimator::new();
        let ctx = AsymptoticCLsContext::new(&mle, &model).map_err(|e| js_err(e.to_string()))?;

        let scan = linspace(scan_start, scan_stop, scan_points);
        let scan_res = ctx
            .upper_limits_qtilde_linear_scan_result(&mle, alpha, &scan)
            .map_err(|e| js_err(e.to_string()))?;

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = PlaygroundResult {
            measurement_name,
            poi_name: poi_name(&model),
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

    // ── 2. MLE Fit ───────────────────────────────────────────────────────

    #[wasm_bindgen]
    pub fn run_fit(workspace_json: &str) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let start_t = web_time::Instant::now();

        let (ws, model) = parse_workspace(workspace_json)?;
        let measurement_name = ws.measurements.first().map(|m| m.name.clone());

        let mle = MaximumLikelihoodEstimator::new();
        let fit = mle.fit_histfactory(&model).map_err(|e| js_err(e.to_string()))?;

        let parameter_names: Vec<String> =
            model.parameters().iter().map(|p| p.name.clone()).collect();

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = FitResultJs {
            measurement_name,
            parameter_names,
            parameters: fit.parameters,
            uncertainties: fit.uncertainties,
            nll: fit.nll,
            converged: fit.converged,
            n_iter: fit.n_iter,
            elapsed_ms,
        };

        serde_wasm_bindgen::to_value(&out).map_err(|e| js_err(e.to_string()))
    }

    // ── 3. Profile Likelihood Scan ───────────────────────────────────────

    #[wasm_bindgen]
    pub fn run_profile_scan(workspace_json: &str, options: JsValue) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();

        let opts: RunOptions = if options.is_null() || options.is_undefined() {
            RunOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|e| js_err(e.to_string()))?
        };

        let scan_start = opts.scan_start.unwrap_or(0.0);
        let scan_stop = opts.scan_stop.unwrap_or(5.0);
        let scan_points = opts.scan_points.unwrap_or(41).min(MAX_SCAN_POINTS);

        if scan_points < 2 {
            return Err(js_err(format!("scan_points must be >= 2, got {scan_points}")));
        }
        if !scan_start.is_finite() || !scan_stop.is_finite() {
            return Err(js_err("scan_start/scan_stop must be finite"));
        }

        let start_t = web_time::Instant::now();

        let (ws, model) = parse_workspace(workspace_json)?;
        let measurement_name = ws.measurements.first().map(|m| m.name.clone());

        let mle = MaximumLikelihoodEstimator::new();
        let mu_values = linspace(scan_start, scan_stop, scan_points);

        let result =
            scan_histfactory(&mle, &model, &mu_values).map_err(|e| js_err(e.to_string()))?;

        let points: Vec<ProfilePointJs> = result
            .points
            .iter()
            .map(|p| ProfilePointJs {
                mu: p.mu,
                delta_nll: p.nll_mu - result.nll_hat,
                q_mu: p.q_mu,
                nll_mu: p.nll_mu,
                converged: p.converged,
            })
            .collect();

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = ProfileScanResultJs {
            measurement_name,
            poi_name: poi_name(&model),
            mu_hat: result.mu_hat,
            nll_hat: result.nll_hat,
            n_params: model.parameters().len(),
            points,
            elapsed_ms,
        };

        serde_wasm_bindgen::to_value(&out).map_err(|e| js_err(e.to_string()))
    }

    // ── 4. Single-point Hypothesis Test ──────────────────────────────────

    #[wasm_bindgen]
    pub fn run_hypotest(workspace_json: &str, mu_test: f64) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();

        if !mu_test.is_finite() {
            return Err(js_err("mu_test must be finite"));
        }

        let start_t = web_time::Instant::now();

        let (ws, model) = parse_workspace(workspace_json)?;
        let measurement_name = ws.measurements.first().map(|m| m.name.clone());

        let mle = MaximumLikelihoodEstimator::new();
        let ctx = AsymptoticCLsContext::new(&mle, &model).map_err(|e| js_err(e.to_string()))?;
        let r = ctx.hypotest_qtilde(&mle, mu_test).map_err(|e| js_err(e.to_string()))?;

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = HypotestResultJs {
            measurement_name,
            poi_name: poi_name(&model),
            mu_test: r.mu_test,
            cls: r.cls,
            clsb: r.clsb,
            clb: r.clb,
            q_mu: r.q_mu,
            q_mu_a: r.q_mu_a,
            mu_hat: r.mu_hat,
            elapsed_ms,
        };

        serde_wasm_bindgen::to_value(&out).map_err(|e| js_err(e.to_string()))
    }
}
