#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use ns_inference::profile_likelihood::scan_histfactory;
    use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
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

    fn parse_workspace(json: &str) -> Result<(Option<String>, HistFactoryModel), JsValue> {
        guard_workspace(json)?;

        let format = ns_translate::hs3::detect::detect_format(json);
        let (measurement_name, model) = match format {
            ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
                let model = ns_translate::hs3::convert::from_hs3_default(json)
                    .map_err(|e| js_err(format!("Failed to load HS3: {e}")))?;
                (None, model)
            }
            ns_translate::hs3::detect::WorkspaceFormat::Pyhf
            | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
                let ws: Workspace = serde_json::from_str(json).map_err(|e| {
                    js_err(format!("Failed to parse workspace as pyhf JSON: {}", e))
                })?;
                let measurement_name = ws.measurements.first().map(|m| m.name.clone());
                let model =
                    HistFactoryModel::from_workspace(&ws).map_err(|e| js_err(e.to_string()))?;
                (measurement_name, model)
            }
        };

        guard_model(&model)?;
        Ok((measurement_name, model))
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

    #[derive(Debug, Clone)]
    struct IngestConfig {
        poi: String,
        observations: Option<std::collections::HashMap<String, Vec<f64>>>,
        measurement_name: String,
    }

    impl Default for IngestConfig {
        fn default() -> Self {
            Self {
                poi: "mu".to_string(),
                observations: None,
                measurement_name: "NormalMeasurement".to_string(),
            }
        }
    }

    fn ingest_config_from_options(options: JsValue) -> Result<IngestConfig, JsValue> {
        let opts: HistogramIngestOptions = if options.is_null() || options.is_undefined() {
            HistogramIngestOptions {
                poi: Some("mu".to_string()),
                observations: None,
                measurement_name: None,
            }
        } else {
            serde_wasm_bindgen::from_value(options).map_err(|e| js_err(e.to_string()))?
        };

        let mut cfg = IngestConfig::default();
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
        cfg: &IngestConfig,
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

        let (measurement_name, model) = parse_workspace(workspace_json)?;

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

        let (measurement_name, model) = parse_workspace(workspace_json)?;

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

        let (measurement_name, model) = parse_workspace(workspace_json)?;

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

    // ── 5. GLM Regression ─────────────────────────────────────────────────

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct GlmInput {
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        #[serde(default = "default_true")]
        include_intercept: bool,
        #[serde(default)]
        model: String, // "linear", "logistic", "poisson"
    }

    fn default_true() -> bool {
        true
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct GlmResultJs {
        model: String,
        parameter_names: Vec<String>,
        parameters: Vec<f64>,
        nll: f64,
        converged: bool,
        n_iter: u64,
        n_obs: usize,
        n_features: usize,
        elapsed_ms: f64,
    }

    use ns_core::traits::LogDensityModel;

    #[wasm_bindgen]
    pub fn run_glm(input_json: &str) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let start_t = web_time::Instant::now();

        let input: GlmInput =
            serde_json::from_str(input_json).map_err(|e| js_err(e.to_string()))?;

        if input.x.is_empty() {
            return Err(js_err("X must be non-empty"));
        }
        let n_obs = input.x.len();
        let n_features = input.x[0].len();
        if input.y.len() != n_obs {
            return Err(js_err(format!(
                "y length ({}) must match X rows ({})",
                input.y.len(),
                n_obs
            )));
        }

        let model_type = if input.model.is_empty() { "linear" } else { input.model.as_str() };

        let cfg =
            ns_inference::OptimizerConfig { max_iter: 500, tol: 1e-8, m: 0, smooth_bounds: false };
        let opt = ns_inference::LbfgsbOptimizer::new(cfg);

        let (param_names, result) = match model_type {
            "linear" => {
                let m = ns_inference::LinearRegressionModel::new(
                    input.x,
                    input.y,
                    input.include_intercept,
                )
                .map_err(|e| js_err(e.to_string()))?;
                let names = m.parameter_names();
                let init = m.parameter_init();
                let bounds = m.parameter_bounds();
                let obj = GlmObjective(&m);
                let res = opt.minimize(&obj, &init, &bounds).map_err(|e| js_err(e.to_string()))?;
                (names, res)
            }
            "logistic" => {
                let y_u8: Vec<u8> = input
                    .y
                    .iter()
                    .map(|&v| {
                        if v == 0.0 {
                            Ok(0u8)
                        } else if v == 1.0 {
                            Ok(1u8)
                        } else {
                            Err(js_err(format!("logistic y must be 0/1, got {v}")))
                        }
                    })
                    .collect::<Result<Vec<u8>, JsValue>>()?;
                let m = ns_inference::LogisticRegressionModel::new(
                    input.x,
                    y_u8,
                    input.include_intercept,
                )
                .map_err(|e| js_err(e.to_string()))?;
                let names = m.parameter_names();
                let init = m.parameter_init();
                let bounds = m.parameter_bounds();
                let obj = GlmObjective(&m);
                let res = opt.minimize(&obj, &init, &bounds).map_err(|e| js_err(e.to_string()))?;
                (names, res)
            }
            "poisson" => {
                let y_u64: Vec<u64> = input
                    .y
                    .iter()
                    .map(|&v| {
                        if v >= 0.0 && v == v.floor() {
                            Ok(v as u64)
                        } else {
                            Err(js_err(format!("poisson y must be non-negative integers, got {v}")))
                        }
                    })
                    .collect::<Result<Vec<u64>, JsValue>>()?;
                let m = ns_inference::PoissonRegressionModel::new(
                    input.x,
                    y_u64,
                    input.include_intercept,
                    None,
                )
                .map_err(|e| js_err(e.to_string()))?;
                let names = m.parameter_names();
                let init = m.parameter_init();
                let bounds = m.parameter_bounds();
                let obj = GlmObjective(&m);
                let res = opt.minimize(&obj, &init, &bounds).map_err(|e| js_err(e.to_string()))?;
                (names, res)
            }
            other => {
                return Err(js_err(format!(
                    "Unknown model type: {other}. Use linear, logistic, or poisson."
                )));
            }
        };

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let out = GlmResultJs {
            model: model_type.to_string(),
            parameter_names: param_names,
            parameters: result.parameters,
            nll: result.fval,
            converged: result.converged,
            n_iter: result.n_iter,
            n_obs,
            n_features,
            elapsed_ms,
        };

        serde_wasm_bindgen::to_value(&out).map_err(|e| js_err(e.to_string()))
    }

    struct GlmObjective<'a, M: LogDensityModel>(&'a M);

    impl<M: LogDensityModel> ns_inference::ObjectiveFunction for GlmObjective<'_, M> {
        fn eval(&self, params: &[f64]) -> ns_core::Result<f64> {
            self.0.nll(params)
        }

        fn gradient(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            self.0.grad_nll(params)
        }
    }

    // ── 4. Single-point Hypothesis Test ──────────────────────────────────

    #[wasm_bindgen]
    pub fn run_hypotest(workspace_json: &str, mu_test: f64) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();

        if !mu_test.is_finite() {
            return Err(js_err("mu_test must be finite"));
        }

        let start_t = web_time::Instant::now();

        let (measurement_name, model) = parse_workspace(workspace_json)?;

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

    // ── 6. Bayesian Sampling (NUTS / MAMS) ───────────────────────────────

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct BayesSampleInput {
        model: String,
        #[serde(default)]
        model_data: serde_json::Value,
        #[serde(default = "default_sampler")]
        sampler: String,
        #[serde(default = "default_n_warmup")]
        n_warmup: usize,
        #[serde(default = "default_n_samples")]
        n_samples: usize,
        #[serde(default = "default_seed")]
        seed: u64,
        #[serde(default = "default_target_accept_nuts")]
        target_accept: f64,
        #[serde(default = "default_max_treedepth")]
        max_treedepth: usize,
    }

    fn default_sampler() -> String {
        "nuts".into()
    }
    fn default_n_warmup() -> usize {
        500
    }
    fn default_n_samples() -> usize {
        1000
    }
    fn default_seed() -> u64 {
        42
    }
    fn default_target_accept_nuts() -> f64 {
        0.8
    }
    fn default_max_treedepth() -> usize {
        10
    }

    fn validate_bayes_input(input: &BayesSampleInput) -> Result<(), JsValue> {
        if input.n_samples == 0 {
            return Err(js_err("Number of samples must be at least 1."));
        }
        if input.n_warmup == 0 {
            return Err(js_err("Number of warmup iterations must be at least 1."));
        }
        if !input.target_accept.is_finite()
            || input.target_accept <= 0.0
            || input.target_accept >= 1.0
        {
            return Err(js_err(
                "Target acceptance rate must be between 0 and 1 (exclusive). Use a dot as decimal separator (e.g. 0.8, not 0,8).",
            ));
        }
        if input.max_treedepth == 0 || input.max_treedepth > 15 {
            return Err(js_err("Max tree depth must be between 1 and 15 in the browser."));
        }
        Ok(())
    }

    fn zscore_columns(x: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, JsValue> {
        if x.is_empty() {
            return Err(js_err("glm_logistic: x must have at least 1 row"));
        }
        let p = x[0].len();
        if p == 0 {
            return Err(js_err("glm_logistic: x must have at least 1 feature"));
        }
        if x.iter().any(|r| r.len() != p) {
            return Err(js_err("glm_logistic: all x rows must have the same length"));
        }
        if x.iter().flatten().any(|v| !v.is_finite()) {
            return Err(js_err("glm_logistic: x must contain only finite values"));
        }

        let n = x.len();
        let mut means = vec![0.0f64; p];
        for row in &x {
            for (j, v) in row.iter().enumerate() {
                means[j] += *v;
            }
        }
        for m in &mut means {
            *m /= n as f64;
        }

        let mut stds = vec![0.0f64; p];
        for row in &x {
            for (j, v) in row.iter().enumerate() {
                let d = *v - means[j];
                stds[j] += d * d;
            }
        }
        for s in &mut stds {
            *s = (*s / n as f64).sqrt();
            if !s.is_finite() || *s < 1e-12 {
                *s = 1.0;
            }
        }

        let x_std: Vec<Vec<f64>> = x
            .into_iter()
            .map(|row| row.into_iter().enumerate().map(|(j, v)| (v - means[j]) / stds[j]).collect())
            .collect();
        Ok(x_std)
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct BayesChainJs {
        draws: Vec<Vec<f64>>,
        divergences: Vec<bool>,
        tree_depths: Vec<usize>,
        accept_probs: Vec<f64>,
        n_leapfrog: Vec<usize>,
        step_size: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct BayesDiagnosticsJs {
        r_hat: Vec<f64>,
        ess_bulk: Vec<f64>,
        ess_tail: Vec<f64>,
        divergence_rate: f64,
        max_treedepth_rate: f64,
        ebfmi: Vec<f64>,
        quality_status: String,
        warnings: Vec<String>,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct BayesSampleResultJs {
        sampler: String,
        model: String,
        param_names: Vec<String>,
        n_warmup: usize,
        n_samples: usize,
        n_chains: usize,
        chains: Vec<BayesChainJs>,
        diagnostics: BayesDiagnosticsJs,
        elapsed_ms: f64,
    }

    const MAX_BAYES_SAMPLES: usize = 50_000;
    const MAX_BAYES_DIM: usize = 200;

    fn dispatch_sample(input: &BayesSampleInput) -> Result<BayesSampleResultJs, JsValue> {
        use ns_inference::diagnostics::{QualityGates, compute_diagnostics, quality_summary};

        let start_t = web_time::Instant::now();

        validate_bayes_input(input)?;

        if input.n_samples > MAX_BAYES_SAMPLES {
            return Err(js_err(format!(
                "Too many samples for browser runtime: {} requested, maximum is {}. Try a smaller value.",
                input.n_samples, MAX_BAYES_SAMPLES
            )));
        }
        if input.n_warmup > MAX_BAYES_SAMPLES {
            return Err(js_err(format!(
                "Too many warmup iterations for browser runtime: {} requested, maximum is {}. Try a smaller value.",
                input.n_warmup, MAX_BAYES_SAMPLES
            )));
        }

        macro_rules! run_sampler {
            ($model:expr, $input:expr) => {{
                let model = $model;
                let dim = ns_core::traits::LogDensityModel::dim(&model);
                if dim > MAX_BAYES_DIM {
                    return Err(js_err(format!(
                        "Model has {} parameters, which exceeds the browser limit of {}.",
                        dim, MAX_BAYES_DIM
                    )));
                }
                let param_names = ns_core::traits::LogDensityModel::parameter_names(&model);

                let chain = match $input.sampler.as_str() {
                    "nuts" => {
                        let config = ns_inference::nuts::NutsConfig {
                            max_treedepth: $input.max_treedepth,
                            target_accept: $input.target_accept,
                            ..Default::default()
                        };
                        ns_inference::nuts::sample_nuts(
                            &model,
                            $input.n_warmup,
                            $input.n_samples,
                            $input.seed,
                            config,
                        )
                        .map_err(|e| js_err(e.to_string()))?
                    }
                    "mams" => {
                        let config = ns_inference::mams::MamsConfig {
                            n_warmup: $input.n_warmup,
                            n_samples: $input.n_samples,
                            target_accept: $input.target_accept,
                            ..Default::default()
                        };
                        ns_inference::mams::sample_mams(&model, config, $input.seed)
                            .map_err(|e| js_err(e.to_string()))?
                    }
                    other => {
                        return Err(js_err(format!(
                            "Unknown sampler: {}. Use 'nuts' or 'mams'.",
                            other
                        )));
                    }
                };

                let sampler_result = ns_inference::chain::SamplerResult {
                    chains: vec![chain],
                    param_names: param_names.clone(),
                    n_warmup: $input.n_warmup,
                    n_samples: $input.n_samples,
                    diagnostics: None,
                };

                let diag = compute_diagnostics(&sampler_result);
                let quality = quality_summary(&diag, 1, $input.n_samples, &QualityGates::default());

                let chains_js: Vec<BayesChainJs> = sampler_result
                    .chains
                    .iter()
                    .map(|c| BayesChainJs {
                        draws: c.draws_constrained.clone(),
                        divergences: c.divergences.clone(),
                        tree_depths: c.tree_depths.clone(),
                        accept_probs: c.accept_probs.clone(),
                        n_leapfrog: c.n_leapfrog.clone(),
                        step_size: c.step_size,
                    })
                    .collect();

                let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

                BayesSampleResultJs {
                    sampler: $input.sampler.clone(),
                    model: $input.model.clone(),
                    param_names,
                    n_warmup: $input.n_warmup,
                    n_samples: $input.n_samples,
                    n_chains: 1,
                    chains: chains_js,
                    diagnostics: BayesDiagnosticsJs {
                        r_hat: diag.r_hat,
                        ess_bulk: diag.ess_bulk,
                        ess_tail: diag.ess_tail,
                        divergence_rate: diag.divergence_rate,
                        max_treedepth_rate: diag.max_treedepth_rate,
                        ebfmi: diag.ebfmi,
                        quality_status: quality.status.to_string(),
                        warnings: quality.warnings.into_iter().chain(quality.failures).collect(),
                    },
                    elapsed_ms,
                }
            }};
        }

        let result = match input.model.as_str() {
            "eight_schools" => {
                let y: Vec<f64> = if input.model_data.get("y").is_some() {
                    serde_json::from_value(input.model_data["y"].clone())
                        .map_err(|e| js_err(format!("eight_schools: bad y: {e}")))?
                } else {
                    vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
                };
                let sigma: Vec<f64> = if input.model_data.get("sigma").is_some() {
                    serde_json::from_value(input.model_data["sigma"].clone())
                        .map_err(|e| js_err(format!("eight_schools: bad sigma: {e}")))?
                } else {
                    vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
                };
                let model = ns_inference::EightSchoolsModel::new(y, sigma, 5.0, 5.0)
                    .map_err(|e| js_err(e.to_string()))?;
                run_sampler!(model, input)
            }
            "glm_logistic" => {
                #[derive(Deserialize)]
                struct GlmData {
                    x: Vec<Vec<f64>>,
                    y: Vec<f64>,
                }
                let d: GlmData = serde_json::from_value(input.model_data.clone())
                    .map_err(|e| js_err(format!("glm_logistic: {e}")))?;
                let x_std = zscore_columns(d.x)?;
                let y_u8: Vec<u8> = d.y.iter().map(|&v| if v == 1.0 { 1u8 } else { 0u8 }).collect();
                let n_pos = y_u8.iter().copied().filter(|&v| v == 1).count();
                if n_pos == 0 || n_pos == y_u8.len() {
                    return Err(js_err(
                        "glm_logistic: y must contain both classes (0 and 1) for Bayesian sampling",
                    ));
                }
                let model = ns_inference::ModelBuilder::logistic_regression(x_std, y_u8, true)
                    .map_err(|e| js_err(e.to_string()))?
                    // Bayesian demo path: use a practical weakly-informative prior
                    // and penalize intercept to avoid runaway trajectories on near-separable data.
                    .with_coef_prior_normal(0.0, 2.5)
                    .map_err(|e| js_err(e.to_string()))?
                    .with_penalize_intercept(true)
                    .build()
                    .map_err(|e| js_err(e.to_string()))?;
                run_sampler!(model, input)
            }
            "glm_linear" => {
                #[derive(Deserialize)]
                struct GlmData {
                    x: Vec<Vec<f64>>,
                    y: Vec<f64>,
                }
                let d: GlmData = serde_json::from_value(input.model_data.clone())
                    .map_err(|e| js_err(format!("glm_linear: {e}")))?;
                let model = ns_inference::ModelBuilder::linear_regression(d.x, d.y, true)
                    .map_err(|e| js_err(e.to_string()))?
                    .build()
                    .map_err(|e| js_err(e.to_string()))?;
                run_sampler!(model, input)
            }
            "glm_poisson" => {
                #[derive(Deserialize)]
                struct GlmData {
                    x: Vec<Vec<f64>>,
                    y: Vec<f64>,
                }
                let d: GlmData = serde_json::from_value(input.model_data.clone())
                    .map_err(|e| js_err(format!("glm_poisson: {e}")))?;
                let y_u64: Vec<u64> = d.y.iter().map(|&v| v as u64).collect();
                let model = ns_inference::ModelBuilder::poisson_regression(d.x, y_u64, true, None)
                    .map_err(|e| js_err(e.to_string()))?
                    .build()
                    .map_err(|e| js_err(e.to_string()))?;
                run_sampler!(model, input)
            }
            "std_normal" => {
                let dim: usize = input
                    .model_data
                    .get("dim")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(2);
                let model = StdNormalModel { dim };
                run_sampler!(model, input)
            }
            other => {
                return Err(js_err(format!(
                    "Unknown model: {}. Available: eight_schools, glm_logistic, glm_linear, glm_poisson, std_normal.",
                    other
                )));
            }
        };

        Ok(result)
    }

    struct StdNormalModel {
        dim: usize,
    }

    impl ns_core::traits::LogDensityModel for StdNormalModel {
        type Prepared<'a>
            = ns_core::traits::PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            self.dim
        }

        fn parameter_names(&self) -> Vec<String> {
            (0..self.dim).map(|i| format!("x[{}]", i)).collect()
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0; self.dim]
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            Ok(0.5 * params.iter().map(|x| x * x).sum::<f64>())
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            Ok(params.to_vec())
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            ns_core::traits::PreparedModelRef::new(self)
        }
    }

    #[wasm_bindgen]
    pub fn run_bayes_sample(input_json: &str) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let input: BayesSampleInput =
            serde_json::from_str(input_json).map_err(|e| js_err(e.to_string()))?;
        let result = dispatch_sample(&input)?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }

    // ── 7. Population PK Tutorial (FOCE / SAEM) ─────────────────────────

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct PopPkInput {
        #[serde(default = "default_pop_pk_estimator")]
        estimator: String,
        #[serde(default = "default_pop_pk_n_subjects")]
        n_subjects: usize,
        #[serde(default = "default_pop_pk_n_obs")]
        n_obs_per_subject: usize,
        #[serde(default = "default_pop_pk_dose")]
        dose: f64,
        #[serde(default = "default_pop_pk_bioav")]
        bioav: f64,
        #[serde(default = "default_pop_pk_theta")]
        theta_true: Vec<f64>,
        #[serde(default = "default_pop_pk_omega")]
        omega_true: Vec<f64>,
        #[serde(default = "default_pop_pk_sigma")]
        sigma: f64,
        #[serde(default = "default_pop_pk_seed")]
        seed: u64,
        #[serde(default = "default_pop_pk_max_outer")]
        max_outer_iter: usize,
        #[serde(default = "default_pop_pk_max_inner")]
        max_inner_iter: usize,
        #[serde(default = "default_pop_pk_n_burn")]
        n_burn: usize,
        #[serde(default = "default_pop_pk_n_iter")]
        n_iter: usize,
    }

    fn default_pop_pk_estimator() -> String {
        "foce".into()
    }
    fn default_pop_pk_n_subjects() -> usize {
        30
    }
    fn default_pop_pk_n_obs() -> usize {
        8
    }
    fn default_pop_pk_dose() -> f64 {
        100.0
    }
    fn default_pop_pk_bioav() -> f64 {
        1.0
    }
    fn default_pop_pk_theta() -> Vec<f64> {
        vec![0.15, 8.0, 1.0]
    }
    fn default_pop_pk_omega() -> Vec<f64> {
        vec![0.20, 0.15, 0.25]
    }
    fn default_pop_pk_sigma() -> f64 {
        0.3
    }
    fn default_pop_pk_seed() -> u64 {
        42
    }
    fn default_pop_pk_max_outer() -> usize {
        200
    }
    fn default_pop_pk_max_inner() -> usize {
        20
    }
    fn default_pop_pk_n_burn() -> usize {
        400
    }
    fn default_pop_pk_n_iter() -> usize {
        200
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct PopPkSubjectJs {
        id: usize,
        times: Vec<f64>,
        y_obs: Vec<f64>,
        y_pred: Vec<f64>,
        y_pop: Vec<f64>,
        eta: Vec<f64>,
    }

    #[derive(Debug, Clone, Serialize)]
    #[serde(rename_all = "camelCase")]
    struct PopPkResultJs {
        estimator: String,
        theta_true: Vec<f64>,
        omega_true: Vec<f64>,
        sigma_true: f64,
        theta_est: Vec<f64>,
        omega_est: Vec<f64>,
        correlation: Vec<Vec<f64>>,
        ofv: f64,
        converged: bool,
        n_iter: usize,
        n_subjects: usize,
        n_obs_per_subject: usize,
        dose: f64,
        subjects: Vec<PopPkSubjectJs>,
        pop_curve_times: Vec<f64>,
        pop_curve_conc: Vec<f64>,
        elapsed_ms: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        ofv_trace: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        acceptance_rates: Option<Vec<f64>>,
    }

    const MAX_PK_SUBJECTS: usize = 100;
    const MAX_PK_OBS: usize = 24;

    fn generate_pop_pk_data(
        input: &PopPkInput,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>), JsValue> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_distr::StandardNormal;

        let mut rng = rand::rngs::StdRng::seed_from_u64(input.seed);
        let theta = &input.theta_true;
        let omega = &input.omega_true;

        let sample_times: Vec<f64> = {
            let n = input.n_obs_per_subject;
            let t_max = 24.0_f64;
            (0..n)
                .map(|i| {
                    let base = t_max * (i as f64 + 0.5) / n as f64;
                    base.max(0.25)
                })
                .collect()
        };

        let mut all_times = Vec::new();
        let mut all_y = Vec::new();
        let mut all_subj = Vec::new();
        let mut subj_times = Vec::new();
        let mut subj_y = Vec::new();

        for s in 0..input.n_subjects {
            let eta_cl: f64 = omega[0] * rng.sample::<f64, _>(StandardNormal);
            let eta_v: f64 = omega[1] * rng.sample::<f64, _>(StandardNormal);
            let eta_ka: f64 = omega[2] * rng.sample::<f64, _>(StandardNormal);

            let cl_i = theta[0] * eta_cl.exp();
            let v_i = theta[1] * eta_v.exp();
            let ka_i = theta[2] * eta_ka.exp();

            let mut st = Vec::with_capacity(sample_times.len());
            let mut sy = Vec::with_capacity(sample_times.len());

            for &t in &sample_times {
                let c = ns_inference::pk::conc_oral(input.dose, input.bioav, cl_i, v_i, ka_i, t);
                let noise: f64 = input.sigma * rng.sample::<f64, _>(StandardNormal);
                let y_obs = (c + noise).max(0.01);

                all_times.push(t);
                all_y.push(y_obs);
                all_subj.push(s);
                st.push(t);
                sy.push(y_obs);
            }
            subj_times.push(st);
            subj_y.push(sy);
        }

        Ok((all_times, all_y, all_subj, subj_times, subj_y))
    }

    #[wasm_bindgen]
    pub fn run_pop_pk(input_json: &str) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let input: PopPkInput =
            serde_json::from_str(input_json).map_err(|e| js_err(e.to_string()))?;

        if input.n_subjects > MAX_PK_SUBJECTS {
            return Err(js_err(format!("n_subjects max {} in browser", MAX_PK_SUBJECTS)));
        }
        if input.n_obs_per_subject > MAX_PK_OBS {
            return Err(js_err(format!("n_obs_per_subject max {} in browser", MAX_PK_OBS)));
        }
        if input.theta_true.len() != 3 {
            return Err(js_err("theta_true must have 3 elements [CL, V, Ka]"));
        }
        if input.omega_true.len() != 3 {
            return Err(js_err("omega_true must have 3 elements [ω_CL, ω_V, ω_Ka]"));
        }

        let start_t = web_time::Instant::now();

        let (times, y, subject_idx, subj_times, subj_y) = generate_pop_pk_data(&input)?;

        let error_model = ns_inference::pk::ErrorModel::Additive(input.sigma);
        let theta_init = input.theta_true.iter().map(|&v| v * 1.1).collect::<Vec<_>>();
        let omega_init = input.omega_true.iter().map(|&v| v * 1.1).collect::<Vec<_>>();

        let (
            theta_est,
            omega_est,
            correlation,
            eta,
            ofv,
            converged,
            n_iter,
            ofv_trace,
            acceptance_rates,
        ) = match input.estimator.as_str() {
            "foce" => {
                let cfg = ns_inference::foce::FoceConfig {
                    max_outer_iter: input.max_outer_iter,
                    max_inner_iter: input.max_inner_iter,
                    tol: 1e-3,
                    interaction: true,
                };
                let est = ns_inference::foce::FoceEstimator::new(cfg);
                let r = est
                    .fit_1cpt_oral(
                        &times,
                        &y,
                        &subject_idx,
                        input.n_subjects,
                        input.dose,
                        input.bioav,
                        error_model,
                        &theta_init,
                        &omega_init,
                    )
                    .map_err(|e| js_err(e.to_string()))?;
                (r.theta, r.omega, r.correlation, r.eta, r.ofv, r.converged, r.n_iter, None, None)
            }
            "saem" => {
                let cfg = ns_inference::saem::SaemConfig {
                    n_burn: input.n_burn,
                    n_iter: input.n_iter,
                    seed: input.seed + 1000,
                    tol: 1e-3,
                    ..Default::default()
                };
                let est = ns_inference::saem::SaemEstimator::new(cfg);
                let (r, diag) = est
                    .fit_1cpt_oral(
                        &times,
                        &y,
                        &subject_idx,
                        input.n_subjects,
                        input.dose,
                        input.bioav,
                        error_model,
                        &theta_init,
                        &omega_init,
                    )
                    .map_err(|e| js_err(e.to_string()))?;
                (
                    r.theta,
                    r.omega,
                    r.correlation,
                    r.eta,
                    r.ofv,
                    r.converged,
                    r.n_iter,
                    Some(diag.ofv_trace),
                    Some(diag.acceptance_rates),
                )
            }
            other => {
                return Err(js_err(format!("Unknown estimator: {}. Use 'foce' or 'saem'.", other)));
            }
        };

        let subjects: Vec<PopPkSubjectJs> = (0..input.n_subjects)
            .map(|s| {
                let st = &subj_times[s];
                let sy = &subj_y[s];

                let y_pred: Vec<f64> = st
                    .iter()
                    .map(|&t| {
                        let cl_i = theta_est[0] * eta[s][0].exp();
                        let v_i = theta_est[1] * eta[s][1].exp();
                        let ka_i = theta_est[2] * eta[s][2].exp();
                        ns_inference::pk::conc_oral(input.dose, input.bioav, cl_i, v_i, ka_i, t)
                    })
                    .collect();

                let y_pop: Vec<f64> = st
                    .iter()
                    .map(|&t| {
                        ns_inference::pk::conc_oral(
                            input.dose,
                            input.bioav,
                            theta_est[0],
                            theta_est[1],
                            theta_est[2],
                            t,
                        )
                    })
                    .collect();

                PopPkSubjectJs {
                    id: s,
                    times: st.clone(),
                    y_obs: sy.clone(),
                    y_pred,
                    y_pop,
                    eta: eta[s].clone(),
                }
            })
            .collect();

        let pop_curve_times: Vec<f64> = (0..100).map(|i| 0.25 + 23.75 * i as f64 / 99.0).collect();
        let pop_curve_conc: Vec<f64> = pop_curve_times
            .iter()
            .map(|&t| {
                ns_inference::pk::conc_oral(
                    input.dose,
                    input.bioav,
                    theta_est[0],
                    theta_est[1],
                    theta_est[2],
                    t,
                )
            })
            .collect();

        let elapsed_ms = start_t.elapsed().as_secs_f64() * 1000.0;

        let result = PopPkResultJs {
            estimator: input.estimator.clone(),
            theta_true: input.theta_true.clone(),
            omega_true: input.omega_true.clone(),
            sigma_true: input.sigma,
            theta_est,
            omega_est,
            correlation,
            ofv,
            converged,
            n_iter,
            n_subjects: input.n_subjects,
            n_obs_per_subject: input.n_obs_per_subject,
            dose: input.dose,
            subjects,
            pop_curve_times,
            pop_curve_conc,
            elapsed_ms,
            ofv_trace,
            acceptance_rates,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }
}
