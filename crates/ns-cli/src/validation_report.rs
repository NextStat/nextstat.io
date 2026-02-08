use anyhow::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::InterpDefaults;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn read_json_value(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn load_model_from_workspace_bytes(
    bytes: &[u8],
    interp_defaults: InterpDefaults,
) -> Result<ns_translate::pyhf::HistFactoryModel> {
    let json_str = std::str::from_utf8(bytes)?;
    let format = ns_translate::hs3::detect::detect_format(json_str);
    let model = match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
            ns_translate::hs3::convert::from_hs3_default(json_str)?
        }
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(json_str)?;
            match interp_defaults {
                InterpDefaults::Pyhf => ns_translate::pyhf::HistFactoryModel::from_workspace_with_settings(
                    &workspace,
                    ns_translate::pyhf::NormSysInterpCode::Code1,
                    ns_translate::pyhf::HistoSysInterpCode::Code0,
                )?,
                InterpDefaults::Root => ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?,
            }
        }
    };
    Ok(model)
}

fn constraint_kind(p: &ns_translate::pyhf::Parameter) -> &'static str {
    // Treat a fixed bound as "fixed" regardless of constraint metadata.
    if (p.bounds.0 - p.bounds.1).abs() < 1e-12 {
        return "fixed";
    }
    match &p.constraint_term {
        Some(ns_translate::pyhf::ConstraintTerm::Uniform) => "uniform",
        Some(ns_translate::pyhf::ConstraintTerm::LogNormal { .. }) => "lognormal",
        Some(ns_translate::pyhf::ConstraintTerm::Gamma { .. }) => "gamma",
        Some(ns_translate::pyhf::ConstraintTerm::NoConstraint) => "fixed",
        None => {
            if p.constrained {
                "normal"
            } else {
                "none"
            }
        }
    }
}

fn suite_status(master: &Value, key: &str) -> Option<String> {
    master
        .get(key)?
        .get("status")?
        .as_str()
        .map(|s| s.to_string())
}

fn map_push_highlight(out: &mut serde_json::Map<String, Value>, msg: String) {
    let v = out
        .entry("highlights".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(arr) = v {
        arr.push(Value::String(msg));
    }
}

fn map_push_worst_case(
    out: &mut serde_json::Map<String, Value>,
    case_name: String,
    metric: &'static str,
    value: f64,
    notes: Option<String>,
) {
    let v = out
        .entry("worst_cases".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(arr) = v {
        let mut obj = serde_json::Map::new();
        obj.insert("case".to_string(), Value::String(case_name));
        obj.insert("metric".to_string(), Value::String(metric.to_string()));
        obj.insert("value".to_string(), serde_json::json!(value));
        if let Some(n) = notes {
            obj.insert("notes".to_string(), Value::String(n));
        }
        arr.push(Value::Object(obj));
    }
}

fn suite_summary(master: &Value, key: &str) -> Option<Value> {
    let status = suite_status(master, key)?;
    let mut out = serde_json::Map::new();
    out.insert("status".to_string(), Value::String(status));

    match key {
        "pyhf" => {
            if let Some(cases) = master
                .get("pyhf")
                .and_then(|v| v.get("report"))
                .and_then(|v| v.get("cases"))
                .and_then(|v| v.as_array())
            {
                out.insert("n_cases".to_string(), Value::Number((cases.len() as u64).into()));
                let mut worst_nll = 0.0f64;
                let mut worst_exp_full = 0.0f64;
                let mut by_nll: Vec<(f64, String, Option<String>)> = Vec::new();
                let mut by_exp: Vec<(f64, String, Option<String>)> = Vec::new();
                for c in cases {
                    let name = c
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let d = c
                        .get("parity")
                        .and_then(|v| v.get("max_abs_delta_nll"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    worst_nll = worst_nll.max(d);
                    let allowed = c
                        .get("parity")
                        .and_then(|v| v.get("nll_allowed"))
                        .and_then(|v| v.as_f64());
                    let notes = allowed.map(|a| format!("allowed={:.3e}", a));
                    by_nll.push((d, name.clone(), notes));

                    let de = c
                        .get("parity")
                        .and_then(|v| v.get("max_abs_delta_expected_full"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    worst_exp_full = worst_exp_full.max(de);
                    by_exp.push((de, name, None));
                }
                out.insert("worst_delta_nll".to_string(), serde_json::json!(worst_nll));
                out.insert(
                    "worst_delta_expected_full".to_string(),
                    serde_json::json!(worst_exp_full),
                );

                map_push_highlight(&mut out, format!("worst |dNLL| = {:.3e}", worst_nll));
                map_push_highlight(
                    &mut out,
                    format!("worst |d expected(full)| = {:.3e}", worst_exp_full),
                );

                by_nll.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (v, name, notes) in by_nll.into_iter().take(5) {
                    map_push_worst_case(&mut out, name, "max_abs_delta_nll", v, notes);
                }
                by_exp.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (v, name, _) in by_exp.into_iter().take(5) {
                    map_push_worst_case(&mut out, name, "max_abs_delta_expected_full", v, None);
                }
            }
        }
        "regression_golden" => {
            if let Some(cases) = master
                .get("regression_golden")
                .and_then(|v| v.get("cases"))
                .and_then(|v| v.as_array())
            {
                out.insert("n_cases".to_string(), Value::Number((cases.len() as u64).into()));
                let n_ok = cases
                    .iter()
                    .filter(|c| c.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count();
                out.insert("n_ok".to_string(), Value::Number((n_ok as u64).into()));
                map_push_highlight(
                    &mut out,
                    format!("cases ok: {}/{}", n_ok, cases.len()),
                );
            }
        }
        "nuts_quality" => {
            if let Some(s) = master.get("nuts_quality") {
                let qs = s
                    .get("quality_status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let d = s.get("diagnostics").unwrap_or(&Value::Null);
                let div = d.get("divergence_rate").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let rhat = d.get("max_r_hat").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let td = d.get("max_treedepth_rate").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let essb = d.get("min_ess_bulk").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let esst = d.get("min_ess_tail").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let ebfmi = d.get("min_ebfmi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);

                map_push_highlight(&mut out, format!("quality status: {}", qs));
                map_push_highlight(
                    &mut out,
                    format!(
                        "smoke diagnostics: div={:.3}, treedepth={:.3}, r_hat={:.3}, ess_bulk={:.1}, ess_tail={:.1}, ebfmi={:.3}",
                        div, td, rhat, essb, esst, ebfmi
                    ),
                );

                let case = "nuts_quality_smoke".to_string();
                map_push_worst_case(&mut out, case.clone(), "divergence_rate", div, None);
                map_push_worst_case(&mut out, case.clone(), "max_treedepth_rate", td, None);
                map_push_worst_case(&mut out, case.clone(), "max_r_hat", rhat, None);
                map_push_worst_case(&mut out, case.clone(), "min_ess_bulk", essb, None);
                map_push_worst_case(&mut out, case.clone(), "min_ess_tail", esst, None);
                map_push_worst_case(&mut out, case, "min_ebfmi", ebfmi, None);
            }
        }
        "nuts_quality_report" => {
            if let Some(rep) = master
                .get("nuts_quality_report")
                .and_then(|v| v.get("report"))
                .and_then(|v| v.as_object())
            {
                let cases = rep.get("cases").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                if !cases.is_empty() {
                    out.insert("n_cases".to_string(), Value::Number((cases.len() as u64).into()));
                    let n_ok = cases
                        .iter()
                        .filter(|c| c.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
                        .count();
                    out.insert("n_ok".to_string(), Value::Number((n_ok as u64).into()));

                    // Worst-case extraction across cases.
                    let mut max_rhat = (0.0f64, "unknown".to_string());
                    let mut max_div = (0.0f64, "unknown".to_string());
                    let mut max_td = (0.0f64, "unknown".to_string());
                    let mut min_ebfmi = (f64::INFINITY, "unknown".to_string());
                    let mut min_essb = (f64::INFINITY, "unknown".to_string());
                    let mut min_esst = (f64::INFINITY, "unknown".to_string());
                    for c in &cases {
                        let name = c.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                        let s = c.get("summary").unwrap_or(&Value::Null);
                        let rhat = s.get("max_r_hat").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        let div = s.get("divergence_rate").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        let td = s.get("max_treedepth_rate").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        let ebfmi = s.get("min_ebfmi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        let essb = s.get("min_ess_bulk").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        let esst = s.get("min_ess_tail").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                        if rhat.is_finite() && rhat > max_rhat.0 {
                            max_rhat = (rhat, name.clone());
                        }
                        if div.is_finite() && div > max_div.0 {
                            max_div = (div, name.clone());
                        }
                        if td.is_finite() && td > max_td.0 {
                            max_td = (td, name.clone());
                        }
                        if ebfmi.is_finite() && ebfmi < min_ebfmi.0 {
                            min_ebfmi = (ebfmi, name.clone());
                        }
                        if essb.is_finite() && essb < min_essb.0 {
                            min_essb = (essb, name.clone());
                        }
                        if esst.is_finite() && esst < min_esst.0 {
                            min_esst = (esst, name);
                        }
                    }
                    map_push_highlight(&mut out, format!("cases ok: {}/{}", n_ok, cases.len()));
                    map_push_worst_case(&mut out, max_rhat.1, "max_r_hat", max_rhat.0, None);
                    map_push_worst_case(&mut out, max_div.1, "divergence_rate", max_div.0, None);
                    map_push_worst_case(&mut out, max_td.1, "max_treedepth_rate", max_td.0, None);
                    map_push_worst_case(&mut out, min_ebfmi.1, "min_ebfmi", min_ebfmi.0, None);
                    map_push_worst_case(&mut out, min_essb.1, "min_ess_bulk", min_essb.0, None);
                    map_push_worst_case(&mut out, min_esst.1, "min_ess_tail", min_esst.0, None);
                }
            }
        }
        "root" => {
            let suite = master.get("root");
            let reason = suite.and_then(|v| v.get("reason")).and_then(|v| v.as_str());
            if let Some(r) = reason {
                map_push_highlight(&mut out, format!("reason: {}", r));
            }
            if let Some(rep) = suite
                .and_then(|v| v.get("report"))
                .and_then(|v| v.as_object())
            {
                if let Some(sum) = rep.get("summary").and_then(|v| v.as_object()) {
                    if let Some(n_cases) = sum.get("n_cases").and_then(|v| v.as_u64()) {
                        out.insert("n_cases".to_string(), Value::Number(n_cases.into()));
                    }
                    if let Some(n_ok) = sum.get("n_ok").and_then(|v| v.as_u64()) {
                        out.insert("n_ok".to_string(), Value::Number(n_ok.into()));
                    }
                    if let (Some(n_ok), Some(n_cases)) = (
                        out.get("n_ok").and_then(|v| v.as_u64()),
                        out.get("n_cases").and_then(|v| v.as_u64()),
                    ) {
                        map_push_highlight(&mut out, format!("cases ok: {}/{}", n_ok, n_cases));
                    }
                }
                if let Some(pr) = rep
                    .get("meta")
                    .and_then(|v| v.get("prereqs"))
                    .and_then(|v| v.as_object())
                {
                    let mut missing: Vec<String> = Vec::new();
                    for k in ["root", "hist2workspace", "uproot"] {
                        if let Some(ok) = pr.get(k).and_then(|v| v.as_bool())
                            && !ok
                        {
                            missing.push(k.to_string());
                        }
                    }
                    if !missing.is_empty() {
                        map_push_highlight(&mut out, format!("missing prereqs: {}", missing.join(", ")));
                    }
                }
            }
        }
        _ => {}
    }

    // Generic hint: preserve human-readable reasons for skipped suites.
    if let Some(r) = master
        .get(key)
        .and_then(|v| v.get("reason"))
        .and_then(|v| v.as_str())
    {
        map_push_highlight(&mut out, format!("reason: {}", r));
    }

    Some(Value::Object(out))
}

fn overall_status(suites: &BTreeMap<String, Value>) -> &'static str {
    for (_k, v) in suites {
        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
        if status == "fail" || status == "error" {
            return "fail";
        }
    }
    "pass"
}

pub fn cmd_validation_report(
    apex2_path: &PathBuf,
    workspace_path: &PathBuf,
    out_path: &PathBuf,
    pdf: Option<&PathBuf>,
    python: Option<&PathBuf>,
    deterministic: bool,
    interp_defaults: InterpDefaults,
) -> Result<()> {
    let apex2_bytes = std::fs::read(apex2_path)?;
    let apex2_sha256 = sha256_hex(&apex2_bytes);
    let apex2_master = serde_json::from_slice::<Value>(&apex2_bytes)?;

    let ws_bytes = std::fs::read(workspace_path)?;
    let ws_sha256 = sha256_hex(&ws_bytes);
    let ws_bytes_len = ws_bytes.len() as u64;
    let model = load_model_from_workspace_bytes(&ws_bytes, interp_defaults)?;

    let channels = model.channel_names();
    let n_channels = model.n_channels();
    let mut n_bins_per_channel: Vec<usize> = Vec::with_capacity(n_channels);
    let mut uniq_samples: HashSet<String> = HashSet::new();
    for ch in 0..n_channels {
        n_bins_per_channel.push(model.channel_bin_count(ch)?);
        for s in model.sample_names(ch) {
            uniq_samples.insert(s);
        }
    }

    let observed = model.observed_main_by_channel();
    let mut total_observed = 0.0f64;
    let mut min_bin = f64::INFINITY;
    let mut max_bin = f64::NEG_INFINITY;
    for ch in &observed {
        for &y in &ch.y {
            total_observed += y;
            min_bin = min_bin.min(y);
            max_bin = max_bin.max(y);
        }
    }
    if !min_bin.is_finite() {
        min_bin = 0.0;
    }
    if !max_bin.is_finite() {
        max_bin = 0.0;
    }

    let poi = model
        .poi_index()
        .and_then(|i| model.parameters().get(i))
        .map(|p| p.name.clone());

    let params_json: Vec<Value> = model
        .parameters()
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name.clone(),
                "init": p.init,
                "bounds": [p.bounds.0, p.bounds.1],
                "constraint": constraint_kind(p),
            })
        })
        .collect();

    let (normsys, histosys) = match interp_defaults {
        InterpDefaults::Pyhf => ("code1", "code0"),
        InterpDefaults::Root => ("code4", "code4p"),
    };

    let python_version = apex2_master
        .get("meta")
        .and_then(|m| m.get("python"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let platform = apex2_master
        .get("meta")
        .and_then(|m| m.get("platform"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let pyhf_version = apex2_master
        .get("pyhf")
        .and_then(|p| p.get("report"))
        .and_then(|r| r.get("meta"))
        .and_then(|m| m.get("pyhf_version"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Build suite summaries from the keys present in the Apex2 master report.
    let mut suites: BTreeMap<String, Value> = BTreeMap::new();
    for k in [
        "pyhf",
        "histfactory_golden",
        "survival",
        "survival_statsmodels",
        "regression_golden",
        "nuts_quality",
        "nuts_quality_report",
        "p6_glm_bench",
        "bias_pulls",
        "sbc",
        "root",
    ] {
        if let Some(v) = suite_summary(&apex2_master, k) {
            suites.insert(k.to_string(), v);
        }
    }
    // Also include any unknown sections that follow the common `{status: ...}` convention.
    if let Some(obj) = apex2_master.as_object() {
        for (k, v) in obj {
            if k == "meta" {
                continue;
            }
            if suites.contains_key(k) {
                continue;
            }
            if v.get("status").and_then(|s| s.as_str()).is_some() {
                if let Some(sum) = suite_summary(&apex2_master, k) {
                    suites.insert(k.clone(), sum);
                }
            }
        }
    }

    let overall = overall_status(&suites);

    let generated_at = if deterministic {
        Value::Null
    } else {
        Value::String(chrono::Utc::now().to_rfc3339())
    };

    // Minimal risk-based notes for regulated review workflows (GxP/CSA/SR 11-7 style).
    // Keep this stable and dependency-light: no external references, just pointers to included evidence.
    let mut risk_items: Vec<Value> = Vec::new();
    {
        let mut evidence = vec!["apex2:suite:pyhf".to_string()];
        if let Some(status) = suite_status(&apex2_master, "pyhf") {
            evidence.push(format!("apex2:suite:pyhf:status={status}"));
        }
        risk_items.push(serde_json::json!({
            "risk": "Numerical mismatch versus reference implementation (pyhf) for NLL and expected_data.",
            "mitigation": "Apex2 pyhf parity suite with worst-case numeric deltas recorded.",
            "evidence": evidence,
        }));
    }
    {
        let mut evidence = vec!["apex2:suite:regression_golden".to_string()];
        if let Some(status) = suite_status(&apex2_master, "regression_golden") {
            evidence.push(format!("apex2:suite:regression_golden:status={status}"));
        }
        risk_items.push(serde_json::json!({
            "risk": "Regression in golden-workspace behavior across versions.",
            "mitigation": "Apex2 regression_golden suite summarizes case counts and pass/fail.",
            "evidence": evidence,
        }));
    }
    {
        let mut evidence = vec!["apex2:suite:timeseries".to_string()];
        if let Some(status) = suite_status(&apex2_master, "timeseries") {
            evidence.push(format!("apex2:suite:timeseries:status={status}"));
        }
        risk_items.push(serde_json::json!({
            "risk": "Incorrect state space / Kalman filtering and forecasting behavior in time series tools.",
            "mitigation": "Apex2 timeseries suite runs Kalman filter/smoother/forecast smoke tests; parity vs statsmodels is checked when dependencies are available.",
            "evidence": evidence,
        }));
    }
    {
        let mut evidence = vec!["apex2:suite:pharma".to_string()];
        if let Some(status) = suite_status(&apex2_master, "pharma") {
            evidence.push(format!("apex2:suite:pharma:status={status}"));
        }
        risk_items.push(serde_json::json!({
            "risk": "Incorrect PK/NLME surface behavior (finite NLL/grad, fit smoke) in pharma-oriented tools.",
            "mitigation": "Apex2 pharma suite runs PK/NLME smoke tests plus a reference check against the analytic 1-compartment oral dosing formula.",
            "evidence": evidence,
        }));
    }
    {
        let mut evidence = vec![
            "deterministic".to_string(),
            "dataset_fingerprint:workspace_sha256".to_string(),
            "apex2_summary:master_report_sha256".to_string(),
        ];
        if deterministic {
            evidence.push("pdf:metadata:pinned_creation_date".to_string());
        }
        risk_items.push(serde_json::json!({
            "risk": "Loss of traceability and reproducibility across runs.",
            "mitigation": "Deterministic mode omits timestamps/timings and uses stable ordering; inputs are fingerprinted by SHA-256.",
            "evidence": evidence,
        }));
    }

    let git_commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let rust_toolchain = if deterministic {
        None
    } else {
        // Best-effort: capture `rustc --version` if available.
        Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };

    let out_json = serde_json::json!({
        "schema_version": "validation_report_v1",
        "generated_at": generated_at,
        "deterministic": deterministic,
        "dataset_fingerprint": {
            "workspace_sha256": ws_sha256,
            "workspace_bytes": ws_bytes_len,
            "channels": channels,
            "n_channels": n_channels,
            "n_bins_per_channel": n_bins_per_channel,
            "n_samples_total": uniq_samples.len(),
            "n_parameters": model.n_params(),
            "observation_summary": {
                "total_observed": total_observed,
                "min_bin": min_bin,
                "max_bin": max_bin,
            }
        },
        "model_spec": {
            "poi": poi,
            "parameters": params_json,
            "interpolation": {"normsys": normsys, "histosys": histosys},
            "objective": "poisson_nll_with_constraints",
            "optimizer": "lbfgsb",
            "eval_mode": "fast",
        },
        "environment": {
            "nextstat_version": ns_core::VERSION,
            "nextstat_git_commit": git_commit,
            "python_version": python_version,
            "platform": platform,
            "rust_toolchain": rust_toolchain,
            "pyhf_version": pyhf_version,
            "determinism_settings": {
                "deterministic": deterministic,
                "interp_defaults": match interp_defaults { InterpDefaults::Pyhf => "pyhf", InterpDefaults::Root => "root" },
                "eval_mode": "fast",
            }
        },
        "apex2_summary": {
            "master_report_sha256": apex2_sha256,
            "suites": suites,
            "overall": overall,
        },
        "regulated_review": {
            "contains_raw_data": false,
            "intended_use": "Provide audit-friendly, reproducible validation evidence for NextStat outputs on a specific workspace and Apex2 master report.",
            "scope": "This report summarizes deterministic fingerprints, environment metadata, and Apex2 suite outcomes. It is an evidence artifact to support risk-based review workflows (e.g., GxP/CSA/SR 11-7).",
            "limitations": [
                "This report does not include raw input data; only hashes, sizes, and summary statistics are included.",
                "Validation results are only as complete as the Apex2 suites executed on the producing environment.",
                "A 'pass' indicates suite-level parity/tolerance checks passed; it does not guarantee fitness for all intended uses."
            ],
            "data_handling": {
                "notes": [
                    "No raw workspace JSON is embedded in this report; only SHA-256 and byte size are recorded.",
                    "Observed data are summarized (min/max/total) without per-bin disclosure."
                ]
            },
            "risk_based_assurance": risk_items,
        }
    });

    crate::write_json_file(out_path, &out_json)?;

    if let Some(pdf_path) = pdf {
        let default_python = {
            let venv = PathBuf::from(".venv/bin/python");
            if venv.exists() { venv } else { PathBuf::from("python3") }
        };
        let python = python.cloned().unwrap_or(default_python);

        // Matplotlib may try to write cache/config under $HOME; in restricted environments this
        // can fail or create noisy warnings. Force a writable, repo-local cache dir.
        let mplconfigdir = PathBuf::from("tmp/mplconfig");
        let _ = std::fs::create_dir_all(&mplconfigdir);

        // Only inject repo-local Python sources when the in-tree compiled extension exists.
        // Otherwise this would shadow an installed wheel (CI / end users) and break imports.
        let local_ext_present = {
            let pkg = PathBuf::from("bindings/ns-py/python/nextstat");
            if pkg.is_dir() {
                std::fs::read_dir(pkg)
                    .ok()
                    .and_then(|it| {
                        for e in it.flatten() {
                            let name = e.file_name().to_string_lossy().to_string();
                            if name.starts_with("_core.")
                                && (name.ends_with(".so")
                                    || name.ends_with(".pyd")
                                    || name.ends_with(".dylib")
                                    || name.ends_with(".dll"))
                            {
                                return Some(());
                            }
                        }
                        None
                    })
                    .is_some()
            } else {
                false
            }
        };
        let force_py_path = std::env::var("NEXTSTAT_FORCE_PYTHONPATH").ok().as_deref() == Some("1");

        let mut py = Command::new(&python);
        py.env("MPLCONFIGDIR", &mplconfigdir);
        if local_ext_present || force_py_path {
            let mut pythonpath = std::ffi::OsString::new();
            pythonpath.push("bindings/ns-py/python");
            if let Some(existing) = std::env::var_os("PYTHONPATH")
                && !existing.is_empty()
            {
                pythonpath.push(":");
                pythonpath.push(existing);
            }
            py.env("PYTHONPATH", pythonpath);
        }

        let status = py
            .arg("-m")
            .arg("nextstat.validation_report")
            .arg("render")
            .arg("--input")
            .arg(out_path)
            .arg("--pdf")
            .arg(pdf_path)
            .status()?;

        if !status.success() {
            anyhow::bail!(
                "validation-report renderer failed (python={}, status={})",
                python.display(),
                status
            );
        }
    }

    Ok(())
}
