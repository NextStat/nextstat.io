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
                let mut worst = 0.0f64;
                for c in cases {
                    let d = c
                        .get("parity")
                        .and_then(|v| v.get("max_abs_delta_nll"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    worst = worst.max(d);
                }
                out.insert("worst_delta_nll".to_string(), serde_json::json!(worst));
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
            }
        }
        _ => {}
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
            "nextstat_git_commit": Value::Null,
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
        }
    });

    crate::write_json_file(out_path, &out_json)?;

    if let Some(pdf_path) = pdf {
        let default_python = {
            let venv = PathBuf::from(".venv/bin/python");
            if venv.exists() { venv } else { PathBuf::from("python3") }
        };
        let python = python.cloned().unwrap_or(default_python);

        let mut pythonpath = std::ffi::OsString::new();
        pythonpath.push("bindings/ns-py/python");
        if let Some(existing) = std::env::var_os("PYTHONPATH")
            && !existing.is_empty()
        {
            pythonpath.push(":");
            pythonpath.push(existing);
        }

        // Matplotlib may try to write cache/config under $HOME; in restricted environments this
        // can fail or create noisy warnings. Force a writable, repo-local cache dir.
        let mplconfigdir = PathBuf::from("tmp/mplconfig");
        let _ = std::fs::create_dir_all(&mplconfigdir);

        let status = Command::new(&python)
            .env("PYTHONPATH", pythonpath)
            .env("MPLCONFIGDIR", &mplconfigdir)
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
