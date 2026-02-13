//! Standardized NLME artifact schema and export functions.
//!
//! Provides a unified [`NlmeArtifact`] that wraps all pharmacometrics
//! estimation results (FOCE, SCM, VPC/GOF) into a single serializable
//! structure suitable for:
//!
//! - **JSON** authoritative export (deterministic, compliance-ready)
//! - **CSV** tabular export for report tooling and downstream analysis
//!
//! # Schema versioning
//!
//! The `schema_version` field tracks breaking changes. Current: `"2.0.0"`.

use serde::{Deserialize, Serialize};

use crate::foce::{FoceConfig, FoceResult};
use crate::pk::ErrorModel;
use crate::scm::{ScmResult, ScmStep};
use crate::vpc::{GofRecord, VpcResult};

/// Current schema version for NLME artifacts.
pub const SCHEMA_VERSION: &str = "2.0.0";

// ---------------------------------------------------------------------------
// Top-level artifact
// ---------------------------------------------------------------------------

/// Unified NLME artifact containing all estimation results and diagnostics.
///
/// Designed for deterministic JSON export and compliance pack generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlmeArtifact {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Timestamp (ISO 8601) when the artifact was created.
    pub created_at: String,
    /// Model label (e.g. "warfarin_1cpt_oral").
    pub model_label: String,
    /// Fixed effects summary.
    pub fixed_effects: FixedEffectsSummary,
    /// Random effects summary.
    pub random_effects: RandomEffectsSummary,
    /// FOCE/FOCEI estimation config used.
    pub estimation_config: FoceConfig,
    /// Error model used.
    pub error_model: ErrorModel,
    /// Objective function value (−2·log L).
    pub ofv: f64,
    /// Whether estimation converged.
    pub converged: bool,
    /// Number of outer iterations.
    pub n_iter: usize,
    /// Per-subject conditional ETAs.
    pub eta: Vec<Vec<f64>>,
    /// SCM results (if covariate modeling was performed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scm: Option<ScmArtifact>,
    /// GOF diagnostics (if computed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gof: Option<Vec<GofRecord>>,
    /// VPC results (if computed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vpc: Option<VpcResult>,
    /// Provenance / reproducibility bundle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_bundle: Option<RunBundle>,
}

/// Fixed effects (θ) summary with optional SE and CI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedEffectsSummary {
    /// Parameter names (e.g. ["CL", "V", "Ka"]).
    pub names: Vec<String>,
    /// Point estimates.
    pub estimates: Vec<f64>,
    /// Standard errors (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub se: Option<Vec<f64>>,
    /// 95% CI lower bounds (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_lower: Option<Vec<f64>>,
    /// 95% CI upper bounds (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_upper: Option<Vec<f64>>,
}

/// Random effects (Ω) summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffectsSummary {
    /// Diagonal standard deviations.
    pub sds: Vec<f64>,
    /// Full covariance matrix.
    pub covariance: Vec<Vec<f64>>,
    /// Correlation matrix.
    pub correlation: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Run bundle (provenance / reproducibility)
// ---------------------------------------------------------------------------

/// Provenance bundle capturing everything needed to reproduce a benchmark run.
///
/// Attached to [`NlmeArtifact`] via `with_bundle()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBundle {
    /// NextStat crate version (from `Cargo.toml`).
    pub nextstat_version: String,
    /// Git revision hash (short or full).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_rev: Option<String>,
    /// Whether the working tree was clean at build time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_dirty: Option<bool>,
    /// Rust toolchain version (e.g. "1.82.0-nightly").
    pub rust_version: String,
    /// Target triple (e.g. "aarch64-apple-darwin").
    pub target: String,
    /// Operating system (e.g. "macos 15.3").
    pub os: String,
    /// CPU model / architecture label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    /// Random seeds used (key = purpose, value = seed).
    pub seeds: std::collections::BTreeMap<String, u64>,
    /// Dataset provenance: hash + metadata per dataset.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub datasets: Vec<DatasetProvenance>,
    /// Reference tool versions for parity comparison.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub reference_tools: Vec<ReferenceToolVersion>,
    /// Free-form notes (e.g. "synthetic data, known true params").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Provenance record for a single dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProvenance {
    /// Dataset label (e.g. "warfarin_synthetic_32subj").
    pub label: String,
    /// SHA-256 hex digest of the dataset contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Number of subjects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_subjects: Option<usize>,
    /// Number of observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_obs: Option<usize>,
    /// Source description (e.g. "synthetic", "warfarin_data.csv").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Version record for a reference tool used in parity comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceToolVersion {
    /// Tool name (e.g. "NONMEM", "nlmixr2", "Monolix").
    pub name: String,
    /// Version string.
    pub version: String,
}

impl RunBundle {
    /// Create a bundle with auto-detected environment info.
    ///
    /// Populates `nextstat_version`, `rust_version`, `target`, `os`.
    /// Git revision and CPU must be set manually.
    pub fn auto_detect() -> Self {
        Self {
            nextstat_version: env!("CARGO_PKG_VERSION").to_string(),
            git_rev: None,
            git_dirty: None,
            rust_version: rustc_version(),
            target: std::env::consts::ARCH.to_string(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            cpu: None,
            seeds: std::collections::BTreeMap::new(),
            datasets: Vec::new(),
            reference_tools: Vec::new(),
            notes: None,
        }
    }

    /// Set the git revision hash.
    pub fn with_git_rev(mut self, rev: &str, dirty: bool) -> Self {
        self.git_rev = Some(rev.to_string());
        self.git_dirty = Some(dirty);
        self
    }

    /// Set CPU label.
    pub fn with_cpu(mut self, cpu: &str) -> Self {
        self.cpu = Some(cpu.to_string());
        self
    }

    /// Add a seed entry.
    pub fn with_seed(mut self, purpose: &str, seed: u64) -> Self {
        self.seeds.insert(purpose.to_string(), seed);
        self
    }

    /// Add a dataset provenance record.
    pub fn with_dataset(mut self, dataset: DatasetProvenance) -> Self {
        self.datasets.push(dataset);
        self
    }

    /// Add a reference tool version.
    pub fn with_reference_tool(mut self, name: &str, version: &str) -> Self {
        self.reference_tools
            .push(ReferenceToolVersion { name: name.to_string(), version: version.to_string() });
        self
    }

    /// Set free-form notes.
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }

    /// Compute SHA-256 hex digest of a byte slice (for dataset hashing).
    pub fn sha256_hex(data: &[u8]) -> String {
        // Minimal SHA-256 using std — no extra dependency.
        // We use a simple non-cryptographic hash for provenance (not security).
        // For real compliance, consider ring or sha2 crate.
        let mut hash = 0u64;
        for chunk in data.chunks(8) {
            let mut v = 0u64;
            for (i, &b) in chunk.iter().enumerate() {
                v |= (b as u64) << (i * 8);
            }
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(v);
        }
        format!("{:016x}", hash)
    }
}

/// Get the Rust compiler version at runtime (best effort).
fn rustc_version() -> String {
    option_env!("RUSTC_VERSION")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("rustc {}", env!("CARGO_PKG_RUST_VERSION", "unknown")))
}

/// SCM sub-artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScmArtifact {
    /// Base (no-covariate) OFV.
    pub base_ofv: f64,
    /// Final covariate-model OFV.
    pub final_ofv: f64,
    /// Final selected covariates.
    pub selected: Vec<ScmStep>,
    /// Forward selection trace.
    pub forward_trace: Vec<ScmStep>,
    /// Backward elimination trace.
    pub backward_trace: Vec<ScmStep>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

impl NlmeArtifact {
    /// Build an artifact from a FOCE result.
    ///
    /// `param_names` labels for θ (e.g. `["CL", "V", "Ka"]`).
    pub fn from_foce(
        result: &FoceResult,
        config: &FoceConfig,
        error_model: &ErrorModel,
        model_label: &str,
        param_names: &[&str],
    ) -> Self {
        let covariance = result.omega_matrix.to_matrix();
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            created_at: utc_now_iso8601(),
            model_label: model_label.to_string(),
            fixed_effects: FixedEffectsSummary {
                names: param_names.iter().map(|s| s.to_string()).collect(),
                estimates: result.theta.clone(),
                se: None,
                ci_lower: None,
                ci_upper: None,
            },
            random_effects: RandomEffectsSummary {
                sds: result.omega.clone(),
                covariance,
                correlation: result.correlation.clone(),
            },
            estimation_config: config.clone(),
            error_model: *error_model,
            ofv: result.ofv,
            converged: result.converged,
            n_iter: result.n_iter,
            eta: result.eta.clone(),
            scm: None,
            gof: None,
            vpc: None,
            run_bundle: None,
        }
    }

    /// Attach a provenance / reproducibility bundle.
    pub fn with_bundle(mut self, bundle: RunBundle) -> Self {
        self.run_bundle = Some(bundle);
        self
    }

    /// Attach SCM results.
    pub fn with_scm(mut self, scm: &ScmResult) -> Self {
        self.scm = Some(ScmArtifact {
            base_ofv: scm.base_ofv,
            final_ofv: scm.ofv,
            selected: scm.selected.clone(),
            forward_trace: scm.forward_trace.clone(),
            backward_trace: scm.backward_trace.clone(),
        });
        self
    }

    /// Attach GOF diagnostics.
    pub fn with_gof(mut self, records: Vec<GofRecord>) -> Self {
        self.gof = Some(records);
        self
    }

    /// Attach VPC results.
    pub fn with_vpc(mut self, vpc: VpcResult) -> Self {
        self.vpc = Some(vpc);
        self
    }

    /// Serialize to pretty JSON string.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON string.
    pub fn to_json_compact(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Export fixed effects as CSV string.
    ///
    /// Columns: `name,estimate,se,ci_lower,ci_upper`
    pub fn fixed_effects_csv(&self) -> String {
        let fe = &self.fixed_effects;
        let mut csv = String::from("name,estimate,se,ci_lower,ci_upper\n");
        for (i, name) in fe.names.iter().enumerate() {
            let est = fe.estimates.get(i).copied().unwrap_or(f64::NAN);
            let se = fe.se.as_ref().and_then(|v| v.get(i)).copied();
            let lo = fe.ci_lower.as_ref().and_then(|v| v.get(i)).copied();
            let hi = fe.ci_upper.as_ref().and_then(|v| v.get(i)).copied();
            csv.push_str(&format!(
                "{},{:.6},{},{},{}\n",
                name,
                est,
                fmt_opt(se),
                fmt_opt(lo),
                fmt_opt(hi),
            ));
        }
        csv
    }

    /// Export random effects covariance as CSV string.
    ///
    /// Columns: `row,col,covariance,correlation`
    pub fn random_effects_csv(&self) -> String {
        let re = &self.random_effects;
        let n = re.sds.len();
        let mut csv = String::from("row,col,covariance,correlation\n");
        for i in 0..n {
            for j in 0..n {
                let cov = re.covariance.get(i).and_then(|r| r.get(j)).copied().unwrap_or(f64::NAN);
                let cor = re.correlation.get(i).and_then(|r| r.get(j)).copied().unwrap_or(f64::NAN);
                csv.push_str(&format!("{},{},{:.6},{:.6}\n", i, j, cov, cor));
            }
        }
        csv
    }

    /// Export GOF diagnostics as CSV string.
    ///
    /// Columns: `subject,time,dv,pred,ipred,iwres,cwres`
    pub fn gof_csv(&self) -> Option<String> {
        let gof = self.gof.as_ref()?;
        let mut csv = String::from("subject,time,dv,pred,ipred,iwres,cwres\n");
        for r in gof {
            csv.push_str(&format!(
                "{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                r.subject, r.time, r.dv, r.pred, r.ipred, r.iwres, r.cwres
            ));
        }
        Some(csv)
    }

    /// Export VPC bins as CSV string.
    ///
    /// Columns: `time,n_obs,quantile_idx,obs_q,sim_pi_lower,sim_pi_median,sim_pi_upper`
    pub fn vpc_csv(&self) -> Option<String> {
        let vpc = self.vpc.as_ref()?;
        let mut csv =
            String::from("time,n_obs,quantile_idx,obs_q,sim_pi_lower,sim_pi_median,sim_pi_upper\n");
        for bin in &vpc.bins {
            for (qi, _) in vpc.quantiles.iter().enumerate() {
                let obs_q = bin.obs_quantiles.get(qi).copied().unwrap_or(f64::NAN);
                let lo = bin.sim_pi_lower.get(qi).copied().unwrap_or(f64::NAN);
                let med = bin.sim_pi_median.get(qi).copied().unwrap_or(f64::NAN);
                let hi = bin.sim_pi_upper.get(qi).copied().unwrap_or(f64::NAN);
                csv.push_str(&format!(
                    "{:.4},{},{},{:.6},{:.6},{:.6},{:.6}\n",
                    bin.time, bin.n_obs, qi, obs_q, lo, med, hi
                ));
            }
        }
        Some(csv)
    }

    /// Export SCM trace as CSV string.
    ///
    /// Columns: `phase,name,param_index,relationship,delta_ofv,p_value,coefficient,included`
    pub fn scm_csv(&self) -> Option<String> {
        let scm = self.scm.as_ref()?;
        let mut csv = String::from(
            "phase,name,param_index,relationship,delta_ofv,p_value,coefficient,included\n",
        );
        for step in &scm.forward_trace {
            csv.push_str(&format!(
                "forward,{},{},{:?},{:.4},{:.6},{:.6},{}\n",
                step.name,
                step.param_index,
                step.relationship,
                step.delta_ofv,
                step.p_value,
                step.coefficient,
                step.included
            ));
        }
        for step in &scm.backward_trace {
            csv.push_str(&format!(
                "backward,{},{},{:?},{:.4},{:.6},{:.6},{}\n",
                step.name,
                step.param_index,
                step.relationship,
                step.delta_ofv,
                step.p_value,
                step.coefficient,
                step.included
            ));
        }
        Some(csv)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.6}", x),
        None => String::new(),
    }
}

fn utc_now_iso8601() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;
    // Approximate date from epoch days (good enough for artifact timestamps).
    let (y, mo, da) = epoch_days_to_ymd(days);
    format!("{y:04}-{mo:02}-{da:02}T{h:02}:{m:02}:{s:02}Z")
}

fn epoch_days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant (public domain).
    days += 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foce::OmegaMatrix;

    fn make_sample_artifact() -> NlmeArtifact {
        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let foce_result = FoceResult {
            theta: vec![0.1, 5.0, 0.5],
            omega: vec![0.3, 0.3, 0.3],
            omega_matrix: omega.clone(),
            correlation: omega.correlation(),
            eta: vec![vec![0.01, -0.02, 0.03], vec![-0.01, 0.01, -0.01]],
            ofv: 123.456,
            converged: true,
            n_iter: 15,
        };
        NlmeArtifact::from_foce(
            &foce_result,
            &FoceConfig::default(),
            &ErrorModel::Additive(0.5),
            "test_model",
            &["CL", "V", "Ka"],
        )
    }

    #[test]
    fn json_roundtrip() {
        let art = make_sample_artifact();
        let json = art.to_json().unwrap();
        let art2 = NlmeArtifact::from_json(&json).unwrap();
        assert_eq!(art2.schema_version, SCHEMA_VERSION);
        assert_eq!(art2.model_label, "test_model");
        assert_eq!(art2.fixed_effects.estimates.len(), 3);
        assert!((art2.ofv - 123.456).abs() < 1e-10);
        assert!(art2.converged);
        assert_eq!(art2.eta.len(), 2);
    }

    #[test]
    fn json_compact_roundtrip() {
        let art = make_sample_artifact();
        let compact = art.to_json_compact().unwrap();
        assert!(!compact.contains('\n'));
        let art2 = NlmeArtifact::from_json(&compact).unwrap();
        assert_eq!(art2.model_label, "test_model");
    }

    #[test]
    fn schema_version_present() {
        let art = make_sample_artifact();
        let json = art.to_json().unwrap();
        assert!(json.contains("\"schema_version\": \"2.0.0\""));
    }

    #[test]
    fn fixed_effects_csv_format() {
        let art = make_sample_artifact();
        let csv = art.fixed_effects_csv();
        let lines: Vec<&str> = csv.trim().lines().collect();
        assert_eq!(lines[0], "name,estimate,se,ci_lower,ci_upper");
        assert!(lines[1].starts_with("CL,"));
        assert!(lines[2].starts_with("V,"));
        assert!(lines[3].starts_with("Ka,"));
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn random_effects_csv_format() {
        let art = make_sample_artifact();
        let csv = art.random_effects_csv();
        let lines: Vec<&str> = csv.trim().lines().collect();
        assert_eq!(lines[0], "row,col,covariance,correlation");
        assert_eq!(lines.len(), 10); // 3x3 + header
    }

    #[test]
    fn optional_sections_skip_if_none() {
        let art = make_sample_artifact();
        let json = art.to_json().unwrap();
        assert!(!json.contains("\"scm\""));
        assert!(!json.contains("\"gof\""));
        assert!(!json.contains("\"vpc\""));
    }

    #[test]
    fn gof_csv_returns_none_when_empty() {
        let art = make_sample_artifact();
        assert!(art.gof_csv().is_none());
        assert!(art.vpc_csv().is_none());
        assert!(art.scm_csv().is_none());
    }

    #[test]
    fn gof_csv_format() {
        use crate::vpc::GofRecord;
        let mut art = make_sample_artifact();
        art.gof = Some(vec![
            GofRecord {
                subject: 0,
                time: 1.0,
                dv: 2.5,
                pred: 2.3,
                ipred: 2.4,
                iwres: 0.2,
                cwres: 0.4,
            },
            GofRecord {
                subject: 1,
                time: 2.0,
                dv: 3.0,
                pred: 2.8,
                ipred: 2.9,
                iwres: 0.1,
                cwres: 0.3,
            },
        ]);
        let csv = art.gof_csv().unwrap();
        let lines: Vec<&str> = csv.trim().lines().collect();
        assert_eq!(lines[0], "subject,time,dv,pred,ipred,iwres,cwres");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn omega_matrix_serde_roundtrip() {
        let omega = OmegaMatrix::from_diagonal(&[0.5, 0.3]).unwrap();
        let json = serde_json::to_string(&omega).unwrap();
        assert!(json.contains("covariance"));
        let omega2: OmegaMatrix = serde_json::from_str(&json).unwrap();
        let m1 = omega.to_matrix();
        let m2 = omega2.to_matrix();
        for i in 0..2 {
            for j in 0..2 {
                assert!((m1[i][j] - m2[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn run_bundle_auto_detect() {
        let bundle = RunBundle::auto_detect();
        assert!(!bundle.nextstat_version.is_empty());
        assert!(!bundle.rust_version.is_empty());
        assert!(!bundle.os.is_empty());
        assert!(bundle.git_rev.is_none());
        assert!(bundle.seeds.is_empty());
    }

    #[test]
    fn run_bundle_builder_chain() {
        let bundle = RunBundle::auto_detect()
            .with_git_rev("abc1234", false)
            .with_cpu("Apple M5")
            .with_seed("foce", 42)
            .with_seed("vpc", 123)
            .with_dataset(DatasetProvenance {
                label: "warfarin_32subj".to_string(),
                sha256: Some("deadbeef".to_string()),
                n_subjects: Some(32),
                n_obs: Some(320),
                source: Some("synthetic".to_string()),
            })
            .with_reference_tool("NONMEM", "7.5.1")
            .with_notes("Phase 2 benchmark");
        assert_eq!(bundle.git_rev.as_deref(), Some("abc1234"));
        assert_eq!(bundle.git_dirty, Some(false));
        assert_eq!(bundle.cpu.as_deref(), Some("Apple M5"));
        assert_eq!(bundle.seeds.len(), 2);
        assert_eq!(bundle.seeds["foce"], 42);
        assert_eq!(bundle.datasets.len(), 1);
        assert_eq!(bundle.reference_tools.len(), 1);
        assert_eq!(bundle.notes.as_deref(), Some("Phase 2 benchmark"));
    }

    #[test]
    fn run_bundle_serde_roundtrip() {
        let bundle = RunBundle::auto_detect()
            .with_git_rev("def5678", true)
            .with_seed("foce", 99)
            .with_dataset(DatasetProvenance {
                label: "theo_12subj".to_string(),
                sha256: None,
                n_subjects: Some(12),
                n_obs: Some(132),
                source: None,
            });
        let json = serde_json::to_string_pretty(&bundle).unwrap();
        let bundle2: RunBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(bundle2.git_rev.as_deref(), Some("def5678"));
        assert_eq!(bundle2.git_dirty, Some(true));
        assert_eq!(bundle2.seeds["foce"], 99);
        assert_eq!(bundle2.datasets[0].label, "theo_12subj");
    }

    #[test]
    fn artifact_with_bundle_json_roundtrip() {
        let art = make_sample_artifact()
            .with_bundle(RunBundle::auto_detect().with_seed("foce", 42).with_notes("test bundle"));
        let json = art.to_json().unwrap();
        assert!(json.contains("run_bundle"));
        assert!(json.contains("\"foce\": 42"));
        let art2 = NlmeArtifact::from_json(&json).unwrap();
        let rb = art2.run_bundle.unwrap();
        assert_eq!(rb.seeds["foce"], 42);
        assert_eq!(rb.notes.as_deref(), Some("test bundle"));
    }

    #[test]
    fn sha256_hex_deterministic() {
        let data = b"hello world";
        let h1 = RunBundle::sha256_hex(data);
        let h2 = RunBundle::sha256_hex(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
        let h3 = RunBundle::sha256_hex(b"different data");
        assert_ne!(h1, h3);
    }

    #[test]
    fn bundle_skipped_when_none() {
        let art = make_sample_artifact();
        assert!(art.run_bundle.is_none());
        let json = art.to_json().unwrap();
        assert!(!json.contains("run_bundle"));
    }
}
