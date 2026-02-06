use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct BundleMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub command: String,
    pub args: serde_json::Value,
    pub input: BundleInputMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleInputMeta {
    pub original_path: String,
    pub input_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_spec_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Manifest {
    bundle_version: u32,
    files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Serialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

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

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

fn ensure_empty_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        if !dir.is_dir() {
            anyhow::bail!("bundle path exists but is not a directory: {}", dir.display());
        }
        // Keep it simple: require empty dir (or non-existent). This avoids accidental overwrites.
        if dir.read_dir()?.next().is_some() {
            anyhow::bail!("bundle directory must be empty: {}", dir.display());
        }
    } else {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct PyhfWorkspaceSpec {
    channels: Vec<ns_translate::pyhf::Channel>,
    measurements: Vec<ns_translate::pyhf::Measurement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PyhfWorkspaceData {
    observations: Vec<ns_translate::pyhf::Observation>,
}

fn try_split_pyhf_workspace(bytes: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let ws: ns_translate::pyhf::Workspace = match serde_json::from_slice(bytes) {
        Ok(ws) => ws,
        Err(_) => return Ok(None),
    };

    let spec = PyhfWorkspaceSpec {
        channels: ws.channels,
        measurements: ws.measurements,
        version: ws.version,
    };

    let mut observations = ws.observations;
    observations.sort_by(|a, b| a.name.cmp(&b.name));
    let data = PyhfWorkspaceData { observations };

    Ok(Some((serde_json::to_vec_pretty(&spec)?, serde_json::to_vec_pretty(&data)?)))
}

pub fn write_bundle(
    bundle_dir: &Path,
    command: &str,
    args: serde_json::Value,
    input_path: &Path,
    output_value: &serde_json::Value,
) -> Result<()> {
    ensure_empty_dir(bundle_dir)?;

    let inputs_dir = bundle_dir.join("inputs");
    let outputs_dir = bundle_dir.join("outputs");
    std::fs::create_dir_all(&inputs_dir)?;
    std::fs::create_dir_all(&outputs_dir)?;

    let input_bytes = std::fs::read(input_path)?;
    let input_sha256 = sha256_hex(&input_bytes);

    let input_copy = inputs_dir.join("input.json");
    std::fs::write(&input_copy, &input_bytes)?;

    let (data_sha256, model_spec_sha256) =
        if let Some((spec_bytes, data_bytes)) = try_split_pyhf_workspace(&input_bytes)? {
            let spec_path = inputs_dir.join("model_spec.json");
            let data_path = inputs_dir.join("data.json");
            std::fs::write(&spec_path, &spec_bytes)?;
            std::fs::write(&data_path, &data_bytes)?;
            (Some(sha256_hex(&data_bytes)), Some(sha256_hex(&spec_bytes)))
        } else {
            (None, None)
        };

    let created_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let meta = BundleMeta {
        tool: "nextstat".to_string(),
        tool_version: ns_core::VERSION.to_string(),
        created_unix_ms,
        command: command.to_string(),
        args,
        input: BundleInputMeta {
            original_path: input_path.display().to_string(),
            input_sha256,
            data_sha256,
            model_spec_sha256,
        },
    };
    let meta_path = bundle_dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    let out_path = outputs_dir.join("result.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(output_value)?)?;

    let mut files = Vec::new();
    for rel in ["meta.json", "inputs/input.json", "outputs/result.json"] {
        let p = bundle_dir.join(rel);
        files.push(ManifestFile {
            path: rel.to_string(),
            bytes: file_size(&p)?,
            sha256: sha256_file(&p)?,
        });
    }
    for rel in ["inputs/model_spec.json", "inputs/data.json"] {
        let p = bundle_dir.join(rel);
        if p.exists() {
            files.push(ManifestFile {
                path: rel.to_string(),
                bytes: file_size(&p)?,
                sha256: sha256_file(&p)?,
            });
        }
    }

    let manifest = Manifest { bundle_version: 1, files };
    let manifest_path = bundle_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    Ok(())
}
