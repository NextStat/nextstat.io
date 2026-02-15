//! NVRTC JIT compiler for user-defined MAMS models.
//!
//! Compiles user CUDA C (defining `user_nll` + `user_grad`) together with the
//! MAMS engine header into PTX at runtime. Supports disk caching to avoid
//! recompilation for the same model + GPU architecture.
//!
//! # Usage
//!
//! ```ignore
//! let compiler = MamsJitCompiler::new()?;
//! let ptx_str = compiler.compile(user_cuda_code)?;
//! // ptx_str is a String containing PTX IR, ready for Ptx::from_src()
//! ```

use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// The MAMS engine header source, embedded at build time.
const ENGINE_SRC: &str = include_str!(env!("CUDA_MAMS_ENGINE_PATH"));

fn jit_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("NVRTC JIT: {msg}"))
}

/// NVRTC JIT compiler for MAMS user models.
///
/// Concatenates user CUDA C code with the MAMS engine header and compiles
/// to PTX via cudarc's NVRTC bindings. Results are cached on disk keyed
/// by SHA-256 of (source + GPU arch).
pub struct MamsJitCompiler {
    /// GPU compute capability, e.g. "sm_89" for Ada.
    arch: String,
    /// Disk cache directory (~/.cache/nextstat/ptx/).
    cache_dir: PathBuf,
}

impl MamsJitCompiler {
    /// Create a JIT compiler for a specific GPU device.
    pub fn new_for_device(device_id: usize) -> ns_core::Result<Self> {
        let arch = detect_gpu_arch_for_device(device_id)?;
        let cache_dir = cache_directory();
        Ok(Self { arch, cache_dir })
    }

    /// Create a new JIT compiler, auto-detecting GPU 0 compute capability.
    pub fn new() -> ns_core::Result<Self> {
        Self::new_for_device(0)
    }

    /// Compile user CUDA C source into PTX.
    ///
    /// The `user_code` must define:
    /// ```c
    /// __device__ double user_nll(const double* x, int dim, const double* model_data);
    /// __device__ void   user_grad(const double* x, double* grad, int dim, const double* model_data);
    /// ```
    ///
    /// Returns PTX as a string. Cached to disk on first compile; subsequent
    /// calls with the same source + arch return the cached PTX in <1ms.
    pub fn compile(&self, user_code: &str) -> ns_core::Result<String> {
        // Full source: user model code + engine header
        let full_source = format!("{}\n{}", user_code, ENGINE_SRC);

        // Cache key: SHA-256(source + arch)
        let cache_key = {
            let mut hasher = Sha256::new();
            hasher.update(full_source.as_bytes());
            hasher.update(self.arch.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let cached_path = self.cache_dir.join(format!("{cache_key}.ptx"));

        // Cache hit?
        if let Ok(ptx) = std::fs::read_to_string(&cached_path) {
            if !ptx.is_empty() {
                log::debug!("NVRTC cache hit: {}", cached_path.display());
                return Ok(ptx);
            }
        }

        // Cache miss: compile via NVRTC
        log::info!("NVRTC JIT compile ({}, {} bytes source)", self.arch, full_source.len());
        let t0 = std::time::Instant::now();

        let ptx_str = compile_nvrtc(&full_source, &self.arch)?;

        let elapsed_ms = t0.elapsed().as_millis();
        log::info!("NVRTC compile done in {}ms, PTX {} bytes", elapsed_ms, ptx_str.len());

        // Write to cache (best-effort)
        if let Err(e) = std::fs::create_dir_all(&self.cache_dir) {
            log::warn!("Failed to create PTX cache dir: {e}");
        } else if let Err(e) = std::fs::write(&cached_path, &ptx_str) {
            log::warn!("Failed to write PTX cache: {e}");
        }

        Ok(ptx_str)
    }

    /// GPU architecture string (e.g. "sm_89").
    pub fn arch(&self) -> &str {
        &self.arch
    }
}

/// Detect GPU compute capability for a specific device via cudarc driver API.
fn detect_gpu_arch_for_device(device_id: usize) -> ns_core::Result<String> {
    use cudarc::driver::result;
    use cudarc::driver::sys;

    unsafe {
        result::init().map_err(|e| jit_err(format!("cuInit: {e}")))?;
        let dev = result::device::get(device_id as i32)
            .map_err(|e| jit_err(format!("cuDeviceGet({device_id}): {e}")))?;

        let major = result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|e| jit_err(format!("get CC major (device {device_id}): {e}")))?;

        let minor = result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|e| jit_err(format!("get CC minor (device {device_id}): {e}")))?;

        Ok(format!("sm_{major}{minor}"))
    }
}

/// Compile CUDA C source to PTX via cudarc NVRTC.
fn compile_nvrtc(source: &str, arch: &str) -> ns_core::Result<String> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    // cudarc CompileOptions.arch requires &'static str.
    // We pass the arch via the raw `options` vec instead.
    let opts = CompileOptions {
        prec_sqrt: Some(true),
        prec_div: Some(true),
        // NO use_fast_math — MAMS requires precise exp/log/sqrt for MH balance
        arch: None, // set via options vec since field requires &'static str
        options: vec![format!("--gpu-architecture={arch}")],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(source, opts)
        .map_err(|e| jit_err(format!("NVRTC compilation failed:\n{e}")))?;

    Ok(ptx.to_src())
}

/// PTX cache directory: `~/.cache/nextstat/ptx/`
fn cache_directory() -> PathBuf {
    if let Some(home) = dirs_home() {
        home.join(".cache").join("nextstat").join("ptx")
    } else {
        std::env::temp_dir().join("nextstat-ptx-cache")
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_src_not_empty() {
        assert!(!ENGINE_SRC.is_empty());
        assert!(ENGINE_SRC.contains("mams_transition"));
        assert!(ENGINE_SRC.contains("user_nll"));
        assert!(ENGINE_SRC.contains("user_grad"));
    }

    #[test]
    fn test_cache_directory() {
        let dir = cache_directory();
        assert!(dir.to_str().unwrap().contains("nextstat"));
    }
}
