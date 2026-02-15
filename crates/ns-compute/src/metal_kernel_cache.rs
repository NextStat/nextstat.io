//! Thread-local cache for runtime-compiled Metal libraries/pipelines.
//!
//! This avoids repeated `new_library_with_source` and
//! `new_compute_pipeline_state_with_function` calls in short-lived CLI runs.

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static LIB_CACHE: RefCell<HashMap<String, Library>> = RefCell::new(HashMap::new());
    static PIPELINE_CACHE: RefCell<HashMap<String, ComputePipelineState>> = RefCell::new(HashMap::new());
}

fn cache_err(namespace: &str, msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal ({namespace}): {msg}"))
}

fn library_key(namespace: &str) -> String {
    namespace.to_string()
}

fn pipeline_key(namespace: &str, function_name: &str) -> String {
    format!("{namespace}::{function_name}")
}

fn get_library(
    device: &Device,
    namespace: &'static str,
    source: &'static str,
) -> ns_core::Result<Library> {
    let key = library_key(namespace);
    if let Some(lib) = LIB_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        return Ok(lib);
    }

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|e| cache_err(namespace, format!("MSL compile: {e}")))?;

    LIB_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, library.clone());
    });

    Ok(library)
}

/// Get a compute pipeline from cache (or compile/build and cache it).
pub(crate) fn get_pipeline(
    device: &Device,
    namespace: &'static str,
    source: &'static str,
    function_name: &'static str,
) -> ns_core::Result<ComputePipelineState> {
    let pkey = pipeline_key(namespace, function_name);
    if let Some(pso) = PIPELINE_CACHE.with(|cache| cache.borrow().get(&pkey).cloned()) {
        return Ok(pso);
    }

    let library = get_library(device, namespace, source)?;
    let function = library
        .get_function(function_name, None)
        .map_err(|e| cache_err(namespace, format!("get {function_name}: {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| cache_err(namespace, format!("pipeline {function_name}: {e}")))?;

    PIPELINE_CACHE.with(|cache| {
        cache.borrow_mut().insert(pkey, pipeline.clone());
    });

    Ok(pipeline)
}
