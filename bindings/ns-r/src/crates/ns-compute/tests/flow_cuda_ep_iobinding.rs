//! CUDA EP + I/O binding smoke test for FlowPdf log_prob -> GPU reduction.
//!
//! Run:
//! - `cargo test -p ns-compute --features "cuda flow-ort-cuda" --test flow_cuda_ep_iobinding`
//!
//! Notes:
//! - Requires runtime CUDA and ONNX Runtime CUDA EP availability.
//! - Uses the `tests/fixtures/flow_test` ONNX fixtures (generated elsewhere).

#![cfg(all(feature = "cuda", feature = "flow-ort-cuda"))]

use std::path::PathBuf;

use ns_compute::cuda_flow_nll::{CudaFlowNllAccelerator, FlowNllConfig};
use ns_unbinned::FlowPdf;
use ns_unbinned::event_store::{EventStore, ObservableSpec};
use ns_unbinned::pdf::UnbinnedPdf;

fn has_cuda() -> bool {
    CudaFlowNllAccelerator::is_available()
}

fn manifest_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // crates/ns-compute
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("tests/fixtures/flow_test/flow_manifest.json");
    p
}

fn has_fixtures() -> bool {
    manifest_path().exists()
}

#[test]
fn test_flow_logprob_cuda_ep_zero_copy_reduce_matches_host_path() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found under tests/fixtures/flow_test");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();

    let obs = ObservableSpec::branch("x", (-6.0, 6.0));
    let xs = vec![0.0, 1.0, -1.0, 0.5, -0.5];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let n_events = store.n_events();
    let config = FlowNllConfig {
        n_events,
        n_procs: 1,
        n_params: 0,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();
    let yields = [100.0f64];
    let params: [f64; 0] = [];

    // Host path: FlowPdf CPU run -> f64 logp -> GPU reduction.
    let mut logp_host = vec![0.0f64; n_events];
    flow.log_prob_batch(&store, &[], &mut logp_host).unwrap();
    let nll_host = accel.nll(&logp_host, &yields, &params).unwrap();

    // Device-resident path: FlowPdf CUDA EP output -> pass device pointer to f32 reducer.
    let logp_dev = match flow.log_prob_batch_cuda(&store, &[]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: FlowPdf CUDA EP path not available: {e}");
            return;
        }
    };
    let ptr = logp_dev.device_ptr_u64();
    assert_ne!(ptr, 0, "expected non-null CUDA device pointer");

    let nll_dev = accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap();

    // Float logp may introduce minor reduction differences vs f64 host path.
    assert!(
        (nll_host - nll_dev).abs() < 1e-5,
        "NLL mismatch: host={nll_host}, device_ptr_f32={nll_dev}"
    );
}
