//! CUDA accelerator for Monte Carlo fault-tree simulation.
//!
//! Wraps the `fault_tree_mc_kernel` CUDA kernel. Uploads flattened tree structure
//! once, then runs chunked MC in a grid-stride loop.
//!
//! This module accepts pre-flattened data (i32/f64 arrays) to avoid
//! a circular dependency on `ns-inference`. The flattening is done in
//! `ns-inference::fault_tree_mc::flatten_spec_for_gpu()`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const PTX_SRC: &str = include_str!(env!("CUDA_FAULT_TREE_MC_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA fault_tree_mc: {msg}"))
}

/// Pre-flattened fault tree data ready for GPU upload.
pub struct FlatFaultTreeData {
    /// Component types: 0=Bernoulli, 1=BernoulliUncertain, 2=WeibullMission.
    pub comp_types: Vec<i32>,
    /// Component params, 3 per component: `[p/mu/k, 0/sigma/lambda, 0/0/mission_time]`.
    pub comp_params: Vec<f64>,
    /// Node types: 0=Component, 1=AND, 2=OR.
    pub node_types: Vec<i32>,
    /// Node data: component index (for type 0), unused otherwise.
    pub node_data: Vec<i32>,
    /// CSR row pointers for children: `[n_nodes+1]`.
    pub children_offsets: Vec<i32>,
    /// Flat children indices.
    pub children_flat: Vec<i32>,
    /// Number of components.
    pub n_components: usize,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Top event node index.
    pub top_node: usize,
}

/// CUDA accelerator for fault-tree Monte Carlo.
pub struct FaultTreeCudaAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,

    d_comp_types: CudaSlice<i32>,
    d_comp_params: CudaSlice<f64>,
    d_node_types: CudaSlice<i32>,
    d_node_data: CudaSlice<i32>,
    d_children_offsets: CudaSlice<i32>,
    d_children_flat: CudaSlice<i32>,

    n_components: i32,
    n_nodes: i32,
    top_node: i32,
}

/// Result from GPU fault-tree MC (mirrors `FaultTreeMcResult` in ns-inference).
pub struct FaultTreeCudaResult {
    /// Total TOP failures.
    pub n_top_failures: u64,
    /// Per-component failures given TOP failed.
    pub comp_fail_given_top: Vec<u64>,
}

impl FaultTreeCudaAccelerator {
    /// Check CUDA availability.
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Create accelerator from pre-flattened data. Uploads tree to GPU.
    pub fn new(data: &FlatFaultTreeData) -> ns_core::Result<Self> {
        let ctx = CudaContext::new(0).map_err(|e| cuda_err(format!("context: {e}")))?;
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load module: {e}")))?;
        let kernel = module
            .load_function("fault_tree_mc_kernel")
            .map_err(|e| cuda_err(format!("load function: {e}")))?;

        let d_comp_types = stream.memcpy_stod(&data.comp_types).map_err(cuda_err)?;
        let d_comp_params = stream.memcpy_stod(&data.comp_params).map_err(cuda_err)?;
        let d_node_types = stream.memcpy_stod(&data.node_types).map_err(cuda_err)?;
        let d_node_data = stream.memcpy_stod(&data.node_data).map_err(cuda_err)?;
        let d_children_offsets = stream.memcpy_stod(&data.children_offsets).map_err(cuda_err)?;
        let d_children_flat = stream.memcpy_stod(&data.children_flat).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            d_comp_types,
            d_comp_params,
            d_node_types,
            d_node_data,
            d_children_offsets,
            d_children_flat,
            n_components: data.n_components as i32,
            n_nodes: data.n_nodes as i32,
            top_node: data.top_node as i32,
        })
    }

    /// Run Monte Carlo simulation on GPU. Returns raw counters.
    ///
    /// Caller (in ns-inference) converts to `FaultTreeMcResult`.
    pub fn run(
        &mut self,
        n_scenarios: usize,
        seed: u64,
        chunk_size: usize,
    ) -> ns_core::Result<FaultTreeCudaResult> {
        let chunk_size = if chunk_size == 0 { 1_000_000 } else { chunk_size };
        let n_comp = self.n_components as usize;

        let mut d_top_fail: CudaSlice<u64> = self.stream.memcpy_stod(&[0u64]).map_err(cuda_err)?;
        let mut d_comp_fail: CudaSlice<u64> =
            self.stream.memcpy_stod(&vec![0u64; n_comp.max(1)]).map_err(cuda_err)?;

        let mut offset = 0usize;
        while offset < n_scenarios {
            let this_chunk = (n_scenarios - offset).min(chunk_size);

            let block_dim = 256u32;
            let grid_dim = ((this_chunk as u32 + block_dim - 1) / block_dim).min(65535);

            let config = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            let n_comp_arg = self.n_components;
            let n_nodes_arg = self.n_nodes;
            let top_node_arg = self.top_node;
            let seed_arg = seed.wrapping_add((offset as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let seed_lo: u32 = (seed_arg & 0xFFFF_FFFF) as u32;
            let seed_hi: u32 = (seed_arg >> 32) as u32;
            let n_scenarios_arg = this_chunk as i32;

            let mut builder = self.stream.launch_builder(&self.kernel);
            builder.arg(&self.d_comp_types);
            builder.arg(&self.d_comp_params);
            builder.arg(&self.d_node_types);
            builder.arg(&self.d_node_data);
            builder.arg(&self.d_children_offsets);
            builder.arg(&self.d_children_flat);
            builder.arg(&mut d_top_fail);
            builder.arg(&mut d_comp_fail);
            builder.arg(&n_comp_arg);
            builder.arg(&n_nodes_arg);
            builder.arg(&top_node_arg);
            builder.arg(&seed_lo);
            builder.arg(&seed_hi);
            builder.arg(&n_scenarios_arg);

            unsafe {
                builder.launch(config).map_err(|e| cuda_err(format!("launch: {e}")))?;
            }

            offset += this_chunk;
        }

        let mut top_fail_host = vec![0u64; 1];
        let mut comp_fail_host = vec![0u64; n_comp.max(1)];
        self.stream.memcpy_dtoh(&d_top_fail, &mut top_fail_host).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&d_comp_fail, &mut comp_fail_host).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        comp_fail_host.truncate(n_comp);

        Ok(FaultTreeCudaResult {
            n_top_failures: top_fail_host[0],
            comp_fail_given_top: comp_fail_host,
        })
    }
}
