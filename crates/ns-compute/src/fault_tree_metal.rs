//! Metal accelerator for Monte Carlo fault-tree simulation.
//!
//! Wraps the `fault_tree_mc_kernel` MSL kernel. Uploads flattened tree structure
//! once, then runs chunked MC dispatches. All computation in f32 (Apple Silicon).
//!
//! Key differences from the CUDA path:
//! - `atomic_uint` (u32) counters — cast to u64 on readback.
//! - Seed split into two `u32` buffers (seed_lo, seed_hi).
//! - Component params uploaded as f32 (kernel is f32 throughout).

use metal::*;
use std::mem;

const MSL_SRC: &str = include_str!("../kernels/fault_tree_mc.metal");

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal fault_tree_mc: {msg}"))
}

/// Pre-flattened fault tree data ready for GPU upload (same layout as CUDA).
pub struct FlatFaultTreeData {
    pub comp_types: Vec<i32>,
    pub comp_params: Vec<f64>,
    pub node_types: Vec<i32>,
    pub node_data: Vec<i32>,
    pub children_offsets: Vec<i32>,
    pub children_flat: Vec<i32>,
    pub n_components: usize,
    pub n_nodes: usize,
    pub top_node: usize,
}

/// Result from GPU fault-tree MC.
pub struct FaultTreeMetalResult {
    pub n_top_failures: u64,
    pub comp_fail_given_top: Vec<u64>,
}

/// Metal accelerator for fault-tree Monte Carlo.
pub struct FaultTreeMetalAccelerator {
    #[allow(dead_code)]
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,

    buf_comp_types: Buffer,
    buf_comp_params: Buffer, // f32 on GPU
    buf_node_types: Buffer,
    buf_node_data: Buffer,
    buf_children_offsets: Buffer,
    buf_children_flat: Buffer,

    n_components: i32,
    n_nodes: i32,
    top_node: i32,
}

impl FaultTreeMetalAccelerator {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create accelerator from pre-flattened data. Uploads tree to GPU.
    pub fn new(data: &FlatFaultTreeData) -> ns_core::Result<Self> {
        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device"))?;
        let queue = device.new_command_queue();

        let pipeline = crate::metal_kernel_cache::get_pipeline(
            &device,
            "fault_tree_mc",
            MSL_SRC,
            "fault_tree_mc_kernel",
        )?;

        let opts = MTLResourceOptions::StorageModeShared;

        let buf_comp_types = Self::buf_from_slice(&device, &data.comp_types, opts);
        // Convert f64 params to f32 for Metal kernel.
        let params_f32: Vec<f32> = data.comp_params.iter().map(|&v| v as f32).collect();
        let buf_comp_params = Self::buf_from_slice(&device, &params_f32, opts);
        let buf_node_types = Self::buf_from_slice(&device, &data.node_types, opts);
        let buf_node_data = Self::buf_from_slice(&device, &data.node_data, opts);
        let buf_children_offsets = Self::buf_from_slice(&device, &data.children_offsets, opts);
        let buf_children_flat = Self::buf_from_slice(&device, &data.children_flat, opts);

        Ok(Self {
            device,
            queue,
            pipeline,
            buf_comp_types,
            buf_comp_params,
            buf_node_types,
            buf_node_data,
            buf_children_offsets,
            buf_children_flat,
            n_components: data.n_components as i32,
            n_nodes: data.n_nodes as i32,
            top_node: data.top_node as i32,
        })
    }

    /// Run Monte Carlo simulation on Metal GPU. Returns raw counters.
    pub fn run(
        &mut self,
        n_scenarios: usize,
        seed: u64,
        chunk_size: usize,
    ) -> ns_core::Result<FaultTreeMetalResult> {
        let chunk_size = if chunk_size == 0 { 1_000_000 } else { chunk_size };
        let n_comp = self.n_components as usize;
        let opts = MTLResourceOptions::StorageModeShared;

        // Allocate atomic counter buffers (u32).
        // +1 for top_fail, n_comp for component failures.
        let buf_top_fail = self.device.new_buffer(mem::size_of::<u32>() as u64, opts);
        let buf_comp_fail =
            self.device.new_buffer((n_comp.max(1) * mem::size_of::<u32>()) as u64, opts);

        // Zero-initialize counters.
        unsafe {
            std::ptr::write_bytes(buf_top_fail.contents() as *mut u8, 0, mem::size_of::<u32>());
            std::ptr::write_bytes(
                buf_comp_fail.contents() as *mut u8,
                0,
                n_comp.max(1) * mem::size_of::<u32>(),
            );
        }

        let mut offset = 0usize;
        while offset < n_scenarios {
            let this_chunk = (n_scenarios - offset).min(chunk_size);
            let chunk_seed = seed.wrapping_add((offset as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let seed_lo = (chunk_seed & 0xFFFFFFFF) as u32;
            let seed_hi = (chunk_seed >> 32) as u32;
            let n_scenarios_arg = this_chunk as i32;

            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);

            encoder.set_buffer(0, Some(&self.buf_comp_types), 0);
            encoder.set_buffer(1, Some(&self.buf_comp_params), 0);
            encoder.set_buffer(2, Some(&self.buf_node_types), 0);
            encoder.set_buffer(3, Some(&self.buf_node_data), 0);
            encoder.set_buffer(4, Some(&self.buf_children_offsets), 0);
            encoder.set_buffer(5, Some(&self.buf_children_flat), 0);
            encoder.set_buffer(6, Some(&buf_top_fail), 0);
            encoder.set_buffer(7, Some(&buf_comp_fail), 0);
            encoder.set_bytes(
                8,
                mem::size_of::<i32>() as u64,
                &self.n_components as *const i32 as *const _,
            );
            encoder.set_bytes(
                9,
                mem::size_of::<i32>() as u64,
                &self.n_nodes as *const i32 as *const _,
            );
            encoder.set_bytes(
                10,
                mem::size_of::<i32>() as u64,
                &self.top_node as *const i32 as *const _,
            );
            encoder.set_bytes(11, mem::size_of::<u32>() as u64, &seed_lo as *const u32 as *const _);
            encoder.set_bytes(12, mem::size_of::<u32>() as u64, &seed_hi as *const u32 as *const _);
            encoder.set_bytes(
                13,
                mem::size_of::<i32>() as u64,
                &n_scenarios_arg as *const i32 as *const _,
            );

            let thread_group_size = self.pipeline.max_total_threads_per_threadgroup().min(256);
            let grid_size = MTLSize::new(this_chunk as u64, 1, 1);
            let group_size = MTLSize::new(thread_group_size, 1, 1);
            encoder.dispatch_threads(grid_size, group_size);
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            offset += this_chunk;
        }

        // Read back u32 counters and cast to u64.
        let top_fail = unsafe { *(buf_top_fail.contents() as *const u32) } as u64;
        let comp_fail: Vec<u64> = unsafe {
            let ptr = buf_comp_fail.contents() as *const u32;
            (0..n_comp).map(|i| *ptr.add(i) as u64).collect()
        };

        Ok(FaultTreeMetalResult { n_top_failures: top_fail, comp_fail_given_top: comp_fail })
    }

    fn buf_from_slice<T>(device: &Device, data: &[T], opts: MTLResourceOptions) -> Buffer {
        if data.is_empty() {
            return device.new_buffer(mem::size_of::<T>().max(4) as u64, opts);
        }
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            mem::size_of_val(data) as u64,
            opts,
        )
    }
}
