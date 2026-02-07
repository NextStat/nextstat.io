fn main() {
    // Link Apple Accelerate framework when the feature is enabled on macOS.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Compile CUDA kernels to PTX when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    {
        let kernel_dir = "kernels";
        let common_header = format!("{}/common.cuh", kernel_dir);
        println!("cargo:rerun-if-changed={}", common_header);

        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

        // --- batch_nll_grad.cu ---
        let batch_src = format!("{}/batch_nll_grad.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", batch_src);
        let batch_ptx = format!("{}/batch_nll_grad.ptx", out_dir);

        let status = std::process::Command::new("nvcc")
            .args([
                "--ptx",
                "-arch=sm_70", // Volta minimum, forward-compatible via JIT
                "-O3",
                "--use_fast_math",
                "-I",
                kernel_dir,
                "-o",
                &batch_ptx,
                &batch_src,
            ])
            .status()
            .expect(
                "nvcc not found — install CUDA toolkit (https://developer.nvidia.com/cuda-toolkit)",
            );
        assert!(status.success(), "nvcc failed to compile {}", batch_src);
        println!("cargo:rustc-env=CUDA_PTX_PATH={}", batch_ptx);

        // --- differentiable_nll_grad.cu ---
        let diff_src = format!("{}/differentiable_nll_grad.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", diff_src);
        let diff_ptx = format!("{}/differentiable_nll_grad.ptx", out_dir);

        // NOTE: No --use_fast_math for differentiable kernel — it replaces
        // divisions/sqrts with less accurate intrinsics, which can introduce
        // gradient noise that hurts NN training convergence. Batch toy kernel
        // keeps --use_fast_math since convergence tolerance is already 1e-3.
        let status = std::process::Command::new("nvcc")
            .args([
                "--ptx",
                "-arch=sm_70",
                "-O3",
                "-I",
                kernel_dir,
                "-o",
                &diff_ptx,
                &diff_src,
            ])
            .status()
            .expect("nvcc not found");
        assert!(status.success(), "nvcc failed to compile {}", diff_src);
        println!("cargo:rustc-env=CUDA_DIFF_PTX_PATH={}", diff_ptx);
    }
}
