fn main() {
    // Link Apple Accelerate framework when the feature is enabled on macOS.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Compile CUDA kernel to PTX when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    {
        let kernel_src = "kernels/batch_nll_grad.cu";
        println!("cargo:rerun-if-changed={}", kernel_src);

        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let ptx_path = format!("{}/batch_nll_grad.ptx", out_dir);

        let status = std::process::Command::new("nvcc")
            .args([
                "--ptx",
                "-arch=sm_70", // Volta minimum, forward-compatible via JIT
                "-O3",
                "--use_fast_math",
                "-o",
                &ptx_path,
                kernel_src,
            ])
            .status()
            .expect(
                "nvcc not found — install CUDA toolkit (https://developer.nvidia.com/cuda-toolkit)",
            );

        assert!(status.success(), "nvcc failed to compile {}", kernel_src);
        println!("cargo:rustc-env=CUDA_PTX_PATH={}", ptx_path);
    }
}
