fn main() {
    // Link Apple Accelerate framework when the feature is enabled on macOS.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Compile CUDA kernels to PTX when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    {
        use std::io::Write as _;

        let warn_nvcc_missing = std::env::var_os("NS_COMPUTE_WARN_NVCC_MISSING").is_some();

        let kernel_dir = "kernels";
        let common_header = format!("{}/common.cuh", kernel_dir);
        println!("cargo:rerun-if-changed={}", common_header);

        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

        fn write_stub_ptx(path: &str, name: &str, why: &str) {
            // Minimal PTX header (no entry functions). This is enough for `include_str!` to work,
            // and avoids hard-failing builds (e.g., clippy) on machines without NVCC installed.
            // If CUDA is actually used at runtime, module/function loading will fail with a clear error.
            let mut f = std::fs::File::create(path).expect("failed to create stub ptx");
            writeln!(
                f,
                "// STUB PTX: {name}\n// reason: {why}\n.version 7.0\n.target sm_70\n.address_size 64\n"
            )
            .expect("failed to write stub ptx");
        }

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
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", batch_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if warn_nvcc_missing {
                    println!(
                        "cargo:warning=ns-compute: nvcc not found; writing stub PTX (CUDA will not work at runtime)"
                    );
                }
                write_stub_ptx(&batch_ptx, "batch_nll_grad", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", batch_src, e),
        }
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
            .args(["--ptx", "-arch=sm_70", "-O3", "-I", kernel_dir, "-o", &diff_ptx, &diff_src])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", diff_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if warn_nvcc_missing {
                    println!(
                        "cargo:warning=ns-compute: nvcc not found; writing stub PTX (CUDA will not work at runtime)"
                    );
                }
                write_stub_ptx(&diff_ptx, "differentiable_nll_grad", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", diff_src, e),
        }
        println!("cargo:rustc-env=CUDA_DIFF_PTX_PATH={}", diff_ptx);
    }
}
