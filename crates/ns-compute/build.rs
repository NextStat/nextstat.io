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

        println!("cargo:rerun-if-env-changed=NS_COMPUTE_FORCE_STUB_PTX");
        let force_stub_ptx = std::env::var("NS_COMPUTE_FORCE_STUB_PTX")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Find nvcc: check CUDA_PATH/CUDA_HOME, then /usr/local/cuda, then PATH.
        let nvcc = if force_stub_ptx {
            // Make Command::new() fail with NotFound so each kernel path
            // falls back to stub PTX generation.
            String::from("__ns_compute_force_stub_nvcc__")
        } else {
            let candidates = [
                std::env::var("CUDA_PATH").ok().map(|p| format!("{}/bin/nvcc", p)),
                std::env::var("CUDA_HOME").ok().map(|p| format!("{}/bin/nvcc", p)),
                Some("/usr/local/cuda/bin/nvcc".to_string()),
            ];
            let mut found = String::from("nvcc"); // fallback to PATH lookup
            for c in candidates.into_iter().flatten() {
                if std::path::Path::new(&c).exists() {
                    found = c;
                    break;
                }
            }
            found
        };
        if force_stub_ptx {
            println!(
                "cargo:warning=ns-compute: NS_COMPUTE_FORCE_STUB_PTX=1, writing stub PTX for CUDA kernels"
            );
        }

        let kernel_dir = "kernels";
        let common_header = format!("{}/common.cuh", kernel_dir);
        println!("cargo:rerun-if-changed={}", common_header);
        let unbinned_common_header = format!("{}/unbinned_common.cuh", kernel_dir);
        println!("cargo:rerun-if-changed={}", unbinned_common_header);

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

        let status = std::process::Command::new(&nvcc)
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
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for batch_nll_grad (CUDA will not work at runtime)"
                );
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
        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-arch=sm_70", "-O3", "-I", kernel_dir, "-o", &diff_ptx, &diff_src])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", diff_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for differentiable_nll_grad (CUDA will not work at runtime)"
                );
                write_stub_ptx(&diff_ptx, "differentiable_nll_grad", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", diff_src, e),
        }
        println!("cargo:rustc-env=CUDA_DIFF_PTX_PATH={}", diff_ptx);

        // --- unbinned_nll_grad.cu ---
        let unbinned_src = format!("{}/unbinned_nll_grad.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", unbinned_src);
        let unbinned_ptx = format!("{}/unbinned_nll_grad.ptx", out_dir);

        // NOTE: No --use_fast_math for unbinned likelihood kernels — they use erf/log1p
        // and are typically used for inference fits where numerical parity matters.
        let status = std::process::Command::new(&nvcc)
            .args([
                "--ptx",
                "-arch=sm_70", // Volta minimum, forward-compatible via JIT
                "-O3",
                "-I",
                kernel_dir,
                "-o",
                &unbinned_ptx,
                &unbinned_src,
            ])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", unbinned_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for unbinned_nll_grad (CUDA will not work at runtime)"
                );
                write_stub_ptx(&unbinned_ptx, "unbinned_nll_grad", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", unbinned_src, e),
        }
        println!("cargo:rustc-env=CUDA_UNBINNED_PTX_PATH={}", unbinned_ptx);

        // --- unbinned_lbfgs_fit.cu (GPU-native persistent L-BFGS optimizer) ---
        let lbfgs_src = format!("{}/unbinned_lbfgs_fit.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", lbfgs_src);
        let lbfgs_ptx = format!("{}/unbinned_lbfgs_fit.ptx", out_dir);

        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-arch=sm_70", "-O3", "-I", kernel_dir, "-o", &lbfgs_ptx, &lbfgs_src])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", lbfgs_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for unbinned_lbfgs_fit (CUDA will not work at runtime)"
                );
                write_stub_ptx(&lbfgs_ptx, "unbinned_lbfgs_fit", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", lbfgs_src, e),
        }
        println!("cargo:rustc-env=CUDA_LBFGS_PTX_PATH={}", lbfgs_ptx);

        // --- unbinned_weight_sys.cu ---
        let unbinned_ws_src = format!("{}/unbinned_weight_sys.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", unbinned_ws_src);
        let unbinned_ws_ptx = format!("{}/unbinned_weight_sys.ptx", out_dir);

        // NOTE: No --use_fast_math for this kernel — weight systematics are used for
        // inference/parity-sensitive workflows.
        let status = std::process::Command::new(&nvcc)
            .args([
                "--ptx",
                "-arch=sm_70",
                "-O3",
                "-I",
                kernel_dir,
                "-o",
                &unbinned_ws_ptx,
                &unbinned_ws_src,
            ])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", unbinned_ws_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for unbinned_weight_sys (CUDA will not work at runtime)"
                );
                write_stub_ptx(&unbinned_ws_ptx, "unbinned_weight_sys", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", unbinned_ws_src, e),
        }
        println!("cargo:rustc-env=CUDA_UNBINNED_WEIGHT_SYS_PTX_PATH={}", unbinned_ws_ptx);

        // --- flow_nll_reduce.cu ---
        let flow_src = format!("{}/flow_nll_reduce.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", flow_src);
        let flow_ptx = format!("{}/flow_nll_reduce.ptx", out_dir);

        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-arch=sm_70", "-O3", "-I", kernel_dir, "-o", &flow_ptx, &flow_src])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", flow_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for flow_nll_reduce (CUDA will not work at runtime)"
                );
                write_stub_ptx(&flow_ptx, "flow_nll_reduce", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", flow_src, e),
        }
        println!("cargo:rustc-env=CUDA_FLOW_PTX_PATH={}", flow_ptx);

        // --- fault_tree_mc.cu ---
        let ft_src = format!("{}/fault_tree_mc.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", ft_src);
        let ft_ptx = format!("{}/fault_tree_mc.ptx", out_dir);

        // Use --use_fast_math: MC throughput > numerical parity.
        let status = std::process::Command::new(&nvcc)
            .args([
                "--ptx",
                "-arch=sm_70",
                "-O3",
                "--use_fast_math",
                "-I",
                kernel_dir,
                "-o",
                &ft_ptx,
                &ft_src,
            ])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", ft_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for fault_tree_mc (CUDA will not work at runtime)"
                );
                write_stub_ptx(&ft_ptx, "fault_tree_mc", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", ft_src, e),
        }
        println!("cargo:rustc-env=CUDA_FAULT_TREE_MC_PTX_PATH={}", ft_ptx);

        // --- mams_engine.cuh (shared engine header for LAPS, used by build-time + JIT) ---
        let mams_engine = format!("{}/mams_engine.cuh", kernel_dir);
        println!("cargo:rerun-if-changed={}", mams_engine);

        // Expose the engine header path so nvrtc_mams.rs can include_str! it for JIT compilation
        let mams_engine_abs = std::fs::canonicalize(&mams_engine)
            .unwrap_or_else(|_| std::path::PathBuf::from(&mams_engine));
        println!("cargo:rustc-env=CUDA_MAMS_ENGINE_PATH={}", mams_engine_abs.display());

        // --- mams_leapfrog.cu (LAPS: GPU MAMS sampler) ---
        let mams_src = format!("{}/mams_leapfrog.cu", kernel_dir);
        println!("cargo:rerun-if-changed={}", mams_src);
        let mams_ptx = format!("{}/mams_leapfrog.ptx", out_dir);

        // NOTE: No --use_fast_math — MAMS energy conservation requires precise
        // exp/log/sqrt for MH detailed balance.
        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-arch=sm_70", "-O3", "-I", kernel_dir, "-o", &mams_ptx, &mams_src])
            .status();
        match status {
            Ok(st) if st.success() => {}
            Ok(st) => panic!("nvcc failed to compile {} (exit={})", mams_src, st),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "cargo:warning=ns-compute: nvcc not found; writing stub PTX for mams_leapfrog (CUDA will not work at runtime)"
                );
                write_stub_ptx(&mams_ptx, "mams_leapfrog", "nvcc not found");
            }
            Err(e) => panic!("failed to spawn nvcc for {}: {}", mams_src, e),
        }
        println!("cargo:rustc-env=CUDA_MAMS_PTX_PATH={}", mams_ptx);

        // glm_cublas_aux is NVRTC-JIT compiled at runtime (see cuda_glm_cublas.rs).
    }
}
