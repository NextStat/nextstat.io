## Environment
```text
## uname
Linux Ubuntu-2404-noble-amd64-base 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
## lscpu
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               20
On-line CPU(s) list:                  0-19
Vendor ID:                            GenuineIntel
BIOS Vendor ID:                       Intel(R) Corporation
Model name:                           13th Gen Intel(R) Core(TM) i5-13500
BIOS Model name:                      13th Gen Intel(R) Core(TM) i5-13500 To Be Filled By O.E.M. CPU @ 2.4GHz
BIOS CPU family:                      205
CPU family:                           6
Model:                                191
Thread(s) per core:                   2
Core(s) per socket:                   14
Socket(s):                            1
Stepping:                             2
CPU(s) scaling MHz:                   35%
CPU max MHz:                          4800.0000
CPU min MHz:                          800.0000
BogoMIPS:                             4992.00
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq tme rdpid movdiri movdir64b fsrm md_clear serialize pconfig arch_lbr ibt flush_l1d arch_capabilities ibpb_exit_to_user
Virtualization:                       VT-x
L1d cache:                            544 KiB (14 instances)
L1i cache:                            704 KiB (14 instances)
L2 cache:                             11.5 MiB (8 instances)
L3 cache:                             24 MiB (1 instance)
NUMA node(s):                         1
NUMA node0 CPU(s):                    0-19
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Mitigation; Clear Register File
Vulnerability Retbleed:               Not affected
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
Vulnerability Vmscape:                Mitigation; IBPB before exit to userspace
## mem
               total        used        free      shared  buff/cache   available
Mem:            62Gi       3.8Gi       3.8Gi        47Mi        55Gi        58Gi
Swap:           31Gi       4.5Mi        31Gi
## nvidia-smi
0, NVIDIA RTX 4000 SFF Ada Generation, 20475 MiB, 580.95.05, 100 %
## nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
## rust
rustc 1.93.1 (01f6ddf75 2026-02-11)
cargo 1.93.1 (083ac5135 2025-12-15)
## nextstat version
nextstat 0.9.0
```

| Case | NS (median) | Competitor (median) | Speedup | Parity | Status |
|------|-------------|---------------------|---------|--------|--------|
| gauss_exp_10k/cuda1_device_sh8_t1000 | 2.153s | 0.706s | 0.3x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_10k/cuda1_device_sh8_t10000 | 13.867s | 7.937s | 0.6x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_10k/cuda1_host_t1000 | 2.308s | 0.706s | 0.3x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_10k/cuda1_host_t10000 | 30.332s | 7.937s | 0.3x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_10k/cuda1_native_t1000 | 2.306s | 0.706s | 0.3x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_10k/cuda1_native_t10000 | 30.455s | 7.937s | 0.3x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_device_sh8_t1000 | 16.216s | 7.467s | 0.5x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_device_sh8_t10000 | 133.230s | 81.287s | 0.6x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_host_t1000 | 14.796s | 7.467s | 0.5x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_host_t10000 | 175.626s | 81.287s | 0.5x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_native_t1000 | 14.792s | 7.467s | 0.5x | ok (conv=1.000, n_error=0) | fail:slower |
| gauss_exp_100k/cuda1_native_t10000 | 172.289s | 81.287s | 0.5x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_10k/cuda1_device_sh8_t1000 | 4.467s | 3.617s | 0.8x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_10k/cuda1_device_sh8_t10000 | 33.600s | 43.620s | 1.3x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_10k/cuda1_host_t1000 | 4.361s | 3.617s | 0.8x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_10k/cuda1_host_t10000 | 51.986s | 43.620s | 0.8x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_10k/cuda1_native_t1000 | 4.360s | 3.617s | 0.8x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_10k/cuda1_native_t10000 | 51.980s | 43.620s | 0.8x | ok (conv=1.000, n_error=0) | fail:slower |
| cb_exp_100k/cuda1_device_sh8_t1000 | 36.274s | 65.063s | 1.8x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_100k/cuda1_device_sh8_t10000 | 154.485s | 505.624s | 3.3x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_100k/cuda1_host_t1000 | 16.898s | 65.063s | 3.9x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_100k/cuda1_host_t10000 | 177.729s | 505.624s | 2.8x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_100k/cuda1_native_t1000 | 18.128s | 65.063s | 3.6x | ok (conv=1.000, n_error=0) | pass |
| cb_exp_100k/cuda1_native_t10000 | 177.744s | 505.624s | 2.8x | ok (conv=1.000, n_error=0) | pass |
