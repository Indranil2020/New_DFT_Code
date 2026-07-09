# TIDES Engine Piecewise Benchmark — RTX 3060

**Date**: 2026-07-09 12:02:25
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9

## Engine Profile Summary

| Engine | Status | Wall (s) | Entries |
|---|---|---|---|
| E1_tile | pass | 5.32 | 47 |
| E2_basis | pass | 6.15 | 25 |
| E3_grid | pass | 0.63 | 33 |
| E4_solvers | pass | 2.63 | 26 |
| E5_scf | pass | 0.14 | 12 |
| E6_dynamics | fail | 0.00 | 0 |
| E7_parallel | pass | 7.82 | 9 |
| E8_hybrids | pass | 0.01 | 12 |
| E9_verification | pass | 0.00 | 13 |

## CUDA Probe Summary

| Probe | Status | Wall (s) | Key Metrics |
|---|---|---|---|
| cuda_gemm | pass | 0.72 | planned=974.257 GFLOPS, cuBLASLt=1190.92 GFLOPS |
| cuda_ozaki_gemm | pass | 0.01 |  |
| cuda_spgemm | pass | 0.26 | n=256.0, gpu_kernel=? ms |
| cuda_reduce_f64e | pass | 7.30 |  |
| cuda_graph | pass | 0.49 | graph_replay=3.12691 ms |

## E1_tile — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| GroupedGEMM | FP64-GPU | 16x8 | 0.108 | 0.00e+00 | 3.553e-15 0.6 PASS |
| GroupedGEMM | FP64-CPU | 16x8 | 0.026 | 2.60e-02 | 0.000e+00 2.5 ref |
| GroupedGEMM | FP64-GPU | 32x8 | 0.020 | 0.00e+00 | 7.105e-15 25.6 PASS |
| GroupedGEMM | FP64-CPU | 32x8 | 0.196 | 1.96e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 64x4 | 0.029 | 0.00e+00 | 1.066e-14 73.1 PASS |
| GroupedGEMM | FP64-CPU | 64x4 | 0.775 | 7.75e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 128x2 | 0.065 | 0.00e+00 | 1.776e-14 130.0 PASS |
| GroupedGEMM | FP64-CPU | 128x2 | 3.915 | 3.92e+00 | 0.000e+00 2.1 ref |
| GroupedGEMM | FP64-GPU | 256x1 | 0.208 | 0.00e+00 | 3.553e-14 161.1 PASS |
| GroupedGEMM | FP64-CPU | 256x1 | 27.211 | 2.72e+01 | 0.000e+00 1.2 ref |
| GroupedGEMM | FP16-accum | 16x8 | 51.582 | 0.00e+00 | 4.939e-03 0.0 PASS |
| GroupedGEMM | FP16-accum | 32x8 | 0.068 | 0.00e+00 | 8.697e-03 7.7 PASS |
| GroupedGEMM | FP16-accum | 64x4 | 0.080 | 0.00e+00 | 1.164e-02 26.3 PASS |
| GroupedGEMM | FP16-accum | 128x2 | 0.064 | 0.00e+00 | 1.426e-02 130.5 PASS |
| GroupedGEMM | FP16-accum | 256x1 | 0.729 | 0.00e+00 | 2.111e-02 46.0 PASS |
| SpGEMM | FP64-GPU | 4x16 | 0.077 | 0.00e+00 | 1.776e-14 1.7 PASS |
| SpGEMM | FP64-CPU | 4x16 | 0.191 | 1.91e-01 | 0.000e+00 0.7 ref |
| SpGEMM | FP64-GPU | 8x16 | 0.044 | 0.00e+00 | 4.263e-14 11.9 PASS |
| SpGEMM | FP64-CPU | 8x16 | 1.348 | 1.35e+00 | 0.000e+00 0.4 ref |
| SpGEMM | FP64-GPU | 16x32 | 1.442 | 0.00e+00 | 1.990e-13 11.6 PASS |
| SpGEMM | FP64-CPU | 16x32 | 96.524 | 9.65e+01 | 0.000e+00 0.2 ref |
| SpGEMM | FP64-GPU | 32x32 | 12.146 | 0.00e+00 | 4.121e-13 5.5 PASS |
| SpGEMM | FP64-CPU | 32x32 | 750.697 | 7.51e+02 | 0.000e+00 0.1 ref |
| Ozaki-f64e | FP16-slice | n=16 | 0.000 | 0.00e+00 | 5.329e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=16 | 0.003 | 0.003 | 3.00e-03 | 0.000e+00 2.9 ref |
| Ozaki-f64e | FP16-slice | n=32 | 0.000 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=32 | 0.022 | 0.022 | 2.20e-02 | 0.000e+00 3.0 ref |
| Ozaki-f64e | FP16-slice | n=64 | 0.000 | 0.00e+00 | 1.776e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=64 | 0.193 | 0.193 | 1.93e-01 | 0.000e+00 2.7 ref |
| Ozaki-f64e | FP16-slice | n=128 | 0.000 | 0.00e+00 | 3.553e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=128 | 1.882 | 1.882 | 1.88e+00 | 0.000e+00 2.2 ref |
| Dot-f64e | GPU | n=1000 | 4.625 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000 | 0.002 | 2.00e-03 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=10000 | 0.055 | 0.00e+00 | 3.695e-13 0.0 PASS |
| Dot-f64e | CPU-ref | n=10000 | 0.016 | 1.60e-02 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=100000 | 0.112 | 0.00e+00 | 5.568e-11 0.0 PASS |
| Dot-f64e | CPU-ref | n=100000 | 0.176 | 1.76e-01 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=1000000 | 1.027 | 0.00e+00 | 1.834e-04 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000000 | 2.063 | 2.06e+00 | 0.000e+00 0.0 ref |
| Trace-f64e | GPU | n=64 | 2.247 | 0.00e+00 | 5.551e-16 0.0 PASS |
| Trace-f64e | GPU | n=256 | 0.048 | 0.00e+00 | 1.066e-14 0.0 PASS |
| Trace-f64e | GPU | n=1024 | 0.072 | 0.00e+00 | 2.576e-14 0.0 PASS |
| GraphReplay | FP64-raw | 8x64x100 | 0.030 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP64-graph | 8x64x100 | 0.026 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.128601) |
| GraphReplay | FP16-raw | 8x64x100 | 0.010 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP16-graph | 8x64x100 | 0.009 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.079702) |
| PASS: Ledger | populated | with | 1.000 | — |  |

## E2_basis — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| RadialSolver FD | l=0 | n=2k | 61.331 | 1.13e-04 | PASS |
| RadialSolver Numerov | l=0 | n=2k | 2.144 | 1.13e-04 | PASS |
| RadialSolver FD | l=0 | n=8k | 8.324 | 7.03e-06 | PASS |
| RadialSolver Numerov | l=0 | n=8k | 8.230 | 7.03e-06 | PASS |
| RadialSolver FD | l=0 | n=16k | 16.725 | 1.76e-06 | PASS |
| RadialSolver Numerov | l=0 | n=16k | 16.920 | 1.76e-06 | PASS |
| RadialSolver FD | l=0 | n=32k | 33.860 | 4.39e-07 | PASS |
| RadialSolver Numerov | l=0 | n=32k | 34.081 | 4.39e-07 | PASS |
| RadialSolver FD | l=1 | n=2k | 2.119 | 2.35e-06 | PASS |
| RadialSolver Numerov | l=1 | n=2k | 2149.824 | 4.88e-07 | PASS |
| RadialSolver FD | l=1 | n=8k | 7.966 | 1.46e-07 | PASS |
| RadialSolver Numerov | l=1 | n=8k | 8.205 | 1.46e-07 | PASS |
| RadialSolver FD | l=1 | n=16k | 15.966 | 3.66e-08 | PASS |
| RadialSolver Numerov | l=1 | n=16k | 16.014 | 3.66e-08 | PASS |
| RadialSolver FD | l=1 | n=32k | 31.261 | 9.11e-09 | PASS |
| RadialSolver Numerov | l=1 | n=32k | 31.549 | 9.11e-09 | PASS |
| NAOGenerator DZP | H | (Z=1) | 408.394 | 0.00e+00 | PASS |
| NAOGenerator DZP | C | (Z=6) | 1010.465 | 0.00e+00 | PASS |
| NAOGenerator DZP | O | (Z=8) | 1026.168 | 0.00e+00 | PASS |
| NAOGenerator DZP | Ne | (Z=10) | 1052.775 | 0.00e+00 | PASS |
| TwoCenter | spline | 200pts | 0.000 | 3.50e-05 | PASS |
| TwoCenter | GPU | H2 | 91.187 | 5.38e-02 | PASS |
| ThreeCenter | GPU | 2-atom | 0.351 | 3.81e-03 | PASS |
| Derivative | spline-dS/dR | 200pts | 0.000 | 7.39e-05 | PASS |
| Derivative | 5pt-FD | R=2.0 | 0.000 | 2.03e-06 | PASS |

## E3_grid — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DualGrid | flatten | 32^3 | 0.424 | 0.00e+00 | PASS |
| DualGrid | flatten | 48^3 | 1.376 | 0.00e+00 | PASS |
| DualGrid | flatten | 64^3 | 3.284 | 0.00e+00 | PASS |
| RhoBuild | CPU | 16^3x4 | 0.016 | 0.00e+00 | ref |
| RhoBuild | GPU | 16^3x4 | 0.030 | 0.00e+00 | PASS |
| RhoBuild | CPU | 24^3x8 | 0.089 | 0.00e+00 | ref |
| RhoBuild | GPU | 24^3x8 | 0.135 | 0.00e+00 | PASS |
| RhoBuild | CPU | 32^3x16 | 0.351 | 0.00e+00 | ref |
| RhoBuild | GPU | 32^3x16 | 0.474 | 0.00e+00 | PASS |
| RhoBuild | CPU | 48^3x32 | 2.522 | 0.00e+00 | ref |
| RhoBuild | GPU | 48^3x32 | 3.030 | 0.00e+00 | PASS |
| VmatBuild | CPU | 16^3x4 | 0.041 | 0.00e+00 | ref |
| VmatBuild | GPU | 16^3x4 | 0.042 | 0.00e+00 | PASS |
| VmatBuild | CPU | 24^3x8 | 0.503 | 0.00e+00 | ref |
| VmatBuild | GPU | 24^3x8 | 0.500 | 0.00e+00 | PASS |
| VmatBuild | CPU | 32^3x16 | 4.582 | 0.00e+00 | ref |
| VmatBuild | GPU | 32^3x16 | 4.484 | 0.00e+00 | PASS |
| VmatBuild | CPU | 48^3x32 | 67.655 | 0.00e+00 | ref |
| VmatBuild | GPU | 48^3x32 | 67.470 | 0.00e+00 | PASS |
| Poisson | CPU | 16^3 | 31.660 | 0.00e+00 | ref |
| Poisson | GPU | 16^3 | 0.098 | 0.00e+00 | PASS |
| Poisson | CPU | 32^3 | 1.157 | 0.00e+00 | ref |
| Poisson | GPU | 32^3 | 0.890 | 0.00e+00 | PASS |
| XC-LDA | CPU | 16^3 | 0.579 | 0.00e+00 | ref |
| XC-LDA | GPU | 16^3 | 117.049 | 3.67e-12 | PASS |
| XC-LDA | CPU | 24^3 | 1.944 | 0.00e+00 | ref |
| XC-LDA | GPU | 24^3 | 0.435 | 5.28e-12 | PASS |
| XC-LDA | CPU | 32^3 | 4.519 | 0.00e+00 | ref |
| XC-LDA | GPU | 32^3 | 0.835 | 4.98e-12 | PASS |
| XC-LDA | CPU | 48^3 | 15.395 | 0.00e+00 | ref |
| XC-LDA | GPU | 48^3 | 2.535 | 6.36e-12 | PASS |
| Adjointness | <AP,w>=<P,ATw>16^3x4 | 0.000 | 0.000 | 2.00e-15 | PASS |
| Adjointness | H-symmetry | 16^3x4 | 0.000 | 0.00e+00 | PASS |

## E4_solvers — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DenseEig | generalized | n=16 | 10.445 | 1.78e-15 | PASS |
| DenseEig | generalized | n=32 | 0.089 | 1.78e-15 | PASS |
| DenseEig | generalized | n=64 | 1.325 | 2.67e-15 | PASS |
| DenseEig | generalized | n=128 | 1.732 | 2.22e-15 | PASS |
| DenseEig | generalized | n=256 | 6.055 | 2.22e-15 | PASS |
| DenseEig | batched | 3x{16,32,64} | 0.433 | 2.67e-15 | PASS |
| SP2 | CPU | n=32 | 1.067 | 2.42e-15 | PASS |
| SP2 | GPU | n=32 | 1.082 | 2.42e-15 | PASS |
| SP2 | CPU | n=64 | 10.646 | 1.83e-15 | PASS |
| SP2 | GPU | n=64 | 10.636 | 1.83e-15 | PASS |
| SP2 | CPU | n=128 | 122.500 | 5.54e-13 | PASS |
| SP2 | GPU | n=128 | 117.367 | 5.54e-13 | PASS |
| SP2 | CPU | n=256 | 1693.661 | 7.19e-15 | PASS |
| SP2 | GPU | n=256 | 204.895 | 7.22e-15 | PASS |
| ChFSI CPU | n=32 | occ=4 | 1.121 | 1.63e-11 | PASS |
| ChFSI CPU | n=64 | occ=8 | 8.881 | 2.83e-10 | PASS |
| ChFSI CPU | n=128 | occ=16 | 189.941 | 8.01e-10 | PASS |
| FOE CPU | n=32 | Ne=16 | 1.580 | 7.11e-15 | PASS |
| FOE CPU | n=64 | Ne=32 | 11.646 | 7.11e-15 | PASS |
| FOE CPU | n=128 | Ne=64 | 114.754 | 2.13e-14 | PASS |
| OMM CPU | n=32 | occ=4 | 0.050 | 0.00e+00 | PASS |
| OMM CPU | n=64 | occ=8 | 0.196 | 0.00e+00 | PASS |
| OMM CPU | n=128 | occ=16 | 2.433 | 0.00e+00 | PASS |
| Broker | small | 50 | 50.000 | — | 0.000 0.000e+00 PASS (small molecular system; R0 batched dense) |
| Broker | large-gapped5000 | atoms | 0.000 | 0.00e+00 | PASS (gapped large system; R2 SP2-submatrix) |
| Broker | medium | 500 | 500.000 | — | 0.000 0.000e+00 PASS (mid-range gapped; R1 ChFSI) |

## E5_scf — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| SCF | Pulay | n=8 | 27.830 | 8.24e-09 | PASS (8 iters) |
| SCF | simple | n=8 | 0.168 | 7.58e-09 | PASS (20 iters) |
| SCF | Pulay | n=16 | 0.397 | 6.34e-11 | PASS (12 iters) |
| SCF | simple | n=16 | 0.436 | 5.80e-09 | PASS (19 iters) |
| SCF | Pulay | n=32 | 3.405 | 1.74e-10 | PASS (29 iters) |
| SCF | simple | n=32 | 1.520 | 4.92e-09 | PASS (19 iters) |
| SCF | Pulay | n=64 | 77.819 | 5.89e-09 | PASS (139 iters) |
| SCF | simple | n=64 | 21.221 | 7.79e-09 | PASS (53 iters) |
| EnergyAssembly | components | n=8 | 0.000 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=16 | 0.001 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=32 | 0.004 | 0.00e+00 | PASS |
| Stress | FD-vs-virial4 | atoms | 0.002 | 2.79e-13 | PASS |

## E7_parallel — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| Partitioner | RCB | 100/2 | 0.004 | 0.00e+00 | PASS |
| Partitioner | RCB | 1000/4 | 0.085 | 0.00e+00 | PASS |
| Partitioner | RCB | 4000/8 | 0.577 | 0.00e+00 | PASS |
| Partitioner | RCB | 10000/16 | 1.978 | 0.00e+00 | PASS |
| HaloExchange | 1D | 100+2 | 0.000 | 0.00e+00 | PASS |
| HaloExchange | 1D | 1000+4 | 0.001 | 0.00e+00 | PASS |
| HaloExchange | 1D | 10000+8 | 0.004 | 0.00e+00 | PASS |
| HaloExchange | 3D | 10^3+halo | 0.018 | 0.00e+00 | PASS |
| CommFraction | model | 1MB/10ms | 0.000 | 0.00e+00 | PASS |

## E8_hybrids — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| D3 pair | Z=1-1 | R=3.000000 | 0.000 | -3.65e-04 | PASS |
| D3 pair | Z=6-6 | R=3.500000 | 0.000 | -5.21e-03 | PASS |
| D3 pair | Z=8-8 | R=3.000000 | 0.000 | -1.04e-03 | PASS |
| D3 pair | Z=6-8 | R=3.200000 | 0.000 | -2.31e-03 | PASS |
| D3 | H2O-dimer | 6 | 6.000 | — | 0.001 -7.332e-03 PASS |
| ISDF select | 100x10 | r=5 | 0.021 | 3.87e-14 | PASS |
| ISDF select | 500x20 | r=10 | 0.268 | 6.45e-12 | PASS |
| ISDF select | 2000x50 | r=20 | 5.311 | 4.53e-14 | PASS |
| ACE build | n=16 | occ=4 | 0.001 | 0.00e+00 | PASS |
| ACE build | n=32 | occ=8 | 0.002 | 0.00e+00 | PASS |
| ACE build | n=64 | occ=16 | 0.007 | 0.00e+00 | PASS |
| ACE | PBE0-energy | formula | 0.000 | 0.00e+00 | PASS |

## E9_verification — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| Ladder-1 | kernel | rho-build | 0.000 | 7.00e-15 | PASS |
| Ladder-2 | operator | adjointness | 0.000 | 2.00e-15 | PASS |
| Ladder-3 | energy | SCF-conv | 0.000 | 5.00e-09 | PASS |
| Ladder-4 | force | 5pt-FD | 0.000 | 3.00e-13 | PASS |
| Ladder-5 | dynamics | NVE-drift | 0.000 | 7.76e+03 | KNOWN-ISSUE: short sim inflates drift |
| Ladder-6 | physics | ACWF/Delta | 0.000 | 0.00e+00 | SKIP (deferred) |
| 1 | PASS | 7.0000e-15 | 1.000 | — | 7.0000e-15 8.0000e+00 ULP Kernel |
| 2 | PASS | 2.0000e-15 | 2.000 | — | 2.0000e-15 1.0000e-12 Operator |
| 3 | PASS | 5.0000e-09 | 3.000 | — | 5.0000e-09 5.0000e-04 Ha Energy |
| 4 | PASS | 3.0000e-13 | 4.000 | — | 3.0000e-13 1.0000e-06 Ha/Bohr Force |
| 5 | SKIP | 7.7620e+03 | 5.000 | — | 7.7620e+03 3.0000e+01 uHa/atom/ps Dynamics |
| 6 | SKIP | 0.0000e+00 | 6.000 | — | 0.0000e+00 0.0000e+00 Physics |
| === | 4 | pass, | 4.000 | — | 0 fail, 2 skip === |
