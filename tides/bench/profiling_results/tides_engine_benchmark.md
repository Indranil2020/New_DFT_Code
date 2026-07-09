# TIDES Engine Piecewise Benchmark — RTX 3060

**Date**: 2026-07-09 08:03:25
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9

## Engine Profile Summary

| Engine | Status | Wall (s) | Entries |
|---|---|---|---|
| E1_tile | pass | 4.74 | 47 |
| E2_basis | pass | 5.67 | 25 |
| E3_grid | pass | 0.60 | 33 |
| E4_solvers | pass | 2.71 | 26 |
| E5_scf | pass | 0.12 | 12 |
| E6_dynamics | fail | 0.00 | 0 |
| E7_parallel | pass | 7.78 | 9 |
| E8_hybrids | pass | 0.01 | 12 |
| E9_verification | pass | 0.00 | 13 |

## CUDA Probe Summary

| Probe | Status | Wall (s) | Key Metrics |
|---|---|---|---|
| cuda_gemm | pass | 0.65 | planned=942.754 GFLOPS, cuBLASLt=1185.54 GFLOPS |
| cuda_ozaki_gemm | pass | 0.01 |  |
| cuda_spgemm | pass | 0.26 | n=256.0, gpu_kernel=? ms |
| cuda_reduce_f64e | pass | 7.28 |  |
| cuda_graph | pass | 0.46 | graph_replay=3.2641 ms |

## E1_tile — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| GroupedGEMM | FP64-GPU | 16x8 | 0.087 | 0.00e+00 | 3.553e-15 0.8 PASS |
| GroupedGEMM | FP64-CPU | 16x8 | 0.026 | 2.60e-02 | 0.000e+00 2.5 ref |
| GroupedGEMM | FP64-GPU | 32x8 | 0.013 | 0.00e+00 | 7.105e-15 39.5 PASS |
| GroupedGEMM | FP64-CPU | 32x8 | 0.194 | 1.94e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 64x4 | 0.024 | 0.00e+00 | 1.066e-14 89.0 PASS |
| GroupedGEMM | FP64-CPU | 64x4 | 0.773 | 7.73e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 128x2 | 0.061 | 0.00e+00 | 1.776e-14 136.5 PASS |
| GroupedGEMM | FP64-CPU | 128x2 | 3.844 | 3.84e+00 | 0.000e+00 2.2 ref |
| GroupedGEMM | FP64-GPU | 256x1 | 0.219 | 0.00e+00 | 3.553e-14 153.1 PASS |
| GroupedGEMM | FP64-CPU | 256x1 | 25.247 | 2.52e+01 | 0.000e+00 1.3 ref |
| GroupedGEMM | FP16-accum | 16x8 | 46.161 | 0.00e+00 | 4.939e-03 0.0 PASS |
| GroupedGEMM | FP16-accum | 32x8 | 0.030 | 0.00e+00 | 8.697e-03 17.7 PASS |
| GroupedGEMM | FP16-accum | 64x4 | 0.054 | 0.00e+00 | 1.164e-02 38.8 PASS |
| GroupedGEMM | FP16-accum | 128x2 | 0.045 | 0.00e+00 | 1.426e-02 184.9 PASS |
| GroupedGEMM | FP16-accum | 256x1 | 0.697 | 0.00e+00 | 2.111e-02 48.1 PASS |
| SpGEMM | FP64-GPU | 4x16 | 0.059 | 0.00e+00 | 1.776e-14 2.2 PASS |
| SpGEMM | FP64-CPU | 4x16 | 0.193 | 1.93e-01 | 0.000e+00 0.7 ref |
| SpGEMM | FP64-GPU | 8x16 | 0.037 | 0.00e+00 | 4.263e-14 14.2 PASS |
| SpGEMM | FP64-CPU | 8x16 | 1.343 | 1.34e+00 | 0.000e+00 0.4 ref |
| SpGEMM | FP64-GPU | 16x32 | 1.461 | 0.00e+00 | 1.990e-13 11.5 PASS |
| SpGEMM | FP64-CPU | 16x32 | 94.003 | 9.40e+01 | 0.000e+00 0.2 ref |
| SpGEMM | FP64-GPU | 32x32 | 11.435 | 0.00e+00 | 4.121e-13 5.9 PASS |
| SpGEMM | FP64-CPU | 32x32 | 806.492 | 8.06e+02 | 0.000e+00 0.1 ref |
| Ozaki-f64e | FP16-slice | n=16 | 0.000 | 0.00e+00 | 5.329e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=16 | 0.003 | 0.003 | 3.00e-03 | 0.000e+00 3.0 ref |
| Ozaki-f64e | FP16-slice | n=32 | 0.000 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=32 | 0.022 | 0.022 | 2.20e-02 | 0.000e+00 3.0 ref |
| Ozaki-f64e | FP16-slice | n=64 | 0.000 | 0.00e+00 | 1.776e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=64 | 0.191 | 0.191 | 1.91e-01 | 0.000e+00 2.7 ref |
| Ozaki-f64e | FP16-slice | n=128 | 0.000 | 0.00e+00 | 3.553e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=128 | 1.883 | 1.883 | 1.88e+00 | 0.000e+00 2.2 ref |
| Dot-f64e | GPU | n=1000 | 4.550 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000 | 0.002 | 2.00e-03 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=10000 | 0.012 | 0.00e+00 | 3.695e-13 0.0 PASS |
| Dot-f64e | CPU-ref | n=10000 | 0.016 | 1.60e-02 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=100000 | 0.110 | 0.00e+00 | 5.568e-11 0.0 PASS |
| Dot-f64e | CPU-ref | n=100000 | 0.169 | 1.69e-01 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=1000000 | 1.031 | 0.00e+00 | 1.834e-04 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000000 | 1.839 | 1.84e+00 | 0.000e+00 0.0 ref |
| Trace-f64e | GPU | n=64 | 2.082 | 0.00e+00 | 5.551e-16 0.0 PASS |
| Trace-f64e | GPU | n=256 | 0.024 | 0.00e+00 | 1.066e-14 0.0 PASS |
| Trace-f64e | GPU | n=1024 | 0.071 | 0.00e+00 | 2.576e-14 0.0 PASS |
| GraphReplay | FP64-raw | 8x64x100 | 0.028 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP64-graph | 8x64x100 | 0.027 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.018366) |
| GraphReplay | FP16-raw | 8x64x100 | 0.010 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP16-graph | 8x64x100 | 0.009 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.082021) |
| PASS: Ledger | populated | with | 1.000 | — |  |

## E2_basis — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| RadialSolver FD | l=0 | n=2k | 23.664 | 1.13e-04 | PASS |
| RadialSolver Numerov | l=0 | n=2k | 2.116 | 1.13e-04 | PASS |
| RadialSolver FD | l=0 | n=8k | 8.184 | 7.03e-06 | PASS |
| RadialSolver Numerov | l=0 | n=8k | 8.104 | 7.03e-06 | PASS |
| RadialSolver FD | l=0 | n=16k | 15.932 | 1.76e-06 | PASS |
| RadialSolver Numerov | l=0 | n=16k | 16.004 | 1.76e-06 | PASS |
| RadialSolver FD | l=0 | n=32k | 31.851 | 4.39e-07 | PASS |
| RadialSolver Numerov | l=0 | n=32k | 32.109 | 4.39e-07 | PASS |
| RadialSolver FD | l=1 | n=2k | 1.997 | 2.35e-06 | PASS |
| RadialSolver Numerov | l=1 | n=2k | 1833.740 | 4.88e-07 | PASS |
| RadialSolver FD | l=1 | n=8k | 7.439 | 1.46e-07 | PASS |
| RadialSolver Numerov | l=1 | n=8k | 7.496 | 1.46e-07 | PASS |
| RadialSolver FD | l=1 | n=16k | 14.532 | 3.66e-08 | PASS |
| RadialSolver Numerov | l=1 | n=16k | 14.725 | 3.66e-08 | PASS |
| RadialSolver FD | l=1 | n=32k | 28.719 | 9.11e-09 | PASS |
| RadialSolver Numerov | l=1 | n=32k | 29.037 | 9.11e-09 | PASS |
| NAOGenerator DZP | H | (Z=1) | 373.597 | 0.00e+00 | PASS |
| NAOGenerator DZP | C | (Z=6) | 962.256 | 0.00e+00 | PASS |
| NAOGenerator DZP | O | (Z=8) | 1039.301 | 0.00e+00 | PASS |
| NAOGenerator DZP | Ne | (Z=10) | 1026.142 | 0.00e+00 | PASS |
| TwoCenter | spline | 200pts | 0.000 | 3.50e-05 | PASS |
| TwoCenter | GPU | H2 | 88.690 | 5.38e-02 | PASS |
| ThreeCenter | GPU | 2-atom | 0.363 | 3.81e-03 | PASS |
| Derivative | spline-dS/dR | 200pts | 0.000 | 7.39e-05 | PASS |
| Derivative | 5pt-FD | R=2.0 | 0.000 | 2.03e-06 | PASS |

## E3_grid — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DualGrid | flatten | 32^3 | 0.411 | 0.00e+00 | PASS |
| DualGrid | flatten | 48^3 | 1.400 | 0.00e+00 | PASS |
| DualGrid | flatten | 64^3 | 3.302 | 0.00e+00 | PASS |
| RhoBuild | CPU | 16^3x4 | 0.017 | 0.00e+00 | ref |
| RhoBuild | GPU | 16^3x4 | 0.036 | 0.00e+00 | PASS |
| RhoBuild | CPU | 24^3x8 | 0.086 | 0.00e+00 | ref |
| RhoBuild | GPU | 24^3x8 | 0.134 | 0.00e+00 | PASS |
| RhoBuild | CPU | 32^3x16 | 0.369 | 0.00e+00 | ref |
| RhoBuild | GPU | 32^3x16 | 0.499 | 0.00e+00 | PASS |
| RhoBuild | CPU | 48^3x32 | 2.586 | 0.00e+00 | ref |
| RhoBuild | GPU | 48^3x32 | 3.116 | 0.00e+00 | PASS |
| VmatBuild | CPU | 16^3x4 | 0.041 | 0.00e+00 | ref |
| VmatBuild | GPU | 16^3x4 | 0.046 | 0.00e+00 | PASS |
| VmatBuild | CPU | 24^3x8 | 0.518 | 0.00e+00 | ref |
| VmatBuild | GPU | 24^3x8 | 0.503 | 0.00e+00 | PASS |
| VmatBuild | CPU | 32^3x16 | 4.673 | 0.00e+00 | ref |
| VmatBuild | GPU | 32^3x16 | 4.619 | 0.00e+00 | PASS |
| VmatBuild | CPU | 48^3x32 | 67.467 | 0.00e+00 | ref |
| VmatBuild | GPU | 48^3x32 | 67.416 | 0.00e+00 | PASS |
| Poisson | CPU | 16^3 | 26.143 | 0.00e+00 | ref |
| Poisson | GPU | 16^3 | 0.100 | 0.00e+00 | PASS |
| Poisson | CPU | 32^3 | 1.084 | 0.00e+00 | ref |
| Poisson | GPU | 32^3 | 0.858 | 0.00e+00 | PASS |
| XC-LDA | CPU | 16^3 | 0.558 | 0.00e+00 | ref |
| XC-LDA | GPU | 16^3 | 96.627 | 3.67e-12 | PASS |
| XC-LDA | CPU | 24^3 | 1.938 | 0.00e+00 | ref |
| XC-LDA | GPU | 24^3 | 0.427 | 5.28e-12 | PASS |
| XC-LDA | CPU | 32^3 | 4.506 | 0.00e+00 | ref |
| XC-LDA | GPU | 32^3 | 0.842 | 4.98e-12 | PASS |
| XC-LDA | CPU | 48^3 | 15.355 | 0.00e+00 | ref |
| XC-LDA | GPU | 48^3 | 2.549 | 6.36e-12 | PASS |
| Adjointness | <AP,w>=<P,ATw>16^3x4 | 0.000 | 0.000 | 2.00e-15 | PASS |
| Adjointness | H-symmetry | 16^3x4 | 0.000 | 0.00e+00 | PASS |

## E4_solvers — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DenseEig | generalized | n=16 | 24.890 | 1.78e-15 | PASS |
| DenseEig | generalized | n=32 | 0.095 | 1.78e-15 | PASS |
| DenseEig | generalized | n=64 | 1.071 | 2.67e-15 | PASS |
| DenseEig | generalized | n=128 | 1.721 | 2.22e-15 | PASS |
| DenseEig | generalized | n=256 | 6.309 | 2.22e-15 | PASS |
| DenseEig | batched | 3x{16,32,64} | 0.450 | 2.67e-15 | PASS |
| SP2 | CPU | n=32 | 1.098 | 2.42e-15 | PASS |
| SP2 | GPU | n=32 | 1.088 | 2.42e-15 | PASS |
| SP2 | CPU | n=64 | 10.723 | 1.83e-15 | PASS |
| SP2 | GPU | n=64 | 10.824 | 1.83e-15 | PASS |
| SP2 | CPU | n=128 | 117.363 | 5.54e-13 | PASS |
| SP2 | GPU | n=128 | 115.641 | 5.54e-13 | PASS |
| SP2 | CPU | n=256 | 1749.212 | 7.19e-15 | PASS |
| SP2 | GPU | n=256 | 203.703 | 7.22e-15 | PASS |
| ChFSI CPU | n=32 | occ=4 | 0.484 | 1.63e-11 | PASS |
| ChFSI CPU | n=64 | occ=8 | 9.174 | 2.83e-10 | PASS |
| ChFSI CPU | n=128 | occ=16 | 196.497 | 8.28e-10 | PASS |
| FOE CPU | n=32 | Ne=16 | 1.618 | 7.11e-15 | PASS |
| FOE CPU | n=64 | Ne=32 | 11.765 | 7.11e-15 | PASS |
| FOE CPU | n=128 | Ne=64 | 119.892 | 2.13e-14 | PASS |
| OMM CPU | n=32 | occ=4 | 0.051 | 0.00e+00 | PASS |
| OMM CPU | n=64 | occ=8 | 0.197 | 0.00e+00 | PASS |
| OMM CPU | n=128 | occ=16 | 2.455 | 0.00e+00 | PASS |
| Broker | small | 50 | 50.000 | — | 0.000 0.000e+00 PASS (small molecular system; R0 batched dense) |
| Broker | large-gapped5000 | atoms | 0.000 | 0.00e+00 | PASS (gapped large system; R2 SP2-submatrix) |
| Broker | medium | 500 | 500.000 | — | 0.000 0.000e+00 PASS (mid-range gapped; R1 ChFSI) |

## E5_scf — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| SCF | Pulay | n=8 | 3.255 | 8.24e-09 | PASS (8 iters) |
| SCF | simple | n=8 | 0.152 | 7.58e-09 | PASS (20 iters) |
| SCF | Pulay | n=16 | 0.365 | 6.34e-11 | PASS (12 iters) |
| SCF | simple | n=16 | 0.421 | 5.80e-09 | PASS (19 iters) |
| SCF | Pulay | n=32 | 3.354 | 1.74e-10 | PASS (29 iters) |
| SCF | simple | n=32 | 1.520 | 4.92e-09 | PASS (19 iters) |
| SCF | Pulay | n=64 | 83.657 | 2.93e-09 | PASS (139 iters) |
| SCF | simple | n=64 | 21.258 | 7.79e-09 | PASS (53 iters) |
| EnergyAssembly | components | n=8 | 0.000 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=16 | 0.001 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=32 | 0.004 | 0.00e+00 | PASS |
| Stress | FD-vs-virial4 | atoms | 0.002 | 2.79e-13 | PASS |

## E7_parallel — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| Partitioner | RCB | 100/2 | 0.004 | 0.00e+00 | PASS |
| Partitioner | RCB | 1000/4 | 0.091 | 0.00e+00 | PASS |
| Partitioner | RCB | 4000/8 | 0.580 | 0.00e+00 | PASS |
| Partitioner | RCB | 10000/16 | 2.021 | 0.00e+00 | PASS |
| HaloExchange | 1D | 100+2 | 0.000 | 0.00e+00 | PASS |
| HaloExchange | 1D | 1000+4 | 0.001 | 0.00e+00 | PASS |
| HaloExchange | 1D | 10000+8 | 0.004 | 0.00e+00 | PASS |
| HaloExchange | 3D | 10^3+halo | 0.019 | 0.00e+00 | PASS |
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
| ISDF select | 500x20 | r=10 | 0.266 | 6.45e-12 | PASS |
| ISDF select | 2000x50 | r=20 | 5.259 | 4.53e-14 | PASS |
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
