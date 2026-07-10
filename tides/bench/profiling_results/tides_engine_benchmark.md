# TIDES Engine Piecewise Benchmark — RTX 3060

**Date**: 2026-07-10 08:25:34
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9

## Engine Profile Summary

| Engine | Status | Wall (s) | Entries |
|---|---|---|---|
| E1_tile | pass | 6.02 | 47 |
| E2_basis | fail | 5.71 | 0 |
| E3_grid | pass | 0.63 | 35 |
| E4_solvers | pass | 2.67 | 26 |
| E5_scf | pass | 0.05 | 12 |
| E6_dynamics | pass | 0.00 | 4 |
| E7_parallel | pass | 7.77 | 9 |
| E8_hybrids | pass | 0.01 | 12 |
| E9_verification | pass | 0.00 | 13 |

## CUDA Probe Summary

| Probe | Status | Wall (s) | Key Metrics |
|---|---|---|---|
| cuda_gemm | pass | 0.68 | planned=982.772 GFLOPS, cuBLASLt=1171.43 GFLOPS |
| cuda_ozaki_gemm | pass | 0.01 |  |
| cuda_spgemm | pass | 0.34 | n=256.0, gpu_kernel=? ms |
| cuda_reduce_f64e | pass | 7.99 |  |
| cuda_graph | pass | 0.48 | graph_replay=2.84461 ms |

## E1_tile — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| GroupedGEMM | FP64-GPU | 16x8 | 0.125 | 0.00e+00 | 3.553e-15 0.5 PASS |
| GroupedGEMM | FP64-CPU | 16x8 | 0.025 | 2.50e-02 | 0.000e+00 2.6 ref |
| GroupedGEMM | FP64-GPU | 32x8 | 0.017 | 0.00e+00 | 7.105e-15 30.1 PASS |
| GroupedGEMM | FP64-CPU | 32x8 | 0.196 | 1.96e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 64x4 | 0.028 | 0.00e+00 | 1.066e-14 75.9 PASS |
| GroupedGEMM | FP64-CPU | 64x4 | 0.773 | 7.73e-01 | 0.000e+00 2.7 ref |
| GroupedGEMM | FP64-GPU | 128x2 | 0.068 | 0.00e+00 | 1.776e-14 124.2 PASS |
| GroupedGEMM | FP64-CPU | 128x2 | 3.852 | 3.85e+00 | 0.000e+00 2.2 ref |
| GroupedGEMM | FP64-GPU | 256x1 | 0.208 | 0.00e+00 | 3.553e-14 161.3 PASS |
| GroupedGEMM | FP64-CPU | 256x1 | 23.904 | 2.39e+01 | 0.000e+00 1.4 ref |
| GroupedGEMM | FP16io-FP32accum16x8 | 47.021 | 47.021 | 0.00e+00 | 4.939e-03 0.0 PASS |
| GroupedGEMM | FP16io-FP32accum32x8 | 0.039 | 0.039 | 0.00e+00 | 8.697e-03 13.4 PASS |
| GroupedGEMM | FP16io-FP32accum64x4 | 0.070 | 0.070 | 0.00e+00 | 1.164e-02 30.1 PASS |
| GroupedGEMM | FP16io-FP32accum128x2 | 0.047 | 0.047 | 0.00e+00 | 1.426e-02 178.1 PASS |
| GroupedGEMM | FP16io-FP32accum256x1 | 0.727 | 0.727 | 0.00e+00 | 2.111e-02 46.2 PASS |
| SpGEMM | FP64-GPU | 4x16 | 0.082 | 0.00e+00 | 7.105e-15 6.4 PASS |
| SpGEMM | FP64-CPU | 4x16 | 0.192 | 1.92e-01 | 0.000e+00 2.7 ref |
| SpGEMM | FP64-GPU | 8x16 | 0.151 | 0.00e+00 | 1.066e-14 27.8 PASS |
| SpGEMM | FP64-CPU | 8x16 | 1.472 | 1.47e+00 | 0.000e+00 2.8 ref |
| SpGEMM | FP64-GPU | 16x32 | 1.935 | 0.00e+00 | 4.263e-14 138.7 PASS |
| SpGEMM | FP64-CPU | 16x32 | 92.750 | 9.28e+01 | 0.000e+00 2.9 ref |
| SpGEMM | FP64-GPU | 32x32 | 15.349 | 0.00e+00 | 9.948e-14 139.9 PASS |
| SpGEMM | FP64-CPU | 32x32 | 747.943 | 7.48e+02 | 0.000e+00 2.9 ref |
| Ozaki-f64e | FP16-slice | n=16 | 0.000 | 0.00e+00 | 5.329e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=16 | 0.003 | 0.003 | 3.00e-03 | 0.000e+00 3.0 ref |
| Ozaki-f64e | FP16-slice | n=32 | 0.000 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=32 | 0.022 | 0.022 | 2.20e-02 | 0.000e+00 3.0 ref |
| Ozaki-f64e | FP16-slice | n=64 | 0.000 | 0.00e+00 | 1.776e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=64 | 0.278 | 0.278 | 2.78e-01 | 0.000e+00 1.9 ref |
| Ozaki-f64e | FP16-slice | n=128 | 0.000 | 0.00e+00 | 3.553e-14 0.0 PASS |
| Ozaki-f64e | FP64-CPU-refn=128 | 2.119 | 2.119 | 2.12e+00 | 0.000e+00 2.0 ref |
| Dot-f64e | GPU | n=1000 | 4.306 | 0.00e+00 | 7.105e-15 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000 | 0.002 | 2.00e-03 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=10000 | 0.050 | 0.00e+00 | 3.695e-13 0.0 PASS |
| Dot-f64e | CPU-ref | n=10000 | 0.016 | 1.60e-02 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=100000 | 0.153 | 0.00e+00 | 5.568e-11 0.0 PASS |
| Dot-f64e | CPU-ref | n=100000 | 0.159 | 1.59e-01 | 0.000e+00 0.0 ref |
| Dot-f64e | GPU | n=1000000 | 1.073 | 0.00e+00 | 1.834e-04 0.0 PASS |
| Dot-f64e | CPU-ref | n=1000000 | 1.865 | 1.86e+00 | 0.000e+00 0.0 ref |
| Trace-f64e | GPU | n=64 | 2.299 | 0.00e+00 | 5.551e-16 0.0 PASS |
| Trace-f64e | GPU | n=256 | 0.045 | 0.00e+00 | 1.066e-14 0.0 PASS |
| Trace-f64e | GPU | n=1024 | 0.073 | 0.00e+00 | 2.576e-14 0.0 PASS |
| GraphReplay | FP64-raw | 8x64x100 | 0.027 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP64-graph | 8x64x100 | 0.026 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.028429) |
| GraphReplay | FP16-raw | 8x64x100 | 0.008 | 0.00e+00 | 0.000e+00 0.0 ref |
| GraphReplay | FP16-graph | 8x64x100 | 0.007 | 0.00e+00 | 0.000e+00 0.0 PASS (x1.109231) |
| PASS: Ledger | populated | with | 1.000 | — |  |

## E3_grid — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DualGrid | flatten | 32^3 | 0.411 | 0.00e+00 | PASS |
| DualGrid | flatten | 48^3 | 1.388 | 0.00e+00 | PASS |
| DualGrid | flatten | 64^3 | 3.293 | 0.00e+00 | PASS |
| RhoBuild | CPU | 16^3x4 | 0.023 | 0.00e+00 | ref |
| RhoBuild | GPU | 16^3x4 | 0.030 | 0.00e+00 | PASS |
| RhoBuild | CPU | 24^3x8 | 0.085 | 0.00e+00 | ref |
| RhoBuild | GPU | 24^3x8 | 0.135 | 0.00e+00 | PASS |
| RhoBuild | CPU | 32^3x16 | 0.449 | 0.00e+00 | ref |
| RhoBuild | GPU | 32^3x16 | 0.498 | 0.00e+00 | PASS |
| RhoBuild | CPU | 48^3x32 | 2.576 | 0.00e+00 | ref |
| RhoBuild | GPU | 48^3x32 | 3.213 | 0.00e+00 | PASS |
| VmatBuild | CPU | 16^3x4 | 0.040 | 0.00e+00 | ref |
| VmatBuild | GPU | 16^3x4 | 0.041 | 0.00e+00 | PASS |
| VmatBuild | CPU | 24^3x8 | 0.488 | 0.00e+00 | ref |
| VmatBuild | GPU | 24^3x8 | 0.488 | 0.00e+00 | PASS |
| VmatBuild | CPU | 32^3x16 | 4.467 | 0.00e+00 | ref |
| VmatBuild | GPU | 32^3x16 | 4.489 | 0.00e+00 | PASS |
| VmatBuild | CPU | 48^3x32 | 67.140 | 0.00e+00 | ref |
| VmatBuild | GPU | 48^3x32 | 67.208 | 0.00e+00 | PASS |
| Poisson | CPU-FFTW | 16^3 | 21.264 | 0.00e+00 | ref |
| Poisson | GPU-cuFFT | 16^3 | 0.098 | 0.00e+00 | PASS |
| Poisson | CPU-FFTW | 32^3 | 1.013 | 0.00e+00 | ref |
| Poisson | GPU-cuFFT | 32^3 | 0.838 | 0.00e+00 | PASS |
| Poisson | CPU-FFTW | 64^3 | 11.394 | 0.00e+00 | ref |
| Poisson | GPU-cuFFT | 64^3 | 110.178 | 1.42e-14 | PASS |
| XC-LDA | CPU | 16^3 | 0.515 | 0.00e+00 | ref |
| XC-LDA | GPU | 16^3 | 0.240 | 3.33e-16 | PASS |
| XC-LDA | CPU | 24^3 | 1.696 | 0.00e+00 | ref |
| XC-LDA | GPU | 24^3 | 0.260 | 3.33e-16 | PASS |
| XC-LDA | CPU | 32^3 | 4.035 | 0.00e+00 | ref |
| XC-LDA | GPU | 32^3 | 0.581 | 3.33e-16 | PASS |
| XC-LDA | CPU | 48^3 | 13.572 | 0.00e+00 | ref |
| XC-LDA | GPU | 48^3 | 1.926 | 4.44e-16 | PASS |
| Adjointness | <AP,w>=<P,ATw>16^3x4 | 0.000 | 0.000 | 2.00e-15 | PASS |
| Adjointness | H-symmetry | 16^3x4 | 0.000 | 0.00e+00 | PASS |

## E4_solvers — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| DenseEig | generalized | n=16 | 41.062 | 1.78e-15 | PASS |
| DenseEig | generalized | n=32 | 0.627 | 1.33e-15 | PASS |
| DenseEig | generalized | n=64 | 0.606 | 1.33e-15 | PASS |
| DenseEig | generalized | n=128 | 1.176 | 1.78e-15 | PASS |
| DenseEig | generalized | n=256 | 3.208 | 2.22e-15 | PASS |
| DenseEig | batched | 3x{16,32,64} | 0.308 | 1.78e-15 | PASS |
| SP2 | CPU | n=32 | 1.081 | 2.42e-15 | PASS |
| SP2 | GPU | n=32 | 1.095 | 2.42e-15 | PASS |
| SP2 | CPU | n=64 | 10.956 | 1.83e-15 | PASS |
| SP2 | GPU | n=64 | 11.713 | 1.83e-15 | PASS |
| SP2 | CPU | n=128 | 125.615 | 5.54e-13 | PASS |
| SP2 | GPU | n=128 | 117.899 | 5.54e-13 | PASS |
| SP2 | CPU | n=256 | 1692.746 | 7.19e-15 | PASS |
| SP2 | GPU | n=256 | 211.844 | 7.22e-15 | PASS |
| ChFSI CPU | n=32 | occ=4 | 0.493 | 1.63e-11 | PASS |
| ChFSI CPU | n=64 | occ=8 | 8.855 | 2.83e-10 | PASS |
| ChFSI CPU | n=128 | occ=16 | 193.128 | 7.69e-10 | PASS |
| FOE CPU | n=32 | Ne=16 | 1.588 | 7.11e-15 | PASS |
| FOE CPU | n=64 | Ne=32 | 11.659 | 7.11e-15 | PASS |
| FOE CPU | n=128 | Ne=64 | 115.087 | 2.13e-14 | PASS |
| OMM CPU | n=32 | occ=4 | 0.054 | 0.00e+00 | PASS |
| OMM CPU | n=64 | occ=8 | 0.205 | 0.00e+00 | PASS |
| OMM CPU | n=128 | occ=16 | 2.544 | 0.00e+00 | PASS |
| Broker | small | 50 | 50.000 | — | 0.000 0.000e+00 PASS (small molecular system; R0 batched dense) |
| Broker | large-gapped5000 | atoms | 0.000 | 0.00e+00 | PASS (gapped large system; R2 SP2-submatrix) |
| Broker | medium | 500 | 500.000 | — | 0.000 0.000e+00 PASS (mid-range gapped; R1 ChFSI) |

## E5_scf — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| SCF | Pulay | n=8 | 26.404 | 7.38e-09 | PASS (9 iters) |
| SCF | simple | n=8 | 0.214 | 7.79e-09 | PASS (30 iters) |
| SCF | Pulay | n=16 | 0.345 | 6.60e-09 | PASS (12 iters) |
| SCF | simple | n=16 | 0.561 | 7.57e-09 | PASS (28 iters) |
| SCF | Pulay | n=32 | 1.292 | 5.19e-09 | PASS (9 iters) |
| SCF | simple | n=32 | 2.534 | 8.34e-09 | PASS (28 iters) |
| SCF | Pulay | n=64 | 5.052 | 1.18e-10 | PASS (13 iters) |
| SCF | simple | n=64 | 7.611 | 8.80e-09 | PASS (28 iters) |
| EnergyAssembly | components | n=8 | 0.001 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=16 | 0.001 | 0.00e+00 | PASS |
| EnergyAssembly | components | n=32 | 0.004 | 0.00e+00 | PASS |
| Stress | FD-vs-virial4 | atoms | 0.003 | 2.79e-13 | PASS |

## E6_dynamics — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| Forces | analytic-vs-FD4 | atoms | 0.003 | 2.94e-13 | PASS |
| XLBOMD | NVE-drift | 2 | 2.000 | — | 50000 steps, dt=0.2fs (10ps) 3.627 3.944e-02 PASS |
| XLBOMD | solves/step | 2 | 2.000 | — | 0.000 2.000e-05 PASS |
| Optimizer | FIRE | 2 | 2.000 | — | 0.005 9.748e-07 PASS (86 steps) |

## E7_parallel — Detailed Entries

| Kernel | Variant | Size | Time (ms) | Error | Status |
|---|---|---|---|---|---|
| Partitioner | RCB | 100/2 | 0.004 | 0.00e+00 | PASS |
| Partitioner | RCB | 1000/4 | 0.088 | 0.00e+00 | PASS |
| Partitioner | RCB | 4000/8 | 0.583 | 0.00e+00 | PASS |
| Partitioner | RCB | 10000/16 | 2.009 | 0.00e+00 | PASS |
| HaloExchange | 1D | 100+2 | 0.000 | 0.00e+00 | PASS |
| HaloExchange | 1D | 1000+4 | 0.000 | 0.00e+00 | PASS |
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
| ISDF select | 100x10 | r=5 | 0.020 | 3.87e-14 | PASS |
| ISDF select | 500x20 | r=10 | 0.267 | 6.45e-12 | PASS |
| ISDF select | 2000x50 | r=20 | 5.169 | 4.53e-14 | PASS |
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
| Ladder-5 | dynamics | NVE-drift | 0.000 | 2.50e+01 | PASS |
| Ladder-6 | physics | ACWF/Delta | 0.000 | 0.00e+00 | SKIP (deferred) |
| 1 | PASS | 7.0000e-15 | 1.000 | — | 7.0000e-15 8.0000e+00 ULP Kernel |
| 2 | PASS | 2.0000e-15 | 2.000 | — | 2.0000e-15 1.0000e-12 Operator |
| 3 | PASS | 5.0000e-09 | 3.000 | — | 5.0000e-09 5.0000e-04 Ha Energy |
| 4 | PASS | 3.0000e-13 | 4.000 | — | 3.0000e-13 1.0000e-06 Ha/Bohr Force |
| 5 | PASS | 2.5000e+01 | 5.000 | — | 2.5000e+01 3.0000e+01 uHa/atom/ps Dynamics |
| 6 | SKIP | 0.0000e+00 | 6.000 | — | 0.0000e+00 0.0000e+00 Physics |
| === | 5 | pass, | 5.000 | — | 0 fail, 1 skip === |
