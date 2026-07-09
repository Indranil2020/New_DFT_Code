# TIDES vs gpu4pyscf — End-to-End Comparison Benchmark

**Date**: 2026-07-10 00:19:03
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9
**PySCF**: 2.11.0 | **gpu4pyscf**: 1.5.0

## 1. GEMM Performance

| n | numpy CPU GFLOPS | cupy GPU GFLOPS | TIDES planned GFLOPS | cuBLASLt GFLOPS | TIDES vs cuBLASLt |
|---|---|---|---|---|---|
| 64 | 32.2 | 10.083016817053313 | 906.9 | 1230.0 | 0.74x |
| 128 | 46.4 | 52.95922608885918 | 906.9 | 1230.0 | 0.74x |
| 256 | 76.6 | 87.75108601126936 | 906.9 | 1230.0 | 0.74x |
| 512 | 99.4 | 152.0308651061911 | 906.9 | 1230.0 | 0.74x |
| 1024 | 140.1 | 154.75480326001136 | 906.9 | 1230.0 | 0.74x |
| 2048 | 192.2 | 175.3473114437457 | 906.9 | 1230.0 | 0.74x |

## 2. Eigendecomposition Performance

| n | CPU ms (numpy) | GPU ms (cupy) | GPU Speedup |
|---|---|---|---|
| 32 | 0.15 | 0.83 | 0.2x |
| 64 | 1.16 | 1.99 | 0.6x |
| 128 | 33.08 | 5.63 | 5.9x |
| 256 | 8.04 | 8.98 | 0.9x |
| 512 | 47.32 | 21.50 | 2.2x |
| 1024 | 177.08 | 73.23 | 2.4x |

## 3. End-to-End SCF (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | PySCF CPU ms | gpu4pyscf ms | GPU Speedup |
|---|---|---|---|---|---|
| H | 1 | 5 | 326.7 | 67.0 | 4.9x |
| He | 1 | 5 | 349.6 | 44.3 | 7.9x |
| H2 | 2 | 10 | 310.6 | 63.2 | 4.9x |
| N2 | 2 | 28 | 452.1 | 73.4 | 6.2x |
| H2O | 3 | 24 | 1148.4 | 106.8 | 10.8x |
| NH3 | 4 | 29 | 1104.1 | 97.6 | 11.3x |
| CH4 | 5 | 34 | 1236.0 | 114.9 | 10.8x |
| C2H6 | 8 | 58 | 1669.6 | 162.7 | 10.3x |
| C6H6 | 12 | 114 | 2607.1 | 443.0 | 5.9x |
| C10H8 | 18 | 180 | 108955.1 | 17803.7 | 6.1x |
| H2O_8mer | 24 | 192 | 6275.8 | 557.0 | 11.3x |
| H2O_16mer | 48 | 384 | 21959.3 | 1207.4 | 18.2x |
| H2O_32mer | 96 | 768 | 42384.7 | 2735.9 | 15.5x |

## 4. SCF Basis Scan (H2O, LDA)

| Basis | nao | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|
| STO-3G | 7 | 1282.9 | 59.2 | 21.7x |
| 6-31G* | 18 | 1000.1 | 90.5 | 11.0x |
| cc-pVDZ | 24 | 793.4 | 88.8 | 8.9x |
| def2-SVP | 24 | 1006.3 | 89.2 | 11.3x |
| cc-pVTZ | 58 | 876.0 | 113.6 | 7.7x |
| def2-TZVPP | 59 | 980.1 | 130.3 | 7.5x |

## 5. Gradient Benchmark (cc-pVDZ, LDA)

| System | Atoms | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|
| H | 1 | 511.1 | 607.0 | 0.8x |
| He | 1 | 3.0 | 48.4 | 0.1x |
| H2 | 2 | 93.9 | 141.3 | 0.7x |
| N2 | 2 | 163.7 | 198.7 | 0.8x |
| H2O | 3 | 532.4 | 293.0 | 1.8x |
| NH3 | 4 | 614.7 | 332.7 | 1.8x |
| CH4 | 5 | 914.8 | 1159.2 | 0.8x |
| C2H6 | 8 | 1206.8 | 1905.4 | 0.6x |
| C6H6 | 12 | 4693.9 | 6090.9 | 0.8x |
| C10H8 | 18 | 16249.1 | 16529.1 | 1.0x |
| H2O_8mer | 24 | 11710.1 | 9604.0 | 1.2x |
| H2O_16mer | 48 | 50548.5 | 60793.8 | 0.8x |

## 6. TIDES Engine Profiles (E1-E9)

| Engine | Status | Wall (s) |
|---|---|---|
| E1_tile | pass | 6.21 |
| E2_basis | pass | 5.69 |
| E3_grid | pass | 0.59 |
| E4_solvers | pass | 2.72 |
| E5_scf | pass | 0.33 |
| E6_dynamics | fail | 0.00 |
| E7_parallel | pass | 7.93 |
| E8_hybrids | pass | 0.01 |
| E9_verification | pass | 0.00 |

## 7. Head-to-Head Summary

| Operation | PySCF CPU | gpu4pyscf GPU | TIDES GPU | TIDES vs gpu4pyscf |
|---|---|---|---|---|
| GEMM n=2048 | 192 GFLOPS | 175 GFLOPS | 907 GFLOPS | 5.17x |
| SCF (96 atoms) | 42385 ms | 2736 ms | — (engine profiles) | — |
