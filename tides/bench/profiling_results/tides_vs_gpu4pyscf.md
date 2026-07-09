# TIDES vs gpu4pyscf — End-to-End Comparison Benchmark

**Date**: 2026-07-09 09:00:54
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9
**PySCF**: 2.11.0 | **gpu4pyscf**: 1.5.0

## 1. GEMM Performance

| n | numpy CPU GFLOPS | cupy GPU GFLOPS | TIDES planned GFLOPS | cuBLASLt GFLOPS | TIDES vs cuBLASLt |
|---|---|---|---|---|---|
| 64 | 8.0 | 9.651328948782753 | 965.9 | 1025.0 | 0.94x |
| 128 | 0.4 | 55.75228172471716 | 965.9 | 1025.0 | 0.94x |
| 256 | 23.8 | 88.43120677520032 | 965.9 | 1025.0 | 0.94x |
| 512 | 101.2 | 152.20059749162766 | 965.9 | 1025.0 | 0.94x |
| 1024 | 100.9 | 154.23656243453865 | 965.9 | 1025.0 | 0.94x |
| 2048 | 92.2 | 164.13752233112143 | 965.9 | 1025.0 | 0.94x |

## 2. Eigendecomposition Performance

| n | CPU ms (numpy) | GPU ms (cupy) | GPU Speedup |
|---|---|---|---|
| 32 | 0.12 | 0.82 | 0.1x |
| 64 | 1.16 | 2.04 | 0.6x |
| 128 | 1.83 | 5.81 | 0.3x |
| 256 | 12.54 | 9.54 | 1.3x |
| 512 | 45.87 | 24.22 | 1.9x |
| 1024 | 1261.07 | 78.11 | 16.1x |

## 3. End-to-End SCF (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | PySCF CPU ms | gpu4pyscf ms | GPU Speedup |
|---|---|---|---|---|---|
| H | 1 | 5 | 2380.1 | 127.2 | 18.7x |
| He | 1 | 5 | 1344.0 | 170.0 | 7.9x |
| H2 | 2 | 10 | 1394.3 | 80.8 | 17.2x |
| N2 | 2 | 28 | 1422.8 | 148.6 | 9.6x |
| H2O | 3 | 24 | 2883.6 | 256.9 | 11.2x |
| NH3 | 4 | 29 | 2578.4 | 187.3 | 13.8x |
| CH4 | 5 | 34 | 3829.9 | 234.9 | 16.3x |
| C2H6 | 8 | 58 | 4202.9 | 347.6 | 12.1x |
| C6H6 | 12 | 114 | 7940.6 | 542.2 | 14.6x |
| C10H8 | 18 | 180 | 250190.8 | 21088.6 | 11.9x |
| H2O_8mer | 24 | 192 | 17229.6 | 683.9 | 25.2x |
| H2O_16mer | 48 | 384 | 63952.7 | 1239.8 | 51.6x |
| H2O_32mer | 96 | 768 | 48048.5 | 2752.2 | 17.5x |

## 4. SCF Basis Scan (H2O, LDA)

| Basis | nao | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|
| STO-3G | 7 | 1557.1 | 61.1 | 25.5x |
| 6-31G* | 18 | 1399.4 | 89.0 | 15.7x |
| cc-pVDZ | 24 | 1181.8 | 92.3 | 12.8x |
| def2-SVP | 24 | 1070.9 | 91.0 | 11.8x |
| cc-pVTZ | 58 | 1253.9 | 117.1 | 10.7x |
| def2-TZVPP | 59 | 1414.6 | 139.1 | 10.2x |

## 5. Gradient Benchmark (cc-pVDZ, LDA)

| System | Atoms | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|
| H | 1 | 4.2 | 41.9 | 0.1x |
| He | 1 | 2.5 | 40.4 | 0.1x |
| H2 | 2 | 152.3 | 131.5 | 1.2x |
| N2 | 2 | 139.8 | 226.3 | 0.6x |
| H2O | 3 | 363.9 | 885.0 | 0.4x |
| NH3 | 4 | 145.8 | 298.4 | 0.5x |
| CH4 | 5 | 371.9 | 384.3 | 1.0x |
| C2H6 | 8 | 1515.4 | 1746.5 | 0.9x |
| C6H6 | 12 | 5545.2 | 6537.2 | 0.8x |
| C10H8 | 18 | 16780.9 | 20774.9 | 0.8x |
| H2O_8mer | 24 | 10893.1 | 12494.1 | 0.9x |
| H2O_16mer | 48 | 52208.3 | 59924.7 | 0.9x |

## 6. TIDES Engine Profiles (E1-E9)

| Engine | Status | Wall (s) |
|---|---|---|
| E1_tile | pass | 4.90 |
| E2_basis | pass | 5.81 |
| E3_grid | pass | 0.62 |
| E4_solvers | pass | 2.68 |
| E5_scf | pass | 0.17 |
| E6_dynamics | fail | 0.00 |
| E7_parallel | pass | 7.86 |
| E8_hybrids | pass | 0.01 |
| E9_verification | pass | 0.00 |

## 7. Head-to-Head Summary

| Operation | PySCF CPU | gpu4pyscf GPU | TIDES GPU | TIDES vs gpu4pyscf |
|---|---|---|---|---|
| GEMM n=2048 | 92 GFLOPS | 164 GFLOPS | 966 GFLOPS | 5.88x |
| SCF (96 atoms) | 48048 ms | 2752 ms | — (engine profiles) | — |
