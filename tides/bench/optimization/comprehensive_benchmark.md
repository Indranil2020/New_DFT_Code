# Comprehensive TIDES vs PySCF/gpu4pyscf Benchmark

**Date**: 2026-07-07 02:35:03
**System**: AMD Ryzen AI 7 350 (8c/16t) | NVIDIA RTX 5050 Laptop (8GB, Blackwell) | 30GB RAM | Intel MPI 2021.18.0
**PySCF**: 2.13.1 | **GPU**: True

## 1. GEMM Performance

| n | PySCF CPU GFLOPS | PySCF CPU ms | PySCF GPU GFLOPS | PySCF GPU ms | TIDES GPU GFLOPS | TIDES GPU ms |
|---|---|---|---|---|---|---|
| 64 | 50.0 | 0.01 | 14.998512595326355 | 0.03 | 235.618 | 0.106912 |
| 128 | 0.3 | 11.99 | 86.89075894235721 | 0.05 | 235.618 | 0.106912 |
| 256 | 1.4 | 24.00 | 117.47187636826068 | 0.29 | 235.618 | 0.106912 |
| 512 | 13.4 | 20.00 | 153.41483330785573 | 1.75 | 235.618 | 0.106912 |
| 1024 | 80.1 | 26.80 | 171.233457097113 | 12.54 | 235.618 | 0.106912 |
| 2048 | 132.2 | 129.98 | 193.6808577836461 | 88.70 | 235.618 | 0.106912 |

**TIDES planned GEMM**: 235.618 GFLOPS (kernel) vs cuBLASLt 1065.22 GFLOPS

## 2. Eigendecomposition Performance

| n | PySCF CPU ms | PySCF GPU ms | TIDES LAPACK ms (n=256) | Speedup CPU→GPU |
|---|---|---|---|---|
| 32 | 0.08 | 1.6378559812437743 | — | 0.0× |
| 64 | 0.28 | 3.8482679810840636 | — | 0.1× |
| 128 | 92.01 | 4.063334024976939 | — | 22.6× |
| 256 | 369.01 | 6.775832996936515 | 9.5 | 54.5× |
| 512 | 2061.93 | 16.716586978873238 | — | 123.3× |
| 1024 | 5657.96 | 72.86992299486883 | — | 77.6× |

## 3. SCF — Basis Set Scan (H2O, LDA)

| Basis | nao | CPU ms | GPU ms | GPU Speedup | CPU Energy | GPU Energy |
|---|---|---|---|---|---|---|
| STO-3G | 7 | 662.3 | 111.21988500235602 | 6.0× | -74.058306 | -74.05830579658426 |
| 6-31G* | 18 | 568.1 | 210.7490299968049 | 2.7× | -75.171199 | -75.17119924779861 |
| cc-pVDZ | 24 | 628.6 | 292.4207880278118 | 2.1× | -75.185570 | -75.1855698103313 |
| cc-pVTZ | 58 | 763.3 | 475.64865698223 | 1.6× | -75.230351 | -75.23035080026516 |
| def2-TZVP | 43 | 687.7 | 345.03415200742893 | 2.0× | -75.233155 | -75.23315490381873 |

## 4. SCF — System Size Scan (water clusters, cc-pVDZ, LDA)

| System | Atoms | nao | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|---|
| H2O | 3 | 24 | 629.8 | 146.89691000967287 | 4.3× |
| H2O_dimer | 6 | 48 | 1317.4 | 303.94932499621063 | 4.3× |
| H2O_tetramer | 12 | 96 | 2621.8 | 425.5910760257393 | 6.2× |
| H2O_hexamer | 18 | 144 | 4793.3 | 563.6710750113707 | 8.5× |

## 5. SCF — XC Functional Scan (H2O/cc-pVDZ)

| XC | CPU ms | GPU ms | GPU Speedup | CPU Energy | GPU Energy |
|---|---|---|---|---|---|
| LDA | 649.4 | 167.32589498860762 | 3.9× | -75.185570 | -75.1855698103313 |
| PBE | 926.0 | 146.3981390115805 | 6.3× | -76.329515 | -76.32951459099036 |
| B3LYP | 961.8 | 1079.3466010072734 | 0.9× | -76.415761 | -76.4157609342192 |
| HF | 87.3 | 976.4768180029932 | 0.1× | -76.020753 | -76.02075342469908 |
| PBE0 | 921.8 | 1151.1474619910587 | 0.8× | -76.334374 | -76.33437372967666 |

## 6. SCF — Atom Scan (cc-pVDZ, LDA)

| Atom | nao | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|---|
| H | 5 | 786.1 | 126.40368699794635 | 6.2× |
| He | 5 | 606.3 | 70.99837702116929 | 8.5× |
| C | 14 | 11797.3 | 2690.0476870068815 | 4.4× |
| N | 14 | 742.7 | 135.54752300842665 | 5.5× |
| O | 14 | 11860.3 | 2813.711652997881 | 4.2× |
| Ne | 14 | 849.0 | 139.48132799123414 | 6.1× |

## 7. Forces (H2O/cc-pVDZ)

| XC | CPU ms | GPU ms | GPU Speedup |
|---|---|---|---|
| LDA | 241.88 | 308.0998980149161 | 0.8× |
| HF | 140.99 | 138.64432999980636 | 1.0× |

## 8. TIDES Engine Profiles (E1-E9)

| Engine | Status | Wall (s) |
|---|---|---|
| E1_tile | pass | 4.63 |
| E2_basis | pass | 4.78 |
| E3_grid | pass | 0.63 |
| E4_solvers | pass | 2.89 |
| E5_scf | pass | 0.19 |
| E6_dynamics | pass | 0.01 |
| E7_parallel | pass | 8.08 |
| E8_hybrids | pass | 0.01 |
| E9_verification | pass | 0.01 |
| cuda_gemm_probe | pass | 0.72 |
| cuda_ozaki_gemm_probe | pass | 0.02 |

## 9. MPI Scaling (TIDES E7)

| nproc | Status | Wall (s) |
|---|---|---|
| nproc=1 | pass | 8.27 |
| nproc=2 | pass | 9.93 |
| nproc=4 | pass | 8.95 |
| nproc=8 | pass | 13.98 |

## 10. Head-to-Head: TIDES vs PySCF (Comparable Operations)

| Operation | PySCF CPU | PySCF GPU | TIDES CPU | TIDES GPU | TIDES vs PySCF GPU |
|---|---|---|---|---|---|
| GEMM n=1024 | 132 GFLOPS | 194 GFLOPS | — | 236 GFLOPS | 1.2× |
| Eig n=1024 | 5658 ms | 73 ms | — | — | — |
