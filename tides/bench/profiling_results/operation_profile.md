# Per-Operation Profiling Report

**Date**: 2026-07-10 00:47:05
**GPU**: NVIDIA GeForce RTX 3060 (12GB, sm_86) | **CUDA**: 12.9
**PySCF**: 2.11.0 | **gpu4pyscf**: 1.5.0

## 1. Per-Operation SCF Breakdown

### 1.1 PySCF CPU (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | Iter | Total ms |
|---|---|---|---|---|
| H2O | 3 | 24 | 3 | 1030.4 |
| NH3 | 4 | 29 | 3 | 1024.3 |
| C6H6 | 12 | 114 | 3 | 2735.5 |
| H2O_8mer | 24 | 192 | 3 | 6560.6 |
| H2O_16mer | 48 | 384 | 3 | 22699.1 |

### 1.2 CPU Per-Operation Detail (ms)

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | jk_matrix | occupation | overlap | veff_coulomb_xc | xc_eval_ao | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | 0.1 | 0.7 | 0.1 | 0.2 | 161.3 | 161.3 | 0.1 | 4.9 | 1019.9 | 140.9 | 857.9 |
| NH3 | 0.1 | 0.7 | 0.1 | 58.8 | 182.1 | 182.1 | 0.1 | 33.7 | 925.9 | 172.4 | 743.2 |
| C6H6 | 37.7 | 129.3 | 0.1 | 68.1 | 183.6 | 183.6 | 0.2 | 18.8 | 2467.8 | 366.0 | 2283.4 |
| H2O_8mer | 36.2 | 459.3 | 0.3 | 54.4 | 347.6 | 347.6 | 0.2 | 38.2 | 5932.8 | 663.2 | 5584.2 |
| H2O_16mer | 39.7 | 2283.9 | 1.7 | 67.8 | 1282.5 | 1282.4 | 0.2 | 25.0 | 20238.0 | 2050.0 | 18953.0 |

### 1.3 CPU Per-Operation % of Total SCF Time

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | jk_matrix | occupation | overlap | veff_coulomb_xc | xc_eval_ao | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | 0.0% | 0.1% | 0.0% | 0.0% | 15.7% | 15.7% | 0.0% | 0.5% | 99.0% | 13.7% | 83.3% |
| NH3 | 0.0% | 0.1% | 0.0% | 5.7% | 17.8% | 17.8% | 0.0% | 3.3% | 90.4% | 16.8% | 72.5% |
| C6H6 | 1.4% | 4.7% | 0.0% | 2.5% | 6.7% | 6.7% | 0.0% | 0.7% | 90.2% | 13.4% | 83.5% |
| H2O_8mer | 0.6% | 7.0% | 0.0% | 0.8% | 5.3% | 5.3% | 0.0% | 0.6% | 90.4% | 10.1% | 85.1% |
| H2O_16mer | 0.2% | 10.1% | 0.0% | 0.3% | 5.6% | 5.6% | 0.0% | 0.1% | 89.2% | 9.0% | 83.5% |

### 1.4 gpu4pyscf GPU (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | Iter | Total ms |
|---|---|---|---|---|
| H2O | 3 | 24 | 1 | 89.4 |
| NH3 | 4 | 29 | 1 | 101.5 |
| C6H6 | 12 | 114 | 1 | 446.4 |
| H2O_8mer | 24 | 192 | 1 | 554.5 |
| H2O_16mer | 48 | 384 | 1 | 1219.2 |

### 1.5 GPU Per-Operation Detail (ms)

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | occupation | overlap | veff_coulomb_xc | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|
| H2O | 1.2 | 0.9 | 0.1 | 26.9 | 5.3 | 0.3 | 24.2 | 30.4 | 24.2 |
| NH3 | 1.4 | 1.1 | 0.1 | 30.3 | 6.0 | 0.3 | 24.7 | 38.2 | 31.2 |
| C6H6 | 1.5 | 5.9 | 0.1 | 35.3 | 7.3 | 0.4 | 35.2 | 358.4 | 350.1 |
| H2O_8mer | 1.4 | 8.1 | 0.1 | 41.7 | 8.2 | 0.3 | 30.7 | 460.9 | 451.8 |
| H2O_16mer | 1.5 | 18.0 | 0.1 | 63.7 | 10.7 | 0.4 | 39.3 | 1078.1 | 1066.4 |

### 1.6 GPU Per-Operation % of Total SCF Time

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | occupation | overlap | veff_coulomb_xc | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|
| H2O | 1.4% | 1.0% | 0.1% | 30.1% | 6.0% | 0.3% | 27.1% | 34.0% | 27.0% |
| NH3 | 1.3% | 1.0% | 0.1% | 29.8% | 6.0% | 0.3% | 24.3% | 37.6% | 30.7% |
| C6H6 | 0.3% | 1.3% | 0.0% | 7.9% | 1.6% | 0.1% | 7.9% | 80.3% | 78.4% |
| H2O_8mer | 0.2% | 1.5% | 0.0% | 7.5% | 1.5% | 0.1% | 5.5% | 83.1% | 81.5% |
| H2O_16mer | 0.1% | 1.5% | 0.0% | 5.2% | 0.9% | 0.0% | 3.2% | 88.4% | 87.5% |

## 2. CPU vs GPU Per-Operation Comparison

| System | Device | Total ms | Top Operation | Top ms | Top % |
|---|---|---|---|---|---|
| H2O | CPU | 1030.4 | veff_coulomb_xc | 1019.9 | 99.0% |
| NH3 | CPU | 1024.3 | veff_coulomb_xc | 925.9 | 90.4% |
| C6H6 | CPU | 2735.5 | veff_coulomb_xc | 2467.8 | 90.2% |
| H2O_8mer | CPU | 6560.6 | veff_coulomb_xc | 5932.8 | 90.4% |
| H2O_16mer | CPU | 22699.1 | veff_coulomb_xc | 20238.0 | 89.2% |
| H2O | GPU | 89.4 | veff_coulomb_xc | 30.4 | 34.0% |
| NH3 | GPU | 101.5 | veff_coulomb_xc | 38.2 | 37.6% |
| C6H6 | GPU | 446.4 | veff_coulomb_xc | 358.4 | 80.3% |
| H2O_8mer | GPU | 554.5 | veff_coulomb_xc | 460.9 | 83.1% |
| H2O_16mer | GPU | 1219.2 | veff_coulomb_xc | 1078.1 | 88.4% |

## 3. TIDES E1 Tile Engine — Per-Kernel Profile

| Kernel | Variant | Size | Kernel ms | GFLOPS | Status |
|---|---|---|---|---|---|
| GroupedGEMM | FP64-GPU | 16x8 | 0.184 | 0.4 | PASS |
| GroupedGEMM | FP64-CPU | 16x8 | 0.026 | 2.5 | ref |
| GroupedGEMM | FP64-GPU | 32x8 | 0.027 | 19.7 | PASS |
| GroupedGEMM | FP64-CPU | 32x8 | 0.286 | 1.8 | ref |
| GroupedGEMM | FP64-GPU | 64x4 | 0.039 | 53.9 | PASS |
| GroupedGEMM | FP64-CPU | 64x4 | 1.117 | 1.9 | ref |
| GroupedGEMM | FP64-GPU | 128x2 | 0.068 | 124.1 | PASS |
| GroupedGEMM | FP64-CPU | 128x2 | 5.804 | 1.4 | ref |
| GroupedGEMM | FP64-GPU | 256x1 | 0.203 | 165.6 | PASS |
| GroupedGEMM | FP64-CPU | 256x1 | 23.597 | 1.4 | ref |
| GroupedGEMM | FP16-accum | 16x8 | 53.927 | 0.0 | PASS |
| GroupedGEMM | FP16-accum | 32x8 | 0.051 | 10.2 | PASS |
| GroupedGEMM | FP16-accum | 64x4 | 0.073 | 28.8 | PASS |
| GroupedGEMM | FP16-accum | 128x2 | 0.052 | 161.1 | PASS |
| GroupedGEMM | FP16-accum | 256x1 | 0.745 | 45.0 | PASS |
| SpGEMM | FP64-GPU | 4x16 | 0.081 | 6.5 | PASS |
| SpGEMM | FP64-CPU | 4x16 | 0.187 | 2.8 | ref |
| SpGEMM | FP64-GPU | 8x16 | 0.144 | 29.1 | PASS |
| SpGEMM | FP64-CPU | 8x16 | 1.576 | 2.7 | ref |
| SpGEMM | FP64-GPU | 16x32 | 1.835 | 146.3 | PASS |
| SpGEMM | FP64-CPU | 16x32 | 101.793 | 2.6 | ref |
| SpGEMM | FP64-GPU | 32x32 | 15.342 | 140.0 | PASS |
| SpGEMM | FP64-CPU | 32x32 | 796.585 | 2.7 | ref |
| Ozaki-f64e | FP16-slice | n=16 | 0.000 | 0.0 | PASS |
| Ozaki-f64e | FP64-CPU-refn=16 | 0.004 | 0.004 | 2.0 | ref |
| Ozaki-f64e | FP16-slice | n=32 | 0.000 | 0.0 | PASS |
| Ozaki-f64e | FP64-CPU-refn=32 | 0.024 | 0.024 | 2.7 | ref |
| Ozaki-f64e | FP16-slice | n=64 | 0.000 | 0.0 | PASS |
| Ozaki-f64e | FP64-CPU-refn=64 | 0.192 | 0.192 | 2.7 | ref |
| Ozaki-f64e | FP16-slice | n=128 | 0.000 | 0.0 | PASS |
| Ozaki-f64e | FP64-CPU-refn=128 | 2.031 | 2.031 | 2.1 | ref |
| Dot-f64e | GPU | n=1000 | 4.337 | 0.0 | PASS |
| Dot-f64e | CPU-ref | n=1000 | 0.002 | 0.0 | ref |
| Dot-f64e | GPU | n=10000 | 0.049 | 0.0 | PASS |
| Dot-f64e | CPU-ref | n=10000 | 0.016 | 0.0 | ref |
| Dot-f64e | GPU | n=100000 | 0.147 | 0.0 | PASS |
| Dot-f64e | CPU-ref | n=100000 | 0.165 | 0.0 | ref |
| Dot-f64e | GPU | n=1000000 | 1.024 | 0.0 | PASS |
| Dot-f64e | CPU-ref | n=1000000 | 1.822 | 0.0 | ref |
| Trace-f64e | GPU | n=64 | 2.131 | 0.0 | PASS |
| Trace-f64e | GPU | n=256 | 0.038 | 0.0 | PASS |
| Trace-f64e | GPU | n=1024 | 0.011 | 0.0 | PASS |
| GraphReplay | FP64-raw | 8x64x100 | 0.027 | 0.0 | ref |
| GraphReplay | FP64-graph | 8x64x100 | 0.026 | 0.0 | PASS |
| GraphReplay | FP16-raw | 8x64x100 | 0.004 | 0.0 | ref |
| GraphReplay | FP16-graph | 8x64x100 | 0.004 | 0.0 | PASS |

## 4. Bottleneck Analysis

### 4.1 CPU Bottlenecks (largest time consumers)

| System | Operation | Total ms | Calls |
|---|---|---|---|
| H2O_16mer | veff_coulomb_xc | 20238.0 | 3 |
| H2O_16mer | xc_nr_rks | 18953.0 | 3 |
| H2O_8mer | veff_coulomb_xc | 5932.8 | 3 |
| H2O_8mer | xc_nr_rks | 5584.2 | 3 |
| C6H6 | veff_coulomb_xc | 2467.8 | 3 |
| H2O_16mer | eigendecomposition | 2283.9 | 3 |
| C6H6 | xc_nr_rks | 2283.4 | 3 |
| H2O_16mer | xc_eval_ao | 2050.0 | 63 |
| H2O_16mer | j_matrix | 1282.5 | 3 |
| H2O_16mer | jk_matrix | 1282.4 | 3 |
| H2O | veff_coulomb_xc | 1019.9 | 3 |
| NH3 | veff_coulomb_xc | 925.9 | 3 |
| H2O | xc_nr_rks | 857.9 | 3 |
| NH3 | xc_nr_rks | 743.2 | 3 |
| H2O_8mer | xc_eval_ao | 663.2 | 33 |

### 4.2 GPU Bottlenecks (largest time consumers)

| System | Operation | Total ms | Calls |
|---|---|---|---|
| H2O_16mer | veff_coulomb_xc | 1078.1 | 2 |
| H2O_16mer | xc_nr_rks | 1066.4 | 2 |
| H2O_8mer | veff_coulomb_xc | 460.9 | 2 |
| H2O_8mer | xc_nr_rks | 451.8 | 2 |
| C6H6 | veff_coulomb_xc | 358.4 | 2 |
| C6H6 | xc_nr_rks | 350.1 | 2 |
| H2O_16mer | hcore | 63.7 | 1 |
| H2O_8mer | hcore | 41.7 | 1 |
| H2O_16mer | overlap | 39.3 | 2 |
| NH3 | veff_coulomb_xc | 38.2 | 2 |
| C6H6 | hcore | 35.3 | 1 |
| C6H6 | overlap | 35.2 | 2 |
| NH3 | xc_nr_rks | 31.2 | 2 |
| H2O_8mer | overlap | 30.7 | 2 |
| H2O | veff_coulomb_xc | 30.4 | 2 |

### 4.3 TIDES GPU Kernel Bottlenecks (slowest kernels)

| Kernel | Variant | Size | Kernel ms | GFLOPS |
|---|---|---|---|---|
| GroupedGEMM | FP16-accum | 16x8 | 53.927 | 0.0 |
| SpGEMM | FP64-GPU | 32x32 | 15.342 | 140.0 |
| Dot-f64e | GPU | n=1000 | 4.337 | 0.0 |
| Trace-f64e | GPU | n=64 | 2.131 | 0.0 |
| SpGEMM | FP64-GPU | 16x32 | 1.835 | 146.3 |
| Dot-f64e | GPU | n=1000000 | 1.024 | 0.0 |
| GroupedGEMM | FP16-accum | 256x1 | 0.745 | 45.0 |
| GroupedGEMM | FP64-GPU | 256x1 | 0.203 | 165.6 |
| GroupedGEMM | FP64-GPU | 16x8 | 0.184 | 0.4 |
| Dot-f64e | GPU | n=100000 | 0.147 | 0.0 |

## 5. Per-Operation Call Counts

### 5.1 CPU Call Counts

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | jk_matrix | occupation | overlap | veff_coulomb_xc | xc_eval_ao | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | 3 | 3 | 4 | 1 | 3 | 3 | 2 | 1 | 3 | 6 | 3 |
| NH3 | 3 | 3 | 4 | 1 | 3 | 3 | 2 | 1 | 3 | 6 | 3 |
| C6H6 | 3 | 3 | 4 | 1 | 3 | 3 | 2 | 1 | 3 | 15 | 3 |
| H2O_8mer | 3 | 3 | 4 | 1 | 3 | 3 | 2 | 1 | 3 | 33 | 3 |
| H2O_16mer | 3 | 3 | 4 | 1 | 3 | 3 | 2 | 1 | 3 | 63 | 3 |

### 5.2 GPU Call Counts

| System | density_matrix | eigendecomposition | fock_assembly | hcore | j_matrix | occupation | overlap | veff_coulomb_xc | xc_nr_rks |
|---|---|---|---|---|---|---|---|---|---|
| H2O | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 |
| NH3 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 |
| C6H6 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 |
| H2O_8mer | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 |
| H2O_16mer | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 |
