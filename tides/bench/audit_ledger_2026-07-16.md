# Clinical Audit Ledger — build_H Performance

**Date:** 2026-07-16  
**System:** Linux x86_64, CUDA GPU pipeline  
**Grid:** h=0.5 Bohr, margin=4.0 Bohr, free-space (non-periodic)  
**Poisson solver:** `PoissonFreeDeviceCache` (device-resident, cached cuFFT Z2Z plans)

---

## Executive Summary

The `build_H` step was profiled across 7 molecules × 2 XC functionals (LDA, PBE) with
CUDA event-level substep timing. The key finding is that the **Poisson FFT itself is
fast (~5–8ms)**, but the `poisson` wall-clock timer is inflated by **hidden async rho
build GPU time** (the `RhoGradientDeviceKernel` queues on the same stream, and the
first blocking cuFFT call waits for it to drain).

### True build_H Breakdown (CH4/LDA, representative)

| Substep | Time (ms) | % of build_H | Notes |
|---|---|---|---|
| RhoGradientDeviceKernel | 32.58 | 70% | O(N²) custom kernel, NOT cuBLAS GEMM |
| Poisson FFT (fwd+inv+multiply) | 5.67 | 12% | cuFFT Z2Z, cached plans |
| Vmat projection + D2H | 3.80 | 8% | BuildWeightedVmatDevice |
| XC evaluation | 4.00 | 9% | LDA: fast; PBE: 31ms (42%) |
| Assemble H | ~0 | 0% | CPU addition |
| **Total** | **46.43** | **100%** | |

### Optimization Priority

1. **RhoGradientDeviceKernel** — replace with cuBLAS GEMM (highest impact)
2. **PBE XC evaluation** — 31ms for CH4, likely kernel launch overhead
3. **Poisson FFT** — already well-optimized at ~5ms

---

## Full Profiling Data

### LDA Results

| Mol | n_basis | np_total | build_H (ms) | rho_hidden (ms) | pois_fft (ms) | pois_total (ms) | xc_eval (ms) | FFT grid |
|---|---|---|---|---|---|---|---|---|
| H   |  8 |  117,649 |  6.99 |  1.82 |  4.28 |  6.48 |  0.48 | 98×98×98 |
| H2  | 16 |  127,253 | 12.91 |  6.02 |  5.41 | 12.09 |  0.79 | 98×98×106 |
| He  |  8 |   59,319 |  4.14 |  1.03 |  2.56 |  3.84 |  0.28 | 78×78×78 |
| LiH | 16 |  127,253 | 10.80 |  4.29 |  5.09 | 10.06 |  0.72 | 98×98×106 |
| CH4 | 40 | 1,202,145 | 46.43 | 32.58 |  5.67 | 42.39 |  4.00 | 110×106×110 |
| H2O | 24 |  132,447 | 20.77 | 12.34 |  5.97 | 19.49 |  1.26 | 98×106×102 |
| NH3 | 32 |  148,877 | 32.48 | 19.27 |  8.38 | 29.96 |  2.49 | 106×106×106 |

### PBE Results

| Mol | n_basis | np_total | build_H (ms) | rho_hidden (ms) | pois_fft (ms) | pois_total (ms) | xc_eval (ms) | FFT grid |
|---|---|---|---|---|---|---|---|---|
| H   |  8 |  117,649 |  7.63 |  1.64 |  4.16 |  6.07 |  1.55 | 98×98×98 |
| H2  | 16 |  127,253 | 13.97 |  5.17 |  5.02 | 10.80 |  3.15 | 98×98×106 |
| He  |  8 |   59,319 |  4.51 |  1.06 |  2.41 |  3.67 |  0.83 | 78×78×78 |
| LiH | 16 |  127,253 | 13.83 |  4.42 |  5.34 | 10.42 |  3.38 | 98×98×106 |
| CH4 | 40 | 1,202,145 | 73.99 | 32.80 |  6.19 | 42.94 | 31.02 | 110×106×110 |
| H2O | 24 |  132,447 | 26.98 | 12.51 |  5.95 | 19.53 |  7.42 | 98×106×102 |
| NH3 | 32 | 1,135,575 | 46.85 | 19.39 |  8.10 | 29.85 | 16.97 | 106×106×106 |

---

## Poisson Substep Breakdown (CUDA Event Timings, TIDES_POISSON_PROFILE=1)

### Poisson FFT Substeps (averaged over SCF iterations, ms)

| Mol | XC | memset_pad | zero_pad | fft_fwd | multiply | fft_inv | extract | energy | GPU_sum | solve_cpu | vmat_cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H   | LDA | 0.05 | 0.05 | 1.91 | 0.14 | 2.03 | 0.01 | 0.12 | 4.28 | 6.01 | 0.29 |
| H2  | LDA | 0.05 | 0.01 | 2.57 | 0.24 | 2.48 | 0.01 | 0.08 | 5.41 | 11.79 | 0.61 |
| He  | LDA | 0.02 | 0.01 | 1.20 | 0.07 | 1.28 | 0.01 | 0.06 | 2.56 | 3.65 | 0.21 |
| LiH | LDA | 0.05 | 0.01 | 2.42 | 0.15 | 2.47 | 0.01 | 0.07 | 5.18 | 9.56 | 0.60 |
| CH4 | LDA | 0.08 | 0.02 | 2.73 | 0.19 | 2.71 | 0.02 | 0.11 | 5.77 | 36.92 | 3.69 |
| H2O | LDA | 0.05 | 0.01 | 2.69 | 0.15 | 2.78 | 0.01 | 0.07 | 5.77 | 17.62 | 1.05 |
| NH3 | LDA | 0.06 | 0.01 | 3.87 | 0.18 | 3.73 | 0.01 | 0.09 | 7.95 | 26.47 | 2.15 |

### Key Observations

1. **FFT forward + inverse dominate GPU time** (~4–7.5ms total).
   - fwd: 1.2–3.9ms, inv: 1.2–3.7ms — roughly equal, as expected for Z2Z.
   - multiply kernel: 0.07–0.24ms — negligible.
   - zero_pad/extract/memset: <0.1ms each — negligible.

2. **solve_cpu >> GPU_sum** — the CPU wall time of `Solve()` far exceeds the GPU event
   times. This is because `cufftExecZ2Z` blocks the CPU until prior stream work (the
   `RhoGradientDeviceKernel`) completes. The difference (`solve_cpu - GPU_sum`) is the
   hidden rho build GPU time.

3. **vmat_cpu is small** (0.2–3.7ms) — the Vmat projection and D2H are not a bottleneck.

4. **FFT grid sizes are ~2× the simulation grid** (zero-padded for free-space convolution).

---

## Root Cause Analysis

### The "Poisson bottleneck" is actually the RhoGradientDeviceKernel

The `poisson` timer measures wall time from `t_poisson0` to `t_poisson1`, which includes:
1. `PoissonFreeDeviceCache::Solve()` — async GPU launches (memset, FFT, multiply, extract)
2. `BuildWeightedVmatDevice` — async GEMM-like projection
3. `cudaMemcpyAsync` for V_H D2H

Since all operations are on `dev_stream`, they queue behind the `RhoGradientDeviceKernel`
from the rho_build step. The first blocking call (cuFFT) waits for the rho kernel to drain.

**Proof:** Adding `cudaStreamSynchronize(dev_stream)` before `Solve()` drops `solve_cpu`
from 38.63ms to 6.16ms for CH4, confirming the stream-queueing hypothesis.

### RhoGradientDeviceKernel Analysis

```cuda
// O(n_basis² × n_points) per thread, 128 threads/block
for (mu = 0; mu < nbasis; ++mu)
  for (nu = 0; nu < nbasis; ++nu)
    density += P[mu*nbasis+nu] * phi_mu * phi_nu;
```

- **CH4:** 40² × 1,202,145 = 1.92B iterations × ~20 FLOPs = ~38B FLOPs → 32.6ms
- **NH3:** 32² × 148,877 = 152M iterations → 19.3ms
- **H2:** 16² × 127,253 = 32.5M iterations → 6.0ms

The kernel is compute-bound with poor memory coalescing (each thread reads nbasis phi
values from global memory). A cuBLAS GEMM approach would be ~10× faster.

### Recommended Fix: Replace with cuBLAS GEMM

```
rho = diag(phi^T @ P @ phi)
```

1. `temp = P @ phi`  → cuBLAS DGEMM (n×n×np, highly optimized)
2. `rho = sum_i phi_i * temp_i`  → element-wise multiply + reduction kernel

This would reduce the 32.6ms CH4 rho build to ~3ms, cutting build_H from 46ms to ~17ms.

---

## Optimizations Already Applied

1. **PoissonFreeDeviceCache** — device-resident solver with cached cuFFT plans, no H2D/D2H
   for rho or V_H, uses main CUDA stream.
2. **Removed cudaStreamSynchronize** after Poisson solve — allows overlap with XC eval.
3. **Conditional profiling** — CUDA event timing gated by `TIDES_POISSON_PROFILE=1` env var,
   zero overhead in production.
4. **Python bindings** — all Poisson substep timings exposed via `BuildHTimings` for
   programmatic analysis.

---

## Next Steps

1. **Replace RhoGradientDeviceKernel with cuBLAS GEMM** (highest impact, ~10× speedup)
2. **Optimize PBE XC evaluation** (31ms for CH4 — investigate kernel efficiency)
3. **Research R2C FFT** for Poisson (real-to-complex would halve FFT work, but Z2Z is
   already only ~5ms)
4. **Consider multigrid Poisson** for very large grids (current FFT is O(N log N), multigrid
   is O(N), but FFT is already fast at these sizes)
