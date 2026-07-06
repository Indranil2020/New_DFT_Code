# TIDES P4: Profiling & Benchmarking Report

## Summary

All CPU and GPU kernels were benchmarked at multiple problem sizes.
GPU implementations match CPU references to machine precision (max diff ≤ 2.7e-12).
GPU speedup increases with problem size as expected (amortizing transfer overhead).

## Benchmark Results

### Rho Build (density from orbitals)

| Grid Size | CPU (ms) | GPU (ms) | Speedup | Max Diff |
|-----------|----------|----------|---------|----------|
| 16^3      | 0.004    | 0.403    | 0.01x   | 2.2e-16  |
| 24^3      | 0.015    | 0.156    | 0.10x   | 2.2e-16  |
| 32^3      | 0.053    | 0.132    | 0.40x   | 2.2e-16  |
| 48^3      | 0.226    | 0.059    | 3.81x   | 2.2e-16  |
| 64^3      | 0.549    | 0.058    | 9.39x   | 2.2e-16  |

**Crossover**: ~40^3. GPU dominates at 48^3+.

### Vmat Build (v→H adjoint map)

| Grid Size | CPU (ms) | GPU (ms) | Speedup | Max Diff |
|-----------|----------|----------|---------|----------|
| 16^3      | 0.015    | 0.423    | 0.03x   | 2.6e-16  |
| 24^3      | 0.048    | 0.092    | 0.52x   | 3.7e-16  |
| 32^3      | 0.116    | 0.187    | 0.62x   | 8.3e-17  |
| 48^3      | 0.418    | 0.269    | 1.56x   | 1.5e-16  |
| 64^3      | 0.961    | 0.408    | 2.36x   | 6.8e-17  |

**Crossover**: ~40^3. GPU dominates at 48^3+.

### XC Evaluation (LDA)

| Grid Size | CPU (ms) | GPU (ms) | Speedup | Max Diff |
|-----------|----------|----------|---------|----------|
| 16^3      | 0.450    | 0.423    | 1.06x   | 2.7e-12  |
| 24^3      | 1.369    | 0.794    | 1.72x   | 2.7e-12  |
| 32^3      | 3.095    | 1.018    | 3.04x   | 2.7e-12  |
| 48^3      | 10.925   | 2.884    | 3.79x   | 2.7e-12  |
| 64^3      | 34.707   | 6.525    | 5.32x   | 2.7e-12  |

**Crossover**: ~16^3. GPU wins immediately. XC is compute-bound (PW92 evaluation).

### SP2 Purification (density matrix)

| Matrix Size | CPU (ms) | GPU (ms) | Speedup | Max Diff |
|-------------|----------|----------|---------|----------|
| n=20        | 0.142    | 3.568    | 0.04x   | 2.2e-16  |
| n=50        | 3.051    | 1.095    | 2.79x   | 5.0e-16  |
| n=100       | 30.386   | 2.128    | 14.28x  | 5.6e-16  |
| n=200       | 232.029  | 9.830    | 23.60x  | 6.7e-16  |
| n=500       | 4897.486 | 54.002   | 90.69x  | 7.8e-16  |

**Crossover**: ~n=40. GPU dominates at n=50+. SP2 is O(n^3) per iteration (cuBLAS GEMM).

## Key Findings

1. **GPU acceleration is effective for all kernels at production sizes.**
   - SP2 sees the largest speedup (91x at n=500) due to cuBLAS GEMM efficiency.
   - XC evaluation benefits from parallel PW92 evaluation (5.3x at 64^3).
   - Rho/Vmat builds see moderate speedup (2-9x) due to memory-bound nature.

2. **Small problems favor CPU** due to H2D/D2H transfer overhead.
   - Crossover points: ~40^3 for grid kernels, ~n=40 for SP2.

3. **Accuracy is preserved**: all GPU results match CPU to ≤ 2.7e-12.
   - SP2: ≤ 7.8e-16 (machine precision).
   - Rho/Vmat: ≤ 3.7e-16 (machine precision).
   - XC: ≤ 2.7e-12 (FP64 rounding in PW92 rational approximation).

4. **Scaling**: GPU kernels scale better with problem size due to:
   - Amortized transfer overhead.
   - Better occupancy at larger grid sizes.
   - cuBLAS GEMM efficiency for SP2 (O(n^3) → GPU parallelism).

## Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| C++ unit tests (WP1-WP9) | 30 | ALL PASS |
| Physics validation (P3.1) | 8 | ALL PASS |
| GPU SP2 tests (P2.6) | 3 | ALL PASS |
| GPU regression (P3.3) | 4×3 | ALL PASS |
| PySCF cross-checks (P3.2) | 7 | ALL PASS |
| Python API tests (WP10) | 25 | ALL PASS |
| **Total** | **77** | **ALL PASS** |
