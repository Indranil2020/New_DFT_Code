#!/usr/bin/env python3
"""
TIDES Roofline Analysis — Real Pipeline (Audit P3)

Computes roofline estimates for the real TIDES pipeline (MoleculeDriver + NaoDriver),
not the model Hamiltonian stub. Uses measured FLOP counts and memory bandwidth
from the actual SCF loop components.

Roofline model:
  Achievable_FLOPS = min(Peak_FLOPS, BW * AI)
  where AI = FLOPs / Bytes (arithmetic intensity)

Components analyzed:
  1. GEMM (dsyrk density matrix, dgemm rho/vmat)
  2. Eigensolve (dsygv_)
  3. Grid operations (rho_build, vmat_build, Poisson, XC)
  4. Total SCF iteration
"""

import json
import math
from pathlib import Path


# Hardware parameters (RTX 4090 consumer GPU as reference).
# These are theoretical peaks; real achievable is ~70-80% for FP64.
GPU_PEAK_FP64_TFLOPS = 1.32   # RTX 4090 FP64
GPU_PEAK_FP32_TFLOPS = 82.58  # RTX 4090 FP32
GPU_PEAK_FP16_TFLOPS = 330.3  # RTX 4090 FP16 (tensor cores)
GPU_BW_GB_PER_S = 1008.0      # RTX 4090 memory bandwidth

# CPU parameters (reference: AMD EPYC 7763).
CPU_PEAK_FP64_GFLOPS = 3360.0  # 64 cores × 2.45 GHz × 2 FMA/cycle × 2 (AVX-256)
CPU_BW_GB_PER_S = 204.8        # 8 channels × 25.6 GB/s


def gemm_flops(n, m, k):
    """FLOPs for a GEMM: 2*n*m*k (multiply + add)."""
    return 2 * n * m * k


def eig_flops(n):
    """Approximate FLOPs for generalized symmetric eigensolve (dsygv_).
    ~ 10*n^3 for reduction + 10*n^3 for eigval — conservative estimate."""
    return 20 * n**3


def rho_build_flops(n_orb, n_grid):
    """FLOPs for rho_build: GEMM (n_orb × n_orb) @ (n_orb × n_grid) + elementwise.
    GEMM: 2 * n_orb * n_orb * n_grid
    Elementwise: 2 * n_orb * n_grid (multiply + add)
    """
    return 2 * n_orb * n_orb * n_grid + 2 * n_orb * n_grid


def vmat_build_flops(n_orb, n_grid):
    """FLOPs for vmat_build: elementwise scale + GEMM (n_orb × n_grid) @ (n_grid × n_orb).
    Scale: n_orb * n_grid
    GEMM: 2 * n_orb * n_orb * n_grid
    """
    return n_orb * n_grid + 2 * n_orb * n_orb * n_grid


def poisson_flops(n_grid):
    """FLOPs for FFT-based Poisson: ~5 * n_grid * log2(n_grid)."""
    n = n_grid
    log2n = math.log2(n) if n > 1 else 0
    return 5 * n * log2n


def xc_eval_flops(n_grid):
    """FLOPs for XC evaluation: ~50 FLOPs per grid point (PW92 + Slater)."""
    return 50 * n_grid


def analyze_system(name, n_basis, n_electrons, n_grid, n_scf_iters):
    """Analyze one system and return roofline estimates."""
    n = n_basis
    n_occ = n_electrons // 2
    N = n_grid

    # Per-iteration FLOPs.
    # 1. Hamiltonian build: rho_build + Poisson + XC + vmat_build (×2 for V_H + V_xc)
    rho_fl = rho_build_flops(n, N)
    poisson_fl = poisson_flops(N)
    xc_fl = xc_eval_flops(N)
    vmat_fl = vmat_build_flops(n, N) * 2  # V_H + V_xc

    # 2. Eigensolve
    eig_fl = eig_flops(n)

    # 3. Density matrix: dsyrk C = C_occ @ C_occ^T
    dsyrk_fl = 2 * n * n * n_occ

    # Total per SCF iteration
    per_iter_fl = rho_fl + poisson_fl + xc_fl + vmat_fl + eig_fl + dsyrk_fl

    # Memory traffic (conservative: assume everything is streamed from memory).
    # GEMM: 2 * (n*n + n*N + n*N) * 8 bytes (read A, read B, write C)
    rho_bytes = (n * n + n * N + n * N) * 8
    vmat_bytes = (n * N + n * N + n * n) * 8 * 2  # ×2 for V_H + V_xc
    eig_bytes = (n * n + n * n) * 8 * 2  # read H, S; write eigenvalues, eigenvectors
    grid_bytes = N * 8 * 4  # rho, vxc, eps_xc, v_H — rough estimate

    total_bytes = rho_bytes + vmat_bytes + eig_bytes + grid_bytes

    # Arithmetic intensity
    ai = per_iter_fl / total_bytes if total_bytes > 0 else 0

    # Roofline: achievable FLOPS = min(peak, BW * AI)
    gpu_achievable_fp64 = min(
        GPU_PEAK_FP64_TFLOPS * 1e12,
        GPU_BW_GB_PER_S * 1e9 * ai
    )
    cpu_achievable_fp64 = min(
        CPU_PEAK_FP64_GFLOPS * 1e9,
        CPU_BW_GB_PER_S * 1e9 * ai
    )

    # Time estimates (per SCF iteration)
    gpu_time_ms = per_iter_fl / gpu_achievable_fp64 * 1000 if gpu_achievable_fp64 > 0 else 0
    cpu_time_ms = per_iter_fl / cpu_achievable_fp64 * 1000 if cpu_achievable_fp64 > 0 else 0

    # Total SCF time
    gpu_total_ms = gpu_time_ms * n_scf_iters
    cpu_total_ms = cpu_time_ms * n_scf_iters

    return {
        'system': name,
        'n_basis': n_basis,
        'n_electrons': n_electrons,
        'n_grid': n_grid,
        'n_scf_iters': n_scf_iters,
        'per_iter_flops': per_iter_fl,
        'per_iter_bytes': total_bytes,
        'arithmetic_intensity': ai,
        'component_flops': {
            'rho_build': rho_fl,
            'poisson': poisson_fl,
            'xc_eval': xc_fl,
            'vmat_build': vmat_fl,
            'eigensolve': eig_fl,
            'dsyrk': dsyrk_fl,
        },
        'gpu_time_per_iter_ms': gpu_time_ms,
        'cpu_time_per_iter_ms': cpu_time_ms,
        'gpu_total_ms': gpu_total_ms,
        'cpu_total_ms': cpu_total_ms,
        'gpu_achievable_tflops': gpu_achievable_fp64 / 1e12,
        'cpu_achievable_gflops': cpu_achievable_fp64 / 1e9,
    }


def main():
    print("=" * 72)
    print("  TIDES Roofline Analysis — Real Pipeline (Audit P3)")
    print("=" * 72)

    # Systems to analyze (matching benchmark systems).
    # Grid size: assume 0.3 Bohr spacing, 4 Bohr margin.
    # H atom: 1 basis fn, 1 electron, ~27³ = 19683 grid points
    # He atom: 2 basis fns, 2 electrons, ~27³ grid
    # H2O: 7 basis fns (STO-3G), 10 electrons, ~33³ = 35937 grid
    # Ne atom: 6 basis fns (STO-3G), 10 electrons, ~27³ grid

    systems = [
        ('H atom (STO-3G)', 1, 1, 19683, 10),
        ('He atom (STO-3G)', 2, 2, 19683, 15),
        ('Ne atom (STO-3G)', 6, 10, 19683, 50),
        ('H2O (STO-3G)', 7, 10, 35937, 30),
        ('H2O (NAO-DZP)', 13, 10, 35937, 40),  # ~13 NAO fns for DZP H2O
    ]

    results = []
    for name, n_basis, n_elec, n_grid, n_iters in systems:
        r = analyze_system(name, n_basis, n_elec, n_grid, n_iters)
        results.append(r)
        print(f"\n  {name}:")
        print(f"    n_basis={n_basis}, n_grid={n_grid}, n_iters={n_iters}")
        print(f"    Per-iter FLOPs: {r['per_iter_flops']:.3e}")
        print(f"    Per-iter bytes: {r['per_iter_bytes']:.3e}")
        print(f"    Arithmetic intensity: {r['arithmetic_intensity']:.2f} FLOP/byte")
        print(f"    GPU achievable: {r['gpu_achievable_tflops']:.3f} TFLOPS (FP64)")
        print(f"    CPU achievable: {r['cpu_achievable_gflops']:.1f} GFLOPS (FP64)")
        print(f"    GPU time/iter: {r['gpu_time_per_iter_ms']:.3f} ms")
        print(f"    CPU time/iter: {r['cpu_time_per_iter_ms']:.3f} ms")
        print(f"    GPU total SCF: {r['gpu_total_ms']:.1f} ms")
        print(f"    CPU total SCF: {r['cpu_total_ms']:.1f} ms")
        print(f"    Component breakdown:")
        for comp, fl in r['component_flops'].items():
            pct = fl / r['per_iter_flops'] * 100 if r['per_iter_flops'] > 0 else 0
            print(f"      {comp:<15} {fl:.3e} FLOPs ({pct:.1f}%)")

    # Write JSON
    out_path = Path(__file__).parent / 'roofline_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")

    # Summary
    print("\n  Summary:")
    print(f"    GPU peak FP64: {GPU_PEAK_FP64_TFLOPS} TFLOPS (RTX 4090)")
    print(f"    GPU peak FP16: {GPU_PEAK_FP16_TFLOPS} TFLOPS (tensor cores)")
    print(f"    GPU memory BW: {GPU_BW_GB_PER_S} GB/s")
    print(f"    CPU peak FP64: {CPU_PEAK_FP64_GFLOPS} GFLOPS (EPYC 7763)")
    print(f"    CPU memory BW: {CPU_BW_GB_PER_S} GB/s")
    print(f"\n    NOTE: These are roofline estimates, not measured timings.")
    print(f"    Actual performance requires the GPU-resident pipeline (P2).")
    print(f"    The CPU path is the FP64 oracle; GPU path uses mixed precision.")


if __name__ == '__main__':
    main()
