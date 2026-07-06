#!/usr/bin/env python3
"""
PySCF vs TIDES profiling comparison.

Benchmarks comparable operations on both CPU and GPU:
  - GEMM (matrix multiply)
  - Eigendecomposition (dense)
  - SCF (single-point energy)
  - Forces (finite difference)
  - Grid operations (XC, Poisson)

Outputs: bench/optimization/pyscf_vs_tides_ledger.json
"""

import json
import time
import numpy as np
from pathlib import Path

# PySCF imports
from pyscf import gto, scf, dft, lib
from pyscf.dft import numint
from pyscf.grad import rhf as rhf_grad

# Try GPU-accelerated PySCF
try:
    import cupy
    from pyscf.dft import rks as rks_gpu
    GPU_PYSCF = True
except ImportError:
    GPU_PYSCF = False

import pyscf
PYSCF_VERSION = pyscf.__version__

LEDGER_PATH = Path(__file__).parent / "optimization" / "pyscf_vs_tides_ledger.json"


def time_fn(fn, warmup=1, repeats=3):
    """Time a function with warmup and repeats."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def profile_gemm_cpu(n):
    """Profile dense GEMM on CPU using numpy."""
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    t = time_fn(lambda: A @ B)
    flops = 2 * n**3
    return {"time_s": t, "gflops": flops / t / 1e9, "n": n}


def profile_gemm_gpu(n):
    """Profile dense GEMM on GPU using cupy."""
    if not GPU_PYSCF:
        return None
    import cupy
    A = cupy.random.randn(n, n)
    B = cupy.random.randn(n, n)
    cupy.cuda.Stream.null.synchronize()
    t = time_fn(lambda: cupy.cuda.Stream.null.synchronize() or (A @ B))
    flops = 2 * n**3
    return {"time_s": t, "gflops": flops / t / 1e9, "n": n}


def profile_eig_cpu(n):
    """Profile dense eigendecomposition on CPU."""
    A = np.random.randn(n, n)
    A = A + A.T  # symmetric
    t = time_fn(lambda: np.linalg.eigh(A))
    return {"time_s": t, "n": n}


def profile_eig_gpu(n):
    """Profile dense eigendecomposition on GPU."""
    if not GPU_PYSCF:
        return None
    import cupy
    A = cupy.random.randn(n, n)
    A = A + A.T
    cupy.cuda.Stream.null.synchronize()
    def _eig():
        cupy.linalg.eigh(A)
        cupy.cuda.Stream.null.synchronize()
    t = time_fn(_eig)
    return {"time_s": t, "n": n}


def profile_scf_cpu(atom, basis, xc="LDA"):
    """Profile single-point SCF on CPU with PySCF."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    if xc.upper() == "HF":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc.lower()
    mf.verbose = 0
    # Warmup
    mf.kernel()
    # Time
    t0 = time.perf_counter()
    e = mf.kernel()
    t = time.perf_counter() - t0
    return {"time_s": t, "energy_ha": float(e), "atom": atom, "basis": basis, "xc": xc}


def profile_scf_gpu(atom, basis, xc="LDA"):
    """Profile single-point SCF on GPU with PySCF (if GPU backend available)."""
    if not GPU_PYSCF:
        return None
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    try:
        if xc.upper() == "HF":
            mf = scf.RHF(mol).to_gpu()
        else:
            mf = dft.RKS(mol).to_gpu()
            mf.xc = xc.lower()
        mf.verbose = 0
        mf.kernel()
        t0 = time.perf_counter()
        e = mf.kernel()
        t = time.perf_counter() - t0
        return {"time_s": t, "energy_ha": float(e), "atom": atom, "basis": basis, "xc": xc}
    except Exception as ex:
        return {"error": str(ex), "atom": atom, "basis": basis, "xc": xc}


def profile_grid_xc_cpu(atom, basis, grid_level=3):
    """Profile XC grid evaluation on CPU."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "lda"
    mf.grids.level = grid_level
    mf.verbose = 0
    mf.kernel()
    # Time grid evaluation
    ni = numint.NumInt()
    t0 = time.perf_counter()
    ni.eval_ao(mol, mf.grids.coords, deriv=1)
    t = time.perf_counter() - t0
    n_grid = len(mf.grids.coords)
    return {"time_s": t, "n_grid": n_grid, "atom": atom, "basis": basis}


def profile_forces_fd_cpu(atom, basis):
    """Profile finite-difference forces on CPU."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    # PySCF analytic nuclear gradients
    t0 = time.perf_counter()
    grad = rhf_grad.Gradients(mf).grad()
    t = time.perf_counter() - t0
    return {"time_s": t, "atom": atom, "basis": basis, "grad_max": float(np.max(np.abs(grad)))}


def main():
    print("=" * 70)
    print("PySCF vs TIDES Profiling Comparison")
    print("=" * 70)
    print(f"PySCF version: {PYSCF_VERSION}")
    print(f"GPU available: {GPU_PYSCF}")
    print()

    results = {
        "metadata": {
            "pyscf_version": PYSCF_VERSION,
            "gpu_available": GPU_PYSCF,
            "numpy_version": np.__version__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "gemm": {"cpu": [], "gpu": []},
        "eig": {"cpu": [], "gpu": []},
        "scf": {"cpu": [], "gpu": []},
        "grid_xc": {"cpu": []},
        "forces": {"cpu": []},
    }

    # --- GEMM ---
    print("--- GEMM Profiling ---")
    for n in [64, 128, 256, 512, 1024]:
        r = profile_gemm_cpu(n)
        results["gemm"]["cpu"].append(r)
        print(f"  CPU n={n}: {r['gflops']:.1f} GFLOPS ({r['time_s']*1000:.2f} ms)")
        if GPU_PYSCF:
            r = profile_gemm_gpu(n)
            if r:
                results["gemm"]["gpu"].append(r)
                print(f"  GPU n={n}: {r['gflops']:.1f} GFLOPS ({r['time_s']*1000:.2f} ms)")

    # --- Eigendecomposition ---
    print("\n--- Eigendecomposition Profiling ---")
    for n in [32, 64, 128, 256, 512]:
        r = profile_eig_cpu(n)
        results["eig"]["cpu"].append(r)
        print(f"  CPU n={n}: {r['time_s']*1000:.2f} ms")
        if GPU_PYSCF:
            r = profile_eig_gpu(n)
            if r:
                results["eig"]["gpu"].append(r)
                print(f"  GPU n={n}: {r['time_s']*1000:.2f} ms")

    # --- SCF ---
    print("\n--- SCF Profiling ---")
    test_systems = [
        ("He", "He 0 0 0", "cc-pVDZ"),
        ("Ne", "Ne 0 0 0", "cc-pVDZ"),
        ("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "cc-pVDZ"),
        ("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "cc-pVTZ"),
        ("CH4", "C 0 0 0; H 0 0 1.09; H 0 1.03 -0.36; H 0.98 -0.51 -0.36; H -0.98 -0.51 -0.36", "cc-pVDZ"),
    ]
    for label, atom, basis in test_systems:
        r = profile_scf_cpu(atom, basis, "LDA")
        results["scf"]["cpu"].append(r)
        print(f"  CPU {label}/{basis}: E={r['energy_ha']:.6f} Ha ({r['time_s']*1000:.1f} ms)")
        if GPU_PYSCF:
            r = profile_scf_gpu(atom, basis, "LDA")
            if r and "error" not in r:
                results["scf"]["gpu"].append(r)
                print(f"  GPU {label}/{basis}: E={r['energy_ha']:.6f} Ha ({r['time_s']*1000:.1f} ms)")
            elif r:
                print(f"  GPU {label}/{basis}: ERROR: {r.get('error', 'unknown')}")

    # --- Grid XC ---
    print("\n--- Grid XC Profiling ---")
    for label, atom, basis in [("He", "He 0 0 0", "cc-pVDZ"), ("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "cc-pVTZ")]:
        r = profile_grid_xc_cpu(atom, basis)
        results["grid_xc"]["cpu"].append(r)
        print(f"  CPU {label}/{basis}: {r['n_grid']} grid pts ({r['time_s']*1000:.2f} ms)")

    # --- Forces ---
    print("\n--- Forces Profiling ---")
    for label, atom, basis in [("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "cc-pVDZ"), ("CH4", "C 0 0 0; H 0 0 1.09; H 0 1.03 -0.36; H 0.98 -0.51 -0.36; H -0.98 -0.51 -0.36", "cc-pVDZ")]:
        r = profile_forces_fd_cpu(atom, basis)
        results["forces"]["cpu"].append(r)
        print(f"  CPU {label}/{basis}: max|grad|={r['grad_max']:.6f} ({r['time_s']*1000:.2f} ms)")

    # --- Write ledger ---
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLedger written to: {LEDGER_PATH}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results["gemm"]["cpu"]:
        print(f"GEMM CPU: {results['gemm']['cpu'][-1]['gflops']:.1f} GFLOPS at n={results['gemm']['cpu'][-1]['n']}")
    if results["gemm"]["gpu"]:
        print(f"GEMM GPU: {results['gemm']['gpu'][-1]['gflops']:.1f} GFLOPS at n={results['gemm']['gpu'][-1]['n']}")
    if results["scf"]["cpu"]:
        print(f"SCF CPU: {len(results['scf']['cpu'])} systems profiled")
    if results["scf"]["gpu"]:
        print(f"SCF GPU: {len(results['scf']['gpu'])} systems profiled")
    else:
        print("SCF GPU: NOT AVAILABLE (no GPU PySCF backend)")


if __name__ == "__main__":
    main()
