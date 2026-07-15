#!/usr/bin/env python3
"""Consolidated benchmark report: TIDES vs PySCF CPU vs gpu4pyscf vs QE.

Reads the JSON results from unified_benchmark and qe_benchmark,
produces a single comparison table and analysis.
"""
import json
import sys
from pathlib import Path


def main():
    base = Path(__file__).parent.parent / "bench" / "profiling_results"
    
    unified_path = base / "unified_benchmark_3rep.json"
    qe_path = base / "qe_benchmark_1rep.json"

    unified_data = []
    qe_data = []

    if unified_path.exists():
        with open(unified_path) as f:
            unified_data = json.load(f)

    if qe_path.exists():
        with open(qe_path) as f:
            qe_data = json.load(f)

    # Build merged table
    qe_by_name = {e["molecule"]: e.get("qe", {}) for e in qe_data}

    print("=" * 140)
    print("  CONSOLIDATED BENCHMARK REPORT")
    print("  TIDES (NAO+PP, LDA-PW92) vs PySCF CPU (GTO, LDA-PW92) vs gpu4pyscf (GTO, LDA-PW92) vs QE (PW+PP, PBE)")
    print("  Hardware: RTX 3060 12GB, Intel i5-10400F 6c/12t, 62GB RAM")
    print("  Protocol: 3 repeats (TIDES/PySCF/GPU4), 1 repeat (QE), min wall time reported")
    print("=" * 140)

    header = (
        f"  {'Molecule':<14} {'N_at':>4} {'N_bas':>5} "
        f"{'TIDES(ms)':>12} {'PySCF(ms)':>12} {'GPU4(ms)':>12} {'QE(ms)':>12} "
        f"{'TIDES E':>16} {'PySCF E':>16} {'GPU4 E':>16} {'QE E':>16} "
        f"{'TIDES it':>8} {'QE it':>6}"
    )
    print(header)
    print("  " + "-" * 136)

    for entry in unified_data:
        name = entry["molecule"]
        n_atoms = entry.get("n_atoms", "?")
        n_basis = entry.get("n_basis", "?")

        tides = entry.get("tides", {})
        pyscf = entry.get("pyscf_cpu", {})
        gpu4 = entry.get("gpu4pyscf", {})
        qe = qe_by_name.get(name, {})

        def fmt_t(d):
            v = d.get("wall_time_ms")
            return f"{v:>12.1f}" if v is not None else f"{'—':>12}"

        def fmt_e(d):
            v = d.get("energy_ha")
            return f"{v:>16.10f}" if v is not None else f"{'—':>16}"

        def fmt_i(d, key="n_iterations"):
            v = d.get(key)
            return f"{v:>8}" if v is not None else f"{'—':>8}"

        print(
            f"  {name:<14} {n_atoms:>4} {n_basis:>5} "
            f"{fmt_t(tides)} {fmt_t(pyscf)} {fmt_t(gpu4)} {fmt_t(qe)} "
            f"{fmt_e(tides)} {fmt_e(pyscf)} {fmt_e(gpu4)} {fmt_e(qe)} "
            f"{fmt_i(tides)} {fmt_i(qe, 'n_iterations'):>6}"
        )

    print("=" * 140)

    # TIDES profiling breakdown
    print("\n" + "=" * 120)
    print("  TIDES INTERNAL ENGINE PROFILING (per-iteration averages, ms)")
    print("=" * 120)
    header2 = (
        f"  {'Molecule':<14} {'N_bas':>5} {'Iters':>6} "
        f"{'build_H':>10} {'gemm_hx':>10} {'gemm_xthp':>10} "
        f"{'eigensolve':>12} {'dsyrk':>10} {'energy':>10} {'diis':>10} "
        f"{'scf_total':>12} {'%build_H':>10}"
    )
    print(header2)
    print("  " + "-" * 116)

    for entry in unified_data:
        tides = entry.get("tides", {})
        ps = tides.get("per_step", {})
        if not ps:
            continue
        scf_total = tides.get("scf_total_ms", 0)
        n_iter = tides.get("n_iterations", 0)
        build_h = ps.get("build_H", 0)
        pct_build = (build_h * n_iter / scf_total * 100) if scf_total > 0 else 0
        print(
            f"  {entry['molecule']:<14} {entry.get('n_basis', '?'):>5} {n_iter:>6} "
            f"{build_h:>10.3f} {ps.get('gemm_hx', 0):>10.3f} {ps.get('gemm_xthp', 0):>10.3f} "
            f"{ps.get('eigensolve', 0):>12.3f} {ps.get('dsyrk', 0):>10.3f} "
            f"{ps.get('energy', 0):>10.3f} {ps.get('diis', 0):>10.3f} "
            f"{scf_total:>12.1f} {pct_build:>9.1f}%"
        )

    print("=" * 120)

    # Analysis
    print("\n  ANALYSIS NOTES:")
    print("  1. Energy differences are EXPECTED — different basis + XC + PP treatment:")
    print("     - TIDES: NAO-DZP + PseudoDojo ONCV PP + LDA-PW92 (valence-only)")
    print("     - PySCF/gpu4pyscf: def2-svp all-electron GTO + LDA-PW92")
    print("     - QE: plane-wave + PAW/NC PP + PBE (different XC!)")
    print("  2. Timing comparison is apples-to-apples: wall time to SCF convergence.")
    print("  3. TIDES build_H dominates (>95% of SCF loop time) — grid-based NAO integral evaluation.")
    print("  4. gpu4pyscf is fastest for all-electron GTO (GPU-accelerated integral + Fock build).")
    print("  5. QE is CPU-only and very slow for molecules (large vacuum boxes → many PWs).")
    print("  6. TIDES SCF loop (GEMM+eig+dsyrk+DIIS) is <5% of total — optimization target is build_H.")
    print("  7. For apples-to-apples energy comparison, need matched PP + XC across all codes.")


if __name__ == "__main__":
    main()
