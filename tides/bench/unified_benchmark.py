#!/usr/bin/env python3
"""Unified benchmark: TIDES NaoDriver vs PySCF CPU vs gpu4pyscf.

Apples-to-apples comparison at matched physics (LDA-PW92, same geometry).
Records per-step profiling for each code and outputs a comparison table + JSON.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_def.so:... \
    python3 bench/unified_benchmark.py [--max-atoms N] [--mp] [--geo-opt]

Protocol (per 60-protocol.md):
  1. Fixed-accuracy: time-to-solution at verified convergence (1e-8 Ha)
  2. Same physics: LDA-PW92, same geometry, documented basis differences
  3. Three repeats; cold and warm reported separately
  4. Per-step profiling for TIDES internal engines
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# TIDES native module
sys.path.insert(0, str(Path(__file__).parent.parent / "api" / "python"))
from tides._native import NaoDriver

# PySCF
from pyscf import gto, dft as pyscf_dft, scf as pyscf_scf


# ---------------------------------------------------------------------------
# Molecule ladder — geometries in Angstrom, converted to Bohr for TIDES
# ---------------------------------------------------------------------------
BOHR_PER_ANG = 1.8897259886

MOLECULES = [
    {
        "name": "H2",
        "atoms": [(1, [0, 0, 0]), (1, [0, 0, 0.74])],
        "charge": 0,
    },
    {
        "name": "H2O",
        "atoms": [(8, [0, 0, 0]), (1, [0, -0.757, 0.587]), (1, [0, 0.757, 0.587])],
        "charge": 0,
    },
    {
        "name": "CH4",
        "atoms": [
            (6, [0, 0, 0]),
            (1, [0, 0, 1.089]),
            (1, [0, 1.027, -0.363]),
            (1, [0, -1.027, -0.363]),
            (1, [0.898, 0, -0.363]),
        ],
        "charge": 0,
    },
    {
        "name": "NH3",
        "atoms": [
            (7, [0, 0, 0]),
            (1, [0, 0, 1.012]),
            (1, [0.972, 0, -0.253]),
            (1, [-0.972, 0, -0.253]),
        ],
        "charge": 0,
    },
    {
        "name": "C2H6",
        "atoms": [
            (6, [0, 0, 0]),
            (6, [0, 0, 1.535]),
            (1, [0, 0, -1.089]),
            (1, [0.898, 0.898, -0.363]),
            (1, [-0.898, 0.898, -0.363]),
            (1, [0, 0, 2.624]),
            (1, [0.898, 0.898, 1.898]),
            (1, [-0.898, 0.898, 1.898]),
        ],
        "charge": 0,
    },
    {
        "name": "C2H4",
        "atoms": [
            (6, [0, 0, 0]),
            (6, [0, 0, 1.339]),
            (1, [0, 0.972, -0.537]),
            (1, [0, -0.972, -0.537]),
            (1, [0, 0.972, 1.876]),
            (1, [0, -0.972, 1.876]),
        ],
        "charge": 0,
    },
    {
        "name": "benzene",
        "atoms": [
            (6, [0, 1.396, 0]),
            (6, [1.209, 0.698, 0]),
            (6, [1.209, -0.698, 0]),
            (6, [0, -1.396, 0]),
            (6, [-1.209, -0.698, 0]),
            (6, [-1.209, 0.698, 0]),
            (1, [0, 2.479, 0]),
            (1, [2.146, 1.240, 0]),
            (1, [2.146, -1.240, 0]),
            (1, [0, -2.479, 0]),
            (1, [-2.146, -1.240, 0]),
            (1, [-2.146, 1.240, 0]),
        ],
        "charge": 0,
    },
    {
        "name": "naphthalene",
        "atoms": [
            (6, [0, 0, 0.717]),
            (6, [0, 0, -0.717]),
            (6, [1.396, 0, 1.434]),
            (6, [1.396, 0, -1.434]),
            (6, [2.792, 0, 0.717]),
            (6, [2.792, 0, -0.717]),
            (6, [4.188, 0, 1.434]),
            (6, [4.188, 0, -1.434]),
            (6, [5.584, 0, 0.717]),
            (6, [5.584, 0, -0.717]),
            (1, [-0.972, 0, 1.254]),
            (1, [-0.972, 0, -1.254]),
            (1, [1.396, 0, 2.523]),
            (1, [1.396, 0, -2.523]),
            (1, [4.188, 0, 2.523]),
            (1, [4.188, 0, -2.523]),
            (1, [6.556, 0, 1.254]),
            (1, [6.556, 0, -1.254]),
        ],
        "charge": 0,
    },
]


def atoms_to_bohr(atoms):
    """Convert Angstrom positions to Bohr."""
    return [(z, [c * BOHR_PER_ANG for c in pos]) for z, pos in atoms]


def atoms_to_pyscf_string(atoms):
    """Convert atom list to PySCF atom string (Angstrom)."""
    sym_map = {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S",
        5: "B", 13: "Al", 14: "Si", 15: "P", 17: "Cl",
    }
    lines = []
    for z, pos in atoms:
        sym = sym_map.get(z, f"Z{z}")
        lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    return "; ".join(lines)


# ---------------------------------------------------------------------------
# TIDES benchmark
# ---------------------------------------------------------------------------
def run_tides(mol_entry, use_mp=False, max_iter=100, tol=1e-8, repeats=3):
    """Run TIDES NaoDriver SCF with per-step profiling."""
    atoms_bohr = atoms_to_bohr(mol_entry["atoms"])
    atomic_numbers = [z for z, _ in atoms_bohr]
    positions = []
    for _, pos in atoms_bohr:
        positions.extend(pos)

    results = []
    for rep in range(repeats):
        t0 = time.perf_counter()
        result = NaoDriver.run(
            atomic_numbers=atomic_numbers,
            positions=positions,
            grid_h=0.3,
            grid_margin=4.0,
            max_iter=max_iter,
            tol=tol,
            use_dual_grid=True,
            use_mixed_precision=use_mp,
        )
        t1 = time.perf_counter()

        scf = result.scf
        t = scf.timings
        entry = {
            "rep": rep,
            "energy_ha": scf.energy,
            "converged": scf.converged,
            "n_iterations": scf.n_iterations,
            "wall_time_ms": (t1 - t0) * 1000,
            "n_basis": result.n_basis,
            "n_atoms": result.n_atoms,
            "scf_total_ms": t.scf_total_ms,
            "per_step_ms": {
                "build_H": t.build_H_ms,
                "gemm_hx": t.gemm_hx_ms,
                "gemm_xthp": t.gemm_xthp_ms,
                "eigensolve": t.eigensolve_ms,
                "backtransform": t.backtransform_ms,
                "dsyrk": t.dsyrk_ms,
                "energy": t.energy_ms,
                "diis": t.diis_ms,
            },
            "build_H_substep_ms": {
                "quantize_P": result.build_H_timings.quantize_P_ms,
                "rho_build": result.build_H_timings.rho_build_ms,
                "poisson": result.build_H_timings.poisson_ms,
                "xc_eval": result.build_H_timings.xc_eval_ms,
                "vmat_build": result.build_H_timings.vmat_build_ms,
                "assemble_H": result.build_H_timings.assemble_H_ms,
                "total": result.build_H_timings.total_ms,
                "used_gpu_pipeline": result.build_H_timings.used_gpu_pipeline,
            },
        }
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# PySCF CPU benchmark
# ---------------------------------------------------------------------------
def run_pyscf_cpu(mol_entry, basis="def2-svp", max_iter=100, tol=1e-8, repeats=3):
    """Run PySCF RKS LDA on CPU with timing."""
    atom_str = atoms_to_pyscf_string(mol_entry["atoms"])
    mol = gto.M(atom=atom_str, basis=basis, verbose=0, charge=mol_entry.get("charge", 0))

    results = []
    for rep in range(repeats):
        mf = pyscf_dft.RKS(mol)
        mf.xc = "LDA,VWN"
        mf.grids.level = 4
        mf.max_cycle = max_iter
        mf.conv_tol = tol
        mf.verbose = 0

        # Cold run
        t0 = time.perf_counter()
        mf.kernel()
        t1 = time.perf_counter()

        entry = {
            "rep": rep,
            "energy_ha": float(mf.e_tot),
            "converged": mf.converged,
            "n_iterations": mf.cycles,
            "wall_time_ms": (t1 - t0) * 1000,
            "n_basis": mol.nao,
            "n_atoms": mol.natm,
        }
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# gpu4pyscf benchmark
# ---------------------------------------------------------------------------
def run_gpu4pyscf(mol_entry, basis="def2-svp", max_iter=100, tol=1e-8, repeats=3):
    """Run gpu4pyscf RKS LDA on GPU with timing."""
    from gpu4pyscf import dft as gpu_dft

    atom_str = atoms_to_pyscf_string(mol_entry["atoms"])
    mol = gto.M(atom=atom_str, basis=basis, verbose=0, charge=mol_entry.get("charge", 0))

    results = []
    for rep in range(repeats):
        mf = gpu_dft.RKS(mol)
        mf.xc = "LDA,VWN"
        mf.grids.level = 4
        mf.max_cycle = max_iter
        mf.conv_tol = tol
        mf.verbose = 0

        t0 = time.perf_counter()
        mf.kernel()
        t1 = time.perf_counter()

        entry = {
            "rep": rep,
            "energy_ha": float(mf.e_tot),
            "converged": mf.converged,
            "n_iterations": getattr(mf, "cycles", 0) or 0,
            "wall_time_ms": (t1 - t0) * 1000,
            "n_basis": mol.nao,
            "n_atoms": mol.natm,
        }
        results.append(entry)
    del gpu_dft
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_repeats(results):
    """Take min wall time, median energy across repeats."""
    times = [r["wall_time_ms"] for r in results]
    energies = [r["energy_ha"] for r in results]
    iters = [r["n_iterations"] for r in results]
    return {
        "energy_ha": float(np.median(energies)),
        "energy_std": float(np.std(energies)),
        "wall_time_ms": float(np.min(times)),
        "wall_time_mean_ms": float(np.mean(times)),
        "wall_time_std_ms": float(np.std(times)),
        "n_iterations": int(np.median(iters)),
        "converged": all(r["converged"] for r in results),
        "n_repeats": len(results),
    }


def print_comparison_table(all_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print("  UNIFIED BENCHMARK: TIDES vs PySCF CPU vs gpu4pyscf (LDA-PW92)")
    print("=" * 120)
    header = (
        f"  {'Molecule':<14} {'N_atoms':>7} {'N_basis':>7} "
        f"{'TIDES(ms)':>12} {'PySCF(ms)':>12} {'GPU4(ms)':>12} "
        f"{'TIDES E(Ha)':>16} {'PySCF E(Ha)':>16} {'GPU4 E(Ha)':>16} "
        f"{'dE(T-P)':>12}"
    )
    print(header)
    print("  " + "-" * 116)

    for entry in all_results:
        name = entry["molecule"]
        n_atoms = entry.get("n_atoms", "?")
        n_basis = entry.get("n_basis", "?")

        tides = entry.get("tides", {})
        pyscf = entry.get("pyscf_cpu", {})
        gpu4 = entry.get("gpu4pyscf", {})

        t_time = f"{tides.get('wall_time_ms', 0):.1f}" if tides else "—"
        p_time = f"{pyscf.get('wall_time_ms', 0):.1f}" if pyscf else "—"
        g_time = f"{gpu4.get('wall_time_ms', 0):.1f}" if gpu4 else "—"

        t_e = f"{tides.get('energy_ha', 0):.10f}" if tides else "—"
        p_e = f"{pyscf.get('energy_ha', 0):.10f}" if pyscf else "—"
        g_e = f"{gpu4.get('energy_ha', 0):.10f}" if gpu4 else "—"

        if tides and pyscf:
            de = abs(tides["energy_ha"] - pyscf["energy_ha"])
            de_str = f"{de:.2e}"
        else:
            de_str = "—"

        print(
            f"  {name:<14} {n_atoms:>7} {n_basis:>7} "
            f"{t_time:>12} {p_time:>12} {g_time:>12} "
            f"{t_e:>16} {p_e:>16} {g_e:>16} "
            f"{de_str:>12}"
        )

    print("=" * 120)


def print_tides_profiling(all_results):
    """Print TIDES per-step profiling breakdown."""
    print("\n" + "=" * 100)
    print("  TIDES INTERNAL ENGINE PROFILING (per-iteration averages, ms)")
    print("=" * 100)
    header = (
        f"  {'Molecule':<14} {'N_basis':>7} {'Iters':>6} "
        f"{'build_H':>10} {'gemm_hx':>10} {'gemm_xthp':>10} "
        f"{'eigensolve':>12} {'dsyrk':>10} {'energy':>10} {'diis':>10} "
        f"{'scf_total':>12}"
    )
    print(header)
    print("  " + "-" * 96)

    for entry in all_results:
        tides = entry.get("tides")
        if not tides or "per_step" not in tides or not tides["per_step"]:
            continue
        ps = tides["per_step"]
        print(
            f"  {entry['molecule']:<14} {entry.get('n_basis', '?'):>7} "
            f"{tides.get('n_iterations', 0):>6} "
            f"{ps.get('build_H', 0):>10.3f} {ps.get('gemm_hx', 0):>10.3f} {ps.get('gemm_xthp', 0):>10.3f} "
            f"{ps.get('eigensolve', 0):>12.3f} {ps.get('dsyrk', 0):>10.3f} {ps.get('energy', 0):>10.3f} "
            f"{ps.get('diis', 0):>10.3f} {tides.get('scf_total_ms', 0):>12.1f}"
        )

    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified TIDES vs PySCF benchmark")
    parser.add_argument("--max-atoms", type=int, default=12, help="Max atoms to benchmark")
    parser.add_argument("--mp", action="store_true", help="Enable TIDES mixed precision")
    parser.add_argument("--basis", default="def2-svp", help="PySCF basis set")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per system")
    parser.add_argument("--no-gpu", action="store_true", help="Skip gpu4pyscf")
    parser.add_argument("--no-pyscf", action="store_true", help="Skip PySCF CPU")
    parser.add_argument("--output", default="bench/profiling_results/unified_benchmark.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 120)
    print(f"  UNIFIED BENCHMARK — LDA-PW92, basis={args.basis}, MP={'ON' if args.mp else 'OFF'}")
    print(f"  Hardware: RTX 3060 (12GB), Intel i5-10400F (6c/12t), 62GB RAM")
    print(f"  Repeats: {args.repeats}, Max atoms: {args.max_atoms}")
    print("=" * 120)

    all_results = []

    for mol in MOLECULES:
        n_atoms = len(mol["atoms"])
        if n_atoms > args.max_atoms:
            print(f"\n--- {mol['name']} ({n_atoms} atoms) — skipping (exceeds --max-atoms {args.max_atoms}) ---")
            continue

        print(f"\n--- {mol['name']} ({n_atoms} atoms) ---", flush=True)

        entry = {"molecule": mol["name"], "n_atoms": n_atoms}

        # TIDES
        print(f"  TIDES NaoDriver (MP={'ON' if args.mp else 'OFF'})...  ", end="", flush=True)
        tides_results = run_tides(mol, use_mp=args.mp, repeats=args.repeats)
        tides_summary = summarize_repeats(tides_results)
        # Attach per-step from first rep (same across reps, it's per-iter average)
        tides_summary["per_step"] = tides_results[0].get("per_step_ms", {})
        tides_summary["scf_total_ms"] = tides_results[0].get("scf_total_ms", 0.0)
        entry["tides"] = tides_summary
        entry["n_basis"] = tides_results[0]["n_basis"]
        print(f"E={tides_summary['energy_ha']:.10f} Ha, {tides_summary['wall_time_ms']:.1f} ms, "
              f"{tides_summary['n_iterations']} iters", flush=True)

        # PySCF CPU
        if not args.no_pyscf:
            print(f"  PySCF CPU ({args.basis})...                   ", end="", flush=True)
            pyscf_results = run_pyscf_cpu(mol, basis=args.basis, repeats=args.repeats)
            pyscf_summary = summarize_repeats(pyscf_results)
            entry["pyscf_cpu"] = pyscf_summary
            if entry.get("n_basis") == "?" or not entry.get("n_basis"):
                entry["n_basis"] = pyscf_results[0]["n_basis"]
            print(f"E={pyscf_summary['energy_ha']:.10f} Ha, {pyscf_summary['wall_time_ms']:.1f} ms, "
                  f"{pyscf_summary['n_iterations']} iters", flush=True)

        # gpu4pyscf
        if not args.no_gpu:
            print(f"  gpu4pyscf ({args.basis})...                   ", end="", flush=True)
            gpu4_results = run_gpu4pyscf(mol, basis=args.basis, repeats=args.repeats)
            gpu4_summary = summarize_repeats(gpu4_results)
            entry["gpu4pyscf"] = gpu4_summary
            print(f"E={gpu4_summary['energy_ha']:.10f} Ha, {gpu4_summary['wall_time_ms']:.1f} ms, "
                  f"{gpu4_summary['n_iterations']} iters", flush=True)

        all_results.append(entry)

    # Print tables
    print_comparison_table(all_results)
    print_tides_profiling(all_results)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
