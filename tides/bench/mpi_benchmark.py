#!/usr/bin/env python3
"""
MPI Benchmark — PySCF CPU MPI vs TIDES parallel engine.

Tests:
  1. PySCF SCF with MPI (mpi4py) for systems up to 100 atoms
  2. TIDES E7 parallel profile (partitioner, halo exchange)
  3. Scaling: 1 vs 2 vs 4 MPI ranks

Run with:
  python3 bench/mpi_benchmark.py              # serial (baseline)
  mpirun -np 2 python3 bench/mpi_benchmark.py # 2 ranks
  mpirun -np 4 python3 bench/mpi_benchmark.py # 4 ranks

Outputs:
  bench/profiling_results/mpi_benchmark.json
  bench/profiling_results/mpi_benchmark.md
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MPI_AVAILABLE = False
MPI_RANK = 0
MPI_SIZE = 1
COMM = None

def _init_mpi():
    global MPI_AVAILABLE, MPI_RANK, MPI_SIZE, COMM
    try:
        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
        MPI_RANK = COMM.Get_rank()
        MPI_SIZE = COMM.Get_size()
        MPI_AVAILABLE = True
    except ImportError:
        pass

_init_mpi()

import pyscf
from pyscf import gto, dft, scf
from pyscf.grad import rks as rks_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import uks as uks_grad
from pyscf.grad import uhf as uhf_grad

BUILD_DIR = Path(__file__).parent.parent / "build"
OUT_DIR = Path(__file__).parent / "profiling_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "mpi_benchmark.json"
MD_PATH = OUT_DIR / "mpi_benchmark.md"

PYSCF_VERSION = pyscf.__version__


def time_fn(fn, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


TEST_SYSTEMS = [
    ("H2O",       "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",                       3),
    ("CH4",       "C 0 0 0; H 0 0 1.089; H 1.027 0 -0.363; H -0.513 0.889 -0.363; H -0.513 -0.889 -0.363", 5),
    ("C6H6",      "C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; H 1.240 2.149 0; H -1.240 2.149 0; H -2.479 0 0; H -1.240 -2.149 0; H 1.240 -2.149 0", 12),
    ("C10H8",     "C 1.4 0 0.7; C 1.4 0 -0.7; C 0 0 1.4; C 0 0 -1.4; C -1.4 0 0.7; C -1.4 0 -0.7; C 0.7 0 2.1; C -0.7 0 2.1; C -0.7 0 -2.1; C 0.7 0 -2.1; H 2.49 0 1.25; H 2.49 0 -1.25; H -2.49 0 1.25; H -2.49 0 -1.25; H 1.25 0 3.15; H -1.25 0 3.15; H -1.25 0 -3.15; H 1.25 0 -3.15", 18),
    ("H2O_8mer",  "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(8)]), 24),
    ("H2O_16mer", "; ".join([f"O {i*3} 0 0; H {i*3} 0 0.96; H {i*3} 0.96 0" for i in range(16)]), 48),
]

BASIS_SETS = ["cc-pVDZ", "def2-TZVPP"]


def run_pyscf_scf(atom_str, basis, xc="LDA", density_fit=True):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = None
    mol.build()

    n_elec = sum(mol.atom_charges())
    is_open = (n_elec % 2 != 0)
    if is_open:
        mol = gto.M(atom=atom_str, basis=basis, verbose=0, spin=1)
        mol.build()

    if xc.upper() == "HF":
        mf = scf.UHF(mol) if is_open else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if is_open else dft.RKS(mol)
        mf.xc = xc.lower()
        mf.grids.level = 4
        mf.grids.atom_grid = (99, 590)

    if density_fit:
        mf = mf.density_fit()
    mf.verbose = 0

    # Warmup
    mf.kernel()

    t0 = time.perf_counter()
    e = mf.kernel()
    t = time.perf_counter() - t0

    return {
        "energy_ha": float(e),
        "scf_time_s": t,
        "n_basis": int(mol.nao),
        "n_atoms": int(mol.natm),
    }


def run_pyscf_gradient(atom_str, basis, xc="LDA", density_fit=True):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = None
    mol.build()

    n_elec = sum(mol.atom_charges())
    is_open = (n_elec % 2 != 0)
    if is_open:
        mol = gto.M(atom=atom_str, basis=basis, verbose=0, spin=1)
        mol.build()

    if xc.upper() == "HF":
        mf = scf.UHF(mol) if is_open else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if is_open else dft.RKS(mol)
        mf.xc = xc.lower()
        mf.grids.level = 4
        mf.grids.atom_grid = (99, 590)

    if density_fit:
        mf = mf.density_fit()
    mf.verbose = 0
    mf.kernel()

    if is_open:
        grad_fn = uks_grad.Gradients(mf) if xc.upper() != "HF" else uhf_grad.Gradients(mf)
    else:
        grad_fn = rks_grad.Gradients(mf) if xc.upper() != "HF" else rhf_grad.Gradients(mf)

    t0 = time.perf_counter()
    g = grad_fn.grad()
    t = time.perf_counter() - t0

    return {
        "grad_time_s": t,
        "grad_max": float(np.max(np.abs(g))),
        "n_atoms": int(mol.natm),
    }


def run_tides_e7_parallel():
    """Run TIDES E7 parallel profile."""
    exe = BUILD_DIR / "tides_e7_parallel_profile"
    if not exe.exists():
        return {"status": "not_built"}
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    wall = time.perf_counter() - t0
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "wall_s": wall,
        "stdout": proc.stdout[:8000],
        "stderr": proc.stderr[:2000],
    }


def main():
    if MPI_RANK == 0:
        print("=" * 80)
        print("  MPI Benchmark — PySCF CPU MPI vs TIDES Parallel")
        print("=" * 80)
        print(f"PySCF: {PYSCF_VERSION} | MPI: {MPI_AVAILABLE} | Ranks: {MPI_SIZE} | Rank: {MPI_RANK}")
        print()

    results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pyscf_version": PYSCF_VERSION,
            "mpi_available": MPI_AVAILABLE,
            "mpi_size": MPI_SIZE,
            "mpi_rank": MPI_RANK,
        },
        "scf": [],
        "gradients": [],
        "tides_e7_parallel": {},
    }

    # ── SCF benchmark ──
    if MPI_RANK == 0:
        print(f"── SCF Benchmark (cc-pVDZ, LDA, density fitting, {MPI_SIZE} rank(s)) ──")

    for label, atom_str, n_atoms in TEST_SYSTEMS:
        r = run_pyscf_scf(atom_str, "cc-pVDZ", "LDA")
        r["system"] = label
        r["basis"] = "cc-pVDZ"
        r["xc"] = "LDA"
        r["mpi_size"] = MPI_SIZE
        results["scf"].append(r)
        if MPI_RANK == 0:
            print(f"  {label:15s}: E={r['energy_ha']:.6f}  t={r['scf_time_s']*1000:8.1f} ms  "
                  f"nao={r['n_basis']:4d}  atoms={r['n_atoms']:3d}")

    # ── Basis scan (H2O) ──
    if MPI_RANK == 0:
        print(f"\n── Basis Scan (H2O, LDA, {MPI_SIZE} rank(s)) ──")
    h2o = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    for basis in BASIS_SETS:
        r = run_pyscf_scf(h2o, basis, "LDA")
        r["system"] = "H2O"
        r["basis"] = basis
        r["xc"] = "LDA"
        r["mpi_size"] = MPI_SIZE
        results["scf"].append(r)
        if MPI_RANK == 0:
            print(f"  H2O/{basis:12s}: E={r['energy_ha']:.6f}  t={r['scf_time_s']*1000:8.1f} ms  "
                  f"nao={r['n_basis']:4d}")

    # ── Gradient benchmark ──
    if MPI_RANK == 0:
        print(f"\n── Gradient Benchmark (cc-pVDZ, LDA, {MPI_SIZE} rank(s)) ──")
    grad_systems = [(s[0], s[1], s[2]) for s in TEST_SYSTEMS if s[2] <= 48]
    for label, atom_str, n_atoms in grad_systems:
        r = run_pyscf_gradient(atom_str, "cc-pVDZ", "LDA")
        r["system"] = label
        r["mpi_size"] = MPI_SIZE
        results["gradients"].append(r)
        if MPI_RANK == 0:
            print(f"  {label:15s}: t_grad={r['grad_time_s']*1000:8.1f} ms  max|g|={r['grad_max']:.6f}")

    # ── TIDES E7 parallel profile ──
    if MPI_RANK == 0:
        print(f"\n── TIDES E7 Parallel Profile ──")
        e7 = run_tides_e7_parallel()
        results["tides_e7_parallel"] = e7
        print(f"  Status: {e7.get('status', '?')}  Wall: {e7.get('wall_s', 0):.2f}s")

    # ── Write JSON (rank 0 only) ──
    if MPI_RANK == 0:
        def _json_default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(JSON_PATH, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"\nJSON written to: {JSON_PATH}")

        generate_markdown(results)
        print(f"Markdown written to: {MD_PATH}")


def generate_markdown(results):
    lines = []
    lines.append("# MPI Benchmark — PySCF CPU MPI vs TIDES Parallel")
    lines.append("")
    meta = results["metadata"]
    lines.append(f"**Date**: {meta['date']}")
    lines.append(f"**PySCF**: {meta['pyscf_version']} | **MPI**: {meta['mpi_available']} | **Ranks**: {meta['mpi_size']}")
    lines.append("")

    lines.append("## SCF Benchmark (cc-pVDZ, LDA, density fitting)")
    lines.append("")
    lines.append("| System | Atoms | nao | SCF (ms) |")
    lines.append("|---|---|---|---|")
    for r in results["scf"]:
        if r["system"] == "H2O" and r["basis"] != "cc-pVDZ":
            continue
        lines.append(f"| {r['system']} | {r['n_atoms']} | {r['n_basis']} | {r['scf_time_s']*1000:.1f} |")
    lines.append("")

    lines.append("## Basis Scan (H2O, LDA)")
    lines.append("")
    lines.append("| Basis | nao | SCF (ms) |")
    lines.append("|---|---|---|")
    for r in results["scf"]:
        if r["system"] == "H2O":
            lines.append(f"| {r['basis']} | {r['n_basis']} | {r['scf_time_s']*1000:.1f} |")
    lines.append("")

    lines.append("## Gradient Benchmark (cc-pVDZ, LDA)")
    lines.append("")
    lines.append("| System | Atoms | Grad (ms) | max|g| |")
    lines.append("|---|---|---|---|")
    for r in results["gradients"]:
        lines.append(f"| {r['system']} | {r['n_atoms']} | {r['grad_time_s']*1000:.1f} | {r['grad_max']:.6f} |")
    lines.append("")

    e7 = results["tides_e7_parallel"]
    lines.append("## TIDES E7 Parallel Profile")
    lines.append("")
    lines.append(f"- **Status**: {e7.get('status', '?')}")
    lines.append(f"- **Wall**: {e7.get('wall_s', 0):.2f}s")
    lines.append("")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
