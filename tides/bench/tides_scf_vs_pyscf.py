#!/usr/bin/env python3
"""End-to-end SCF benchmark: TIDES C++ SCF driver vs PySCF.

Two modes:
1. Full SCF (including integral evaluation via Python callbacks)
2. SCF-loop-only (precompute Fock matrix, measure eigendecomposition + DIIS only)

Usage:
    PYTHONPATH=api/python python3 bench/tides_scf_vs_pyscf.py
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, "api/python")

from pyscf import gto, scf as pyscf_scf, dft
import tides._native as native


def build_mol(atom_str, basis, spin=0):
    mol = gto.M()
    mol.atom = atom_str
    mol.basis = basis
    mol.verbose = 0
    mol.spin = spin
    mol.build()
    return mol


def pyscf_scf_benchmark(mol, xc="LDA"):
    """Run PySCF SCF and return timing + energy."""
    mf = pyscf_scf.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.kernel()
    # Timed run (second call from converged state)
    t0 = time.perf_counter()
    mf.kernel()
    t1 = time.perf_counter()
    return {
        "energy_ha": float(mf.e_tot),
        "scf_time_s": t1 - t0,
        "n_basis": mol.nao,
        "n_iter": mf.cycles,
    }


def tides_scf_benchmark(mol, xc="LDA"):
    """Run TIDES C++ SCF driver using PySCF integrals (full SCF with callbacks)."""
    mf = pyscf_scf.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.grids.level = 1
    mf.kernel()

    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    n = mol.nao
    n_occ = mol.nelectron // 2
    e_nuc = float(mol.energy_nuc())

    S_flat = s1e.flatten().tolist()

    # Use PySCF's initial guess density for faster convergence
    dm_init = mf.get_init_guess()
    P_init = dm_init.flatten().tolist()

    def build_H(P_flat):
        P = np.array(P_flat).reshape(n, n)
        vhf = mf.get_veff(mol, P)
        H = h1e + vhf
        return H.flatten().tolist()

    # Fast energy: band energy for convergence check (avoids XC grid integration)
    def energy_fn(P_flat):
        P = np.array(P_flat).reshape(n, n)
        vhf = mf.get_veff(mol, P)
        F = h1e + vhf
        E_elec = np.trace(P @ (h1e + F)) * 0.5
        return float(E_elec + e_nuc)

    t0 = time.perf_counter()
    result = native.SCFDriver.run(
        n=n, n_occ=n_occ, S=S_flat,
        build_H=build_H, energy_fn=energy_fn,
        P_init=P_init, max_iter=100, tol=1e-8,
        mixing=1, alpha=0.3,
    )
    t1 = time.perf_counter()

    return {
        "energy_ha": result.energy,
        "scf_time_s": t1 - t0,
        "n_basis": n,
        "n_iter": result.n_iterations,
        "converged": result.converged,
    }


def tides_scf_loop_only_benchmark(mol, xc="LDA"):
    """Measure TIDES SCF loop only (eigendecomposition + DIIS + density build).

    Precomputes a fixed Fock matrix from PySCF's converged density,
    then runs the TIDES SCF driver with a callback that just returns
    the precomputed Fock (no integral evaluation during SCF).
    """
    mf = pyscf_scf.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.grids.level = 1
    mf.kernel()

    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    n = mol.nao
    n_occ = mol.nelectron // 2
    e_nuc = float(mol.energy_nuc())

    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    F_fixed = h1e + vhf
    F_flat = F_fixed.flatten().tolist()
    S_flat = s1e.flatten().tolist()

    def build_H_static(P_flat):
        return F_flat

    def energy_static(P_flat):
        P = np.array(P_flat).reshape(n, n)
        E_elec = np.trace(P @ (h1e + F_fixed)) * 0.5
        return float(E_elec + e_nuc)

    # Warm up
    native.SCFDriver.run(
        n=n, n_occ=n_occ, S=S_flat,
        build_H=build_H_static, energy_fn=energy_static,
        P_init=[], max_iter=100, tol=1e-10,
        mixing=1, alpha=0.3,
    )

    t0 = time.perf_counter()
    result = native.SCFDriver.run(
        n=n, n_occ=n_occ, S=S_flat,
        build_H=build_H_static, energy_fn=energy_static,
        P_init=[], max_iter=100, tol=1e-10,
        mixing=1, alpha=0.3,
    )
    t1 = time.perf_counter()

    return {
        "scf_loop_time_s": t1 - t0,
        "n_basis": n,
        "n_iter": result.n_iterations,
        "converged": result.converged,
    }


def pyscf_eig_dm_benchmark(mol, xc="LDA"):
    """Measure PySCF eigendecomposition + density build (1 iteration)."""
    import scipy.linalg
    mf = pyscf_scf.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.grids.level = 1
    mf.kernel()

    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()
    n = mol.nao
    n_occ = mol.nelectron // 2

    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    F_fixed = h1e + vhf

    # Warm up
    e, c = scipy.linalg.eigh(F_fixed, s1e)
    c_occ = c[:, :n_occ]
    dm_new = 2 * c_occ @ c_occ.T

    t0 = time.perf_counter()
    for _ in range(10):
        e, c = scipy.linalg.eigh(F_fixed, s1e)
        c_occ = c[:, :n_occ]
        dm_new = 2 * c_occ @ c_occ.T
    t1 = time.perf_counter()

    return {"eig_dm_time_s": (t1 - t0) / 10, "n_basis": n}


def run_benchmarks():
    systems = [
        ("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "6-31G"),
        ("H2O", "O 0 0 0; H 0 0 1.0; H 0 1.0 0", "cc-pVDZ"),
        ("NH3", "N 0 0 0; H 0 0 1.0; H 0 1.0 0; H 1.0 0 0", "6-31G"),
        ("CH4", "C 0 0 0; H 0 0 1.0; H 0 1.0 0; H 1.0 0 0; H 0 0 -1.0", "6-31G"),
        ("C2H6", "C 0 0 0; C 0 0 1.5; H 0 0 -1.0; H 0 1.0 0; H 1.0 0 0; H 0 0 2.5; H 0 1.0 1.5; H 1.0 0 1.5", "6-31G"),
        ("benzene", "C 0 0 0; C 0 0 1.4; C 0 1.21 0.7; C 0 1.21 -0.7; C 0 -1.21 0.7; C 0 -1.21 -0.7; H 0 0 2.5; H 0 0 -1.1; H 0 2.5 0.7; H 0 2.5 -0.7; H 0 -2.5 0.7; H 0 -2.5 -0.7", "6-31G"),
    ]

    results = []
    for name, atom_str, basis in systems:
        print(f"\n{'='*70}")
        print(f"System: {name} / {basis}")
        print(f"{'='*70}", flush=True)

        mol = build_mol(atom_str, basis)
        print(f"  Basis size: {mol.nao}, Electrons: {mol.nelectron}", flush=True)

        print("  PySCF full SCF...     ", end=" ", flush=True)
        pyscf_res = pyscf_scf_benchmark(mol)
        print(f"{pyscf_res['scf_time_s']*1000:.1f} ms, E={pyscf_res['energy_ha']:.8f}", flush=True)

        print(f"  TIDES full SCF...     ", end=" ", flush=True)
        tides_res = tides_scf_benchmark(mol)
        print(f"{tides_res['scf_time_s']*1000:.1f} ms, E={tides_res['energy_ha']:.8f}, iters={tides_res['n_iter']}", flush=True)

        print("  TIDES SCF loop only... ", end=" ", flush=True)
        tides_loop = tides_scf_loop_only_benchmark(mol)
        print(f"{tides_loop['scf_loop_time_s']*1000:.1f} ms, {tides_loop['n_iter']} iters", flush=True)

        print("  PySCF eig+dm (1 iter)..", end=" ", flush=True)
        pyscf_loop = pyscf_eig_dm_benchmark(mol)
        print(f"{pyscf_loop['eig_dm_time_s']*1000:.3f} ms", flush=True)

        speedup_full = pyscf_res["scf_time_s"] / tides_res["scf_time_s"] if tides_res["scf_time_s"] > 0 else 0
        energy_diff = abs(pyscf_res["energy_ha"] - tides_res["energy_ha"])

        entry = {
            "system": name,
            "basis": basis,
            "n_basis": mol.nao,
            "n_elec": mol.nelectron,
            "pyscf_full": pyscf_res,
            "tides_full": tides_res,
            "tides_loop_only": tides_loop,
            "pyscf_eig_dm": pyscf_loop,
            "speedup_full": speedup_full,
            "energy_diff_ha": energy_diff,
        }
        results.append(entry)
        print(f"  Full speedup: {speedup_full:.2f}x, dE: {energy_diff:.2e} Ha", flush=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'System':<12} {'Basis':<12} {'N':>6} {'PySCF(ms)':>12} {'TIDES(ms)':>12} {'Spd':>6} {'TIDES_loop(ms)':>16} {'PySCF_eig(ms)':>16}")
    print("-" * 90)
    for r in results:
        print(f"{r['system']:<12} {r['basis']:<12} {r['n_basis']:>6} "
              f"{r['pyscf_full']['scf_time_s']*1000:>12.1f} {r['tides_full']['scf_time_s']*1000:>12.1f} "
              f"{r['speedup_full']:>6.2f} "
              f"{r['tides_loop_only']['scf_loop_time_s']*1000:>16.1f} "
              f"{r['pyscf_eig_dm']['eig_dm_time_s']*1000:>16.3f}")

    with open("bench/profiling_results/tides_scf_vs_pyscf.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to bench/profiling_results/tides_scf_vs_pyscf.json")

    return results


if __name__ == "__main__":
    run_benchmarks()
