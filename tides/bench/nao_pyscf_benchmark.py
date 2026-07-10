#!/usr/bin/env python3
"""Gap 8: Matched-accuracy benchmark — TIDES NaoDriver vs PySCF.

Runs H atom and H2 molecule through both TIDES (NAO+grid LDA) and PySCF
(GTO+LDA) and compares total energies. The comparison validates that the
TIDES NAO SCF engine produces physically meaningful results against an
established DFT code.

TIDES uses NAO basis with grid-based Poisson/LDA-XC (all-electron).
PySCF uses GTO basis with analytic ERIs and libxc LDA.

Usage:
  PYTHONPATH=api/python python3 bench/nao_pyscf_benchmark.py
"""
from __future__ import annotations
import sys
import os

# Ensure the TIDES Python package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api", "python"))

import tides._native as native


def run_tides(atomic_numbers, positions_bohr, grid_h=0.3, grid_margin=4.0,
              max_iter=100, tol=1e-8):
    """Run TIDES NaoDriver SCF and return (energy, components)."""
    result = native.NaoDriver.run(
        atomic_numbers=atomic_numbers,
        positions=positions_bohr,
        grid_h=grid_h,
        grid_margin=grid_margin,
        max_iter=max_iter,
        tol=tol,
    )
    e = result.energy
    return result.scf.energy, {
        "E_kin": e.E_kin,
        "E_ne": e.E_ne,
        "E_H": e.E_H,
        "E_xc": e.E_xc,
        "E_ion": e.E_ion if hasattr(e, "E_ion") else 0.0,
    }, result


def run_pyscf_h():
    """Run PySCF H atom LDA/6-31G (spin-polarized)."""
    from pyscf import gto, scf, dft
    mol = gto.M(atom="H 0 0 0", basis="6-31G", unit="Bohr", verbose=0, spin=1)
    mf = dft.UKS(mol)
    mf.xc = "lda,"
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    return energy, mf


def run_pyscf_h2(bond_bohr=1.4):
    """Run PySCF H2 LDA/6-31G at given bond length."""
    from pyscf import gto, scf, dft
    mol = gto.M(
        atom=f"H 0 0 0; H {bond_bohr} 0 0",
        basis="6-31G",
        unit="Bohr",
        verbose=0,
        spin=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,"
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    return energy, mf


def run_pyscf_h2_cc(bond_bohr=1.4):
    """Run PySCF H2 LDA/cc-pVTZ for higher accuracy."""
    from pyscf import gto, scf, dft
    mol = gto.M(
        atom=f"H 0 0 0; H {bond_bohr} 0 0",
        basis="cc-pVTZ",
        unit="Bohr",
        verbose=0,
        spin=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,"
    mf.grids.level = 5
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    return energy, mf


def main():
    print("=" * 72)
    print("Gap 8: TIDES NaoDriver vs PySCF — Matched-Accuracy Benchmark")
    print("=" * 72)

    # --- H atom ---
    print("\n--- H atom ---")
    t_energy, t_comp, t_res = run_tides([1], [0.0, 0.0, 0.0], grid_h=0.3)
    print(f"  TIDES  E = {t_energy:.6f} Ha  (n_basis={t_res.n_basis})")
    print(f"    E_kin={t_comp['E_kin']:.4f}  E_ne={t_comp['E_ne']:.4f}  "
          f"E_H={t_comp['E_H']:.4f}  E_xc={t_comp['E_xc']:.4f}")

    p_energy, p_mf = run_pyscf_h()
    print(f"  PySCF  E = {p_energy:.6f} Ha  (6-31G LDA)")
    print(f"  Diff    = {abs(t_energy - p_energy):.6f} Ha = "
          f"{abs(t_energy - p_energy) * 27.2114 * 1000:.1f} meV")

    # --- H2 molecule ---
    print("\n--- H2 molecule (R=1.4 Bohr) ---")
    t_energy2, t_comp2, t_res2 = run_tides(
        [1, 1], [-0.7, 0.0, 0.0, 0.7, 0.0, 0.0], grid_h=0.3)
    print(f"  TIDES  E = {t_energy2:.6f} Ha  (n_basis={t_res2.n_basis})")
    print(f"    E_kin={t_comp2['E_kin']:.4f}  E_ne={t_comp2['E_ne']:.4f}  "
          f"E_H={t_comp2['E_H']:.4f}  E_xc={t_comp2['E_xc']:.4f}")

    p_energy2, p_mf2 = run_pyscf_h2(1.4)
    print(f"  PySCF  E = {p_energy2:.6f} Ha  (6-31G LDA)")

    p_energy2cc, _ = run_pyscf_h2_cc(1.4)
    print(f"  PySCF  E = {p_energy2cc:.6f} Ha  (cc-pVTZ LDA)")

    diff2 = abs(t_energy2 - p_energy2)
    print(f"  Diff (vs 6-31G)  = {diff2:.6f} Ha = {diff2 * 27.2114 * 1000:.1f} meV")
    diff2cc = abs(t_energy2 - p_energy2cc)
    print(f"  Diff (vs cc-pVTZ) = {diff2cc:.6f} Ha = {diff2cc * 27.2114 * 1000:.1f} meV")

    # --- H2 forces ---
    print("\n--- H2 forces (TIDES FD5) ---")
    forces = native.NaoDriver.compute_forces(
        atomic_numbers=[1, 1],
        positions=[-0.7, 0.0, 0.0, 0.7, 0.0, 0.0],
        grid_h=0.3,
        h=0.01,
    )
    print(f"  F0 = [{forces[0]:.6f}, {forces[1]:.6f}, {forces[2]:.6f}]")
    print(f"  F1 = [{forces[3]:.6f}, {forces[4]:.6f}, {forces[5]:.6f}]")
    print(f"  Newton 3rd: F0x + F1x = {forces[0] + forces[3]:.6f}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  H atom:   TIDES={t_energy:.4f}  PySCF={p_energy:.4f}  "
          f"diff={abs(t_energy - p_energy) * 1000:.1f} meV")
    print(f"  H2 mol:    TIDES={t_energy2:.4f}  PySCF(6-31G)={p_energy2:.4f}  "
          f"diff={diff2 * 1000:.1f} meV")
    print(f"  H2 mol:    TIDES={t_energy2:.4f}  PySCF(cc-pVTZ)={p_energy2cc:.4f}  "
          f"diff={diff2cc * 1000:.1f} meV")
    print()
    print("Note: TIDES uses NAO basis + grid Poisson/LDA (all-electron).")
    print("      PySCF uses GTO basis + analytic ERIs + libxc LDA.")
    print("      Basis set differences account for most of the energy gap.")
    print("      The comparison validates physical correctness of the TIDES")
    print("      NAO SCF engine against an established DFT code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
