#!/usr/bin/env python3
"""Self-test for the Becke terms oracle (task p06b-becke-oracle).

Generates a **synthetic** dump directory (2 atoms, s and p functions built
from *known* Gaussians sampled on a radial grid) where exact S/T/V are
computable analytically (or via high-precision numerical integration), then
runs the Becke oracle and asserts it matches to the required tolerances:

    S  ≤ 1e-8
    T  ≤ 1e-6
    V  ≤ 1e-6  (-Z/r)

This is the acceptance test for task p06b.  It requires **no TIDES C++** —
everything is Python + numpy + scipy.

Run::

    python3 tides/verification/runners/selftest_becke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import List, Tuple

import numpy as np

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from becke_terms_oracle import (  # noqa: E402
    run_becke_oracle,
    compute_matrices,
    load_dump,
    BasisEvaluator,
    real_sph_harm,
    generate_radial_grid,
    generate_angular_grid,
    _becke_cell_weight,
    build_becke_mesh,
)
from compare_pyscf_terms import z_to_symbol  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic-basis definition — same pattern as selftest_synthetic.py
# ---------------------------------------------------------------------------

# Two atoms: H (Z=1) at origin and He (Z=2) at [2.5, 0, 0]
# This gives a non-trivial V_ext for validation.
ATOMS = [
    {"Z": 1, "pos_bohr": [0.0, 0.0, 0.0]},
    {"Z": 2, "pos_bohr": [2.5, 0.0, 0.0]},
]

# Basis: atom 0 has s + p, atom 1 has s + p
# Each fn is built from known Gaussians so analytic values are computable
# via PySCF (we don't use PySCF in the oracle, but we use it for the reference)
SYNTHETIC_BASIS = [
    # atom 0 — s function (2 primitives)
    {"atom": 0, "l": 0, "m": 0, "fn": 0, "primitives": [(0.5, 1.0), (2.0, 0.4)]},
    # atom 0 — p function (1 primitive)
    {"atom": 0, "l": 1, "m": -1, "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 0, "l": 1, "m": 0, "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 0, "l": 1, "m": 1, "fn": 1, "primitives": [(0.8, 1.0)]},
    # atom 1 — s function (1 primitive, different from atom 0)
    {"atom": 1, "l": 0, "m": 0, "fn": 0, "primitives": [(0.7, 1.0)]},
    # atom 1 — p function
    {"atom": 1, "l": 1, "m": -1, "fn": 1, "primitives": [(1.2, 1.0)]},
    {"atom": 1, "l": 1, "m": 0, "fn": 1, "primitives": [(1.2, 1.0)]},
    {"atom": 1, "l": 1, "m": 1, "fn": 1, "primitives": [(1.2, 1.0)]},
]

R_MAX = 12.0  # Gaussians are negligible beyond ~10 Bohr; tighter r_cut improves grid density
N_R = 800


def radial_value(r: np.ndarray, l: int, primitives: List[Tuple[float, float]]) -> np.ndarray:
    """R(r) = Σ_k c_k · r^l · exp(-α_k·r²)"""
    R = np.zeros_like(r)
    for alpha, coeff in primitives:
        R += coeff * r ** l * np.exp(-alpha * r ** 2)
    return R


def generate_synthetic_dump(dump_dir: str) -> dict:
    """Create a synthetic dump directory and return its metadata.

    The dump is built from *known* Gaussian primitives so that exact S/T/V
    are computable by PySCF directly.  The Becke oracle must recover these
    matrices to the required precision.
    """
    os.makedirs(dump_dir, exist_ok=True)

    # --- Build metadata ----------------------------------------------------
    basis_entries = []
    for entry in SYNTHETIC_BASIS:
        basis_entries.append({
            "atom": entry["atom"],
            "l": entry["l"],
            "m": entry["m"],
            "fn": entry["fn"],
        })
    meta = {
        "n_basis": len(SYNTHETIC_BASIS),
        "grid_h": 0.3,
        "atoms": ATOMS,
        "basis": basis_entries,
    }

    # --- Write radial files (unique by atom+fn) ---------------------------
    r_grid = np.linspace(0.0, R_MAX, N_R)
    written = set()
    for entry in SYNTHETIC_BASIS:
        key = (entry["atom"], entry["fn"])
        if key in written:
            continue
        R = radial_value(r_grid, entry["l"], entry["primitives"])
        # Normalize so that ∫ R² r² dr = 1 (matching PySCF contraction normalization)
        norm_sq = np.trapz(R ** 2 * r_grid ** 2, r_grid)
        assert norm_sq > 0, f"Zero-norm radial for atom {entry['atom']} fn {entry['fn']}"
        R = R / np.sqrt(norm_sq)
        np.savetxt(
            os.path.join(dump_dir, f"radial_{entry['atom']}_{entry['fn']}.txt"),
            np.column_stack([r_grid, R]),
            fmt="%.17g",
        )
        written.add(key)

    # --- Build reference matrices via PySCF --------------------------------
    from pyscf import gto
    from fit_nao_gaussians import primitive_normalization

    atom_list = [[z_to_symbol(a["Z"]), a["pos_bohr"]] for a in ATOMS]

    # Collect per-(atom, l) primitives
    atom_l_prims: dict = {}
    for entry in SYNTHETIC_BASIS:
        key = (entry["atom"], entry["l"])
        if key not in atom_l_prims:
            atom_l_prims[key] = []
        for p in entry["primitives"]:
            atom_l_prims[key].append(p)

    # Build element-keyed basis dict
    # Both atoms are different elements (H and He)
    element_basis: dict = {}
    for atom_idx in range(len(ATOMS)):
        sym = z_to_symbol(ATOMS[atom_idx]["Z"])
        if sym not in element_basis:
            contractions = []
            for l_val in [0, 1]:
                key = (atom_idx, l_val)
                if key not in atom_l_prims:
                    continue
                contraction = [l_val]
                for alpha, coeff in atom_l_prims[key]:
                    nk = primitive_normalization(alpha, l_val)
                    contraction.append([float(alpha), float(coeff) / nk])
                contractions.append(contraction)
            element_basis[sym] = contractions

    total_Z = sum(a["Z"] for a in ATOMS)
    spin = total_Z % 2

    mol = gto.M(atom=atom_list, basis=element_basis, unit="Bohr", spin=spin, verbose=0)
    mol.cart = False
    mol.build()

    S_ref = mol.intor("int1e_ovlp")
    T_ref = mol.intor("int1e_kin")
    V_ref = mol.intor("int1e_nuc")

    # --- Map PySCF AO order → TIDES basis order ---------------------------
    # PySCF real spherical harmonic m-order (determined empirically):
    #   l=0: [0]
    #   l=1: [+1, -1, 0]  →  [px, py, pz]
    PYSCF_M_ORDER = {
        0: [0],
        1: [1, -1, 0],
    }

    pyscf_alm = []
    for ish in range(mol.nbas):
        a = int(mol.bas_atom(ish))
        l = int(mol.bas_angular(ish))
        nctr = int(mol.bas_nctr(ish))
        for _ in range(nctr):
            for m in PYSCF_M_ORDER[l]:
                pyscf_alm.append((a, l, m))

    tides_alm = [(e["atom"], e["l"], e["m"]) for e in SYNTHETIC_BASIS]

    pyscf_idx_of = {}
    for j, alm in enumerate(pyscf_alm):
        pyscf_idx_of[alm] = j

    tides_to_pyscf = []
    for alm in tides_alm:
        assert alm in pyscf_idx_of, f"TIDES AO {alm} not found in PySCF"
        tides_to_pyscf.append(pyscf_idx_of[alm])

    n = len(tides_alm)
    S_tides = np.zeros((n, n))
    T_tides = np.zeros((n, n))
    V_tides = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi = tides_to_pyscf[i]
            pj = tides_to_pyscf[j]
            S_tides[i, j] = S_ref[pi, pj]
            T_tides[i, j] = T_ref[pi, pj]
            V_tides[i, j] = V_ref[pi, pj]

    # Write matrices
    np.savetxt(os.path.join(dump_dir, "S.txt"), S_tides.ravel(), fmt="%.17g")
    np.savetxt(os.path.join(dump_dir, "T.txt"), T_tides.ravel(), fmt="%.17g")
    np.savetxt(os.path.join(dump_dir, "V_ext.txt"), V_tides.ravel(), fmt="%.17g")

    # Write metadata
    with open(os.path.join(dump_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # Set LD_PRELOAD for MKL (for PySCF reference computation)
    mkl_libs = [
        "/usr/lib/x86_64-linux-gnu/libmkl_core.so",
        "/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so",
        "/usr/lib/x86_64-linux-gnu/libiomp5.so",
    ]
    existing = os.environ.get("LD_PRELOAD", "")
    parts = [p for p in existing.split(":") if p]
    for lib in mkl_libs:
        if lib not in parts and os.path.exists(lib):
            parts.append(lib)
    if parts:
        os.environ["LD_PRELOAD"] = ":".join(parts)

    tmpdir = tempfile.mkdtemp(prefix="tides_p06b_selftest_")
    try:
        print(f"[selftest] Generating synthetic dump in {tmpdir}")
        generate_synthetic_dump(tmpdir)

        # Run Becke oracle with higher grid density for the selftest
        print("[selftest] Running Becke oracle (n_radial=200, lebedev_order=41)")
        result = run_becke_oracle(
            tmpdir, n_radial=200, lebedev_order=41, compute_v=True
        )

        print()
        print(f"  n_basis           = {result['n_basis']}")
        print(f"  n_mesh_points     = {result['n_mesh_points']}")
        print(f"  Y_lm valid        = {result['ylm_convention_valid']}  "
              f"(dev = {result['ylm_max_onsite_dev']:.6e})")
        print(f"  max|dS|           = {result['max_dS']:.6e}  "
              f"(tol {1e-8:.6e})")
        print(f"  max|dT|           = {result['max_dT']:.6e}  "
              f"(tol {1e-6:.6e})")
        if result["max_dV"] is not None:
            print(f"  max|dV_ext|       = {result['max_dV']:.6e}  "
                  f"(tol {1e-6:.6e})")
        print()
        print("  Worst offenders:")
        print(f"    dS: {result['worst_S']}")
        print(f"    dT: {result['worst_T']}")
        if result["worst_V"]:
            print(f"    dV: {result['worst_V']}")
        print()

        # Assertions (no try/except control flow)
        assert result["max_dS"] <= 1e-8, (
            f"S mismatch: {result['max_dS']:.6e} > 1e-8"
        )
        assert result["max_dT"] <= 1e-6, (
            f"T mismatch: {result['max_dT']:.6e} > 1e-6"
        )
        if result["max_dV"] is not None:
            assert result["max_dV"] <= 1e-6, (
                f"V mismatch: {result['max_dV']:.6e} > 1e-6"
            )

        print("[selftest] PASSED — all matrices within tolerance")
        sys.exit(0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
