#!/usr/bin/env python3
"""Self-test for the P0.6 PySCF comparison harness.

Generates a **synthetic** dump directory (2 atoms, s and p functions built
from *known* Gaussians sampled on a radial grid) where the exact S/T/V are
computable by PySCF directly, then runs the full comparison pipeline and
asserts it passes.

This is the acceptance test for task ``p06a-pyscf-harness``.  It requires
**no TIDES C++** — everything is Python + PySCF.

Run::

    python3 tides/verification/runners/selftest_synthetic.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import List, Tuple

import numpy as np
from pyscf import gto

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pyscf_terms import compare_dump, z_to_symbol  # noqa: E402
from fit_nao_gaussians import (  # noqa: E402
    coeffs_for_pyscf,
    primitive_normalization,
)

# ---------------------------------------------------------------------------
# Synthetic-basis definition
# ---------------------------------------------------------------------------

# Two H-like atoms with s and p functions built from known Gaussian primitives.
# (atom_index, l, [(exp, coeff), ...])
SYNTHETIC_BASIS = [
    # atom 0 — s function (2 primitives)
    {"atom": 0, "l": 0, "m": 0,  "fn": 0, "primitives": [(0.5, 1.0), (2.0, 0.4)]},
    # atom 0 — p function (1 primitive)  [different fn index!]
    {"atom": 0, "l": 1, "m": 0,  "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 0, "l": 1, "m": 1,  "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 0, "l": 1, "m": -1, "fn": 1, "primitives": [(0.8, 1.0)]},
    # atom 1 — same element (H) → same radial functions as atom 0
    {"atom": 1, "l": 0, "m": 0,  "fn": 0, "primitives": [(0.5, 1.0), (2.0, 0.4)]},
    {"atom": 1, "l": 1, "m": 0,  "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 1, "l": 1, "m": 1,  "fn": 1, "primitives": [(0.8, 1.0)]},
    {"atom": 1, "l": 1, "m": -1, "fn": 1, "primitives": [(0.8, 1.0)]},
]

# Atom positions (Bohr)
ATOMS = [
    {"Z": 1, "pos_bohr": [0.0, 0.0, 0.0]},
    {"Z": 1, "pos_bohr": [2.5, 0.0, 0.0]},
]

# Radial grid for sampling the Gaussian functions
R_MAX = 30.0
N_R = 800


# ---------------------------------------------------------------------------
# Radial function evaluation
# ---------------------------------------------------------------------------

def radial_value(r: np.ndarray, l: int, primitives: List[Tuple[float, float]]) -> np.ndarray:
    """Evaluate the radial part  R(r) = Σ_k c_k · r^l · exp(-α_k·r²)
    using **unnormalised** Gaussians.
    """
    R = np.zeros_like(r)
    for alpha, coeff in primitives:
        R += coeff * r ** l * np.exp(-alpha * r ** 2)
    return R


# ---------------------------------------------------------------------------
# Synthetic dump generation
# ---------------------------------------------------------------------------

def generate_synthetic_dump(dump_dir: str) -> dict:
    """Create a synthetic dump directory and return its metadata.

    The dump is built from *known* Gaussian primitives so that the exact S/T/V
    are computable by PySCF directly.  The comparison harness must recover
    these matrices to fit precision.
    """
    os.makedirs(dump_dir, exist_ok=True)

    # --- Build metadata ----------------------------------------------------
    basis_entries = []
    for i, entry in enumerate(SYNTHETIC_BASIS):
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
        np.savetxt(
            os.path.join(dump_dir, f"radial_{entry['atom']}_{entry['fn']}.txt"),
            np.column_stack([r_grid, R]),
            fmt="%.17g",
        )
        written.add(key)

    # --- Build PySCF molecule to get reference matrices --------------------
    # Use dict basis (element → contractions).  Both atoms are H, so they
    # share the same basis — this mirrors the real NAO case where same-element
    # atoms share radial functions.
    atom_list = [[z_to_symbol(a["Z"]), a["pos_bohr"]] for a in ATOMS]

    # Collect per-(atom, l) primitives (each (atom, l) has one fn here)
    from collections import defaultdict
    atom_l_prims: dict = {}
    for entry in SYNTHETIC_BASIS:
        key = (entry["atom"], entry["l"])
        if key not in atom_l_prims:
            atom_l_prims[key] = []
        for p in entry["primitives"]:
            atom_l_prims[key].append(p)

    # Build element-keyed basis dict (both atoms are H → same basis)
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
    #   l=2: [0, +1, -1, +2, -2]  (not needed for this synthetic test)
    PYSCF_M_ORDER = {
        0: [0],
        1: [1, -1, 0],   # px, py, pz
    }

    # Build PySCF (atom, l, m) list
    pyscf_alm = []
    for ish in range(mol.nbas):
        a = int(mol.bas_atom(ish))
        l = int(mol.bas_angular(ish))
        nctr = int(mol.bas_nctr(ish))
        for _ in range(nctr):
            for m in PYSCF_M_ORDER[l]:
                pyscf_alm.append((a, l, m))

    # Build TIDES basis order: [(atom, l, m)] from SYNTHETIC_BASIS
    tides_alm = [(e["atom"], e["l"], e["m"]) for e in SYNTHETIC_BASIS]

    # Map: for each TIDES AO, find the PySCF AO with matching (atom, l, m)
    pyscf_idx_of = {}
    for j, alm in enumerate(pyscf_alm):
        pyscf_idx_of[alm] = j

    tides_to_pyscf = []
    for alm in tides_alm:
        assert alm in pyscf_idx_of, f"TIDES AO {alm} not found in PySCF"
        tides_to_pyscf.append(pyscf_idx_of[alm])

    # Build TIDES-ordered matrices (reorder PySCF matrices into TIDES order)
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
    import json
    with open(os.path.join(dump_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Set LD_PRELOAD for MKL (documented requirement for PySCF on this system)
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

    tmpdir = tempfile.mkdtemp(prefix="tides_p06a_selftest_")
    try:
        print(f"[selftest] Generating synthetic dump in {tmpdir}")
        generate_synthetic_dump(tmpdir)

        print("[selftest] Running comparison pipeline")
        result = compare_dump(tmpdir, n_gauss=14)

        print()
        print(f"  n_basis           = {result['n_basis']}")
        print(f"  max fit residual  = {result['max_fit_residual']:.6e}")
        print(f"  max|dS|           = {result['max_dS']:.6e}  (tol {result['tol_s']:.6e})")
        print(f"  max|dT|           = {result['max_dT']:.6e}  (tol {result['tol_t']:.6e})")
        print(f"  max|dV_ext|       = {result['max_dV']:.6e}  (tol {result['tol_v']:.6e})")
        print()
        print("  Per-function fit residuals:")
        for name, info in result["per_function_residuals"].items():
            print(f"    {name}  (l={info['l']}):  {info['residual']:.6e}")
        print()

        # Assertions (no try/except control flow)
        assert result["max_dS"] <= result["tol_s"], (
            f"S mismatch: {result['max_dS']:.6e} > {result['tol_s']:.6e}"
        )
        assert result["max_dT"] <= result["tol_t"], (
            f"T mismatch: {result['max_dT']:.6e} > {result['tol_t']:.6e}"
        )
        assert result["max_dV"] <= result["tol_v"], (
            f"V mismatch: {result['max_dV']:.6e} > {result['tol_v']:.6e}"
        )
        assert result["passed"], "Pipeline reported FAIL"

        print("[selftest] PASSED — all matrices within tolerance")
        sys.exit(0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
