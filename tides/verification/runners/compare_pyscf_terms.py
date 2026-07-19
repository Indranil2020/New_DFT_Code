#!/usr/bin/env python3
"""Compare TIDES S, T, V_ext matrices against PySCF.

Given a dump directory produced by TIDES (env ``TIDES_DUMP_HMAT_DIR=<dir>``),
this script:

1.  Fits each TIDES radial function  R(r)  to even-tempered Gaussians
    (see :mod:`fit_nao_gaussians`).
2.  Builds the same molecule in PySCF (``gto.M``, unit Bohr, custom basis
    from the fits, spherical harmonics).
3.  Computes  S / T / V_nuc  via  ``mol.intor('int1e_ovlp' | 'int1e_kin' | 'int1e_nuc')``.
4.  **Automatically** determines the TIDES→PySCF basis-ordering permutation
    and per-function sign mapping by matching the S matrix (validated when
    S agrees).
5.  Reports ``max|dS|``, ``max|dT|``, ``max|dV_ext|`` and per-element worst
    offenders.
6.  Exits 0 iff all matrices are within tolerance.

Tolerance logic::

    tol_S = tol_T = max(1e-5, 10 × max_fit_residual)
    tol_V         = max(1e-4, 10 × max_fit_residual)

(see ``tides/verification/tolerances.yaml``  →  ``p0_6``)

No try/except — all error conditions use assertions and explicit checks.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
from pyscf import gto

# Reuse the fitting module (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_nao_gaussians import (  # noqa: E402
    DEFAULT_N_GAUSS,
    coeffs_for_pyscf,
    fit_all_radials,
)

# ---------------------------------------------------------------------------
# Element symbol lookup  (Z = 1 … 36 covers all common DFT test systems)
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS: List[str] = [
    "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na","Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc","Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga","Ge", "As", "Se", "Br", "Kr",
]


def z_to_symbol(Z: int) -> str:
    assert 1 <= Z < len(ELEMENT_SYMBOLS), f"Unsupported atomic number Z={Z}"
    return ELEMENT_SYMBOLS[Z]


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------

def load_dump(dump_dir: str) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load a TIDES dump directory.

    Returns ``(meta, S, T, V_ext)`` where each matrix is *n_basis* × *n_basis*.
    """
    meta_path = os.path.join(dump_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)

    n = int(meta["n_basis"])

    def _load_matrix(name: str) -> np.ndarray:
        data = np.loadtxt(os.path.join(dump_dir, name))
        assert data.size == n * n, (
            f"{name}: expected {n}×{n} = {n*n} elements, got {data.size}"
        )
        return data.reshape(n, n)

    S = _load_matrix("S.txt")
    T = _load_matrix("T.txt")
    V = _load_matrix("V_ext.txt")
    return meta, S, T, V


# ---------------------------------------------------------------------------
# PySCF molecule construction
# ---------------------------------------------------------------------------

def build_pyscf_mol(meta: dict, dump_dir: str, radial_fits: dict) -> gto.Mole:
    """Build a PySCF :class:`~pyscf.gto.Mole` from the fitted Gaussian basis."""
    atoms = meta["atoms"]

    # --- atom list (symbol + position) ---------------------------------
    atom_list: List[List] = []
    for atom in atoms:
        sym = z_to_symbol(int(atom["Z"]))
        pos = [float(x) for x in atom["pos_bohr"]]
        atom_list.append([sym, pos])

    # --- basis dict (element → contractions) ---------------------------
    # PySCF assigns the same basis to all atoms of the same element.
    # NAO basis is per-element, so same-element atoms share radial functions.
    # We only need one set of radial functions per element (from the first
    # atom of that element); duplicates from other atoms of the same element
    # are skipped.
    element_basis: Dict[str, list] = {}
    atom_to_symbol: Dict[int, str] = {}
    element_first_atom: Dict[str, int] = {}
    for atom_idx, atom in enumerate(atoms):
        sym = z_to_symbol(int(atom["Z"]))
        atom_to_symbol[atom_idx] = sym
        if sym not in element_first_atom:
            element_first_atom[sym] = atom_idx
            element_basis[sym] = []

    for (atom_idx, fn_idx), (l, alphas, coeffs, _res) in radial_fits.items():
        sym = atom_to_symbol[atom_idx]
        # Only add contractions from the first atom of each element
        if atom_idx != element_first_atom[sym]:
            continue
        contraction = [l]  # PySCF format: [l, [exp, coeff], [exp, coeff], ...]
        pairs = coeffs_for_pyscf(alphas, coeffs, l)
        contraction.extend(pairs)
        element_basis[sym].append(contraction)

    # Sanity: every element used by at least one atom has a basis
    for atom_idx in range(len(atoms)):
        sym = atom_to_symbol[atom_idx]
        assert len(element_basis[sym]) > 0, f"Element {sym} (atom {atom_idx}) has no basis"

    # --- spin (only needed so PySCF's build() doesn't complain) ---------
    total_Z = sum(int(a["Z"]) for a in atoms)
    spin = total_Z % 2

    mol = gto.M(atom=atom_list, basis=element_basis, unit="Bohr",
                spin=spin, verbose=0)
    mol.cart = False
    mol.build()
    return mol


def get_pyscf_ao_info(mol: gto.Mole) -> List[Tuple[int, int]]:
    """Return a list of ``(atom_index, l)`` for every PySCF AO (spherical)."""
    info: List[Tuple[int, int]] = []
    for ish in range(mol.nbas):
        atom = int(mol.bas_atom(ish))
        l = int(mol.bas_angular(ish))
        nctr = int(mol.bas_nctr(ish))
        for _ in range(nctr * (2 * l + 1)):
            info.append((atom, l))
    return info


# ---------------------------------------------------------------------------
# Permutation + sign determination
# ---------------------------------------------------------------------------

def _signs_from_perm(
    S_tides: np.ndarray,
    S_pyscf: np.ndarray,
    perm: np.ndarray,
    n: int,
) -> np.ndarray:
    """Determine per-AO sign factors  s_i = ±1  given the permutation.

    Given the permutation,  S_tides[i,j] = s_i · s_j · S_pyscf[perm[i], perm[j]].
    We fix s₀ = +1 and propagate via BFS through the overlap graph.
    """
    signs = np.zeros(n, dtype=float)
    signs[0] = 1.0
    threshold = 1e-12

    queue = deque([0])
    visited = {0}
    while queue:
        i = queue.popleft()
        for j in range(n):
            if j in visited:
                continue
            s_ij_pyscf = S_pyscf[perm[i], perm[j]]
            if abs(S_tides[i, j]) > threshold and abs(s_ij_pyscf) > threshold:
                ratio = S_tides[i, j] / s_ij_pyscf
                signs[j] = signs[i] * (1.0 if ratio >= 0 else -1.0)
                visited.add(j)
                queue.append(j)

    # Unconnected AOs default to +1 (sign doesn't affect near-zero overlaps)
    for i in range(n):
        if signs[i] == 0.0:
            signs[i] = 1.0
    return signs


def _full_mismatch(
    S_tides: np.ndarray,
    S_pyscf: np.ndarray,
    perm: np.ndarray,
    n: int,
) -> float:
    """Compute the sign-invariant S-matrix mismatch for a full permutation.

    Uses absolute values so the result is independent of per-AO signs.
    """
    idx = np.ix_(perm, perm)
    diff = np.abs(S_tides) - np.abs(S_pyscf[idx])
    return float(np.sum(diff * diff))


def _match_group_bruteforce(
    S_tides: np.ndarray,
    S_pyscf: np.ndarray,
    tides_indices: List[int],
    pyscf_indices: List[int],
    perm_so_far: np.ndarray,
    n: int,
) -> List[int]:
    """Match TIDES AOs to PySCF AOs within a single ``(atom, l)`` group.

    Since groups are small (2l+1 ≤ 7 for l ≤ 3), we brute-force all
    permutations within the group and pick the one that minimises the
    **full** S-matrix mismatch (sign-invariant via absolute values).
    The full-matrix comparison is necessary because within-group sub-matrices
    are often identical up to permutation (e.g. diagonal identity blocks),
    so they cannot distinguish m-orderings on their own.

    ``perm_so_far`` contains the partial permutation for already-matched
    groups; the entries for this group are filled in for each candidate.

    Returns ``local_perm`` where  ``local_perm[a] = b``  means
    TIDES AO ``tides_indices[a]``  ↔  PySCF AO ``pyscf_indices[b]``.
    """
    ng = len(tides_indices)
    assert ng <= 7, f"Group too large for brute-force: {ng} AOs (l > 3?)"

    best_perm = list(range(ng))
    best_mismatch = float("inf")

    for perm_tuple in permutations(range(ng)):
        # Fill in this group's permutation entries
        perm = perm_so_far.copy()
        for a, b in enumerate(perm_tuple):
            perm[tides_indices[a]] = pyscf_indices[b]
        mismatch = _full_mismatch(S_tides, S_pyscf, perm, n)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_perm = list(perm_tuple)

    return best_perm


def find_permutation_and_signs(
    meta: dict,
    S_tides: np.ndarray,
    S_pyscf: np.ndarray,
    pyscf_ao_info: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine TIDES→PySCF permutation and per-function signs.

    Groups AOs by ``(atom, l)``.  Within each group, brute-forces all
    permutations (groups are ≤ 7 elements) and picks the one minimising the
    full S-matrix mismatch (sign-invariant).  Then propagates per-AO signs
    via BFS through the overlap graph.

    Returns ``(perm, signs)`` where:
    - ``perm[i]`` = PySCF AO index corresponding to TIDES AO *i*
    - ``signs[i]`` = ±1 sign factor for TIDES AO *i*
    """
    n = len(meta["basis"])
    assert len(pyscf_ao_info) == n, (
        f"AO count mismatch: TIDES basis has {n} entries, "
        f"PySCF has {len(pyscf_ao_info)} AOs"
    )

    # Group TIDES AOs by (atom, l)
    tides_groups: Dict[Tuple[int, int], List[int]] = {}
    for i, entry in enumerate(meta["basis"]):
        key = (int(entry["atom"]), int(entry["l"]))
        tides_groups.setdefault(key, []).append(i)

    # Group PySCF AOs by (atom, l)
    pyscf_groups: Dict[Tuple[int, int], List[int]] = {}
    for j, key in enumerate(pyscf_ao_info):
        pyscf_groups.setdefault(key, []).append(j)

    # Process groups in order of increasing size (small groups first →
    # fewer permutations to evaluate for the larger groups later, since
    # the partial permutation constrains the search).
    sorted_groups = sorted(tides_groups.items(), key=lambda kv: len(kv[1]))

    perm = np.zeros(n, dtype=int)
    matched_tides = set()
    matched_pyscf = set()

    for key, ti_list in sorted_groups:
        assert key in pyscf_groups, f"No PySCF AOs for group (atom={key[0]}, l={key[1]})"
        pj_list = pyscf_groups[key]
        assert len(ti_list) == len(pj_list), (
            f"Group {key}: TIDES has {len(ti_list)} AOs, PySCF has {len(pj_list)}"
        )
        local_perm = _match_group_bruteforce(
            S_tides, S_pyscf, ti_list, pj_list, perm, n
        )
        for a, b in enumerate(local_perm):
            perm[ti_list[a]] = pj_list[b]
            matched_tides.add(ti_list[a])
            matched_pyscf.add(pj_list[b])

    signs = _signs_from_perm(S_tides, S_pyscf, perm, n)
    return perm, signs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _worst_elements(dM: np.ndarray, n_worst: int = 3) -> List[Tuple[int, int, float]]:
    """Return the *n_worst* largest-magnitude elements of *dM* as ``(i, j, val)``."""
    flat = np.argsort(np.abs(dM).ravel())[::-1][:n_worst]
    out: List[Tuple[int, int, float]] = []
    for idx in flat:
        i, j = np.unravel_index(idx, dM.shape)
        out.append((int(i), int(j), float(dM[i, j])))
    return out


def compare_dump(
    dump_dir: str,
    n_gauss: int = DEFAULT_N_GAUSS,
) -> dict:
    """Run the full comparison pipeline on *dump_dir*.

    Returns a details dict with keys:
    ``max_dS, max_dT, max_dV, max_fit_residual, tol_s, tol_t, tol_v,
    passed, n_basis, worst_S, worst_T, worst_V, per_function_residuals``.
    """
    meta, S_tides, T_tides, V_tides = load_dump(dump_dir)

    radial_fits, max_residual = fit_all_radials(meta, dump_dir, n_gauss=n_gauss)

    mol = build_pyscf_mol(meta, dump_dir, radial_fits)
    pyscf_ao_info = get_pyscf_ao_info(mol)

    S_pyscf = mol.intor("int1e_ovlp")
    T_pyscf = mol.intor("int1e_kin")
    V_pyscf = mol.intor("int1e_nuc")

    perm, signs = find_permutation_and_signs(meta, S_tides, S_pyscf, pyscf_ao_info)

    # Apply permutation and signs to PySCF matrices:
    # matched[i,j] = signs[i] · signs[j] · pyscf[perm[i], perm[j]]
    s_outer = signs[:, None] * signs[None, :]
    idx = np.ix_(perm, perm)
    S_matched = s_outer * S_pyscf[idx]
    T_matched = s_outer * T_pyscf[idx]
    V_matched = s_outer * V_pyscf[idx]

    dS = S_tides - S_matched
    dT = T_tides - T_matched
    dV = V_tides - V_matched

    max_dS = float(np.max(np.abs(dS)))
    max_dT = float(np.max(np.abs(dT)))
    max_dV = float(np.max(np.abs(dV)))

    tol_s = max(1e-5, 10.0 * max_residual)
    tol_t = max(1e-5, 10.0 * max_residual)
    tol_v = max(1e-4, 10.0 * max_residual)

    passed = max_dS <= tol_s and max_dT <= tol_t and max_dV <= tol_v

    # Per-function fit residuals
    per_fn = {}
    for (atom, fn), (l, _, _, res) in radial_fits.items():
        per_fn[f"radial_{atom}_{fn}"] = {"l": l, "residual": res}

    return {
        "max_dS": max_dS,
        "max_dT": max_dT,
        "max_dV": max_dV,
        "max_fit_residual": max_residual,
        "tol_s": tol_s,
        "tol_t": tol_t,
        "tol_v": tol_v,
        "passed": passed,
        "n_basis": int(meta["n_basis"]),
        "worst_S": _worst_elements(dS),
        "worst_T": _worst_elements(dT),
        "worst_V": _worst_elements(dV),
        "per_function_residuals": per_fn,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: compare_pyscf_terms.py <dump_dir> [n_gauss]")
        sys.exit(1)
    dump_dir = sys.argv[1]
    n_gauss = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_GAUSS

    d = compare_dump(dump_dir, n_gauss=n_gauss)

    print(f"TIDES vs PySCF comparison  (n_basis = {d['n_basis']})")
    print(f"  max|dS|      = {d['max_dS']:.6e}   (tol {d['tol_s']:.6e})")
    print(f"  max|dT|      = {d['max_dT']:.6e}   (tol {d['tol_t']:.6e})")
    print(f"  max|dV_ext|  = {d['max_dV']:.6e}   (tol {d['tol_v']:.6e})")
    print(f"  max fit residual = {d['max_fit_residual']:.6e}")
    print()
    print("  Per-function fit residuals:")
    for name, info in d["per_function_residuals"].items():
        print(f"    {name}  (l={info['l']}):  {info['residual']:.6e}")
    print()
    print("  Worst offenders (i, j, value):")
    print(f"    dS: {d['worst_S']}")
    print(f"    dT: {d['worst_T']}")
    print(f"    dV: {d['worst_V']}")
    print()
    print(f"  {'PASSED' if d['passed'] else 'FAILED'}")
    sys.exit(0 if d["passed"] else 1)


if __name__ == "__main__":
    main()
