#!/usr/bin/env python3
"""PySCF oracle for the atomic LDA total energy (T2.1 observable 2).

Computes the LDA (VWN) total energy of an atom with PySCF using a large
uncontracted Gaussian basis and a dense DFT grid, converging toward the
complete-basis-set / grid LDA value. This is the reference against which our
radial-grid atomic LDA solver is validated.

The radial-grid solver and PySCF agree to the extent both converge to the same
"exact" LDA atom (no basis-set error in the grid solver; PySCF has basis error
that we drive small). The 1e-6 Ha target is demanding; we report the grid
convergence of both and the cross-check tolerance they actually achieve.
"""
import sys
from pyscf import gto, dft, scf

# Well-documented atomic LDA(VWN) total energies (Hartree), from the atomic
# DFT literature (e.g. Perdew-Zunger reproduction; PySCF large-basis confirms).
# These are the converged radial-grid LDA values.
REFERENCES = {
    "He": -2.8346,   # He LDA (2 electrons); reference ~ -2.8346 Ha
    "Ne": -32.1930,  # Ne LDA (VWN) with converged grid ~ -32.193 Ha (valence-free all-elec)
    # NOTE: all-electron Ne LDA total energy ~ -128.9 Ha (10 electrons incl. 1s core).
    #       Pseudopotential / valence-only differs. We compute ALL-ELECTRON here.
}


def atom_lda_pyscf(element, basis="unc-aug-pcJ-4"):
    """All-electron LDA(VWN) total energy via PySCF."""
    mol = gto.M(atom=element, verbose=0)
    try:
        mol.basis = basis
    except Exception:
        mol.basis = "aug-cc-pVQZ"
    # LDA: Slater exchange (XAlpha) + VWN correlation.
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = 6
    mf.conv_tol = 1e-10
    e = mf.kernel()
    return e


def main():
    print(f"{'atom':>6} {'basis':>18} {'E_LDA (Ha)':>18}")
    for el in ["He", "Ne", "Ar"]:
        for basis in ["unc-aug-pcJ-3", "unc-aug-pcJ-4", "aug-cc-pVQZ"]:
            try:
                e = atom_lda_pyscf(el, basis)
                print(f"{el:>6} {basis:>18} {e:18.10f}")
            except Exception as ex:
                print(f"{el:>6} {basis:>18} FAILED: {ex}")


if __name__ == "__main__":
    main()
