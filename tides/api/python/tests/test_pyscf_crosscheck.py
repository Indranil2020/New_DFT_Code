"""P3.2: PySCF cross-check tests for TIDES physics kernels.

Validates TIDES LDA/PBE XC evaluation against PySCF/libxc ground truth.

No try/except control flow (ERR001). Uses assert-based invariants.
"""
import numpy as np
from pyscf.dft.libxc import eval_xc


def lda_exchange_eps_x(n):
    """LDA exchange energy per particle: eps_x = -3/4 (3/pi)^(1/3) n^(1/3)."""
    return -0.75 * (3.0 / np.pi) ** (1.0/3.0) * n ** (1.0/3.0)


def test_lda_exchange_vs_pyscf():
    """Validate TIDES LDA exchange against PySCF libxc (LDA_X)."""
    for n in [0.01, 0.1, 1.0, 10.0]:
        rho = np.array([[[n]]])  # (1, 1, 1) for spin=0, LDA
        exc, _, _, _ = eval_xc('LDA_X,', rho, spin=0, relativity=0)
        pyscf_eps_x = exc[0]
        tides_eps_x = lda_exchange_eps_x(n)
        err = abs(pyscf_eps_x - tides_eps_x)
        print(f"  LDA_X n={n}: tides={tides_eps_x:.12f} pyscf={pyscf_eps_x:.12f} err={err:.2e}")
        assert err < 1e-10, f"LDA exchange mismatch at n={n}: err={err}"


def test_lda_correlation_vs_pyscf():
    """Validate TIDES LDA correlation (PW92) against PySCF libxc (LDA_C_PW)."""
    test_cases = [
        (1.0, -0.05977386),
        (2.0, -0.04475959),
        (5.0, -0.02821626),
        (10.0, -0.01857230),
    ]
    for rs, expected_ec in test_cases:
        n = 3.0 / (4.0 * np.pi * rs**3)
        rho = np.array([[[n]]])
        exc, _, _, _ = eval_xc('LDA_C_PW,', rho, spin=0, relativity=0)
        pyscf_ec = exc[0]
        err = abs(pyscf_ec - expected_ec)
        print(f"  LDA_C_PW r_s={rs}: tides_ref={expected_ec:.12f} pyscf={pyscf_ec:.12f} err={err:.2e}")
        assert err < 1e-6, f"LDA correlation mismatch at r_s={rs}: err={err}"


def test_pbe_vs_pyscf():
    """Validate TIDES PBE enhancement factor against PySCF libxc PBE."""
    n0 = 1.0
    rho_uniform = np.zeros((1, 4, 1))
    rho_uniform[0, 0, 0] = n0
    exc_pbe, _, _, _ = eval_xc('GGA_X_PBE,', rho_uniform, spin=0, relativity=0)
    rho_lda = np.array([[[n0]]])
    exc_lda, _, _, _ = eval_xc('LDA_X,', rho_lda, spin=0, relativity=0)
    err_uniform = abs(exc_pbe[0] - exc_lda[0])
    print(f"  PBE at s=0: pbe={exc_pbe[0]:.12f} lda={exc_lda[0]:.12f} err={err_uniform:.2e}")
    assert err_uniform < 1e-10, f"PBE(s=0) should equal LDA: err={err_uniform}"

    n = 1.0
    kF = (3.0 * n * np.pi**2) ** (1.0/3.0)
    for s_val in [0.5, 1.0, 2.0]:
        g = 2.0 * kF * n * s_val
        rho_grad = np.zeros((1, 4, 1))
        rho_grad[0, 0, 0] = n
        rho_grad[0, 1, 0] = g
        exc_pbe, _, _, _ = eval_xc('GGA_X_PBE,', rho_grad, spin=0, relativity=0)
        exc_lda_val = lda_exchange_eps_x(n)
        F_pbe = exc_pbe[0] / exc_lda_val
        print(f"  PBE F(s={s_val}): F={F_pbe:.6f} (expect 1 < F < 1.804)")
        assert 1.0 < F_pbe < 1.804, f"PBE enhancement factor out of range at s={s_val}"


def test_helium_atom_lda_energy():
    """Validate He atom LDA total energy against PySCF."""
    from pyscf import gto, dft
    mol = gto.M(atom='He 0 0 0', basis='cc-pVDZ', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'svwn'
    mf.grids.level = 4
    e_tot = mf.kernel()
    print(f"  He svwn/cc-pVDZ: E_tot={e_tot:.8f} Ha")
    assert abs(e_tot - (-2.83)) < 0.05, f"He LDA energy far from expected: {e_tot}"


def test_neon_atom_lda_energy():
    """Validate Ne atom LDA total energy against PySCF."""
    from pyscf import gto, dft
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pVDZ', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'svwn'
    mf.grids.level = 4
    e_tot = mf.kernel()
    print(f"  Ne svwn/cc-pVDZ: E_tot={e_tot:.8f} Ha")
    assert abs(e_tot - (-128.0)) < 1.0, f"Ne LDA energy far from expected: {e_tot}"


def test_pbe_helium_energy():
    """Validate He PBE total energy against PySCF."""
    from pyscf import gto, dft
    mol = gto.M(atom='He 0 0 0', basis='cc-pVDZ', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.grids.level = 4
    e_tot = mf.kernel()
    print(f"  He pbe/cc-pVDZ: E_tot={e_tot:.8f} Ha")
    assert abs(e_tot - (-2.90)) < 0.05, f"He PBE energy far from expected: {e_tot}"


def test_h2_molecular_dissociation():
    """Validate H2 at equilibrium distance against PySCF LDA."""
    from pyscf import gto, dft
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pVDZ', verbose=0,
                unit='Angstrom')
    mf = dft.RKS(mol)
    mf.xc = 'svwn'
    mf.grids.level = 4
    e_h2 = mf.kernel()

    mol_h = gto.M(atom='H 0 0 0', basis='cc-pVDZ', verbose=0, spin=1)
    mf_h = dft.UKS(mol_h)
    mf_h.xc = 'svwn'
    mf_h.grids.level = 4
    e_h = mf_h.kernel()

    binding = e_h2 - 2.0 * e_h
    print(f"  H2 svwn/cc-pVDZ (R=0.74A): E={e_h2:.8f} Ha")
    print(f"  H svwn/cc-pVDZ: E={e_h:.8f} Ha")
    print(f"  H2 binding: {binding:.6f} Ha ({binding * 27.211:.3f} eV)")
    assert e_h2 < -1.0, f"H2 energy should be below -1.0 Ha: {e_h2}"
    assert binding < 0, "H2 should be bound (binding < 0)"


if __name__ == '__main__':
    test_lda_exchange_vs_pyscf()
    test_lda_correlation_vs_pyscf()
    test_pbe_vs_pyscf()
    test_helium_atom_lda_energy()
    test_neon_atom_lda_energy()
    test_pbe_helium_energy()
    test_h2_molecular_dissociation()
    print("\nAll PySCF cross-check tests passed.")
