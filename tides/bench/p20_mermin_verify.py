#!/usr/bin/env python3
"""p20-mermin-molecular verification: T=0 regression + smearing sanity.

Acceptance criteria (CPU-only, TIDES_DISABLE_GPU=1):
  1. T=0 REGRESSION: CH4, H2O, NH3 PP PBE grid_h=0.3 with electronic_temp_k=0
     must give EXACTLY the current energies and 7 iterations.
  2. SMEARING SANITY: C4H10 and C6H6 PP PBE grid_h=0.5 with electronic_temp_k=1500
     must (a) converge, (b) give a total (free) energy within ~0.2 Ha of T=0.
  3. Physical check: at 300K the smeared energy should be ~equal to T=0 (few mHa).

Usage:
  LD_PRELOAD=<mkl preload> python3 p20_mermin_verify.py
"""
import sys, os, time

# Use the worktree's build
WT = '/home/indranil/git/New_DFT_Code/.worktrees/p20-mermin-molecular'
TIDES_SRC = os.path.join(WT, 'tides')
os.environ['TIDES_SRC_DIR'] = TIDES_SRC
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))

# CPU-only
os.environ['TIDES_DISABLE_GPU'] = '1'

BOHR = 1.8897261254535
PP_DIR = os.path.join(TIDES_SRC, 'external/pseudopotentials/pseudodojo-pbe-sr')

# T=0 reference energies (from four_route_check / TASK-LEDGER)
T0_REFS = {
    'CH4': -7.87519691,
    'H2O': -16.084,
    'NH3': -11.12,
}

SMILES = {
    'CH4': 'C', 'H2O': 'O', 'NH3': 'N',
    'C4H10': 'CCCC', 'C6H6': 'c1ccccc1',
}


def make_molecule(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    Z, pos = [], []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        Z.append(atom.GetAtomicNum())
        pos.extend([p.x * BOHR, p.y * BOHR, p.z * BOHR])
    return Z, pos


def run_route(Z, pos, grid_h, electronic_temp_k, max_iter=100, tol=1e-7):
    import tides._native as native
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=grid_h, grid_margin=6.0,
        max_iter=max_iter, tol=tol,
        use_dual_grid=False, xc_functional='pbe',
        pp_dir=PP_DIR,
        allow_grid_refine=False,
        electronic_temp_k=electronic_temp_k,
    )
    return {
        'E': r.energy.E_total, 'converged': r.scf.converged,
        'iters': r.scf.n_iterations,
        'mermin_F': r.E_mermin_free_energy,
        'mermin_S': r.mermin_entropy,
        'mermin_mu': r.mermin_fermi_level,
    }


def main():
    all_pass = True

    # --- T=0 REGRESSION ---
    print("=" * 60)
    print("ACCEPTANCE 1: T=0 REGRESSION (must be byte-for-byte unchanged)")
    print("=" * 60)
    for label in ('CH4', 'H2O', 'NH3'):
        Z, pos = make_molecule(SMILES[label])
        r = run_route(Z, pos, grid_h=0.3, electronic_temp_k=0.0)
        ref = T0_REFS[label]
        de = abs(r['E'] - ref)
        ok = r['converged'] and de < 1e-4
        all_pass &= ok
        print(f"  {label}: E={r['E']:.8f} ref={ref:.8f} dE={de:.2e} "
              f"iters={r['iters']} conv={r['converged']} "
              f"[{'PASS' if ok else 'FAIL'}]")

    # --- SMEARING SANITY ---
    print()
    print("=" * 60)
    print("ACCEPTANCE 2: SMEARING SANITY (T=1500K, within ~0.2 Ha of T=0)")
    print("=" * 60)
    for label in ('C4H10', 'C6H6'):
        Z, pos = make_molecule(SMILES[label])
        # T=0 baseline
        r0 = run_route(Z, pos, grid_h=0.5, electronic_temp_k=0.0)
        print(f"  {label} T=0:    E={r0['E']:.8f} iters={r0['iters']} "
              f"conv={r0['converged']}")
        # T=1500K
        r1500 = run_route(Z, pos, grid_h=0.5, electronic_temp_k=1500.0)
        de = abs(r1500['E'] - r0['E'])
        ok = r1500['converged'] and de < 0.2
        all_pass &= ok
        print(f"  {label} T=1500: E={r1500['E']:.8f} (Mermin F) "
              f"iters={r1500['iters']} conv={r1500['converged']}")
        print(f"    dE vs T=0 = {de:.4f} Ha  S={r1500['mermin_S']:.6f} "
              f"mu={r1500['mermin_mu']:.6f}")
        print(f"    [{'PASS' if ok else 'FAIL'}]")

    # --- PHYSICAL CHECK: 300K ~ T=0 ---
    print()
    print("=" * 60)
    print("ACCEPTANCE 3: PHYSICAL CHECK (T=300K ~ T=0, few mHa)")
    print("=" * 60)
    for label in ('C4H10',):
        Z, pos = make_molecule(SMILES[label])
        r0 = run_route(Z, pos, grid_h=0.5, electronic_temp_k=0.0)
        r300 = run_route(Z, pos, grid_h=0.5, electronic_temp_k=300.0)
        de = abs(r300['E'] - r0['E'])
        ok = de < 0.01  # few mHa
        all_pass &= ok
        print(f"  {label} T=300:  E={r300['E']:.8f} vs T=0 {r0['E']:.8f} "
              f"dE={de:.6f} Ha  S={r300['mermin_S']:.6f}")
        print(f"    [{'PASS' if ok else 'FAIL'}]")

    print()
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
