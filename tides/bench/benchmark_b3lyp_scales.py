#!/usr/bin/env python3
"""Benchmark TIDES B3LYP at 10/50/100-atom scales with and without screening.

Generates molecules via RDKit, runs TIDES SCF, and records timings/energies.
Results are printed in a table format suitable for the audit ledger.
"""
import sys, os, time, json
sys.path.insert(0, '/home/indranil/git/New_DFT_Code/tides/api/python')
import tides._native as native
from rdkit import Chem
from rdkit.Chem import AllChem

BOHR = 1.8897261254535

def make_molecule(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    Z, pos = [], []
    for atom in mol.GetAtoms():
        Z.append(atom.GetAtomicNum())
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.extend([p.x * BOHR, p.y * BOHR, p.z * BOHR])
    return Z, pos

def run_tides(Z, pos, xc='b3lyp', grid_h=0.5, max_iter=80, tol=5e-6,
              level_shift=0.0, mixing=0, alpha=0.3, screen=True):
    env = dict(os.environ)
    env['TIDES_LEVEL_SHIFT'] = str(level_shift)
    env['TIDES_SCF_MIXING'] = str(mixing)
    env['TIDES_SCF_ALPHA'] = str(alpha)
    if screen:
        env['TIDES_VMAT_GEMM'] = '1'
        env['TIDES_RHO_GEMM'] = '1'
        env['TIDES_GRID_SCREEN'] = '1'
    else:
        env['TIDES_VMAT_GEMM'] = '0'
        env['TIDES_RHO_GEMM'] = '0'
        env['TIDES_GRID_SCREEN'] = '0'
    # Apply env
    for k, v in env.items():
        os.environ[k] = v

    t0 = time.time()
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=grid_h, grid_margin=6.0,
        max_iter=max_iter, tol=tol,
        use_dual_grid=False, xc_functional=xc,
        allow_grid_refine=False
    )
    wall = time.time() - t0
    return r, wall

# Benchmark molecules
# ~10 atoms: CH4 (5), C2H6 (8), H2O (3), NH3 (4)
# ~14 atoms: butane C4H10
# ~50 atoms: decane C10H22 (32), or larger
# ~100 atoms: eicosane C20H42 (62), or icosane

molecules = {
    'CH4 (5 atoms)':       ('C', 5),
    'C2H6 (8 atoms)':      ('CC', 8),
    'C4H10 butane (14)':   ('CCCC', 14),
    'C6H6 benzene (12)':   ('c1ccccc1', 12),
    'C8H18 octane (26)':   ('CCCCCCCC', 26),
    'C10H22 decane (32)':  ('CCCCCCCCCC', 32),
    'C14H30 (44)':         ('C'*14, 44),
    'C20H42 (62)':         ('C'*20, 62),
}

results = []
for label, (smiles, natoms) in molecules.items():
    print(f"\n{'='*60}")
    print(f"  {label}  SMILES={smiles}")
    print(f"{'='*60}")
    Z, pos = make_molecule(smiles)
    actual_n = len(Z)
    n_basis = actual_n * 8  # DZ ~8 fns/atom

    if actual_n > 20:
        max_iter = 80
        level_shift = 0.3
    else:
        max_iter = 80
        level_shift = 0.0

    for screen in [True, False]:
        screen_label = "screen" if screen else "noscreen"
        print(f"\n--- {label} B3LYP {screen_label} ---")
        r, wall = run_tides(Z, pos, xc='b3lyp', grid_h=0.5,
                           max_iter=max_iter, tol=5e-6,
                           level_shift=level_shift, mixing=0, alpha=0.3,
                           screen=screen)
        bh = r.build_H_timings
        scf = r.scf
        en = r.energy
        entry = {
            'label': label, 'smiles': smiles, 'natoms': actual_n,
            'n_basis': n_basis, 'screen': screen_label,
            'E_total': en.E_total, 'converged': scf.converged,
            'iters': bh.n_iterations,
            'wall_s': round(wall, 2),
            'build_H_ms': round(bh.total_ms, 2),
            'xc_eval_ms': round(bh.xc_eval_ms, 2),
            'poisson_ms': round(bh.poisson_ms, 2),
            'rho_build_ms': round(bh.rho_build_ms, 2),
        }
        results.append(entry)
        print(f"  E={en.E_total:.6f}  converged={scf.converged}  iters={bh.n_iterations}")
        print(f"  wall={wall:.1f}s  build_H={bh.total_ms:.1f}ms  xc={bh.xc_eval_ms:.1f}ms")
        print(f"  poisson={bh.poisson_ms:.1f}ms  rho={bh.rho_build_ms:.1f}ms")
        if not screen:
            break
        if actual_n > 20:
            break  # skip no-screen for large molecules

print(f"\n{'='*80}")
print("BENCHMARK SUMMARY")
print(f"{'='*80}")
print(f"{'Label':<25} {'N':>4} {'Screen':>8} {'E_total':>14} {'Conv':>5} {'Iters':>5} {'Wall(s)':>8} {'build_H(ms)':>12}")
print("-"*80)
for e in results:
    print(f"{e['label']:<25} {e['natoms']:>4} {e['screen']:>8} {e['E_total']:>14.4f} {str(e['converged']):>5} {e['iters']:>5} {e['wall_s']:>8.1f} {e['build_H_ms']:>12.1f}")

# Save as JSON
with open('/home/indranil/git/New_DFT_Code/tides/bench/profiling_results/b3lyp_scale_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to bench/profiling_results/b3lyp_scale_benchmark.json")
