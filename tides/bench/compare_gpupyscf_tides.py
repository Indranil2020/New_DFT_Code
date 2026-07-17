#!/usr/bin/env python3
"""GPU-PySCF vs TIDES comparison for molecules up to 14 atoms.

Runs B3LYP single-point calculations in 4 configurations:
  1. GPU-PySCF all-electron (def2-svp)
  2. GPU-PySCF pseudopotential (gth-dzvp + gth-pbe PP)
  3. TIDES all-electron (NAO-DZP, no PP)
  4. TIDES pseudopotential (NAO-DZP + PseudoDojo PP)

Same RDKit geometries for all. Energies won't match exactly across
different basis sets, but PP-vs-PP and AE-vs-AE should be closer.
Primary goal: compare wall-clock performance at same level of theory.
"""
import sys, os, time, json

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
os.environ['TIDES_SRC_DIR'] = TIDES_SRC
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))

BOHR = 1.8897261254535

def make_molecule(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms = []
    pos_ang = []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
        p = conf.GetAtomPosition(atom.GetIdx())
        pos_ang.append([p.x, p.y, p.z])
    pos_bohr = [[c * BOHR for c in r] for r in pos_ang]
    Z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return atoms, Z, pos_ang, pos_bohr

def run_gpupyscf(atoms, pos_ang, xc='b3lyp', basis='def2-svp', pseudo=None):
    from pyscf import gto, dft
    import gpu4pyscf
    kwargs = dict(
        atom=[(a, r) for a, r in zip(atoms, pos_ang)],
        basis=basis,
        unit='Angstrom',
    )
    if pseudo:
        kwargs['pseudo'] = pseudo
    mol = gto.M(**kwargs)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-7
    t0 = time.time()
    e = mf.kernel()
    wall = time.time() - t0
    return {
        'engine': 'gpu4pyscf',
        'basis': basis,
        'pseudo': pseudo or 'none',
        'xc': xc,
        'conv_tol': 1e-7,
        'E_total': e,
        'converged': mf.converged,
        'n_iters': mf.cycles,
        'wall_s': round(wall, 3),
    }

def run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=True):
    import tides._native as native
    if not use_pp:
        del os.environ['TIDES_SRC_DIR']
    if use_pp:
        os.environ['TIDES_LEVEL_SHIFT'] = '0.2'
    t0 = time.time()
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos_bohr_flat,
        grid_h=0.5, grid_margin=6.0,
        max_iter=200, tol=1e-7,
        use_dual_grid=False, xc_functional=xc,
        allow_grid_refine=False,
    )
    wall = time.time() - t0
    if not use_pp:
        os.environ['TIDES_SRC_DIR'] = TIDES_SRC
    if 'TIDES_LEVEL_SHIFT' in os.environ:
        del os.environ['TIDES_LEVEL_SHIFT']
    bh = r.build_H_timings
    return {
        'engine': 'tides',
        'basis': 'NAO-DZP',
        'pseudo': 'PseudoDojo' if use_pp else 'none',
        'xc': xc,
        'E_total': r.energy.E_total,
        'converged': r.scf.converged,
        'n_iters': bh.n_iterations,
        'conv_tol': 1e-7,
        'wall_s': round(wall, 3),
        'build_H_ms': round(bh.total_ms, 2),
        'xc_eval_ms': round(bh.xc_eval_ms, 2),
        'poisson_ms': round(bh.poisson_ms, 2),
        'gpu_pipeline': bh.used_gpu_pipeline,
    }

molecules = [
    ('CH4',      'C',       5),
    ('H2O',      'O',       3),
    ('NH3',      'N',       4),
    ('C2H6',     'CC',      8),
    ('C4H10',    'CCCC',   14),
    ('C6H6',     'c1ccccc1', 12),
]

results = []
for label, smiles, natoms in molecules:
    print(f"\n{'='*70}")
    print(f"  {label} ({natoms} atoms)  SMILES={smiles}")
    print(f"{'='*70}")

    atoms, Z, pos_ang, pos_bohr = make_molecule(smiles)
    pos_bohr_flat = [c for r in pos_bohr for c in r]

    # --- GPU-PySCF all-electron ---
    print(f"  --- GPU-PySCF AE (def2-svp, B3LYP) ---")
    r_gpu_ae = run_gpupyscf(atoms, pos_ang, xc='b3lyp', basis='def2-svp')
    print(f"  E={r_gpu_ae['E_total']:.6f}  conv={r_gpu_ae['converged']}  iters={r_gpu_ae['n_iters']}  wall={r_gpu_ae['wall_s']:.2f}s")

    # --- GPU-PySCF pseudopotential ---
    print(f"  --- GPU-PySCF PP (gth-dzvp + gth-pbe, B3LYP) ---")
    r_gpu_pp = run_gpupyscf(atoms, pos_ang, xc='b3lyp', basis='gth-dzvp', pseudo='gth-pbe')
    print(f"  E={r_gpu_pp['E_total']:.6f}  conv={r_gpu_pp['converged']}  iters={r_gpu_pp['n_iters']}  wall={r_gpu_pp['wall_s']:.2f}s")

    # --- TIDES all-electron ---
    print(f"  --- TIDES AE (NAO-DZP, B3LYP) ---")
    r_tides_ae = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=False)
    gpu_str = "GPU" if r_tides_ae['gpu_pipeline'] else "CPU"
    print(f"  E={r_tides_ae['E_total']:.6f}  conv={r_tides_ae['converged']}  iters={r_tides_ae['n_iters']}  wall={r_tides_ae['wall_s']:.2f}s  [{gpu_str}]")

    # --- TIDES pseudopotential ---
    print(f"  --- TIDES PP (NAO-DZP + PseudoDojo, B3LYP) ---")
    r_tides_pp = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=True)
    gpu_str = "GPU" if r_tides_pp['gpu_pipeline'] else "CPU"
    print(f"  E={r_tides_pp['E_total']:.6f}  conv={r_tides_pp['converged']}  iters={r_tides_pp['n_iters']}  wall={r_tides_pp['wall_s']:.2f}s  [{gpu_str}]")

    dE_pp = r_tides_pp['E_total'] - r_gpu_pp['E_total']
    dE_ae = r_tides_ae['E_total'] - r_gpu_ae['E_total']
    speedup_pp = r_gpu_pp['wall_s'] / r_tides_pp['wall_s'] if r_tides_pp['wall_s'] > 0 else float('inf')
    speedup_ae = r_gpu_ae['wall_s'] / r_tides_ae['wall_s'] if r_tides_ae['wall_s'] > 0 else float('inf')
    entry = {
        'label': label, 'natoms': natoms,
        'gpu_ae': r_gpu_ae, 'gpu_pp': r_gpu_pp,
        'tides_ae': r_tides_ae, 'tides_pp': r_tides_pp,
        'dE_ae': round(dE_ae, 6), 'dE_pp': round(dE_pp, 6),
        'speedup_ae': round(speedup_ae, 3), 'speedup_pp': round(speedup_pp, 3),
    }
    results.append(entry)
    print(f"  AE: dE={dE_ae:+.4f}  speedup={speedup_ae:.2f}x")
    print(f"  PP: dE={dE_pp:+.4f}  speedup={speedup_pp:.2f}x")

print(f"\n{'='*110}")
print("SUMMARY: GPU-PySCF vs TIDES (B3LYP) — 4 configurations")
print(f"{'='*110}")
print(f"{'Mol':<10} {'N':>3} | {'GPU-AE':>12} {'TIDES-AE':>12} {'dE_AE':>9} {'sp_AE':>6} | {'GPU-PP':>12} {'TIDES-PP':>12} {'dE_PP':>9} {'sp_PP':>6}")
print(f"{'':>14} | {'it':>3} {'it':>3} {'conv':>4} {'conv':>4} | {'it':>3} {'it':>3} {'conv':>4} {'conv':>4}")
print("-"*110)
for e in results:
    print(f"{e['label']:<10} {e['natoms']:>3} | {e['gpu_ae']['E_total']:>12.4f} {e['tides_ae']['E_total']:>12.4f} {e['dE_ae']:>+9.4f} {e['speedup_ae']:>5.2f}x | {e['gpu_pp']['E_total']:>12.4f} {e['tides_pp']['E_total']:>12.4f} {e['dE_pp']:>+9.4f} {e['speedup_pp']:>5.2f}x")
    ae_g = 'Y' if e['gpu_ae']['converged'] else 'N'
    ae_t = 'Y' if e['tides_ae']['converged'] else 'N'
    pp_g = 'Y' if e['gpu_pp']['converged'] else 'N'
    pp_t = 'Y' if e['tides_pp']['converged'] else 'N'
    print(f"{'':>14} | {e['gpu_ae']['n_iters']:>3} {e['tides_ae']['n_iters']:>3} {ae_g:>4} {ae_t:>4} | {e['gpu_pp']['n_iters']:>3} {e['tides_pp']['n_iters']:>3} {pp_g:>4} {pp_t:>4}")

print(f"\n{'='*110}")
print("TIMING SUMMARY (wall seconds)")
print(f"{'='*110}")
print(f"{'Mol':<10} {'N':>3} | {'GPU-AE':>8} {'TIDES-AE':>9} | {'GPU-PP':>8} {'TIDES-PP':>9} | {'GPU?':>5} | {'tol':>8}")
print("-"*90)
for e in results:
    gpu = "Yes" if e['tides_pp']['gpu_pipeline'] else "No"
    print(f"{e['label']:<10} {e['natoms']:>3} | {e['gpu_ae']['wall_s']:>7.2f}s {e['tides_ae']['wall_s']:>8.2f}s | {e['gpu_pp']['wall_s']:>7.2f}s {e['tides_pp']['wall_s']:>8.2f}s | {gpu:>5} | 1e-7")

out_path = '/home/indranil/git/New_DFT_Code/tides/bench/profiling_results/gpupyscf_vs_tides.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
