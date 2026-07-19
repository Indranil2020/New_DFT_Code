#!/usr/bin/env python3
"""Quick TIDES-only benchmark to measure optimization impact.
Compares against saved gpu4pyscf reference data.
Does NOT re-run PySCF/gpu4pyscf.
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
    atoms, pos_ang, Z = [], [], []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
        p = conf.GetAtomPosition(atom.GetIdx())
        pos_ang.append([p.x, p.y, p.z])
        Z.append(atom.GetAtomicNum())
    pos_bohr = [[c * BOHR for c in r] for r in pos_ang]
    pos_flat = [c for r in pos_bohr for c in r]
    return atoms, Z, pos_flat

def run_tides(Z, pos_flat, xc='b3lyp', use_pp=False, use_mp=False, use_graph=False):
    from tides._native import NaoDriver
    import os
    # Use a non-existent directory to disable PP loading when use_pp=False.
    # Passing empty string triggers DefaultPpDir() which finds data/pseudodojo.
    pp_dir = os.path.join(TIDES_SRC, 'external', 'pseudopotentials', 'pseudodojo-pbe-sr') if use_pp else '/nonexistent_tides_pp_dir'
    t0 = time.time()
    grid_h = float(os.environ.get('TIDES_GRID_H', '0.3'))
    grid_margin = float(os.environ.get('TIDES_GRID_MARGIN', '6.0'))
    r = NaoDriver.run(
        atomic_numbers=Z, positions=pos_flat,
        grid_h=grid_h, grid_margin=grid_margin,
        max_iter=100, tol=1e-7,
        use_dual_grid=False,
        xc_functional=xc,
        pp_dir=pp_dir,
        allow_grid_refine=False,
        use_mixed_precision=use_mp,
        use_cuda_graph=use_graph,
    )
    wall = time.time() - t0
    bh = r.build_H_timings
    return {
        'E_total': r.energy.E_total,
        'converged': r.scf.converged,
        'n_iters': bh.n_iterations,
        'wall_s': round(wall, 3),
        'build_H_ms': round(bh.total_ms, 2),
        'xc_eval_ms': round(bh.xc_eval_ms, 2),
        'poisson_ms': round(bh.poisson_ms, 2),
        'gpu_pipeline': bh.used_gpu_pipeline,
    }

# Load saved gpu4pyscf reference
ref_path = os.path.join(TIDES_SRC, 'bench/profiling_results/PYSCF_GPU_REFERENCE_DATA.json')
with open(ref_path) as f:
    ref = json.load(f)

ref_mols = {m['label']: m for m in ref['comparison_b3lyp']['molecules']}

test_mols = [
    ('CH4',  'C',          True),
    ('H2O',  'O',          True),
    ('C2H6', 'CC',         True),
    ('NH3',  'N',          True),
]

print(f"\n{'='*80}")
print(f"  TIDES Optimization Benchmark (OPT-1 through OPT-5)")
print(f"{'='*80}")
print(f"  Comparing against saved gpu4pyscf reference (no PySCF re-run)")
print()

results = []
use_mp = os.environ.get('TIDES_USE_MIXED', '0') == '1'
use_graph = os.environ.get('TIDES_USE_CUDA_GRAPH', '0') == '1'
for label, smiles, use_pp in test_mols:
    atoms, Z, pos_flat = make_molecule(smiles)
    print(f"\n--- {label} ({len(Z)} atoms, {'PP' if use_pp else 'AE'}) ---")

    tides_res = run_tides(Z, pos_flat, xc='b3lyp', use_pp=use_pp, use_mp=use_mp, use_graph=use_graph)
    gpu_ref = ref_mols[label]['gpu_pp'] if use_pp else ref_mols[label]['gpu_ae']

    speedup = gpu_ref['wall_s'] / tides_res['wall_s'] if tides_res['wall_s'] > 0 else 0
    iter_ratio = gpu_ref['n_iters'] / tides_res['n_iters'] if tides_res['n_iters'] > 0 else 0

    print(f"  TIDES:  E={tides_res['E_total']:.6f}  iters={tides_res['n_iters']}  wall={tides_res['wall_s']}s  build_H={tides_res['build_H_ms']}ms  xc={tides_res['xc_eval_ms']}ms")
    print(f"  gpu4p:  E={gpu_ref['E_total']:.6f}  iters={gpu_ref['n_iters']}  wall={gpu_ref['wall_s']}s")
    print(f"  Speedup: {speedup:.3f}x  (TIDES/gpu4pyscf)")
    print(f"  Iter ratio: {iter_ratio:.2f}  (gpu iters / tides iters)")

    results.append({
        'label': label,
        'natoms': len(Z),
        'tides': tides_res,
        'gpu4pyscf': gpu_ref,
        'speedup': speedup,
        'iter_ratio': iter_ratio,
    })

print(f"\n{'='*80}")
print(f"  Summary")
print(f"{'='*80}")
print(f"{'Molecule':<10} {'TIDES wall(s)':<15} {'gpu4p wall(s)':<15} {'Speedup':<10} {'TIDES iters':<12} {'gpu iters':<10}")
for r in results:
    print(f"{r['label']:<10} {r['tides']['wall_s']:<15} {r['gpu4pyscf']['wall_s']:<15} {r['speedup']:<10.3f} {r['tides']['n_iters']:<12} {r['gpu4pyscf']['n_iters']:<10}")

# Save results
out_path = os.path.join(TIDES_SRC, 'bench/profiling_results/optimization_bench_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
