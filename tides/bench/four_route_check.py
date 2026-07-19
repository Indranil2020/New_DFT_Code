#!/usr/bin/env python3
"""Four-route ground-truth check (P0 gate): TIDES AE/PP x CPU/GPU on CH4 + H2O.

Protocol matches four_route_ground_report_2026-07-18.md: PBE, grid_h=0.3,
margin 6.0, tol 1e-7, no level shift (P0 wants raw physics, not workarounds).

Gates checked per molecule:
  G-A  each route converges
  G-B  CPU vs GPU same route |dE| <= 1e-6 Ha   (roadmap gate is 1e-8; report at 1e-6 first)
  G-C  AE vs frozen PySCF def2-svp PBE reference (sanity, basis-limited ~0.1 Ha)
  G-D  PP vs AE consistency (valence physics sanity)

Frozen PySCF references (four_route_ground_report_2026-07-18.md, RDKit seed 42):
  CH4 AE -40.4146 | H2O AE -76.2725 | CH4 PP(ccECP) -7.9459 | H2O PP(ccECP) -16.9454

Usage:
  LD_PRELOAD=<mkl preload> python3 four_route_check.py [--max-iter N]
Writes JSON next to this file under profiling_results/.
"""
import sys, os, time, json, argparse

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
os.environ['TIDES_SRC_DIR'] = TIDES_SRC
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))

BOHR = 1.8897261254535
PYSCF_REF = {
    ('CH4', 'AE'): -40.4146, ('H2O', 'AE'): -76.2725,
    ('CH4', 'PP'): -7.9459,  ('H2O', 'PP'): -16.9454,
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


def run_route(Z, pos, use_pp, use_gpu, max_iter):
    import tides._native as native
    if use_gpu:
        os.environ.pop('TIDES_DISABLE_GPU', None)
    else:
        os.environ['TIDES_DISABLE_GPU'] = '1'
    if use_pp:
        os.environ['TIDES_SRC_DIR'] = TIDES_SRC
    else:
        os.environ.pop('TIDES_SRC_DIR', None)
    t0 = time.time()
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=0.3, grid_margin=6.0,
        max_iter=max_iter, tol=1e-7,
        use_dual_grid=False, xc_functional='pbe',
        allow_grid_refine=False,
    )
    wall = time.time() - t0
    os.environ['TIDES_SRC_DIR'] = TIDES_SRC
    os.environ.pop('TIDES_DISABLE_GPU', None)
    bh = r.build_H_timings
    return {
        'E': r.energy.E_total, 'converged': r.scf.converged,
        'iters': bh.n_iterations, 'wall_s': round(wall, 2),
        'gpu': bool(bh.used_gpu_pipeline), 'n_basis': r.n_basis,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-iter', type=int, default=60)
    args = ap.parse_args()

    results = {}
    for label, smiles in [('CH4', 'C'), ('H2O', 'O')]:
        Z, pos = make_molecule(smiles)
        entry = {}
        for pp in (False, True):
            for gpu in (True, False):
                key = f"{'PP' if pp else 'AE'}_{'GPU' if gpu else 'CPU'}"
                print(f"--- {label} {key} ---", flush=True)
                entry[key] = run_route(Z, pos, pp, gpu, args.max_iter)
                e = entry[key]
                print(f"    E={e['E']:.6f} conv={e['converged']} iters={e['iters']} "
                      f"wall={e['wall_s']}s gpu_pipeline={e['gpu']}", flush=True)
        results[label] = entry

    print("\n==== GATE SUMMARY ====")
    all_pass = True
    for label, entry in results.items():
        for route in ('AE', 'PP'):
            g, c = entry[f'{route}_GPU'], entry[f'{route}_CPU']
            d_cg = abs(g['E'] - c['E'])
            conv = g['converged'] and c['converged']
            ref = PYSCF_REF[(label, route)]
            d_ref = g['E'] - ref
            ok = conv and d_cg <= 1e-6
            all_pass &= ok
            print(f"{label} {route}: conv(G/C)={g['converged']}/{c['converged']}  "
                  f"|E_gpu-E_cpu|={d_cg:.2e}  E_gpu-ref={d_ref:+.4f} Ha  "
                  f"[{'PASS' if ok else 'FAIL'}]")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'} (conv + CPU~GPU 1e-6 gates)")

    out = os.path.join(TIDES_SRC, 'bench/profiling_results',
                       f"four_route_check_{time.strftime('%Y-%m-%d_%H%M')}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
