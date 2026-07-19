#!/usr/bin/env python3
"""Fair TIDES vs PySCF benchmark — performance + self-consistency.

Design principles:
  1. Same RDKit geometries for all engines (fixed seed).
  2. Matched basis sizes: NAO-DZP (~8 fns/atom) vs def2-svp (~7 fns/atom).
  3. AE and PP compared separately — never mix.
  4. Cold + warm runs for TIDES (basis cache warms up on cold run).
  5. VRAM scaling: 5→62 atoms, log GPU/CPU fallback point.
  6. Substep profiling: build_H decomposition (rho, poisson, xc, vmat).
  7. Self-consistency check: each engine must converge and produce stable energy.
  8. C6H6 PP: 250 max_iter, level_shift=0.2 to test full convergence.

Configurations per molecule:
  - TIDES AE (NAO-DZP, no PP) — cold + warm
  - TIDES PP (NAO-DZP + PseudoDojo) — cold + warm
  - GPU-PySCF AE (def2-svp)
  - GPU-PySCF PP (gth-dzvp + gth-pbe)
  - CPU-PySCF AE (def2-svp)
  - CPU-PySCF PP (gth-dzvp + gth-pbe)
"""
import sys, os, time, json, subprocess

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
os.environ['TIDES_SRC_DIR'] = TIDES_SRC
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))

BOHR = 1.8897261254535

# ── Molecule ladder: 5→62 atoms ──────────────────────────────────────────────
MOLECULES = [
    ('CH4',    'C',           5),
    ('H2O',    'O',           3),
    ('NH3',    'N',           4),
    ('C2H6',   'CC',          8),
    ('C6H6',   'c1ccccc1',   12),
    ('C4H10',  'CCCC',       14),
    ('C8H18',  'CCCCCCCC',   26),
    ('C10H22', 'CCCCCCCCCC', 32),
    ('C14H30', 'C'*14,        44),
    ('C20H42', 'C'*20,        62),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_molecule(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms, Z, pos_ang, pos_bohr = [], [], [], []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        pos_ang.append([p.x, p.y, p.z])
        pos_bohr.append([p.x * BOHR, p.y * BOHR, p.z * BOHR])
        Z.append(atom.GetAtomicNum())
    pos_bohr_flat = [c for r in pos_bohr for c in r]
    return atoms, Z, pos_ang, pos_bohr_flat


def get_gpu_info():
    r = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
        capture_output=True, text=True)
    return r.stdout.strip()


def level_shift_for(natoms):
    if natoms > 20:
        return 0.3
    if natoms > 4:
        return 0.2
    return 0.0


def max_iter_for(natoms, is_pp=False):
    if is_pp and natoms >= 12:
        return 250
    if natoms > 20:
        return 120
    return 100


# ── TIDES runner ──────────────────────────────────────────────────────────────
def run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=True, natoms=5, label=''):
    import tides._native as native
    if not use_pp:
        del os.environ['TIDES_SRC_DIR']
    ls = level_shift_for(natoms)
    if ls > 0:
        os.environ['TIDES_LEVEL_SHIFT'] = str(ls)
    mi = max_iter_for(natoms, is_pp=use_pp)
    t0 = time.time()
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos_bohr_flat,
        grid_h=0.5, grid_margin=6.0,
        max_iter=mi, tol=1e-7,
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
        'rho_build_ms': round(bh.rho_build_ms, 2),
        'vmat_build_ms': round(bh.vmat_build_ms, 2),
        'assemble_H_ms': round(bh.assemble_H_ms, 2),
        'gpu_pipeline': bh.used_gpu_pipeline,
        'n_basis': r.n_basis,
        'level_shift': ls,
        'max_iter': mi,
    }


def run_tides_cpu(Z, pos_bohr_flat, **kw):
    os.environ['TIDES_DISABLE_GPU'] = '1'
    r = run_tides(Z, pos_bohr_flat, **kw)
    if 'TIDES_DISABLE_GPU' in os.environ:
        del os.environ['TIDES_DISABLE_GPU']
    r['engine'] = 'tides_cpu'
    return r


# ── PySCF runners ─────────────────────────────────────────────────────────────
def run_pyscf_gpu(atoms, pos_ang, xc='b3lyp', basis='def2-svp', pseudo=None, natoms=5):
    from pyscf import gto, dft
    import gpu4pyscf  # noqa: F401
    kwargs = dict(
        atom=[(a, r) for a, r in zip(atoms, pos_ang)],
        basis=basis, unit='Angstrom',
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
        'engine': 'gpu4pyscf', 'basis': basis,
        'pseudo': pseudo or 'none', 'xc': xc,
        'E_total': e, 'converged': mf.converged,
        'n_iters': mf.cycles, 'conv_tol': 1e-7,
        'wall_s': round(wall, 3), 'n_basis': mol.nao,
    }


def run_pyscf_cpu(atoms, pos_ang, xc='b3lyp', basis='def2-svp', pseudo=None, natoms=5):
    from pyscf import gto, dft
    kwargs = dict(
        atom=[(a, r) for a, r in zip(atoms, pos_ang)],
        basis=basis, unit='Angstrom',
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
        'engine': 'pyscf_cpu', 'basis': basis,
        'pseudo': pseudo or 'none', 'xc': xc,
        'E_total': e, 'converged': mf.converged,
        'n_iters': mf.cycles, 'conv_tol': 1e-7,
        'wall_s': round(wall, 3), 'n_basis': mol.nao,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────
def main():
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info}")
    print(f"TIDES_SRC: {TIDES_SRC}")
    print()

    results = []

    for label, smiles, natoms in MOLECULES:
        print(f"\n{'='*80}")
        print(f"  {label} ({natoms} atoms)  SMILES={smiles}")
        print(f"{'='*80}")

        atoms, Z, pos_ang, pos_bohr_flat = make_molecule(smiles)
        actual_n = len(Z)
        print(f"  Atoms: {atoms}")
        print(f"  Z: {Z}")

        entry = {'label': label, 'smiles': smiles, 'natoms': actual_n}

        # ── TIDES AE: cold (basis gen) then warm (cached) ───────────────────
        print(f"\n  --- TIDES AE cold (NAO-DZP, B3LYP) ---")
        r_tides_ae_cold = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=False, natoms=actual_n)
        gpu_str = "GPU" if r_tides_ae_cold['gpu_pipeline'] else "CPU"
        print(f"  E={r_tides_ae_cold['E_total']:.6f}  conv={r_tides_ae_cold['converged']}  "
              f"iters={r_tides_ae_cold['n_iters']}  wall={r_tides_ae_cold['wall_s']:.2f}s  "
              f"nbf={r_tides_ae_cold['n_basis']}  [{gpu_str}]")
        print(f"    build_H={r_tides_ae_cold['build_H_ms']:.1f}ms  xc={r_tides_ae_cold['xc_eval_ms']:.1f}ms  "
              f"poisson={r_tides_ae_cold['poisson_ms']:.1f}ms  vmat={r_tides_ae_cold['vmat_build_ms']:.1f}ms")

        print(f"  --- TIDES AE warm (cached basis) ---")
        r_tides_ae_warm = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=False, natoms=actual_n)
        gpu_str = "GPU" if r_tides_ae_warm['gpu_pipeline'] else "CPU"
        print(f"  E={r_tides_ae_warm['E_total']:.6f}  conv={r_tides_ae_warm['converged']}  "
              f"iters={r_tides_ae_warm['n_iters']}  wall={r_tides_ae_warm['wall_s']:.2f}s  [{gpu_str}]")

        # ── TIDES PP: cold then warm ────────────────────────────────────────
        print(f"\n  --- TIDES PP cold (NAO-DZP + PseudoDojo, B3LYP) ---")
        r_tides_pp_cold = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=True, natoms=actual_n)
        gpu_str = "GPU" if r_tides_pp_cold['gpu_pipeline'] else "CPU"
        print(f"  E={r_tides_pp_cold['E_total']:.6f}  conv={r_tides_pp_cold['converged']}  "
              f"iters={r_tides_pp_cold['n_iters']}  wall={r_tides_pp_cold['wall_s']:.2f}s  "
              f"nbf={r_tides_pp_cold['n_basis']}  [{gpu_str}]  ls={r_tides_pp_cold['level_shift']}")
        print(f"    build_H={r_tides_pp_cold['build_H_ms']:.1f}ms  xc={r_tides_pp_cold['xc_eval_ms']:.1f}ms  "
              f"poisson={r_tides_pp_cold['poisson_ms']:.1f}ms  vmat={r_tides_pp_cold['vmat_build_ms']:.1f}ms")

        print(f"  --- TIDES PP warm (cached basis) ---")
        r_tides_pp_warm = run_tides(Z, pos_bohr_flat, xc='b3lyp', use_pp=True, natoms=actual_n)
        gpu_str = "GPU" if r_tides_pp_warm['gpu_pipeline'] else "CPU"
        print(f"  E={r_tides_pp_warm['E_total']:.6f}  conv={r_tides_pp_warm['converged']}  "
              f"iters={r_tides_pp_warm['n_iters']}  wall={r_tides_pp_warm['wall_s']:.2f}s  [{gpu_str}]")

        # ── GPU-PySCF AE ─────────────────────────────────────────────────────
        r_gpu_ae = None
        print(f"\n  --- GPU-PySCF AE (def2-svp, B3LYP) ---")
        r_gpu_ae = run_pyscf_gpu(atoms, pos_ang, xc='b3lyp', basis='def2-svp', natoms=actual_n)
        print(f"  E={r_gpu_ae['E_total']:.6f}  conv={r_gpu_ae['converged']}  "
              f"iters={r_gpu_ae['n_iters']}  wall={r_gpu_ae['wall_s']:.2f}s  nbf={r_gpu_ae['n_basis']}")

        # ── GPU-PySCF PP ─────────────────────────────────────────────────────
        r_gpu_pp = None
        print(f"  --- GPU-PySCF PP (gth-dzvp + gth-pbe, B3LYP) ---")
        r_gpu_pp = run_pyscf_gpu(atoms, pos_ang, xc='b3lyp', basis='gth-dzvp', pseudo='gth-pbe', natoms=actual_n)
        print(f"  E={r_gpu_pp['E_total']:.6f}  conv={r_gpu_pp['converged']}  "
              f"iters={r_gpu_pp['n_iters']}  wall={r_gpu_pp['wall_s']:.2f}s  nbf={r_gpu_pp['n_basis']}")

        # ── CPU PySCF AE ─────────────────────────────────────────────────────
        r_cpu_ae = None
        print(f"  --- CPU PySCF AE (def2-svp, B3LYP) ---")
        r_cpu_ae = run_pyscf_cpu(atoms, pos_ang, xc='b3lyp', basis='def2-svp', natoms=actual_n)
        print(f"  E={r_cpu_ae['E_total']:.6f}  conv={r_cpu_ae['converged']}  "
              f"iters={r_cpu_ae['n_iters']}  wall={r_cpu_ae['wall_s']:.2f}s  nbf={r_cpu_ae['n_basis']}")

        # ── CPU PySCF PP ─────────────────────────────────────────────────────
        r_cpu_pp = None
        print(f"  --- CPU PySCF PP (gth-dzvp + gth-pbe, B3LYP) ---")
        r_cpu_pp = run_pyscf_cpu(atoms, pos_ang, xc='b3lyp', basis='gth-dzvp', pseudo='gth-pbe', natoms=actual_n)
        print(f"  E={r_cpu_pp['E_total']:.6f}  conv={r_cpu_pp['converged']}  "
              f"iters={r_cpu_pp['n_iters']}  wall={r_cpu_pp['wall_s']:.2f}s  nbf={r_cpu_pp['n_basis']}")

        # ── Self-consistency checks ──────────────────────────────────────────
        ae_energy_stable = abs(r_tides_ae_warm['E_total'] - r_tides_ae_cold['E_total']) < 1e-6
        pp_energy_stable = abs(r_tides_pp_warm['E_total'] - r_tides_pp_cold['E_total']) < 1e-6

        entry.update({
            'tides_ae_cold': r_tides_ae_cold,
            'tides_ae_warm': r_tides_ae_warm,
            'tides_pp_cold': r_tides_pp_cold,
            'tides_pp_warm': r_tides_pp_warm,
            'gpu_ae': r_gpu_ae,
            'gpu_pp': r_gpu_pp,
            'cpu_ae': r_cpu_ae,
            'cpu_pp': r_cpu_pp,
            'ae_energy_stable': ae_energy_stable,
            'pp_energy_stable': pp_energy_stable,
        })
        results.append(entry)

        # Per-molecule summary
        ae_warm = r_tides_ae_warm
        pp_warm = r_tides_pp_warm
        print(f"\n  SUMMARY {label}:")
        print(f"    AE: TIDES {ae_warm['wall_s']:.2f}s (cold {r_tides_ae_cold['wall_s']:.2f}s)  "
              f"GPU-PySCF {r_gpu_ae['wall_s']:.2f}s  CPU-PySCF {r_cpu_ae['wall_s']:.2f}s")
        print(f"    PP: TIDES {pp_warm['wall_s']:.2f}s (cold {r_tides_pp_cold['wall_s']:.2f}s)  "
              f"GPU-PySCF {r_gpu_pp['wall_s']:.2f}s  CPU-PySCF {r_cpu_pp['wall_s']:.2f}s")
        print(f"    AE conv: TIDES={ae_warm['converged']}  GPU={r_gpu_ae['converged']}  CPU={r_cpu_ae['converged']}")
        print(f"    PP conv: TIDES={pp_warm['converged']}  GPU={r_gpu_pp['converged']}  CPU={r_cpu_pp['converged']}")
        print(f"    Energy stable: AE={ae_energy_stable}  PP={pp_energy_stable}")

    # ── Final summary tables ────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("FAIR BENCHMARK SUMMARY — TIDES vs PySCF (B3LYP)")
    print(f"GPU: {gpu_info}")
    print(f"{'='*120}")

    # Table 1: Performance (warm timings)
    print(f"\n{'Mol':<8} {'N':>3} | {'TIDES-AE':>10} {'GPU-AE':>10} {'CPU-AE':>10} | "
          f"{'TIDES-PP':>10} {'GPU-PP':>10} {'CPU-PP':>10} | {'GPU?':>5}")
    print("-" * 100)
    for e in results:
        ae = e['tides_ae_warm']
        pp = e['tides_pp_warm']
        gpu = "Yes" if pp['gpu_pipeline'] else "No"
        ga = e['gpu_ae'] or {'wall_s': 0}
        gp = e['gpu_pp'] or {'wall_s': 0}
        ca = e['cpu_ae'] or {'wall_s': 0}
        cp = e['cpu_pp'] or {'wall_s': 0}
        print(f"{e['label']:<8} {e['natoms']:>3} | "
              f"{ae['wall_s']:>9.2f}s {ga['wall_s']:>9.2f}s {ca['wall_s']:>9.2f}s | "
              f"{pp['wall_s']:>9.2f}s {gp['wall_s']:>9.2f}s {cp['wall_s']:>9.2f}s | {gpu:>5}")

    # Table 2: Speedup ratios (warm TIDES vs GPU-PySCF)
    print(f"\n{'Mol':<8} {'N':>3} | {'AE speedup':>12} {'PP speedup':>12} | "
          f"{'AE cold':>10} {'PP cold':>10} | {'Cache x':>8}")
    print("-" * 80)
    for e in results:
        ae_w = e['tides_ae_warm']
        pp_w = e['tides_pp_warm']
        ae_c = e['tides_ae_cold']
        pp_c = e['tides_pp_cold']
        ga = e['gpu_ae'] or {'wall_s': 0.001}
        gp = e['gpu_pp'] or {'wall_s': 0.001}
        sp_ae = ga['wall_s'] / ae_w['wall_s'] if ae_w['wall_s'] > 0 else 0
        sp_pp = gp['wall_s'] / pp_w['wall_s'] if pp_w['wall_s'] > 0 else 0
        cache_ae = ae_c['wall_s'] / ae_w['wall_s'] if ae_w['wall_s'] > 0 else 0
        print(f"{e['label']:<8} {e['natoms']:>3} | "
              f"{sp_ae:>11.2f}x {sp_pp:>11.2f}x | "
              f"{ae_c['wall_s']:>9.2f}s {pp_c['wall_s']:>9.2f}s | {cache_ae:>7.1f}x")

    # Table 3: Convergence + self-consistency
    print(f"\n{'Mol':<8} {'N':>3} | {'AE conv':>16} {'PP conv':>16} | {'AE stable':>10} {'PP stable':>10}")
    print("-" * 80)
    for e in results:
        ae = e['tides_ae_warm']
        pp = e['tides_pp_warm']
        ga = e['gpu_ae'] or {'converged': False}
        gp = e['gpu_pp'] or {'converged': False}
        ca = e['cpu_ae'] or {'converged': False}
        cp = e['cpu_pp'] or {'converged': False}
        ae_conv = f"T={ae['converged']}/G={ga['converged']}/C={ca['converged']}"
        pp_conv = f"T={pp['converged']}/G={gp['converged']}/C={cp['converged']}"
        print(f"{e['label']:<8} {e['natoms']:>3} | {ae_conv:>16} {pp_conv:>16} | "
              f"{str(e['ae_energy_stable']):>10} {str(e['pp_energy_stable']):>10}")

    # Table 4: Iteration counts
    print(f"\n{'Mol':<8} {'N':>3} | {'AE iters':>20} | {'PP iters':>20}")
    print("-" * 70)
    for e in results:
        ae = e['tides_ae_warm']
        pp = e['tides_pp_warm']
        ga = e['gpu_ae'] or {'n_iters': 0}
        gp = e['gpu_pp'] or {'n_iters': 0}
        ca = e['cpu_ae'] or {'n_iters': 0}
        cp = e['cpu_pp'] or {'n_iters': 0}
        ae_it = f"T={ae['n_iters']:>3} G={ga['n_iters']:>3} C={ca['n_iters']:>3}"
        pp_it = f"T={pp['n_iters']:>3} G={gp['n_iters']:>3} C={cp['n_iters']:>3}"
        print(f"{e['label']:<8} {e['natoms']:>3} | {ae_it:>20} | {pp_it:>20}")

    # Table 5: TIDES build_H substep profiling (warm PP, ms/iter)
    print(f"\nTIDES build_H substep profiling (warm PP, ms/iter):")
    print(f"{'Mol':<8} {'N':>3} | {'total':>8} {'rho':>8} {'poisson':>8} {'xc':>8} {'vmat':>8} {'asmbl':>8} | {'GPU?':>5}")
    print("-" * 80)
    for e in results:
        pp = e['tides_pp_warm']
        gpu = "Yes" if pp['gpu_pipeline'] else "No"
        print(f"{e['label']:<8} {e['natoms']:>3} | "
              f"{pp['build_H_ms']:>7.1f} {pp['rho_build_ms']:>7.1f} {pp['poisson_ms']:>7.1f} "
              f"{pp['xc_eval_ms']:>7.1f} {pp['vmat_build_ms']:>7.1f} {pp['assemble_H_ms']:>7.1f} | {gpu:>5}")

    # Table 6: Basis sizes
    print(f"\nBasis sizes (nbf):")
    print(f"{'Mol':<8} {'N':>3} | {'TIDES':>8} {'def2-svp':>10} {'gth-dzvp':>10}")
    print("-" * 50)
    for e in results:
        ae = e['tides_ae_warm']
        ga = e['gpu_ae'] or {'n_basis': 0}
        gp = e['gpu_pp'] or {'n_basis': 0}
        print(f"{e['label']:<8} {e['natoms']:>3} | {ae['n_basis']:>8} {ga['n_basis']:>10} {gp['n_basis']:>10}")

    # Table 7: VRAM scaling
    print(f"\nVRAM scaling (TIDES PP, GPU fallback point):")
    print(f"{'Mol':<8} {'N':>3} | {'GPU?':>5} | {'wall_warm':>10} {'build_H':>10} | {'nbf':>6}")
    print("-" * 60)
    for e in results:
        pp = e['tides_pp_warm']
        gpu = "Yes" if pp['gpu_pipeline'] else "No"
        print(f"{e['label']:<8} {e['natoms']:>3} | {gpu:>5} | "
              f"{pp['wall_s']:>9.2f}s {pp['build_H_ms']:>9.1f}ms | {pp['n_basis']:>6}")

    # Save JSON
    out_path = os.path.join(TIDES_SRC, 'bench/profiling_results/fair_benchmark_2026-07-17.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
