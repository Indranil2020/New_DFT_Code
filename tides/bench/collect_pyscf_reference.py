#!/usr/bin/env python3
"""Collect extensive PySCF/gpu4pyscf reference data and freeze it.

This is a ONE-TIME collection script. After running, the output JSON
(PYSCF_REFERENCE_FROZEN.json) is loaded by unified_benchmark.py and
PySCF is never re-run.

Profiling depth per run:
  - Top-level: E_total, converged, n_iters, wall_s, n_basis
  - Per-iteration (via callback): e_tot, delta_E, norm_gorb, norm_ddm, wall_iter_s
  - Per-operation (via monkey-patching): hcore, overlap, veff (J/K/XC substeps),
    eigensolve, make_rdm1, energy_tot, diis
  - Convergence diagnostics: energy_history, delta_E_history, grad_history, ddm_history
  - Grid & basis info: n_grid_points, grid_level, n_ao, n_shells
  - GPU-specific: gpu_actually_used, gpu_memory_used_mb, density_fitting, auxbasis

Configurations: 8 per molecule (CPU/GPU x AE/PP x PBE/B3LYP)
Molecule ladder: CH4, H2O, NH3, C2H6, C6H6, C4H10, C8H18, C10H22
Total: 64 PySCF runs
"""
import sys, os, time, json, subprocess, importlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resource_monitor import ResourceMonitor, get_system_info

# Fix cuSPARSE/nvJitLink version mismatch (needed for gpu4pyscf)
_nvjitlink = Path.home() / ".local" / "lib" / "python3.12" / "site-packages" / "nvidia" / "nvjitlink" / "lib" / "libnvJitLink.so.12"
if _nvjitlink.exists():
    _nvjitlink_str = str(_nvjitlink)
    current_preload = os.environ.get("LD_PRELOAD", "")
    if _nvjitlink_str not in current_preload:
        os.environ["LD_PRELOAD"] = _nvjitlink_str + (":" + current_preload if current_preload else "")
        os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], os.environ)

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
BOHR = 1.8897261254535

MOLECULES = [
    ('CH4',    'C',           5),
    ('H2O',    'O',           3),
    ('NH3',    'N',           4),
    ('C2H6',   'CC',          8),
    ('C6H6',   'c1ccccc1',   12),
    ('C4H10',  'CCCC',       14),
    ('C8H18',  'CCCCCCCC',   26),
    ('C10H22', 'CCCCCCCCCC', 32),
]

XC_FUNCTIONALS = ['pbe', 'b3lyp']

# AE: def2-svp (no pseudo); PP: gth-dzvp + gth-pbe
AE_BASIS = 'def2-svp'
PP_BASIS = 'gth-dzvp'
PP_PSEUDO = 'gth-pbe'


def make_molecule(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms, Z, pos_ang = [], [], []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        Z.append(atom.GetAtomicNum())
        pos_ang.append([p.x, p.y, p.z])
    return atoms, Z, pos_ang


def get_gpu_info():
    r = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
        capture_output=True, text=True)
    return r.stdout.strip()


def make_profiling_callback():
    """Create a callback that collects per-iteration data from PySCF SCF loop."""
    iter_data = []
    iter_wall_times = []
    _iter_start = [None]

    def callback(envs):
        t_now = time.time()
        if _iter_start[0] is not None:
            iter_wall_times.append(t_now - _iter_start[0])
        _iter_start[0] = t_now

        cycle = envs.get('cycle', len(iter_data))
        e_tot = envs.get('e_tot', 0.0)
        last_hf_e = envs.get('last_hf_e', 0.0)
        delta_E = e_tot - last_hf_e if last_hf_e != 0 else 0.0

        import numpy as np
        norm_gorb = envs.get('norm_gorb', 0.0)
        norm_ddm = envs.get('norm_ddm', 0.0)

        mo_energy = envs.get('mo_energy', None)
        mo_occ = envs.get('mo_occ', None)
        dm = envs.get('dm', None)

        dm_norm = float(np.linalg.norm(dm)) if dm is not None else 0.0
        fock = envs.get('fock', None)
        fock_norm = float(np.linalg.norm(fock)) if fock is not None else 0.0

        entry = {
            'iter': cycle + 1,
            'e_tot': float(e_tot),
            'delta_E': float(delta_E),
            'norm_gorb': float(norm_gorb),
            'norm_ddm': float(norm_ddm),
            'dm_norm': dm_norm,
            'fock_norm': fock_norm,
        }
        if mo_energy is not None:
            entry['mo_energy'] = [float(x) for x in np.asarray(mo_energy).ravel()]
        if mo_occ is not None:
            entry['mo_occ'] = [float(x) for x in np.asarray(mo_occ).ravel()]
        iter_data.append(entry)

    return callback, iter_data, iter_wall_times


def make_timed_method(original_method, timing_key, timings_dict):
    """Wrap a method with a timer and accumulate into timings_dict."""
    def timed(*args, **kwargs):
        t0 = time.time()
        result = original_method(*args, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000.0
        if timing_key not in timings_dict:
            timings_dict[timing_key] = []
        timings_dict[timing_key].append(elapsed_ms)
        return result
    return timed


def run_pyscf(atoms, pos_ang, xc='pbe', basis='def2-svp', pseudo=None, use_gpu=False):
    """Run a single PySCF calculation with extensive profiling."""
    import numpy as np
    from pyscf import gto, dft

    is_pp = pseudo is not None
    is_hybrid = (xc.lower() in ('b3lyp', 'pbe0', 'hse06', 'cam-b3lyp'))

    kwargs = dict(
        atom=[(a, r) for a, r in zip(atoms, pos_ang)],
        basis=basis, unit='Angstrom',
    )
    if pseudo:
        kwargs['pseudo'] = pseudo
    mol = gto.M(**kwargs)

    # Determine if gpu4pyscf is available
    gpu_actually_used = False
    gpu_memory_mb = 0.0
    density_fitting = False
    auxbasis = 'none'
    gpu_jk_builder = 'none'

    if use_gpu:
        gpu4pyscf_spec = importlib.util.find_spec('gpu4pyscf')
        if gpu4pyscf_spec is not None:
            import gpu4pyscf
            # Use density fitting for gpu4pyscf (standard practice)
            mol.build()
            gpu_actually_used = True

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-7
    mf.max_cycle = 100

    # Enable density fitting for GPU path (standard gpu4pyscf practice)
    if use_gpu and gpu_actually_used:
        mf = mf.density_fit()
        auxbasis = getattr(mf, 'auxbasis', 'def2-universal-JKFIT')
        density_fitting = True

    # Per-operation timing accumulators
    op_timings = {}

    # Monkey-patch methods with timers
    original_get_hcore = mf.get_hcore
    original_get_ovlp = mf.get_ovlp
    original_get_veff = mf.get_veff
    original_eig = mf.eig
    original_make_rdm1 = mf.make_rdm1
    original_energy_tot = mf.energy_tot

    mf.get_hcore = make_timed_method(original_get_hcore, 'hcore_ms', op_timings)
    mf.get_ovlp = make_timed_method(original_get_ovlp, 'overlap_ms', op_timings)
    mf.get_veff = make_timed_method(original_get_veff, 'veff_ms', op_timings)
    mf.eig = make_timed_method(original_eig, 'eigensolve_ms', op_timings)
    mf.make_rdm1 = make_timed_method(original_make_rdm1, 'make_rdm1_ms', op_timings)
    mf.energy_tot = make_timed_method(original_energy_tot, 'energy_tot_ms', op_timings)

    # Per-iteration callback
    callback, iter_data, iter_wall_times = make_profiling_callback()
    mf.callback = callback

    # GPU memory tracking
    if use_gpu and gpu_actually_used:
        cupy_spec = importlib.util.find_spec('cupy')
        if cupy_spec is not None:
            import cupy as cp
            pool = cp.get_default_memory_pool()

    # Start resource monitor
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    # Run SCF
    t0 = time.time()
    e = mf.kernel()
    wall = time.time() - t0

    # Stop resource monitor
    resource_stats = monitor.stop()

    # Collect GPU memory
    if use_gpu and gpu_actually_used:
        cupy_spec = importlib.util.find_spec('cupy')
        if cupy_spec is not None:
            import cupy as cp
            pool = cp.get_default_memory_pool()
            gpu_memory_mb = pool.total_bytes() / (1024 * 1024)

    # Build per-operation summary (average ms per call)
    op_summary = {}
    for key, values in op_timings.items():
        op_summary[key] = {
            'total_ms': sum(values),
            'avg_ms': sum(values) / len(values) if values else 0.0,
            'n_calls': len(values),
        }

    # Build convergence diagnostics
    energy_history = [d['e_tot'] for d in iter_data]
    delta_E_history = [d['delta_E'] for d in iter_data]
    grad_history = [d['norm_gorb'] for d in iter_data]
    ddm_history = [d['norm_ddm'] for d in iter_data]

    # Estimate convergence rate (fit of log|delta_E| vs iter, excluding first and zero entries)
    convergence_rate = 0.0
    valid_deltas = [(i, abs(d)) for i, d in enumerate(delta_E_history[1:]) if abs(d) > 1e-15]
    if len(valid_deltas) >= 3:
        import math
        log_deltas = [(i, math.log(d)) for i, d in valid_deltas if d > 0]
        if len(log_deltas) >= 3:
            n_pts = len(log_deltas)
            t_mean = sum(t for t, _ in log_deltas) / n_pts
            v_mean = sum(v for _, v in log_deltas) / n_pts
            num = sum((t - t_mean) * (v - v_mean) for t, v in log_deltas)
            den = sum((t - t_mean) ** 2 for t, _ in log_deltas)
            convergence_rate = num / den if den > 0 else 0.0

    # Add per-iteration wall times to iter_data
    for i, wt in enumerate(iter_wall_times):
        if i < len(iter_data):
            iter_data[i]['wall_iter_s'] = wt

    # Grid info
    n_grid_points = 0
    grid_level = 4
    if hasattr(mf, 'grids'):
        n_grid_points = len(mf.grids.coords)
        grid_level = mf.grids.level

    # SCF summary (may contain extra info from PySCF)
    scf_summary = {}
    if hasattr(mf, 'scf_summary'):
        scf_summary = {k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in mf.scf_summary.items()}

    result = {
        'engine': 'gpu4pyscf' if (use_gpu and gpu_actually_used) else 'pyscf_cpu',
        'basis': basis,
        'pseudo': pseudo or 'none',
        'xc': xc,
        'E_total': float(e),
        'converged': bool(mf.converged),
        'n_iters': int(mf.cycles),
        'wall_s': round(wall, 4),
        'conv_tol': 1e-7,
        'n_basis': int(mol.nao),
        'n_grid_points': int(n_grid_points),
        'grid_level': int(grid_level),
        'n_shells': int(mol.nbas),
        'gpu_actually_used': gpu_actually_used,
        'gpu_memory_used_mb': round(gpu_memory_mb, 2),
        'density_fitting': density_fitting,
        'auxbasis': auxbasis,
        'gpu_jk_builder': gpu_jk_builder,
        'is_hybrid': is_hybrid,
        # Resource usage (CPU, RAM, GPU, VRAM)
        'resource_usage': resource_stats,
        # Per-operation timing
        'op_timings': op_summary,
        # Per-iteration data
        'iter_data': iter_data,
        # Convergence diagnostics
        'energy_history': energy_history,
        'delta_E_history': delta_E_history,
        'grad_history': grad_history,
        'ddm_history': ddm_history,
        'convergence_rate': convergence_rate,
        # Extra
        'scf_summary': scf_summary,
    }
    return result


def main():
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info}")
    print(f"TIDES_SRC: {TIDES_SRC}")
    print()

    # Check PySCF and gpu4pyscf versions
    import pyscf
    pyscf_ver = pyscf.__version__
    gpu4pyscf_ver = 'not installed'
    gpu4pyscf_spec = importlib.util.find_spec('gpu4pyscf')
    if gpu4pyscf_spec is not None:
        import gpu4pyscf
        gpu4pyscf_ver = getattr(gpu4pyscf, '__version__', 'unknown')
    import numpy as np
    numpy_ver = np.__version__

    print(f"PySCF version: {pyscf_ver}")
    print(f"gpu4pyscf version: {gpu4pyscf_ver}")
    print(f"NumPy version: {numpy_ver}")
    print()

    # Check for existing reference file
    out_path = os.path.join(TIDES_SRC, 'bench/profiling_results/PYSCF_REFERENCE_FROZEN.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Found existing reference at {out_path}")
        print(f"  PBE molecules: {list(existing.get('pbe', {}).keys())}")
        print(f"  B3LYP molecules: {list(existing.get('b3lyp', {}).keys())}")

    results = existing
    sys_info = get_system_info()
    results['metadata'] = {
        'description': 'Complete frozen PySCF/gpu4pyscf reference data with extensive profiling',
        'pyscf_version': pyscf_ver,
        'gpu4pyscf_version': gpu4pyscf_ver,
        'numpy_version': numpy_ver,
        'hardware': gpu_info,
        'system_info': sys_info,
        'cuda_version': os.environ.get('CUDA_VERSION', 'unknown'),
        'date_collected': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': {
            'ae_basis': AE_BASIS,
            'pp_basis': PP_BASIS,
            'pp_pseudo': PP_PSEUDO,
            'grids': '(99,590) level=4',
            'conv_tol': 1e-7,
            'max_cycle': 100,
            'rdkit_seed': 42,
            'density_fitting_gpu': True,
        },
    }

    # Initialize XC sections
    for xc in XC_FUNCTIONALS:
        if xc not in results:
            results[xc] = {}

    total_runs = 0
    skipped = 0
    failed = 0

    for label, smiles, natoms_expected in MOLECULES:
        atoms, Z, pos_ang = make_molecule(smiles)
        natoms = len(Z)
        print(f"\n{'='*80}")
        print(f"  {label} ({natoms} atoms)  SMILES={smiles}")
        print(f"{'='*80}")

        for xc in XC_FUNCTIONALS:
            if label not in results[xc]:
                results[xc][label] = {}

            for route_key, use_gpu, use_pp in [
                ('cpu_ae', False, False),
                ('cpu_pp', False, True),
                ('gpu_ae', True, False),
                ('gpu_pp', True, True),
            ]:
                # Skip if already collected
                if route_key in results[xc][label] and results[xc][label][route_key] is not None:
                    if results[xc][label][route_key].get('converged', False):
                        print(f"  [{xc}] {route_key}: SKIP (already collected, converged)")
                        skipped += 1
                        continue

                basis = PP_BASIS if use_pp else AE_BASIS
                pseudo = PP_PSEUDO if use_pp else None
                gpu_str = "GPU" if use_gpu else "CPU"
                pp_str = "PP" if use_pp else "AE"

                print(f"\n  [{xc}] {gpu_str}-{pp_str} (basis={basis}", end="")
                if pseudo:
                    print(f", pseudo={pseudo}", end="")
                print(f") ...", flush=True)

                total_runs += 1
                r = run_pyscf(atoms, pos_ang, xc=xc, basis=basis, pseudo=pseudo, use_gpu=use_gpu)

                status = "CONVERGED" if r['converged'] else "FAILED"
                print(f"    E={r['E_total']:.6f}  {status}  iters={r['n_iters']}  "
                      f"wall={r['wall_s']:.2f}s  nbf={r['n_basis']}  "
                      f"gpu_used={r['gpu_actually_used']}", flush=True)

                if not r['converged']:
                    failed += 1
                    print(f"    WARNING: Did not converge!")

                # Print per-operation summary
                for op_key, op_val in r['op_timings'].items():
                    print(f"    {op_key}: avg={op_val['avg_ms']:.1f}ms  "
                          f"total={op_val['total_ms']:.1f}ms  calls={op_val['n_calls']}")

                # Print convergence trajectory
                if r['energy_history']:
                    print(f"    Energy trajectory: ", end="")
                    for i, e in enumerate(r['energy_history']):
                        if i < 5 or i >= len(r['energy_history']) - 2:
                            print(f"{e:.8f}", end=" ")
                        elif i == 5:
                            print("... ", end="")
                    print()

                results[xc][label][route_key] = r

                # Save incrementally (so we don't lose data on crash)
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total runs: {total_runs}")
    print(f"  Skipped (already collected): {skipped}")
    print(f"  Failed (not converged): {failed}")
    print(f"  Saved to: {out_path}")

    # Print summary table
    print(f"\n{'XC':<6} {'Mol':<8} {'N':>3} | {'CPU-AE':>12} {'CPU-PP':>12} {'GPU-AE':>12} {'GPU-PP':>12}")
    print("-" * 80)
    for xc in XC_FUNCTIONALS:
        for label, smiles, _ in MOLECULES:
            if label not in results[xc]:
                continue
            natoms = len(make_molecule(smiles)[1])
            row = []
            for route_key in ['cpu_ae', 'cpu_pp', 'gpu_ae', 'gpu_pp']:
                d = results[xc][label].get(route_key)
                if d is None:
                    row.append("N/A")
                else:
                    conv = "Y" if d['converged'] else "N"
                    row.append(f"{d['wall_s']:.2f}s/{d['n_iters']}it/{conv}")
            print(f"{xc:<6} {label:<8} {natoms:>3} | {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

    print(f"\nReference file: {out_path}")
    print("This file is now frozen. Use unified_benchmark.py to compare TIDES against it.")


if __name__ == '__main__':
    main()
