#!/usr/bin/env python3
"""Unified TIDES benchmark — runs all 4 routes, compares against frozen PySCF reference.

Loads PYSCF_REFERENCE_FROZEN.json (never runs PySCF).
Runs TIDES for: AE-GPU, AE-CPU, PP-GPU, PP-CPU x PBE + B3LYP x molecule ladder.
Outputs 9 dashboard tables per XC functional + JSON with full side-by-side data.

Usage:
  LD_PRELOAD=<mkl preload> python3 bench/unified_benchmark.py [--xc pbe,b3lyp] [--molecules CH4,H2O,...]
"""
import sys, os, time, json, argparse, math
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
os.environ['TIDES_SRC_DIR'] = TIDES_SRC
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))

BOHR = 1.8897261254535
PP_DIR = os.path.join(TIDES_SRC, 'external/pseudopotentials/pseudodojo-pbe-sr')
REFERENCE_PATH = os.path.join(TIDES_SRC, 'bench/profiling_results/PYSCF_REFERENCE_FROZEN.json')

# TIDES GPU VRAM limit: GPU alloc fails at ~96 basis functions on RTX 3060 (12GB)
# Beyond this, TIDES falls back to CPU silently — skip GPU routes to keep comparison honest
TIDES_GPU_MAX_BASIS = 90

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

ALL_XC = ['pbe', 'b3lyp']


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


def level_shift_for(natoms, xc):
    if xc == 'b3lyp':
        if natoms > 20:
            return 0.3
        if natoms > 4:
            return 0.2
        return 0.0
    return 0.0


def max_iter_for(natoms, is_pp, xc):
    if xc == 'b3lyp':
        if is_pp and natoms >= 12:
            return 250
        if natoms > 20:
            return 120
        return 100
    return 80


def run_tides(Z, pos, xc='pbe', use_pp=True, use_gpu=True, natoms=5, label=''):
    """Run TIDES with full profiling data extraction."""
    import tides._native as native

    if use_gpu:
        os.environ.pop('TIDES_DISABLE_GPU', None)
    else:
        os.environ['TIDES_DISABLE_GPU'] = '1'

    ls = level_shift_for(natoms, xc)
    if ls > 0:
        os.environ['TIDES_LEVEL_SHIFT'] = str(ls)
    else:
        os.environ.pop('TIDES_LEVEL_SHIFT', None)

    mi = max_iter_for(natoms, is_pp=use_pp, xc=xc)

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    t0 = time.time()
    r = native.NaoDriver.run(
        atomic_numbers=Z, positions=pos,
        grid_h=0.3, grid_margin=6.0,
        max_iter=mi, tol=1e-7,
        use_dual_grid=False, xc_functional=xc,
        pp_dir=PP_DIR if use_pp else '/nonexistent',
        allow_grid_refine=False,
    )
    wall = time.time() - t0

    resource_stats = monitor.stop()

    os.environ.pop('TIDES_DISABLE_GPU', None)
    os.environ.pop('TIDES_LEVEL_SHIFT', None)

    bh = r.build_H_timings
    scf = r.scf
    scf_t = scf.timings

    energy_history = list(scf.energy_history) if hasattr(scf, 'energy_history') else []

    delta_E_history = []
    for i in range(1, len(energy_history)):
        delta_E_history.append(energy_history[i] - energy_history[i - 1])

    convergence_rate = 0.0
    valid_deltas = [(i, abs(d)) for i, d in enumerate(delta_E_history) if abs(d) > 1e-15]
    if len(valid_deltas) >= 3:
        log_deltas = [(i, math.log(d)) for i, d in valid_deltas if d > 0]
        if len(log_deltas) >= 3:
            n_pts = len(log_deltas)
            t_mean = sum(t for t, _ in log_deltas) / n_pts
            v_mean = sum(v for _, v in log_deltas) / n_pts
            num = sum((t - t_mean) * (v - v_mean) for t, v in log_deltas)
            den = sum((t - t_mean) ** 2 for t, _ in log_deltas)
            convergence_rate = num / den if den > 0 else 0.0

    result = {
        'engine': 'tides_gpu' if use_gpu else 'tides_cpu',
        'basis': 'NAO-DZP',
        'pseudo': 'PseudoDojo' if use_pp else 'none',
        'xc': xc,
        'E_total': float(r.energy.E_total),
        'converged': bool(scf.converged),
        'n_iters': int(bh.n_iterations),
        'wall_s': round(wall, 4),
        'conv_tol': 1e-7,
        'n_basis': int(r.n_basis),
        'n_electrons': int(r.n_electrons),
        'n_atoms': int(r.n_atoms),
        'grid_h': float(r.grid_h),
        'grid_n': list(r.grid_n) if hasattr(r, 'grid_n') else [0, 0, 0],
        'gpu_pipeline': bool(bh.used_gpu_pipeline),
        'level_shift': ls,
        'max_iter': mi,
        'resource_usage': resource_stats,
        'build_H_timings': {
            'quantize_P_ms': float(bh.quantize_P_ms),
            'rho_build_ms': float(bh.rho_build_ms),
            'poisson_ms': float(bh.poisson_ms),
            'xc_eval_ms': float(bh.xc_eval_ms),
            'vmat_build_ms': float(bh.vmat_build_ms),
            'assemble_H_ms': float(bh.assemble_H_ms),
            'total_ms': float(bh.total_ms),
            'n_iterations': int(bh.n_iterations),
            'used_gpu_pipeline': bool(bh.used_gpu_pipeline),
        },
        'poisson_substeps': {
            'memset_ms': float(bh.poisson_memset_ms),
            'zeropad_ms': float(bh.poisson_zeropad_ms),
            'fft_fwd_ms': float(bh.poisson_fft_fwd_ms),
            'multiply_ms': float(bh.poisson_multiply_ms),
            'fft_inv_ms': float(bh.poisson_fft_inv_ms),
            'extract_ms': float(bh.poisson_extract_ms),
            'energy_ms': float(bh.poisson_energy_ms),
            'fft_dims': [int(bh.poisson_fft_n0), int(bh.poisson_fft_n1), int(bh.poisson_fft_n2)],
        },
        'scf_timings': {
            'build_H_ms': float(scf_t.build_H_ms),
            'gemm_hx_ms': float(scf_t.gemm_hx_ms),
            'gemm_xthp_ms': float(scf_t.gemm_xthp_ms),
            'eigensolve_ms': float(scf_t.eigensolve_ms),
            'backtransform_ms': float(scf_t.backtransform_ms),
            'dsyrk_ms': float(scf_t.dsyrk_ms),
            'energy_ms': float(scf_t.energy_ms),
            'diis_ms': float(scf_t.diis_ms),
            'scf_total_ms': float(scf_t.scf_total_ms),
            'n_iterations': int(scf_t.n_iterations),
        },
        'energy_history': [float(e) for e in energy_history],
        'delta_E_history': [float(d) for d in delta_E_history],
        'convergence_rate': convergence_rate,
        'avg_iter_wall_ms': float(scf_t.scf_total_ms) / max(int(scf_t.n_iterations), 1),
    }
    return result


def load_reference():
    if not os.path.exists(REFERENCE_PATH):
        print(f"ERROR: Reference file not found: {REFERENCE_PATH}")
        print("Run collect_pyscf_reference.py first to generate it.")
        sys.exit(1)
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def fmt_wall(wall_s):
    if wall_s is None or wall_s == 0:
        return "  N/A"
    return f"{wall_s:>7.2f}s"


def fmt_iters(n):
    if n is None:
        return " N/A"
    return f"{n:>4}"


def fmt_conv(conv):
    if conv is None:
        return " N/A"
    return " YES" if conv else " NO "


def fmt_speedup(tides_wall, pyscf_wall):
    if tides_wall is None or pyscf_wall is None or tides_wall == 0:
        return "  N/A"
    ratio = pyscf_wall / tides_wall
    return f"{ratio:>5.2f}x"


def fmt_ms(ms):
    if ms is None or ms == 0:
        return "    N/A"
    return f"{ms:>7.1f}"


def fmt_pct(num, denom):
    if denom is None or denom == 0 or num is None:
        return "  N/A"
    return f"{num / denom * 100:>5.1f}%"


def print_dashboard(xc, tides_results, pyscf_ref):
    """Print all 7 dashboard tables for one XC functional."""
    print(f"\n{'='*140}")
    print(f"  UNIFIED BENCHMARK DASHBOARD — {xc.upper()}")
    print(f"{'='*140}")

    molecules = [m[0] for m in MOLECULES if m[0] in tides_results]

    # ── Table 1: Speed (warm) ──
    print(f"\n{'='*140}")
    print("TABLE 1 — SPEED (wall time, seconds)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'TIDES AE-GPU':>12} {'PySCF GPU-AE':>12} {'speedup':>8} | "
          f"{'TIDES PP-GPU':>12} {'PySCF GPU-PP':>12} {'speedup':>8} | "
          f"{'TIDES AE-CPU':>12} {'PySCF CPU-AE':>12} {'speedup':>8} | "
          f"{'TIDES PP-CPU':>12} {'PySCF CPU-PP':>12} {'speedup':>8}")
    print("-" * 140)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        parts = [f"{label:<8} {natoms:>3}"]
        for tides_key, pyscf_key in [
            ('ae_gpu', 'gpu_ae'), ('pp_gpu', 'gpu_pp'),
            ('ae_cpu', 'cpu_ae'), ('pp_cpu', 'cpu_pp'),
        ]:
            t = tides_results[label].get(tides_key, {})
            p = pyscf_ref.get(label, {}).get(pyscf_key, {})
            tw = t.get('wall_s', 0)
            pw = p.get('wall_s', 0)
            sp = fmt_speedup(tw, pw) if tw and pw else "  N/A"
            parts.append(f"{fmt_wall(tw)} {fmt_wall(pw)} {sp:>8}")
        print(f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]}")

    # ── Table 2: Convergence ──
    print(f"\n{'='*140}")
    print("TABLE 2 — CONVERGENCE")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'AE-GPU':>16} {'PP-GPU':>16} | "
          f"{'AE-CPU':>16} {'PP-CPU':>16}")
    print("-" * 100)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        parts = [f"{label:<8} {natoms:>3}"]
        for tides_key, pyscf_key in [
            ('ae_gpu', 'gpu_ae'), ('pp_gpu', 'gpu_pp'),
            ('ae_cpu', 'cpu_ae'), ('pp_cpu', 'cpu_pp'),
        ]:
            t = tides_results[label].get(tides_key, {})
            p = pyscf_ref.get(label, {}).get(pyscf_key, {})
            tc = fmt_conv(t.get('converged'))
            pc = fmt_conv(p.get('converged'))
            parts.append(f"T={tc} P={pc}")
        print(f"{parts[0]} | {parts[1]:>16} {parts[2]:>16} | {parts[3]:>16} {parts[4]:>16}")

    # ── Table 3: Iterations ──
    print(f"\n{'='*140}")
    print("TABLE 3 — SCF ITERATIONS")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'AE-GPU':>20} {'PP-GPU':>20} | "
          f"{'AE-CPU':>20} {'PP-CPU':>20}")
    print("-" * 110)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        parts = [f"{label:<8} {natoms:>3}"]
        for tides_key, pyscf_key in [
            ('ae_gpu', 'gpu_ae'), ('pp_gpu', 'gpu_pp'),
            ('ae_cpu', 'cpu_ae'), ('pp_cpu', 'cpu_pp'),
        ]:
            t = tides_results[label].get(tides_key, {})
            p = pyscf_ref.get(label, {}).get(pyscf_key, {})
            ti = t.get('n_iters', 0)
            pi = p.get('n_iters', 0)
            ratio = f"{ti / pi:.1f}x" if pi and ti else "N/A"
            parts.append(f"T={fmt_iters(ti)} P={fmt_iters(pi)} {ratio:>6}")
        print(f"{parts[0]} | {parts[1]:>20} {parts[2]:>20} | {parts[3]:>20} {parts[4]:>20}")

    # ── Table 4: Per-iteration cost (ms/iter) ──
    print(f"\n{'='*140}")
    print("TABLE 4 — PER-ITERATION COST (ms/iter, averaged)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'TIDES AE-GPU':>40} | {'PySCF GPU-AE':>30} | ratio")
    print(f"{'':>12} | "
          f"{'build_H':>8} {'eig':>8} {'diis':>8} {'total':>8} | "
          f"{'veff':>8} {'eig':>8} {'total':>8} | "
          f"{'iter':>6}")
    print("-" * 120)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        t = tides_results[label].get('ae_gpu', {})
        p = pyscf_ref.get(label, {}).get('gpu_ae', {})

        t_bh = t.get('build_H_timings', {})
        t_scf = t.get('scf_timings', {})
        t_total = t_scf.get('scf_total_ms', 0)

        p_ops = p.get('op_timings', {})
        p_veff = p_ops.get('veff_ms', {}).get('avg_ms', 0)
        p_eig = p_ops.get('eigensolve_ms', {}).get('avg_ms', 0)
        p_total = 0
        for op_key, op_val in p_ops.items():
            p_total += op_val.get('avg_ms', 0)

        ratio = f"{t_total / p_total:.2f}x" if p_total and t_total else "N/A"

        print(f"{label:<8} {natoms:>3} | "
              f"{fmt_ms(t_bh.get('total_ms', 0))} {fmt_ms(t_scf.get('eigensolve_ms', 0))} "
              f"{fmt_ms(t_scf.get('diis_ms', 0))} {fmt_ms(t_total)} | "
              f"{fmt_ms(p_veff)} {fmt_ms(p_eig)} {fmt_ms(p_total)} | "
              f"{ratio:>6}")

    # ── Table 5: Operation breakdown (% of SCF) ──
    print(f"\n{'='*140}")
    print("TABLE 5 — OPERATION BREAKDOWN (% of SCF time, AE-GPU vs GPU-AE)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'TIDES':>50} | {'PySCF':>50}")
    print(f"{'':>12} | "
          f"{'rho%':>6} {'pois%':>6} {'xc%':>6} {'vmat%':>6} {'eig%':>6} {'diis%':>6} {'other%':>6} | "
          f"{'hcore%':>6} {'veff%':>6} {'eig%':>6} {'rdm1%':>6} {'etot%':>6} {'other%':>6}")
    print("-" * 140)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        t = tides_results[label].get('ae_gpu', {})
        p = pyscf_ref.get(label, {}).get('gpu_ae', {})

        t_bh = t.get('build_H_timings', {})
        t_scf = t.get('scf_timings', {})
        t_total = t_scf.get('scf_total_ms', 0)

        t_rho = t_bh.get('rho_build_ms', 0)
        t_pois = t_bh.get('poisson_ms', 0)
        t_xc = t_bh.get('xc_eval_ms', 0)
        t_vmat = t_bh.get('vmat_build_ms', 0)
        t_eig = t_scf.get('eigensolve_ms', 0)
        t_diis = t_scf.get('diis_ms', 0)
        t_known = t_rho + t_pois + t_xc + t_vmat + t_eig + t_diis
        t_other = t_total - t_known

        p_ops = p.get('op_timings', {})
        p_total = sum(op.get('avg_ms', 0) for op in p_ops.values())
        p_hcore = p_ops.get('hcore_ms', {}).get('avg_ms', 0)
        p_veff = p_ops.get('veff_ms', {}).get('avg_ms', 0)
        p_eig = p_ops.get('eigensolve_ms', {}).get('avg_ms', 0)
        p_rdm1 = p_ops.get('make_rdm1_ms', {}).get('avg_ms', 0)
        p_etot = p_ops.get('energy_tot_ms', {}).get('avg_ms', 0)
        p_known = p_hcore + p_veff + p_eig + p_rdm1 + p_etot
        p_other = p_total - p_known

        print(f"{label:<8} {natoms:>3} | "
              f"{fmt_pct(t_rho, t_total)} {fmt_pct(t_pois, t_total)} {fmt_pct(t_xc, t_total)} "
              f"{fmt_pct(t_vmat, t_total)} {fmt_pct(t_eig, t_total)} {fmt_pct(t_diis, t_total)} "
              f"{fmt_pct(t_other, t_total)} | "
              f"{fmt_pct(p_hcore, p_total)} {fmt_pct(p_veff, p_total)} {fmt_pct(p_eig, p_total)} "
              f"{fmt_pct(p_rdm1, p_total)} {fmt_pct(p_etot, p_total)} {fmt_pct(p_other, p_total)}")

    # ── Table 6: Convergence trajectory ──
    print(f"\n{'='*140}")
    print("TABLE 6 — CONVERGENCE TRAJECTORY (energy per iteration, AE-GPU vs GPU-AE)")
    print(f"{'='*140}")
    for label in molecules:
        t = tides_results[label].get('ae_gpu', {})
        p = pyscf_ref.get(label, {}).get('gpu_ae', {})
        t_hist = t.get('energy_history', [])
        p_hist = p.get('energy_history', [])
        natoms = t.get('n_atoms', 0)

        print(f"\n  {label} ({natoms} atoms):")
        print(f"    TIDES ({len(t_hist)} iters):  ", end="")
        for i, e in enumerate(t_hist):
            if i < 5 or i >= len(t_hist) - 2:
                print(f"{e:.10f}", end="  ")
            elif i == 5:
                print("...  ", end="")
        print()
        print(f"    PySCF ({len(p_hist)} iters):  ", end="")
        for i, e in enumerate(p_hist):
            if i < 5 or i >= len(p_hist) - 2:
                print(f"{e:.10f}", end="  ")
            elif i == 5:
                print("...  ", end="")
        print()

        t_deltas = t.get('delta_E_history', [])
        p_deltas = p.get('delta_E_history', [])
        if t_deltas:
            print(f"    TIDES |dE|:  ", end="")
            for i, d in enumerate(t_deltas):
                if abs(d) > 0:
                    print(f"{abs(d):.2e}", end="  ")
            print()
        if p_deltas:
            print(f"    PySCF |dE|:  ", end="")
            for i, d in enumerate(p_deltas):
                if abs(d) > 0:
                    print(f"{abs(d):.2e}", end="  ")
            print()

    # ── Table 7: Full profiling detail ──
    print(f"\n{'='*140}")
    print("TABLE 7 — FULL PROFILING DETAIL (AE-GPU vs GPU-AE, ms/iter)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'TIDES substeps':>60} | {'PySCF operations':>50}")
    print(f"{'':>12} | "
          f"{'quant':>6} {'rho':>6} {'pois':>6} {'xc':>6} {'vmat':>6} {'asmbl':>6} "
          f"{'eig':>6} {'diis':>6} {'total':>6} | "
          f"{'hcore':>6} {'veff':>6} {'eig':>6} {'rdm1':>6} {'etot':>6} {'total':>6}")
    print("-" * 140)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0)
        t = tides_results[label].get('ae_gpu', {})
        p = pyscf_ref.get(label, {}).get('gpu_ae', {})

        t_bh = t.get('build_H_timings', {})
        t_scf = t.get('scf_timings', {})
        p_ops = p.get('op_timings', {})

        p_hcore = p_ops.get('hcore_ms', {}).get('avg_ms', 0)
        p_veff = p_ops.get('veff_ms', {}).get('avg_ms', 0)
        p_eig = p_ops.get('eigensolve_ms', {}).get('avg_ms', 0)
        p_rdm1 = p_ops.get('make_rdm1_ms', {}).get('avg_ms', 0)
        p_etot = p_ops.get('energy_tot_ms', {}).get('avg_ms', 0)
        p_total = sum(op.get('avg_ms', 0) for op in p_ops.values())

        print(f"{label:<8} {natoms:>3} | "
              f"{fmt_ms(t_bh.get('quantize_P_ms', 0))} {fmt_ms(t_bh.get('rho_build_ms', 0))} "
              f"{fmt_ms(t_bh.get('poisson_ms', 0))} {fmt_ms(t_bh.get('xc_eval_ms', 0))} "
              f"{fmt_ms(t_bh.get('vmat_build_ms', 0))} {fmt_ms(t_bh.get('assemble_H_ms', 0))} "
              f"{fmt_ms(t_scf.get('eigensolve_ms', 0))} {fmt_ms(t_scf.get('diis_ms', 0))} "
              f"{fmt_ms(t_scf.get('scf_total_ms', 0))} | "
              f"{fmt_ms(p_hcore)} {fmt_ms(p_veff)} {fmt_ms(p_eig)} {fmt_ms(p_rdm1)} "
              f"{fmt_ms(p_etot)} {fmt_ms(p_total)}")

    # ── Table 8: Resource usage ──
    print(f"\n{'='*140}")
    print("TABLE 8 — RESOURCE USAGE (CPU%, RAM, GPU%, VRAM, power, temp)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} | "
          f"{'TIDES AE-GPU':>30} | {'PySCF GPU-AE':>30} | "
          f"{'TIDES PP-GPU':>30} | {'PySCF GPU-PP':>30} | "
          f"{'TIDES AE-CPU':>30} | {'PySCF CPU-AE':>30}")
    print(f"{'':>12} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5} | "
          f"{'CPU%':>5} {'RAM':>6} {'GPU%':>5} {'VRAM':>6} {'W':>5}")
    print("-" * 210)
    for label in molecules:
        natoms = tides_results[label].get('ae_gpu', {}).get('n_atoms', 0) or \
                 tides_results[label].get('ae_cpu', {}).get('n_atoms', 0)
        parts = [f"{label:<8} {natoms:>3}"]
        for tides_key, pyscf_key in [
            ('ae_gpu', 'gpu_ae'), ('pp_gpu', 'gpu_pp'), ('ae_cpu', 'cpu_ae'),
        ]:
            # TIDES
            t = tides_results[label].get(tides_key, {})
            if t.get('skipped_reason') == 'GPU_OOM':
                parts.append(f"{'GPU_OOM':>30}")
            else:
                tr = t.get('resource_usage', {})
                t_cpu = tr.get('proc_cpu_pct_peak', 0)
                t_ram = tr.get('proc_rss_mb_peak', 0)
                t_gpu = tr.get('gpu_util_pct_avg', 0)
                t_vram = tr.get('vram_used_mb_peak', 0)
                t_w = tr.get('gpu_power_w_avg', 0)
                parts.append(f"{t_cpu:>5.0f} {t_ram:>6.0f} {t_gpu:>5.0f} {t_vram:>6.0f} {t_w:>5.1f}")
            # PySCF
            p = pyscf_ref.get(label, {}).get(pyscf_key, {})
            pr = p.get('resource_usage', {})
            p_cpu = pr.get('proc_cpu_pct_peak', 0)
            p_ram = pr.get('proc_rss_mb_peak', 0)
            p_gpu = pr.get('gpu_util_pct_avg', 0)
            p_vram = pr.get('vram_used_mb_peak', 0)
            p_w = pr.get('gpu_power_w_avg', 0)
            parts.append(f"{p_cpu:>5.0f} {p_ram:>6.0f} {p_gpu:>5.0f} {p_vram:>6.0f} {p_w:>5.1f}")
        print(f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} | {parts[5]} | {parts[6]}")

    # ── Table 9: GPU VRAM limitation analysis ──
    print(f"\n{'='*140}")
    print("TABLE 9 — GPU VRAM LIMITATION ANALYSIS (TIDES vs PySCF on this hardware)")
    print(f"{'='*140}")
    print(f"{'Mol':<8} {'N':>3} {'nbf_AE':>6} {'nbf_PP':>6} | "
          f"{'TIDES AE-GPU':>20} {'TIDES PP-GPU':>20} | "
          f"{'PySCF GPU-AE':>20} {'PySCF GPU-PP':>20} | "
          f"{'VRAM gap':>20}")
    print("-" * 140)
    for label in molecules:
        t_ae = tides_results[label].get('ae_gpu', {})
        t_pp = tides_results[label].get('pp_gpu', {})
        p_ae = pyscf_ref.get(label, {}).get('gpu_ae', {})
        p_pp = pyscf_ref.get(label, {}).get('gpu_pp', {})
        nbf_ae = p_ae.get('n_basis', 0)
        nbf_pp = p_pp.get('n_basis', 0)
        natoms = t_ae.get('n_atoms', 0) or tides_results[label].get('ae_cpu', {}).get('n_atoms', 0)

        t_ae_status = 'GPU_OOM' if t_ae.get('skipped_reason') == 'GPU_OOM' else \
                      ('CONV' if t_ae.get('converged') else 'FAIL')
        t_pp_status = 'GPU_OOM' if t_pp.get('skipped_reason') == 'GPU_OOM' else \
                      ('CONV' if t_pp.get('converged') else 'FAIL')
        p_ae_status = 'CONV' if p_ae.get('converged') else 'FAIL'
        p_pp_status = 'CONV' if p_pp.get('converged') else 'FAIL'

        p_ae_vram = p_ae.get('resource_usage', {}).get('vram_used_mb_peak', 0)
        p_pp_vram = p_pp.get('resource_usage', {}).get('vram_used_mb_peak', 0)
        t_ae_vram = t_ae.get('resource_usage', {}).get('vram_used_mb_peak', 0) if t_ae_status != 'GPU_OOM' else 0

        gap = f"PySCF {p_ae_vram:.0f}MB vs TIDES {t_ae_vram:.0f}MB" if t_ae_vram else \
              f"PySCF {p_ae_vram:.0f}MB vs TIDES OOM"

        print(f"{label:<8} {natoms:>3} {nbf_ae:>6} {nbf_pp:>6} | "
              f"{t_ae_status:>20} {t_pp_status:>20} | "
              f"{p_ae_status:>20} {p_pp_status:>20} | "
              f"{gap:>20}")


def main():
    ap = argparse.ArgumentParser(description='Unified TIDES benchmark vs frozen PySCF reference')
    ap.add_argument('--xc', default='pbe,b3lyp', help='comma list: pbe,b3lyp')
    ap.add_argument('--molecules', default='', help='comma list (default: all)')
    ap.add_argument('--routes', default='all', help='all or comma list: ae_gpu,ae_cpu,pp_gpu,pp_cpu')
    args = ap.parse_args()

    xc_list = [x.strip() for x in args.xc.split(',')]
    mol_labels = [x.strip() for x in args.molecules.split(',')] if args.molecules else None
    route_list = None
    if args.routes != 'all':
        route_list = [x.strip() for x in args.routes.split(',')]

    print("Loading frozen PySCF reference...")
    ref = load_reference()
    print(f"  PBE molecules: {list(ref.get('pbe', {}).keys())}")
    print(f"  B3LYP molecules: {list(ref.get('b3lyp', {}).keys())}")
    print(f"  Reference date: {ref.get('metadata', {}).get('date_collected', 'unknown')}")
    print()

    all_routes = ['ae_gpu', 'pp_gpu', 'ae_cpu', 'pp_cpu']
    active_routes = route_list if route_list else all_routes

    molecules = [(l, s, n) for l, s, n in MOLECULES
                 if mol_labels is None or l in mol_labels]

    all_results = {}

    for xc in xc_list:
        print(f"\n{'#'*80}")
        print(f"# Running TIDES — XC={xc.upper()}")
        print(f"{'#'*80}")

        pyscf_ref = ref.get(xc, {})
        all_results[xc] = {}

        for label, smiles, _ in molecules:
            Z, pos = make_molecule(smiles)
            natoms = len(Z)
            all_results[xc][label] = {}

            for route_key in active_routes:
                use_pp = 'pp' in route_key
                use_gpu = 'gpu' in route_key
                pp_str = "PP" if use_pp else "AE"
                gpu_str = "GPU" if use_gpu else "CPU"

                # Check if GPU route would exceed VRAM limit (skip to avoid CPU fallback)
                if use_gpu:
                    # Estimate basis size from PySCF reference
                    p_ref = pyscf_ref.get(label, {})
                    ref_key = 'gpu_pp' if use_pp else 'gpu_ae'
                    est_nbf = p_ref.get(ref_key, {}).get('n_basis', 0)
                    if est_nbf > TIDES_GPU_MAX_BASIS:
                        print(f"\n  [{xc.upper()}] {label} ({natoms} atoms) — {gpu_str}-{pp_str}: SKIPPED (GPU OOM, nbf={est_nbf} > {TIDES_GPU_MAX_BASIS})", flush=True)
                        all_results[xc][label][route_key] = {
                            'engine': 'tides',
                            'E_total': 0.0,
                            'converged': False,
                            'n_iters': 0,
                            'wall_s': 0.0,
                            'conv_tol': 1e-7,
                            'n_basis': est_nbf,
                            'n_electrons': 0,
                            'n_atoms': natoms,
                            'gpu_pipeline': False,
                            'level_shift': 0,
                            'max_iter': 0,
                            'resource_usage': {'error': 'GPU OOM — skipped'},
                            'build_H_timings': {},
                            'poisson_substeps': {},
                            'scf_timings': {},
                            'energy_history': [],
                            'delta_E_history': [],
                            'convergence_rate': 0,
                            'avg_iter_wall_ms': 0,
                            'skipped_reason': 'GPU_OOM',
                        }
                        continue

                print(f"\n  [{xc.upper()}] {label} ({natoms} atoms) — {gpu_str}-{pp_str}", flush=True)
                r = run_tides(Z, pos, xc=xc, use_pp=use_pp, use_gpu=use_gpu,
                              natoms=natoms, label=label)

                status = "CONVERGED" if r['converged'] else "FAILED"
                print(f"    E={r['E_total']:.6f}  {status}  iters={r['n_iters']}  "
                      f"wall={r['wall_s']:.2f}s  nbf={r['n_basis']}  "
                      f"gpu={r['gpu_pipeline']}", flush=True)

                bh = r['build_H_timings']
                print(f"    build_H: rho={bh['rho_build_ms']:.1f}ms  pois={bh['poisson_ms']:.1f}ms  "
                      f"xc={bh['xc_eval_ms']:.1f}ms  vmat={bh['vmat_build_ms']:.1f}ms  "
                      f"total={bh['total_ms']:.1f}ms")

                st = r['scf_timings']
                print(f"    SCF: eig={st['eigensolve_ms']:.1f}ms  diis={st['diis_ms']:.1f}ms  "
                      f"total={st['scf_total_ms']:.1f}ms/iter")

                if r['energy_history']:
                    print(f"    Energy: ", end="")
                    for i, e in enumerate(r['energy_history']):
                        if i < 3 or i >= len(r['energy_history']) - 1:
                            print(f"{e:.10f}", end="  ")
                        elif i == 3:
                            print("...  ", end="")
                    print()

                all_results[xc][label][route_key] = r

            # Per-molecule comparison
            print(f"\n  COMPARISON {label} [{xc.upper()}]:")
            pyscf_route_map = {
                'ae_gpu': 'gpu_ae', 'pp_gpu': 'gpu_pp',
                'ae_cpu': 'cpu_ae', 'pp_cpu': 'cpu_pp',
            }
            for route_key in active_routes:
                t = all_results[xc][label].get(route_key, {})
                p = pyscf_ref.get(label, {}).get(pyscf_route_map.get(route_key, ''), {})

                if t.get('skipped_reason') == 'GPU_OOM':
                    print(f"    {route_key:>8}: TIDES GPU_OOM (skipped, nbf={t.get('n_basis',0)})  "
                          f"PySCF {p.get('wall_s',0):.2f}s/{p.get('n_iters',0)}it/conv={p.get('converged',False)}  "
                          f"[PySCF GPU handles this — TIDES VRAM limit]")
                    continue

                tw = t.get('wall_s', 0)
                pw = p.get('wall_s', 0)
                ti = t.get('n_iters', 0)
                pi = p.get('n_iters', 0)
                tc = t.get('converged', False)
                pc = p.get('converged', False)
                sp = f"{pw / tw:.2f}x" if tw and pw else "N/A"

                print(f"    {route_key:>8}: TIDES {tw:.2f}s/{ti}it/conv={tc}  "
                      f"PySCF {pw:.2f}s/{pi}it/conv={pc}  speedup={sp}")

        # Print dashboard for this XC
        print_dashboard(xc, all_results[xc], pyscf_ref)

    # Save JSON output
    out_path = os.path.join(TIDES_SRC, 'bench/profiling_results',
                            f"unified_benchmark_{time.strftime('%Y-%m-%d_%H%M')}.json")
    sys_info = get_system_info()
    output = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tides_src': TIDES_SRC,
            'reference_file': REFERENCE_PATH,
            'xc_functionals': xc_list,
            'molecules': [m[0] for m in molecules],
            'system_info': sys_info,
        },
        'tides': all_results,
        'pyscf_reference': {xc: ref.get(xc, {}) for xc in xc_list},
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
