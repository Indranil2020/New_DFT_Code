#!/usr/bin/env python3
"""
TIDES vs PySCF Benchmark Script

Compares TIDES CPU reference implementations against PySCF for:
  1. He atom LDA total energy
  2. Ne atom LDA total energy
  3. H2O molecule LDA total energy + forces
  4. H2 molecule dissociation curve (5 points)
  5. Timing comparison for each system

Outputs a structured benchmark report to stdout and JSON ledger.
"""

import json
import time
import sys
from pathlib import Path

# PySCF imports
from pyscf import gto, dft, scf as pyscf_scf
from pyscf import grad as pyscf_grad

def run_pyscf_atom(symbol, basis='ccpvdz', xc='LDA,VWN'):
    """Run PySCF atomic LDA calculation."""
    mol = gto.M(atom=symbol, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    t0 = time.perf_counter()
    e = mf.kernel()
    t1 = time.perf_counter()
    return {
        'system': f"{symbol} atom",
        'method': 'PySCF RKS LDA',
        'basis': basis,
        'energy_Ha': e,
        'time_ms': (t1 - t0) * 1000,
        'n_atoms': 1,
    }

def run_pyscf_molecule(atoms, basis='ccpvdz', xc='LDA,VWN'):
    """Run PySCF molecular LDA calculation with forces."""
    mol = gto.M(atom=atoms, basis=basis, verbose=0, unit='Angstrom')
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    t0 = time.perf_counter()
    e = mf.kernel()
    t1 = time.perf_counter()

    # Forces
    g = pyscf_grad.RKS(mf).grad()
    t2 = time.perf_counter()

    return {
        'system': 'H2O molecule',
        'method': 'PySCF RKS LDA',
        'basis': basis,
        'energy_Ha': e,
        'forces_Ha_per_Bohr': g.tolist(),
        'time_ms': (t1 - t0) * 1000,
        'force_time_ms': (t2 - t1) * 1000,
        'n_atoms': mol.natm,
    }

def run_pyscf_h2_curve(basis='ccpvdz', xc='LDA,VWN'):
    """H2 dissociation curve: 5 points from 0.5 to 3.0 Angstrom."""
    results = []
    for r_ang in [0.5, 0.74, 1.0, 2.0, 3.0]:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r_ang}", basis=basis, verbose=0, unit='Angstrom')
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 4
        t0 = time.perf_counter()
        e = mf.kernel()
        t1 = time.perf_counter()
        results.append({
            'r_Ang': r_ang,
            'energy_Ha': e,
            'time_ms': (t1 - t0) * 1000,
        })
    return results

def run_tides_atom(symbol, basis='DZP'):
    """Run TIDES atomic LDA calculation via Python API.

    AUDIT C7/P1.5: Now uses real MoleculeDriver through nanobind when available.
    Falls back to stub refusal when only the model Hamiltonian is available.
    """
    try:
        from tides.core import TidesCalculator
        from tides.config import TidesConfig, SystemConfig, BasisConfig, SCFConfig, GridConfig
        from tides._native import MoleculeDriver

        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'C': 6, 'N': 7, 'O': 8, 'Ne': 10}
        Z = Z_map.get(symbol, 1)

        # Use MoleculeDriver directly (real GTO-based SCF).
        mol = MoleculeDriver.build_molecule(
            atomic_numbers=[Z],
            positions=[0.0, 0.0, 0.0],
        )
        if mol.n_basis == 0:
            return {
                'system': f"{symbol} atom",
                'method': f'TIDES (STO-3G unavailable for Z={Z})',
                'error': f'STO-3G basis not available for {symbol}',
                'energy_Ha': None,
                'time_ms': None,
            }

        t0 = time.perf_counter()
        result = MoleculeDriver.run(mol=mol, grid_h=0.25, grid_margin=4.0,
                                     max_iter=200, tol=1e-8,
                                     use_grid_hartree=False,
                                     xc_functional='lda')
        t1 = time.perf_counter()

        t = result.timings
        return {
            'system': f"{symbol} atom",
            'method': f'TIDES MoleculeDriver LDA (STO-3G, GEMM+XC engine)',
            'basis': 'STO-3G',
            'energy_Ha': result.scf.energy,
            'time_ms': (t1 - t0) * 1000,
            'wall_time_ms': result.wall_time_ms,
            'n_atoms': 1,
            'n_iterations': result.scf.n_iterations,
            'converged': result.scf.converged,
            'E_kin': result.energy.E_kin,
            'E_ne': result.energy.E_ne,
            'E_H': result.energy.E_H,
            'E_xc': result.energy.E_xc,
            'E_ion': result.energy.E_ion,
            'pipeline_timings': {
                'rho_build_ms': t.rho_build_ms,
                'xc_eval_ms': t.xc_eval_ms,
                'poisson_ms': t.poisson_ms,
                'vmat_build_ms': t.vmat_build_ms,
                'scf_total_ms': t.scf_total_ms,
                'n_iterations': t.n_iterations,
                'used_gpu_xc': t.used_gpu_xc,
                'used_grid_hartree': t.used_grid_hartree,
                'xc_functional': t.xc_functional,
            },
        }
    except ImportError:
        return {
            'system': f"{symbol} atom",
            'method': f'TIDES (STUB — model Hamiltonian, not real DFT)',
            'error': 'MoleculeDriver not available in native backend. '
                     'Build nanobind bindings first. Benchmark refused per audit A1.',
            'energy_Ha': None,
            'time_ms': None,
        }
    except Exception as e:
        return {
            'system': f"{symbol} atom",
            'method': f'TIDES MoleculeDriver LDA (STO-3G)',
            'error': str(e),
            'energy_Ha': None,
            'time_ms': None,
        }

def main():
    print("=" * 72)
    print("  TIDES vs PySCF Benchmark — Real Pipeline (GEMM + Fused XC Engine)")
    print("=" * 72)

    all_results = []

    # --- He atom ---
    print("\n--- He atom LDA ---")
    pyscf_he = run_pyscf_atom('He', basis='ccpvdz')
    print(f"  PySCF (cc-pVDZ):  E = {pyscf_he['energy_Ha']:.10f} Ha  ({pyscf_he['time_ms']:.1f} ms)")
    all_results.append(pyscf_he)

    # Same-basis reference (STO-3G) for honest comparison with TIDES.
    pyscf_he_sto3g = run_pyscf_atom('He', basis='sto3g')
    print(f"  PySCF (STO-3G):   E = {pyscf_he_sto3g['energy_Ha']:.10f} Ha  ({pyscf_he_sto3g['time_ms']:.1f} ms)")
    all_results.append(pyscf_he_sto3g)

    tides_he = run_tides_atom('He', basis='DZP')
    if tides_he.get('energy_Ha') is not None:
        print(f"  TIDES (STO-3G):   E = {tides_he['energy_Ha']:.10f} Ha  ({tides_he['time_ms']:.1f} ms)")
        delta = abs(pyscf_he_sto3g['energy_Ha'] - tides_he['energy_Ha'])
        print(f"  ΔE (same basis) = {delta:.2e} Ha  ({delta*27.2114:.4f} eV)")
        tides_he['delta_vs_pyscf_Ha'] = delta
    else:
        print(f"  TIDES:  ERROR: {tides_he.get('error', 'unknown')}")
    all_results.append(tides_he)

    # --- Ne atom ---
    print("\n--- Ne atom LDA ---")
    pyscf_ne = run_pyscf_atom('Ne', basis='ccpvdz')
    print(f"  PySCF (cc-pVDZ):  E = {pyscf_ne['energy_Ha']:.10f} Ha  ({pyscf_ne['time_ms']:.1f} ms)")
    all_results.append(pyscf_ne)

    pyscf_ne_sto3g = run_pyscf_atom('Ne', basis='sto3g')
    print(f"  PySCF (STO-3G):   E = {pyscf_ne_sto3g['energy_Ha']:.10f} Ha  ({pyscf_ne_sto3g['time_ms']:.1f} ms)")
    all_results.append(pyscf_ne_sto3g)

    tides_ne = run_tides_atom('Ne', basis='DZP')
    if tides_ne.get('energy_Ha') is not None:
        print(f"  TIDES (STO-3G):   E = {tides_ne['energy_Ha']:.10f} Ha  ({tides_ne['time_ms']:.1f} ms)")
        delta = abs(pyscf_ne_sto3g['energy_Ha'] - tides_ne['energy_Ha'])
        print(f"  ΔE (same basis) = {delta:.2e} Ha  ({delta*27.2114:.4f} eV)")
        tides_ne['delta_vs_pyscf_Ha'] = delta
    else:
        print(f"  TIDES:  ERROR: {tides_ne.get('error', 'unknown')}")
    all_results.append(tides_ne)

    # --- H2O molecule ---
    print("\n--- H2O molecule LDA ---")
    h2o_geom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    pyscf_h2o = run_pyscf_molecule(h2o_geom, basis='ccpvdz')
    print(f"  PySCF:  E = {pyscf_h2o['energy_Ha']:.10f} Ha  ({pyscf_h2o['time_ms']:.1f} ms)")
    print(f"          Forces time: {pyscf_h2o['force_time_ms']:.1f} ms")
    all_results.append(pyscf_h2o)

    # --- H2 dissociation curve ---
    print("\n--- H2 dissociation curve (PySCF reference) ---")
    h2_curve = run_pyscf_h2_curve(basis='ccpvdz')
    for pt in h2_curve:
        print(f"  r={pt['r_Ang']:.2f} Å:  E = {pt['energy_Ha']:.10f} Ha  ({pt['time_ms']:.1f} ms)")
    all_results.append({'system': 'H2 curve', 'points': h2_curve})

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("  Benchmark Summary")
    print("=" * 72)
    print(f"  {'System':<20} {'Method':<25} {'Energy (Ha)':<18} {'Time (ms)':<12}")
    print("  " + "-" * 75)
    for r in all_results:
        if 'error' in r:
            print(f"  {r['system']:<20} {r['method']:<25} {'ERROR':<18} {'—':<12}")
        elif 'points' in r:
            print(f"  {'H2 curve':<20} {'PySCF RKS LDA':<25} {'(5 points)':<18} {'—':<12}")
        else:
            e_str = f"{r['energy_Ha']:.10f}" if r.get('energy_Ha') is not None else "—"
            t_str = f"{r['time_ms']:.1f}" if r.get('time_ms') is not None else "—"
            print(f"  {r['system']:<20} {r['method']:<25} {e_str:<18} {t_str:<12}")

    # --- Write JSON ledger ---
    ledger_path = Path(__file__).parent / 'pyscf_benchmark_results.json'
    with open(ledger_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results written to: {ledger_path}")

    # --- Delta summary ---
    print("\n  Energy Deltas (TIDES vs PySCF):")
    print("  AUDIT C7: MoleculeDriver now wired through nanobind.")
    print("  P2 FIX: SCF loop uses GEMM rho/vmat + fused Tier-0 XC engine.")
    print("  TIDES uses STO-3G; PySCF uses cc-pVDZ — basis mismatch expected.")
    print("  Delta is meaningful only vs same-basis PySCF reference.\n")
    for r in all_results:
        if 'delta_vs_pyscf_Ha' in r:
            d = r['delta_vs_pyscf_Ha']
            print(f"    {r['system']:<20} \u0394E = {d:.2e} Ha (STO-3G vs cc-pVDZ)")
            if 'pipeline_timings' in r:
                pt = r['pipeline_timings']
                print(f"      Pipeline: rho={pt['rho_build_ms']:.2f}ms "
                      f"xc={pt['xc_eval_ms']:.2f}ms "
                      f"vmat={pt['vmat_build_ms']:.2f}ms "
                      f"scf_total={pt['scf_total_ms']:.1f}ms "
                      f"iters={pt['n_iterations']} "
                      f"gpu_xc={pt['used_gpu_xc']}")
        elif r.get('error'):
            print(f"    {r['system']:<20} ERROR: {r['error'][:60]}")

    # --- PySCF reference data for TIDES validation ---
    print("\n  PySCF Reference Energies (target for TIDES C++ backend):")
    print(f"    He atom  (cc-pVDZ, LDA):  {all_results[0]['energy_Ha']:.10f} Ha")
    print(f"    Ne atom  (cc-pVDZ, LDA):  {all_results[2]['energy_Ha']:.10f} Ha")
    print(f"    H2O      (cc-pVDZ, LDA):  {all_results[4]['energy_Ha']:.10f} Ha")
    print(f"    H2 curve (cc-pVDZ, LDA):  5 points in JSON ledger")

if __name__ == '__main__':
    main()
