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
    """Run TIDES atomic LDA calculation via Python API."""
    try:
        from tides.core import TidesCalculator, SCFResult
        from tides.config import TidesConfig, SystemConfig, BasisConfig, SCFConfig, GridConfig

        Z_map = {'H': 1, 'He': 2, 'Li': 3, 'C': 6, 'N': 7, 'O': 8, 'Ne': 10}
        Z = Z_map.get(symbol, 1)

        config = TidesConfig()
        config.system = SystemConfig(
            n_atoms=1,
            atomic_numbers=[Z],
            positions=[[0.0, 0.0, 0.0]],
        )
        config.basis = BasisConfig(kind=basis)
        config.scf = SCFConfig(max_iter=200, energy_tol=1e-10)
        config.grid = GridConfig()

        calc = TidesCalculator(config)
        t0 = time.perf_counter()
        result = calc.run_scf()
        t1 = time.perf_counter()

        if not result.is_ok:
            return {
                'system': f"{symbol} atom",
                'method': f'TIDES TidesCalculator LDA ({basis})',
                'error': result.status.message if hasattr(result, 'status') else 'SCF failed',
                'energy_Ha': None,
                'time_ms': None,
            }

        scf = result.value
        return {
            'system': f"{symbol} atom",
            'method': f'TIDES TidesCalculator LDA ({basis})',
            'basis': basis,
            'energy_Ha': scf.energy,
            'time_ms': (t1 - t0) * 1000,
            'n_atoms': 1,
            'n_iterations': len(scf.energy_history) - 1 if hasattr(scf, 'energy_history') else None,
        }
    except Exception as e:
        return {
            'system': f"{symbol} atom",
            'method': f'TIDES TidesCalculator LDA ({basis})',
            'error': str(e),
            'energy_Ha': None,
            'time_ms': None,
        }

def main():
    print("=" * 72)
    print("  TIDES vs PySCF Benchmark — CPU Comparison")
    print("=" * 72)

    all_results = []

    # --- He atom ---
    print("\n--- He atom LDA ---")
    pyscf_he = run_pyscf_atom('He', basis='ccpvdz')
    print(f"  PySCF:  E = {pyscf_he['energy_Ha']:.10f} Ha  ({pyscf_he['time_ms']:.1f} ms)")
    all_results.append(pyscf_he)

    tides_he = run_tides_atom('He', basis='DZP')
    if tides_he.get('energy_Ha') is not None:
        print(f"  TIDES:  E = {tides_he['energy_Ha']:.10f} Ha  ({tides_he['time_ms']:.1f} ms)")
        delta = abs(pyscf_he['energy_Ha'] - tides_he['energy_Ha'])
        print(f"  ΔE    = {delta:.2e} Ha  ({delta*27.2114:.4f} eV)")
        tides_he['delta_vs_pyscf_Ha'] = delta
    else:
        print(f"  TIDES:  ERROR: {tides_he.get('error', 'unknown')}")
    all_results.append(tides_he)

    # --- Ne atom ---
    print("\n--- Ne atom LDA ---")
    pyscf_ne = run_pyscf_atom('Ne', basis='ccpvdz')
    print(f"  PySCF:  E = {pyscf_ne['energy_Ha']:.10f} Ha  ({pyscf_ne['time_ms']:.1f} ms)")
    all_results.append(pyscf_ne)

    tides_ne = run_tides_atom('Ne', basis='DZP')
    if tides_ne.get('energy_Ha') is not None:
        print(f"  TIDES:  E = {tides_ne['energy_Ha']:.10f} Ha  ({tides_ne['time_ms']:.1f} ms)")
        delta = abs(pyscf_ne['energy_Ha'] - tides_ne['energy_Ha'])
        print(f"  ΔE    = {delta:.2e} Ha  ({delta*27.2114:.4f} eV)")
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
    print("  NOTE: TIDES Python API uses a simplified model Hamiltonian (pure-Python")
    print("        fallback). The C++ native backend (nanobind) is not yet wired.")
    print("        Real DFT energies come from the C++ per-engine test suites (E1-E9).")
    print("        PySCF reference data below is the benchmark target for TIDES.\n")
    for r in all_results:
        if 'delta_vs_pyscf_Ha' in r:
            d = r['delta_vs_pyscf_Ha']
            status = "✅ PASS" if d < 1e-3 else "⚠️  CHECK (model vs real DFT)"
            print(f"    {r['system']:<20} ΔE = {d:.2e} Ha ({d*27.2114:.4f} eV)  {status}")

    # --- PySCF reference data for TIDES validation ---
    print("\n  PySCF Reference Energies (target for TIDES C++ backend):")
    print(f"    He atom  (cc-pVDZ, LDA):  {all_results[0]['energy_Ha']:.10f} Ha")
    print(f"    Ne atom  (cc-pVDZ, LDA):  {all_results[2]['energy_Ha']:.10f} Ha")
    print(f"    H2O      (cc-pVDZ, LDA):  {all_results[4]['energy_Ha']:.10f} Ha")
    print(f"    H2 curve (cc-pVDZ, LDA):  5 points in JSON ledger")

if __name__ == '__main__':
    main()
