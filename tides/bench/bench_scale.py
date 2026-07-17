#!/usr/bin/env python3
"""Comprehensive scale benchmark with audit ledger.

Tests 10/50/100 atom molecules with LDA, PBE, B3LYP.
Records detailed timing breakdown and energy accuracy.
Outputs structured ledger entries.
"""
import sys, os, time, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'python'))
import tides._native as native

os.environ['TIDES_VMAT_GEMM'] = '1'
os.environ['TIDES_RHO_GEMM'] = '1'

LEDGER_FILE = os.path.join(os.path.dirname(__file__), 'audit_ledger.jsonl')

def ledger_write(entry):
    entry['timestamp'] = datetime.datetime.now().isoformat()
    with open(LEDGER_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"  [ledger] {entry['event']}")

# --- Molecule definitions ---

# 12-atom: C6H6 (benzene) - our reference molecule
mol_c6h6 = {
    'name': 'C6H6',
    'nat': 12,
    'Z': [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    'pos': [
        1.40, 0, 0, 0.70, 1.21, 0, -0.70, 1.21, 0,
        -1.40, 0, 0, -0.70, -1.21, 0, 0.70, -1.21, 0,
        2.49, 0, 0, 1.25, 2.16, 0, -1.25, 2.16, 0,
        -2.49, 0, 0, -1.25, -2.16, 0, 1.25, -2.16, 0,
    ],
}

# 10-atom: C4H6 (2-butyne)
mol_c4h6 = {
    'name': 'C4H6',
    'nat': 10,
    'Z': [6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    'pos': [
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.20,
        0.0, 0.0, 2.40,
        0.0, 0.0, 3.60,
        0.0, 1.03, -0.36,
        0.89, -0.51, -0.36,
        -0.89, -0.51, -0.36,
        0.0, 1.03, 3.96,
        0.89, -0.51, 3.96,
        -0.89, -0.51, 3.96,
    ],
}

# Linear chain molecule generator
def make_chain(n_c):
    n_h = n_c
    nat = n_c + n_h
    Z = [6] * n_c + [1] * n_h
    pos = []
    for i in range(n_c):
        pos.extend([0.0, 0.0, i * 1.40])
    for i in range(n_h):
        pos.extend([0.0, 1.09, i * 1.40])
    return {'name': f'C{n_c}H{n_h}', 'nat': nat, 'Z': Z, 'pos': pos}

mol_c25h25 = make_chain(25)   # 50 atoms
mol_c50h50 = make_chain(50)   # 100 atoms

# Reference energies from FP64 runs (C6H6 only — established baseline)
ref_energies = {
    ('C6H6', 'lda'):   105.610897,
    ('C6H6', 'pbe'):   105.566208,
    ('C6H6', 'b3lyp'): 104.975022,
}

molecules = [
    mol_c4h6,      # 10 atoms
    mol_c6h6,      # 12 atoms (reference)
    mol_c25h25,    # 50 atoms
    # mol_c50h50,  # 100 atoms — enable after 50-atom works
]

functionals = ['lda', 'pbe', 'b3lyp']

print("=" * 130)
print("TIDES SCF SCALE BENCHMARK — FP64 GEMM (single-stream, cuBLAS DEFAULT algo)")
print(f"GPU: RT 3060 (28 SMs, 0.2 TFLOPS FP64, 12.7 TFLOPS FP32)")
print(f"Date: {datetime.datetime.now().isoformat()}")
print("=" * 130)

ledger_write({
    'event': 'benchmark_start',
    'config': {
        'gpu': 'RT 3060',
        'fp64_tflops': 0.2,
        'fp32_tflops': 12.7,
        'vmat_gemm': True,
        'rho_gemm': True,
        'splitk': False,
        'stream_overlap': False,
        'mixed_kernel': False,
    }
})

all_results = []

for mol in molecules:
    for xc in functionals:
        name = mol['name']
        nat = mol['nat']
        print(f"\n{'='*80}")
        print(f"  {name} ({nat} atoms) / {xc.upper()}")
        print(f"{'='*80}")

        t0 = time.time()
        r = native.NaoDriver.run(
            atomic_numbers=mol['Z'], positions=mol['pos'],
            grid_h=0.5, grid_margin=4.0,
            max_iter=100, tol=1e-6,
            use_dual_grid=False, xc_functional=xc)
        wall = time.time() - t0

        bt = r.build_H_timings
        ref = ref_energies.get((name, xc))
        dE = (r.energy.E_total - ref) if ref is not None else None

        # Print detailed timing breakdown
        print(f"  n_basis:     {r.n_basis}")
        print(f"  Timings (avg per SCF iter, ms):")
        print(f"    quantize_P:  {bt.quantize_P_ms:8.2f}")
        print(f"    rho_build:   {bt.rho_build_ms:8.2f}")
        print(f"    poisson:     {bt.poisson_ms:8.2f}")
        print(f"    xc_eval:     {bt.xc_eval_ms:8.2f}")
        print(f"    vmat_build:  {bt.vmat_build_ms:8.2f}")
        print(f"    assemble_H:  {bt.assemble_H_ms:8.2f}")
        print(f"    total:       {bt.total_ms:8.2f}")
        print(f"  GPU event timings (avg, ms):")
        print(f"    gpu_poisson: {bt.poisson_solve_cpu_ms:8.2f}")
        print(f"    gpu_xc_ker:  {bt.poisson_vmat_cpu_ms:8.2f}")
        print(f"  Energy: {r.energy.E_total:.6f}")
        if dE is not None:
            print(f"  dE vs ref: {dE:+.6f}")
        print(f"  SCF iters: {bt.n_iterations}")
        print(f"  Wall time: {wall:.1f}s")

        entry = {
            'event': 'benchmark_result',
            'mol': name,
            'nat': nat,
            'nbasis': r.n_basis,
            'xc': xc,
            'timings_ms': {
                'quantize_P': bt.quantize_P_ms,
                'rho_build': bt.rho_build_ms,
                'poisson': bt.poisson_ms,
                'xc_eval': bt.xc_eval_ms,
                'vmat_build': bt.vmat_build_ms,
                'assemble_H': bt.assemble_H_ms,
                'total': bt.total_ms,
                'gpu_poisson': bt.poisson_solve_cpu_ms,
                'gpu_xc_ker': bt.poisson_vmat_cpu_ms,
            },
            'energy': r.energy.E_total,
            'dE_vs_ref': dE,
            'n_iters': bt.n_iterations,
            'wall_s': wall,
        }
        ledger_write(entry)
        all_results.append(entry)

# Summary table
print("\n" + "=" * 130)
print("SUMMARY")
print("=" * 130)
hdr = f"{'Mol':10s} {'nat':>4s} {'nb':>4s} {'XC':6s} {'bH_ms':>8s} {'pois':>6s} {'xc':>8s} {'gpu_p':>7s} {'gpu_xk':>7s} {'E_total':>12s} {'dE':>10s} {'it':>3s} {'wall':>7s}"
print(hdr)
for r in all_results:
    t = r['timings_ms']
    dE_str = f"{r['dE_vs_ref']:+.6f}" if r['dE_vs_ref'] is not None else "       ---"
    print(f"{r['mol']:10s} {r['nat']:4d} {r['nbasis']:4d} {r['xc']:6s} {t['total']:8.1f} {t['poisson']:6.1f} {t['xc_eval']:8.1f} {t['gpu_poisson']:7.1f} {t['gpu_xc_ker']:7.2f} {r['energy']:12.6f} {dE_str:>10s} {r['n_iters']:3d} {r['wall_s']:7.1f}s")

print(f"\nLedger written to: {LEDGER_FILE}")
