#!/usr/bin/env python3
"""Generic benchmark: 10-atom, 100-atom molecules with LDA, PBE, B3LYP.

Tests scaling and generic applicability of GPU optimizations.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'python'))
import tides._native as native

# --- Molecule definitions ---

# 10-atom: C4H6 (2-butyne) — 4C + 6H = 10 atoms
mol_10atom = {
    'Z': [6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    'pos': [
        0.0, 0.0, 0.0,      # C1
        0.0, 0.0, 1.20,     # C2
        0.0, 0.0, 2.40,     # C3
        0.0, 0.0, 3.60,     # C4
        0.0, 1.03, -0.36,   # H1
        0.89, -0.51, -0.36, # H2
        -0.89, -0.51, -0.36,# H3
        0.0, 1.03, 3.96,    # H4
        0.89, -0.51, 3.96,  # H5
        -0.89, -0.51, 3.96, # H6
    ],
}

# 100-atom: C50H50 (linear chain) — 50C + 50H = 100 atoms
def make_100atom():
    Z = []
    pos = []
    for i in range(50):
        Z.append(6)
        pos.extend([0.0, 0.0, i * 1.40])
    for i in range(50):
        Z.append(1)
        pos.extend([0.0, 1.09, i * 1.40])
    return {'Z': Z, 'pos': pos}

mol_100atom = make_100atom()

# C6H6 for reference
mol_c6h6 = {
    'Z': [6,6,6,6,6,6,1,1,1,1,1,1],
    'pos': [1.40,0,0, 0.70,1.21,0, -0.70,1.21,0, -1.40,0,0, -0.70,-1.21,0, 0.70,-1.21,0,
            2.49,0,0, 1.25,2.16,0, -1.25,2.16,0, -2.49,0,0, -1.25,-2.16,0, 1.25,-2.16,0],
}

molecules = [
    ('C6H6',   mol_c6h6),
    ('C4H6',   mol_10atom),
    ('C50H50', mol_100atom),
]

functionals = ['lda', 'pbe', 'b3lyp']

os.environ['TIDES_VMAT_GEMM'] = '1'
os.environ['TIDES_RHO_GEMM'] = '1'

print("=" * 100)
print("GENERIC BENCHMARK: GPU-optimized TIDES build_H")
print("=" * 100)

for xc in functionals:
    print(f"\n--- XC = {xc.upper()} ---")
    print(f"{'Mol':8s} {'n_at':>4s} {'n_b':>4s}  {'bH_ms':>7s} {'pois':>7s} {'rho':>6s} {'xc':>7s} {'vmat':>6s}  {'gpu_p':>7s} {'gpu_x':>7s}  {'E_total':>12s}  {'it':>3s}  {'wall_s':>7s}")

    for name, mol in molecules:
        nat = len(mol['Z'])
        if nat > 50 and xc != 'lda':
            # Skip very large + expensive functional combos for time
            print(f"{name:8s} {nat:4d}      --- skipped (too expensive) ---")
            continue

        t0 = time.time()
        r = native.NaoDriver.run(
            atomic_numbers=mol['Z'], positions=mol['pos'],
            grid_h=0.5, grid_margin=4.0,
            max_iter=100, tol=1e-6,
            use_dual_grid=False, xc_functional=xc)
        wall = time.time() - t0
        bt = r.build_H_timings
        print(f"{name:8s} {nat:4d} {r.n_basis:4d}  {bt.total_ms:7.1f} {bt.poisson_ms:7.1f} "
              f"{bt.rho_build_ms:6.3f} {bt.xc_eval_ms:7.1f} {bt.vmat_build_ms:6.1f}  "
              f"{bt.poisson_solve_cpu_ms:7.1f} {bt.poisson_vmat_cpu_ms:7.1f}  "
              f"{r.energy.E_total:12.6f}  {bt.n_iterations:3d}  {wall:7.1f}")
