#!/usr/bin/env python3
"""P0.5 probe: isolated C/N/O atoms, TIDES AE vs AtomicLDA spherical reference.

Isolates V_ext/core handling with ZERO cross-atom terms. If isolated C is off
by ~Ha, the AE carbon bug has an on-site component; if clean, it is purely
cross-atom. LDA only (matched to AtomicLDA physics). Restricted KS on open-shell
atoms differs from the spherical fractional-occupation reference at the ~0.1 Ha
level; we look for MULTI-Ha discrepancies and grid_h sensitivity.

Usage: LD_PRELOAD=<mkl> python3 single_atom_check.py
"""
import sys, os, time

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))
os.environ.pop('TIDES_SRC_DIR', None)   # AE: no pseudopotentials

import tides._native as native

for Z, sym in [(6, 'C'), (7, 'N'), (8, 'O')]:
    for gh in (0.3, 0.2):
        t0 = time.time()
        r = native.NaoDriver.run(
            atomic_numbers=[Z], positions=[0.0, 0.0, 0.0],
            grid_h=gh, grid_margin=6.0,
            max_iter=80, tol=1e-7,
            use_dual_grid=False, xc_functional='lda',
            allow_grid_refine=False,
        )
        print(f"{sym} (Z={Z}) grid_h={gh}: E={r.energy.E_total:.6f} Ha  "
              f"conv={r.scf.converged}  iters={r.build_H_timings.n_iterations}  "
              f"xc={r.xc_functional}  wall={time.time()-t0:.1f}s", flush=True)
