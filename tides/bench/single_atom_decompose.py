#!/usr/bin/env python3
"""P0.5b: decompose single-O energies vs grid spacing and grid alignment.

Hypothesis: multi-Ha grid sensitivity of isolated atoms comes from nucleus-on-
grid-point alignment (BUG-5 family) and/or unresolved core density in E_H/E_xc.
Compare E_kin/E_ne/E_H/E_xc for O at origin (grid-aligned) vs shifted off-node,
at h=0.3 and 0.2; also N for contrast.
"""
import sys, os

TIDES_SRC = '/home/indranil/git/New_DFT_Code/tides'
sys.path.insert(0, os.path.join(TIDES_SRC, 'api', 'python'))
os.environ.pop('TIDES_SRC_DIR', None)

import tides._native as native


def run(Z, pos, gh):
    r = native.NaoDriver.run(
        atomic_numbers=[Z], positions=list(pos),
        grid_h=gh, grid_margin=6.0, max_iter=80, tol=1e-7,
        use_dual_grid=False, xc_functional='lda', allow_grid_refine=False)
    e = r.energy
    return (e.E_total, e.E_kin, e.E_ne, e.E_H, e.E_xc, r.scf.converged,
            r.build_H_timings.n_iterations)


print(f"{'case':<26} {'E_total':>12} {'E_kin':>10} {'E_ne':>12} {'E_H':>10} "
      f"{'E_xc':>10} conv iters", flush=True)
for Z, sym in [(8, 'O'), (7, 'N')]:
    for gh in (0.3, 0.2):
        for tag, pos in [('origin', (0.0, 0.0, 0.0)),
                         ('shift(.05,.03,.07)', (0.05, 0.03, 0.07)),
                         ('shift(h/2)', (gh / 2, gh / 2, gh / 2))]:
            E, K, NE, H, XC, conv, it = run(Z, pos, gh)
            print(f"{sym} h={gh} {tag:<18} {E:>12.4f} {K:>10.3f} {NE:>12.3f} "
                  f"{H:>10.3f} {XC:>10.3f} {str(conv):>5} {it}", flush=True)
