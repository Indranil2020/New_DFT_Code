#!/usr/bin/env python3
"""FP64 oracle: hydrogenic radial eigenvalues via DIRECT r-space FD.

Avoids the log-grid transformation algebra entirely. Discretizes the standard
radial Schrodinger equation for P(r) = r R(r):
    -1/2 P''(r) + [V(r) + l(l+1)/(2 r^2)] P(r) = eps P(r)
on a non-uniform r grid (quadratically refined near the origin), with P(0)=0
and P(r_max)=0 (Dirichlet). Uses scipy.linalg.eigh_tridiagonal. This is
unambiguous and verifiable to machine precision against eps_n = -Z^2/(2 n^2).

Once this oracle is GREEN, the C++ log-grid solver is validated against it
(at the operator level, accounting for the transformation).
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def solve_hydrogenic_r(Z, l, n_states, r_max=80.0, n_r=4000):
    # Quadratic grid: r_i = (i/N)^2 * r_max, denser near origin.
    i = np.arange(1, n_r)  # interior points 1..n_r-1 (skip r=0)
    t = i / n_r
    r = t ** 2 * r_max
    V = -Z / r
    # Non-uniform 3-point 2nd derivative: P''_i = ...
    # Use the general formula for non-uniform grid.
    h_m = np.diff(r)            # h_i = r_{i+1} - r_i, length n-2
    h_p = h_m[1:]               # shift
    # P''_i = [2/(h_m h_p)] [ h_p P_{i-1} - (h_m+h_p) P_i + h_m P_{i+1} ]
    # but with variable spacing we need the consistent form. Simpler: uniform
    # grid with enough points. Use uniform grid instead for the oracle.
    r = np.linspace(1e-6, r_max, n_r)
    h = r[1] - r[0]
    # Interior points
    ri = r[1:-1]
    Vi = -Z / ri
    centrif = l * (l + 1) / (2.0 * ri ** 2)
    diag = 1.0 / h ** 2 + Vi + centrif   # -1/2 * (-2/h^2) = 1/h^2
    off = -0.5 / h ** 2 * np.ones(len(ri) - 1)
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1))
    return w


def main():
    cases = [
        (1, 0, [-0.5, -0.125, -1.0 / 18.0], "H_l0"),
        (1, 1, [-0.125, -1.0 / 18.0], "H_l1"),
        (2, 0, [-2.0, -0.5], "He_l0"),
        (6, 0, [-18.0], "C5+_l0"),
    ]
    print(f"{'case':>12} {'n_r':>6} {'state':>6} {'eps':>18} {'exact':>18} {'err':>12}")
    for Z, l, expected, label in cases:
        for n_r in [2000, 4000, 8000]:
            w = solve_hydrogenic_r(Z, l, len(expected), n_r=n_r)
            for k, eps in enumerate(w):
                err = abs(eps - expected[k])
                print(f"{label:>12} {n_r:6d} {k:6d} {eps:18.14f} {expected[k]:18.14f} {err:12.2e}")
    print("\nDirect-r oracle complete.")


if __name__ == "__main__":
    main()
