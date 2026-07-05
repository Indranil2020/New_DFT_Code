#!/usr/bin/env python3
"""Independent FP64 oracle for the hydrogenic radial problem on a log grid.

DERIVATION (the bit that was wrong before):
  Radial Schrodinger (Hartree a.u.):  -1/2 P''(r) + [V(r) + l(l+1)/(2r^2)] P = eps P
  with P(r) = r R(r), P(0)=P(inf)=0.
  Change of variable x = ln r, so r = e^x, and let P(r) = e^{x/2} u(x).
  Then (chain rule):  d/dr = (1/r) d/dx,  d^2/dr^2 = (1/r^2)(d^2/dx^2 - d/dx).
  Substituting and using P = r^{1/2} u:
      -1/2 (1/r^2)(u'' - u') r^{1/2} + [V + l(l+1)/(2r^2)] r^{1/2} u = eps r^{1/2} u
  Multiply through by r^{3/2}:
      -1/2 (u'' - u') + [V r^2 + l(l+1)/2] u = eps r^2 u
  The first-derivative term is removed by the standard trick: the operator
  -1/2(d^2/dx^2 - d/dx) is NOT self-adjoint in the u basis. The SELF-ADJOINT
  form uses P(r) = u(x) directly (NO e^{x/2} factor), giving:
      -1/(2r^2) u''(x) + [V(r) + l(l+1)/(2r^2)] u(x) = eps u(x)
  Multiply by r^2 to clear the 1/r^2 on the kinetic term:
      -1/2 u''(x) + [V(r) r^2 + l(l+1)/2] u(x) = eps r^2 u(x)
  This is a GENERALIZED eigenproblem  H u = eps M u  with
      H = -1/2 d^2/dx^2 + diag(V(r)*r^2 + l(l+1)/2),   M = diag(r^2).
  Reducing to standard form via z = r u:  H' z = eps z with
      H' = diag(1/r) H diag(1/r) = -1/(2 r^2) d^2/dx^2 1/r^2 ... (messy).

  CLEANEST CORRECT FORM (what we implement): keep the generalized problem
      H u = eps M u,  H = K + diag(W),  M = diag(r^2),
      K = -1/2 d^2/dx^2,  W_i = V(r_i) r_i^2 + l(l+1)/2.
  Solve as standard via Cholesky of M = diag(r) since M is diagonal positive:
      z = r u,  H_tilde = diag(1/r) H diag(1/r),  H_tilde z = eps z.
      H_tilde_ij = K_ij/(r_i r_j) + delta_ij * [V_i r_i + l(l+1)/(2 r_i^2)].
  Wait — V*r^2 * 1/(r*r) = V. So:
      H_tilde_ii = K_ii/(r_i^2) + V_i + l(l+1)/(2 r_i^2)
      H_tilde_ij = K_ij/(r_i r_j)   (off-diagonal, symmetric since r constant? NO)
  Off-diagonal is NOT symmetric with naive division. The self-adjoint reduction
  requires H_tilde = D^{-1/2} H D^{-1/2} with D = M = diag(r^2), so D^{-1/2} =
  diag(1/r). For the off-diagonal K_ij (which is symmetric), the reduced entry
  K_ij/(r_i r_j) is symmetric. GOOD. And the diagonal V*r^2 becomes V*r^2/r_i^2 = V.
  So:
      H_tilde_ii = K_ii/r_i^2 + V_i + l(l+1)/(2 r_i^2)
      H_tilde_ij = K_ij/(r_i r_j)
  This is the form that was MISSING the V_i (bare) and had the centrifugal
  weighting wrong. Let's verify numerically.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def coulomb(r, Z):
    return -Z / r


def solve_hydrogenic(Z, l, n_states, x_min=-14.0, x_max=8.0, h=0.02):
    x = np.arange(x_min, x_max + 0.5 * h, h)
    r = np.exp(x)
    V = coulomb(r, Z)
    n = len(x)
    m = n - 2  # interior, Dirichlet at both ends
    rr = r[1:-1]
    VV = V[1:-1]
    # Kinetic K = -1/2 d^2/dx^2, 3-point stencil on interior.
    diag_K = 1.0 / h ** 2          # -1/2 * (-2/h^2)
    off_K = -0.5 / h ** 2
    # H_tilde (self-adjoint reduction, M = diag(r^2)):
    diag = diag_K / (rr ** 2) + VV + (l + 0.5) ** 2 / (2.0 * rr ** 2)
    # Wait: l(l+1)/2 / r^2 = (l(l+1)/2)/r^2. With (l+1/2)^2/2 = (l^2+l+1/4)/2.
    # These differ by 1/8. Use the EXACT l(l+1)/2 form:
    diag = diag_K / (rr ** 2) + VV + l * (l + 1) / (2.0 * rr ** 2)
    off = off_K / (rr[:-1] * rr[1:])  # K_ij/(r_i r_j) for neighbors i,i+1
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1))
    states = []
    for k in range(n_states):
        eps = w[k]
        u = np.zeros(n)
        u[1:-1] = v[:, k]
        norm = np.sqrt(np.trapezoid(u * u, x))
        u /= norm
        R = np.zeros(n)
        R[1:-1] = u[1:-1] / np.sqrt(rr)
        # actually R(r) = u(x)/r^{1/2} since z=r u => u = z; R = z/r = u/r^{1/2}? recheck
        states.append((eps, R, r.copy()))
    return states


def main():
    cases = [
        (1, 0, [-0.5, -0.125, -1.0 / 18.0], "H_l0"),
        (1, 1, [-0.125, -1.0 / 18.0], "H_l1"),
        (2, 0, [-2.0, -0.5], "He_l0"),
        (6, 0, [-18.0], "C5+_l0"),
    ]
    print(f"{'case':>12} {'h':>8} {'state':>6} {'eps':>16} {'exact':>16} {'err':>12}")
    for Z, l, expected, label in cases:
        for h in [0.04, 0.02, 0.01]:
            states = solve_hydrogenic(Z, l, len(expected), h=h)
            for k, (eps, R, r) in enumerate(states):
                err = abs(eps - expected[k])
                print(f"{label:>12} {h:8.4f} {k:6d} {eps:16.12f} {expected[k]:16.12f} {err:12.2e}")
    print("\nOracle built.")


if __name__ == "__main__":
    main()
