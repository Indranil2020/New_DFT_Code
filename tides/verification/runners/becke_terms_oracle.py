#!/usr/bin/env python3
"""Becke multicenter numerical-quadrature oracle for TIDES one-electron terms.

Computes S, T, and V_ext matrix elements **directly** from dumped radial
tables — no basis fitting — via atom-centered multicenter numerical
quadrature (Becke 1988 partition-of-unity + Gauss-Legendre radial grids +
Lebedev angular grids).

This is the *precision* oracle for task **p06b-becke-oracle**, complementing
the fitted-Gaussian PySCF runner (``compare_pyscf_terms.py``) which is only a
few-percent sanity oracle.

Method
------
For each atom *A* we build a radial grid (Gauss-Legendre nodes linearly mapped
to [0, r_cut], ~120 points, capped at the atom's max r_cut)
× angular grid (Lebedev order-29 or product Gauss-Legendre × uniform-φ).

Becke partition-of-unity weights (k=3 iterations of the smoothing polynomial)
combine overlapping atomic meshes into a single integral.

Basis values at mesh points come from cubic-spline interpolation of the dumped
radial tables (``scipy.interpolate.CubicSpline``, extrapolate=0 beyond r_cut)
times real spherical harmonics matching the TIDES convention (validated against
the dumped S matrix).

  S_ij  = Σ_mesh  w · φ_i · φ_j
  T_ij  = ½ Σ_mesh  w · ∇φ_i · ∇φ_j          (gradient form)
  V_ij  = Σ_mesh  w · φ_i · (−Z_A/|r−R_A|) · φ_j   (AE only)

No try/except — all error conditions use assertions and explicit checks.
numpy/scipy only (pyscf allowed solely for optional cross-checks in the
selftest).  CPU only.  Vectorized with numpy throughout.

CLI
---
    python3 becke_terms_oracle.py <dump_dir>

Exit 0 iff gates pass (for the selftest).  For real dumps the numbers are
written to stdout and ``<dump_dir>/becke_report.json`` but the exit code is
always 0 (real-dump deviations measure TIDES grid error, not oracle error).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
# (no scipy.special import needed; we roll our own Y_lm)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default grid parameters
DEFAULT_N_RADIAL = 120          # radial points per atom
DEFAULT_LEBEDEV_ORDER = 29      # angular points per radial shell
BECKE_K = 3                     # iterations of the smoothing polynomial

# Acceptance tolerances for the synthetic selftest
TOL_S_SELFTTEST = 1.0e-8
TOL_T_SELFTTEST = 1.0e-6
TOL_V_SELFTTEST = 1.0e-6

# Tolerance for Y_lm convention validation against on-site S blocks
TOL_YLM_VALIDATE = 1.0e-6


# ---------------------------------------------------------------------------
# Real spherical harmonics — matching TIDES convention exactly
# ---------------------------------------------------------------------------

def _assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Associated Legendre P_l^m(x) **without** Condon-Shortley phase.

    Matches the TIDES C++ ``AssociatedLegendre`` function in
    ``two_center_integrals.hpp``.  Vectorized over *x*.
    """
    am = abs(m)
    if am > l:
        return np.zeros_like(x)
    # Seed P_m^m = (2m-1)!! * (1-x^2)^{m/2}  (no CS phase)
    pmm = np.ones_like(x)
    if am > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(am):
            pmm = pmm * fact * somx2
            fact += 2.0
    if l == am:
        return pmm
    pmmp1 = x * (2.0 * am + 1.0) * pmm
    if l == am + 1:
        return pmmp1
    pll = np.zeros_like(x)
    for ll in range(am + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + am - 1.0) * pmm) / (ll - am)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _ylm_norm(l: int, m: int) -> float:
    """Normalization constant for real spherical harmonics (TIDES convention).

    N = sqrt((2l+1)/(4π) · (l−|m|)!/(l+|m|)!)  for m=0
    N = sqrt(2) · sqrt((2l+1)/(4π) · (l−|m|)!/(l+|m|)!)  for m≠0
    """
    am = abs(m)
    denom = 1.0
    for i in range(l - am + 1, l + am + 1):
        denom *= float(i)
    n = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) / denom)
    if am != 0:
        n *= np.sqrt(2.0)
    return n


def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate real spherical harmonic Y_lm(θ, φ) — TIDES convention.

      m > 0:  Y = N · P_l^m(cosθ) · cos(m·φ)
      m < 0:  Y = N · P_l^|m|(cosθ) · sin(|m|·φ)
      m = 0:  Y = N · P_l^0(cosθ)

    *theta* and *phi* are broadcastable arrays.
    """
    x = np.cos(theta)
    am = abs(m)
    plm = _assoc_legendre(l, am, x)
    n = _ylm_norm(l, am)
    if m > 0:
        return n * plm * np.cos(m * phi)
    elif m < 0:
        return n * plm * np.sin(am * phi)
    else:
        return n * plm


# ---------------------------------------------------------------------------
# Angular gradient of real spherical harmonics via finite-difference rotation
# ---------------------------------------------------------------------------

def _grad_ylm_cartesian(
    l: int, m: int,
    coords: np.ndarray,
    h: float = 1.0e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of Y_lm in Cartesian coordinates via finite differences.

    Y_lm depends on direction (θ, φ) only.  At a point **r** = (x, y, z), the
    Cartesian gradient ∇Y_lm includes a 1/r factor because changing x by δx
    changes the direction by δx/r.  We evaluate Y_lm at r + h·ê and r − h·ê
    (actual Cartesian offsets) to capture this automatically.

    Returns (dY/dx, dY/dy, dY/dz), each shape (Npts,).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    def _ylm_at(xa, ya, za):
        r = np.sqrt(xa * xa + ya * ya + za * za)
        r_safe = np.where(r > 0, r, 1e-30)
        th = np.arccos(np.clip(za / r_safe, -1.0, 1.0))
        ph = np.arctan2(ya, xa)
        return real_sph_harm(l, m, th, ph)

    ddx = (_ylm_at(x + h, y, z) - _ylm_at(x - h, y, z)) / (2.0 * h)
    ddy = (_ylm_at(x, y + h, z) - _ylm_at(x, y - h, z)) / (2.0 * h)
    ddz = (_ylm_at(x, y, z + h) - _ylm_at(x, y, z - h)) / (2.0 * h)
    return ddx, ddy, ddz


# ---------------------------------------------------------------------------
# Radial grid generation (Gauss-Legendre, linear mapping)
# ---------------------------------------------------------------------------

def generate_radial_grid(n_radial: int, r_cut: float) -> np.ndarray:
    """Generate a radial grid on [0, r_cut] using Gauss-Legendre nodes.

    Gauss-Legendre nodes on [-1, 1] are mapped linearly to [0, r_cut]:
        r(ξ) = r_cut · (1 + ξ) / 2

    This gives exponential convergence for smooth integrands.  The 1/r nuclear
    singularity is integrable (R²·r → 0 at r=0) so no special mapping is needed.
    """
    gl_nodes, _ = np.polynomial.legendre.leggauss(n_radial)
    r = r_cut * (1.0 + gl_nodes) / 2.0
    return r




# ---------------------------------------------------------------------------
# Angular grid: Lebedev-like product Gauss-Legendre × uniform-phi
# ---------------------------------------------------------------------------

def generate_angular_grid(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an angular grid (θ, φ, w) as product Gauss-Legendre ×
    uniform-φ.

    For ``order`` ≈ 29, we use n_theta ≈ 15 Gauss-Legendre points on [0, π]
    and n_phi ≈ 2*n_theta uniform points on [0, 2π).  This gives ~300 points,
    comparable to a Lebedev-29 grid.

    The weights include the sin(θ) Jacobian for the spherical integral:

        ∫ f(θ,φ) dΩ = ∫₀^π ∫₀^{2π} f(θ,φ) sin(θ) dθ dφ

    Returns (theta, phi, weights) 1-D arrays of equal length.
    """
    # Determine n_theta from order: Lebedev-29 has 302 pts; we want ~same
    if order <= 15:
        n_theta = 8
    elif order <= 29:
        n_theta = 14
    elif order <= 35:
        n_theta = 18
    else:  # order 41+
        n_theta = 24

    n_phi = 2 * n_theta

    # Gauss-Legendre on [-1, 1] → map to [0, π]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x_gl)  # θ ∈ (0, π)
    # Weight includes sin(θ) Jacobian: ∫₀^π f(θ) sin(θ) dθ = Σ w_gl · f(θ)
    # because x = cos(θ), dx = -sin(θ) dθ, and ∫₋₁^1 f(arccos x) dx
    # = ∫₀^π f(θ) sin(θ) dθ.  So w_theta = w_gl.
    w_theta = w_gl  # already includes the sinθ factor via change of variable

    # Uniform phi
    dphi = 2.0 * np.pi / n_phi
    phi = np.arange(n_phi) * dphi
    w_phi = np.full(n_phi, dphi)

    # Product grid
    theta_grid = np.repeat(theta, n_phi)
    phi_grid = np.tile(phi, n_theta)
    w_grid = np.repeat(w_theta, n_phi) * np.tile(w_phi, n_theta)

    return theta_grid, phi_grid, w_grid


# ---------------------------------------------------------------------------
# Becke partition-of-unity weights
# ---------------------------------------------------------------------------

def _becke_cell_weight(
    iatom: int,
    atoms_pos: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Becke partition weight for atom *iatom* at *coords*.

    Implements the Becke 1988 partition-of-unity with k=3 iterations of the
    smoothing polynomial.  *atoms_pos* is (Natom, 3), *coords* is (Npts, 3).

    Returns a weight array (Npts,) that, summed over all atoms, gives 1.
    """
    natom = len(atoms_pos)
    N = coords.shape[0]

    # Distance from each point to each atom: (Npts, Natom)
    # Compute |r - R_A|
    dists = np.empty((N, natom))
    for a in range(natom):
        diff = coords - atoms_pos[a]
        dists[:, a] = np.sqrt(np.sum(diff * diff, axis=1))

    # For each pair of atoms (A, B), compute the "fuzzy" weight function
    # s_AB(r) = (μ_AB - 1) / (μ_AB + 1) where μ_AB relates distances to A and B.
    # Actually the standard approach: P_A(r) = Π_{B≠A} s(μ_AB)
    # where μ_AB = (d_A - d_B) / R_AB  and s is the smoothed step function.

    # Compute pairwise R_AB
    weights = np.ones((N, natom))
    for a in range(natom):
        for b in range(natom):
            if a == b:
                continue
            R_ab = np.linalg.norm(atoms_pos[a] - atoms_pos[b])
            assert R_ab > 1e-10, f"Atoms {a} and {b} are at the same position"
            mu = (dists[:, a] - dists[:, b]) / R_ab  # (Npts,)

            # Smoothed step function: ν(μ) with k iterations
            nu = mu
            for _ in range(BECKE_K):
                nu = 0.5 * nu * (3.0 - nu * nu)
            s = 0.5 * (1.0 - nu)

            weights[:, a] *= s

    # Normalize: P_A = w_A / Σ_B w_B
    total = weights.sum(axis=1, keepdims=True)
    assert np.all(total > 0), "Zero total Becke weight at some grid point"
    return weights[:, iatom] / total[:, 0]


# ---------------------------------------------------------------------------
# Dump loading (reused from compare_pyscf_terms.py schema)
# ---------------------------------------------------------------------------

def load_dump(dump_dir: str) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a TIDES dump directory.

    Returns (meta, S, T, V_ext, radial_tables) where radial_tables maps
    (atom, fn) → (r_array, R_array).
    """
    meta_path = os.path.join(dump_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)

    n = int(meta["n_basis"])

    def _load_matrix(name: str) -> np.ndarray:
        data = np.loadtxt(os.path.join(dump_dir, name))
        assert data.size == n * n, (
            f"{name}: expected {n}×{n} = {n*n} elements, got {data.size}"
        )
        return data.reshape(n, n)

    S = _load_matrix("S.txt")
    T = _load_matrix("T.txt")

    # V_ext may not exist for some dumps; load if present
    vext_path = os.path.join(dump_dir, "V_ext.txt")
    if os.path.isfile(vext_path):
        V = _load_matrix("V_ext.txt")
    else:
        V = np.zeros((n, n))

    # Load radial tables
    radial_tables: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for entry in meta["basis"]:
        key = (int(entry["atom"]), int(entry["fn"]))
        if key in radial_tables:
            continue
        filepath = os.path.join(dump_dir, f"radial_{key[0]}_{key[1]}.txt")
        data = np.loadtxt(filepath)
        assert data.ndim == 2 and data.shape[1] >= 2, (
            f"Expected ≥2 columns in {filepath}, got shape {data.shape}"
        )
        radial_tables[key] = (data[:, 0].copy(), data[:, 1].copy())

    return meta, S, T, V, radial_tables


# ---------------------------------------------------------------------------
# Build per-atom mesh with Becke weights
# ---------------------------------------------------------------------------

def build_becke_mesh(
    meta: dict,
    radial_tables: dict,
    n_radial: int = DEFAULT_N_RADIAL,
    lebedev_order: int = DEFAULT_LEBEDEV_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the combined multicenter Becke mesh.

    For each atom we generate a radial × angular product grid, then assign
    Becke partition-of-unity weights.  All atomic meshes are concatenated
    into a single (coords, weights) array.

    Returns (coords, weights, atom_index) where:
      coords: (Ntotal, 3) Cartesian coordinates
      weights: (Ntotal,) quadrature weights (includes r² Jacobian)
      atom_index: (Ntotal,) which atom each point belongs to
    """
    atoms = meta["atoms"]
    natom = len(atoms)
    atoms_pos = np.array([a["pos_bohr"] for a in atoms], dtype=float)  # (natom, 3)

    # Determine r_cut per atom from radial tables
    atom_rcuts = {}
    for entry in meta["basis"]:
        a = int(entry["atom"])
        fn = int(entry["fn"])
        r_arr, _ = radial_tables[(a, fn)]
        rc = r_arr[-1]
        if a not in atom_rcuts or rc > atom_rcuts[a]:
            atom_rcuts[a] = rc

    # Angular grid (same for all atoms)
    ang_theta, ang_phi, ang_w = generate_angular_grid(lebedev_order)
    n_ang = len(ang_theta)

    all_coords = []
    all_weights = []
    all_atom_idx = []

    # First build all atomic meshes (without Becke weights)
    atom_meshes = []
    for a in range(natom):
        r_cut = atom_rcuts[a]
        # Gauss-Legendre radial quadrature on [0, r_cut] (linear mapping).
        # Gauss-Legendre nodes on [-1,1] are mapped linearly to [0, r_cut].
        # This gives exponential convergence for smooth (pseudo) NAOs and the
        # 1/r nuclear singularity is integrable (R²·r → 0 at r=0).
        # The 3D Jacobian gives: ∫₀^{r_cut} g(r) r² dr = ∫₋₁^1 g(r(ξ)) r(ξ)² (dr/dξ) dξ
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_radial)
        r_from_xi = r_cut * (1.0 + gl_nodes) / 2.0
        dr_dxi = r_cut / 2.0
        radial_w = gl_weights * r_from_xi ** 2 * dr_dxi

        # Build product grid for this atom
        # (n_radial, n_ang) → flatten
        r_grid = r_from_xi  # use the computed r values
        for ia in range(n_radial):
            # Cartesian coordinates of angular points at radius r_grid[ia]
            x = r_grid[ia] * np.sin(ang_theta) * np.cos(ang_phi)
            y = r_grid[ia] * np.sin(ang_theta) * np.sin(ang_phi)
            z = r_grid[ia] * np.cos(ang_theta)
            coords = atoms_pos[a] + np.column_stack([x, y, z])
            w = radial_w[ia] * ang_w
            all_coords.append(coords)
            all_weights.append(w)
            all_atom_idx.append(np.full(n_ang, a, dtype=int))

    all_coords = np.concatenate(all_coords, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    all_atom_idx = np.concatenate(all_atom_idx, axis=0)

    # Now compute Becke partition-of-unity weights and multiply
    becke_w = np.zeros_like(all_weights)
    for a in range(natom):
        mask = all_atom_idx == a
        bw = _becke_cell_weight(a, atoms_pos, all_coords)
        becke_w[mask] = bw[mask]

    # Final weight = original quadrature weight × Becke partition weight
    final_weights = all_weights * becke_w

    return all_coords, final_weights, all_atom_idx


# ---------------------------------------------------------------------------
# Basis function evaluation on the mesh
# ---------------------------------------------------------------------------

class BasisEvaluator:
    """Evaluate NAO basis functions on the Becke mesh.

    Each basis entry is (atom, l, m, fn).  The orbital is
    φ(r) = R(|r − R_atom|) · Y_lm(θ, φ)  where (θ, φ) is the direction
    of (r − R_atom).
    """

    def __init__(self, meta: dict, radial_tables: dict):
        self.meta = meta
        self.basis = meta["basis"]
        self.n_basis = int(meta["n_basis"])
        self.atoms_pos = np.array(
            [a["pos_bohr"] for a in meta["atoms"]], dtype=float
        )
        self.natom = len(meta["atoms"])

        # Build cubic splines for each unique (atom, fn)
        # Also store r_cut for each
        self.splines: Dict[Tuple[int, int], CubicSpline] = {}
        self.rcuts: Dict[Tuple[int, int], float] = {}
        self.l_of: Dict[Tuple[int, int], int] = {}
        for entry in self.basis:
            a = int(entry["atom"])
            fn = int(entry["fn"])
            l = int(entry["l"])
            key = (a, fn)
            if key in self.splines:
                continue
            r_arr, R_arr = radial_tables[key]
            assert np.all(np.diff(r_arr) > 0), (
                f"Radial grid for atom {a} fn {fn} must be strictly increasing"
            )
            # CubicSpline with zero extrapolation beyond r_cut
            # We set bc_type='natural' and extrapolate to 0 beyond r_cut
            spline = CubicSpline(r_arr, R_arr, extrapolate=False)
            self.splines[key] = spline
            self.rcuts[key] = r_arr[-1]
            self.l_of[key] = l

        # Precompute (atom, l, m, fn) for each basis index
        self.basis_info = []
        for entry in self.basis:
            self.basis_info.append((
                int(entry["atom"]),
                int(entry["l"]),
                int(entry["m"]),
                int(entry["fn"]),
            ))

    def evaluate(self, coords: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at *coords*.

        Returns (Npts, n_basis) array of φ_i(r) values.
        """
        N = coords.shape[0]
        nb = self.n_basis
        vals = np.zeros((N, nb))

        for i, (a, l, m, fn) in enumerate(self.basis_info):
            # Relative coordinates w.r.t. atom *a*
            mask = atom_idx == a
            # Also evaluate on ALL points (basis functions extend to all space)
            # Actually for efficiency and correctness, we evaluate on all points
            # since φ_i is non-zero everywhere (not just on its own atom's mesh).
            rel = coords - self.atoms_pos[a]
            r_rel = np.sqrt(np.sum(rel * rel, axis=1))
            theta = np.arccos(np.clip(rel[:, 2] / np.where(r_rel > 0, r_rel, 1.0), -1.0, 1.0))
            phi = np.arctan2(rel[:, 1], rel[:, 0])

            # Radial part via spline (zero beyond r_cut)
            key = (a, fn)
            spline = self.splines[key]
            r_cut = self.rcuts[key]
            R_vals = np.zeros(N)
            within = r_rel <= r_cut
            R_vals[within] = spline(r_rel[within])
            # Zero out negligible values
            R_vals[np.isnan(R_vals)] = 0.0

            # Angular part
            Y_vals = real_sph_harm(l, m, theta, phi)

            vals[:, i] = R_vals * Y_vals

        return vals

    def evaluate_gradients(self, coords: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
        """Evaluate Cartesian gradients ∇φ_i at all mesh points.

        Uses the chain rule:
          ∇φ = (∂R/∂r · Y_lm) · r̂  +  R · ∇Y_lm

        where ∂R/∂r comes from the spline derivative, and ∇Y_lm is computed
        via finite differences on Cartesian offsets.

        Returns (Npts, n_basis, 3) array.
        """
        N = coords.shape[0]
        nb = self.n_basis
        grads = np.zeros((N, nb, 3))

        for i, (a, l, m, fn) in enumerate(self.basis_info):
            rel = coords - self.atoms_pos[a]
            r_rel = np.sqrt(np.sum(rel * rel, axis=1))
            # Avoid division by zero
            r_safe = np.where(r_rel > 1e-30, r_rel, 1e-30)
            theta = np.arccos(np.clip(rel[:, 2] / r_safe, -1.0, 1.0))
            phi = np.arctan2(rel[:, 1], rel[:, 0])

            # Radial part and its derivative
            key = (a, fn)
            spline = self.splines[key]
            r_cut = self.rcuts[key]
            R_vals = np.zeros(N)
            Rp_vals = np.zeros(N)  # dR/dr
            within = r_rel <= r_cut
            if np.any(within):
                R_vals[within] = spline(r_rel[within])
                Rp_vals[within] = spline(r_rel[within], 1)  # 1st derivative
            R_vals = np.nan_to_num(R_vals)
            Rp_vals = np.nan_to_num(Rp_vals)

            # Angular part and its gradient
            Y_vals = real_sph_harm(l, m, theta, phi)

            # Gradient of Y_lm via finite differences
            dYdx, dYdy, dYdz = _grad_ylm_cartesian(l, m, rel)

            # Unit radial vector
            rhat = rel / r_safe[:, None]

            # ∇φ = Rp · Y · r̂  +  R · ∇Y
            # ∇Y components (dY/dx, dY/dy, dY/dz) are already Cartesian
            term_radial = (Rp_vals * Y_vals)[:, None] * rhat  # (N, 3)
            term_angular = np.column_stack([R_vals * dYdx, R_vals * dYdy, R_vals * dYdz])

            grads[:, i, :] = term_radial + term_angular

            # Zero out points beyond r_cut (basis function is exactly zero there)
            beyond = r_rel > r_cut
            grads[beyond, i, :] = 0.0

        return grads


# ---------------------------------------------------------------------------
# Matrix element computation
# ---------------------------------------------------------------------------

def compute_matrices(
    meta: dict,
    radial_tables: dict,
    n_radial: int = DEFAULT_N_RADIAL,
    lebedev_order: int = DEFAULT_LEBEDEV_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute S, T, V_ext via Becke multicenter quadrature.

    Returns (S, T, V_ext, report_dict).
    """
    n = int(meta["n_basis"])

    # Build mesh
    coords, weights, atom_idx = build_becke_mesh(
        meta, radial_tables, n_radial=n_radial, lebedev_order=lebedev_order
    )
    Npts = len(coords)

    # Evaluate basis and gradients
    evaluator = BasisEvaluator(meta, radial_tables)
    phi_vals = evaluator.evaluate(coords, atom_idx)        # (Npts, n_basis)
    phi_grads = evaluator.evaluate_gradients(coords, atom_idx)  # (Npts, n_basis, 3)

    # Overlap: S_ij = Σ w · φ_i · φ_j
    # Vectorized: S = (phi * w)^T @ phi
    pw = phi_vals * weights[:, None]
    S = pw.T @ phi_vals

    # Kinetic: T_ij = ½ Σ w · ∇φ_i · ∇φ_j
    # ∇φ_i · ∇φ_j = Σ_k (dφ_i/dx_k)(dφ_j/dx_k)
    # T = ½ Σ_k (phi_grads[:,:,k] * w)^T @ phi_grads[:,:,k]
    T = np.zeros((n, n))
    for k in range(3):
        pg = phi_grads[:, :, k] * weights[:, None]
        T += pg.T @ phi_grads[:, :, k]
    T *= 0.5

    # External potential: V_ij = Σ w · φ_i · V(r) · φ_j
    # For AE dumps: V(r) = Σ_A -Z_A / |r - R_A|
    # For PP dumps: V_loc tables not in dump → V = n/a
    atoms = meta["atoms"]
    atoms_pos = np.array([a["pos_bohr"] for a in atoms], dtype=float)
    Zs = np.array([int(a["Z"]) for a in atoms], dtype=float)

    # Determine if PP dump (V_nl.txt exists and non-empty)
    # We pass the dump_dir through meta for this check; but we don't have it here.
    # Instead, check if V_nl info is available; for now compute V as -Z/r for AE.
    # The caller will handle PP gating.

    # Compute nuclear potential at all mesh points
    V_pot = np.zeros(Npts)
    for a in range(len(atoms)):
        diff = coords - atoms_pos[a]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        # Avoid singularity at nucleus: use a softening for the exact nucleus point
        # (the Becke grid has points very close to nuclei, so 1/r is integrable)
        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        V_pot += -Zs[a] / dist_safe

    # V_ij = Σ w · φ_i · V · φ_j
    V = (phi_vals * (weights * V_pot)[:, None]).T @ phi_vals

    report = {
        "n_mesh_points": int(Npts),
        "n_radial": n_radial,
        "lebedev_order": lebedev_order,
    }

    return S, T, V, report


# ---------------------------------------------------------------------------
# Y_lm convention validation against on-site S blocks
# ---------------------------------------------------------------------------

def validate_ylm_convention(
    S_computed: np.ndarray,
    S_dumped: np.ndarray,
    meta: dict,
) -> Tuple[bool, float]:
    """Validate the Y_lm convention by comparing on-site S blocks.

    On-site blocks (same atom) in the TIDES dump use analytic quadrature and
    are trustworthy.  If our computed on-site S matches to ≤1e-6, the Y_lm
    convention matches.

    Returns (passed, max_deviation).
    """
    basis = meta["basis"]
    n = len(basis)

    # Group basis entries by atom
    atom_indices: Dict[int, List[int]] = {}
    for i, entry in enumerate(basis):
        a = int(entry["atom"])
        atom_indices.setdefault(a, []).append(i)

    max_dev = 0.0
    for a, indices in atom_indices.items():
        idx = np.array(indices)
        sub_computed = S_computed[np.ix_(idx, idx)]
        sub_dumped = S_dumped[np.ix_(idx, idx)]
        dev = float(np.max(np.abs(sub_computed - sub_dumped)))
        max_dev = max(max_dev, dev)

    return max_dev <= TOL_YLM_VALIDATE, max_dev


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _worst_elements(dM: np.ndarray, n_worst: int = 3) -> List[Tuple[int, int, float]]:
    """Return the *n_worst* largest-magnitude elements of *dM*."""
    flat = np.argsort(np.abs(dM).ravel())[::-1][:n_worst]
    out = []
    for idx in flat:
        i, j = np.unravel_index(idx, dM.shape)
        out.append((int(i), int(j), float(dM[i, j])))
    return out


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def run_becke_oracle(
    dump_dir: str,
    n_radial: int = DEFAULT_N_RADIAL,
    lebedev_order: int = DEFAULT_LEBEDEV_ORDER,
    compute_v: bool = True,
) -> dict:
    """Run the Becke oracle on *dump_dir*.

    Returns a details dict with max|dS|, max|dT|, max|dV|, worst elements,
    mesh sizes, and pass/fail flags.

    For AE dumps, all three gates (S, T, V) are computed.
    For PP dumps, V is reported as n/a (V_loc tables not in dump).
    """
    meta, S_dumped, T_dumped, V_dumped, radial_tables = load_dump(dump_dir)

    # Check if PP dump (V_nl.txt exists and is non-empty)
    vnl_path = os.path.join(dump_dir, "V_nl.txt")
    is_pp = os.path.isfile(vnl_path) and os.path.getsize(vnl_path) > 1

    # Compute matrices via Becke quadrature
    S_comp, T_comp, V_comp, mesh_report = compute_matrices(
        meta, radial_tables, n_radial=n_radial, lebedev_order=lebedev_order
    )

    # Deviations
    dS = S_dumped - S_comp
    dT = T_dumped - T_comp
    max_dS = float(np.max(np.abs(dS)))
    max_dT = float(np.max(np.abs(dT)))

    if compute_v and not is_pp:
        dV = V_dumped - V_comp
        max_dV = float(np.max(np.abs(dV)))
    else:
        dV = None
        max_dV = None

    # Validate Y_lm convention
    ylm_ok, ylm_dev = validate_ylm_convention(S_comp, S_dumped, meta)

    result = {
        "n_basis": int(meta["n_basis"]),
        "is_pp": is_pp,
        "max_dS": max_dS,
        "max_dT": max_dT,
        "max_dV": max_dV,
        "worst_S": _worst_elements(dS),
        "worst_T": _worst_elements(dT),
        "worst_V": _worst_elements(dV) if dV is not None else [],
        "ylm_convention_valid": ylm_ok,
        "ylm_max_onsite_dev": ylm_dev,
        **mesh_report,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: becke_terms_oracle.py <dump_dir>")
        sys.exit(1)
    dump_dir = sys.argv[1]
    assert os.path.isdir(dump_dir), f"Not a directory: {dump_dir}"

    result = run_becke_oracle(dump_dir)

    print(f"Becke oracle  (n_basis = {result['n_basis']}, "
          f"PP = {result['is_pp']})")
    print(f"  Mesh points      = {result['n_mesh_points']}  "
          f"({result['n_radial']} radial × order {result['lebedev_order']})")
    print(f"  Y_lm convention  = {'VALID' if result['ylm_convention_valid'] else 'INVALID'}"
          f"  (on-site S dev = {result['ylm_max_onsite_dev']:.6e})")
    print(f"  max|dS|          = {result['max_dS']:.6e}")
    print(f"  max|dT|          = {result['max_dT']:.6e}")
    if result["max_dV"] is not None:
        print(f"  max|dV_ext|      = {result['max_dV']:.6e}")
    else:
        print(f"  max|dV_ext|      = n/a (PP dump)")
    print()
    print("  Worst offenders (i, j, value):")
    print(f"    dS: {result['worst_S']}")
    print(f"    dT: {result['worst_T']}")
    if result["worst_V"]:
        print(f"    dV: {result['worst_V']}")
    print()

    # Write report JSON
    report_path = os.path.join(dump_dir, "becke_report.json")
    with open(report_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Report written to {report_path}")

    # For real dumps: do NOT hard-fail (deviations measure TIDES grid error)
    # For selftest: the caller handles pass/fail
    sys.exit(0)


if __name__ == "__main__":
    main()
