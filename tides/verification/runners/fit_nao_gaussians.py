#!/usr/bin/env python3
"""Least-squares fit of NAO radial functions to even-tempered Gaussians.

Each radial function R(r) (the radial part of a numerical atomic orbital,
where the full orbital is  R(r) * Y_lm) is fitted to a sum of even-tempered
Gaussian primitives::

    R(r) ≈ Σ_k  c_k · r^l · exp(-α_k · r²)

The exponent ladder  {α_k}  is even-tempered (geometric progression) and
fixed; only the coefficients  {c_k}  are determined by linear least squares.

Because PySCF normalises *both* primitives and the contraction, the fitted
coefficients must be divided by the primitive normalisation constant before
being handed to PySCF.  The helper :func:`coeffs_for_pyscf` performs this
conversion so that the PySCF basis reproduces R(r) to fit precision.

No try/except — all error conditions are handled with assertions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import gamma as gamma_func

# Default even-tempered exponent ladder bounds (Bohr⁻²).
# α_min captures the most diffuse tail; α_max captures the core region.
# α_max is bounded by what the TIDES radial grid itself resolves: with
# dr = 0.02 Bohr the sharpest representable feature has width ~dr, i.e.
# α ~ 1/(2·dr²) ≈ 1e3; exponents far above that are invisible to the data
# and only ill-condition the fit.
DEFAULT_ALPHA_MIN = 0.01
DEFAULT_ALPHA_MAX = 100.0
DEFAULT_N_GAUSS = 14
# NOTE: widening this ladder (e.g. α_max 2e3, n 20) makes neighbouring
# even-tempered primitives near-collinear; the fit then carries huge ±
# coefficient pairs that S tolerates but T (~α·S) amplifies by ~α. Confined
# NAOs are fit to ~1e-2 relative at best (the r_cut kink is not Gaussian-
# representable), so this harness is a SANITY oracle at the few-% level.
# The precision oracle for S/T/V_ext is the Becke two-center quadrature
# runner (becke_terms_oracle.py), which uses the dumped radials directly.


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def primitive_normalization(alpha: float, l: int) -> float:
    """Normalisation constant *N* for the primitive  r^l · exp(-α·r²).

    The 3-D norm is::

        ∫ |N · r^l · exp(-α·r²) · Y_lm|² d³r  =  1

    Since ∫|Y_lm|² dΩ = 1, the radial integral must equal unity::

        N² · ∫₀^∞ r^{2l+2} · exp(-2α·r²) dr  =  1
        N  =  √[ 2 · (2α)^{l+3/2} / Γ(l+3/2) ]
    """
    half = l + 1.5
    return float(np.sqrt(2.0 * (2.0 * alpha) ** half / gamma_func(half)))


def even_tempered_exponents(
    n: int,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> np.ndarray:
    """Return *n* even-tempered exponents spanning ``[alpha_min, alpha_max]``.

    The ladder is geometric:  α_k = α_min · (α_max/α_min)^{k/(n-1)}.
    """
    assert n >= 1, "Need at least 1 Gaussian"
    if n == 1:
        return np.array([alpha_min])
    log_min = np.log(alpha_min)
    log_max = np.log(alpha_max)
    return np.exp(np.linspace(log_min, log_max, n))


def fit_radial_to_gaussians(
    r: np.ndarray,
    R: np.ndarray,
    l: int,
    n_gauss: int = DEFAULT_N_GAUSS,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit *R(r)* to  Σ_k c_k · r^l · exp(-α_k·r²)  (unnormalised primitives).

    Parameters
    ----------
    r : radial grid (1-D, Bohr).
    R : radial function values on *r*.
    l : angular-momentum quantum number.
    n_gauss : number of even-tempered Gaussians (12–16 recommended).

    Returns
    -------
    alphas : exponent ladder  (n_gauss,)
    coeffs : fitted coefficients for **unnormalised** primitives  (n_gauss,)
    residual : relative L² residual  √(∫|R−R_fit|²r²dr) / √(∫|R|²r²dr)
    """
    assert r.ndim == 1 and R.ndim == 1
    assert r.shape == R.shape
    assert len(r) > n_gauss, f"Need more grid points ({len(r)}) than Gaussians ({n_gauss})"
    assert np.all(r >= 0), "Radial grid must be non-negative"

    alphas = even_tempered_exponents(n_gauss, alpha_min, alpha_max)

    # Design matrix with *normalised* primitives so column norms are
    # comparable and SVD truncation (rcond) is meaningful — with a wide
    # exponent ladder an unregularised fit puts huge +/- coefficients on
    # near-degenerate steep primitives, which S tolerates but T (~ α·S)
    # amplifies catastrophically.
    norms = np.array([primitive_normalization(a, l) for a in alphas])
    A = np.empty((len(r), n_gauss))
    for k in range(n_gauss):
        A[:, k] = norms[k] * r ** l * np.exp(-alphas[k] * r ** 2)

    # Least squares weighted by r so the fit metric matches the 3-D norm
    # ∫|R−fit|² r² dr (an unweighted fit over-weights the far tail of the
    # uniform radial grid). rcond truncates the near-degenerate directions.
    w = r
    coeffs_n, _res, _rank, _sv = np.linalg.lstsq(A * w[:, None], R * w,
                                                 rcond=1e-10)
    coeffs = coeffs_n * norms  # back to unnormalised-primitive convention

    # Relative L² residual (with r² weight, i.e. the 3-D norm)
    R_fit = A @ coeffs_n  # A holds normalised primitives
    diff = R - R_fit
    num = float(np.sqrt(np.trapz(diff ** 2 * r ** 2, r)))
    den = float(np.sqrt(np.trapz(R ** 2 * r ** 2, r)))
    assert den > 0.0, "Zero-norm radial function — cannot fit"
    residual = num / den

    return alphas, coeffs, residual


def coeffs_for_pyscf(alphas: np.ndarray, coeffs: np.ndarray, l: int) -> List[List[float]]:
    """Convert fitted (unnormalised) coefficients to PySCF basis-entry format.

    PySCF applies primitive normalisation *N_k* internally, so we must pass
    ``c_k / N_k`` so that the PySCF contraction reproduces
    ``Σ_k c_k · r^l · exp(-α_k·r²)``  (the fitted function).

    Returns a list of ``[exp, coeff]`` pairs (the PySCF contraction format).
    """
    pairs: List[List[float]] = []
    for k in range(len(alphas)):
        nk = primitive_normalization(float(alphas[k]), l)
        pairs.append([float(alphas[k]), float(coeffs[k]) / nk])
    return pairs


# ---------------------------------------------------------------------------
# Dump-level helpers
# ---------------------------------------------------------------------------

def load_radial(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column radial file  *r  R(r)*."""
    data = np.loadtxt(filepath)
    assert data.ndim == 2 and data.shape[1] >= 2, (
        f"Expected ≥2 columns in {filepath}, got shape {data.shape}"
    )
    return data[:, 0], data[:, 1]


def fit_all_radials(
    meta: dict,
    dump_dir: str,
    n_gauss: int = DEFAULT_N_GAUSS,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, np.ndarray, np.ndarray, float]], float]:
    """Fit every unique radial function referenced in *meta['basis']*.

    Returns
    -------
    radial_fits : ``(atom, fn) → (l, alphas, coeffs, residual)``
    max_residual : worst relative residual across all functions
    """
    radial_fits: Dict[Tuple[int, int], Tuple[int, np.ndarray, np.ndarray, float]] = {}
    max_residual = 0.0

    for entry in meta["basis"]:
        key = (entry["atom"], entry["fn"])
        if key in radial_fits:
            continue
        filepath = os.path.join(dump_dir, f"radial_{entry['atom']}_{entry['fn']}.txt")
        r, R = load_radial(filepath)
        l = int(entry["l"])
        alphas, coeffs, residual = fit_radial_to_gaussians(r, R, l, n_gauss=n_gauss)
        radial_fits[key] = (l, alphas, coeffs, residual)
        max_residual = max(max_residual, residual)

    return radial_fits, max_residual


# ---------------------------------------------------------------------------
# CLI (optional — used by compare_pyscf_terms.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: fit_nao_gaussians.py <radial_file> <l> [n_gauss]")
        sys.exit(1)
    fp = sys.argv[1]
    ang_l = int(sys.argv[2])
    ng = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_GAUSS

    r_grid, R_vals = load_radial(fp)
    a, c, res = fit_radial_to_gaussians(r_grid, R_vals, ang_l, n_gauss=ng)
    print(f"Fit residual (rel L2): {res:.6e}")
    print(f"{'alpha':>14s}  {'coeff':>14s}")
    for ai, ci in zip(a, c):
        print(f"{ai:14.8f}  {ci:14.8f}")
