# TIDES P0.6 Verification Runners — PySCF Comparison Harness

This directory contains the Python verification harness for TIDES task **P0.6**
(per-term H-matrix dump & comparison against PySCF).

## Files

| File | Purpose |
|---|---|
| `fit_nao_gaussians.py` | Least-squares fit of each TIDES radial function R(r) to 12–16 even-tempered Gaussians `r^l · exp(-α·r²)`. Reports relative L² fit residual per function. |
| `compare_pyscf_terms.py` | Builds the same molecule in PySCF (custom basis from the fits, spherical, Bohr), computes S/T/V_nuc via `mol.intor`, **automatically** determines the TIDES→PySCF basis-ordering permutation and per-function sign mapping by matching the S matrix, then reports `max|dS|`, `max|dT|`, `max|dV_ext|` and worst offenders. Exit 0 iff all within tolerance. |
| `selftest_synthetic.py` | **Acceptance test.** Generates a synthetic dump (2 atoms, s+p functions from known Gaussians) where exact S/T/V are computable by PySCF, runs the full pipeline, and asserts it passes. **No TIDES C++ required.** |

## Dump Schema

Produced by TIDES with `TIDES_DUMP_HMAT_DIR=<dir>`:

- `meta.json`: `{"n_basis", "grid_h", "atoms": [{"Z", "pos_bohr"}], "basis": [{"atom", "l", "m", "fn"}]}`
- `S.txt`, `T.txt`, `V_ext.txt`: `n_basis²` doubles, row-major, one per line (`%.17g`)
- `radial_<atom>_<fn>.txt`: two whitespace columns `r R(r)` (Bohr, radial part; orbital = `R(r)·Y_lm`)

## Usage

### Self-test (acceptance)

```bash
python3 tides/verification/runners/selftest_synthetic.py
# Exits 0 with max errors printed on success.
```

### Compare a real TIDES dump

```bash
python3 tides/verification/runners/compare_pyscf_terms.py <dump_dir> [n_gauss]
# n_gauss defaults to 14.
```

### Fit a single radial function (debugging)

```bash
python3 tides/verification/runners/fit_nao_gaussians.py <radial_file> <l> [n_gauss]
```

## Tolerance

Defined in `tides/verification/tolerances.yaml` → `p0_6`:

```yaml
p0_6:
  s_max: 1.0e-5
  t_max: 1.0e-5
  vext_max: 1.0e-4
```

The effective per-run tolerance is `max(nominal, 10 × max_fit_residual)` so that
a looser Gaussian fit does not cause spurious failures.

## PySCF / MKL Environment

PySCF on this system links against Intel MKL.  When running PySCF-based scripts
directly (outside the self-test launcher, which sets this automatically), export:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so
```

## Design Notes

- **Basis format**: PySCF custom basis entries use the form
  `[l, [exp₁, coeff₁], [exp₂, coeff₂], …]` where each 2-element list is one
  primitive `(exp, coeff)`.  PySCF normalises **both** primitives and the
  contraction, so fitted (unnormalised) coefficients are divided by the
  primitive normalisation constant before being passed to PySCF
  (see `coeffs_for_pyscf` in `fit_nao_gaussians.py`).

- **Spherical harmonic ordering**: PySCF (libint) orders real spherical
  harmonics as `m = 0, +1, −1, +2, −2, …`.  For `l=1` this is `pz, px, py`.
  The permutation finder in `compare_pyscf_terms.py` handles this
  automatically by matching S-matrix rows.

- **Permutation & sign determination**: The TIDES basis order may differ from
  PySCF's, and individual functions may carry an overall sign.  The harness
  groups AOs by `(atom, l)`, uses the Hungarian algorithm on the
  sign-invariant S-matrix row correlations to find the optimal assignment,
  then propagates signs via BFS through the overlap graph (validated when S
  agrees).

- **No try/except**: Following the team standard (Rule 13 for C++; same
  principle here), all error conditions use assertions and explicit checks.
