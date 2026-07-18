# Phase 0 Diagnostic — PP CPU semi-on-site correction

## Objective
Confirm the cause of non-physical energies / divergence for CH4 and H2O in the
TIDES CPU PP path (`BuildAnalyticVextPP`) and verify whether disabling the
semi-on-site (and on-site analytic) corrections makes CPU PP match GPU PP.

## Method
Added environment-variable gates in `core/scf/nao_driver.hpp`:

- `TIDES_PP_SEMIONSITE=0` — disables the semi-on-site analytic correction loop.
- `TIDES_PP_ONSITE=0` — disables the analytic on-site radial-integral replacement.

Default behaviour for both is **enabled** (unset or any value other than `0`).

GPU PP path is unchanged and is the reference.

## Key results

### 1. Semi-on-site is the dominant CPU PP bug
| system | grid_h | GPU PP default | CPU PP default | CPU PP semi=0 |
|--------|--------|----------------|----------------|---------------|
| CH4    | 0.3    | -8.3424        | -114.7 (non-physical) | -8.2137 |
| H2O    | 0.3    | -17.9469       | diverges       | -17.7053 |
| H2     | 0.3    | -1.1058        | -1.4058        | -1.0768 |

Setting `TIDES_PP_SEMIONSITE=0` restores physical, converged energies.

### 2. On-site replacement has almost no effect
For CH4 / H2O with semi=0, toggling `TIDES_PP_ONSITE` changes the total energy
by ~0.004 Ha. It is not the source of the route-specific error.

### 3. CPU PP can match GPU PP under the gate — cubic grids
CH4 at `grid_h=0.2`:
- GPU: `-7.932225474872737`
- CPU semi=0, onsite=0: `-7.932225474872496`
- difference: `2.4e-13 Ha`

This proves that, with the semi-on-site correction disabled, the CPU and GPU
projection paths are numerically equivalent for at least one realistic system.

### 4. H2O / O residual is not the semi-on-site bug
For H2O at `grid_h=0.2` CPU semi=0 still differs from GPU by ~0.23 Ha.
Investigation shows:

- `tides_cuda_pp_build_tests` passes (`BuildVlocDevice` and
  `BuildWeightedVmatDevice` match the CPU reference to round-off, including
  for real pseudopotentials).
- For a single O atom the printed `V_ext` diagonal and `H=T+V_ext`
  eigenvalues are identical between CPU and GPU, but the SCF initial energy
  (`E_ne` and `P_trace`) differs at `iter=0` despite an identical Hamiltonian.
- The residual therefore appears to be downstream of the V_ext projection,
  most likely in the SAD initial-density / eigensolver degeneracy handling
  rather than in `BuildAnalyticVextPP` itself.

### 5. Unit-test status after the diagnostic changes
- `tides_real_pp_scf_tests`: **PASS** (deterministic, |dE| ~1e-15).
- `tides_cuda_pp_build_tests`: **PASS** (round-off).
- `tides_nao_driver_tests`: 4 pre-existing failures (H2 AE, H2 dual-grid AE,
  XL-BOMD shadow forces, HSE06 correction).
- `tides_pp_scf_validation_tests`: 1 pre-existing failure (He PP vs AE
  discrepancy).
- `tides_nao_pseudo_tests`: 1 pre-existing failure (H2 PP vs AE discrepancy).

These failures are in all-electron, hybrid-functional, or He-only paths and are
not introduced by the PP env gates.

## Conclusion
The CPU PP non-physical energies for CH4/H2O are caused by the semi-on-site
analytic correction in `BuildAnalyticVextPP`. Disabling it (`TIDES_PP_SEMIONSITE=0`)
makes CPU PP energies physically reasonable and lets CH4 match the GPU PP
reference to round-off.

A small residual (~0.23 Ha) remains for H2O at `grid_h=0.2`; it is not removed
by disabling the on-site replacement and does not correlate with the V_ext
projection kernel (which validates to machine precision). It should be tracked
as a separate SCF/eigensolver degeneracy or initial-density issue, not as part
of the PP CPU semi-on-site bug.

## Recommendations
1. Fix the semi-on-site correction physics (frame-consistent spherical-harmonic
   evaluation or replace with a dense Lebedev radial/quadrature oracle).
2. Keep the env gates until the fix is verified, to permit quick A/B testing.
3. Investigate the H2O/O residual separately, starting from the SAD initial
   guess and eigensolver output for degenerate p-shells.
