---
Description: Phase 1/2/3 context brief for RALPH protocol execution
---

# Context Brief — TIDES NAO Product Path Hardening

## Objective
Execute Phase 1 (analytic two-center integrals), Phase 2 (physics completeness), and Phase 3 (architecture integration) to move the NAO product path from a grid-integrated prototype to a real DFT engine.

## Acceptance Criteria

### Phase 1 — Two-Center Integrals (A1)
- `E2` spline accuracy test passes: max spline error < 1.0e-5 (currently 3.5e-5).
- `E2` GPU two-center symmetry passes: max |S_ij - S_ji| < 1.0e-12 (currently 3.8e-3).
- `T2.4` rotation-invariance tests pass for l_a, l_b in {0, 1, 2}.
- `NaoDriver` assembles S and T using analytic two-center integrals (spline + Slater-Koster), not 3D grid integration.
- NAO SCF energy for H atom and H2 reaches PySCF reference within 0.01 Ha (currently 0.15 / 0.25 Ha).
- `tolerances.yaml` spline_value and rotation_invariance gates are met.

### Phase 2 — Physics Completeness (A2)
- SCF driver supports `nspin=2` (UKS) with distinct alpha/beta density matrices.
- `NaoDriver` can load real UPF2 pseudopotential files from a path.
- Pulay forces use full eigenvalue-weighted sum (sum_k eps_k C_k^T dS C_k), not eps_avg * Tr(P dS).

### Phase 3 — Architecture Integration (A3)
- `TileMat` is used for at least one matrix operation inside the SCF loop (P, H, or S product), not just stats.
- A dual-grid (coarse/fine) option exists and is used for Hartree/Poisson or XC.
- Stress tensor is computed and matches finite-difference strain to < 1.0e-6 Ha.

## Current Evidence
- `nao_driver.hpp` builds S/T by direct 3D grid integration (lines 438-486). This is O(n^2 * N_grid) and numerically noisy.
- `two_center_integrals.hpp` has `CubicSpline`, `RealSphericalHarmonics`, `TwoCenterAngularCoupling`, and `TwoCenterIntegralEval`, but `NaoDriver` does not use them.
- `TabulateOverlapSS` / `TabulateKineticSS` in `nao_driver.hpp` are s-s only and unused.
- `TwoCenterAngularCoupling` p-p, p-d, d-d are simplified (arbitrary channel splits). s-p, s-d are okay for a first pass but the angular normalization is not fully verified.
- `two_center.cu` uses `y_a * y_b` as angular factor, which is wrong for two-center integrals (should be Slater-Koster angular projection).
- `tolerances.yaml` wp2.two_center.spline_value = 1.0e-5, rotation_invariance = 1.0e-12.
- `molecule_driver_tests.cpp` tolerances: H2 0.2 Ha, He 0.1 Ha, H2O 3.0 Ha. NAO benchmark: H 0.15 Ha, H2 0.25 Ha.

## Dependencies
- `basis/two_center_integrals.hpp` (CPU math)
- `basis/two_center.cu` (GPU kernel)
- `core/scf/nao_driver.hpp` (integration point)
- `core/scf/scf_driver.hpp` (SCF loop, spin extension)
- `core/scf/molecule_driver.hpp` (force formula reference)
- `verification/tolerances.yaml` (gates)
- `tides/CMakeLists.txt` (build targets)

## Ambiguity / Risk
- Full Slater-Koster table for p-p, p-d, d-d requires careful Gaunt/Clebsch-Gordan factors. We will implement a verified set for s-s, s-p, p-p, s-d and document the rest.
- Real pseudopotential loading requires a working UPF2 reader (`basis/pseudo/upf2_reader.hpp`) and test files. We will add a file-loading path and a synthetic round-trip test.
- Tile substrate integration in SCF loop is a large change; we will implement a minimal P@H via TileMat for n >= 32 while keeping the dense path as fallback.
- Dual grid and stress are deferred behind the Phase 1/2 correctness gates.

## Phase Order
RALPH phases sequential: R -> A -> L -> P -> H. Phase 1 gates must pass before Phase 2/3 features are merged into the product path.
