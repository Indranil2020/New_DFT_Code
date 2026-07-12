---
Description: Phase A architecture plan for RALPH protocol execution
---

# Architecture Plan — Phase 1/2/3

## Phase 1 — Analytic Two-Center Integrals

### Components
1. `basis/two_center_integrals.hpp` (extension):
   - `SlaterKosterTable`: precomputed angular factors for (l_a, m_a, l_b, m_b) pairs.
   - `NaoTwoCenterBuilder` or `TabulateOverlap` / `TabulateKinetic` functions that operate on `NaoBasisFunction` and support all (l_a, l_b) up to l=2.
   - `BuildTwoCenterMatrix(S, T, ...)` that takes a list of atoms + basis mapping and returns dense S/T.

2. `basis/two_center.cu` (GPU fix):
   - Replace `y_a * y_b` with `SlaterKosterAngular(l_a, m_a, l_b, m_b, theta, phi)`.
   - Ensure pair (a,b) and (b,a) emit exactly symmetric contributions (or launch only upper-triangle and mirror atomically with consistent rounding).
   - For s-s: angular factor = 1.0. For s-p: angular = Y_{1m} projected on bond. For p-p: sigma + pi split. For s-d: Y_{2m} projection.

3. `core/scf/nao_driver.hpp` (integration):
   - Replace grid-integrated S/T (lines 438-486) with a call to the analytic two-center builder.
   - Keep grid-orbital evaluation for `VmatBuilder` (rho, V_ext, V_xc, V_H) because the grid XC/Poisson path still needs it.
   - Optionally keep a fallback `use_grid_integrals` flag for cross-validation.

4. `tests/per_engine/e2_basis/e2_test_profile.cpp` / `core/basis/tests/two_center_tests.cpp`:
   - Add Slater-Koster rotation-invariance tests for (l_a, l_b) pairs.
   - Retighten spline gate to 1e-5 and GPU symmetry gate to 1e-12.

### Interfaces
```cpp
// Radial table for a pair of NAO functions.
struct NaoTwoCenterTable {
  CubicSpline S;       // overlap vs R
  CubicSpline T;       // kinetic vs R
  int l_a, l_b;
};

// Build a radial table from two NaoBasisFunction.
NaoTwoCenterTable TabulateNaoTwoCenter(
    const NaoBasisFunction& fa,
    const NaoBasisFunction& fb,
    int n_R = 1024);

// Assemble dense S/T for all atom/basis pairs.
void AssembleTwoCenterMatrices(
    const std::vector<NaoAtom>& atoms,
    const std::vector<BasisIdx>& basis_map,
    std::vector<double>& S,
    std::vector<double>& T);

// Slater-Koster angular factor.
double SlaterKosterAngular(int l_a, int m_a, int l_b, int m_b,
                           double cos_theta, double phi);
```

### Phase 1 Acceptance Gates
- `two_center_tests` passes.
- `e2_test_profile` spline error < 1e-5 and GPU symmetry < 1e-12.
- `nao_driver_tests` and `nao_benchmark_tests` reach tolerances < 0.01 Ha.

## Phase 2 — Physics Completeness

### Components
1. `core/scf/scf_driver.hpp`:
   - Add `nspin` parameter to `SCFResult` and `Run`.
   - For `nspin=2`, maintain P_alpha and P_beta, solve generalized eigenproblem once (same H, S) for both spins, occupy n_occ_alpha and n_occ_beta lowest orbitals.
   - Pass `nspin` into `energy_fn`.

2. `core/scf/nao_driver.hpp`:
   - Accept `nspin` parameter (default 1).
   - For `nspin=2`, set n_occ_alpha = n_electrons/2, n_occ_beta = (n_electrons + 1)/2, or user-provided.
   - XC engine already supports `nspin=2`; `xc_in.nspin` must be set.

3. `basis/pseudo/upf2_reader.hpp`:
   - Add `LoadPseudopotentialsFromPaths(std::vector<std::string> paths)` returning `std::vector<Pseudopotential>`.
   - Validate norm conservation and ghost detector.

4. `core/scf/molecule_driver.hpp`:
   - Replace diagonal Pulay approximation with full sum over occupied orbitals: `sum_k eps_k * (C_k^T dS C_k)`.
   - Coefficients C_k are in `SCFResult.eigenvectors`.

### Phase 2 Acceptance Gates
- New `spin_tests` or `nao_driver` with `nspin=2` runs and produces correct H atom doublet energy.
- `pseudo_tests` loads a real UPF2 file and validates.
- `molecule_driver` forces match FD5 within 1e-6 Ha/Bohr.

## Phase 3 — Architecture Integration

### Components
1. `core/scf/nao_driver.hpp`:
   - For `n >= 32`, convert S, H, P to `TileMat` and use `SpGemmFilteredFp64` for `trace(P@H)` inside the SCF loop or in `energy_fn`.
   - Set `result.tile_substrate_used = true` when a tile operation is used for convergence/energy.

2. `grid/dual_grid.hpp`:
   - Extend `DualGrid` to store a coarse grid (for Poisson) and a fine grid (for rho/XC).
   - Implement restriction and prolongation between grids.
   - Use coarse grid for `PoissonSolver::Solve` and fine grid for `VmatBuilder` and `XCGridEvaluator`.

3. `core/scf/nao_driver.hpp` (stress):
   - Compute stress tensor via finite differences of the total energy with respect to uniform lattice strain (3x3 matrix).
   - Use central differences: `dE/deps_{ij}`.

### Phase 3 Acceptance Gates
- `tile` operation is in the SCF loop (not just stats) and trace error < 1e-12 vs dense.
- Dual grid Poisson/XC energy difference vs single fine grid < 1e-6 Ha.
- Stress tensor passes FD strain check within 1e-6 Ha.

## Sequencing
1. Phase 1 (all gates pass) before Phase 2/3.
2. Phase 2/3 can be partially parallelized once Phase 1 correctness is established.
3. Each Phase L item is followed by a Phase P test run; loop to L if gates fail.
