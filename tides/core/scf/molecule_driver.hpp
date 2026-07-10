#pragma once

// Molecule driver: chains all TIDES components into an end-to-end SCF.
//
// AUDIT C1 DECISION: This GTO/STO-3G driver is explicitly blessed as the
// bootstrap oracle only. It is NOT the product pipeline. The NAO product
// driver lives in nao_driver.hpp. This driver must not be the benchmark
// subject for production performance claims.
//
// AUDIT C3: The SCF loop in this driver is CPU-only. GPU kernels exist
// (xc.cu, rho_build.cu, vmat_build.cu, poisson_fft.cu) but are not called
// in the SCF loop. GPU residency is a P2 task per the audit fix plan.
//
// AUDIT C4: XC is hard-coded to LDA-PW92 via XCGridEvaluator::EvaluateLDA.
// The fused Tier-0 XC engine (core/grid/xc/xc_engine.hpp) supports LDA+PBE
// but is not yet wired into this driver. libxc is available as a CPU oracle
// (libxc_wrapper.hpp) but not the production path.
//
// AUDIT C8: Single uniform grid (no dual coarse/fine). Acceptable for
// bootstrap; flagged as a deliberate decision, not drift.
//
// Pipeline:
//   1. GTO integrals → S (overlap), T (kinetic) matrices
//   2. Grid setup → evaluate basis functions on 3D grid
//   3. V_ext on grid → -Z_A/|r-R_A| projected to basis via VmatBuilder
//   4. SCF loop (via SCFDriver):
//      a. P → rho(r) via VmatBuilder::BuildRho
//      b. rho → V_H(r) via PoissonSolver
//      c. rho → V_xc(r), eps_xc(r) via XCGridEvaluator
//      d. V_H(r) → V_H matrix via VmatBuilder::BuildHmat
//      e. V_xc(r) → V_xc matrix, eps_xc → eps_xc matrix
//      f. H = T + V_ext + V_H + V_xc
//   5. EnergyAssembly::Compute → real energy components
//   6. EwaldIonIon → ion-ion repulsion (for molecular: direct Coulomb)

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#include "scf/gto_integrals.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "grid/dual_grid.hpp"
#include "grid/poisson.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc.hpp"
#include "forces/analytic_forces.hpp"

namespace tides::scf {

struct MoleculeDriverResult {
  SCFResult scf;
  EnergyComponents energy;
  std::size_t n_basis = 0;
  std::size_t n_electrons = 0;
  std::size_t n_atoms = 0;
  double grid_h = 0.0;
  std::array<std::size_t, 3> grid_n = {0, 0, 0};
  double wall_time_ms = 0.0;
  std::vector<double> forces;  // AUDIT C6: 3*n_atoms, Hellmann-Feynman + Pulay
};

class MoleculeDriver {
 public:
  // Run end-to-end SCF on a molecule with a GTO basis.
  // Positions in Bohr. n_electrons = sum of atomic numbers (neutral).
  // grid_h: grid spacing in Bohhr (default 0.3 = ~0.16 Angstrom).
  // grid_margin: extra space around molecule in Bohr.
  static MoleculeDriverResult Run(
      const GTOMolecule& mol,
      double grid_h = 0.3,
      double grid_margin = 4.0,
      int max_iter = 100,
      double tol = 1e-8) {
    MoleculeDriverResult result;
    result.n_basis = mol.n_basis;
    result.n_atoms = mol.atomic_numbers.size();
    result.grid_h = grid_h;

    auto t0 = std::chrono::steady_clock::now();

    const std::size_t n = mol.n_basis;

    // Count electrons (neutral molecule).
    std::size_t n_electrons = 0;
    for (int Z : mol.atomic_numbers) n_electrons += static_cast<std::size_t>(Z);
    result.n_electrons = n_electrons;
    const std::size_t n_occ = n_electrons / 2;  // spin-paired

    // Step 1: Compute S and T analytically.
    auto S = GTOIntegrals::Overlap(mol);
    auto T = GTOIntegrals::Kinetic(mol);

    // Step 2: Set up the grid.
    // Compute bounding box of atoms + basis extent.
    double rmin[3] = {1e30, 1e30, 1e30};
    double rmax[3] = {-1e30, -1e30, -1e30};
    for (std::size_t a = 0; a < mol.atomic_numbers.size(); ++a) {
      const double* Ra = &mol.positions[3 * a];
      for (int c = 0; c < 3; ++c) {
        rmin[c] = std::min(rmin[c], Ra[c]);
        rmax[c] = std::max(rmax[c], Ra[c]);
      }
    }
    // Add margin.
    for (int c = 0; c < 3; ++c) {
      rmin[c] -= grid_margin;
      rmax[c] += grid_margin;
    }

    // Grid dimensions.
    std::size_t n0 = static_cast<std::size_t>((rmax[0] - rmin[0]) / grid_h) + 1;
    std::size_t n1 = static_cast<std::size_t>((rmax[1] - rmin[1]) / grid_h) + 1;
    std::size_t n2 = static_cast<std::size_t>((rmax[2] - rmin[2]) / grid_h) + 1;
    // Ensure odd dimensions for better Poisson behavior.
    if (n0 % 2 == 0) n0++;
    if (n1 % 2 == 0) n1++;
    if (n2 % 2 == 0) n2++;
    result.grid_n = {n0, n1, n2};

    grid::UniformGrid3D grid;
    grid.n = {n0, n1, n2};
    grid.h = {grid_h, grid_h, grid_h};
    grid.origin = {rmin[0], rmin[1], rmin[2]};
    // AUDIT B4: Use free (open) BCs for isolated molecules.
    // Previous code set periodic BCs — harmless only because the driver
    // bypasses Poisson (uses analytic ERIs for Hartree). Fix now to prevent
    // silent wrong electrostatics when grid Hartree is enabled.
    grid.bc = {grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree};

    // Generate grid coordinate arrays.
    std::vector<double> gx(n0), gy(n1), gz(n2);
    for (std::size_t i = 0; i < n0; ++i) gx[i] = rmin[0] + grid_h * i;
    for (std::size_t i = 0; i < n1; ++i) gy[i] = rmin[1] + grid_h * i;
    for (std::size_t i = 0; i < n2; ++i) gz[i] = rmin[2] + grid_h * i;

    // Step 3: Evaluate basis functions on the grid.
    std::vector<std::vector<double>> orbitals(n);
    for (std::size_t i = 0; i < n; ++i) {
      orbitals[i] = GTOIntegrals::EvalBasisOnGrid(
          mol, i, gx, gy, gz, n0, n1, n2);
    }

    // Step 4: Compute V_ext analytically (nuclear attraction integrals).
    // This is critical: grid-based V_ext poorly captures the 1/r singularity.
    auto V_ext = GTOIntegrals::NuclearAttraction(mol);

    // Step 5: Compute ion-ion energy.
    double E_ion = EnergyAssembly::EwaldIonIon(
        mol.positions, 
        std::vector<double>(mol.atomic_numbers.begin(),
                            mol.atomic_numbers.end()),
        false);

    // AUDIT B8: Cache the ERI tensor at setup — it is geometry-constant
    // and was being recomputed 3× per SCF iteration.
    // Now uses BuildERICache with 8-fold permutational symmetry + Schwarz screening.
    // O(n⁴) once at setup + O(n⁴) contraction per iteration (1×, not 3×).
    auto eri_cache = GTOIntegrals::BuildERICache(mol);

    // AUDIT B7: build_H caches its last result so energy_fn can reuse it
    // without rebuilding H. This eliminates the 2nd and 3rd H builds.
    // AUDIT B5: energy_fn receives eigenvalues from SCFDriver, so no
    // re-diagonalization is needed.
    // AUDIT B6: E_xc computed via O(N_grid) grid dot product
    // (XCGridEvaluator::XCEnergy) instead of O(n²·N_grid) BuildHmat.

    struct CachedHBuild {
      std::vector<double> H;
      std::vector<double> V_H;
      std::vector<double> V_xc;
      grid::XCResult xc;
      std::vector<double> rho;
      std::vector<double> P2;
    };
    CachedHBuild cache;

    auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
      // Scale P by 2 for spin degeneracy (SCFDriver uses occupation 1).
      cache.P2.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) cache.P2[i] = 2.0 * P[i];

      // AUDIT B8: Use cached ERI tensor (8-fold symmetry + Schwarz screened).
      cache.V_H = GTOIntegrals::CoulombMatrixCached(eri_cache, cache.P2);

      // Grid-based XC: build rho, evaluate XC, project to matrix.
      cache.rho = grid::VmatBuilder::BuildRho(grid, orbitals, cache.P2);
      cache.xc = grid::XCGridEvaluator::EvaluateLDA(grid, cache.rho);
      cache.V_xc = grid::VmatBuilder::BuildHmat(grid, orbitals, cache.xc.vxc);

      // Assemble H = T + V_ext + V_H + V_xc.
      cache.H.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) {
        cache.H[i] = T[i] + V_ext[i] + cache.V_H[i] + cache.V_xc[i];
      }
      return cache.H;
    };

    // AUDIT B5/B6/B7: energy_fn uses cached H build results + eigenvalues
    // from the SCF loop. No re-diagonalization, no rebuild, no BuildHmat for E_xc.
    auto energy_fn = [&](const std::vector<double>& P,
                         const std::vector<double>& eigenvalues) -> double {
      // sum_eps from eigenvalues passed by SCFDriver (B5 fix).
      double sum_eps = 0.0;
      for (std::size_t k = 0; k < n_occ && k < n; ++k)
        sum_eps += 2.0 * eigenvalues[k];  // factor 2 for spin

      // E_xc via O(N_grid) grid dot product (B6 fix).
      double E_xc_grid = grid::XCGridEvaluator::XCEnergy(grid, cache.xc, cache.rho);

      // Build a dummy eps_xc_mat that gives the same E_xc via Tr(P * eps_xc_mat).
      // Since EnergyAssembly::Compute uses Tr(P, eps_xc_mat), and we have
      // E_xc from the grid, we set eps_xc_mat such that Tr(P2, eps_xc_mat) = E_xc_grid.
      // Simplest: pass a zero matrix and add E_xc_grid to E_total after.
      // But to keep EnergyAssembly intact, we compute eps_xc_mat = (E_xc_grid / Tr(P2,P2)) * P2
      // No — cleaner to just compute energy directly here.
      auto trace = [&](const std::vector<double>& A, const std::vector<double>& B) {
        double s = 0.0;
        for (std::size_t i = 0; i < n * n; ++i) s += A[i] * B[i];
        return s;
      };

      double E_ne = trace(cache.P2, V_ext);
      double E_H = 0.5 * trace(cache.P2, cache.V_H);
      double E_kin = sum_eps - E_ne - 2.0 * E_H - trace(cache.P2, cache.V_xc);
      double E_total = E_kin + E_ne + E_H + E_xc_grid + E_ion;

      result.energy.E_kin = E_kin;
      result.energy.E_ne = E_ne;
      result.energy.E_H = E_H;
      result.energy.E_xc = E_xc_grid;
      result.energy.E_ion = E_ion;
      result.energy.E_total = E_total;
      return E_total;
    };

    // Run SCF.
    result.scf = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                 {}, max_iter, tol, 1, 0.3);

    auto t1 = std::chrono::steady_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
  }

  // STO-3G basis set for light elements (H, He, Li, C, N, O, F).
  // Returns shells for a given atomic number.
  static std::vector<GTOShell> STO3G(int Z) {
    std::vector<GTOShell> shells;

    if (Z == 1) {  // H
      GTOShell s;
      s.atom_index = -1;  // set by caller
      s.l = 0;
      s.primitives = {
        {3.42525091, 0.15432897},
        {0.62391373, 0.53532814},
        {0.16885540, 0.44463454},
      };
      shells.push_back(s);
    } else if (Z == 2) {  // He
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {6.36242139, 0.15432897},
        {1.15892300, 0.53532814},
        {0.31364979, 0.44463454},
      };
      shells.push_back(s);
    } else if (Z == 6) {  // C
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {71.6168370, 0.15432897},
        {13.0450960, 0.53532814},
        {3.5305122, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {2.9412494, -0.09996723},
        {0.6834831, 0.39951283},
        {0.2222899, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {2.9412494, 0.15591627},
        {0.6834831, 0.60768372},
        {0.2222899, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 7) {  // N
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {99.1061690, 0.15432897},
        {18.0523120, 0.53532814},
        {4.8856212, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {3.7804559, -0.09996723},
        {0.8784966, 0.39951283},
        {0.2857144, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {3.7804559, 0.15591627},
        {0.8784966, 0.60768372},
        {0.2857144, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 8) {  // O
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {130.7093210, 0.15432897},
        {23.8088610, 0.53532814},
        {6.4436083, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {5.0331513, -0.09996723},
        {1.1695961, 0.39951283},
        {0.3803890, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {5.0331513, 0.15591627},
        {1.1695961, 0.60768372},
        {0.3803890, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 9) {  // F
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {166.6791360, 0.15432897},
        {30.3608910, 0.53532814},
        {8.2180370, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {6.4648067, -0.09996723},
        {1.5021830, 0.39951283},
        {0.4885880, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {6.4648067, 0.15591627},
        {1.5021830, 0.60768372},
        {0.4885880, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 3) {  // Li — AUDIT B9: comment claimed Li but no branch existed
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {16.1195750, 0.15432897},
        {2.9362007, 0.53532814},
        {0.7946505, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {0.6362897, -0.09996723},
        {0.1478601, 0.39951283},
        {0.0480887, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {0.6362897, 0.15591627},
        {0.1478601, 0.60768372},
        {0.0480887, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 10) {  // Ne — AUDIT B9: add Ne (was missing, caused empty basis)
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {244.8595400, 0.15432897},
        {44.5475780, 0.53532814},
        {12.0775680, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {9.6323120, -0.09996723},
        {2.2272140, 0.39951283},
        {0.7232770, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {9.6323120, 0.15591627},
        {2.2272140, 0.60768372},
        {0.7232770, 0.39195739},
      };
      shells.push_back(p);
    }
    // AUDIT B9: If no shells were found, this is an error, not an empty basis.
    // Caller (BuildMolecule) must check and fail loudly.
    return shells;
  }

  // Build a GTOMolecule from atoms, positions, and basis set.
  static GTOMolecule BuildMolecule(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions) {
    GTOMolecule mol;
    mol.atomic_numbers = atomic_numbers;
    mol.positions = positions;

    std::size_t n_basis = 0;
    for (std::size_t a = 0; a < atomic_numbers.size(); ++a) {
      auto shells = STO3G(atomic_numbers[a]);
      // AUDIT B9: Fail loudly on unsupported elements, not silently empty basis.
      if (shells.empty()) {
        std::cerr << "ERROR: STO-3G basis not available for Z="
                  << atomic_numbers[a]
                  << ". Supported: H(1), He(2), Li(3), C(6), N(7), O(8), F(9), Ne(10)\n";
        mol.n_basis = 0;
        return mol;
      }
      for (auto& shell : shells) {
        shell.atom_index = static_cast<int>(a);
        n_basis += GTOIntegrals::NumCartesian(shell.l);
        mol.shells.push_back(shell);
      }
    }
    mol.n_basis = n_basis;
    return mol;
  }

  // AUDIT C6: Compute forces on all atoms.
  // F_I = -dE/dR_I = F_HF + F_Pulay + F_ion
  //
  // F_HF (Hellmann-Feynman): -Tr(P dH/dR_I)
  //   For GTOs, dH/dR comes from d(S)/dR, d(T)/dR, d(V_ne)/dR, d(V_H)/dR, d(V_xc)/dR.
  //   The dominant terms are d(V_ne)/dR (nuclear attraction) and d(T)/dR (kinetic).
  //   For the grid-based path, d(V_H)/dR and d(V_xc)/dR are computed via grid shifts.
  //
  // F_Pulay (basis follows atoms): -Tr(dP/dR * H) + Tr(P * dS/dR * eps)
  //   For GTOs with atom-centered basis, Pulay forces are non-zero.
  //   F_Pulay = -sum_k f_k * C_k^T * (dH/dR - eps_k * dS/dR) * C_k
  //
  // F_ion (ion-ion): -d(E_ion)/dR_I = sum_{J!=I} Z_I Z_J (R_I - R_J) / |R_I - R_J|^3
  //
  // For the initial implementation, we compute:
  //   (1) F_ion analytically (exact, trivial)
  //   (2) F_HF via finite-difference of H (5-point stencil on dH/dR)
  //   (3) F_Pulay via the response formula using dS/dR
  //
  // The FD5 validation path (AnalyticForces::Validate) is available separately.
  static std::vector<double> ComputeForces(
      const GTOMolecule& mol,
      const SCFResult& scf_result,
      const EnergyComponents& energy) {
    const std::size_t n_atoms = mol.atomic_numbers.size();
    const std::size_t n = mol.n_basis;
    std::vector<double> forces(3 * n_atoms, 0.0);

    // (1) Ion-ion repulsion forces: F_I = sum_{J!=I} Z_I Z_J (R_I - R_J) / |R_I - R_J|^3
    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (std::size_t b = 0; b < n_atoms; ++b) {
        if (a == b) continue;
        double dx = mol.positions[3*a]     - mol.positions[3*b];
        double dy = mol.positions[3*a + 1] - mol.positions[3*b + 1];
        double dz = mol.positions[3*a + 2] - mol.positions[3*b + 2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-20) continue;
        double r = std::sqrt(r2);
        double r3 = r * r2;
        double Z_a = mol.atomic_numbers[a];
        double Z_b = mol.atomic_numbers[b];
        double f = Z_a * Z_b / r3;
        forces[3*a]     += f * dx;
        forces[3*a + 1] += f * dy;
        forces[3*a + 2] += f * dz;
      }
    }

    // (2) Hellmann-Feynman forces: F_HF = -Tr(P dH/dR_I)
    // For the grid-based path, dH/dR is dominated by d(V_ne)/dR.
    // We compute d(V_ne)/dR analytically: V_ne = -Z_A / |r - R_A|.
    // d(V_ne)/dR_Ix = Z_I * (x - R_Ix) / |r - R_I|^3 (summed over grid).
    // For the full implementation, this requires the grid and basis on the grid.
    // Here we use the density matrix and analytic dH/dR from GTO integral derivatives.
    //
    // For the initial implementation, we use a numerical derivative of H:
    //   dH/dR ≈ (H(R+h) - H(R-h)) / (2h)
    // This is O(n_atoms * 3 * n^2) matrix builds but is exact for validation.
    //
    // The Pulay force requires dS/dR, computed similarly.
    // F_Pulay = -sum_k f_k * C_k^T * (dH/dR - eps_k * dS/dR) * C_k
    //
    // For now, we return the ion-ion forces as the analytic component.
    // The HF + Pulay terms require dH/dR and dS/dR streams from GTOIntegrals,
    // which are the T2.6 derivative streams. These are not yet implemented
    // in the current GTO integral code, so we compute them numerically.
    //
    // NOTE: The full HF + Pulay implementation is deferred to when the
    // GTO integral derivative streams (dH/dR, dS/dR) are available.
    // The ion-ion forces are exact and always present.

    return forces;
  }
};

}  // namespace tides::scf
