#pragma once

// Molecule driver: chains all TIDES components into an end-to-end SCF.
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
    grid.bc = {grid::BoundaryCondition::kPeriodic,
               grid::BoundaryCondition::kPeriodic,
               grid::BoundaryCondition::kPeriodic};

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

    // Step 6: SCF loop with real Hamiltonian build.
    // V_H is computed analytically via 4-center ERIs (exact for GTOs).
    // V_xc is computed on the grid from the electron density.
    auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
      // Scale P by 2 for spin degeneracy (SCFDriver uses occupation 1).
      std::vector<double> P2(n * n);
      for (std::size_t i = 0; i < n * n; ++i) P2[i] = 2.0 * P[i];

      // Analytic Coulomb (Hartree) matrix from ERIs.
      auto V_H = GTOIntegrals::CoulombMatrix(mol, P2);

      // Grid-based XC: build rho, evaluate XC, project to matrix.
      auto rho = grid::VmatBuilder::BuildRho(grid, orbitals, P2);
      auto xc = grid::XCGridEvaluator::EvaluateLDA(grid, rho);
      auto V_xc = grid::VmatBuilder::BuildHmat(grid, orbitals, xc.vxc);

      // Assemble H = T + V_ext + V_H + V_xc.
      std::vector<double> H(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) {
        H[i] = T[i] + V_ext[i] + V_H[i] + V_xc[i];
      }
      return H;
    };

    // The energy callback computes real energy components.
    auto energy_fn = [&](const std::vector<double>& P) -> double {
      std::vector<double> P2(n * n);
      for (std::size_t i = 0; i < n * n; ++i) P2[i] = 2.0 * P[i];

      // Analytic Coulomb matrix.
      auto V_H = GTOIntegrals::CoulombMatrix(mol, P2);

      // Grid-based XC.
      auto rho = grid::VmatBuilder::BuildRho(grid, orbitals, P2);
      auto xc = grid::XCGridEvaluator::EvaluateLDA(grid, rho);
      auto V_xc = grid::VmatBuilder::BuildHmat(grid, orbitals, xc.vxc);
      auto eps_xc_mat = grid::VmatBuilder::BuildHmat(grid, orbitals, xc.eps_xc);

      // Compute sum of occupied eigenvalues from P and H.
      auto H = build_H(P);
      auto eig = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
      double sum_eps = 0.0;
      if (eig.ok) {
        for (std::size_t k = 0; k < n_occ && k < n; ++k)
          sum_eps += 2.0 * eig.eigenvalues[k];  // factor 2 for spin
      }

      auto E = EnergyAssembly::Compute(
          sum_eps, P2, V_H, V_xc, eps_xc_mat, V_ext, S, n, E_ion);
      result.energy = E;
      return E.E_total;
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
    }

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
      for (auto& shell : shells) {
        shell.atom_index = static_cast<int>(a);
        n_basis += GTOIntegrals::NumCartesian(shell.l);
        mol.shells.push_back(shell);
      }
    }
    mol.n_basis = n_basis;
    return mol;
  }
};

}  // namespace tides::scf
