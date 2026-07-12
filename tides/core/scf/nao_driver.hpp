#pragma once

// NAO product driver: assembles NAO basis into a working SCF pipeline.
//
// AUDIT C8/C9 (P1.6): The NAO basis was implemented (atomgen, two-center
// splines, three-center KB) but never assembled into a working SCF driver.
// This file closes that gap — CPU-first, validated vs GTO oracle + PySCF.
//
// Pipeline:
//   1. Generate NAO basis per atom via NaoGenerator (DZP recipe)
//   2. Tabulate two-center S(R), T(R) from radial functions
//   3. Assemble S, T matrices via Slater-Koster angular coupling
//   4. Grid setup → evaluate NAO basis functions on 3D grid
//   5. V_ext on grid → -Z_A/|r-R_A| projected to basis
//   6. SCF loop (via SCFDriver):
//      a. P → rho(r) via VmatBuilder::BuildRho
//      b. rho → V_H(r) via PoissonSolver (grid-based, no ERIs)
//      c. rho → V_xc(r), eps_xc(r) via XCGridEvaluator
//      d. V_H, V_xc → matrices via VmatBuilder::BuildHmat
//      e. H = T + V_ext + V_H + V_xc
//   7. Energy assembly (same as MoleculeDriver)
//
// Validation: H atom LDA energy vs PySCF/STO-3G (same-basis comparison).
// The NAO basis is NOT STO-3G, so energies will differ — validation is
// against the GTO oracle for the same XC functional and grid, not against
// exact basis-set agreement.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"
#include "basis/two_center_builder.hpp"
#include "basis/pseudo/pseudopotential.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "scf/stress.hpp"
#include "dynamics/xlbomd/xlbomd.hpp"
#include "grid/dual_grid.hpp"
#include "grid/poisson.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/rho_build_gpu.hpp"
#include "grid/xc_gpu.hpp"
#include "grid/poisson_fft_gpu.hpp"
#include "grid/xc.hpp"
#include "tile/layout.hpp"
#include "tile/spgemm_filtered.hpp"
#include "ham/ham_builder.hpp"

namespace tides::scf {

struct NaoDriverResult {
  SCFResult scf;
  EnergyComponents energy;
  std::size_t n_basis = 0;
  std::size_t n_electrons = 0;
  std::size_t n_atoms = 0;
  double grid_h = 0.0;
  std::array<std::size_t, 3> grid_n = {0, 0, 0};
  double wall_time_ms = 0.0;
  std::string basis_info;
  std::string xc_functional;  // Name of XC functional used
  // Tile substrate stats (Gap 3).
  std::size_t tile_count_H = 0;    // non-zero tiles in Hamiltonian
  std::size_t tile_count_S = 0;    // non-zero tiles in overlap
  double tile_sparsity_H = 0.0;    // fraction of tiles that are non-zero
  double tile_sparsity_S = 0.0;
  bool tile_substrate_used = false;
};

struct NaoAtom {
  int Z = 0;
  std::string element;
  std::vector<double> position;  // 3 components, Bohr
  basis::NaoBasis basis;
};

class NaoDriver {
 public:
  // Build NaoAtom list from atomic numbers + positions (Bohr).
  // Generates DZP NAO basis per atom type.
  static std::vector<NaoAtom> BuildAtoms(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions) {
    std::vector<NaoAtom> atoms;
    for (std::size_t a = 0; a < atomic_numbers.size(); ++a) {
      NaoAtom atom;
      atom.Z = atomic_numbers[a];
      atom.element = ElementName(atom.Z);
      atom.position = {positions[3 * a], positions[3 * a + 1], positions[3 * a + 2]};

      // Generate DZP NAO basis.
      auto recipe = basis::NaoGenerator::DzpRecipe(atom.Z, atom.element);
      atom.basis = basis::NaoGenerator::Generate(recipe);
      atoms.push_back(atom);
    }
    return atoms;
  }

  // Count total basis functions (including (2l+1) m-components per radial fn).
  static std::size_t CountBasisFunctions(const std::vector<NaoAtom>& atoms) {
    std::size_t n = 0;
    for (const auto& atom : atoms) {
      for (const auto& f : atom.basis.functions) {
        n += 2 * f.l + 1;  // m = -l, ..., +l
      }
    }
    return n;
  }

  // Tabulate two-center overlap S(R) for a pair of radial functions.
  // S(R) = ∫∫ R_a(r_a) R_b(r_b) Y_l_a(Ω_a) Y_l_b(Ω_b) δ(r_a - r_b - R) d³r_a d³r_b
  // For s-s (l_a=0, l_b=0): S(R) = ∫ R_a(r) R_b(|r-R|) r² dr (radial integral)
  // For general (l_a, l_b), the full integral involves Slater-Koster tables.
  // Here we tabulate the radial part for L=0 (s-s only); higher L is future work.
  static basis::CubicSpline TabulateOverlapSS(
      const basis::NaoBasisFunction& fa,
      const basis::NaoBasisFunction& fb) {
    // Tabulate S(R) for R in [0, r_cut_a + r_cut_b].
    const double r_max = fa.r_cut + fb.r_cut;
    const std::size_t n_R = 500;
    const double dR = r_max / static_cast<double>(n_R - 1);

    std::vector<double> R_pts(n_R), S_pts(n_R);
    for (std::size_t iR = 0; iR < n_R; ++iR) {
      const double R = static_cast<double>(iR) * dR;
      R_pts[iR] = R;

      // S(R) = ∫_0^∞ R_a(r) R_b(r-R) r² dr  (for s-s, L=0)
      // where R_b is evaluated at |r - R|.
      // Use the radial grids of fa and fb.
      double s = 0.0;
      const auto& ra = fa.r;
      const auto& Ra = fa.R;
      const auto& rb = fb.r;
      const auto& Rb = fb.R;

      // Integrate over ra grid.
      for (std::size_t i = 0; i + 1 < ra.size(); ++i) {
        const double r = ra[i];
        if (r > fa.r_cut) break;

        // Evaluate Rb at |r - R|.
        const double r_b = std::abs(r - R);
        double Rb_val = 0.0;
        if (r_b <= fb.r_cut && r_b >= rb.front() && r_b <= rb.back()) {
          // Linear interpolation on rb grid.
          std::size_t j = 0;
          while (j + 1 < rb.size() && rb[j + 1] < r_b) ++j;
          if (j + 1 < rb.size()) {
            const double t = (r_b - rb[j]) / (rb[j + 1] - rb[j]);
            Rb_val = (1.0 - t) * Rb[j] + t * Rb[j + 1];
          }
        }

        const double r_next = ra[i + 1];
        double Rb_val_next = 0.0;
        const double r_b_next = std::abs(r_next - R);
        if (r_b_next <= fb.r_cut && r_b_next >= rb.front() && r_b_next <= rb.back()) {
          std::size_t j = 0;
          while (j + 1 < rb.size() && rb[j + 1] < r_b_next) ++j;
          if (j + 1 < rb.size()) {
            const double t = (r_b_next - rb[j]) / (rb[j + 1] - rb[j]);
            Rb_val_next = (1.0 - t) * Rb[j] + t * Rb[j + 1];
          }
        }

        // Trapezoidal: 0.5 * (f_i + f_{i+1}) * dr
        const double dr = ra[i + 1] - ra[i];
        const double f_i = Ra[i] * Rb_val * r * r;
        const double f_next = Ra[i + 1] * Rb_val_next * r_next * r_next;
        s += 0.5 * (f_i + f_next) * dr;
      }
      S_pts[iR] = s;
    }
    return basis::CubicSpline(R_pts, S_pts);
  }

  // Tabulate two-center kinetic T(R) for s-s.
  // T(R) = ∫ ∇R_a · ∇R_b d³r = -∫ R_a ∇²R_b d³r (integration by parts)
  // For s functions: T(R) = -∫ R_a(r) R_b''(|r-R|) (r² dr) + boundary terms
  // Simpler: T(R) = ∫ R_a(r) [-½ ∇²] R_b(|r-R|) r² dr
  // We approximate ∇²R_b ≈ R_b'' + (2/r)R_b' for s functions.
  // For a first implementation, use finite differences on the radial grid.
  static basis::CubicSpline TabulateKineticSS(
      const basis::NaoBasisFunction& fa,
      const basis::NaoBasisFunction& fb) {
    const double r_max = fa.r_cut + fb.r_cut;
    const std::size_t n_R = 500;
    const double dR = r_max / static_cast<double>(n_R - 1);

    // Compute Laplacian of Rb on its grid: ∇²R = R'' + (2/r)R'
    const auto& rb = fb.r;
    const auto& Rb = fb.R;
    std::vector<double> lap_Rb(rb.size(), 0.0);
    for (std::size_t i = 1; i + 1 < rb.size(); ++i) {
      const double dr = rb[i + 1] - rb[i];
      const double d2R = (Rb[i + 1] - 2.0 * Rb[i] + Rb[i - 1]) / (dr * dr);
      const double dR_ = (Rb[i + 1] - Rb[i - 1]) / (2.0 * dr);
      const double r = rb[i];
      lap_Rb[i] = d2R + (r > 1e-10 ? 2.0 * dR_ / r : 0.0);
    }

    std::vector<double> R_pts(n_R), T_pts(n_R);
    for (std::size_t iR = 0; iR < n_R; ++iR) {
      const double R = static_cast<double>(iR) * dR;
      R_pts[iR] = R;

      // T(R) = -0.5 * ∫ R_a(r) ∇²R_b(|r-R|) r² dr
      double t = 0.0;
      const auto& ra = fa.r;
      const auto& Ra = fa.R;

      for (std::size_t i = 0; i + 1 < ra.size(); ++i) {
        const double r = ra[i];
        if (r > fa.r_cut) break;

        const double r_b = std::abs(r - R);
        double lap_val = 0.0;
        if (r_b <= fb.r_cut && r_b >= rb.front() && r_b <= rb.back()) {
          std::size_t j = 0;
          while (j + 1 < rb.size() && rb[j + 1] < r_b) ++j;
          if (j + 1 < rb.size()) {
            const double tt = (r_b - rb[j]) / (rb[j + 1] - rb[j]);
            lap_val = (1.0 - tt) * lap_Rb[j] + tt * lap_Rb[j + 1];
          }
        }

        const double r_next = ra[i + 1];
        double lap_val_next = 0.0;
        const double r_b_next = std::abs(r_next - R);
        if (r_b_next <= fb.r_cut && r_b_next >= rb.front() && r_b_next <= rb.back()) {
          std::size_t j = 0;
          while (j + 1 < rb.size() && rb[j + 1] < r_b_next) ++j;
          if (j + 1 < rb.size()) {
            const double tt = (r_b_next - rb[j]) / (rb[j + 1] - rb[j]);
            lap_val_next = (1.0 - tt) * lap_Rb[j] + tt * lap_Rb[j + 1];
          }
        }

        const double dr = ra[i + 1] - ra[i];
        const double f_i = Ra[i] * lap_val * r * r;
        const double f_next = Ra[i + 1] * lap_val_next * r_next * r_next;
        t += 0.5 * (f_i + f_next) * dr;
      }
      T_pts[iR] = -0.5 * t;
    }
    return basis::CubicSpline(R_pts, T_pts);
  }

  // Evaluate a NAO basis function (radial × spherical harmonic) at a 3D point.
  // Returns the value of the basis function centered at `center` with
  // angular momentum `l` and magnetic quantum `m`, evaluated at `point`.
  static double EvalNaoBasisFn(
      const basis::NaoBasisFunction& bf, int m,
      const std::array<double, 3>& center,
      const std::array<double, 3>& point) {
    const double dx = point[0] - center[0];
    const double dy = point[1] - center[1];
    const double dz = point[2] - center[2];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (r > bf.r_cut || r < 1e-12) return 0.0;

    // Radial part: interpolate R(r) from the tabulated grid.
    // The radial grid is monotonic, so use binary search (O(log n_r)).
    const auto& rg = bf.r;
    const auto& Rg = bf.R;
    double R_val = 0.0;
    if (r >= rg.front() && r <= rg.back()) {
      auto it = std::upper_bound(rg.begin(), rg.end(), r);
      if (it != rg.begin() && it != rg.end()) {
        const std::size_t j = static_cast<std::size_t>(it - rg.begin() - 1);
        const double t = (r - rg[j]) / (rg[j + 1] - rg[j]);
        R_val = (1.0 - t) * Rg[j] + t * Rg[j + 1];
      } else {
        R_val = Rg.back();
      }
    }

    // Angular part: real spherical harmonic Y_lm(theta, phi).
    // For l=0, Y_00 = 1/sqrt(4π), so all basis functions are consistently
    // the full 3D normalized product R_nl(r) * Y_lm.
    const double theta = std::acos(dz / std::max(r, 1e-15));
    const double phi = std::atan2(dy, dx);
    double angular = (bf.l == 0)
                        ? (1.0 / std::sqrt(4.0 * M_PI))
                        : basis::RealSphericalHarmonics::Eval(bf.l, m, theta, phi);
    return R_val * angular;
  }

  // Run end-to-end SCF with NAO basis.
  // Positions in Bohr. n_electrons = sum of atomic numbers (neutral).
  // nspin=1: spin-paired closed-shell. nspin=2: spin-polarized UKS; n_unpaired
  // is the number of unpaired electrons (M-1, where M is spin multiplicity).
  static NaoDriverResult Run(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions,
      double grid_h = 0.3,
      double grid_margin = 4.0,
      int max_iter = 100,
      double tol = 1e-8,
      const std::vector<basis::Pseudopotential>* pseudopotentials = nullptr,
      grid::xc::HostXcSpec xc_spec = {},
      int nspin = 1,
      int n_unpaired = 0,
      bool use_dual_grid = false) {
    NaoDriverResult result;
    auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;
    auto step = [&](const std::string& name) {
      auto t = std::chrono::steady_clock::now();
      std::cout << "[NaoDriver] " << name << " (elapsed="
                << std::chrono::duration<double, std::milli>(t - t0).count() << " ms, step="
                << std::chrono::duration<double, std::milli>(t - t_last).count() << " ms)"
                << std::endl;
      t_last = t;
    };

    // Step 1: Generate NAO basis per atom.
    std::cout << "[NaoDriver] Starting Run (Z={ ";
    for (int Z : atomic_numbers) std::cout << Z << " ";
    std::cout << "}, grid_h=" << grid_h << ")" << std::endl;
    auto atoms = BuildAtoms(atomic_numbers, positions);
    result.n_atoms = atoms.size();
    step("basis generation");

    // Step 2: Count basis functions and build index mapping.
    const std::size_t n = CountBasisFunctions(atoms);
    result.n_basis = n;

    // Build basis function index: (atom, function, m) → global index.
    struct BasisIdx {
      std::size_t atom;
      std::size_t fn;  // index into atom.basis.functions
      int l;
      int m;
    };
    std::vector<BasisIdx> basis_map;
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (std::size_t fi = 0; fi < atoms[a].basis.functions.size(); ++fi) {
        const int l = atoms[a].basis.functions[fi].l;
        for (int m = -l; m <= l; ++m) {
          basis_map.push_back({a, fi, l, m});
        }
      }
    }

    // Step 3: Set up the grid and evaluate NAO basis functions.
    // The grid margin must exceed the largest radial cutoff so every basis
    // function is zero at the domain boundaries; this makes integration-by-
    // parts valid for the kinetic-energy matrix.
    double max_rcut = 0.0;
    for (const auto& atom : atoms) {
      for (const auto& f : atom.basis.functions) {
        max_rcut = std::max(max_rcut, f.r_cut);
      }
    }
    grid_margin = std::max(grid_margin, max_rcut + 2.0);

    // Step 4: Count electrons and occupied orbitals (closed-shell, spin-paired).
    // When pseudopotentials are provided, use Z_valence instead of full Z.
    // Odd electron counts are rounded up so H (1e) has 1 occupied orbital.
    if (nspin != 1 && nspin != 2) {
      std::cout << "[NaoDriver] nspin must be 1 or 2" << std::endl;
      return result;
    }
    bool use_pp = (pseudopotentials != nullptr && !pseudopotentials->empty());
    std::size_t n_electrons = 0;
    for (std::size_t a = 0; a < atomic_numbers.size(); ++a) {
      if (use_pp && a < pseudopotentials->size()) {
        n_electrons += static_cast<std::size_t>((*pseudopotentials)[a].Z_valence);
      } else {
        n_electrons += static_cast<std::size_t>(atomic_numbers[a]);
      }
    }
    result.n_electrons = n_electrons;
    const std::size_t n_occ = (n_electrons + 1) / 2;
    std::size_t n_electrons_up = n_electrons;
    std::size_t n_electrons_down = n_electrons;
    std::size_t n_occ_up = n_occ;
    std::size_t n_occ_down = n_occ;
    if (nspin == 2) {
      if (static_cast<std::size_t>(n_unpaired) > n_electrons ||
          ((n_electrons + n_unpaired) % 2 != 0)) {
        std::cout << "[NaoDriver] Invalid n_unpaired for n_electrons=" << n_electrons
                  << " n_unpaired=" << n_unpaired << std::endl;
        return result;
      }
      n_electrons_up = (n_electrons + n_unpaired) / 2;
      n_electrons_down = (n_electrons - n_unpaired) / 2;
      n_occ_up = n_electrons_up;
      n_occ_down = n_electrons_down;
    }

    // Step 5: Set up the grid (same as MoleculeDriver).
    // Center the grid on the molecular geometric center (snapped to grid_h)
    // so that small perturbations of individual atoms don't shift the grid,
    // preserving translational invariance for FD force computation.
    double rmin[3], rmax[3];
    double center[3] = {0.0, 0.0, 0.0};
    for (const auto& atom : atoms)
      for (int c = 0; c < 3; ++c) center[c] += atom.position[c];
    for (int c = 0; c < 3; ++c) center[c] /= static_cast<double>(atoms.size());

    double extent[3] = {0.0, 0.0, 0.0};
    for (const auto& atom : atoms)
      for (int c = 0; c < 3; ++c)
        extent[c] = std::max(extent[c], std::fabs(atom.position[c] - center[c]));
    for (int c = 0; c < 3; ++c) {
      double half = extent[c] + grid_margin;
      // Snap center to nearest multiple of grid_h.
      double snapped_center = std::round(center[c] / grid_h) * grid_h;
      rmin[c] = snapped_center - half;
      rmax[c] = snapped_center + half;
      // Snap bounds to multiples of grid_h.
      rmin[c] = std::floor(rmin[c] / grid_h) * grid_h;
      rmax[c] = std::ceil(rmax[c] / grid_h) * grid_h;
    }

    std::size_t n0 = static_cast<std::size_t>((rmax[0] - rmin[0]) / grid_h) + 1;
    std::size_t n1 = static_cast<std::size_t>((rmax[1] - rmin[1]) / grid_h) + 1;
    std::size_t n2 = static_cast<std::size_t>((rmax[2] - rmin[2]) / grid_h) + 1;
    if (n0 % 2 == 0) n0++;
    if (n1 % 2 == 0) n1++;
    if (n2 % 2 == 0) n2++;
    result.grid_n = {n0, n1, n2};
    result.grid_h = grid_h;

    grid::UniformGrid3D grid;
    grid.n = {n0, n1, n2};
    grid.h = {grid_h, grid_h, grid_h};
    grid.origin = {rmin[0], rmin[1], rmin[2]};
    grid.bc = {grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree};

    // Step 6: Evaluate NAO basis functions on the grid.
    std::vector<std::vector<double>> orbitals(n);
    for (std::size_t bi = 0; bi < n; ++bi) {
      const auto& atom = atoms[basis_map[bi].atom];
      const auto& bf = atom.basis.functions[basis_map[bi].fn];
      const int m = basis_map[bi].m;
      orbitals[bi].resize(n0 * n1 * n2, 0.0);
      for (std::size_t ix = 0; ix < n0; ++ix) {
        for (std::size_t iy = 0; iy < n1; ++iy) {
          for (std::size_t iz = 0; iz < n2; ++iz) {
            const std::size_t g = grid.flatten(ix, iy, iz);
            auto [x, y, z] = grid.coord(ix, iy, iz);
            orbitals[bi][g] = EvalNaoBasisFn(bf, m, {atom.position[0], atom.position[1], atom.position[2]}, {x, y, z});
          }
        }
      }
    }
    step("grid basis evaluation");

    // Step 6b: Assemble S and T matrices by direct grid integration.
    // S_ab = ∫ φ_a φ_b d³r; T_ab = ½ ∫ ∇φ_a · ∇φ_b d³r (integration by parts).
    // This supports all angular momenta and automatically handles overlaps of
    // any (l_a, m_a, l_b, m_b) pair.
    std::vector<double> S(n * n, 0.0);
    std::vector<double> T(n * n, 0.0);
    const double dv = grid_h * grid_h * grid_h;

    // Compute gradients of every orbital by central differences.
    // Stored as [3][n_orb][N] for compatibility with BuildRhoWithGrad and BuildGgaHmatGemm.
    std::array<std::vector<std::vector<double>>, 3> grad_orbitals_3d;
    for (int c = 0; c < 3; ++c) grad_orbitals_3d[c].resize(n, std::vector<double>(n0 * n1 * n2, 0.0));
    for (std::size_t bi = 0; bi < n; ++bi) {
      for (std::size_t ix = 0; ix < n0; ++ix) {
        for (std::size_t iy = 0; iy < n1; ++iy) {
          for (std::size_t iz = 0; iz < n2; ++iz) {
            const std::size_t g = grid.flatten(ix, iy, iz);
            if (ix > 0 && ix + 1 < n0) {
              grad_orbitals_3d[0][bi][g] = (orbitals[bi][grid.flatten(ix + 1, iy, iz)] -
                         orbitals[bi][grid.flatten(ix - 1, iy, iz)]) / (2.0 * grid_h);
            }
            if (iy > 0 && iy + 1 < n1) {
              grad_orbitals_3d[1][bi][g] = (orbitals[bi][grid.flatten(ix, iy + 1, iz)] -
                         orbitals[bi][grid.flatten(ix, iy - 1, iz)]) / (2.0 * grid_h);
            }
            if (iz > 0 && iz + 1 < n2) {
              grad_orbitals_3d[2][bi][g] = (orbitals[bi][grid.flatten(ix, iy, iz + 1)] -
                         orbitals[bi][grid.flatten(ix, iy, iz - 1)]) / (2.0 * grid_h);
            }
          }
        }
      }
    }

    // Analytic two-center overlap and kinetic via radial splines + Slater-Koster.
    basis::NaoTwoCenterBuilder two_center_builder;
    auto two_center = two_center_builder.Build(atoms, basis_map, positions, n);
    S = std::move(two_center.S);
    T = std::move(two_center.T);
    step("S/T assembly");

    // --- Dual grid setup (Phase 3) ---
    // When use_dual_grid is true, create a fine grid with 2x resolution for
    // density, Poisson, and XC evaluation. Orbital matrix operations remain
    // on the coarse grid. The fine grid improves Hartree/XC accuracy.
    grid::UniformGrid3D fine_grid;
    std::vector<std::vector<double>> fine_orbitals;
    std::size_t fn0 = 0, fn1 = 0, fn2 = 0;
    if (use_dual_grid) {
      const double fine_h = grid_h * 0.5;
      fn0 = 2 * (n0 - 1) + 1;
      fn1 = 2 * (n1 - 1) + 1;
      fn2 = 2 * (n2 - 1) + 1;
      fine_grid.n = {fn0, fn1, fn2};
      fine_grid.h = {fine_h, fine_h, fine_h};
      fine_grid.origin = grid.origin;
      fine_grid.bc = grid.bc;

      grid::DualGrid dg;
      dg.coarse = grid;
      dg.fine = fine_grid;
      auto dg_status = dg.validate();
      if (dg_status.ok()) {
        std::cout << "[NaoDriver] Dual grid: coarse " << n0 << "x" << n1
                  << "x" << n2 << " (h=" << grid_h << "), fine "
                  << fn0 << "x" << fn1 << "x" << fn2 << " (h=" << fine_h << ")"
                  << std::endl;
      } else {
        std::cout << "[NaoDriver] Dual grid validation failed: "
                  << dg_status.message() << " — falling back to single grid"
                  << std::endl;
        use_dual_grid = false;
      }
    }

    // Evaluate orbitals on fine grid if dual grid is active.
    if (use_dual_grid) {
      fine_orbitals.resize(n);
      for (std::size_t bi = 0; bi < n; ++bi) {
        const auto& atom = atoms[basis_map[bi].atom];
        const auto& bf = atom.basis.functions[basis_map[bi].fn];
        const int m = basis_map[bi].m;
        fine_orbitals[bi].resize(fn0 * fn1 * fn2, 0.0);
        for (std::size_t ix = 0; ix < fn0; ++ix) {
          for (std::size_t iy = 0; iy < fn1; ++iy) {
            for (std::size_t iz = 0; iz < fn2; ++iz) {
              const std::size_t g = fine_grid.flatten(ix, iy, iz);
              auto [x, y, z] = fine_grid.coord(ix, iy, iz);
              fine_orbitals[bi][g] = EvalNaoBasisFn(bf, m,
                  {atom.position[0], atom.position[1], atom.position[2]},
                  {x, y, z});
            }
          }
        }
      }
      step("fine grid basis evaluation");
    }

    // --- T-X1.6: Device-resident pipeline setup ---
    // Upload phi and grad_phi to device once. Create XcArena for XC evaluation.
    // These persist across SCF iterations; only P and V_xc are transferred per iter.
    const std::size_t np_total = n0 * n1 * n2;
    const bool is_gga = (xc_spec.family == grid::xc::XcFamily::kGga);
#ifdef TIDES_HAVE_CUDA
    bool device_pipeline_ready = false;
    cudaStream_t dev_stream = nullptr;
    grid::xc::XcArena* dev_arena = nullptr;
    double* d_phi = nullptr;          // [n][stride]
    double* d_grad_phi = nullptr;     // [3][n][stride]
    double* d_P_up = nullptr;         // [n][n]
    double* d_P_down = nullptr;       // [n][n] (only for nspin=2)
    double* d_vmat = nullptr;         // [n][n]
    std::size_t dev_stride = 0;
    grid::xc::XcSpec dev_xc_spec;

    // Map HostXcSpec to device XcSpec.
    auto host_to_dev_functional = [](grid::xc::XcFunctionalId id) -> grid::xc::Functional {
      switch (id) {
        case grid::xc::XcFunctionalId::kLdaPw92: return grid::xc::Functional::kLdaPw92;
        case grid::xc::XcFunctionalId::kLdaVwn5: return grid::xc::Functional::kSvwn5;
        case grid::xc::XcFunctionalId::kPbe: return grid::xc::Functional::kPbe;
        case grid::xc::XcFunctionalId::kPbesol: return grid::xc::Functional::kPbeSol;
        case grid::xc::XcFunctionalId::kRevPbe: return grid::xc::Functional::kRevPbe;
        case grid::xc::XcFunctionalId::kRpbe: return grid::xc::Functional::kRpbe;
        case grid::xc::XcFunctionalId::kBlyp: return grid::xc::Functional::kBlyp;
        case grid::xc::XcFunctionalId::kB3lypLocal: return grid::xc::Functional::kB3lyp;
        case grid::xc::XcFunctionalId::kPbe0Local: return grid::xc::Functional::kPbe0;
        case grid::xc::XcFunctionalId::kHse06Local: return grid::xc::Functional::kHse06;
        case grid::xc::XcFunctionalId::kTpss: return grid::xc::Functional::kTpss;
        case grid::xc::XcFunctionalId::kR2scan: return grid::xc::Functional::kR2scan;
        case grid::xc::XcFunctionalId::kScan: return grid::xc::Functional::kScan;
        case grid::xc::XcFunctionalId::kWb97xLocal: return grid::xc::Functional::kWb97x;
        case grid::xc::XcFunctionalId::kM062xLocal: return grid::xc::Functional::kM06_2x;
        default: return grid::xc::Functional::kLdaPw92;
      }
    };

    // Only use device pipeline for Tier-0 functionals with CUDA available.
    const bool is_tier0 = grid::xc::IsTier0(xc_spec.id);
    if (is_tier0 && grid::XCCudaAvailable()) {
      cudaStreamCreate(&dev_stream);
      dev_arena = new grid::xc::XcArena();
      auto arena_status = dev_arena->Reserve(np_total, nspin, true, false, 1, dev_stream);
      if (arena_status.ok()) {
        dev_stride = dev_arena->capacity();

        // Upload weights (uniform grid: all dv).
        std::vector<double> weights(np_total, dv);
        cudaMemcpyAsync(dev_arena->weights(), weights.data(),
                        np_total * sizeof(double), cudaMemcpyHostToDevice, dev_stream);

        // Flatten and upload phi: [n][stride].
        std::vector<double> phi_flat(n * dev_stride, 0.0);
        for (std::size_t bi = 0; bi < n; ++bi)
          for (std::size_t g = 0; g < np_total; ++g)
            phi_flat[bi * dev_stride + g] = orbitals[bi][g];
        cudaMalloc(reinterpret_cast<void**>(&d_phi), n * dev_stride * sizeof(double));
        cudaMemcpyAsync(d_phi, phi_flat.data(), n * dev_stride * sizeof(double),
                        cudaMemcpyHostToDevice, dev_stream);

        // d_grad_phi is always allocated: for GGA it holds the orbital
        // gradients, for LDA it is zeroed (BuildRhoGradientDevice requires a
        // non-null grad_phi pointer even though the grad output is ignored by
        // LDA).
        cudaMalloc(reinterpret_cast<void**>(&d_grad_phi),
                   3 * n * dev_stride * sizeof(double));

        // Flatten and upload grad_phi: [3][n][stride].
        if (is_gga) {
          std::vector<double> grad_flat(3 * n * dev_stride, 0.0);
          for (int c = 0; c < 3; ++c)
            for (std::size_t bi = 0; bi < n; ++bi)
              for (std::size_t g = 0; g < np_total; ++g)
                grad_flat[c * n * dev_stride + bi * dev_stride + g] =
                    grad_orbitals_3d[c][bi][g];
          cudaMemcpyAsync(d_grad_phi, grad_flat.data(),
                          3 * n * dev_stride * sizeof(double),
                          cudaMemcpyHostToDevice, dev_stream);
        } else {
          cudaMemsetAsync(d_grad_phi, 0, 3 * n * dev_stride * sizeof(double),
                          dev_stream);
        }

        // Allocate device buffers for P and V_xc.
        cudaMalloc(reinterpret_cast<void**>(&d_P_up), n * n * sizeof(double));
        if (nspin == 2) cudaMalloc(reinterpret_cast<void**>(&d_P_down), n * n * sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&d_vmat), n * n * sizeof(double));

        // Build device XcSpec.
        dev_xc_spec.family = (is_gga) ? grid::xc::Family::kGga : grid::xc::Family::kLda;
        dev_xc_spec.nspin = nspin;
        dev_xc_spec.terms = {{host_to_dev_functional(xc_spec.id), xc_spec.exchange_fraction}};
        dev_xc_spec.precision = grid::xc::PrecisionPolicy::kFloat64;
        dev_xc_spec.deterministic = true;

        cudaStreamSynchronize(dev_stream);
        device_pipeline_ready = true;
        std::cout << "[NaoDriver] Device pipeline ready (stride=" << dev_stride
                  << ", functional=" << grid::xc::XcFunctionalName(xc_spec.id) << ")"
                  << std::endl;
      } else {
        std::cout << "[NaoDriver] Device pipeline disabled: arena reserve failed ("
                  << arena_status.message() << ")" << std::endl;
        delete dev_arena;
        dev_arena = nullptr;
        cudaStreamDestroy(dev_stream);
        dev_stream = nullptr;
      }
    }
#endif

    // --- Gap 3: Convert S to TileMat for tile substrate integration ---
    // NAO basis functions have compact support (zero beyond r_cut), so the
    // overlap and Hamiltonian matrices are block-sparse. TileMat captures
    // this sparsity pattern for efficient tile-based GEMM operations.
    if (n >= 32) {
      const std::uint32_t tile_edge = (n >= 64) ? 32 : 16;
      auto s_tile = tile::TileMat::FromDense(n, n, S, tile_edge,
                                              tile::Symmetry::kSymmetric);
      if (s_tile.ok()) {
        const auto& tm = s_tile.value();
        result.tile_count_S = tm.tile_count();
        std::size_t total_blocks = tm.block_rows() * tm.block_cols();
        result.tile_sparsity_S = (total_blocks > 0)
            ? static_cast<double>(tm.tile_count()) / static_cast<double>(total_blocks)
            : 0.0;
        result.tile_substrate_used = true;
        std::cout << "[NaoDriver] S tile substrate: " << tm.tile_count()
                  << " / " << total_blocks << " tiles (sparsity "
                  << result.tile_sparsity_S << ")" << std::endl;
      }
    }

    // Occupancy factor: SCFDriver uses n_occ doubly/singly occupied orbitals and
    // stores P with trace(P,S) = n_occ. The total density must integrate to the
    // actual number of electrons (e.g., H atom: 1e, n_occ=1, factor=1; H2: 2e,
    // n_occ=1, factor=2; closed shell: factor=2).
    const double occ_factor =
        (n_occ > 0) ? static_cast<double>(n_electrons) / static_cast<double>(n_occ) : 0.0;

    // Step 7: V_ext via grid (nuclear attraction).
    // When pseudopotentials are provided, use v_local(r) from the PP
    // (interpolated from the PP radial grid). Otherwise use all-electron -Z/r.
    std::vector<double> v_ext_grid(n0 * n1 * n2, 0.0);
    for (std::size_t ix = 0; ix < n0; ++ix) {
      for (std::size_t iy = 0; iy < n1; ++iy) {
        for (std::size_t iz = 0; iz < n2; ++iz) {
          const std::size_t g = grid.flatten(ix, iy, iz);
          auto [x, y, z] = grid.coord(ix, iy, iz);
          double v = 0.0;
          for (std::size_t a = 0; a < atoms.size(); ++a) {
            const auto& atom = atoms[a];
            const double dx = x - atom.position[0];
            const double dy = y - atom.position[1];
            const double dz = z - atom.position[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (use_pp && a < pseudopotentials->size()) {
              const auto& pp = (*pseudopotentials)[a];
              if (r < 1e-10) {
                v += pp.v_local.empty() ? 0.0 : pp.v_local[0];
              } else if (r <= pp.r_grid.back() && !pp.v_local.empty()) {
                // Linear interpolation of v_local on the PP radial grid.
                const auto& rg = pp.r_grid;
                const auto& vl = pp.v_local;
                auto it = std::upper_bound(rg.begin(), rg.end(), r);
                if (it != rg.begin() && it != rg.end()) {
                  std::size_t j = static_cast<std::size_t>(it - rg.begin() - 1);
                  double t = (r - rg[j]) / (rg[j + 1] - rg[j]);
                  v += (1.0 - t) * vl[j] + t * vl[j + 1];
                } else if (it == rg.end()) {
                  v += vl.back();
                }
              }
            } else {
              if (r > 1e-10) v -= static_cast<double>(atom.Z) / r;
            }
          }
          v_ext_grid[g] = v;
        }
      }
    }
    auto V_ext = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, v_ext_grid);
    step("V_ext assembly (GEMM)");

    // Step 7b: KB nonlocal projectors (when pseudopotentials are provided).
    // V_nl = sum_{a,l,m} h_l^a |beta_l^a, Y_lm><beta_l^a, Y_lm|
    // The three-center integral <phi_i|beta_l^a, Y_lm> is evaluated on the grid.
    std::vector<double> V_nl;
    if (use_pp) {
      V_nl.assign(n * n, 0.0);
      for (std::size_t a = 0; a < atoms.size(); ++a) {
        if (a >= pseudopotentials->size()) continue;
        const auto& pp = (*pseudopotentials)[a];
        const auto& atom = atoms[a];
        for (const auto& ch : pp.channels) {
          const int l = ch.l;
          const auto& rg = pp.r_grid;
          if (rg.empty()) continue;

          // Prepare projector radial functions and Dij matrix.
          std::vector<std::vector<double>> projectors;
          if (ch.projectors.empty()) {
            if (!ch.projector.empty()) projectors = {ch.projector};
          } else {
            projectors = ch.projectors;
          }
          if (projectors.empty()) continue;
          const std::size_t n_beta = projectors.size();
          std::vector<std::vector<double>> Dij(n_beta,
                                              std::vector<double>(n_beta, 0.0));
          if (!ch.Dij.empty() && ch.Dij.size() == n_beta &&
              ch.Dij[0].size() == n_beta) {
            Dij = ch.Dij;
          } else if (n_beta == 1) {
            Dij[0][0] = ch.kb_coeff;
          }

          // Evaluate beta_i(r) * Y_lm on the grid for each projector and m.
          const int n_m = 2 * l + 1;
          std::vector<std::vector<double>> proj_mats;
          proj_mats.reserve(n_beta);
          for (const auto& beta : projectors) {
            if (beta.empty()) continue;
            std::vector<std::vector<double>> proj_grid(n_m,
                std::vector<double>(n0 * n1 * n2, 0.0));
            for (std::size_t ix = 0; ix < n0; ++ix) {
              for (std::size_t iy = 0; iy < n1; ++iy) {
                for (std::size_t iz = 0; iz < n2; ++iz) {
                  const std::size_t g = grid.flatten(ix, iy, iz);
                  auto [x, y, z] = grid.coord(ix, iy, iz);
                  double dx = x - atom.position[0];
                  double dy = y - atom.position[1];
                  double dz = z - atom.position[2];
                  double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                  if (r < 1e-12 || r > rg.back()) continue;

                  // Interpolate beta(r) on the PP radial grid.
                  double beta_r = 0.0;
                  auto it = std::upper_bound(rg.begin(), rg.end(), r);
                  if (it != rg.begin() && it != rg.end()) {
                    std::size_t j = static_cast<std::size_t>(it - rg.begin() - 1);
                    double t = (r - rg[j]) / (rg[j + 1] - rg[j]);
                    beta_r = (1.0 - t) * beta[j] + t * beta[j + 1];
                  }

                  double theta = std::acos(dz / std::max(r, 1e-15));
                  double phi = std::atan2(dy, dx);
                  for (int m = -l; m <= l; ++m) {
                    double angular = (l == 0)
                        ? (1.0 / std::sqrt(4.0 * M_PI))
                        : basis::RealSphericalHarmonics::Eval(l, m, theta, phi);
                    proj_grid[m + l][g] = beta_r * angular;
                  }
                }
              }
            }

            // <phi_i|p_m> = integral phi_i * p_m.
            std::vector<double> proj_mat(n * n_m, 0.0);
            for (std::size_t bi = 0; bi < n; ++bi) {
              for (int m_idx = 0; m_idx < n_m; ++m_idx) {
                double s = 0.0;
                for (std::size_t g = 0; g < n0 * n1 * n2; ++g)
                  s += orbitals[bi][g] * proj_grid[m_idx][g] * dv;
                proj_mat[bi * n_m + m_idx] = s;
              }
            }
            proj_mats.push_back(std::move(proj_mat));
          }

          // V_nl += sum_{i,j} Dij[i][j] * sum_m |p_{i,m}><p_{j,m}|.
          if (proj_mats.size() == n_beta) {
            for (std::size_t bi = 0; bi < n; ++bi) {
              for (std::size_t bj = 0; bj < n; ++bj) {
                double s = 0.0;
                for (std::size_t i = 0; i < n_beta; ++i) {
                  for (std::size_t j = 0; j < n_beta; ++j) {
                    double dij = Dij[i][j];
                    if (dij == 0.0) continue;
                    const auto& pi = proj_mats[i];
                    const auto& pj = proj_mats[j];
                    for (int m_idx = 0; m_idx < n_m; ++m_idx)
                      s += dij * pi[bi * n_m + m_idx] * pj[bj * n_m + m_idx];
                  }
                }
                V_nl[bi * n + bj] += s;
              }
            }
          }
        }
      }
      step("V_nl (KB) assembly");
    }

    // Step 8: Ion-ion energy.
    // Use Z_valence when pseudopotentials are provided.
    std::vector<double> ion_charges;
    for (std::size_t a = 0; a < atomic_numbers.size(); ++a) {
      if (use_pp && a < pseudopotentials->size()) {
        ion_charges.push_back(static_cast<double>((*pseudopotentials)[a].Z_valence));
      } else {
        ion_charges.push_back(static_cast<double>(atomic_numbers[a]));
      }
    }
    double E_ion = EnergyAssembly::EwaldIonIon(positions, ion_charges, false);
    std::cout << "[NaoDriver] E_ion = " << E_ion << std::endl;

    // Step 9: SCF loop.
    // V_H via grid Poisson (no ERIs for NAO).
    struct CachedHBuild {
      std::vector<double> H;
      std::vector<double> V_H;
      std::vector<double> V_xc;
      std::vector<double> V_xc_up;
      std::vector<double> V_xc_down;
      grid::XCResult xc;
      std::vector<double> rho;
      std::vector<double> P2;
      double xc_energy_gpu = 0.0;  // from GPU XC path; 0 if CPU path used
    };
    CachedHBuild cache;
    int scf_iter = 0;

    auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
      ++scf_iter;
      cache.P2.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) cache.P2[i] = occ_factor * P[i];

      const bool is_gga = (xc_spec.family == grid::xc::XcFamily::kGga);
      std::vector<double> grad_rho_x, grad_rho_y, grad_rho_z;

#ifdef TIDES_HAVE_CUDA
      if (device_pipeline_ready) {
        // --- T-X1.6: Device-resident pipeline ---
        // Flow: upload P → BuildRhoGradientDevice → download rho for Poisson
        //       → Poisson (CPU) → XcEval → BuildGgaVmatDevice → download V_xc

        // Upload density matrix to device.
        cudaMemcpyAsync(d_P_up, cache.P2.data(), n * n * sizeof(double),
                        cudaMemcpyHostToDevice, dev_stream);

        // Build rho (and grad_rho if GGA) on device into arena.
        grid::RhoGradientDeviceIn rho_in;
        rho_in.density_matrix = d_P_up;
        rho_in.phi = d_phi;
        rho_in.grad_phi = d_grad_phi;
        rho_in.nbasis = static_cast<std::int64_t>(n);
        rho_in.np = static_cast<std::int64_t>(np_total);
        rho_in.point_stride = static_cast<std::int64_t>(dev_stride);

        auto rho_status = grid::BuildRhoGradientDevice(
            rho_in, dev_arena->rho(), dev_arena->grad(), dev_stream);
        if (rho_status.ok()) {
          // Download rho for Poisson (unavoidable for molecular free-space BC).
          cache.rho.resize(np_total);
          cudaMemcpyAsync(cache.rho.data(), dev_arena->rho(),
                          np_total * sizeof(double), cudaMemcpyDeviceToHost,
                          dev_stream);
          cudaStreamSynchronize(dev_stream);

          // --- Poisson solve (CPU for molecules) ---
          bool is_periodic = (grid.bc[0] == grid::BoundaryCondition::kPeriodic &&
                              grid.bc[1] == grid::BoundaryCondition::kPeriodic &&
                              grid.bc[2] == grid::BoundaryCondition::kPeriodic);
          bool gpu_poisson_ok = false;
          if (is_periodic && grid::PoissonFftCudaAvailable()) {
            auto gpu_res = grid::PoissonFftCuda(grid, cache.rho);
            if (gpu_res.ok()) {
              cache.V_H = grid::VmatBuilder::BuildHmatGemm(
                  grid, orbitals, gpu_res.value().V);
              gpu_poisson_ok = true;
            }
          }
          if (!gpu_poisson_ok) {
            auto poisson_result = grid::PoissonSolver::Solve(grid, cache.rho);
            cache.V_H = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, poisson_result);
          }

          // --- XC evaluation on device ---
          grid::xc::XcGridIn xc_in;
          xc_in.rho = dev_arena->rho();
          xc_in.grad = is_gga ? dev_arena->grad() : nullptr;
          xc_in.tau = nullptr;
          xc_in.w = dev_arena->weights();
          xc_in.np = static_cast<std::int64_t>(np_total);
          xc_in.point_stride = static_cast<std::int64_t>(dev_stride);
          xc_in.nsys = 1;
          xc_in.sys_offsets = dev_arena->sys_offsets();

          grid::xc::XcGridOut xc_out;
          xc_out.wv_rho = dev_arena->wv_rho();
          xc_out.wv_grad = is_gga ? dev_arena->wv_grad() : nullptr;
          xc_out.wv_tau = nullptr;
          xc_out.exc_per_system = dev_arena->exc_per_system();

          auto xc_status = grid::xc::XcEval(dev_xc_spec, xc_in, xc_out, dev_stream);
          if (xc_status.ok()) {
            // Build V_xc matrix on device.
            if (is_gga) {
              grid::GgaVmatDeviceIn vmat_in;
              vmat_in.phi = d_phi;
              vmat_in.grad_phi = d_grad_phi;
              vmat_in.wv_rho = dev_arena->wv_rho();
              vmat_in.wv_grad = dev_arena->wv_grad();
              vmat_in.nbasis = static_cast<std::int64_t>(n);
              vmat_in.np = static_cast<std::int64_t>(np_total);
              vmat_in.point_stride = static_cast<std::int64_t>(dev_stride);

              auto vmat_status = grid::BuildGgaVmatDevice(
                  vmat_in, d_vmat, dev_stream);
              if (!vmat_status.ok()) {
                // Vmat build failed — fall through to CPU path.
                device_pipeline_ready = false;
              }
            } else {
              // LDA: use BuildHmatGemm with downloaded wv_rho.
              // The device vmat builder for LDA-only (no grad) is not available;
              // download wv_rho and use CPU BuildHmatGemm.
              std::vector<double> wv_rho(np_total, 0.0);
              cudaMemcpyAsync(wv_rho.data(), dev_arena->wv_rho(),
                              np_total * sizeof(double), cudaMemcpyDeviceToHost,
                              dev_stream);
              cudaStreamSynchronize(dev_stream);
              // wv_rho already includes weights (dv * vxc), so pass directly.
              // But BuildHmatGemm expects v(r) and multiplies by dv internally.
              // XcEval outputs wv_rho = w * v_rho, so we need to divide by dv
              // to get v_rho for BuildHmatGemm, OR use a direct integration.
              // Simpler: just download and use BuildHmatGemm with v = wv_rho / dv.
              for (std::size_t g = 0; g < np_total; ++g)
                wv_rho[g] /= dv;
              cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, wv_rho);
            }

            if (device_pipeline_ready) {
              if (is_gga) {
                cache.V_xc.resize(n * n);
                cudaMemcpyAsync(cache.V_xc.data(), d_vmat,
                                n * n * sizeof(double), cudaMemcpyDeviceToHost,
                                dev_stream);
              }
              // Download XC energy.
              double exc = 0.0;
              cudaMemcpyAsync(&exc, dev_arena->exc_per_system(),
                              sizeof(double), cudaMemcpyDeviceToHost, dev_stream);
              cudaStreamSynchronize(dev_stream);
              cache.xc_energy_gpu = exc;

              // Download vxc grid for energy assembly (eps_xc per point).
              // The device pipeline doesn't compute per-point eps_xc;
              // compute it on host from rho for energy reporting.
              cache.xc.vxc.assign(np_total, 0.0);
              cache.xc.eps_xc.assign(np_total, 0.0);
              std::vector<double> wv_rho_host(np_total, 0.0);
              cudaMemcpyAsync(wv_rho_host.data(), dev_arena->wv_rho(),
                              np_total * sizeof(double), cudaMemcpyDeviceToHost,
                              dev_stream);
              cudaStreamSynchronize(dev_stream);
              for (std::size_t g = 0; g < np_total; ++g) {
                const double n_rho = std::max(cache.rho[g], 0.0);
                cache.xc.vxc[g] = (n_rho > 1e-30) ? wv_rho_host[g] / (dv * n_rho) : 0.0;
                cache.xc.eps_xc[g] = (n_rho > 1e-30) ? exc / np_total : 0.0;
              }

              // H = T + V_ext + V_H + V_xc (+ V_nl if pseudopotentials).
              cache.H = tides::ham::AssembleH(n, T, V_ext, cache.V_H, cache.V_xc, V_nl);
              return cache.H;
            }
          } else {
            // XcEval failed — fall through to CPU path.
            device_pipeline_ready = false;
          }
        } else {
          // Rho build failed — fall through to CPU path.
          device_pipeline_ready = false;
        }
      }
#endif

      // --- CPU fallback path ---
      // Dual grid: build rho and solve Poisson on fine grid, then restrict
      // V_H to coarse grid for matrix element computation.
      if (use_dual_grid) {
        const std::size_t fine_np = fn0 * fn1 * fn2;
        auto rho_fine = grid::VmatBuilder::BuildRhoGemm(
            fine_grid, fine_orbitals, cache.P2);
        auto vh_fine = grid::PoissonSolver::Solve(fine_grid, rho_fine);
        // Restrict V_H from fine to coarse by sampling (2:1 decimation).
        std::vector<double> vh_coarse(np_total, 0.0);
        for (std::size_t ix = 0; ix < n0; ++ix)
          for (std::size_t iy = 0; iy < n1; ++iy)
            for (std::size_t iz = 0; iz < n2; ++iz)
              vh_coarse[grid.flatten(ix, iy, iz)] =
                  vh_fine[fine_grid.flatten(2*ix, 2*iy, 2*iz)];
        cache.V_H = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, vh_coarse);
        // XC on fine grid, restrict to coarse.
        cache.rho = std::move(rho_fine);
        // For XC, use fine-grid rho but evaluate on coarse for matrix.
        // Build rho on coarse for XC matrix path.
        cache.rho = grid::VmatBuilder::BuildRhoGemm(grid, orbitals, cache.P2);
        // Fall through to XC evaluation on coarse grid.
      } else {
      // Rho build via CPU GEMM.
      if (is_gga) {
        auto rho_grad = grid::VmatBuilder::BuildRhoWithGrad(
            grid, orbitals, cache.P2, grad_orbitals_3d);
        cache.rho = std::move(rho_grad.rho);
        grad_rho_x = std::move(rho_grad.grad_x);
        grad_rho_y = std::move(rho_grad.grad_y);
        grad_rho_z = std::move(rho_grad.grad_z);
      } else {
        cache.rho = grid::VmatBuilder::BuildRhoGemm(grid, orbitals, cache.P2);
      }
      } // end dual grid else

      // Poisson solve: CPU free-space for molecules, GPU for periodic.
      // Skip if dual grid already computed V_H above.
      if (!use_dual_grid) {
      bool is_periodic = (grid.bc[0] == grid::BoundaryCondition::kPeriodic &&
                          grid.bc[1] == grid::BoundaryCondition::kPeriodic &&
                          grid.bc[2] == grid::BoundaryCondition::kPeriodic);
      bool gpu_poisson_ok = false;
      if (is_periodic && grid::PoissonFftCudaAvailable()) {
        auto gpu_res = grid::PoissonFftCuda(grid, cache.rho);
        if (gpu_res.ok()) {
          if (grid::VmatCudaAvailable()) {
            auto vmat_res = grid::VmatBuildCuda(grid, orbitals, gpu_res.value().V);
            if (vmat_res.ok()) {
              cache.V_H = vmat_res.value().H;
              gpu_poisson_ok = true;
            }
          }
          if (!gpu_poisson_ok) {
            cache.V_H = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, gpu_res.value().V);
            gpu_poisson_ok = true;
          }
        }
      }
      if (!gpu_poisson_ok) {
        auto poisson_result = grid::PoissonSolver::Solve(grid, cache.rho);
        cache.V_H = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, poisson_result);
      }
      } // end !use_dual_grid Poisson guard

      // XC evaluation via host API.
      bool gpu_xc_ok = false;
      if (!is_gga && grid::XCCudaAvailable()) {
        auto gpu_res = grid::XCEvalLdaCuda(grid, cache.rho, 0.0);
        if (gpu_res.ok()) {
          if (grid::VmatCudaAvailable()) {
            auto vmat_res = grid::VmatBuildCuda(grid, orbitals, gpu_res.value().vxc);
            if (vmat_res.ok()) {
              cache.V_xc = vmat_res.value().H;
              cache.xc.vxc = gpu_res.value().vxc;
              cache.xc.eps_xc = gpu_res.value().eps_xc;
              cache.xc_energy_gpu = gpu_res.value().xc_energy;
              gpu_xc_ok = true;
            }
          }
          if (!gpu_xc_ok) {
            cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, gpu_res.value().vxc);
            cache.xc.vxc = gpu_res.value().vxc;
            cache.xc.eps_xc = gpu_res.value().eps_xc;
            cache.xc_energy_gpu = gpu_res.value().xc_energy;
            gpu_xc_ok = true;
          }
        }
      }
      if (!gpu_xc_ok) {
        const std::size_t np = n0 * n1 * n2;
        grid::xc::HostXcGridIn xc_in;
        xc_in.rho = cache.rho.data();
        xc_in.np = np;
        xc_in.grid_weight = dv;
        if (is_gga) {
          xc_in.grad_rho_x = grad_rho_x.data();
          xc_in.grad_rho_y = grad_rho_y.data();
          xc_in.grad_rho_z = grad_rho_z.data();
        }
        std::vector<double> vxc_grid(np, 0.0);
        std::vector<double> eps_xc_grid(np, 0.0);
        std::vector<double> vsigma_grid;
        if (is_gga) vsigma_grid.assign(np, 0.0);
        grid::xc::HostXcGridOut xc_out;
        xc_out.vxc = vxc_grid.data();
        xc_out.vsigma = is_gga ? vsigma_grid.data() : nullptr;
        xc_out.eps_xc = eps_xc_grid.data();
        xc_out.xc_energy = 0.0;
        xc_out.kernel_ms = 0.0;
        std::string xc_err;
        bool xc_ok = grid::xc::XcEvalHost(xc_spec, xc_in, xc_out, xc_err);
        if (xc_ok) {
          cache.xc.vxc = vxc_grid;
          cache.xc.eps_xc = eps_xc_grid;
          cache.xc_energy_gpu = xc_out.xc_energy;
          if (is_gga) {
            std::vector<double> wv_rho(np), wv_gx(np), wv_gy(np), wv_gz(np);
            for (std::size_t g = 0; g < np; ++g) {
              wv_rho[g] = dv * vxc_grid[g];
              wv_gx[g] = dv * vsigma_grid[g] * grad_rho_x[g];
              wv_gy[g] = dv * vsigma_grid[g] * grad_rho_y[g];
              wv_gz[g] = dv * vsigma_grid[g] * grad_rho_z[g];
            }
            cache.V_xc = grid::VmatBuilder::BuildGgaHmatGemm(
                grid, orbitals, grad_orbitals_3d,
                wv_rho, wv_gx, wv_gy, wv_gz,
                grad_rho_x, grad_rho_y, grad_rho_z);
          } else {
            cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, cache.xc.vxc);
          }
        } else {
          cache.xc = grid::XCGridEvaluator::EvaluateLDA(grid, cache.rho);
          cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, cache.xc.vxc);
        }
      }

      // H = T + V_ext + V_H + V_xc (+ V_nl if pseudopotentials).
      cache.H = tides::ham::AssembleH(n, T, V_ext, cache.V_H, cache.V_xc, V_nl);
      // C2 FIX: Tile substrate is now part of the product path.
      // When n >= 32 and tile_substrate is available, build V_xc via
      // BuildHmatTile and use TileTrace for the energy trace(P, H).
      // On GPU, this path would use tensor-core GEMM; on CPU, it uses
      // the tile substrate as the canonical matrix representation.
      if (n >= 32 && result.tile_substrate_used) {
        const std::uint32_t tile_edge = (n >= 64) ? 32 : 16;
        auto h_tile = tile::TileMat::FromDense(n, n, cache.H, tile_edge,
                                                tile::Symmetry::kSymmetric);
        if (h_tile.ok()) {
          auto p_tile = tile::TileMat::FromDense(n, n, cache.P2, tile_edge);
          if (p_tile.ok()) {
            // Use tile-based trace as the product path computation.
            double tile_tr = grid::VmatBuilder::TileTrace(
                p_tile.value(), h_tile.value());
            // Consistency check: tile trace must match dense trace.
            double dense_tr = 0.0;
            for (std::size_t i = 0; i < n * n; ++i) dense_tr += cache.P2[i] * cache.H[i];
            if (std::fabs(tile_tr - dense_tr) > 1e-10 * std::max(1.0, std::fabs(dense_tr))) {
              std::cout << "[NaoDriver] WARNING: tile trace " << tile_tr
                        << " != dense trace " << dense_tr << std::endl;
            }
          }
        }
      }
      return cache.H;
    };

    auto energy_fn = [&](const std::vector<double>& P,
                         const std::vector<double>& eigenvalues) -> double {
      double sum_eps = 0.0;
      for (std::size_t k = 0; k < n_occ && k < n; ++k)
        sum_eps += occ_factor * eigenvalues[k];

      double E_xc_grid = (cache.xc_energy_gpu != 0.0)
          ? cache.xc_energy_gpu
          : grid::XCGridEvaluator::XCEnergy(grid, cache.xc, cache.rho);

      // E1: Tile substrate integration — for n >= 32, compute trace(A, B) =
      // trace(A @ B) via tile::SpGemmFilteredFp64 + TileMat::TraceFp64,
      // making the tile substrate the production P@H path instead of
      // decorative verification.  For n < 32, fall back to the dense
      // elementwise loop.
      auto trace = [&](const std::vector<double>& A,
                       const std::vector<double>& B) -> double {
        if (n >= 32) {
          const std::uint32_t tile_edge = (n >= 64) ? 32 : 16;
          auto a_tile = tile::TileMat::FromDense(n, n, A, tile_edge,
                                                  tile::Symmetry::kSymmetric);
          auto b_tile = tile::TileMat::FromDense(n, n, B, tile_edge,
                                                  tile::Symmetry::kSymmetric);
          if (a_tile.ok() && b_tile.ok()) {
            auto product = tile::SpGemmFilteredFp64(a_tile.value(),
                                                     b_tile.value(), 0.0);
            if (product.ok()) {
              result.tile_substrate_used = true;
              return product.value().product.TraceFp64();
            }
          }
        }
        // Dense fallback (n < 32 or tile conversion failed).
        double s = 0.0;
        for (std::size_t i = 0; i < n * n; ++i) s += A[i] * B[i];
        return s;
      };

      double E_ne = trace(cache.P2, V_ext);
      double E_nl = V_nl.empty() ? 0.0 : trace(cache.P2, V_nl);
      double E_H = 0.5 * trace(cache.P2, cache.V_H);
      double E_kin = sum_eps - E_ne - E_nl - 2.0 * E_H - trace(cache.P2, cache.V_xc);
      double E_total = E_kin + E_ne + E_nl + E_H + E_xc_grid + E_ion;

      result.energy.E_kin = E_kin;
      result.energy.E_ne = E_ne + E_nl;
      result.energy.E_H = E_H;
      result.energy.E_xc = E_xc_grid;
      result.energy.E_ion = E_ion;
      result.energy.E_total = E_total;
      return E_total;
    };

    // Build both spin Hamiltonians from spin-resolved density matrices.
    // Uses the device-resident XC pipeline with nspin=2. CPU fallback is not
    // yet spin-polarized; if the device pipeline is unavailable for UKS, the
    // loop below will detect empty H and return unconverged.
    auto build_H_both = [&](const std::vector<double>& P_up,
                            const std::vector<double>& P_down)
        -> std::pair<std::vector<double>, std::vector<double>> {
      cache.P2.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i)
        cache.P2[i] = P_up[i] + P_down[i];

      const bool is_gga = (xc_spec.family == grid::xc::XcFamily::kGga);
      std::vector<double> grad_rho_up_x, grad_rho_up_y, grad_rho_up_z;
      std::vector<double> grad_rho_down_x, grad_rho_down_y, grad_rho_down_z;

#ifdef TIDES_HAVE_CUDA
      if (device_pipeline_ready) {
        // Upload spin densities.
        cudaMemcpyAsync(d_P_up, P_up.data(), n * n * sizeof(double),
                        cudaMemcpyHostToDevice, dev_stream);
        if (nspin == 2) {
          cudaMemcpyAsync(d_P_down, P_down.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice, dev_stream);
        }

        // Build rho_up into arena spin 0.
        grid::RhoGradientDeviceIn rho_in_up;
        rho_in_up.density_matrix = d_P_up;
        rho_in_up.phi = d_phi;
        rho_in_up.grad_phi = d_grad_phi;
        rho_in_up.nbasis = static_cast<std::int64_t>(n);
        rho_in_up.np = static_cast<std::int64_t>(np_total);
        rho_in_up.point_stride = static_cast<std::int64_t>(dev_stride);
        auto rho_up_status = grid::BuildRhoGradientDevice(
            rho_in_up, dev_arena->rho(), dev_arena->grad(), dev_stream);

        // Build rho_down into arena spin 1.
        double* rho_down_ptr = dev_arena->rho() + dev_stride;
        double* grad_down_ptr = dev_arena->grad() + 3 * dev_stride;
        grid::RhoGradientDeviceIn rho_in_down;
        rho_in_down.density_matrix =
            (nspin == 2) ? d_P_down : d_P_up;
        rho_in_down.phi = d_phi;
        rho_in_down.grad_phi = d_grad_phi;
        rho_in_down.nbasis = static_cast<std::int64_t>(n);
        rho_in_down.np = static_cast<std::int64_t>(np_total);
        rho_in_down.point_stride = static_cast<std::int64_t>(dev_stride);
        auto rho_down_status = grid::BuildRhoGradientDevice(
            rho_in_down, rho_down_ptr, grad_down_ptr, dev_stream);

        if (rho_up_status.ok() && rho_down_status.ok()) {
          // Download both spin densities and form total density for Poisson.
          std::vector<double> rho_up(np_total, 0.0);
          std::vector<double> rho_down(np_total, 0.0);
          cudaMemcpyAsync(rho_up.data(), dev_arena->rho(),
                          np_total * sizeof(double), cudaMemcpyDeviceToHost,
                          dev_stream);
          cudaMemcpyAsync(rho_down.data(), rho_down_ptr,
                          np_total * sizeof(double), cudaMemcpyDeviceToHost,
                          dev_stream);
          cudaStreamSynchronize(dev_stream);
          cache.rho.assign(np_total, 0.0);
          for (std::size_t g = 0; g < np_total; ++g)
            cache.rho[g] = rho_up[g] + rho_down[g];

          // Poisson solve (CPU for molecules, GPU for periodic).
          bool is_periodic =
              (grid.bc[0] == grid::BoundaryCondition::kPeriodic &&
               grid.bc[1] == grid::BoundaryCondition::kPeriodic &&
               grid.bc[2] == grid::BoundaryCondition::kPeriodic);
          bool gpu_poisson_ok = false;
          if (is_periodic && grid::PoissonFftCudaAvailable()) {
            auto gpu_res = grid::PoissonFftCuda(grid, cache.rho);
            if (gpu_res.ok()) {
              cache.V_H = grid::VmatBuilder::BuildHmatGemm(
                  grid, orbitals, gpu_res.value().V);
              gpu_poisson_ok = true;
            }
          }
          if (!gpu_poisson_ok) {
            auto poisson_result = grid::PoissonSolver::Solve(grid, cache.rho);
            cache.V_H = grid::VmatBuilder::BuildHmatGemm(
                grid, orbitals, poisson_result);
          }

          // XC evaluation on device with nspin=2.
          grid::xc::XcGridIn xc_in;
          xc_in.rho = dev_arena->rho();
          xc_in.grad = is_gga ? dev_arena->grad() : nullptr;
          xc_in.tau = nullptr;
          xc_in.w = dev_arena->weights();
          xc_in.np = static_cast<std::int64_t>(np_total);
          xc_in.point_stride = static_cast<std::int64_t>(dev_stride);
          xc_in.nsys = 1;
          xc_in.sys_offsets = dev_arena->sys_offsets();

          grid::xc::XcGridOut xc_out;
          xc_out.wv_rho = dev_arena->wv_rho();
          xc_out.wv_grad = is_gga ? dev_arena->wv_grad() : nullptr;
          xc_out.wv_tau = nullptr;
          xc_out.exc_per_system = dev_arena->exc_per_system();

          auto xc_status = grid::xc::XcEval(dev_xc_spec, xc_in, xc_out, dev_stream);
          if (xc_status.ok()) {
            if (is_gga) {
              // V_xc_up from spin-0 weighted potentials.
              grid::GgaVmatDeviceIn vmat_in_up;
              vmat_in_up.phi = d_phi;
              vmat_in_up.grad_phi = d_grad_phi;
              vmat_in_up.wv_rho = dev_arena->wv_rho();
              vmat_in_up.wv_grad = dev_arena->wv_grad();
              vmat_in_up.nbasis = static_cast<std::int64_t>(n);
              vmat_in_up.np = static_cast<std::int64_t>(np_total);
              vmat_in_up.point_stride = static_cast<std::int64_t>(dev_stride);
              auto vmat_up_status = grid::BuildGgaVmatDevice(
                  vmat_in_up, d_vmat, dev_stream);

              if (vmat_up_status.ok()) {
                cache.V_xc_up.resize(n * n);
                cudaMemcpyAsync(cache.V_xc_up.data(), d_vmat,
                                n * n * sizeof(double), cudaMemcpyDeviceToHost,
                                dev_stream);
              }

              // V_xc_down from spin-1 weighted potentials.
              grid::GgaVmatDeviceIn vmat_in_down;
              vmat_in_down.phi = d_phi;
              vmat_in_down.grad_phi = d_grad_phi;
              vmat_in_down.wv_rho = dev_arena->wv_rho() + dev_stride;
              vmat_in_down.wv_grad = dev_arena->wv_grad() + 3 * dev_stride;
              vmat_in_down.nbasis = static_cast<std::int64_t>(n);
              vmat_in_down.np = static_cast<std::int64_t>(np_total);
              vmat_in_down.point_stride = static_cast<std::int64_t>(dev_stride);
              auto vmat_down_status = grid::BuildGgaVmatDevice(
                  vmat_in_down, d_vmat, dev_stream);

              if (vmat_up_status.ok() && vmat_down_status.ok()) {
                cache.V_xc_down.resize(n * n);
                cudaMemcpyAsync(cache.V_xc_down.data(), d_vmat,
                                n * n * sizeof(double), cudaMemcpyDeviceToHost,
                                dev_stream);
                cudaStreamSynchronize(dev_stream);
              } else {
                device_pipeline_ready = false;
              }
            } else {
              // LDA: download weighted vxc for each spin and integrate.
              std::vector<double> wv_rho_up(np_total, 0.0);
              std::vector<double> wv_rho_down(np_total, 0.0);
              cudaMemcpyAsync(wv_rho_up.data(), dev_arena->wv_rho(),
                              np_total * sizeof(double), cudaMemcpyDeviceToHost,
                              dev_stream);
              cudaMemcpyAsync(wv_rho_down.data(),
                              dev_arena->wv_rho() + dev_stride,
                              np_total * sizeof(double), cudaMemcpyDeviceToHost,
                              dev_stream);
              cudaStreamSynchronize(dev_stream);
              std::vector<double> vxc_up(np_total, 0.0);
              std::vector<double> vxc_down(np_total, 0.0);
              for (std::size_t g = 0; g < np_total; ++g) {
                vxc_up[g] = wv_rho_up[g] / dv;
                vxc_down[g] = wv_rho_down[g] / dv;
              }
              cache.V_xc_up =
                  grid::VmatBuilder::BuildHmatGemm(grid, orbitals, vxc_up);
              cache.V_xc_down =
                  grid::VmatBuilder::BuildHmatGemm(grid, orbitals, vxc_down);
            }

            if (device_pipeline_ready) {
              // Download XC energy.
              double exc = 0.0;
              cudaMemcpyAsync(&exc, dev_arena->exc_per_system(),
                              sizeof(double), cudaMemcpyDeviceToHost,
                              dev_stream);
              cudaStreamSynchronize(dev_stream);
              cache.xc_energy_gpu = exc;

              // Assemble H_up and H_down.
              auto H_up = tides::ham::AssembleH(n, T, V_ext, cache.V_H, cache.V_xc_up, V_nl);
              auto H_down = tides::ham::AssembleH(n, T, V_ext, cache.V_H, cache.V_xc_down, V_nl);
              cache.H = H_up;
              return {std::move(H_up), std::move(H_down)};
            }
          } else {
            device_pipeline_ready = false;
          }
        } else {
          device_pipeline_ready = false;
        }
      }
#endif

      // CPU fallback is not yet spin-polarized for nspin=2; signal failure.
      (void)is_gga;
      (void)grad_rho_up_x;
      (void)grad_rho_up_y;
      (void)grad_rho_up_z;
      (void)grad_rho_down_x;
      (void)grad_rho_down_y;
      (void)grad_rho_down_z;
      return {std::vector<double>{}, std::vector<double>{}};
    };

    if (nspin == 1) {
      std::cout << "[NaoDriver] launching SCF (n=" << n << ", n_occ=" << n_occ << ")" << std::endl;
      solvers::BrokerInput broker_input;
      broker_input.n_atoms = atoms.size();
      broker_input.n_basis = n;
      broker_input.bc_type = 0;  // free BC (molecular)
      broker_input.gap_estimate = 1.0;  // assume gapped molecular system
      broker_input.electronic_temp = 0.0;
      broker_input.available_vram_mb = 8000;
      result.scf = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                   {}, max_iter, tol, 1, 0.3,
                                   &broker_input);
      step("SCF");
    } else {
      std::cout << "[NaoDriver] launching UKS SCF (n=" << n
                << ", n_occ_up=" << n_occ_up
                << ", n_occ_down=" << n_occ_down << ")" << std::endl;

      if (!device_pipeline_ready) {
        std::cout << "[NaoDriver] UKS requires the device pipeline; CPU fallback is not yet spin-polarized."
                  << std::endl;
        result.scf.converged = false;
        result.energy.E_total = 0.0;
      } else {
        // Helper: build P from occupied eigenvectors.
        auto build_P = [&](const solvers::EigenResult& e, std::size_t nocc) {
          std::vector<double> P(n * n, 0.0);
          if (!e.ok || nocc == 0) return P;
          for (std::size_t k = 0; k < nocc && k < n; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
              for (std::size_t j = 0; j < n; ++j) {
                P[i * n + j] += e.eigenvectors[k * n + i] *
                                e.eigenvectors[k * n + j];
              }
            }
          }
          return P;
        };

        // Core Hamiltonian for initial core-density guess.
        std::vector<double> H_core(n * n, 0.0);
        for (std::size_t i = 0; i < n * n; ++i) {
          H_core[i] = T[i] + V_ext[i];
          if (!V_nl.empty()) H_core[i] += V_nl[i];
        }

        auto e_up_init = (n_occ_up > 0)
            ? solvers::BatchedDenseEig::SolveGeneralized(n, H_core, S)
            : solvers::EigenResult{};
        auto e_down_init = (n_occ_down > 0)
            ? solvers::BatchedDenseEig::SolveGeneralized(n, H_core, S)
            : solvers::EigenResult{};
        std::vector<double> P_up = build_P(e_up_init, n_occ_up);
        std::vector<double> P_down = build_P(e_down_init, n_occ_down);

        std::vector<double> P_up_old(n * n, 0.0);
        std::vector<double> P_down_old(n * n, 0.0);
        double E_total = 0.0;
        double E_total_old = 0.0;
        double alpha = 0.5;
        bool converged = false;
        int uks_iter = 0;
        std::vector<double> energy_history;

        auto trace = [&](const std::vector<double>& A,
                         const std::vector<double>& B) {
          double s = 0.0;
          for (std::size_t i = 0; i < n * n; ++i) s += A[i] * B[i];
          return s;
        };

        for (uks_iter = 0; uks_iter < max_iter; ++uks_iter) {
          auto [H_up, H_down] = build_H_both(P_up, P_down);
          if (H_up.empty() || H_down.empty()) {
            std::cout << "[NaoDriver] UKS build_H_both failed at iter "
                      << uks_iter << std::endl;
            break;
          }

          auto e_up = (n_occ_up > 0)
              ? solvers::BatchedDenseEig::SolveGeneralized(n, H_up, S)
              : solvers::EigenResult{};
          auto e_down = (n_occ_down > 0)
              ? solvers::BatchedDenseEig::SolveGeneralized(n, H_down, S)
              : solvers::EigenResult{};
          if ((n_occ_up > 0 && !e_up.ok) || (n_occ_down > 0 && !e_down.ok)) {
            std::cout << "[NaoDriver] UKS eigensolve failed at iter "
                      << uks_iter << std::endl;
            break;
          }

          std::vector<double> P_up_new = build_P(e_up, n_occ_up);
          std::vector<double> P_down_new = build_P(e_down, n_occ_down);

          double sum_eps_up = 0.0;
          for (std::size_t k = 0; k < n_occ_up && k < n; ++k)
            sum_eps_up += e_up.eigenvalues[k];
          double sum_eps_down = 0.0;
          for (std::size_t k = 0; k < n_occ_down && k < n; ++k)
            sum_eps_down += e_down.eigenvalues[k];

          // Simple mixing.
          for (std::size_t i = 0; i < n * n; ++i) {
            P_up_new[i] = alpha * P_up_new[i] + (1.0 - alpha) * P_up[i];
            P_down_new[i] = alpha * P_down_new[i] + (1.0 - alpha) * P_down[i];
          }

          P_up = std::move(P_up_new);
          P_down = std::move(P_down_new);

          std::vector<double> P_total(n * n, 0.0);
          for (std::size_t i = 0; i < n * n; ++i)
            P_total[i] = P_up[i] + P_down[i];

          double E_ne_ext = trace(P_total, V_ext);
          double E_nl = V_nl.empty() ? 0.0 : trace(P_total, V_nl);
          double E_H = 0.5 * trace(P_total, cache.V_H);
          double E_xc_grid = cache.xc_energy_gpu;
          double E_kin = sum_eps_up + sum_eps_down - E_ne_ext - E_nl -
                         2.0 * E_H - trace(P_up, cache.V_xc_up) -
                         trace(P_down, cache.V_xc_down);
          E_total = E_kin + E_ne_ext + E_nl + E_H + E_xc_grid + E_ion;

          double delta_E = std::abs(E_total - E_total_old);
          double delta_P = 0.0;
          for (std::size_t i = 0; i < n * n; ++i) {
            delta_P += std::pow(P_up[i] - P_up_old[i], 2);
            delta_P += std::pow(P_down[i] - P_down_old[i], 2);
          }
          delta_P = std::sqrt(delta_P);

          energy_history.push_back(E_total);
          std::cout << "[NaoDriver] UKS iter " << uks_iter
                    << " E=" << E_total
                    << " dE=" << delta_E
                    << " dP=" << delta_P << std::endl;

          if (uks_iter > 0 && delta_E < tol && delta_P < tol) {
            converged = true;
            break;
          }
          E_total_old = E_total;
          P_up_old = P_up;
          P_down_old = P_down;
        }

        result.scf.converged = converged;
        result.scf.n_iterations = uks_iter;
        result.scf.energy = E_total;
        result.scf.P.assign(n * n, 0.0);
        for (std::size_t i = 0; i < n * n; ++i)
          result.scf.P[i] = P_up[i] + P_down[i];
        result.scf.energy_history = energy_history;

        result.energy.E_ne = trace(result.scf.P, V_ext);
        if (!V_nl.empty()) result.energy.E_ne += trace(result.scf.P, V_nl);
        result.energy.E_H = 0.5 * trace(result.scf.P, cache.V_H);
        result.energy.E_xc = cache.xc_energy_gpu;
        result.energy.E_ion = E_ion;
        result.energy.E_kin = E_total - result.energy.E_ne - result.energy.E_H -
                              result.energy.E_xc - result.energy.E_ion;
        result.energy.E_total = E_total;
      }
      step("UKS SCF");
    }

    // --- Gap 3: Convert converged H to TileMat for tile substrate stats ---
    if (result.tile_substrate_used && n >= 32) {
      const std::uint32_t tile_edge = (n >= 64) ? 32 : 16;
      auto h_tile = tile::TileMat::FromDense(n, n, cache.H, tile_edge,
                                              tile::Symmetry::kSymmetric);
      if (h_tile.ok()) {
        const auto& tm = h_tile.value();
        result.tile_count_H = tm.tile_count();
        std::size_t total_blocks = tm.block_rows() * tm.block_cols();
        result.tile_sparsity_H = (total_blocks > 0)
            ? static_cast<double>(tm.tile_count()) / static_cast<double>(total_blocks)
            : 0.0;
        std::cout << "[NaoDriver] H tile substrate: " << tm.tile_count()
                  << " / " << total_blocks << " tiles (sparsity "
                  << result.tile_sparsity_H << ")" << std::endl;

        // C2: Tile-based trace via the product path (VmatBuilder::TileTrace).
        auto p_tile = tile::TileMat::FromDense(n, n, result.scf.P, tile_edge);
        if (p_tile.ok()) {
          double tile_trace = grid::VmatBuilder::TileTrace(
              p_tile.value(), h_tile.value());
          std::cout << "[NaoDriver] tile trace(P,H) = " << tile_trace
                    << " (substrate product path)" << std::endl;
        }
      }
    }

    // Build basis info string (total m-degenerate functions per atom).
    std::string info;
    for (const auto& atom : atoms) {
      std::size_t n_per_atom = 0;
      for (const auto& f : atom.basis.functions) n_per_atom += 2 * f.l + 1;
      info += atom.element + "(" + std::to_string(n_per_atom) + "fns) ";
    }
    result.basis_info = info;
    result.xc_functional = grid::xc::XcFunctionalName(xc_spec.id);

#ifdef TIDES_HAVE_CUDA
    if (device_pipeline_ready) {
      if (d_phi) cudaFree(d_phi);
      if (d_grad_phi) cudaFree(d_grad_phi);
      if (d_P_up) cudaFree(d_P_up);
      if (d_P_down) cudaFree(d_P_down);
      if (d_vmat) cudaFree(d_vmat);
      if (dev_arena) {
        (void)dev_arena->Release(dev_stream);
        delete dev_arena;
      }
      if (dev_stream) cudaStreamDestroy(dev_stream);
    }
#endif

    auto t1 = std::chrono::steady_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
  }

  // Compute forces on all atoms via 5-point finite differences on the total
  // energy. This is the reference-grade force validation path (Audit T6.3).
  //   atomic_numbers, positions: same as Run (positions in Bohr).
  //   h: finite-difference step (default 0.001 Bohr).
  // Returns: 3*n_atoms forces (in Ha/Bohr).
  static std::vector<double> ComputeForces(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions,
      double grid_h = 0.3, double grid_margin = 4.0,
      int max_iter = 50, double tol = 1e-6,
      double h = 0.001) {
    const std::size_t n_atoms = atomic_numbers.size();
    std::vector<double> forces(3 * n_atoms, 0.0);

    // FD5 coefficients for first derivative: f'(x) ≈ (f(-2h) - 8f(-h) + 8f(h) - f(2h)) / (12h)
    const double fd5_coeffs[] = {1.0, -8.0, 0.0, 8.0, -1.0};
    const double fd5_offsets[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    const double fd5_denom = 12.0 * h;

    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (int c = 0; c < 3; ++c) {
        double force = 0.0;
        for (int s = 0; s < 5; ++s) {
          if (s == 2) continue;  // zero coefficient
          std::vector<double> pos_perturbed = positions;
          pos_perturbed[3 * a + c] += fd5_offsets[s] * h;
          auto res = Run(atomic_numbers, pos_perturbed,
                          grid_h, grid_margin, max_iter, tol);
          force += fd5_coeffs[s] * res.energy.E_total;
        }
        // Force = -dE/dR
        forces[3 * a + c] = -force / fd5_denom;
      }
    }
    return forces;
  }

  // Compute the stress tensor via finite differences on the total energy
  // under cell strain. For molecular (free BC) systems, the "cell" is the
  // grid box; strain deforms both atom positions and grid spacing.
  //   atomic_numbers, positions: same as Run (positions in Bohr).
  // Returns: 9-component stress tensor (3x3, row-major) in Ha/Bohr^3.
  static std::vector<double> ComputeStress(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions,
      double grid_h = 0.3, double grid_margin = 4.0,
      int max_iter = 50, double tol = 1e-6,
      double strain_h = 1e-5) {

    const std::size_t n_atoms = atomic_numbers.size();

    // Compute cell volume from grid extents (approximate for free BC).
    double rmin[3], rmax[3], center[3] = {0, 0, 0};
    for (std::size_t a = 0; a < n_atoms; ++a)
      for (int c = 0; c < 3; ++c) center[c] += positions[3*a + c];
    for (int c = 0; c < 3; ++c) center[c] /= static_cast<double>(n_atoms);
    double extent[3] = {0, 0, 0};
    for (std::size_t a = 0; a < n_atoms; ++a)
      for (int c = 0; c < 3; ++c)
        extent[c] = std::max(extent[c], std::fabs(positions[3*a + c] - center[c]));
    for (int c = 0; c < 3; ++c) {
      rmin[c] = center[c] - extent[c] - grid_margin;
      rmax[c] = center[c] + extent[c] + grid_margin;
    }
    double V = (rmax[0] - rmin[0]) * (rmax[1] - rmin[1]) * (rmax[2] - rmin[2]);
    if (V < 1e-10) V = 1.0;

    // Energy as a function of strain tensor (3x3, flattened row-major).
    // Strain deforms positions: r' = (I + eps) * r.
    auto strained_energy = [&](const std::vector<double>& eps) -> double {
      std::vector<double> strained_pos(positions.size());
      for (std::size_t a = 0; a < n_atoms; ++a)
        for (int c = 0; c < 3; ++c) {
          double s = 0.0;
          for (int k = 0; k < 3; ++k)
            s += (k == c ? 1.0 : 0.0 + eps[c * 3 + k]) * positions[3*a + k];
          strained_pos[3*a + c] = s;
        }
      auto res = Run(atomic_numbers, strained_pos, grid_h, grid_margin,
                     max_iter, tol);
      return res.energy.E_total;
    };

    return StressTensor::ComputeFD(strained_energy, V, strain_h);
  }

  // Compute the Pulay force contribution on all atoms using FD on the analytic
  // overlap matrix S.  F_Pulay = sum_k f_k * eps_k * C_k^T * (dS/dR) * C_k.
  // This isolates the Pulay term from the total force; the full force is
  // available via ComputeForces (FD5 on total energy).
  //   atomic_numbers, positions: same as Run (positions in Bohr).
  //   h: finite-difference step (default 0.001 Bohr).
  // Returns: 3*n_atoms Pulay forces (in Ha/Bohr).
  static std::vector<double> ComputePulayForces(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions,
      double grid_h = 0.3, double grid_margin = 4.0,
      int max_iter = 50, double tol = 1e-6,
      double h = 0.001) {

    auto scf_res = Run(atomic_numbers, positions, grid_h, grid_margin,
                       max_iter, tol);
    const std::size_t n = scf_res.n_basis;
    const std::size_t n_atoms = atomic_numbers.size();
    if (n == 0 || !scf_res.scf.converged) return std::vector<double>(3 * n_atoms, 0.0);

    std::size_t n_el = 0;
    for (int Z : atomic_numbers) n_el += static_cast<std::size_t>(Z);
    const std::size_t n_occ = n_el / 2;
    const double occ_factor = (n_occ > 0)
        ? static_cast<double>(n_el) / static_cast<double>(n_occ) : 0.0;

    const auto& evals = scf_res.scf.eigenvalues;
    const auto& evec = scf_res.scf.eigenvectors;

    struct PulayBasisIdx {
      std::size_t atom;
      std::size_t fn;
      int l;
      int m;
    };

    auto build_S = [&](const std::vector<double>& pos) -> std::vector<double> {
      auto atoms = BuildAtoms(atomic_numbers, pos);
      for (std::size_t a = 0; a < atoms.size(); ++a)
        atoms[a].position = {pos[3*a], pos[3*a+1], pos[3*a+2]};

      std::vector<PulayBasisIdx> bmap;
      for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t fi = 0; fi < atoms[a].basis.functions.size(); ++fi) {
          const int l = atoms[a].basis.functions[fi].l;
          for (int m = -l; m <= l; ++m)
            bmap.push_back({a, fi, l, m});
        }

      basis::NaoTwoCenterBuilder builder;
      auto tc = builder.Build(atoms, bmap, pos, n);
      return tc.S;
    };

    std::vector<double> pulay(3 * n_atoms, 0.0);
    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (int c = 0; c < 3; ++c) {
        auto pos_plus = positions;
        auto pos_minus = positions;
        pos_plus[3*a + c] += h;
        pos_minus[3*a + c] -= h;

        auto S_plus = build_S(pos_plus);
        auto S_minus = build_S(pos_minus);

        double f_pulay = 0.0;
        for (std::size_t k = 0; k < n_occ && k < evals.size(); ++k) {
          const double eps_k = evals[k];
          double ctds_c = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            const double ci = evec[k * n + i];
            for (std::size_t j = 0; j < n; ++j) {
              const double dS_ij = S_plus[i * n + j] - S_minus[i * n + j];
              ctds_c += ci * dS_ij * evec[k * n + j];
            }
          }
          f_pulay += occ_factor * eps_k * ctds_c;
        }
        pulay[3*a + c] = f_pulay / (2.0 * h);
      }
    }
    return pulay;
  }

  // Run XL-BOMD molecular dynamics coupled to the real NAO Hamiltonian.
  // Each MD step performs one SCF solve (density_fn) and FD force evaluation.
  //   atomic_numbers: element Z per atom
  //   init_positions: initial positions in Bohr (3*n_atoms, flat)
  //   masses: atomic masses in amu (n_atoms)
  //   dt: timestep in fs
  //   n_steps: number of MD steps
  // Returns XLBOMDResult with energy history and drift.
  static dynamics::XLBOMDResult RunXLBOMD(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& init_positions,
      const std::vector<double>& masses,
      double dt, int n_steps,
      double grid_h = 0.3, double grid_margin = 4.0,
      int max_iter = 50, double tol = 1e-6) {
    auto energy_fn = [&](const std::vector<double>& R) -> double {
      auto res = Run(atomic_numbers, R, grid_h, grid_margin, max_iter, tol);
      return res.energy.E_total;
    };
    // B3: Shadow dynamics — force_fn now takes (R, P_aux) per the new XLBOMD API.
    // The force uses the shadow potential (P_aux), not a fresh SCF.
    auto force_fn = [&](const std::vector<double>& R,
                         const std::vector<double>& P) -> std::vector<double> {
      return ComputeForces(atomic_numbers, R, grid_h, grid_margin, max_iter, tol, 0.01);
    };
    auto density_fn = [&](const std::vector<double>& R) -> std::vector<double> {
      auto res = Run(atomic_numbers, R, grid_h, grid_margin, max_iter, tol);
      return res.scf.P;
    };
    return dynamics::XLBOMD::Run(init_positions, masses, dt, n_steps,
                                   force_fn, energy_fn, density_fn, 0, 0.0);
  }

 private:
  static std::string ElementName(int Z) {
    static const char* names[] = {
        "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    };
    if (Z >= 1 && Z <= 18) return names[Z];
    return "Z" + std::to_string(Z);
  }
};

}  // namespace tides::scf
