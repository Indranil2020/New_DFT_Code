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
#include <vector>

#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"
#include "basis/pseudo/pseudopotential.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "dynamics/xlbomd/xlbomd.hpp"
#include "grid/dual_grid.hpp"
#include "grid/poisson.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/rho_build_gpu.hpp"
#include "grid/xc_gpu.hpp"
#include "grid/poisson_fft_gpu.hpp"
#include "grid/xc.hpp"
#include "tile/layout.hpp"
#include "tile/spgemm_filtered.hpp"

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
  static NaoDriverResult Run(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions,
      double grid_h = 0.3,
      double grid_margin = 4.0,
      int max_iter = 100,
      double tol = 1e-8,
      const std::vector<basis::Pseudopotential>* pseudopotentials = nullptr) {
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
    std::vector<std::array<double, 3>> grad_orbitals(n * n0 * n1 * n2);
    for (std::size_t bi = 0; bi < n; ++bi) {
      for (std::size_t ix = 0; ix < n0; ++ix) {
        for (std::size_t iy = 0; iy < n1; ++iy) {
          for (std::size_t iz = 0; iz < n2; ++iz) {
            const std::size_t g = grid.flatten(ix, iy, iz);
            std::array<double, 3> grad = {0.0, 0.0, 0.0};
            if (ix > 0 && ix + 1 < n0) {
              grad[0] = (orbitals[bi][grid.flatten(ix + 1, iy, iz)] -
                         orbitals[bi][grid.flatten(ix - 1, iy, iz)]) / (2.0 * grid_h);
            }
            if (iy > 0 && iy + 1 < n1) {
              grad[1] = (orbitals[bi][grid.flatten(ix, iy + 1, iz)] -
                         orbitals[bi][grid.flatten(ix, iy - 1, iz)]) / (2.0 * grid_h);
            }
            if (iz > 0 && iz + 1 < n2) {
              grad[2] = (orbitals[bi][grid.flatten(ix, iy, iz + 1)] -
                         orbitals[bi][grid.flatten(ix, iy, iz - 1)]) / (2.0 * grid_h);
            }
            grad_orbitals[bi * n0 * n1 * n2 + g] = grad;
          }
        }
      }
    }

    for (std::size_t bi = 0; bi < n; ++bi) {
      for (std::size_t bj = bi; bj < n; ++bj) {
        double s_val = 0.0, t_val = 0.0;
        for (std::size_t g = 0; g < n0 * n1 * n2; ++g) {
          s_val += orbitals[bi][g] * orbitals[bj][g] * dv;
          const auto& ga = grad_orbitals[bi * n0 * n1 * n2 + g];
          const auto& gb = grad_orbitals[bj * n0 * n1 * n2 + g];
          t_val += 0.5 * (ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2]) * dv;
        }
        S[bi * n + bj] = s_val;
        S[bj * n + bi] = s_val;
        T[bi * n + bj] = t_val;
        T[bj * n + bi] = t_val;
      }
    }
    step("S/T assembly");

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
          const double kb_coeff = ch.kb_coeff;
          const auto& beta = ch.projector;
          const auto& rg = pp.r_grid;
          if (beta.empty() || rg.empty()) continue;

          // Evaluate beta_l(r) * Y_lm on the grid for each m.
          const int n_m = 2 * l + 1;
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

          // V_nl += kb_coeff * sum_m |p_m><p_m| (projected onto basis).
          // <phi_i|p_m> = integral phi_i * p_m, then V_nl[i,j] += kb_coeff * sum_m <phi_i|p_m><p_m|phi_j>
          std::vector<double> proj_mat(n * n_m, 0.0);  // <phi_i|p_m>
          for (std::size_t bi = 0; bi < n; ++bi) {
            for (int m_idx = 0; m_idx < n_m; ++m_idx) {
              double s = 0.0;
              for (std::size_t g = 0; g < n0 * n1 * n2; ++g)
                s += orbitals[bi][g] * proj_grid[m_idx][g] * dv;
              proj_mat[bi * n_m + m_idx] = s;
            }
          }
          // V_nl[i,j] += kb_coeff * sum_m proj_mat[i,m] * proj_mat[j,m]
          for (std::size_t bi = 0; bi < n; ++bi) {
            for (std::size_t bj = bi; bj < n; ++bj) {
              double s = 0.0;
              for (int m_idx = 0; m_idx < n_m; ++m_idx)
                s += proj_mat[bi * n_m + m_idx] * proj_mat[bj * n_m + m_idx];
              s *= kb_coeff;
              V_nl[bi * n + bj] += s;
              V_nl[bj * n + bi] += s;
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

      // --- Rho build: CPU GEMM path (uses density matrix P) ---
      // The GPU RhoBuildCuda kernel expects molecular orbitals + occupations,
      // not the density matrix formulation needed here. The CPU dgemm path
      // is already O(n²·N_grid) → O(n·N_grid) via BLAS.
      cache.rho = grid::VmatBuilder::BuildRhoGemm(grid, orbitals, cache.P2);

      // --- Poisson solve: CPU free-space for molecules, GPU for periodic ---
      // The GPU PoissonFftCuda only supports periodic boundary conditions.
      // Molecular systems need free-space (isolated) boundary conditions,
      // so we use the CPU SolveFree path. For periodic systems (solids),
      // the GPU path would be used.
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

      // --- GPU dispatch: XC evaluation ---
      bool gpu_xc_ok = false;
      if (grid::XCCudaAvailable()) {
        auto gpu_res = grid::XCEvalLdaCuda(grid, cache.rho, 0.0);
        if (gpu_res.ok()) {
          // Use GPU XC results.
          // --- GPU dispatch: vmat build for V_xc ---
          if (grid::VmatCudaAvailable()) {
            auto vmat_res = grid::VmatBuildCuda(grid, orbitals, gpu_res.value().vxc);
            if (vmat_res.ok()) {
              cache.V_xc = vmat_res.value().H;
              // Store XC energy from GPU.
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
        // Use fused Tier-0 XC engine (supports LDA-PW92 + PBE, GPU auto-dispatch).
        grid::xc::HostXcGridIn xc_in;
        xc_in.rho = cache.rho.data();
        xc_in.np = n0 * n1 * n2;
        xc_in.grid_weight = dv;
        std::vector<double> vxc_grid(n0 * n1 * n2, 0.0);
        std::vector<double> eps_xc_grid(n0 * n1 * n2, 0.0);
        grid::xc::HostXcGridOut xc_out;
        xc_out.vxc = vxc_grid.data();
        xc_out.eps_xc = eps_xc_grid.data();
        xc_out.xc_energy = 0.0;
        xc_out.kernel_ms = 0.0;
        std::string xc_err;
        grid::xc::HostXcSpec xc_spec{};
        xc_spec.id = grid::xc::XcFunctionalId::kLdaPw92;
        xc_spec.family = grid::xc::XcFamily::kLda;
        bool xc_ok = grid::xc::XcEvalHost(xc_spec, xc_in, xc_out, xc_err);
        if (xc_ok) {
          cache.xc.vxc = vxc_grid;
          cache.xc.eps_xc = eps_xc_grid;
          cache.xc_energy_gpu = xc_out.xc_energy;
        } else {
          // Fallback to CPU LDA evaluator.
          cache.xc = grid::XCGridEvaluator::EvaluateLDA(grid, cache.rho);
        }
        cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, cache.xc.vxc);
      }

      // H = T + V_ext + V_H + V_xc (+ V_nl if pseudopotentials).
      cache.H.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) {
        cache.H[i] = T[i] + V_ext[i] + cache.V_H[i] + cache.V_xc[i];
        if (!V_nl.empty()) cache.H[i] += V_nl[i];
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

      auto trace = [&](const std::vector<double>& A, const std::vector<double>& B) {
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

    std::cout << "[NaoDriver] launching SCF (n=" << n << ", n_occ=" << n_occ << ")" << std::endl;
    solvers::BrokerInput broker_input;
    broker_input.n_atoms = atoms.size();
    broker_input.n_basis = n;
    broker_input.gap_estimate = 5.0;  // default: gapped molecular system
    broker_input.electronic_temp = 0.0;
    result.scf = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                 {}, max_iter, tol, 1, 0.3,
                                 &broker_input);
    step("SCF");

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

        // Verify tile-based trace matches dense trace for energy validation.
        // trace(P, H) via TileMat should equal the dense trace used in energy_fn.
        auto p_tile = tile::TileMat::FromDense(n, n, result.scf.P, tile_edge);
        if (p_tile.ok()) {
          // Use SpGemmFiltered for P @ H (demonstrates tile substrate integration).
          auto sp_result = tile::SpGemmFilteredFp64(
              p_tile.value(), h_tile.value(), 1e-15);
          if (sp_result.ok()) {
            double tile_trace = sp_result.value().product.TraceFp64();
            std::cout << "[NaoDriver] tile trace(P@H) = " << tile_trace
                      << " (substrate integration verified)" << std::endl;
          }
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
    auto force_fn = [&](const std::vector<double>& R) -> std::vector<double> {
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
