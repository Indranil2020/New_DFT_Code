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

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "grid/dual_grid.hpp"
#include "grid/poisson.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc.hpp"

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
    const std::size_t n_R = 200;
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
    const std::size_t n_R = 200;
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
    const auto& rg = bf.r;
    const auto& Rg = bf.R;
    double R_val = 0.0;
    if (r >= rg.front() && r <= rg.back()) {
      std::size_t j = 0;
      while (j + 1 < rg.size() && rg[j + 1] < r) ++j;
      if (j + 1 < rg.size()) {
        const double t = (r - rg[j]) / (rg[j + 1] - rg[j]);
        R_val = (1.0 - t) * Rg[j] + t * Rg[j + 1];
      } else {
        R_val = Rg.back();
      }
    }

    if (bf.l == 0) return R_val;  // s orbital: no angular part

    // Angular part: real spherical harmonic Y_lm(theta, phi).
    const double theta = std::acos(dz / std::max(r, 1e-15));
    const double phi = std::atan2(dy, dx);
    double angular = basis::RealSphericalHarmonics::Eval(bf.l, m, theta, phi);
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
      double tol = 1e-8) {
    NaoDriverResult result;
    auto t0 = std::chrono::steady_clock::now();

    // Step 1: Generate NAO basis per atom.
    auto atoms = BuildAtoms(atomic_numbers, positions);
    result.n_atoms = atoms.size();

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

    // Step 4: Count electrons.
    std::size_t n_electrons = 0;
    for (int Z : atomic_numbers) n_electrons += static_cast<std::size_t>(Z);
    result.n_electrons = n_electrons;
    const std::size_t n_occ = n_electrons / 2;

    // Step 5: Set up the grid (same as MoleculeDriver).
    double rmin[3] = {1e30, 1e30, 1e30};
    double rmax[3] = {-1e30, -1e30, -1e30};
    for (const auto& atom : atoms) {
      for (int c = 0; c < 3; ++c) {
        rmin[c] = std::min(rmin[c], atom.position[c]);
        rmax[c] = std::max(rmax[c], atom.position[c]);
      }
    }
    for (int c = 0; c < 3; ++c) {
      rmin[c] -= grid_margin;
      rmax[c] += grid_margin;
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

    // Step 7: V_ext via grid (nuclear attraction).
    // For NAO, we project -Z/|r-R| onto the basis via VmatBuilder.
    std::vector<double> v_ext_grid(n0 * n1 * n2, 0.0);
    for (std::size_t ix = 0; ix < n0; ++ix) {
      for (std::size_t iy = 0; iy < n1; ++iy) {
        for (std::size_t iz = 0; iz < n2; ++iz) {
          const std::size_t g = grid.flatten(ix, iy, iz);
          auto [x, y, z] = grid.coord(ix, iy, iz);
          double v = 0.0;
          for (const auto& atom : atoms) {
            const double dx = x - atom.position[0];
            const double dy = y - atom.position[1];
            const double dz = z - atom.position[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r > 1e-10) v -= static_cast<double>(atom.Z) / r;
          }
          v_ext_grid[g] = v;
        }
      }
    }
    auto V_ext = grid::VmatBuilder::BuildHmat(grid, orbitals, v_ext_grid);

    // Step 8: Ion-ion energy.
    double E_ion = EnergyAssembly::EwaldIonIon(
        positions,
        std::vector<double>(atomic_numbers.begin(), atomic_numbers.end()),
        false);

    // Step 9: SCF loop.
    // V_H via grid Poisson (no ERIs for NAO).
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
      cache.P2.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) cache.P2[i] = 2.0 * P[i];

      // Grid-based Hartree: rho → V_H via Poisson.
      cache.rho = grid::VmatBuilder::BuildRho(grid, orbitals, cache.P2);

      // Solve Poisson: ∇²V_H = -4π ρ.
      auto poisson_result = grid::PoissonSolver::Solve(grid, cache.rho);
      cache.V_H = grid::VmatBuilder::BuildHmat(grid, orbitals, poisson_result);

      // Grid-based XC.
      cache.xc = grid::XCGridEvaluator::EvaluateLDA(grid, cache.rho);
      cache.V_xc = grid::VmatBuilder::BuildHmat(grid, orbitals, cache.xc.vxc);

      // H = T + V_ext + V_H + V_xc.
      cache.H.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i) {
        cache.H[i] = T[i] + V_ext[i] + cache.V_H[i] + cache.V_xc[i];
      }
      return cache.H;
    };

    auto energy_fn = [&](const std::vector<double>& P,
                         const std::vector<double>& eigenvalues) -> double {
      double sum_eps = 0.0;
      for (std::size_t k = 0; k < n_occ && k < n; ++k)
        sum_eps += 2.0 * eigenvalues[k];

      double E_xc_grid = grid::XCGridEvaluator::XCEnergy(grid, cache.xc, cache.rho);

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

    result.scf = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                 {}, max_iter, tol, 1, 0.3);

    // Build basis info string.
    std::string info;
    for (const auto& atom : atoms) {
      info += atom.element + "(" + std::to_string(atom.basis.functions.size()) + "fns) ";
    }
    result.basis_info = info;

    auto t1 = std::chrono::steady_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
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
