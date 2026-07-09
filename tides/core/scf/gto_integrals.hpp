#pragma once

// GTO (Gaussian-Type Orbital) integral library.
// Computes overlap S and kinetic T integrals for contracted Gaussian
// basis sets using the Obara-Saika recursion.
//
// Nuclear attraction V_ext is computed on the grid via the existing
// VmatBuilder infrastructure (more robust than analytic OS recursion
// for nuclear attraction).
//
// Supports s, p, and d Cartesian Gaussians.

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace tides::scf {

struct GTOPrimitive {
  double exponent;
  double coefficient;
};

struct GTOShell {
  int atom_index;
  int l;  // angular momentum (0=s, 1=p, 2=d)
  std::vector<GTOPrimitive> primitives;
};

struct GTOMolecule {
  std::vector<double> positions;  // 3*n_atoms (Bohr)
  std::vector<int> atomic_numbers;
  std::vector<GTOShell> shells;
  std::size_t n_basis = 0;
};

class GTOIntegrals {
 public:
  // Compute the overlap matrix S (n_basis x n_basis).
  static std::vector<double> Overlap(const GTOMolecule& mol) {
    const std::size_t n = mol.n_basis;
    std::vector<double> S(n * n, 0.0);
    std::size_t i = 0;
    for (const auto& shell_i : mol.shells) {
      for (int mi = 0; mi < NumCartesian(shell_i.l); ++mi, ++i) {
        std::size_t j = 0;
        for (const auto& shell_j : mol.shells) {
          for (int mj = 0; mj < NumCartesian(shell_j.l); ++mj, ++j) {
            if (j > i) continue;
            double val = OverlapPair(mol, shell_i, mi, shell_j, mj);
            S[i * n + j] = val;
            S[j * n + i] = val;
          }
        }
      }
    }
    return S;
  }

  // Compute the nuclear attraction matrix V_ext (n_basis x n_basis).
  // V_ext[i,j] = -sum_A Z_A * <phi_i | 1/|r - R_A| | phi_j>
  // Uses analytic Obara-Saika recursion with Boys functions.
  static std::vector<double> NuclearAttraction(const GTOMolecule& mol) {
    const std::size_t n = mol.n_basis;
    std::vector<double> V(n * n, 0.0);
    std::size_t i = 0;
    for (const auto& shell_i : mol.shells) {
      for (int mi = 0; mi < NumCartesian(shell_i.l); ++mi, ++i) {
        std::size_t j = 0;
        for (const auto& shell_j : mol.shells) {
          for (int mj = 0; mj < NumCartesian(shell_j.l); ++mj, ++j) {
            if (j > i) continue;
            double val = 0.0;
            for (std::size_t a = 0; a < mol.atomic_numbers.size(); ++a) {
              val -= mol.atomic_numbers[a] *
                     NuclearPair(mol, shell_i, mi, shell_j, mj, a);
            }
            V[i * n + j] = val;
            V[j * n + i] = val;
          }
        }
      }
    }
    return V;
  }

  // Compute the kinetic energy matrix T (n_basis x n_basis).
  static std::vector<double> Kinetic(const GTOMolecule& mol) {
    const std::size_t n = mol.n_basis;
    std::vector<double> T(n * n, 0.0);
    std::size_t i = 0;
    for (const auto& shell_i : mol.shells) {
      for (int mi = 0; mi < NumCartesian(shell_i.l); ++mi, ++i) {
        std::size_t j = 0;
        for (const auto& shell_j : mol.shells) {
          for (int mj = 0; mj < NumCartesian(shell_j.l); ++mj, ++j) {
            if (j > i) continue;
            double val = KineticPair(mol, shell_i, mi, shell_j, mj);
            T[i * n + j] = val;
            T[j * n + i] = val;
          }
        }
      }
    }
    return T;
  }

  // Evaluate a basis function on a 3D grid.
  static std::vector<double> EvalBasisOnGrid(
      const GTOMolecule& mol, std::size_t basis_idx,
      const std::vector<double>& grid_x,
      const std::vector<double>& grid_y,
      const std::vector<double>& grid_z,
      std::size_t n0, std::size_t n1, std::size_t n2) {
    const std::size_t N = n0 * n1 * n2;
    std::vector<double> vals(N, 0.0);

    auto [shell_idx, m_idx] = BasisIndexToShell(mol, basis_idx);
    const auto& shell = mol.shells[shell_idx];
    const double* Ra = &mol.positions[3 * shell.atom_index];

    for (std::size_t g = 0; g < N; ++g) {
      const std::size_t iz = g / (n0 * n1);
      const std::size_t rem = g % (n0 * n1);
      const std::size_t iy = rem / n0;
      const std::size_t ix = rem % n0;
      const double rx = grid_x[ix] - Ra[0];
      const double ry = grid_y[iy] - Ra[1];
      const double rz = grid_z[iz] - Ra[2];
      const double r2 = rx * rx + ry * ry + rz * rz;

      double angular = AngularFunction(shell.l, m_idx, rx, ry, rz);
      const auto [lx, ly, lz] = CartExp(shell.l, m_idx);
      double radial = 0.0;
      for (const auto& prim : shell.primitives)
        radial += prim.coefficient * PrimNorm(prim.exponent, lx, ly, lz) *
                   std::exp(-prim.exponent * r2);
      vals[g] = angular * radial;
    }
    return vals;
  }

  // Count Cartesian functions for angular momentum l.
  // s=1, p=3, d=6 (Cartesian, not spherical).
  static int NumCartesian(int l) {
    return (l + 1) * (l + 2) / 2;
  }

  // --- Electron Repulsion Integrals (ERI) ---
  // Computes (ij|kl) = integral integral phi_i(r) phi_j(r) 1/|r-r'|
  //                    phi_k(r') phi_l(r') dV dV'
  // using Obara-Saika recursion from the ss|ss base case.
  //
  // For s and p functions only (STO-3G for light elements).

  // Build the Coulomb (Hartree) matrix: V_H[i,j] = sum_{k,l} P[k,l] * (ij|kl)
  static std::vector<double> CoulombMatrix(
      const GTOMolecule& mol, const std::vector<double>& P) {
    const std::size_t n = mol.n_basis;
    std::vector<double> V_H(n * n, 0.0);
    std::size_t i = 0;
    for (const auto& si : mol.shells) {
      for (int mi = 0; mi < NumCartesian(si.l); ++mi, ++i) {
        std::size_t j = 0;
        for (const auto& sj : mol.shells) {
          for (int mj = 0; mj < NumCartesian(sj.l); ++mj, ++j) {
            std::size_t k = 0;
            for (const auto& sk : mol.shells) {
              for (int mk = 0; mk < NumCartesian(sk.l); ++mk, ++k) {
                std::size_t l = 0;
                for (const auto& sl : mol.shells) {
                  for (int ml = 0; ml < NumCartesian(sl.l); ++ml, ++l) {
                    double eri = ERI(mol, si, mi, sj, mj, sk, mk, sl, ml);
                    if (std::fabs(eri) < 1e-15) continue;
                    V_H[i * n + j] += P[k * n + l] * eri;
                  }
                }
              }
            }
          }
        }
      }
    }
    return V_H;
  }

  // Compute the XC energy density matrix: eps_xc_mat[i,j] = <phi_i|eps_xc(rho)|phi_j>
  // This requires the XC energy density on the grid, so we still use the grid for XC.
  // But V_H is now analytic.

 private:
  static std::pair<std::size_t, int> BasisIndexToShell(
      const GTOMolecule& mol, std::size_t idx) {
    std::size_t i = 0;
    for (std::size_t s = 0; s < mol.shells.size(); ++s) {
      for (int m = 0; m < NumCartesian(mol.shells[s].l); ++m, ++i) {
        if (i == idx) return {s, m};
      }
    }
    return {0, 0};
  }

  // Cartesian Gaussian angular function.
  // l=0: {1}
  // l=1: {x, y, z}
  // l=2: {xx, xy, xz, yy, yz, zz}  (6 Cartesian d functions)
  static double AngularFunction(int l, int m, double x, double y, double z) {
    if (l == 0) return 1.0;
    if (l == 1) {
      if (m == 0) return x;
      if (m == 1) return y;
      if (m == 2) return z;
    }
    if (l == 2) {
      if (m == 0) return x * x;
      if (m == 1) return x * y;
      if (m == 2) return x * z;
      if (m == 3) return y * y;
      if (m == 4) return y * z;
      if (m == 5) return z * z;
    }
    return 1.0;
  }

  // Cartesian exponents (lx, ly, lz) for shell l, magnetic index m.
  static std::array<int, 3> CartExp(int l, int m) {
    if (l == 0) return {0, 0, 0};
    if (l == 1) {
      if (m == 0) return {1, 0, 0};
      if (m == 1) return {0, 1, 0};
      if (m == 2) return {0, 0, 1};
    }
    if (l == 2) {
      if (m == 0) return {2, 0, 0};
      if (m == 1) return {1, 1, 0};
      if (m == 2) return {1, 0, 1};
      if (m == 3) return {0, 2, 0};
      if (m == 4) return {0, 1, 1};
      if (m == 5) return {0, 0, 2};
    }
    return {0, 0, 0};
  }

  // Double factorial: (-1)!! = 1, 0!! = 1, 1!! = 1, 2!! = 2, 3!! = 3, ...
  static double DoubleFact(int n) {
    if (n < 0) return 1.0;  // (-1)!! = 1
    double r = 1.0;
    for (int k = n; k > 0; k -= 2) r *= static_cast<double>(k);
    return r;
  }

  // Normalization factor for a Cartesian Gaussian primitive.
  // N = (2a/pi)^(3/4) * (4a)^((lx+ly+lz)/2) / sqrt(df(2lx-1)*df(2ly-1)*df(2lz-1))
  static double PrimNorm(double alpha, int lx, int ly, int lz) {
    const double base = std::pow(2.0 * alpha / M_PI, 0.75);
    const int l_sum = lx + ly + lz;
    const double angular = std::pow(4.0 * alpha, static_cast<double>(l_sum) / 2.0);
    const double df = std::sqrt(DoubleFact(2 * lx - 1) *
                                 DoubleFact(2 * ly - 1) *
                                 DoubleFact(2 * lz - 1));
    return base * angular / df;
  }

  // Boys function F_m(t) via upward recursion from F_0.
  // F_0(t) = 0.5 * sqrt(pi/t) * erf(sqrt(t))
  // F_{m+1}(t) = ((2m+1)*F_m(t) - exp(-t)) / (2t)
  static double BoysFm(int m, double t) {
    if (t < 1e-15) return 1.0 / (2.0 * m + 1.0);
    double f0 = 0.5 * std::sqrt(M_PI / t) * std::erf(std::sqrt(t));
    if (m == 0) return f0;
    double fm = f0;
    for (int k = 0; k < m; ++k)
      fm = ((2.0 * k + 1.0) * fm - std::exp(-t)) / (2.0 * t);
    return fm;
  }

  // 1D nuclear attraction OS recursion coefficients.
  // Returns vector of (coefficient, boys_index) pairs for the 1D integral.
  // For s/p functions (i,j <= 1):
  //   [0,0]: {(1, 0)}
  //   [1,0]: {(PA, 0), (PC, 1)}
  //   [0,1]: {(PB, 0), (PC, 1)}
  //   [1,1]: {(PA*PB + 1/(2p), 0), ((PA+PB)*PC, 1), (PC^2, 2)}
  // For d functions (i or j = 2), use the full recursion.
  struct CoeffPair { double coeff; int m_idx; };
  static std::vector<CoeffPair> Nuclear1DCoeffs(int i, int j, double PA,
                                                 double PB, double PC,
                                                 double p) {
    if (i == 0 && j == 0) return {{1.0, 0}};
    if (i == 1 && j == 0) return {{PA, 0}, {PC, 1}};
    if (i == 0 && j == 1) return {{PB, 0}, {PC, 1}};
    if (i == 1 && j == 1)
      return {{PA * PB + 1.0 / (2.0 * p), 0},
              {(PA + PB) * PC, 1},
              {PC * PC, 2}};
    // General recursion for higher l (d functions).
    // V_{i+1,j} = PA * V_{i,j} + 1/(2p) * (i * V_{i-1,j} + j * V_{i,j-1}) + PC * V_{i,j}^{m+1}
    // Build by reducing the larger index.
    std::vector<CoeffPair> result;
    if (i > 0) {
      auto lower = Nuclear1DCoeffs(i - 1, j, PA, PB, PC, p);
      auto lower2 = (i > 1) ? Nuclear1DCoeffs(i - 2, j, PA, PB, PC, p)
                            : std::vector<CoeffPair>{};
      auto shift = Nuclear1DCoeffs(i - 1, j, PA, PB, PC, p);
      for (auto& c : shift) c.m_idx += 1;
      for (const auto& c : lower) result.push_back({PA * c.coeff, c.m_idx});
      for (const auto& c : lower2) result.push_back({static_cast<double>(i - 1) / (2.0 * p) * c.coeff, c.m_idx});
      for (const auto& c : shift) result.push_back({PC * c.coeff, c.m_idx});
    } else if (j > 0) {
      auto lower = Nuclear1DCoeffs(i, j - 1, PA, PB, PC, p);
      auto lower2 = (j > 1) ? Nuclear1DCoeffs(i, j - 2, PA, PB, PC, p)
                            : std::vector<CoeffPair>{};
      auto shift = Nuclear1DCoeffs(i, j - 1, PA, PB, PC, p);
      for (auto& c : shift) c.m_idx += 1;
      for (const auto& c : lower) result.push_back({PB * c.coeff, c.m_idx});
      for (const auto& c : lower2) result.push_back({static_cast<double>(j - 1) / (2.0 * p) * c.coeff, c.m_idx});
      for (const auto& c : shift) result.push_back({PC * c.coeff, c.m_idx});
    }
    return result;
  }

  // Nuclear attraction for a pair of shells, for a given nucleus C.
  static double NuclearPair(
      const GTOMolecule& mol,
      const GTOShell& si, int mi,
      const GTOShell& sj, int mj,
      std::size_t atom_c) {
    const double* Ra = &mol.positions[3 * si.atom_index];
    const double* Rb = &mol.positions[3 * sj.atom_index];
    const double* Rc = &mol.positions[3 * atom_c];
    const auto [lx_i, ly_i, lz_i] = CartExp(si.l, mi);
    const auto [lx_j, ly_j, lz_j] = CartExp(sj.l, mj);

    double total = 0.0;
    for (const auto& pi : si.primitives) {
      for (const auto& pj : sj.primitives) {
        const double a = pi.exponent, b = pj.exponent;
        const double p = a + b;
        const double mu = a * b / p;
        const double AB2 = (Ra[0]-Rb[0])*(Ra[0]-Rb[0]) +
                           (Ra[1]-Rb[1])*(Ra[1]-Rb[1]) +
                           (Ra[2]-Rb[2])*(Ra[2]-Rb[2]);
        const double prefac = 2.0 * M_PI / p * std::exp(-mu * AB2);

        const double Px = (a * Ra[0] + b * Rb[0]) / p;
        const double Py = (a * Ra[1] + b * Rb[1]) / p;
        const double Pz = (a * Ra[2] + b * Rb[2]) / p;
        const double PC2 = (Px - Rc[0]) * (Px - Rc[0]) +
                           (Py - Rc[1]) * (Py - Rc[1]) +
                           (Pz - Rc[2]) * (Pz - Rc[2]);
        const double t = p * PC2;

        // 1D recursion coefficients for each direction.
        auto cx = Nuclear1DCoeffs(lx_i, lx_j, Px - Ra[0], Px - Rb[0],
                                  Px - Rc[0], p);
        auto cy = Nuclear1DCoeffs(ly_i, ly_j, Py - Ra[1], Py - Rb[1],
                                  Py - Rc[1], p);
        auto cz = Nuclear1DCoeffs(lz_i, lz_j, Pz - Ra[2], Pz - Rb[2],
                                  Pz - Rc[2], p);

        // Combine: sum over all (c_x, m_x) * (c_y, m_y) * (c_z, m_z) * F_{m_x+m_y+m_z}(t)
        double prim_val = 0.0;
        for (const auto& ex : cx)
          for (const auto& ey : cy)
            for (const auto& ez : cz) {
              const int m = ex.m_idx + ey.m_idx + ez.m_idx;
              prim_val += ex.coeff * ey.coeff * ez.coeff * BoysFm(m, t);
            }

        const double Na = PrimNorm(a, lx_i, ly_i, lz_i);
        const double Nb = PrimNorm(b, lx_j, ly_j, lz_j);
        total += pi.coefficient * pj.coefficient * Na * Nb * prefac * prim_val;
      }
    }
    return total;
  }

  // 1D overlap using Obara-Saika recursion.
  // S_{i,j} with base S_{0,0} = 1 (prefactor factored out).
  // S_{i+1,j} = PA * S_{i,j} + 1/(2p) * (i * S_{i-1,j} + j * S_{i,j-1})
  // S_{i,j+1} = PB * S_{i,j} + 1/(2p) * (i * S_{i-1,j} + j * S_{i,j-1})
  static double Overlap1D(int i, int j, double PA, double PB, double p) {
    if (i < 0 || j < 0) return 0.0;
    if (i == 0 && j == 0) return 1.0;
    const double inv_2p = 1.0 / (2.0 * p);
    if (j > 0) {
      return PB * Overlap1D(i, j - 1, PA, PB, p) +
             inv_2p * (static_cast<double>(i) * Overlap1D(i - 1, j - 1, PA, PB, p) +
                       static_cast<double>(j - 1) * Overlap1D(i, j - 2, PA, PB, p));
    }
    // j == 0, i > 0
    return PA * Overlap1D(i - 1, 0, PA, PB, p) +
           inv_2p * static_cast<double>(i - 1) * Overlap1D(i - 2, 0, PA, PB, p);
  }

  // 1D kinetic energy contribution.
  // T_x = li*lj*S(i-1,j-1) - 2b*li*S(i-1,j+1) - 2a*lj*S(i+1,j-1) + 4ab*S(i+1,j+1)
  static double Kinetic1D(int i, int j, double PA, double PB,
                           double a, double b, double p) {
    double t = 0.0;
    if (i > 0 && j > 0)
      t += static_cast<double>(i * j) * Overlap1D(i - 1, j - 1, PA, PB, p);
    if (i > 0)
      t -= 2.0 * b * static_cast<double>(i) * Overlap1D(i - 1, j + 1, PA, PB, p);
    if (j > 0)
      t -= 2.0 * a * static_cast<double>(j) * Overlap1D(i + 1, j - 1, PA, PB, p);
    t += 4.0 * a * b * Overlap1D(i + 1, j + 1, PA, PB, p);
    return t;
  }

  // Overlap for a pair of shells.
  static double OverlapPair(
      const GTOMolecule& mol,
      const GTOShell& si, int mi,
      const GTOShell& sj, int mj) {
    const double* Ra = &mol.positions[3 * si.atom_index];
    const double* Rb = &mol.positions[3 * sj.atom_index];
    const auto [lx_i, ly_i, lz_i] = CartExp(si.l, mi);
    const auto [lx_j, ly_j, lz_j] = CartExp(sj.l, mj);

    double total = 0.0;
    for (const auto& pi : si.primitives) {
      for (const auto& pj : sj.primitives) {
        const double a = pi.exponent, b = pj.exponent;
        const double p = a + b;
        const double mu = a * b / p;
        const double AB2 = (Ra[0]-Rb[0])*(Ra[0]-Rb[0]) +
                           (Ra[1]-Rb[1])*(Ra[1]-Rb[1]) +
                           (Ra[2]-Rb[2])*(Ra[2]-Rb[2]);
        const double prefac = std::pow(M_PI / p, 1.5) * std::exp(-mu * AB2);

        const double Px = (a * Ra[0] + b * Rb[0]) / p;
        const double Py = (a * Ra[1] + b * Rb[1]) / p;
        const double Pz = (a * Ra[2] + b * Rb[2]) / p;

        double sx = Overlap1D(lx_i, lx_j, Px - Ra[0], Px - Rb[0], p);
        double sy = Overlap1D(ly_i, ly_j, Py - Ra[1], Py - Rb[1], p);
        double sz = Overlap1D(lz_i, lz_j, Pz - Ra[2], Pz - Rb[2], p);

        const double Na = PrimNorm(a, lx_i, ly_i, lz_i);
        const double Nb = PrimNorm(b, lx_j, ly_j, lz_j);
        total += pi.coefficient * pj.coefficient * Na * Nb * prefac * sx * sy * sz;
      }
    }
    return total;
  }

  // Kinetic energy for a pair of shells.
  static double KineticPair(
      const GTOMolecule& mol,
      const GTOShell& si, int mi,
      const GTOShell& sj, int mj) {
    const double* Ra = &mol.positions[3 * si.atom_index];
    const double* Rb = &mol.positions[3 * sj.atom_index];
    const auto [lx_i, ly_i, lz_i] = CartExp(si.l, mi);
    const auto [lx_j, ly_j, lz_j] = CartExp(sj.l, mj);

    double total = 0.0;
    for (const auto& pi : si.primitives) {
      for (const auto& pj : sj.primitives) {
        const double a = pi.exponent, b = pj.exponent;
        const double p = a + b;
        const double mu = a * b / p;
        const double AB2 = (Ra[0]-Rb[0])*(Ra[0]-Rb[0]) +
                           (Ra[1]-Rb[1])*(Ra[1]-Rb[1]) +
                           (Ra[2]-Rb[2])*(Ra[2]-Rb[2]);
        const double prefac = std::pow(M_PI / p, 1.5) * std::exp(-mu * AB2);

        const double Px = (a * Ra[0] + b * Rb[0]) / p;
        const double Py = (a * Ra[1] + b * Rb[1]) / p;
        const double Pz = (a * Ra[2] + b * Rb[2]) / p;

        double sx = Overlap1D(lx_i, lx_j, Px - Ra[0], Px - Rb[0], p);
        double sy = Overlap1D(ly_i, ly_j, Py - Ra[1], Py - Rb[1], p);
        double sz = Overlap1D(lz_i, lz_j, Pz - Ra[2], Pz - Rb[2], p);

        double tx = Kinetic1D(lx_i, lx_j, Px - Ra[0], Px - Rb[0], a, b, p);
        double ty = Kinetic1D(ly_i, ly_j, Py - Ra[1], Py - Rb[1], a, b, p);
        double tz = Kinetic1D(lz_i, lz_j, Pz - Ra[2], Pz - Rb[2], a, b, p);

        const double Na = PrimNorm(a, lx_i, ly_i, lz_i);
        const double Nb = PrimNorm(b, lx_j, ly_j, lz_j);
        total += pi.coefficient * pj.coefficient * Na * Nb * 0.5 * prefac *
                 (tx * sy * sz + sx * ty * sz + sx * sy * tz);
      }
    }
    return total;
  }

  // 4-center ERI for contracted shells.
 private:
  static double ERI(
      const GTOMolecule& mol,
      const GTOShell& si, int mi,
      const GTOShell& sj, int mj,
      const GTOShell& sk, int mk,
      const GTOShell& sl, int ml) {
    const double* Ra = &mol.positions[3 * si.atom_index];
    const double* Rb = &mol.positions[3 * sj.atom_index];
    const double* Rc = &mol.positions[3 * sk.atom_index];
    const double* Rd = &mol.positions[3 * sl.atom_index];
    const auto [lx_i, ly_i, lz_i] = CartExp(si.l, mi);
    const auto [lx_j, ly_j, lz_j] = CartExp(sj.l, mj);
    const auto [lx_k, ly_k, lz_k] = CartExp(sk.l, mk);
    const auto [lx_l, ly_l, lz_l] = CartExp(sl.l, ml);

    double total = 0.0;
    for (const auto& pi : si.primitives) {
      for (const auto& pj : sj.primitives) {
        for (const auto& pk : sk.primitives) {
          for (const auto& pl : sl.primitives) {
            const double a = pi.exponent, b = pj.exponent;
            const double c = pk.exponent, d = pl.exponent;
            const double p = a + b, q = c + d;
            const double mu = a * b / p, nu = c * d / q;
            const double AB2 = (Ra[0]-Rb[0])*(Ra[0]-Rb[0]) +
                               (Ra[1]-Rb[1])*(Ra[1]-Rb[1]) +
                               (Ra[2]-Rb[2])*(Ra[2]-Rb[2]);
            const double CD2 = (Rc[0]-Rd[0])*(Rc[0]-Rd[0]) +
                               (Rc[1]-Rd[1])*(Rc[1]-Rd[1]) +
                               (Rc[2]-Rd[2])*(Rc[2]-Rd[2]);
            const double Px = (a*Ra[0] + b*Rb[0]) / p;
            const double Py = (a*Ra[1] + b*Rb[1]) / p;
            const double Pz = (a*Ra[2] + b*Rb[2]) / p;
            const double Qx = (c*Rc[0] + d*Rd[0]) / q;
            const double Qy = (c*Rc[1] + d*Rd[1]) / q;
            const double Qz = (c*Rc[2] + d*Rd[2]) / q;
            const double PQ2 = (Px-Qx)*(Px-Qx) + (Py-Qy)*(Py-Qy) + (Pz-Qz)*(Pz-Qz);

            const double prefac = 2.0 * std::pow(M_PI, 2.5) /
                                  (p * q * std::sqrt(p + q)) *
                                  std::exp(-mu * AB2) * std::exp(-nu * CD2);
            const double t = p * q / (p + q) * PQ2;

            // W = (p*P + q*Q) / (p + q) is the center of the combined Gaussian.
            // WP = W - P is required by the OS recursion.
            const double pq = p + q;
            const double WPx = (q * Qx - q * Px) / pq;
            const double WPy = (q * Qy - q * Py) / pq;
            const double WPz = (q * Qz - q * Pz) / pq;

            // OS recursion for the 4-center integral.
            // Build from ss|ss base case using the vertical recurrence.
            double prim_val = ERI_OS(
                lx_i, ly_i, lz_i, lx_j, ly_j, lz_j,
                lx_k, ly_k, lz_k, lx_l, ly_l, lz_l,
                Px - Ra[0], Py - Ra[1], Pz - Ra[2],
                Px - Rb[0], Py - Rb[1], Pz - Rb[2],
                Qx - Rc[0], Qy - Rc[1], Qz - Rc[2],
                Qx - Rd[0], Qy - Rd[1], Qz - Rd[2],
                WPx, WPy, WPz,
                p, q, t);

            const double Na = PrimNorm(a, lx_i, ly_i, lz_i);
            const double Nb = PrimNorm(b, lx_j, ly_j, lz_j);
            const double Nc = PrimNorm(c, lx_k, ly_k, lz_k);
            const double Nd = PrimNorm(d, lx_l, ly_l, lz_l);

            total += pi.coefficient * pj.coefficient *
                     pk.coefficient * pl.coefficient *
                     Na * Nb * Nc * Nd * prefac * prim_val;
          }
        }
      }
    }
    return total;
  }

  // OS recursion for ERI. For s and p functions (max l=1 per direction),
  // we use direct formulas. For d functions, use full recursion.
  // This implements the vertical recurrence relations (VRR) from Obara-Saika.
  static double ERI_OS(
      int lx_i, int ly_i, int lz_i,
      int lx_j, int ly_j, int lz_j,
      int lx_k, int ly_k, int lz_k,
      int lx_l, int ly_l, int lz_l,
      double PAx, double PAy, double PAz,
      double PBx, double PBy, double PBz,
      double QCx, double QCy, double QCz,
      double QDx, double QDy, double QDz,
      double WPx, double WPy, double WPz,
      double p, double q, double t) {
    // For s and p functions, we handle each Cartesian direction independently
    // using the 1D OS recursion for ERIs.
    //
    // The ss|ss base case: (s|s)(s|s) = F_0(t)
    // The VRR for increasing i (bra, first center):
    //   (i+1|j) = (PA + WP) * (i|j) + 1/(2p) * (i * (i-1|j) + j * (i|j-1))
    //             + 1/(2(p+q)) * (k * (i-1|j)(k-1|l) + l * (i|j)(k|l-1))
    //             + WP * (i|j)(k|l)^(m+1)
    //
    // This is complex. For s and p only (lx,ly,lz <= 1 for each function),
    // the maximum total angular momentum per direction is 2 (pp|pp).
    // We use the direct approach: compute the 1D integrals for each direction
    // and combine.
    //
    // Actually, the standard approach is to use the McMurchie-Davidson or
    // Rys quadrature. For simplicity with s and p functions, let me use
    // the direct product approach:
    //
    // (ij|kl) = sum_m sum_n coeff_x[m,n] * coeff_y[m,n] * coeff_z[m,n] * F_{m+n}(t)
    //
    // where the coefficients come from the 1D OS recursion for each direction.

    // For s and p functions, the 1D recursion is simple.
    // We use the transfer equation for bra and ket separately.

    // Actually, let me use the simplest approach: since we only support s and p,
    // the maximum angular momentum per Cartesian direction is 1 for each function,
    // so the total per direction is at most 2 (e.g., px * px = lx_total = 2).
    // The ERI can be computed as a sum of Boys functions with coefficients
    // from the 1D recursion.

    // Use the Head-Gordon/Pople approach: compute 1D coefficients for each
    // direction, then combine.
    auto cx = ERI_1D(lx_i, lx_j, lx_k, lx_l, PAx, PBx, QCx, QDx, WPx, p, q);
    auto cy = ERI_1D(ly_i, ly_j, ly_k, ly_l, PAy, PBy, QCy, QDy, WPy, p, q);
    auto cz = ERI_1D(lz_i, lz_j, lz_k, lz_l, PAz, PBz, QCz, QDz, WPz, p, q);

    double val = 0.0;
    for (const auto& ex : cx)
      for (const auto& ey : cy)
        for (const auto& ez : cz) {
          const int m = ex.m_idx + ey.m_idx + ez.m_idx;
          val += ex.coeff * ey.coeff * ez.coeff * BoysFm(m, t);
        }
    return val;
  }

  // 1D ERI coefficients using OS recursion.
  // Returns list of (coefficient, boys_index) pairs.
  // For s and p (i,j,k,l <= 1 per direction):
  static std::vector<CoeffPair> ERI_1D(
      int i, int j, int k, int l,
      double PA, double PB, double QC, double QD, double WP,
      double p, double q) {
    // Base case: (0,0|0,0) = {(1, 0)}
    if (i == 0 && j == 0 && k == 0 && l == 0) return {{1.0, 0}};

    // Use the VRR to increase i, then transfer for j, k, l.
    // For s/p functions, the maximum index is 1, so we handle cases directly.

    // (1,0|0,0) = PA * (0,0|0,0) + WP * (0,0|0,0)^{m+1}
    if (i == 1 && j == 0 && k == 0 && l == 0)
      return {{PA, 0}, {WP, 1}};

    // (0,1|0,0) = PB * (0,0|0,0) + WP * (0,0|0,0)^{m+1}
    if (i == 0 && j == 1 && k == 0 && l == 0)
      return {{PB, 0}, {WP, 1}};

    // (0,0|1,0) = QC * (0,0|0,0) - WP * (0,0|0,0)^{m+1}
    if (i == 0 && j == 0 && k == 1 && l == 0)
      return {{QC, 0}, {-WP, 1}};

    // (0,0|0,1) = QD * (0,0|0,0) - WP * (0,0|0,0)^{m+1}
    if (i == 0 && j == 0 && k == 0 && l == 1)
      return {{QD, 0}, {-WP, 1}};

    // (1,1|0,0) = PA*PB*(0,0|0,0) + (PA+PB)*WP*(0,0|0,0)^{m+1}
    //            + WP^2*(0,0|0,0)^{m+2} + 1/(2p)*(0,0|0,0)
    if (i == 1 && j == 1 && k == 0 && l == 0) {
      const double inv_2p = 1.0 / (2.0 * p);
      return {{PA * PB + inv_2p, 0},
              {(PA + PB) * WP, 1},
              {WP * WP, 2}};
    }

    // (1,0|1,0) = PA*QC*(0,0|0,0) + (QC*WP - PA*WP)*(0,0|0,0)^{m+1}
    //            - WP^2*(0,0|0,0)^{m+2} + 1/(2(p+q))*(0,0|0,0)
    if (i == 1 && j == 0 && k == 1 && l == 0) {
      const double inv_2pq = 1.0 / (2.0 * (p + q));
      return {{PA * QC + inv_2pq, 0},
              {QC * WP - PA * WP, 1},
              {-WP * WP, 2}};
    }

    // (1,0|0,1) = PA*QD*(0,0|0,0) + (QD*WP - PA*WP)*(0,0|0,0)^{m+1}
    //            - WP^2*(0,0|0,0)^{m+2} + 1/(2(p+q))*(0,0|0,0)
    if (i == 1 && j == 0 && k == 0 && l == 1) {
      const double inv_2pq = 1.0 / (2.0 * (p + q));
      return {{PA * QD + inv_2pq, 0},
              {QD * WP - PA * WP, 1},
              {-WP * WP, 2}};
    }

    // (0,1|1,0) = PB*QC*(0,0|0,0) + (QC*WP - PB*WP)*(0,0|0,0)^{m+1}
    //            - WP^2*(0,0|0,0)^{m+2} + 1/(2(p+q))*(0,0|0,0)
    if (i == 0 && j == 1 && k == 1 && l == 0) {
      const double inv_2pq = 1.0 / (2.0 * (p + q));
      return {{PB * QC + inv_2pq, 0},
              {QC * WP - PB * WP, 1},
              {-WP * WP, 2}};
    }

    // (0,1|0,1) = PB*QD*(0,0|0,0) + (QD*WP - PB*WP)*(0,0|0,0)^{m+1}
    //            - WP^2*(0,0|0,0)^{m+2} + 1/(2(p+q))*(0,0|0,0)
    if (i == 0 && j == 1 && k == 0 && l == 1) {
      const double inv_2pq = 1.0 / (2.0 * (p + q));
      return {{PB * QD + inv_2pq, 0},
              {QD * WP - PB * WP, 1},
              {-WP * WP, 2}};
    }

    // (0,0|1,1) = QC*QD*(0,0|0,0) - (QC+QD)*WP*(0,0|0,0)^{m+1}
    //            + WP^2*(0,0|0,0)^{m+2} + 1/(2q)*(0,0|0,0)
    if (i == 0 && j == 0 && k == 1 && l == 1) {
      const double inv_2q = 1.0 / (2.0 * q);
      return {{QC * QD + inv_2q, 0},
              {-(QC + QD) * WP, 1},
              {WP * WP, 2}};
    }

    // For higher combinations (pp|pp etc.), use the general recursion.
    // The 1D OS VRR includes cross-terms between bra and ket:
    // [i+1,j|k,l] = PA*[i,j|k,l] + 1/(2p)*(i*[i-1,j|k,l] + j*[i,j-1|k,l])
    //              + 1/(2(p+q))*(k*[i,j|k-1,l] + l*[i,j|k,l-1])
    //              + WP*[i,j|k,l]^{m+1}
    // [i,j|k+1,l] = QC*[i,j|k,l] + 1/(2q)*(k*[i,j|k-1,l] + l*[i,j|k,l-1])
    //              + 1/(2(p+q))*(i*[i-1,j|k,l] + j*[i,j-1|k,l])
    //              - WP*[i,j|k,l]^{m+1}
    const double inv_2p = 1.0 / (2.0 * p);
    const double inv_2q = 1.0 / (2.0 * q);
    const double inv_2pq = 1.0 / (2.0 * (p + q));

    // VRR: increase i
    if (i > 0) {
      auto base = ERI_1D(i - 1, j, k, l, PA, PB, QC, QD, WP, p, q);
      auto shift = ERI_1D(i - 1, j, k, l, PA, PB, QC, QD, WP, p, q);
      for (auto& c : shift) c.m_idx += 1;
      std::vector<CoeffPair> result;
      for (const auto& c : base) result.push_back({PA * c.coeff, c.m_idx});
      if (i > 1) {
        auto base2 = ERI_1D(i - 2, j, k, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base2)
          result.push_back({static_cast<double>(i - 1) * inv_2p * c.coeff, c.m_idx});
      }
      if (j > 0) {
        auto base_j = ERI_1D(i - 1, j - 1, k, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_j)
          result.push_back({static_cast<double>(j) * inv_2p * c.coeff, c.m_idx});
      }
      if (k > 0) {
        auto base_k = ERI_1D(i - 1, j, k - 1, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_k)
          result.push_back({static_cast<double>(k) * inv_2pq * c.coeff, c.m_idx});
      }
      if (l > 0) {
        auto base_l = ERI_1D(i - 1, j, k, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_l)
          result.push_back({static_cast<double>(l) * inv_2pq * c.coeff, c.m_idx});
      }
      for (const auto& c : shift) result.push_back({WP * c.coeff, c.m_idx});
      return result;
    }
    // Transfer: increase j
    if (j > 0) {
      auto base = ERI_1D(i, j - 1, k, l, PA, PB, QC, QD, WP, p, q);
      auto shift = ERI_1D(i, j - 1, k, l, PA, PB, QC, QD, WP, p, q);
      for (auto& c : shift) c.m_idx += 1;
      std::vector<CoeffPair> result;
      for (const auto& c : base) result.push_back({PB * c.coeff, c.m_idx});
      if (j > 1) {
        auto base2 = ERI_1D(i, j - 2, k, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base2)
          result.push_back({static_cast<double>(j - 1) * inv_2p * c.coeff, c.m_idx});
      }
      if (i > 0) {
        auto base_i = ERI_1D(i - 1, j - 1, k, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_i)
          result.push_back({static_cast<double>(i) * inv_2p * c.coeff, c.m_idx});
      }
      if (k > 0) {
        auto base_k = ERI_1D(i, j - 1, k - 1, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_k)
          result.push_back({static_cast<double>(k) * inv_2pq * c.coeff, c.m_idx});
      }
      if (l > 0) {
        auto base_l = ERI_1D(i, j - 1, k, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_l)
          result.push_back({static_cast<double>(l) * inv_2pq * c.coeff, c.m_idx});
      }
      for (const auto& c : shift) result.push_back({WP * c.coeff, c.m_idx});
      return result;
    }
    // VRR for ket: increase k
    if (k > 0) {
      auto base = ERI_1D(i, j, k - 1, l, PA, PB, QC, QD, WP, p, q);
      auto shift = ERI_1D(i, j, k - 1, l, PA, PB, QC, QD, WP, p, q);
      for (auto& c : shift) c.m_idx += 1;
      std::vector<CoeffPair> result;
      for (const auto& c : base) result.push_back({QC * c.coeff, c.m_idx});
      if (k > 1) {
        auto base2 = ERI_1D(i, j, k - 2, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base2)
          result.push_back({static_cast<double>(k - 1) * inv_2q * c.coeff, c.m_idx});
      }
      if (l > 0) {
        auto base_l = ERI_1D(i, j, k - 1, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_l)
          result.push_back({static_cast<double>(l) * inv_2q * c.coeff, c.m_idx});
      }
      if (i > 0) {
        auto base_i = ERI_1D(i - 1, j, k - 1, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_i)
          result.push_back({static_cast<double>(i) * inv_2pq * c.coeff, c.m_idx});
      }
      if (j > 0) {
        auto base_j = ERI_1D(i, j - 1, k - 1, l, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_j)
          result.push_back({static_cast<double>(j) * inv_2pq * c.coeff, c.m_idx});
      }
      for (const auto& c : shift) result.push_back({-WP * c.coeff, c.m_idx});
      return result;
    }
    // Transfer for ket: increase l
    if (l > 0) {
      auto base = ERI_1D(i, j, k, l - 1, PA, PB, QC, QD, WP, p, q);
      auto shift = ERI_1D(i, j, k, l - 1, PA, PB, QC, QD, WP, p, q);
      for (auto& c : shift) c.m_idx += 1;
      std::vector<CoeffPair> result;
      for (const auto& c : base) result.push_back({QD * c.coeff, c.m_idx});
      if (l > 1) {
        auto base2 = ERI_1D(i, j, k, l - 2, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base2)
          result.push_back({static_cast<double>(l - 1) * inv_2q * c.coeff, c.m_idx});
      }
      if (k > 0) {
        auto base_k = ERI_1D(i, j, k - 1, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_k)
          result.push_back({static_cast<double>(k) * inv_2q * c.coeff, c.m_idx});
      }
      if (i > 0) {
        auto base_i = ERI_1D(i - 1, j, k, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_i)
          result.push_back({static_cast<double>(i) * inv_2pq * c.coeff, c.m_idx});
      }
      if (j > 0) {
        auto base_j = ERI_1D(i, j - 1, k, l - 1, PA, PB, QC, QD, WP, p, q);
        for (const auto& c : base_j)
          result.push_back({static_cast<double>(j) * inv_2pq * c.coeff, c.m_idx});
      }
      for (const auto& c : shift) result.push_back({-WP * c.coeff, c.m_idx});
      return result;
    }
    return {{1.0, 0}};
  }

 public:
};

}  // namespace tides::scf
