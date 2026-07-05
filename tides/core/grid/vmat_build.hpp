#pragma once

// T3.3: v(r) -> H tiles adjoint map.
// The density build (T3.2) maps P -> rho(r): rho(r) = sum_ij P_ij phi_i(r) phi_j(r).
// The adjoint map maps a potential v(r) back to the Hamiltonian tile:
//   H_ij = integral v(r) phi_i(r) phi_j(r) d^3r
// These two operations are adjoints: <A P, v> = <P, A^T v> where A is the
// orbital-product operator. The T3.3 observable is:
//   adjointness |<A P, w> - <P, A^T w>| <= 1e-12 on 100 random pairs (FP64).
//
// For the CPU reference we implement both the forward (P -> rho) and adjoint
// (v -> H) using direct grid integration. The GPU tile-batched path uses WP1
// TileMat grouped GEMM for the orbital products; this CPU path is the FP64
// oracle.

#include <cmath>
#include <cstddef>
#include <vector>

#include "grid/dual_grid.hpp"

namespace tides::grid {

class VmatBuilder {
 public:
  // Forward: density from density matrix P and orbitals.
  // rho(r) = sum_{ij} P_{ij} * phi_i(r) * phi_j(r)
  // P is symmetric; we sum over ALL i,j (not just lower triangle) so the
  // adjointness identity <A P, w> = <P, A^T w> holds exactly.
  static std::vector<double> BuildRho(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    std::vector<double> rho(N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t j = 0; j < n_orb; ++j) {
        const double Pij = P[i * n_orb + j];
        if (std::fabs(Pij) < 1e-30) continue;
        for (std::size_t g = 0; g < N; ++g)
          rho[g] += Pij * orbitals[i][g] * orbitals[j][g];
      }
    }
    return rho;
  }

  // Adjoint: Hamiltonian tile from potential v(r) and orbitals.
  // H_ij = integral v(r) phi_i(r) phi_j(r) d^3r
  // Returns a symmetric matrix (lower triangle, row-major).
  static std::vector<double> BuildHmat(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& v) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    std::vector<double> H(n_orb * n_orb, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t j = i; j < n_orb; ++j) {
        double s = 0.0;
        for (std::size_t g = 0; g < N; ++g)
          s += v[g] * orbitals[i][g] * orbitals[j][g];
        H[i * n_orb + j] = s * dv;
        H[j * n_orb + i] = s * dv;  // symmetric
      }
    }
    return H;
  }

  // Adjointness check: <A P, w> should equal <P, A^T w> for random P, w.
  // A is the forward operator (P -> rho), A^T is the adjoint (v -> H).
  // <A P, w> = integral (A P)(r) * w(r) d^3r = integral rho_P(r) * w(r) d^3r
  // <P, A^T w> = sum_{ij} P_ij * H_w[ij] = sum_{ij} P_ij * integral w(r) phi_i phi_j d^3r
  // These are equal by construction (Fubini's theorem); the test validates the
  // implementation doesn't have a transposition/discretization bug.
  static double CheckAdjointness(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P, const std::vector<double>& w) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;

    // Forward: rho_P = A(P)
    auto rho_P = BuildRho(grid, orbitals, P);
    // <A P, w> = integral rho_P(r) * w(r) d^3r
    double AP_w = 0.0;
    for (std::size_t g = 0; g < N; ++g)
      AP_w += rho_P[g] * w[g] * dv;

    // Adjoint: H_w = A^T(w)
    auto H_w = BuildHmat(grid, orbitals, w);
    // <P, A^T w> = sum_{ij} P_ij * H_w[ij]
    double P_ATw = 0.0;
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = 0; j < n_orb; ++j)
        P_ATw += P[i * n_orb + j] * H_w[i * n_orb + j];

    return std::fabs(AP_w - P_ATw);
  }
};

}  // namespace tides::grid
