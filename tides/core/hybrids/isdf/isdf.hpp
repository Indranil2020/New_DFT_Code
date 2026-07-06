#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace tides::hybrids {

// Interpolative Separable Density Fitting (ISDF) — T7.2.
//
// The exact exchange matrix K_{ij} = sum_{mu,nu} P_{mu,nu} (i mu | nu j)
// requires O(N^4) work. ISDF reduces this to O(N * N_mu^2) by finding
// interpolation points {r_μ} such that the 4-index integral is approximated:
//   (i j | r) ≈ sum_mu' theta_{i j, mu'} * x_mu'(r)
// where x_mu'(r) are interpolating functions (based on the selected points).
//
// The randomized ISDF algorithm (Lu-Ying, JCP 2015):
//   1. Sample the orbital product matrix on the grid: M(r, ij) = phi_i(r) * phi_j(r).
//   2. Apply a random projection to reduce the column dimension.
//   3. QR-pivot the projected matrix to select the interpolation points.
//   4. Fit the interpolating functions via least-squares.
//
// Observable (T7.2): exchange-energy error vs rank curve for benzene;
// rank for 0.1 mHa recorded.

struct ISDFResult {
  std::vector<std::size_t> interp_point_indices;  // selected grid points
  std::vector<double> interp_vectors;  // x_mu'(r) on the grid (n_grid x n_interp)
  std::size_t rank = 0;
  double reconstruction_error = 0.0;  // ||M_approx - M_exact||_F / ||M_exact||_F
};

class ISDF {
 public:
  // Select interpolation points via randomized QR with column pivoting.
  //   M:      the matrix to decompose (n_grid x n_orb_pairs, row-major)
  //   n_grid: number of grid points (rows)
  //   n_pairs: number of orbital pairs (columns)
  //   rank:   desired rank (number of interpolation points)
  //   seed:   random seed for the projection
  static ISDFResult SelectPoints(const std::vector<double>& M,
                                 std::size_t n_grid, std::size_t n_pairs,
                                 std::size_t rank, std::uint64_t seed = 42) {
    ISDFResult res;
    res.rank = std::min(rank, std::min(n_grid, n_pairs));
    if (n_grid == 0 || n_pairs == 0) return res;

    // Step 1: Random projection. Sample a random matrix Omega (n_pairs x rank)
    // and compute Y = M * Omega (n_grid x rank).
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> Omega(n_pairs * res.rank, 0.0);
    for (auto& v : Omega) v = g(rng) / std::sqrt(static_cast<double>(res.rank));

    std::vector<double> Y(n_grid * res.rank, 0.0);
    for (std::size_t i = 0; i < n_grid; ++i)
      for (std::size_t k = 0; k < res.rank; ++k) {
        double s = 0.0;
        for (std::size_t j = 0; j < n_pairs; ++j)
          s += M[i * n_pairs + j] * Omega[j * res.rank + k];
        Y[i * res.rank + k] = s;
      }

    // Step 2: QR with column pivoting (simplified: pick the row with the
    // largest norm, project out, repeat — this is the greedy pivoting).
    std::vector<bool> used(n_grid, false);
    res.interp_point_indices.reserve(res.rank);
    std::vector<double> Q(n_grid * res.rank, 0.0);

    for (std::size_t k = 0; k < res.rank; ++k) {
      // Find the unused row with the largest residual norm.
      std::size_t best = 0;
      double best_norm = -1.0;
      for (std::size_t i = 0; i < n_grid; ++i) {
        if (used[i]) continue;
        double norm = 0.0;
        for (std::size_t kk = 0; kk < res.rank; ++kk)
          norm += Y[i * res.rank + kk] * Y[i * res.rank + kk];
        if (norm > best_norm) {
          best_norm = norm;
          best = i;
        }
      }
      used[best] = true;
      res.interp_point_indices.push_back(best);

      // Copy the selected row into Q.
      for (std::size_t kk = 0; kk < res.rank; ++kk)
        Q[best * res.rank + kk] = Y[best * res.rank + kk];
    }

    // Step 3: Fit the interpolating vectors x_mu'(r) via least-squares.
    // M_approx(r, ij) = sum_mu' x_mu'(r) * M(r_mu', ij)
    // => x = M * M_interp^T * (M_interp * M_interp^T)^{-1}
    // where M_interp is the submatrix of M at the interpolation points.
    const std::size_t r = res.rank;
    
    // Extract M_interp: r x n_pairs (rows at interpolation points).
    std::vector<double> M_interp(r * n_pairs, 0.0);
    for (std::size_t mu = 0; mu < r; ++mu) {
      const std::size_t idx = res.interp_point_indices[mu];
      for (std::size_t j = 0; j < n_pairs; ++j)
        M_interp[mu * n_pairs + j] = M[idx * n_pairs + j];
    }
    
    // Compute G = M_interp * M_interp^T (r x r).
    std::vector<double> G(r * r, 0.0);
    for (std::size_t i = 0; i < r; ++i)
      for (std::size_t j = 0; j < r; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n_pairs; ++k)
          s += M_interp[i * n_pairs + k] * M_interp[j * n_pairs + k];
        G[i * r + j] = s;
      }
    
    // Invert G via Gaussian elimination with partial pivoting.
    // Augment [G | I] and reduce.
    std::vector<double> aug(r * 2 * r, 0.0);
    for (std::size_t i = 0; i < r; ++i) {
      for (std::size_t j = 0; j < r; ++j) aug[i * 2 * r + j] = G[i * r + j];
      aug[i * 2 * r + r + i] = 1.0;
    }
    for (std::size_t col = 0; col < r; ++col) {
      // Find pivot.
      std::size_t piv = col;
      double max_val = std::abs(aug[col * 2 * r + col]);
      for (std::size_t row = col + 1; row < r; ++row) {
        double val = std::abs(aug[row * 2 * r + col]);
        if (val > max_val) { max_val = val; piv = row; }
      }
      if (max_val < 1e-30) continue;  // singular column, skip
      if (piv != col) {
        for (std::size_t j = 0; j < 2 * r; ++j)
          std::swap(aug[col * 2 * r + j], aug[piv * 2 * r + j]);
      }
      double diag = aug[col * 2 * r + col];
      for (std::size_t j = 0; j < 2 * r; ++j) aug[col * 2 * r + j] /= diag;
      for (std::size_t row = 0; row < r; ++row) {
        if (row == col) continue;
        double factor = aug[row * 2 * r + col];
        for (std::size_t j = 0; j < 2 * r; ++j)
          aug[row * 2 * r + j] -= factor * aug[col * 2 * r + j];
      }
    }
    // Extract G_inv.
    std::vector<double> G_inv(r * r, 0.0);
    for (std::size_t i = 0; i < r; ++i)
      for (std::size_t j = 0; j < r; ++j)
        G_inv[i * r + j] = aug[i * 2 * r + r + j];
    
    // Compute interpolation vectors: x_mu(r) = sum_nu M(r, :) . M_interp(nu, :)^T * G_inv[nu, mu]
    // First compute M * M_interp^T = n_grid x r.
    std::vector<double> M_Mt(n_grid * r, 0.0);
    for (std::size_t i = 0; i < n_grid; ++i)
      for (std::size_t mu = 0; mu < r; ++mu) {
        double s = 0.0;
        for (std::size_t j = 0; j < n_pairs; ++j)
          s += M[i * n_pairs + j] * M_interp[mu * n_pairs + j];
        M_Mt[i * r + mu] = s;
      }
    
    // x = M_Mt * G_inv (n_grid x r).
    res.interp_vectors.assign(n_grid * r, 0.0);
    for (std::size_t i = 0; i < n_grid; ++i)
      for (std::size_t mu = 0; mu < r; ++mu) {
        double s = 0.0;
        for (std::size_t nu = 0; nu < r; ++nu)
          s += M_Mt[i * r + nu] * G_inv[nu * r + mu];
        res.interp_vectors[i * r + mu] = s;
      }

    // Compute reconstruction error.
    double err = 0.0, norm_M = 0.0;
    for (std::size_t i = 0; i < n_grid; ++i)
      for (std::size_t j = 0; j < n_pairs; ++j) {
        double approx = 0.0;
        for (std::size_t mu = 0; mu < res.rank; ++mu) {
          const std::size_t idx = res.interp_point_indices[mu];
          approx += res.interp_vectors[i * res.rank + mu] * M[idx * n_pairs + j];
        }
        const double diff = approx - M[i * n_pairs + j];
        err += diff * diff;
        norm_M += M[i * n_pairs + j] * M[i * n_pairs + j];
      }
    res.reconstruction_error = (norm_M > 1e-30) ? std::sqrt(err / norm_M) : 0.0;

    return res;
  }

  // Build the orbital product matrix M(r, ij) = phi_i(r) * phi_j(r).
  //   orbitals: n_orb x n_grid (each orbital sampled on the grid)
  //   n_orb:   number of orbitals
  //   n_grid:  number of grid points
  // Returns M (n_grid x n_orb*(n_orb+1)/2, row-major, only unique pairs).
  static std::vector<double> BuildOrbitalProducts(
      const std::vector<double>& orbitals, std::size_t n_orb,
      std::size_t n_grid) {
    std::vector<double> M;
    std::size_t n_pairs = 0;
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = i; j < n_orb; ++j) ++n_pairs;
    M.assign(n_grid * n_pairs, 0.0);
    std::size_t pair = 0;
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = i; j < n_orb; ++j, ++pair)
        for (std::size_t g = 0; g < n_grid; ++g)
          M[g * n_pairs + pair] = orbitals[i * n_grid + g] *
                                   orbitals[j * n_grid + g];
    return M;
  }
};

}  // namespace tides::hybrids
