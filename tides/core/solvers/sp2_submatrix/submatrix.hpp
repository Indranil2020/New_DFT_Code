#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <vector>

#include "solvers/sp2_submatrix/sp2.hpp"

namespace tides::solvers {

// Submatrix construction (T5.2): per-atom principal submatrices from the
// sparsity graph; batch by size class.
//
// For each atom a, extract the principal submatrix of H and S over the
// sparsity neighborhood of a (the atom itself + its neighbors within the
// density-matrix decay radius). Purify each submatrix independently (dense,
// batched on GPU in production), then write back the block column.
//
// Observable (T5.2): equals global SP2 on 500-2000-atom proxies within
// declared block tolerance.
//
// For the CPU reference we use a simple neighbor list (distance-based) and
// dense submatrices. The GPU path uses WP1 TileMat for batched dense SP2.

struct SubmatrixResult {
  std::vector<double> P;           // reconstructed global density matrix
  std::vector<std::size_t> atom_order;  // atoms processed
  int n_submatrices = 0;
  double max_block_error = 0.0;   // max ||P_block^2 - P_block||
};

class SubmatrixBuilder {
 public:
  // Build per-atom submatrices, purify each, and reconstruct the global P.
  //   n:       matrix dimension
  //   H, S:    global matrices (row-major, n x n)
  //   n_e:     number of electrons
  //   n_occ:   number of occupied states
  //   neighbor_list: for each atom, the list of neighbor atom indices
  //   lambda_min, lambda_max: spectral bounds of the global problem
  static SubmatrixResult BuildAndPurify(
      std::size_t n, const std::vector<double>& H,
      const std::vector<double>& S, double n_e, double mu,
      const std::vector<std::vector<std::size_t>>& neighbor_list,
      double lambda_min, double lambda_max) {
    SubmatrixResult res;
    res.P.assign(n * n, 0.0);
    res.n_submatrices = 0;
    res.max_block_error = 0.0;

    const std::size_t n_atoms = neighbor_list.size();
    if (n_atoms == 0 || n == 0) return res;

    for (std::size_t a = 0; a < n_atoms; ++a) {
      // Collect the indices of this atom's neighborhood (itself + neighbors).
      std::vector<std::size_t> indices = {a};
      for (std::size_t nb : neighbor_list[a]) {
        if (nb < n && nb != a) indices.push_back(nb);
      }
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

      const std::size_t block_n = indices.size();
      if (block_n == 0) continue;

      // Extract the submatrix H_block, S_block (block_n x block_n).
      std::vector<double> H_block(block_n * block_n, 0.0);
      std::vector<double> S_block(block_n * block_n, 0.0);
      for (std::size_t i = 0; i < block_n; ++i)
        for (std::size_t j = 0; j < block_n; ++j) {
          H_block[i * block_n + j] = H[indices[i] * n + indices[j]];
          S_block[i * block_n + j] = S[indices[i] * n + indices[j]];
        }

      // Purify the submatrix.
      auto sp2 = SP2Purification::Purify(block_n, H_block, S_block, n_e, mu,
                                          lambda_min, lambda_max);
      res.n_submatrices++;

      // Write back the block column (the rows of P owned by atom a).
      for (std::size_t i = 0; i < block_n; ++i)
        for (std::size_t j = 0; j < block_n; ++j)
          res.P[indices[i] * n + indices[j]] = sp2.P[i * block_n + j];

      res.max_block_error = std::max(res.max_block_error, sp2.idempotency_err);
    }

    return res;
  }

  // Build a simple distance-based neighbor list for a 1D chain (proxy for
  // a-Si:H linear systems). Atom i's neighbors are i-1 and i+1 (and optionally
  // i-2, i+2 for larger radius).
  static std::vector<std::vector<std::size_t>> ChainNeighborList(
      std::size_t n_atoms, int radius = 1) {
    std::vector<std::vector<std::size_t>> nl(n_atoms);
    for (std::size_t a = 0; a < n_atoms; ++a)
      for (int r = 1; r <= radius; ++r) {
        if (a >= static_cast<std::size_t>(r)) nl[a].push_back(a - r);
        if (a + static_cast<std::size_t>(r) < n_atoms) nl[a].push_back(a + r);
      }
    return nl;
  }
};

}  // namespace tides::solvers
