#pragma once

// Shared Hamiltonian assembly: H = T + V_ext + V_H + V_xc + V_nl.
//
// AUDIT C1 FIX: Extracts the H assembly pattern shared by nao_driver and
// molecule_driver into a single location. Both drivers were duplicating the
// same loop. This is the canonical Hamiltonian builder for the product path.

#include <cstddef>
#include <vector>

namespace tides::ham {

// Assemble H = T + V_ext + V_H + V_xc (+ V_nl if non-empty).
// All matrices are n×n row-major. V_nl may be empty (no pseudopotential).
inline std::vector<double> AssembleH(
    std::size_t n,
    const std::vector<double>& T,
    const std::vector<double>& V_ext,
    const std::vector<double>& V_H,
    const std::vector<double>& V_xc,
    const std::vector<double>& V_nl) {
  std::vector<double> H(n * n, 0.0);
  for (std::size_t i = 0; i < n * n; ++i) {
    H[i] = T[i] + V_ext[i] + V_H[i] + V_xc[i];
    if (!V_nl.empty()) H[i] += V_nl[i];
  }
  return H;
}

}  // namespace tides::ham
