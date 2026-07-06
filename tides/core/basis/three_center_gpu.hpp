#pragma once

#include <cstddef>
#include <vector>

#include "basis/two_center_integrals.hpp"
#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::basis {

// Three-center integral result: V_nl matrix (nonlocal pseudopotential).
struct ThreeCenterGpuResult {
  std::vector<double> V_nl;  // n_basis x n_basis nonlocal PP matrix
  double kernel_ms = 0.0;
  std::size_t n_triplets = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool ThreeCenterCudaAvailable();

// Assemble the three-center KB nonlocal pseudopotential matrix on GPU.
//
// V_nl[ab] = Σ_c Σ_l h_l^c × ⟨φ_a|β_l^c⟩ × ⟨β_l^c|φ_b⟩
//
// Each factor ⟨φ_a|β_l^c⟩ is a two-center integral evaluated via spline
// at R = |r_a - r_c|, with Slater-Koster angular coupling.
//
// @param positions   Atomic positions (3*n_atoms)
// @param l_per_atom  Angular momentum per atom
// @param basis_offsets  Starting basis index per atom
// @param n_basis     Total basis size
// @param kb_centers  Indices of atoms that have KB projectors
// @param kb_l        Angular momentum per KB channel
// @param kb_coeff    KB coefficient h_l per KB channel
// @param phi_beta_splines  Tabulated ⟨φ|β_l⟩ radial integrals (one per KB channel)
// @param beta_phi_splines  Tabulated ⟨β_l|φ⟩ radial integrals (one per KB channel)
[[nodiscard]] Result<ThreeCenterGpuResult> AssembleThreeCenterCuda(
    const std::vector<double>& positions,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const std::vector<int>& kb_centers,
    const std::vector<int>& kb_l,
    const std::vector<double>& kb_coeff,
    const std::vector<CubicSpline>& phi_beta_splines,
    const std::vector<CubicSpline>& beta_phi_splines);

}  // namespace tides::basis
