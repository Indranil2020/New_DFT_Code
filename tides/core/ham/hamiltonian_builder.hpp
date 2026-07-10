#pragma once

// Hamiltonian builder — shared H assembly from component matrices.
//
// Both MoleculeDriver (GTO) and NaoDriver (NAO) build H = T + V_ext + V_H + V_xc
// inside their SCF build_H lambdas. This module extracts that common pattern
// into a reusable class so the logic is not duplicated.
//
// The builder is stateless w.r.t. the SCF iteration — it holds references to
// the fixed matrices (T, V_ext) and accepts callbacks for the P-dependent
// parts (rho build, Hartree, XC, vmat build). This matches the existing
// callback architecture used by SCFDriver.
//
// WP6 T6.1: "The SCF driver calls build_H(P) which assembles the full
// Hamiltonian from fixed (T, V_ext) and P-dependent (V_H, V_xc) parts."

#include <functional>
#include <string>
#include <vector>

#include "common/config.hpp"

namespace tides::ham {

// Energy components returned by the energy function.
struct EnergyComponents {
  double E_kin = 0.0;
  double E_ne = 0.0;
  double E_H = 0.0;
  double E_xc = 0.0;
  double E_ion = 0.0;
  double E_total = 0.0;
};

// Callbacks for P-dependent quantities.
// Each takes the density matrix P (n×n, row-major) and returns the result.
struct HamiltonianCallbacks {
  // Build density on grid from P: rho(r) = sum_ij P_ij phi_i(r) phi_j(r)
  std::function<std::vector<double>(const std::vector<double>&)> build_rho;

  // Build Hartree potential matrix from rho: V_H_ij = integral V_H(r) phi_i phi_j
  // Returns the n×n matrix and optionally the Hartree energy.
  std::function<std::pair<std::vector<double>, double>(
      const std::vector<double>&)> build_hartree;

  // Build XC potential matrix and energy from rho.
  // Returns (V_xc matrix, E_xc energy).
  std::function<std::pair<std::vector<double>, double>(
      const std::vector<double>&)> build_xc;
};

// Hamiltonian builder assembles H = T + V_ext + V_H(P) + V_xc(P).
class HamiltonianBuilder {
 public:
  // n:       matrix dimension (basis size)
  // T:       kinetic energy matrix (n×n, fixed)
  // V_ext:   external (nuclear attraction) matrix (n×n, fixed)
  // occ_factor: occupation scaling factor (2.0 for spin-paired, n_e/n_occ)
  HamiltonianBuilder(std::size_t n,
                     std::vector<double> T,
                     std::vector<double> V_ext,
                     double occ_factor)
      : n_(n), T_(std::move(T)), V_ext_(std::move(V_ext)),
        occ_factor_(occ_factor) {}

  // Build H from density matrix P using the provided callbacks.
  // Returns H (n×n row-major). Also caches intermediate results for energy_fn.
  std::vector<double> BuildH(const std::vector<double>& P,
                             const HamiltonianCallbacks& cb) {
    // Scale P by occupation factor.
    P2_.assign(n_ * n_, 0.0);
    for (std::size_t i = 0; i < n_ * n_; ++i) P2_[i] = occ_factor_ * P[i];

    // Build rho, V_H, V_xc via callbacks.
    cached_rho_ = cb.build_rho(P2_);
    auto [vh, e_h] = cb.build_hartree(cached_rho_);
    auto [vxc, e_xc] = cb.build_xc(cached_rho_);

    // Cache for energy computation.
    cached_V_H_ = std::move(vh);
    cached_V_xc_ = std::move(vxc);
    cached_E_H_ = e_h;
    cached_E_xc_ = e_xc;

    // H = T + V_ext + V_H + V_xc.
    std::vector<double> H(n_ * n_, 0.0);
    for (std::size_t i = 0; i < n_ * n_; ++i) {
      H[i] = T_[i] + V_ext_[i] + cached_V_H_[i] + cached_V_xc_[i];
    }
    return H;
  }

  // Compute total energy from cached H build + eigenvalues.
  // E_total = E_kin + E_ne + E_H + E_xc + E_ion
  // where E_kin = sum_eps - E_ne - 2*E_H - trace(P2, V_xc) (double-counting).
  double ComputeEnergy(const std::vector<double>& eigenvalues,
                       std::size_t n_occ, double E_ion) {
    auto trace = [this](const std::vector<double>& A,
                        const std::vector<double>& B) {
      double s = 0.0;
      for (std::size_t i = 0; i < n_ * n_; ++i) s += A[i] * B[i];
      return s;
    };

    double sum_eps = 0.0;
    for (std::size_t k = 0; k < n_occ && k < n_; ++k)
      sum_eps += occ_factor_ * eigenvalues[k];

    double E_ne = trace(P2_, V_ext_);
    double E_H = cached_E_H_;
    double E_xc = cached_E_xc_;
    double E_kin = sum_eps - E_ne - 2.0 * E_H - trace(P2_, cached_V_xc_);
    double E_total = E_kin + E_ne + E_H + E_xc + E_ion;

    return E_total;
  }

  // Accessors for cached components.
  const std::vector<double>& rho() const { return cached_rho_; }
  const std::vector<double>& P2() const { return P2_; }
  const std::vector<double>& V_H() const { return cached_V_H_; }
  const std::vector<double>& V_xc() const { return cached_V_xc_; }

 private:
  std::size_t n_;
  std::vector<double> T_;
  std::vector<double> V_ext_;
  double occ_factor_;

  // Cached from last BuildH call.
  std::vector<double> P2_;
  std::vector<double> cached_rho_;
  std::vector<double> cached_V_H_;
  std::vector<double> cached_V_xc_;
  double cached_E_H_ = 0.0;
  double cached_E_xc_ = 0.0;
};

}  // namespace tides::ham
