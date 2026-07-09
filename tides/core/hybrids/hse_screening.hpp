#pragma once

// T7.4: Short-range HSE screening in tiles.
//
// HSE (Heyd-Scuseria-Ernzerhof) uses the error-function screened Coulomb
// operator to split exchange into short-range (SR) and long-range (LR):
//   1/r = erfc(omega * r) / r  +  erf(omega * r) / r
//          \_ SR (exact)        \_ LR (DFT)
//
// In the tile framework, this means:
//   1. For each pair of atoms (a, b), compute the distance |R_a - R_b|.
//   2. If |R_a - R_b| > r_cut_screen, the SR exchange tile is zero
//      (screened out). This is the "short-range screening" — tiles beyond
//      the screening radius are dropped from the exact exchange.
//   3. The screening parameter omega controls the range separation.
//      HSE06: omega = 0.11 Bohr^-1, alpha = 0.25 (SR exact exchange fraction).
//
// Observable (T7.4): the screened exchange matrix has the correct sparsity
// pattern (tiles beyond r_cut are zero), and the exchange energy matches
// a dense reference within tolerance.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/status.hpp"

namespace tides::hybrids {

// HSE screening parameters.
struct HSEParams {
  double omega = 0.11;       // screening parameter (Bohr^-1), HSE06 default
  double alpha_sr = 0.25;    // SR exact exchange fraction
  double alpha_lr = 0.0;     // LR exact exchange fraction (0 for HSE06)
  double r_cut_screen = 0.0; // if > 0, hard cutoff for SR exchange tiles
};

// Compute the screened Coulomb interaction for a pair at distance r.
// SR part: V_SR(r) = erfc(omega * r) / r
// LR part: V_LR(r) = erf(omega * r) / r
struct ScreenedCoulomb {
  double omega;

  explicit ScreenedCoulomb(double omega_ = 0.11) : omega(omega_) {}

  // Short-range: erfc(omega*r) / r
  [[nodiscard]] double SR(double r) const {
    if (r < 1e-30) return 2.0 / std::sqrt(M_PI) * omega;  // limit r->0
    return std::erfc(omega * r) / r;
  }

  // Long-range: erf(omega*r) / r
  [[nodiscard]] double LR(double r) const {
    if (r < 1e-30) return 0.0;  // limit r->0
    return std::erf(omega * r) / r;
  }

  // Full Coulomb (should equal SR + LR = 1/r).
  [[nodiscard]] double Full(double r) const {
    if (r < 1e-30) return 0.0;
    return 1.0 / r;
  }
};

// Build the screened exchange matrix with tile-level screening.
//
// For each pair of basis functions (i, j) centered on atoms (a, b):
//   K_SR[i][j] = alpha_sr * erfc(omega * |R_a - R_b|) / |R_a - R_b| * P[i][j]
//
// If r_cut_screen > 0 and |R_a - R_b| > r_cut_screen, the tile is skipped
// (set to zero). This implements the short-range screening in the tile
// framework: tiles beyond the screening radius are dropped.
//
// @param n           Matrix dimension (n_basis)
// @param P           Density matrix (n x n, row-major)
// @param atom_centers  For each basis function index, the atom index it belongs to
// @param atom_positions  For each atom, 3D position (3 * n_atoms)
// @param params      HSE screening parameters
// @return            Screened exchange matrix K_SR (n x n, row-major)
[[nodiscard]] inline std::vector<double> BuildScreenedExchange(
    std::size_t n, const std::vector<double>& P,
    const std::vector<std::size_t>& atom_centers,
    const std::vector<double>& atom_positions,
    const HSEParams& params) {
  std::vector<double> K_SR(n * n, 0.0);
  if (n == 0 || P.size() != n * n || atom_centers.size() != n) return K_SR;

  ScreenedCoulomb sc(params.omega);
  const std::size_t n_atoms = atom_positions.size() / 3;

  // Track which tiles are screened out for diagnostics.
  std::size_t n_tiles_total = 0, n_tiles_screened = 0;

  for (std::size_t i = 0; i < n; ++i) {
    std::size_t a = atom_centers[i];
    if (a >= n_atoms) continue;
    double ax = atom_positions[3 * a];
    double ay = atom_positions[3 * a + 1];
    double az = atom_positions[3 * a + 2];

    for (std::size_t j = 0; j < n; ++j) {
      std::size_t b = atom_centers[j];
      if (b >= n_atoms) continue;
      n_tiles_total++;

      double bx = atom_positions[3 * b];
      double by = atom_positions[3 * b + 1];
      double bz = atom_positions[3 * b + 2];
      double dx = ax - bx, dy = ay - by, dz = az - bz;
      double r = std::sqrt(dx * dx + dy * dy + dz * dz);

      // Hard cutoff screening.
      if (params.r_cut_screen > 0.0 && r > params.r_cut_screen) {
        n_tiles_screened++;
        continue;  // tile is zero (screened out)
      }

      // SR screened Coulomb weight.
      double v_sr = sc.SR(r);
      K_SR[i * n + j] = params.alpha_sr * v_sr * P[i * n + j];
    }
  }

  return K_SR;
}

// Compute the screened exchange energy: E_x_SR = -0.5 * Tr(P * K_SR)
[[nodiscard]] inline double ScreenedExchangeEnergy(
    std::size_t n, const std::vector<double>& P,
    const std::vector<double>& K_SR) {
  double E = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      E += P[i * n + j] * K_SR[j * n + i];
  return -0.5 * E;
}

// Screening diagnostics: count how many atom pairs are within/outside
// the screening radius.
struct ScreeningStats {
  std::size_t n_pairs_total = 0;
  std::size_t n_pairs_screened = 0;  // outside r_cut
  std::size_t n_pairs_active = 0;    // within r_cut
  double fraction_screened = 0.0;
};

[[nodiscard]] inline ScreeningStats ComputeScreeningStats(
    const std::vector<double>& atom_positions,
    double r_cut_screen) {
  ScreeningStats stats;
  const std::size_t n_atoms = atom_positions.size() / 3;
  if (r_cut_screen <= 0.0 || n_atoms == 0) return stats;

  for (std::size_t a = 0; a < n_atoms; ++a)
    for (std::size_t b = a + 1; b < n_atoms; ++b) {
      double dx = atom_positions[3 * a] - atom_positions[3 * b];
      double dy = atom_positions[3 * a + 1] - atom_positions[3 * b + 1];
      double dz = atom_positions[3 * a + 2] - atom_positions[3 * b + 2];
      double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      stats.n_pairs_total++;
      if (r > r_cut_screen)
        stats.n_pairs_screened++;
      else
        stats.n_pairs_active++;
    }
  stats.fraction_screened = (stats.n_pairs_total > 0)
      ? static_cast<double>(stats.n_pairs_screened) /
        static_cast<double>(stats.n_pairs_total)
      : 0.0;
  return stats;
}

}  // namespace tides::hybrids
