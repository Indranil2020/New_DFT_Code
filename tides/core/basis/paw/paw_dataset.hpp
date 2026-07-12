#pragma once

// PAW (Projector Augmented-Wave) data structures for TIDES.
//
// PAW maps smooth pseudo-quantities to all-electron quantities via a linear
// transformation (Blöchl, PRB 50, 17953, 1994):
//
//   |psi> = |psi_tilde> + sum_i (|phi_i> - |phi_tilde_i>) <p_i|psi_tilde>
//
// where phi_i are all-electron partial waves, phi_tilde_i are pseudo partial
// waves, and p_i are projector functions. All are atom-centered radial
// functions times spherical harmonics — structurally identical to NAO.
//
// The PAW correction is a per-atom, local operation: each atom has an
// augmentation sphere of radius r_c. The correction adds a small dense
// (n_proj x n_proj) block to the diagonal of H and the density — no
// cross-tile coupling. This makes PAW naturally compatible with the tile
// substrate (diagonal-block correction, batched across atoms).
//
// Reference: PAW_FEASIBILITY_MEMO.md §3 (integration plan).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tides::basis::paw {

// A single PAW partial-wave channel: (l, radial functions, projectors).
struct PAWChannel {
  int l = 0;
  // All-electron partial wave: phi_i(r) * Y_lm
  std::vector<double> phi;       // radial part on r_grid
  // Pseudo partial wave: phi_tilde_i(r) * Y_lm (smooth, matches phi outside r_c)
  std::vector<double> phi_tilde;
  // Projector function: p_i(r) * Y_lm (localized within r_c)
  std::vector<double> projector;
  // Kinetic energy eigenvalue for this partial wave (Ha)
  double eigenvalue = 0.0;
};

// Per-atom PAW dataset.
struct PAWAtomData {
  std::string element = "";
  int Z_valence = 0;          // valence charge
  int Z_core = 0;             // frozen core charge
  double r_c = 0.0;           // augmentation sphere radius (Bohr)
  std::vector<double> r_grid; // radial grid
  std::vector<PAWChannel> channels;

  // D_ij matrix: coupling between projectors (n_proj x n_proj, Ha)
  // Stored flat in channel order. Diagonal entries are the eigenvalues;
  // off-diagonal are typically zero for single-projector-per-l.
  std::vector<std::vector<double>> Dij;

  // Core density (frozen core) on r_grid, for XC and Hartree.
  std::vector<double> core_density;

  // Compensation charge multipole moments (for multipole compensation
  // in periodic systems; zero for isolated molecules).
  std::vector<double> compensation_moments;
};

// Create a synthetic PAW dataset for H (Z=1, no core).
// This is a minimal PAW with one s-channel: the projector is a Gaussian
// centered at the atom, the partial waves are simple exponentials, and
// D_ij = [eigenvalue]. The correction should be small since H has no core.
PAWAtomData MakeSimplePAWH() {
  PAWAtomData paw;
  paw.element = "H";
  paw.Z_valence = 1;
  paw.Z_core = 0;
  paw.r_c = 1.2;  // typical PAW augmentation radius for H

  // Radial grid: 0.001 to 3.0 Bohr, 300 points (log-spaced).
  const int n_r = 300;
  paw.r_grid.resize(n_r);
  for (int i = 0; i < n_r; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(n_r - 1);
    // Log-linear grid: small r well-sampled.
    paw.r_grid[i] = 0.001 * std::exp(t * std::log(3000.0));
  }

  PAWChannel ch;
  ch.l = 0;
  ch.eigenvalue = -0.5;  // H 1s energy (Ha)

  // phi (all-electron): exact H 1s = 2 * exp(-r)
  // phi_tilde (pseudo): smooth version = 2 * exp(-r) * (1 - exp(-r^2/rc^2))
  // projector: Gaussian-like function localized within r_c
  ch.phi.resize(n_r);
  ch.phi_tilde.resize(n_r);
  ch.projector.resize(n_r);
  for (int i = 0; i < n_r; ++i) {
    double r = paw.r_grid[i];
    ch.phi[i] = 2.0 * std::exp(-r);
    double smooth = 1.0 - std::exp(-r * r / (paw.r_c * paw.r_c));
    ch.phi_tilde[i] = ch.phi[i] * smooth;
    // Projector: normalized Gaussian truncated at r_c
    double sigma = paw.r_c * 0.5;
    ch.projector[i] = std::exp(-r * r / (2.0 * sigma * sigma)) /
                     (sigma * std::sqrt(std::sqrt(M_PI)));
  }
  paw.channels.push_back(ch);

  // D_ij: 1x1 matrix with the eigenvalue.
  paw.Dij.assign(1, std::vector<double>(1, ch.eigenvalue));

  // No core density for H.
  paw.core_density.assign(n_r, 0.0);
  paw.compensation_moments = {0.0};

  return paw;
}

// Create a synthetic PAW dataset for He (Z=2, no core for simplicity).
// Similar to H but with Z=2 effective charge.
PAWAtomData MakeSimplePAWHe() {
  PAWAtomData paw;
  paw.element = "He";
  paw.Z_valence = 2;
  paw.Z_core = 0;
  paw.r_c = 1.5;

  const int n_r = 300;
  paw.r_grid.resize(n_r);
  for (int i = 0; i < n_r; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(n_r - 1);
    paw.r_grid[i] = 0.001 * std::exp(t * std::log(3000.0));
  }

  PAWChannel ch;
  ch.l = 0;
  ch.eigenvalue = -2.0;  // He 1s approximate energy (Ha)

  // phi: approximate He 1s = 2 * Z_eff * exp(-Z_eff * r), Z_eff ~ 1.7
  const double Z_eff = 1.7;
  ch.phi.resize(n_r);
  ch.phi_tilde.resize(n_r);
  ch.projector.resize(n_r);
  for (int i = 0; i < n_r; ++i) {
    double r = paw.r_grid[i];
    ch.phi[i] = 2.0 * Z_eff * std::exp(-Z_eff * r);
    double smooth = 1.0 - std::exp(-r * r / (paw.r_c * paw.r_c));
    ch.phi_tilde[i] = ch.phi[i] * smooth;
    double sigma = paw.r_c * 0.5;
    ch.projector[i] = std::exp(-r * r / (2.0 * sigma * sigma)) /
                     (sigma * std::sqrt(std::sqrt(M_PI)));
  }
  paw.channels.push_back(ch);

  paw.Dij.assign(1, std::vector<double>(1, ch.eigenvalue));
  paw.core_density.assign(n_r, 0.0);
  paw.compensation_moments = {0.0};

  return paw;
}

// Compute the PAW occupancy matrix D_ij = sum_n f_n <psi_tilde_n|p_i><p_j|psi_tilde_n>
// In the NAO basis, <psi_tilde_n|p_i> = sum_mu C_nmu <phi_mu|p_i>, so:
//   D_ij = sum_mu,nu P_munu <phi_mu|p_i><p_j|phi_nu>
// where P is the density matrix and <phi_mu|p_i> are three-center integrals.
//
// For the on-site correction, we compute:
//   H_PAW[i,j] = D_ij * (h_ij^AE - h_ij^PS)
// where h_ij^AE and h_ij^PS are the all-electron and pseudo matrix elements
// of the Hamiltonian within the augmentation sphere.
//
// The total PAW correction to the energy is:
//   E_PAW = sum_a sum_ij D_ij^a * (h_ij^AE,a - h_ij^PS,a)
//         + core-valence XC + double-counting corrections

// Compute projector overlaps <phi_mu | p_i> for all basis functions mu and
// projectors i on a 3D grid. This is a three-center integral evaluated by
// grid integration (same as the KB projector path in NaoDriver).
struct PAWProjectorOverlaps {
  // overlaps[proj_idx] = vector of n_basis overlaps
  std::vector<std::vector<double>> overlaps;
  // n_projectors per atom
  std::size_t n_projectors = 0;
};

// Compute the PAW on-site energy correction for a single atom.
// E_PAW = sum_ij D_ij * (h_ij^AE - h_ij^PS)
// For the synthetic datasets, h_ij^AE - h_ij^PS is approximated by the
// difference between the all-electron and pseudo partial-wave kinetic
// energies, which for our simple Gaussians is proportional to eigenvalue
// times the overlap of phi and phi_tilde.
inline double ComputeOnSiteEnergy(const PAWAtomData& paw,
                                  const std::vector<double>& D_ij_flat) {
  double energy = 0.0;
  std::size_t n_proj = paw.Dij.size();
  if (n_proj == 0) return 0.0;

  // The on-site correction: sum_ij D_ij * Delta_h_ij
  // where Delta_h_ij = h_ij^AE - h_ij^PS
  // For the synthetic case, Delta_h is the difference between the
  // all-electron and pseudo eigenvalues weighted by the phi-phi_tilde overlap.
  for (std::size_t i = 0; i < n_proj && i < D_ij_flat.size(); ++i) {
    for (std::size_t j = 0; j < n_proj && j < D_ij_flat.size(); ++j) {
      double delta_h = 0.0;
      if (i == j && i < paw.channels.size()) {
        // Compute overlap <phi_i | phi_tilde_i> - 1 (deviation from identity)
        double overlap = 0.0;
        const auto& phi = paw.channels[i].phi;
        const auto& phi_t = paw.channels[i].phi_tilde;
        const auto& rg = paw.r_grid;
        for (std::size_t k = 0; k + 1 < rg.size() && k < phi.size() && k < phi_t.size(); ++k) {
          double dr = rg[k + 1] - rg[k];
          overlap += 0.5 * (phi[k] * phi_t[k] * rg[k] * rg[k] +
                            phi[k + 1] * phi_t[k + 1] * rg[k + 1] * rg[k + 1]) * dr;
        }
        // Delta_h = eigenvalue * (1 - overlap^2): the correction vanishes
        // when phi = phi_tilde (no correction needed) and is proportional
        // to the eigenvalue otherwise.
        delta_h = paw.channels[i].eigenvalue * (1.0 - overlap * overlap);
      }
      energy += D_ij_flat[i * n_proj + j] * delta_h;
    }
  }
  return energy;
}

// Compute the PAW density correction on the radial grid.
// Delta_n(r) = sum_ij D_ij * (phi_i(r)*phi_j(r) - phi_tilde_i(r)*phi_tilde_j(r))
// This is the difference between all-electron and pseudo density within
// the augmentation sphere. Outside r_c, phi = phi_tilde, so Delta_n = 0.
inline std::vector<double> ComputeDensityCorrection(
    const PAWAtomData& paw,
    const std::vector<double>& D_ij_flat) {
  std::vector<double> delta_n(paw.r_grid.size(), 0.0);
  std::size_t n_proj = paw.Dij.size();
  if (n_proj == 0) return delta_n;

  for (std::size_t i = 0; i < n_proj && i < paw.channels.size(); ++i) {
    for (std::size_t j = 0; j < n_proj && j < paw.channels.size(); ++j) {
      double dij = (i * n_proj + j < D_ij_flat.size()) ?
          D_ij_flat[i * n_proj + j] : 0.0;
      if (dij == 0.0) continue;
      for (std::size_t k = 0; k < paw.r_grid.size(); ++k) {
        double phi_phi = paw.channels[i].phi[k] * paw.channels[j].phi[k];
        double pt_pt = paw.channels[i].phi_tilde[k] * paw.channels[j].phi_tilde[k];
        delta_n[k] += dij * (phi_phi - pt_pt);
      }
    }
  }
  return delta_n;
}

}  // namespace tides::basis::paw
