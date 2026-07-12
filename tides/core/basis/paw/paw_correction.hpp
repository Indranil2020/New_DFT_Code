#pragma once

// PAW on-site correction to the SCF Hamiltonian and energy.
//
// This module computes the PAW correction that modifies the Hamiltonian
// and density within the augmentation sphere of each atom. The correction
// is a small dense (n_proj x n_proj) operation per atom — naturally
// compatible with the tile substrate (diagonal-block correction).
//
// Integration point: NaoDriver::build_H adds the PAW correction to H after
// assembling the standard H = T + V_ext + V_H + V_xc (+ V_nl).
//
// The PAW correction to H is:
//   H_PAW = sum_a sum_ij h_ij^a |p_i^a><p_j^a|
// where h_ij^a = (h_ij^AE - h_ij^PS) is the difference between all-electron
// and pseudo Hamiltonian matrix elements within the augmentation sphere.
//
// The correction to the energy is:
//   E_PAW = sum_a sum_ij D_ij^a * h_ij^a
// where D_ij = sum_munu P_munu <phi_mu|p_i><p_j|phi_nu> is the occupancy
// matrix computed from the density matrix P.

#include <algorithm>
#include <cmath>
#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/paw/paw_dataset.hpp"

namespace tides::basis::paw {

// Compute the PAW on-site Hamiltonian correction matrix.
// This is a (n_basis x n_basis) matrix added to H:
//   H_PAW[mu,nu] = sum_a sum_ij sum_m <phi_mu|p_i^a,m> * h_ij^a * <p_j^a,m|phi_nu>
// where the sum over m runs from -l to +l for each channel.
//
// Parameters:
//   n_basis: number of NAO basis functions
//   atoms: atom positions (flat 3*n_atoms)
//   paw_data: per-atom PAW datasets
//   projector_overlaps: precomputed <phi_mu | p_i^a,m> for each atom
//   orbitals: NAO basis functions on the grid (for computing overlaps if needed)
//   grid: grid info
//   n0, n1, n2: grid dimensions
//   dv: grid cell volume
class PAWCorrection {
 public:
  // Compute the on-site H correction for all atoms.
  // Returns a (n_basis x n_basis) matrix to add to H.
  static std::vector<double> ComputeOnSiteH(
      std::size_t n_basis,
      const std::vector<double>& atom_positions,
      const std::vector<PAWAtomData>& paw_data,
      const std::vector<std::vector<std::vector<double>>>& projector_overlaps,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& r_grid_1d,
      int n0, int n1, int n2, double dv, double grid_h) {
    std::vector<double> H_paw(n_basis * n_basis, 0.0);

    for (std::size_t a = 0; a < paw_data.size(); ++a) {
      const auto& paw = paw_data[a];
      std::size_t n_proj = paw.Dij.size();
      if (n_proj == 0) continue;

      // For each projector pair (i, j) and each angular momentum m:
      // H_PAW[mu, nu] += sum_ij h_ij * sum_m <mu|p_i,m><p_j,m|nu>
      //
      // h_ij = D_ij[ae] - D_ij[ps] (difference of AE and PS Hamiltonian)
      // For our synthetic data, h_ij = eigenvalue * (1 - overlap^2) for i==j.
      std::vector<double> h_onsite(n_proj * n_proj, 0.0);
      for (std::size_t i = 0; i < n_proj; ++i) {
        for (std::size_t j = 0; j < n_proj; ++j) {
          if (i == j && i < paw.channels.size()) {
            double overlap = 0.0;
            const auto& phi = paw.channels[i].phi;
            const auto& phi_t = paw.channels[i].phi_tilde;
            const auto& rg = paw.r_grid;
            for (std::size_t k = 0; k + 1 < rg.size() && k < phi.size() && k < phi_t.size(); ++k) {
              double dr = rg[k + 1] - rg[k];
              overlap += 0.5 * (phi[k] * phi_t[k] * rg[k] * rg[k] +
                               phi[k + 1] * phi_t[k + 1] * rg[k + 1] * rg[k + 1]) * dr;
            }
            h_onsite[i * n_proj + j] = paw.channels[i].eigenvalue * (1.0 - overlap * overlap);
          }
        }
      }

      // Compute projector-basis overlaps on the grid if not precomputed.
      // <phi_mu | p_i, m> = integral phi_mu(r) * p_i(r-R_a) * Y_lm(r-R_a) dr
      const auto& overlaps_a = (a < projector_overlaps.size()) ?
          projector_overlaps[a] : std::vector<std::vector<double>>();

      if (!overlaps_a.empty()) {
        // Use precomputed overlaps.
        for (std::size_t i = 0; i < n_proj; ++i) {
          for (std::size_t j = 0; j < n_proj; ++j) {
            double h_ij = h_onsite[i * n_proj + j];
            if (h_ij == 0.0) continue;
            // H_PAW[mu, nu] += h_ij * sum_m <mu|p_i,m> * <p_j,m|nu>
            // For single-projector-per-l (n_proj = n_channels), m is implied.
            const auto& ov_i = (i < overlaps_a.size()) ? overlaps_a[i] : std::vector<double>();
            const auto& ov_j = (j < overlaps_a.size()) ? overlaps_a[j] : std::vector<double>();
            if (ov_i.empty() || ov_j.empty()) continue;
            for (std::size_t mu = 0; mu < n_basis; ++mu) {
              for (std::size_t nu = 0; nu < n_basis; ++nu) {
                H_paw[mu * n_basis + nu] += h_ij * ov_i[mu] * ov_j[nu];
              }
            }
          }
        }
      } else {
        // Compute overlaps on-the-fly on the grid.
        // For simplicity, evaluate projector * Y_lm on the grid and integrate.
        std::size_t np = static_cast<std::size_t>(n0) * n1 * n2;
        double ax = atom_positions[3 * a];
        double ay = atom_positions[3 * a + 1];
        double az = atom_positions[3 * a + 2];

        for (std::size_t ch_idx = 0; ch_idx < paw.channels.size(); ++ch_idx) {
          const auto& ch = paw.channels[ch_idx];
          int l = ch.l;
          int n_m = 2 * l + 1;
          const auto& rg = paw.r_grid;
          if (rg.empty()) continue;

          // Evaluate projector(r) * Y_lm for each m on the grid.
          std::vector<std::vector<double>> proj_grid(n_m, std::vector<double>(np, 0.0));
          for (int ix = 0; ix < n0; ++ix) {
            for (int iy = 0; iy < n1; ++iy) {
              for (int iz = 0; iz < n2; ++iz) {
                std::size_t g = static_cast<std::size_t>(ix) * n1 * n2 +
                                static_cast<std::size_t>(iy) * n2 + iz;
                double x = static_cast<double>(ix) * grid_h;
                double y = static_cast<double>(iy) * grid_h;
                double z = static_cast<double>(iz) * grid_h;
                double dx = x - ax, dy = y - ay, dz = z - az;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12 || r > paw.r_c) continue;

                // Interpolate projector on the PAW radial grid.
                double beta_r = 0.0;
                auto it = std::upper_bound(rg.begin(), rg.end(), r);
                if (it != rg.begin() && it != rg.end()) {
                  std::size_t k = static_cast<std::size_t>(it - rg.begin() - 1);
                  double t = (r - rg[k]) / (rg[k + 1] - rg[k]);
                  beta_r = (1.0 - t) * ch.projector[k] + t * ch.projector[k + 1];
                }

                double theta = std::acos(dz / std::max(r, 1e-15));
                double phi_ang = std::atan2(dy, dx);
                // Real spherical harmonics.
                for (int m = -l; m <= l; ++m) {
                  double angular = (l == 0) ? (1.0 / std::sqrt(4.0 * M_PI)) :
                      RealSphericalHarmonic(l, m, theta, phi_ang);
                  proj_grid[m + l][g] = beta_r * angular;
                }
              }
            }
          }

          // Compute <phi_mu | p_m> for each basis function and m.
          std::size_t proj_global_idx = ch_idx;  // index within this atom's projectors
          for (int m_idx = 0; m_idx < n_m; ++m_idx) {
            std::vector<double> ov(n_basis, 0.0);
            for (std::size_t mu = 0; mu < n_basis; ++mu) {
              double s = 0.0;
              for (std::size_t g = 0; g < np; ++g)
                s += orbitals[mu][g] * proj_grid[m_idx][g] * dv;
              ov[mu] = s;
            }

            // Add h_ij * <mu|p_i,m> * <p_j,m|nu> to H_PAW.
            for (std::size_t j = 0; j < paw.channels.size(); ++j) {
              double h_ij = h_onsite[proj_global_idx * n_proj + j];
              if (h_ij == 0.0) continue;
              // For the same m: need <p_j,m|nu> as well.
              // For simplicity with single-projector: use the same overlap.
              // (This is exact when each channel has one projector and l matches.)
              for (std::size_t mu = 0; mu < n_basis; ++mu) {
                for (std::size_t nu = 0; nu < n_basis; ++nu) {
                  H_paw[mu * n_basis + nu] += h_ij * ov[mu] * ov[nu];
                }
              }
            }
          }
        }
      }
    }

    return H_paw;
  }

  // Compute the PAW energy correction.
  // E_PAW = sum_a sum_ij D_ij^a * h_ij^a
  // where D_ij = sum_munu P_munu <phi_mu|p_i><p_j|phi_nu>
  static double ComputeEnergyCorrection(
      std::size_t n_basis,
      const std::vector<double>& P,
      const std::vector<PAWAtomData>& paw_data,
      const std::vector<std::vector<std::vector<double>>>& projector_overlaps) {
    double energy = 0.0;
    for (std::size_t a = 0; a < paw_data.size(); ++a) {
      const auto& paw = paw_data[a];
      std::size_t n_proj = paw.Dij.size();
      if (n_proj == 0) continue;

      const auto& overlaps_a = (a < projector_overlaps.size()) ?
          projector_overlaps[a] : std::vector<std::vector<double>>();
      if (overlaps_a.empty()) continue;

      // D_ij = sum_munu P_munu * <mu|p_i> * <p_j|nu>
      std::vector<double> D_ij(n_proj * n_proj, 0.0);
      for (std::size_t i = 0; i < n_proj; ++i) {
        const auto& ov_i = (i < overlaps_a.size()) ? overlaps_a[i] : std::vector<double>();
        if (ov_i.empty()) continue;
        for (std::size_t j = 0; j < n_proj; ++j) {
          const auto& ov_j = (j < overlaps_a.size()) ? overlaps_a[j] : std::vector<double>();
          if (ov_j.empty()) continue;
          double dij = 0.0;
          for (std::size_t mu = 0; mu < n_basis; ++mu) {
            for (std::size_t nu = 0; nu < n_basis; ++nu) {
              dij += P[mu * n_basis + nu] * ov_i[mu] * ov_j[nu];
            }
          }
          D_ij[i * n_proj + j] = dij;
        }
      }

      // E_PAW += sum_ij D_ij * h_ij
      for (std::size_t i = 0; i < n_proj; ++i) {
        for (std::size_t j = 0; j < n_proj; ++j) {
          if (i == j && i < paw.channels.size()) {
            double overlap = 0.0;
            const auto& phi = paw.channels[i].phi;
            const auto& phi_t = paw.channels[i].phi_tilde;
            const auto& rg = paw.r_grid;
            for (std::size_t k = 0; k + 1 < rg.size() && k < phi.size() && k < phi_t.size(); ++k) {
              double dr = rg[k + 1] - rg[k];
              overlap += 0.5 * (phi[k] * phi_t[k] * rg[k] * rg[k] +
                               phi[k + 1] * phi_t[k + 1] * rg[k + 1] * rg[k + 1]) * dr;
            }
            double h_ij = paw.channels[i].eigenvalue * (1.0 - overlap * overlap);
            energy += D_ij[i * n_proj + j] * h_ij;
          }
        }
      }
    }
    return energy;
  }

 private:
  // Real spherical harmonic evaluation (same as in two_center_integrals.hpp).
  static double RealSphericalHarmonic(int l, int m, double theta, double phi) {
    if (l == 0) return 1.0 / std::sqrt(4.0 * M_PI);
    if (l == 1) {
      if (m == -1) return std::sqrt(3.0 / (4.0 * M_PI)) * std::sin(theta) * std::sin(phi);
      if (m == 0) return std::sqrt(3.0 / (4.0 * M_PI)) * std::cos(theta);
      if (m == 1) return std::sqrt(3.0 / (4.0 * M_PI)) * std::sin(theta) * std::cos(phi);
    }
    if (l == 2) {
      double st = std::sin(theta), ct = std::cos(theta);
      if (m == -2) return std::sqrt(15.0 / (16.0 * M_PI)) * st * st * std::sin(2.0 * phi);
      if (m == -1) return std::sqrt(15.0 / (4.0 * M_PI)) * st * ct * std::sin(phi);
      if (m == 0) return std::sqrt(5.0 / (16.0 * M_PI)) * (3.0 * ct * ct - 1.0);
      if (m == 1) return std::sqrt(15.0 / (4.0 * M_PI)) * st * ct * std::cos(phi);
      if (m == 2) return std::sqrt(15.0 / (16.0 * M_PI)) * st * st * std::cos(2.0 * phi);
    }
    return 0.0;
  }
};

}  // namespace tides::basis::paw
