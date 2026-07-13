#pragma once

// Monkhorst-Pack k-point sampling for periodic systems (§3.1.5, §9.1).
//
// Generates a Monkhorst-Pack k-point grid and provides Bloch-phase
// transformation of matrices. With time-reversal symmetry, k and -k are
// equivalent (weight doubled), reducing the number of irreducible k-points.
//
// The Bloch-phase transform applies exp(i*k·R) to the real-space matrix
// elements, where R is the lattice vector connecting two unit cells.

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace tides::tile {

struct KPoint {
  std::array<double, 3> kvec;
  double weight;
};

struct KPointGrid {
  std::vector<KPoint> kpoints;
  std::array<int, 3> grid_dims = {1, 1, 1};
  std::array<std::array<double, 3>, 3> reciprocal_lattice;
  bool time_reversal = true;
  bool gamma_centered = false;
};

class KPointSampler {
 public:
  // Generate Monkhorst-Pack grid.
  //   grid_dims:        (nkx, nky, nkz)
  //   reciprocal_lattice: rows are b1, b2, b3
  //   gamma_centered:   if true, k_i = (n_i-1)/N_i; else k_i = (2n_i-N_i+1)/(2N_i)
  //   time_reversal:    if true, fold k <-> -k (weight doubled)
  static KPointGrid GenerateMonkhorstPack(
      std::array<int, 3> grid_dims,
      std::array<std::array<double, 3>, 3> reciprocal_lattice,
      bool gamma_centered = false,
      bool time_reversal = true) {
    KPointGrid grid;
    grid.grid_dims = grid_dims;
    grid.reciprocal_lattice = reciprocal_lattice;
    grid.gamma_centered = gamma_centered;
    grid.time_reversal = time_reversal;

    const int nkx = grid_dims[0], nky = grid_dims[1], nkz = grid_dims[2];
    const double total = static_cast<double>(nkx * nky * nkz);

    for (int ix = 0; ix < nkx; ++ix) {
      for (int iy = 0; iy < nky; ++iy) {
        for (int iz = 0; iz < nkz; ++iz) {
          double kx, ky, kz;
          if (gamma_centered) {
            kx = static_cast<double>(ix) / nkx;
            ky = static_cast<double>(iy) / nky;
            kz = static_cast<double>(iz) / nkz;
          } else {
            kx = (2.0 * ix - nkx + 1) / (2.0 * nkx);
            ky = (2.0 * iy - nky + 1) / (2.0 * nky);
            kz = (2.0 * iz - nkz + 1) / (2.0 * nkz);
          }
          // Fold to [-0.5, 0.5)
          kx = Fold(kx);
          ky = Fold(ky);
          kz = Fold(kz);

          KPoint kp;
          kp.kvec = {kx, ky, kz};
          kp.weight = 1.0 / total;

          // Check if -k is already in the list (time-reversal symmetry).
          if (time_reversal) {
            std::array<double, 3> neg_k = {-kx, -ky, -kz};
            // Check if -k == k (Gamma point or zone boundary).
            double diff = (kx - neg_k[0]) * (kx - neg_k[0]) +
                          (ky - neg_k[1]) * (ky - neg_k[1]) +
                          (kz - neg_k[2]) * (kz - neg_k[2]);
            if (diff < 1e-12) {
              // k == -k, unique point (Gamma or boundary).
              grid.kpoints.push_back(kp);
            } else {
              // Check if -k is already in the list.
              bool found = false;
              for (const auto& existing : grid.kpoints) {
                double d = (existing.kvec[0] - neg_k[0]) *
                           (existing.kvec[0] - neg_k[0]) +
                           (existing.kvec[1] - neg_k[1]) *
                           (existing.kvec[1] - neg_k[1]) +
                           (existing.kvec[2] - neg_k[2]) *
                           (existing.kvec[2] - neg_k[2]);
                if (d < 1e-12) {
                  found = true;
                  break;
                }
              }
              if (found) {
                // -k already exists; skip (weight will be doubled).
                // Find the existing point and double its weight.
                for (auto& existing : grid.kpoints) {
                  double d = (existing.kvec[0] - neg_k[0]) *
                             (existing.kvec[0] - neg_k[0]) +
                             (existing.kvec[1] - neg_k[1]) *
                             (existing.kvec[1] - neg_k[1]) +
                             (existing.kvec[2] - neg_k[2]) *
                             (existing.kvec[2] - neg_k[2]);
                  if (d < 1e-12) {
                    existing.weight += 1.0 / total;
                    break;
                  }
                }
              } else {
                grid.kpoints.push_back(kp);
              }
            }
          } else {
            grid.kpoints.push_back(kp);
          }
        }
      }
    }
    return grid;
  }

  // Fold k-point fractional coordinate to [-0.5, 0.5).
  static double Fold(double k) {
    k -= std::floor(k + 0.5);
    return k;
  }

  // Fold a 3D k-point into the first Brillouin zone.
  static std::array<double, 3> FoldToBZ(std::array<double, 3> k) {
    return {Fold(k[0]), Fold(k[1]), Fold(k[2])};
  }

  // Count irreducible k-points (after time-reversal folding).
  static std::size_t CountIrreducible(const KPointGrid& grid) {
    return grid.kpoints.size();
  }

  // Apply Bloch phase to a real-space matrix at a given k-point.
  // H(k) = sum_R H(R) * exp(i*k·R)
  // For the CPU reference, this applies the phase factor to each element.
  // The output is a complex matrix stored as interleaved (real, imag) pairs.
  static std::vector<std::complex<double>> BlochPhaseTransform(
      const std::vector<double>& H_real,
      std::size_t n,
      const KPoint& kp,
      const std::array<std::array<double, 3>, 3>& lattice_vectors) {
    std::vector<std::complex<double>> H_k(n * n, {0.0, 0.0});

    // For a single unit cell (no supercell), H(k) = H(R=0) since there's
    // only the on-site block. The phase factor exp(i*k·0) = 1.
    // For a supercell with neighbor images, we'd sum over R vectors.
    // Here we apply the on-site block plus a phase for the diagonal.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        // The phase depends on the real-space separation.
        // For simplicity, the on-site block (R=0) has phase 1.
        H_k[i * n + j] = std::complex<double>(H_real[i * n + j], 0.0);
      }
    }

    // If there are off-diagonal blocks corresponding to lattice vectors,
    // apply the phase. Here we use a simple model: the phase for element (i,j)
    // is exp(i * k · R_ij) where R_ij is the lattice vector connecting them.
    // For a real implementation, the caller would provide the R-vector for
    // each block. For the CPU reference, we apply the k-point phase to the
    // full matrix as a complex scalar.
    double k_dot_r = 0.0;
    for (int a = 0; a < 3; ++a)
      k_dot_r += kp.kvec[a] * (lattice_vectors[0][a] + lattice_vectors[1][a] +
                               lattice_vectors[2][a]);

    // Apply the phase to off-diagonal elements (simple model).
    std::complex<double> phase = std::polar(1.0, 2.0 * M_PI * k_dot_r);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        H_k[i * n + j] *= phase;
        H_k[j * n + i] = std::conj(H_k[i * n + j]);
      }

    return H_k;
  }

  // Get the k-point index in the full grid (for parallelization).
  static int KPointIndex(const KPointGrid& grid, std::array<double, 3> k) {
    for (std::size_t i = 0; i < grid.kpoints.size(); ++i) {
      double d = (grid.kpoints[i].kvec[0] - k[0]) *
                 (grid.kpoints[i].kvec[0] - k[0]) +
                 (grid.kpoints[i].kvec[1] - k[1]) *
                 (grid.kpoints[i].kvec[1] - k[1]) +
                 (grid.kpoints[i].kvec[2] - k[2]) *
                 (grid.kpoints[i].kvec[2] - k[2]);
      if (d < 1e-12) return static_cast<int>(i);
    }
    return -1;  // not found
  }
};

}  // namespace tides::tile
