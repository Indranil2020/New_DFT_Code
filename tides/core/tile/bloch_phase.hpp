#pragma once

// T2.8: Bloch-phase (complex) tiles for periodic R0/R1.
//
// For periodic systems, the Hamiltonian and overlap matrices in the
// k-space representation are:
//   H(k) = sum_R H(R) * exp(i * k . R)
//   S(k) = sum_R S(R) * exp(i * k . R)
// where R are lattice vectors and H(R), S(R) are the real-space tile
// matrices for each periodic image.
//
// This module provides:
//   - ComplexTile: a dense complex (double) tile for k-space operations
//   - BlochPhaseMap: applies Bloch phase factors to real-space tiles
//     and assembles the k-space H(k) and S(k) as dense complex matrices
//   - The output complex matrices can be fed to the R0/R1 eigensolvers
//     (which need a complex generalized eigensolver path)

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/status.hpp"

namespace tides::tile {

// A dense complex matrix (row-major, n x n) using std::complex<double>.
struct ComplexMatrix {
  std::size_t n = 0;
  std::vector<std::complex<double>> data;  // n*n, row-major

  ComplexMatrix() = default;
  explicit ComplexMatrix(std::size_t n_)
      : n(n_), data(n_ * n_, std::complex<double>(0.0, 0.0)) {}

  std::complex<double>& operator()(std::size_t i, std::size_t j) {
    return data[i * n + j];
  }
  const std::complex<double>& operator()(std::size_t i, std::size_t j) const {
    return data[i * n + j];
  }
};

// Bloch phase factor: exp(i * k . R)
inline std::complex<double> BlochPhase(const std::array<double, 3>& k,
                                       const std::array<double, 3>& R) {
  double phase = k[0] * R[0] + k[1] * R[1] + k[2] * R[2];
  return std::complex<double>(std::cos(phase), std::sin(phase));
}

// A real-space tile matrix for a single periodic image R.
// Each image has a dense n x n real matrix (Hamiltonian or overlap).
struct PeriodicImage {
  std::array<double, 3> R;       // lattice vector
  std::vector<double> matrix;    // n x n, row-major, real
};

// Assemble the k-space complex matrix from real-space periodic images.
//   M(k) = sum_R M(R) * exp(i * k . R)
//
// @param n       Matrix dimension
// @param k       k-point in fractional coordinates (or Cartesian — must match R)
// @param images  Real-space periodic images (R vectors + matrices)
// @return        Complex n x n matrix M(k)
[[nodiscard]] inline ComplexMatrix AssembleKSpace(
    std::size_t n, const std::array<double, 3>& k,
    const std::vector<PeriodicImage>& images) {
  ComplexMatrix result(n);
  for (const auto& img : images) {
    if (img.matrix.size() != n * n) continue;
    std::complex<double> phase = BlochPhase(k, img.R);
    for (std::size_t i = 0; i < n * n; ++i)
      result.data[i] += phase * img.matrix[i];
  }
  return result;
}

// Extract the Hermitian part: M_h = 0.5 * (M + M^dagger).
// For valid k-space Hamiltonians, M(k) should already be Hermitian
// (up to numerical noise). This enforces it.
[[nodiscard]] inline ComplexMatrix HermitianPart(const ComplexMatrix& M) {
  ComplexMatrix result(M.n);
  for (std::size_t i = 0; i < M.n; ++i)
    for (std::size_t j = 0; j < M.n; ++j)
      result(i, j) = 0.5 * (M(i, j) + std::conj(M(j, i)));
  return result;
}

// Complex matrix-matrix product: C = A * B (both n x n, complex).
[[nodiscard]] inline ComplexMatrix ComplexMatMul(std::size_t n,
                                                  const ComplexMatrix& A,
                                                  const ComplexMatrix& B) {
  ComplexMatrix C(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      std::complex<double> s(0.0, 0.0);
      for (std::size_t k = 0; k < n; ++k)
        s += A(i, k) * B(k, j);
      C(i, j) = s;
    }
  return C;
}

// Trace of complex matrix: tr(M) = sum_i M_ii.
[[nodiscard]] inline std::complex<double> ComplexTrace(
    const ComplexMatrix& M) {
  std::complex<double> tr(0.0, 0.0);
  for (std::size_t i = 0; i < M.n; ++i) tr += M(i, i);
  return tr;
}

// Frobenius norm of complex matrix.
[[nodiscard]] inline double ComplexFrobeniusNorm(const ComplexMatrix& M) {
  double s = 0.0;
  for (std::size_t i = 0; i < M.data.size(); ++i)
    s += std::norm(M.data[i]);
  return std::sqrt(s);
}

// Generate a simple k-point mesh (Monkhorst-Pack grid).
// Returns a list of k-points in fractional coordinates.
[[nodiscard]] inline std::vector<std::array<double, 3>>
MonkhorstPackGrid(std::size_t nk) {
  std::vector<std::array<double, 3>> kpts;
  if (nk == 0) return kpts;
  for (std::size_t ix = 0; ix < nk; ++ix)
    for (std::size_t iy = 0; iy < nk; ++iy)
      for (std::size_t iz = 0; iz < nk; ++iz) {
        double kx = (2.0 * ix + 1.0) / (2.0 * nk) - 0.5;
        double ky = (2.0 * iy + 1.0) / (2.0 * nk) - 0.5;
        double kz = (2.0 * iz + 1.0) / (2.0 * nk) - 0.5;
        kpts.push_back({kx, ky, kz});
      }
  return kpts;
}

}  // namespace tides::tile
