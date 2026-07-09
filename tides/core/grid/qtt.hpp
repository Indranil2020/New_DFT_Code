#pragma once

// T3.7/T3.8: QTT (Quantized Tensor Train) prototypes for density compression
// and Poisson solving.
//
// QTT quantizes each grid index into binary bits, reshaping a 2^n x 2^n x 2^n
// tensor into a 3n-dimensional tensor, then decomposes it into a Tensor Train.
// Storage: O(3n * r^2) vs O(2^{3n}), where r is the TT rank.
//
// For smooth functions (Gaussians, orbitals), QTT ranks are small (r ~ 10-30),
// giving exponential compression. The Poisson equation can be solved in QTT
// format using TT matrix operations.
//
// This is a research prototype (gate R-1 at M30).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace tides::grid {

// A single TT core: a 3D array of shape (r_left, mode, r_right).
struct TTCORE {
  std::size_t r_left = 0;
  std::size_t mode = 0;
  std::size_t r_right = 0;
  std::vector<double> data;  // size r_left * mode * r_right, row-major

  double& at(std::size_t i, std::size_t j, std::size_t k) {
    return data[i * mode * r_right + j * r_right + k];
  }
  const double& at(std::size_t i, std::size_t j, std::size_t k) const {
    return data[i * mode * r_right + j * r_right + k];
  }
};

// Tensor Train representation of a multi-dimensional array.
struct TensorTrain {
  std::vector<TTCORE> cores;
  std::vector<std::size_t> modes;  // mode sizes per dimension

  [[nodiscard]] std::size_t ndim() const { return modes.size(); }

  // Total parameter count (storage cost).
  [[nodiscard]] std::size_t param_count() const {
    std::size_t total = 0;
    for (const auto& c : cores) total += c.r_left * c.mode * c.r_right;
    return total;
  }

  // Max TT rank.
  [[nodiscard]] std::size_t max_rank() const {
    std::size_t r = 0;
    for (const auto& c : cores) r = std::max(r, std::max(c.r_left, c.r_right));
    return r;
  }
};

// SVD-based TT decomposition and operations.
class QTT {
 public:
  // Decompose a d-dimensional tensor into TT format via sequential SVD.
  //   shape: mode sizes per dimension
  //   data: flat array, row-major (last index fastest)
  //   eps: relative accuracy for rank truncation
  static TensorTrain Decompose(const std::vector<double>& data,
                                const std::vector<std::size_t>& shape,
                                double eps = 1e-12) {
    const std::size_t d = shape.size();
    TensorTrain tt;
    tt.modes = shape;

    // Reshape into matrix [shape[0] x rest], SVD, then iterate.
    std::size_t total = 1;
    for (auto s : shape) total *= s;
    if (data.size() != total) return tt;

    // Working matrix: rows = current mode, cols = remaining.
    std::size_t rows = shape[0];
    std::size_t cols = total / rows;
    std::vector<double> mat(data.begin(), data.end());

    std::size_t r_left = 1;

    for (std::size_t dim = 0; dim < d; ++dim) {
      // SVD: mat [rows x cols] -> U [rows x k] * S [k] * V^T [k x cols]
      auto [U, S, Vt, k] = SVDTruncated(mat, rows, cols, eps);

      // Truncate rank.
      std::size_t r = k;
      // Keep core: shape (r_left, shape[dim], r)
      TTCORE core;
      core.r_left = r_left;
      core.mode = shape[dim];
      core.r_right = r;
      core.data.resize(r_left * shape[dim] * r, 0.0);

      // Fill core from U: U is [rows x r], rows = r_left * shape[dim]
      for (std::size_t i = 0; i < r_left; ++i)
        for (std::size_t j = 0; j < shape[dim]; ++j)
          for (std::size_t kk = 0; kk < r; ++kk)
            core.at(i, j, kk) = U[(i * shape[dim] + j) * r + kk];

      // For the last dimension, multiply core by S (and Vt which is [1x1]=1).
      if (dim + 1 == d) {
        for (std::size_t i = 0; i < r_left; ++i)
          for (std::size_t j = 0; j < shape[dim]; ++j)
            for (std::size_t kk = 0; kk < r; ++kk)
              core.at(i, j, kk) *= S[kk];
      }

      tt.cores.push_back(std::move(core));

      // Prepare next iteration: mat = diag(S) * Vt
      if (dim + 1 < d) {
        // Vt has shape [r x cols] (cols is still the old value here).
        // mat = S * Vt, same shape [r x cols].
        mat.resize(r * cols, 0.0);
        for (std::size_t i = 0; i < r; ++i)
          for (std::size_t j = 0; j < cols; ++j)
            mat[i * cols + j] = S[i] * Vt[i * cols + j];
        // Reshape: [r x cols] -> [r * shape[dim+1] x cols / shape[dim+1]]
        // (row-major data stays the same, just reinterpret dimensions).
        rows = r * shape[dim + 1];
        cols = cols / shape[dim + 1];
        r_left = r;
      }
    }

    return tt;
  }

  // Reconstruct the full tensor from TT representation.
  static std::vector<double> Reconstruct(const TensorTrain& tt) {
    if (tt.cores.empty()) return {};

    // Start with first core: [1 x mode0 x r0]
    std::vector<double> result = tt.cores[0].data;
    std::size_t current_rows = 1;
    std::size_t current_mode = tt.cores[0].mode;
    std::size_t current_r = tt.cores[0].r_right;

    for (std::size_t c = 1; c < tt.cores.size(); ++c) {
      const auto& core = tt.cores[c];
      // result is [current_rows * current_mode x current_r]
      // core is [current_r x core.mode x core.r_right]
      // Contract over current_r.
      std::size_t new_rows = current_rows * current_mode;
      std::size_t new_cols = core.mode * core.r_right;
      std::vector<double> contracted(new_rows * new_cols, 0.0);

      for (std::size_t i = 0; i < new_rows; ++i)
        for (std::size_t j = 0; j < core.mode; ++j)
          for (std::size_t kk = 0; kk < core.r_right; ++kk) {
            double s = 0.0;
            for (std::size_t r = 0; r < current_r; ++r)
              s += result[i * current_r + r] * core.at(r, j, kk);
            contracted[i * new_cols + j * core.r_right + kk] = s;
          }

      result = std::move(contracted);
      current_rows = new_rows;
      current_mode = core.mode;
      current_r = core.r_right;
    }

    // result is now [total_modes x 1] — reshape to flat.
    return result;
  }

  // Quantize a 3D grid function into QTT format.
  // Grid must be 2^n x 2^n x 2^n. Each index is split into n binary bits,
  // creating a 3n-dimensional tensor.
  //   n0, n1, n2: grid dimensions (must be powers of 2)
  //   data: flat 3D array (n0 * n1 * n2, row-major: x fastest)
  //   eps: TT rounding accuracy
  static TensorTrain Quantize3D(const std::vector<double>& data,
                                 std::size_t n0, std::size_t n1, std::size_t n2,
                                 double eps = 1e-10) {
    // Check power-of-2.
    std::size_t nb0 = Log2(n0);
    std::size_t nb1 = Log2(n1);
    std::size_t nb2 = Log2(n2);
    if ((1UL << nb0) != n0 || (1UL << nb1) != n1 || (1UL << nb2) != n2)
      return {};

    // Interleave bits: index (ix, iy, iz) -> bit pattern
    // QTT index order: (ix_bit0, iy_bit0, iz_bit0, ix_bit1, iy_bit1, iz_bit1, ...)
    std::size_t total_dims = 3 * (nb0 + nb1 + nb2) / 3;  // = nb0 if all equal
    // For simplicity, require all dims equal (cubic grid).
    if (nb0 != nb1 || nb1 != nb2) return {};
    std::size_t nbits = nb0;
    total_dims = 3 * nbits;

    // Build the quantized tensor.
    std::vector<std::size_t> qshape(total_dims, 2);
    std::vector<double> qdata(1UL << total_dims, 0.0);

    for (std::size_t idx = 0; idx < n0 * n1 * n2; ++idx) {
      std::size_t iz = idx / (n0 * n1);
      std::size_t rem = idx % (n0 * n1);
      std::size_t iy = rem / n0;
      std::size_t ix = rem % n0;

      // Map to quantized index.
      std::size_t qidx = 0;
      for (std::size_t b = 0; b < nbits; ++b) {
        std::size_t bit_x = (ix >> b) & 1;
        std::size_t bit_y = (iy >> b) & 1;
        std::size_t bit_z = (iz >> b) & 1;
        qidx |= (bit_x << (3 * b));
        qidx |= (bit_y << (3 * b + 1));
        qidx |= (bit_z << (3 * b + 2));
      }
      qdata[qidx] = data[idx];
    }

    return Decompose(qdata, qshape, eps);
  }

  // Reconstruct from QTT back to 3D grid.
  static std::vector<double> Reconstruct3D(const TensorTrain& tt,
                                            std::size_t n0, std::size_t n1,
                                            std::size_t n2) {
    auto qdata = Reconstruct(tt);
    if (qdata.empty()) return {};

    std::size_t nbits = Log2(n0);
    std::vector<double> data(n0 * n1 * n2, 0.0);

    for (std::size_t idx = 0; idx < n0 * n1 * n2; ++idx) {
      std::size_t iz = idx / (n0 * n1);
      std::size_t rem = idx % (n0 * n1);
      std::size_t iy = rem / n0;
      std::size_t ix = rem % n0;

      std::size_t qidx = 0;
      for (std::size_t b = 0; b < nbits; ++b) {
        qidx |= ((ix >> b) & 1) << (3 * b);
        qidx |= ((iy >> b) & 1) << (3 * b + 1);
        qidx |= ((iz >> b) & 1) << (3 * b + 2);
      }
      data[idx] = qdata[qidx];
    }
    return data;
  }

  // Compression ratio: original_size / tt_param_count.
  static double CompressionRatio(const TensorTrain& tt,
                                  std::size_t original_size) {
    std::size_t params = tt.param_count();
    if (params == 0) return 0.0;
    return static_cast<double>(original_size) / static_cast<double>(params);
  }

  // TT rounding (re-compress by truncating small singular values).
  static TensorTrain Round(const TensorTrain& tt, double eps = 1e-12) {
    // Reconstruct then re-decompose with tighter eps.
    auto full = Reconstruct(tt);
    return Decompose(full, tt.modes, eps);
  }

  // TT addition (element-wise sum of two TTs with same modes).
  static TensorTrain Add(const TensorTrain& a, const TensorTrain& b) {
    if (a.modes != b.modes) return {};

    TensorTrain result;
    result.modes = a.modes;
    result.cores.resize(a.cores.size());

    for (std::size_t d = 0; d < a.cores.size(); ++d) {
      const auto& ca = a.cores[d];
      const auto& cb = b.cores[d];
      auto& cr = result.cores[d];

      if (d == 0) {
        // r_left = 1 for both; concatenate along mode.
        cr.r_left = 1;
        cr.mode = ca.mode;
        cr.r_right = ca.r_right + cb.r_right;
        cr.data.resize(cr.r_left * cr.mode * cr.r_right, 0.0);
        for (std::size_t j = 0; j < ca.mode; ++j) {
          for (std::size_t k = 0; k < ca.r_right; ++k)
            cr.at(0, j, k) = ca.at(0, j, k);
          for (std::size_t k = 0; k < cb.r_right; ++k)
            cr.at(0, j, ca.r_right + k) = cb.at(0, j, k);
        }
      } else if (d == a.cores.size() - 1) {
        // r_right = 1 for both; concatenate along mode.
        cr.r_left = ca.r_left + cb.r_left;
        cr.mode = ca.mode;
        cr.r_right = 1;
        cr.data.resize(cr.r_left * cr.mode * cr.r_right, 0.0);
        for (std::size_t i = 0; i < ca.r_left; ++i)
          for (std::size_t j = 0; j < ca.mode; ++j)
            cr.at(i, j, 0) = ca.at(i, j, 0);
        for (std::size_t i = 0; i < cb.r_left; ++i)
          for (std::size_t j = 0; j < ca.mode; ++j)
            cr.at(ca.r_left + i, j, 0) = cb.at(i, j, 0);
      } else {
        // Block-diagonal concatenation.
        cr.r_left = ca.r_left + cb.r_left;
        cr.mode = ca.mode;
        cr.r_right = ca.r_right + cb.r_right;
        cr.data.resize(cr.r_left * cr.mode * cr.r_right, 0.0);
        for (std::size_t i = 0; i < ca.r_left; ++i)
          for (std::size_t j = 0; j < ca.mode; ++j)
            for (std::size_t k = 0; k < ca.r_right; ++k)
              cr.at(i, j, k) = ca.at(i, j, k);
        for (std::size_t i = 0; i < cb.r_left; ++i)
          for (std::size_t j = 0; j < ca.mode; ++j)
            for (std::size_t k = 0; k < cb.r_right; ++k)
              cr.at(ca.r_left + i, j, ca.r_right + k) = cb.at(i, j, k);
      }
    }

    // Round to compress.
    return Round(result, 1e-14);
  }

  // TT dot product: <a, b> = sum of element-wise products.
  static double Dot(const TensorTrain& a, const TensorTrain& b) {
    if (a.modes != b.modes) return 0.0;
    // Contract cores sequentially.
    // r = identity matrix (1x1)
    std::vector<double> r = {1.0};
    std::size_t r_rows = 1, r_cols = 1;

    for (std::size_t d = 0; d < a.cores.size(); ++d) {
      const auto& ca = a.cores[d];
      const auto& cb = b.cores[d];
      // r is [r_rows x r_cols] (from previous contraction)
      // ca is [ca.r_left x ca.mode x ca.r_right]
      // cb is [cb.r_left x cb.mode x cb.r_right]
      // Contract: new_r[i, k] = sum_{j, p, q} r[p, q] * ca[q, j, i] * cb[p, j, k]
      // But r indices map to (a_rank, b_rank) from previous step.
      std::size_t new_rows = ca.r_right;
      std::size_t new_cols = cb.r_right;
      std::vector<double> new_r(new_rows * new_cols, 0.0);

      for (std::size_t i = 0; i < new_rows; ++i)
        for (std::size_t k = 0; k < new_cols; ++k) {
          double s = 0.0;
          for (std::size_t j = 0; j < ca.mode; ++j)
            for (std::size_t p = 0; p < r_rows; ++p)
              for (std::size_t q = 0; q < r_cols; ++q)
                s += r[p * r_cols + q] * ca.at(q, j, i) * cb.at(p, j, k);
          new_r[i * new_cols + k] = s;
        }

      r = std::move(new_r);
      r_rows = new_rows;
      r_cols = new_cols;
    }

    // r should be 1x1 at the end.
    return r[0];
  }

  // TT norm squared: <tt, tt>.
  static double NormSq(const TensorTrain& tt) {
    return Dot(tt, tt);
  }

  // TT norm.
  static double Norm(const TensorTrain& tt) {
    return std::sqrt(NormSq(tt));
  }

  // Poisson solver in QTT format (prototype).
  // Solves: -laplacian(V) = 4*pi*rho
  // Uses TT-based Richardson iteration: V_{k+1} = V_k + omega * (4*pi*rho - L*V_k)
  // where L is the finite-difference Laplacian applied in TT format.
  // This is a simple prototype; production would use AMEn/DMRG solvers.
  static TensorTrain SolvePoissonQTT(
      const TensorTrain& rho_tt,
      const std::vector<std::size_t>& grid_shape,
      double dv, int max_iter = 100, double omega = 0.005, double tol = 1e-6) {
    // Start with V = 0.
    TensorTrain V = ZerosLike(rho_tt);

    for (int iter = 0; iter < max_iter; ++iter) {
      // R = 4*pi*rho - L*V
      auto LV = ApplyLaplacian(V, grid_shape, dv);
      auto four_pi_rho = Scale(rho_tt, 4.0 * M_PI);
      auto R = Sub(four_pi_rho, LV);

      double res_norm = Norm(R);
      if (iter % 10 == 0)
        std::cout << "  QTT-Poisson iter " << iter
                  << " residual=" << res_norm << '\n';
      if (res_norm < tol) break;

      // V = V + omega * R
      auto correction = Scale(R, omega);
      V = Add(V, correction);
      // Truncate ranks to control growth.
      V = Round(V, 1e-10);
    }

    return V;
  }

 private:
  static std::size_t Log2(std::size_t n) {
    if (n == 0) return 0;
    std::size_t b = 0;
    while ((1UL << b) < n) ++b;
    return b;
  }

  // Truncated SVD via one-sided Jacobi (no external dependency).
  // Returns U [m x k], S [k], Vt [k x n], k = effective rank.
  struct SVDResult {
    std::vector<double> U;
    std::vector<double> S;
    std::vector<double> Vt;
    std::size_t k;
  };

  static SVDResult SVDTruncated(const std::vector<double>& mat,
                                 std::size_t m, std::size_t n, double eps) {
    // Use the smaller dimension for eigendecomposition to avoid huge matrices.
    const std::size_t k_max = std::min(m, n);

    SVDResult result;

    if (m <= n) {
      // Compute C = A * A^T (m x m), eigendecompose for U.
      std::vector<double> C(m * m, 0.0);
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
          double s = 0.0;
          for (std::size_t r = 0; r < n; ++r)
            s += mat[i * n + r] * mat[j * n + r];
          C[i * m + j] = s;
        }

      std::vector<double> eigvals(m, 0.0);
      std::vector<double> eigvecs = C;
      JacobiEig(eigvecs, eigvals, m, 100);

      std::vector<std::size_t> idx(m);
      for (std::size_t i = 0; i < m; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(),
                [&](std::size_t a, std::size_t b) { return eigvals[a] > eigvals[b]; });

      double total_sq = 0.0;
      for (std::size_t i = 0; i < m; ++i) total_sq += eigvals[i];
      double threshold = eps * eps * total_sq;
      double cum_sq = 0.0;
      std::size_t k = 0;
      for (std::size_t i = 0; i < k_max; ++i) {
        cum_sq += eigvals[idx[i]];
        k = i + 1;
        if (total_sq - cum_sq < threshold) break;
      }

      result.k = k;
      result.S.resize(k);
      result.U.resize(m * k, 0.0);
      result.Vt.resize(k * n, 0.0);

      for (std::size_t i = 0; i < k; ++i) {
        std::size_t ei = idx[i];
        double sigma = std::sqrt(std::max(0.0, eigvals[ei]));
        result.S[i] = sigma;

        // U column i = eigenvector ei (normalized).
        double unorm = 0.0;
        for (std::size_t j = 0; j < m; ++j)
          unorm += eigvecs[j * m + ei] * eigvecs[j * m + ei];
        unorm = std::sqrt(unorm);
        if (unorm < 1e-30) unorm = 1.0;
        for (std::size_t j = 0; j < m; ++j)
          result.U[j * k + i] = eigvecs[j * m + ei] / unorm;

        // Vt row i = A^T * U_i / sigma.
        if (sigma > 1e-30) {
          for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < m; ++r)
              s += mat[r * n + j] * result.U[r * k + i];
            result.Vt[i * n + j] = s / sigma;
          }
        }
      }
    } else {
      // Compute C = A^T * A (n x n), eigendecompose for V.
      std::vector<double> C(n * n, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          double s = 0.0;
          for (std::size_t r = 0; r < m; ++r)
            s += mat[r * n + i] * mat[r * n + j];
          C[i * n + j] = s;
        }

      std::vector<double> eigvals(n, 0.0);
      std::vector<double> eigvecs = C;
      JacobiEig(eigvecs, eigvals, n, 100);

      std::vector<std::size_t> idx(n);
      for (std::size_t i = 0; i < n; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(),
                [&](std::size_t a, std::size_t b) { return eigvals[a] > eigvals[b]; });

      double total_sq = 0.0;
      for (std::size_t i = 0; i < n; ++i) total_sq += eigvals[i];
      double threshold = eps * eps * total_sq;
      double cum_sq = 0.0;
      std::size_t k = 0;
      for (std::size_t i = 0; i < k_max; ++i) {
        cum_sq += eigvals[idx[i]];
        k = i + 1;
        if (total_sq - cum_sq < threshold) break;
      }

      result.k = k;
      result.S.resize(k);
      result.U.resize(m * k, 0.0);
      result.Vt.resize(k * n, 0.0);

      for (std::size_t i = 0; i < k; ++i) {
        std::size_t ei = idx[i];
        double sigma = std::sqrt(std::max(0.0, eigvals[ei]));
        result.S[i] = sigma;

        // Vt row i = eigenvector ei (normalized).
        double vnorm = 0.0;
        for (std::size_t j = 0; j < n; ++j)
          vnorm += eigvecs[j * n + ei] * eigvecs[j * n + ei];
        vnorm = std::sqrt(vnorm);
        if (vnorm < 1e-30) vnorm = 1.0;
        for (std::size_t j = 0; j < n; ++j)
          result.Vt[i * n + j] = eigvecs[j * n + ei] / vnorm;

        // U column i = A * V_i / sigma.
        if (sigma > 1e-30) {
          for (std::size_t r = 0; r < m; ++r) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j)
              s += mat[r * n + j] * result.Vt[i * n + j];
            result.U[r * k + i] = s / sigma;
          }
        }
      }
    }

    return result;
  }

  // Jacobi eigenvalue decomposition for symmetric matrix.
  // A is overwritten with eigenvectors as columns. eigvals gets eigenvalues.
  static void JacobiEig(std::vector<double>& A, std::vector<double>& eigvals,
                         std::size_t n, int max_sweeps) {
    // V accumulates eigenvectors (start as identity).
    std::vector<double> V(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) V[i * n + i] = 1.0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
      double off = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
          off += A[i * n + j] * A[i * n + j];
      if (off < 1e-30) break;

      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = p + 1; q < n; ++q) {
          double apq = A[p * n + q];
          if (std::fabs(apq) < 1e-30) continue;
          double app = A[p * n + p];
          double aqq = A[q * n + q];
          double theta = (aqq - app) / (2.0 * apq);
          double t = (theta >= 0 ? 1.0 : -1.0) /
                     (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
          double c = 1.0 / std::sqrt(t * t + 1.0);
          double s = t * c;

          // Update A.
          A[p * n + p] = app - t * apq;
          A[q * n + q] = aqq + t * apq;
          A[p * n + q] = 0.0;
          A[q * n + p] = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
              double aip = A[i * n + p];
              double aiq = A[i * n + q];
              A[i * n + p] = c * aip - s * aiq;
              A[p * n + i] = A[i * n + p];
              A[i * n + q] = s * aip + c * aiq;
              A[q * n + i] = A[i * n + q];
            }
          }
          // Accumulate eigenvectors: V = V * R(p,q,theta).
          for (std::size_t i = 0; i < n; ++i) {
            double vip = V[i * n + p];
            double viq = V[i * n + q];
            V[i * n + p] = c * vip - s * viq;
            V[i * n + q] = s * vip + c * viq;
          }
        }
    }

    // Extract eigenvalues from diagonal, eigenvectors are in V.
    for (std::size_t i = 0; i < n; ++i) eigvals[i] = A[i * n + i];
    // Overwrite A with V (eigenvectors as columns).
    A = V;
  }

  // Create a zero TT with same structure as input.
  static TensorTrain ZerosLike(const TensorTrain& tt) {
    TensorTrain result;
    result.modes = tt.modes;
    result.cores.resize(tt.cores.size());
    for (std::size_t d = 0; d < tt.cores.size(); ++d) {
      result.cores[d].r_left = tt.cores[d].r_left;
      result.cores[d].mode = tt.cores[d].mode;
      result.cores[d].r_right = tt.cores[d].r_right;
      result.cores[d].data.assign(tt.cores[d].data.size(), 0.0);
    }
    return result;
  }

  // Scale a TT by a scalar.
  static TensorTrain Scale(const TensorTrain& tt, double alpha) {
    TensorTrain result;
    result.modes = tt.modes;
    result.cores = tt.cores;
    // Scale first core.
    for (auto& v : result.cores[0].data) v *= alpha;
    return result;
  }

  // Subtract: a - b.
  static TensorTrain Sub(const TensorTrain& a, const TensorTrain& b) {
    auto neg_b = Scale(b, -1.0);
    return Add(a, neg_b);
  }

  // Apply finite-difference Laplacian to a TT representing a 3D grid function.
  // The grid is grid_shape[0] x grid_shape[1] x grid_shape[2].
  // L*V is computed by reconstructing, applying FD, and re-decomposing.
  // (Prototype: full reconstruction. Production would use TT-matrix operators.)
  static TensorTrain ApplyLaplacian(const TensorTrain& V,
                                     const std::vector<std::size_t>& grid_shape,
                                     double dv) {
    // Reconstruct to full grid.
    auto full = Reconstruct(V);
    if (full.empty()) return V;

    // Determine grid dimensions from QTT modes.
    // QTT has 3*nbits modes of size 2 each.
    std::size_t total_dims = V.modes.size();
    if (total_dims % 3 != 0) return V;
    std::size_t nbits = total_dims / 3;
    std::size_t n = 1UL << nbits;

    // Un-quantize to 3D.
    std::vector<double> v3d(n * n * n, 0.0);
    for (std::size_t idx = 0; idx < n * n * n; ++idx) {
      std::size_t iz = idx / (n * n);
      std::size_t rem = idx % (n * n);
      std::size_t iy = rem / n;
      std::size_t ix = rem % n;

      std::size_t qidx = 0;
      for (std::size_t b = 0; b < nbits; ++b) {
        qidx |= ((ix >> b) & 1) << (3 * b);
        qidx |= ((iy >> b) & 1) << (3 * b + 1);
        qidx |= ((iz >> b) & 1) << (3 * b + 2);
      }
      v3d[idx] = full[qidx];
    }

    // Apply FD Laplacian: L(V) = (V[i+1] - 2V[i] + V[i-1]) / h^2 per axis.
    double h = std::cbrt(dv);
    std::vector<double> lap(n * n * n, 0.0);
    auto idx3d = [n](std::size_t x, std::size_t y, std::size_t z) {
      return x + n * (y + n * z);
    };

    for (std::size_t z = 0; z < n; ++z)
      for (std::size_t y = 0; y < n; ++y)
        for (std::size_t x = 0; x < n; ++x) {
          double v = -6.0 * v3d[idx3d(x, y, z)];
          if (x > 0) v += v3d[idx3d(x - 1, y, z)];
          if (x + 1 < n) v += v3d[idx3d(x + 1, y, z)];
          if (y > 0) v += v3d[idx3d(x, y - 1, z)];
          if (y + 1 < n) v += v3d[idx3d(x, y + 1, z)];
          if (z > 0) v += v3d[idx3d(x, y, z - 1)];
          if (z + 1 < n) v += v3d[idx3d(x, y, z + 1)];
          lap[idx3d(x, y, z)] = v / (h * h);
        }

    // Re-quantize.
    return Quantize3D(lap, n, n, n, 1e-12);
  }
};

}  // namespace tides::grid
