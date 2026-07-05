// T2.1 foundation: self-test for the self-contained Jacobi symmetric
// eigensolver. Validates against constructed spectra with known eigenvalues,
// the residual ||A v - lambda v||, eigenvector orthonormality, and exact
// reconstruction A = V diag(lambda) V^T. This is an analytical/math correctness
// gate, not a language unit test.

#include "basis/atomgen/symmetric_eigensolver.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::atomgen::SymmetricEigensolver;

int Fail(const std::string& msg) {
  std::cerr << "eigensolver_tests: " << msg << '\n';
  return 1;
}

double Norm2(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v * v;
  return std::sqrt(s);
}

// Construct A = Q diag(lambda) Q^T with a random orthogonal Q (via QR of a
// random matrix using modified Gram-Schmidt). Returns A (row-major) and Q.
void BuildFromSpectrum(const std::vector<double>& lambda,
                       std::uint64_t seed, std::vector<double>& a,
                       std::vector<double>& q) {
  const std::size_t n = lambda.size();
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  q.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n * n; ++i) q[i] = g(rng);
  // Orthonormalize columns of q via modified Gram-Schmidt.
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      double dot = 0.0;
      for (std::size_t i = 0; i < n; ++i) dot += q[i * n + j] * q[i * n + k];
      for (std::size_t i = 0; i < n; ++i) q[i * n + j] -= dot * q[i * n + k];
    }
    double nrm = 0.0;
    for (std::size_t i = 0; i < n; ++i) nrm += q[i * n + j] * q[i * n + j];
    nrm = std::sqrt(nrm);
    for (std::size_t i = 0; i < n; ++i) q[i * n + j] /= nrm;
  }
  a.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += q[i * n + k] * lambda[k] * q[j * n + k];
      a[i * n + j] = s;
    }
}

int CheckSpectrum(std::size_t n, const std::vector<double>& lambda,
                  std::uint64_t seed, const std::string& label,
                  double val_tol, double vec_tol) {
  std::vector<double> a, q;
  BuildFromSpectrum(lambda, seed, a, q);

  std::vector<double> a_copy = a;  // for reconstruction / residual checks
  std::vector<double> w, v;
  SymmetricEigensolver::Solve(a, n, w, v);

  // 1. Eigenvalues match the constructed spectrum (sorted ascending).
  std::vector<double> lam_sorted = lambda;
  std::sort(lam_sorted.begin(), lam_sorted.end());
  double max_val_err = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    max_val_err = std::max(max_val_err, std::fabs(w[i] - lam_sorted[i]));
  if (max_val_err > val_tol) {
    std::ostringstream os;
    os << label << ": eigenvalue error " << max_val_err << " > " << val_tol;
    return Fail(os.str());
  }

  // 2. Residual ||A v_k - w_k v_k|| for each eigenpair.
  double max_res = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    std::vector<double> r(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) s += a_copy[i * n + j] * v[j * n + k];
      r[i] = s - w[k] * v[i * n + k];
    }
    max_res = std::max(max_res, Norm2(r));
  }
  if (max_res > vec_tol) {
    std::ostringstream os;
    os << label << ": residual " << max_res << " > " << vec_tol;
    return Fail(os.str());
  }

  // 3. Orthonormality of eigenvectors: V^T V = I.
  double max_orth = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += v[k * n + i] * v[k * n + j];
      double target = (i == j) ? 1.0 : 0.0;
      max_orth = std::max(max_orth, std::fabs(s - target));
    }
  if (max_orth > vec_tol) {
    std::ostringstream os;
    os << label << ": orthonormality error " << max_orth << " > " << vec_tol;
    return Fail(os.str());
  }

  // 4. Reconstruction A = V diag(w) V^T.
  double max_recon = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += v[i * n + k] * w[k] * v[j * n + k];
      max_recon = std::max(max_recon, std::fabs(s - a_copy[i * n + j]));
    }
  if (max_recon > vec_tol) {
    std::ostringstream os;
    os << label << ": reconstruction error " << max_recon << " > " << vec_tol;
    return Fail(os.str());
  }

  std::cout << label << ": n=" << n << " val_err=" << max_val_err
            << " residual=" << max_res << " orth=" << max_orth
            << " recon=" << max_recon << '\n';
  return 0;
}

}  // namespace

int main() {
  // Analytical 2x2: [[2,1],[1,2]] -> {1, 3}.
  {
    std::vector<double> a = {2.0, 1.0, 1.0, 2.0};
    std::vector<double> w, v;
    SymmetricEigensolver::Solve(a, 2, w, v);
    if (std::fabs(w[0] - 1.0) > 1e-13 || std::fabs(w[1] - 3.0) > 1e-13)
      return Fail("2x2 eigenvalues wrong");
    // Eigenvectors: v0 ~ (1,-1)/sqrt2, v1 ~ (1,1)/sqrt2 (up to sign).
    std::cout << "2x2: w={" << w[0] << "," << w[1] << "} OK\n";
  }

  // Diagonal matrix.
  {
    std::vector<double> a = {5.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0};
    std::vector<double> w;
    SymmetricEigensolver::ValuesOnly(a, 3, w);
    if (std::fabs(w[0] + 3.0) > 1e-13 || std::fabs(w[1] - 2.0) > 1e-13 ||
        std::fabs(w[2] - 5.0) > 1e-13)
      return Fail("diagonal eigenvalues wrong");
    std::cout << "diagonal 3x3: w={" << w[0] << "," << w[1] << "," << w[2]
              << "} OK\n";
  }

  // Constructed random spectra: eigenpairs must be accurate to ~machine
  // precision *relative to the spectral scale* (the intent of an analytical
  // eigensolver gate). Absolute tolerances scale with max|lambda|.
  const auto rel_tol_for = [](const std::vector<double>& lam) {
    double m = 1.0;
    for (double v : lam) m = std::max(m, std::fabs(v));
    // ~1e-13 relative; eigenvectors/residuals ~1e-11 relative (Jacobi floor).
    return std::make_pair(1e-13 * m, 1e-11 * m);
  };

  {
    auto lam = std::vector<double>{-7.0, -3.0, -1.0, 0.0, 0.5, 2.0, 4.0, 9.0};
    auto [vt, et] = rel_tol_for(lam);
    if (CheckSpectrum(8, lam, 11, "rand_8_mixed", vt, et)) return 1;
  }
  {
    auto lam = std::vector<double>{-100.0, -50.0, -10.0, -1.0, -0.1, 0.0, 0.01,
                                    0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0,
                                    5000.0, 12345.0};
    auto [vt, et] = rel_tol_for(lam);
    if (CheckSpectrum(16, lam, 23, "rand_16_wide_dynamic", vt, et)) return 1;
  }
  {
    auto lam = std::vector<double>{2.0, 2.0, 2.0, 5.0, 5.0, -1.0};
    auto [vt, et] = rel_tol_for(lam);
    if (CheckSpectrum(6, lam, 7, "rand_6_degenerate", vt, et)) return 1;
  }

  std::cout << "eigensolver_tests: ALL GREEN\n";
  return 0;
}
