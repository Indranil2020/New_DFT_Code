#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "solvers/sp2_submatrix/sp2.hpp"

namespace tides::solvers {

// Fermi-operator expansion (FOE) — R3 regime, metallic, finite-Te.
//
// P = f_FD((H - mu S) / kT_e) approximated by a Chebyshev polynomial expansion
// of the Fermi-Dirac function. The expansion is applied as repeated matrix
// products on the same tiles (per 20-math/23): "all work = repeated
// spgemm_filtered on the same tiles => same substrate, no diagonalization."
//
// The Fermi-Dirac function f(x) = 1/(1 + exp(x)) is expanded as:
//   f(x) = sum_k c_k T_k(x_normalized)
// where T_k are Chebyshev polynomials and c_k are Chebyshev coefficients
// computed via Clenshaw-Curtis quadrature. The polynomial order p scales as
// beta * spectral_width (per 20-math/23: "p ~ beta * DeltaH").
//
// Observable (T5.5): order-vs-beta curve matches theory; free energy match
// <=1 meV/atom vs dense control.

struct FOEResult {
  std::vector<double> P;        // density matrix (row-major, n x n)
  double mu = 0.0;              // chemical potential found
  int chebyshev_order = 0;      // polynomial order used
  double trace_PS = 0.0;       // tr(P S) = N_e
  bool ok = false;
};

class FermiOperatorExpansion {
 public:
  // Compute the density matrix via Chebyshev expansion of the Fermi function.
  //   n:       matrix dimension
  //   H, S:    symmetric matrices (row-major); S SPD
  //   mu:      chemical potential
  //   kT_e:    electronic temperature (Hartree)
  //   lambda_min, lambda_max: spectral bounds of (H - mu S)
  //   order:   Chebyshev polynomial order (p ~ beta * DeltaH)
  static FOEResult Compute(std::size_t n, const std::vector<double>& H,
                           const std::vector<double>& S, double mu,
                           double kT_e, double lambda_min, double lambda_max,
                           int order = 30) {
    FOEResult res;
    if (n == 0 || H.size() != n * n || S.size() != n * n) return res;
    if (kT_e <= 0 || lambda_max <= lambda_min) return res;

    // The Fermi function f(x) = 1/(1+exp(x/kT)), where x = eigenvalue of
    // (H - mu S). We expand f in Chebyshev polynomials on the spectral range
    // of (H - mu S): [lambda_min - mu, lambda_max - mu] (for S=I).
    // For the general case, the spectral range of the generalized problem
    // shifted by mu is [lambda_min - mu, lambda_max - mu] (since the
    // generalized eigenvalues don't shift simply with mu*S; but for the
    // standard problem S=I they do). We use the shifted bounds.
    const double lo_shifted = lambda_min - mu;
    const double hi_shifted = lambda_max - mu;
    const double scale = hi_shifted - lo_shifted;
    const double shift = (hi_shifted + lo_shifted) / 2.0;

    // Compute Chebyshev coefficients of f(x) on [-1, 1] (mapped from the
    // spectral range). f_mapped(t) = 1/(1 + exp((x(t) - mu)/kT)).
    // x(t) = shift + scale*t/2, so x ranges in [shift - scale/2, shift + scale/2]
    // = [lambda_min, lambda_max].
    std::vector<double> coeffs(order + 1, 0.0);
    const int n_quad = std::max(2 * order + 1, 64);
    for (int k = 0; k <= order; ++k) {
      double sum = 0.0;
      for (int j = 0; j < n_quad; ++j) {
        const double theta_j = M_PI * (j + 0.5) / n_quad;
        const double t = std::cos(theta_j);
        const double x = shift + scale * t / 2.0;
        // x is already in the shifted domain [lambda_min-mu, lambda_max-mu].
        // The Fermi function is f(e) = 1/(1+exp((e-mu)/kT)). Here e-mu = x.
        const double arg = x / kT_e;
        // Overflow-safe Fermi function.
        double f_val;
        if (arg > 500) f_val = 0.0;
        else if (arg < -500) f_val = 1.0;
        else f_val = 1.0 / (1.0 + std::exp(arg));
        sum += f_val * std::cos(static_cast<double>(k) * theta_j);
      }
      coeffs[k] = (k == 0) ? sum / n_quad : 2.0 * sum / n_quad;
    }

    // Apply the Chebyshev expansion to the matrix (H - mu S).
    // X maps the eigenvalues of (H - mu S) to [-1, 1].
    // eigenvalue of (H - mu S) is (e - mu). We want:
    //   t = 2*(e - mu - shift_x)/scale  in [-1, 1]
    // where shift_x = (lo_shifted + hi_shifted)/2 and scale = hi_shifted - lo_shifted.
    // So X = 2*(H - mu*S - shift_x*I)/scale = ((H-mu*S) - shift_x*I)/(scale/2).
    // Note: shift_x = (lambda_min-mu + lambda_max-mu)/2 = (lambda_min+lambda_max)/2 - mu.
    // And scale = lambda_max - lambda_min (mu cancels).
    // So X = (H - mu*S - ((lambda_min+lambda_max)/2 - mu)*I) / ((lambda_max-lambda_min)/2)
    //      = (H - mu*S - (lambda_min+lambda_max)/2 * I + mu*I) / ((lambda_max-lambda_min)/2)
    //      = (H - (lambda_min+lambda_max)/2 * I) / ((lambda_max-lambda_min)/2)
    // The mu cancels in X! X maps e -> [-1,1] regardless of mu. The mu
    // dependence is entirely in the Chebyshev coefficients (the Fermi function).
    const double center = (lambda_min + lambda_max) / 2.0;
    const double half_width = (lambda_max - lambda_min) / 2.0;
    std::vector<double> X(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        X[i * n + j] = (H[i * n + j] - mu * S[i * n + j] -
                       (i == j ? center - mu : 0.0)) / half_width;

    // P = c_0 I + c_1 X + c_2 T_2(X) + ... via Clenshaw recurrence.
    // T_0 = I, T_1 = X, T_k = 2 X T_{k-1} - T_{k-2}.
    std::vector<double> T_prev2(n * n, 0.0);  // T_0 = I
    for (std::size_t i = 0; i < n; ++i) T_prev2[i * n + i] = 1.0;
    std::vector<double> T_prev1 = X;  // T_1 = X

    // P = coeffs[0] * I + coeffs[1] * X
    res.P.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n * n; ++i) {
      res.P[i] = coeffs[0] * T_prev2[i] + coeffs[1] * T_prev1[i];
    }

    for (int k = 2; k <= order; ++k) {
      // T_k = 2 X T_{k-1} - T_{k-2}
      auto XT = MatMul(n, X, T_prev1);
      std::vector<double> Tk(n * n, 0.0);
      for (std::size_t i = 0; i < n * n; ++i)
        Tk[i] = 2.0 * XT[i] - T_prev2[i];
      // Add coeffs[k] * T_k to P.
      for (std::size_t i = 0; i < n * n; ++i)
        res.P[i] += coeffs[k] * Tk[i];
      T_prev2 = T_prev1;
      T_prev1 = Tk;
    }

    res.mu = mu;
    res.chebyshev_order = order;
    res.trace_PS = SP2Purification::TraceS(n, res.P, S);
    res.ok = true;
    return res;
  }

  // Theoretical Chebyshev order needed: p ~ beta * DeltaH (per 20-math/23).
  // beta = 1/kT, DeltaH = spectral width.
  static int TheoreticalOrder(double kT_e, double spectral_width) {
    if (kT_e <= 0) return 1000;  // T=0: no polynomial expansion (use SP2)
    const double beta = 1.0 / kT_e;
    return static_cast<int>(std::ceil(beta * spectral_width));
  }

 private:
  static std::vector<double> MatMul(std::size_t n, const std::vector<double>& A,
                                    const std::vector<double>& B) {
    std::vector<double> C(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
      }
    return C;
  }
};

}  // namespace tides::solvers
