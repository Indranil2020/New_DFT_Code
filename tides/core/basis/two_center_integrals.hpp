#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::basis {

// Cubic Hermite spline for tabulated two-center integrals as a function of
// interatomic distance R. Two-center integrals (overlap S, kinetic T, and
// Kleinman-Bylander nonlocal V_nl) depend only on R = |R_a - R_b| and the
// angular momenta (l_a, l_b, L), so tabulating them on a 1D R-grid and
// evaluating via spline is the standard NAO approach (SIESTA/FHI-aims lineage).
//
// The spline supports monotone-scaled inputs and returns value + derivative
// (for force contributions, T2.6).
class CubicSpline {
 public:
  CubicSpline() = default;

  // Build from tabulated (x, y) points. Computes the second derivatives d2y
  // via the natural cubic spline tridiagonal solve (O(n)).
  CubicSpline(std::vector<double> x, std::vector<double> y)
      : x_(std::move(x)), y_(std::move(y)) {
    const std::size_t n = x_.size();
    if (n < 2) return;
    d2y_.assign(n, 0.0);
    if (n == 2) return;  // linear; d2y = 0
    std::vector<double> u(n, 0.0);
    d2y_[0] = 0.0;  // natural spline
    u[0] = 0.0;
    for (std::size_t i = 1; i < n - 1; ++i) {
      const double dx0 = x_[i] - x_[i - 1];
      const double dx1 = x_[i + 1] - x_[i];
      const double sig = dx0 / dx1;
      const double p = sig * d2y_[i - 1] + 2.0;
      d2y_[i] = (sig - 1.0) / p;
      u[i] = (6.0 * ((y_[i + 1] - y_[i]) / dx1 - (y_[i] - y_[i - 1]) / dx0) /
                  (dx0 + dx1) - sig * u[i - 1]) / p;
    }
    double qn = 0.0, un = 0.0;  // natural spline at the end
    d2y_[n - 1] = (un - qn * u[n - 2]) / (qn * d2y_[n - 2] + 1.0);
    for (std::size_t i = n - 1; i-- > 0;)
      d2y_[i] = d2y_[i] * d2y_[i + 1] + u[i];
  }

  // Evaluate the spline at xq. Extrapolates linearly outside the range.
  [[nodiscard]] double Eval(double xq) const {
    return EvalWithDeriv(xq).first;
  }

  // Evaluate value and first derivative at xq.
  [[nodiscard]] std::pair<double, double> EvalWithDeriv(double xq) const {
    const std::size_t n = x_.size();
    if (n == 0) return {0.0, 0.0};
    if (n == 1) return {y_[0], 0.0};
    // Find bracketing interval [i, i+1].
    std::size_t i;
    if (xq <= x_[0]) {
      // Linear extrapolation below.
      const double dx = x_[1] - x_[0];
      const double slope = (y_[1] - y_[0]) / dx;
      return {y_[0] + slope * (xq - x_[0]), slope};
    }
    if (xq >= x_[n - 1]) {
      const double dx = x_[n - 1] - x_[n - 2];
      const double slope = (y_[n - 1] - y_[n - 2]) / dx;
      return {y_[n - 1] + slope * (xq - x_[n - 1]), slope};
    }
    // Binary search for the interval.
    std::size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (x_[mid] <= xq) lo = mid; else hi = mid;
    }
    i = lo;
    const double h = x_[i + 1] - x_[i];
    const double a = (x_[i + 1] - xq) / h;
    const double b = (xq - x_[i]) / h;
    const double yq = a * y_[i] + b * y_[i + 1] +
                      ((a * a * a - a) * d2y_[i] +
                       (b * b * b - b) * d2y_[i + 1]) * (h * h) / 6.0;
    const double dyq = (y_[i + 1] - y_[i]) / h +
                       ((3.0 * b * b - 1.0) * d2y_[i + 1] -
                        (3.0 * a * a - 1.0) * d2y_[i]) * h / 6.0;
    return {yq, dyq};
  }

  [[nodiscard]] std::size_t size() const { return x_.size(); }
  [[nodiscard]] const std::vector<double>& x() const { return x_; }
  [[nodiscard]] const std::vector<double>& y() const { return y_; }
  [[nodiscard]] const std::vector<double>& d2y() const { return d2y_; }

 private:
  std::vector<double> x_, y_, d2y_;
};

// Real spherical harmonics Y_lm(theta, phi). Used by the Slater-Koster rotation
// machinery. The real harmonics (tesseral harmonics) are the standard choice
// for NAO codes (block-diagonal real representation).
//
// For the rotation-invariance test we need the angular coupling coefficients
// (the Slater-Koster tables). Rather than hardcode the full SK table, we build
// integrals from the angular overlap integrals directly: the two-center
// integral for (l_a, m_a, l_b, m_b) is a product of a radial part (from the
// spline) and an angular part (from the rotation-coupled spherical harmonics).
class RealSphericalHarmonics {
 public:
  // Evaluate the real spherical harmonic Y_lm at direction (theta, phi).
  // Uses the standard Condon-Shortley phase convention for the associated
  // Legendre polynomials. Real harmonics (tesseral):
  //   m > 0: Y_lm = N_lm * P_l^m(cos theta) * cos(m*phi)
  //   m < 0: Y_lm = N_l|m| * P_l^|m|(cos theta) * sin(|m|*phi)
  //   m = 0: Y_l0 = N_l0 * P_l^0(cos theta)
  // with N_lm = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!).
  static double Eval(int l, int m, double theta, double phi) {
    const double x = std::cos(theta);
    const int am = std::abs(m);
    const double Plm = AssociatedLegendre(l, am, x);
    const double N = Normalization(l, am);
    double angular;
    if (m > 0) {
      angular = N * Plm * std::cos(static_cast<double>(m) * phi);
    } else if (m < 0) {
      angular = N * Plm * std::sin(static_cast<double>(am) * phi);
    } else {
      angular = N * Plm;
    }
    return angular;
  }

 private:
  // Normalization for REAL spherical harmonics. For m != 0 the real form
  // combines the +m and -m complex harmonics, giving an extra sqrt(2) factor:
  //   m = 0: N = sqrt((2l+1)/(4pi))
  //   m != 0: N = sqrt(2) * sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
  // with P_l^m computed WITHOUT the Condon-Shortley phase.
  static double Normalization(int l, int m) {
    double denom = 1.0;
    for (int i = l - m + 1; i <= l + m; ++i) denom *= static_cast<double>(i);
    double n = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) / denom);
    if (m != 0) n *= std::sqrt(2.0);
    return n;
  }

  // Associated Legendre P_l^m(x) WITHOUT the Condon-Shortley phase (the real
  // spherical harmonics convention drops the (−1)^m factor). This gives
  // P_l^m(x) >= 0 for x in [0,1] at m=l (the standard "geodesy" convention).
  static double AssociatedLegendre(int l, int m, double x) {
    if (m > l) return 0.0;
    // Compute P_m^m (seed): P_m^m = (2m-1)!! * (1-x^2)^{m/2} (no CS phase).
    double pmm = 1.0;
    if (m > 0) {
      double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
      double fact = 1.0;
      for (int i = 1; i <= m; ++i) {
        pmm *= fact * somx2;
        fact += 2.0;
      }
    }
    if (l == m) return pmm;
    // P_{m+1}^m = x * (2m+1) * P_m^m.
    double pmmp1 = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) return pmmp1;
    // Recurse: (l-m) P_l^m = (2l-1) x P_{l-1}^m - (l+m-1) P_{l-2}^m.
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
      pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) /
            (ll - m);
      pmm = pmmp1;
      pmmp1 = pll;
    }
    return pll;
  }
};

}  // namespace tides::basis
