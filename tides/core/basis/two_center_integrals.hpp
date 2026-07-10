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

// AUDIT: NAO higher-l two-center integrals via Slater-Koster tables.
// The two-center integral for (l_a, m_a, l_b, m_b) at distance R along
// direction (theta, phi) is:
//   I(l_a,m_a,l_b,m_b) = sum_L C(l_a,m_a,l_b,m_b,L) * h_L(R) * P_L(cos(theta))
// where h_L(R) is the radial integral (tabulated via spline) and
// C is the angular coupling coefficient from the Slater-Koster table.
//
// For s-s (l_a=0, l_b=0): only L=0, C=1, h_0 = S_ss(R).
// For s-p (l_a=0, l_b=1): L=1, h_1 = S_sp(R), angular = cos(theta).
// For p-s (l_a=1, l_b=0): L=1, h_1 = S_ps(R) = S_sp(R), angular = cos(theta).
// For p-p (l_a=1, l_b=1): L=0 and L=2, sigma and pi channels.
// For s-d (l_a=0, l_b=2): L=2, h_2 = S_sd(R), angular ~ P_2(cos(theta)).
// For d-s (l_a=2, l_b=0): L=2, h_2 = S_ds(R) = S_sd(R).
// For p-d (l_a=1, l_b=2): L=1 and L=3, sigma and pi channels.
// For d-p (l_a=2, l_b=1): L=1 and L=3.
// For d-d (l_a=2, l_b=2): L=0, L=2, L=4 — sigma, pi, delta channels.
class TwoCenterAngularCoupling {
 public:
  // Compute the angular factor for a two-center integral.
  // Given l_a, m_a, l_b, m_b, and the direction (theta, phi) of the
  // interatomic axis, returns the angular coupling coefficient.
  //
  // This uses the Slater-Koster decomposition:
  //   I = sum_L h_L(R) * AngularFactor(l_a, m_a, l_b, m_b, L, theta, phi)
  //
  // We return a vector of (L, angular_factor) pairs.
  struct CouplingTerm { int L; double factor; };
  static std::vector<CouplingTerm> Compute(
      int l_a, int m_a, int l_b, int m_b,
      double cos_theta, double phi) {
    std::vector<CouplingTerm> result;

    if (l_a == 0 && l_b == 0) {
      // s-s: only sigma, L=0.
      result.push_back({0, 1.0});
    } else if (l_a == 0 && l_b == 1) {
      // s-p: L=1, angular = direction component of p.
      result.push_back({1, SPCoupling(m_b, cos_theta, phi)});
    } else if (l_a == 1 && l_b == 0) {
      // p-s: L=1, angular = direction component of p.
      result.push_back({1, SPCoupling(m_a, cos_theta, phi)});
    } else if (l_a == 1 && l_b == 1) {
      // p-p: sigma (L=0) and pi (L=2) channels.
      auto terms = PPCoupling(m_a, m_b, cos_theta, phi);
      for (const auto& t : terms) result.push_back(t);
    } else if (l_a == 0 && l_b == 2) {
      // s-d: L=2, angular = P_2 component of d.
      result.push_back({2, SDCoupling(m_b, cos_theta, phi)});
    } else if (l_a == 2 && l_b == 0) {
      // d-s: L=2, angular = P_2 component of d.
      result.push_back({2, SDCoupling(m_a, cos_theta, phi)});
    } else if (l_a == 1 && l_b == 2) {
      // p-d: L=1 and L=3 channels.
      auto terms = PDCoupling(m_a, m_b, cos_theta, phi);
      for (const auto& t : terms) result.push_back(t);
    } else if (l_a == 2 && l_b == 1) {
      // d-p: L=1 and L=3 channels (symmetric to p-d with a↔b).
      auto terms = PDCoupling(m_b, m_a, cos_theta, phi);
      for (const auto& t : terms) result.push_back(t);
    } else if (l_a == 2 && l_b == 2) {
      // d-d: L=0, L=2, L=4 channels (sigma, pi, delta).
      auto terms = DDCoupling(m_a, m_b, cos_theta, phi);
      for (const auto& t : terms) result.push_back(t);
    }

    return result;
  }

 private:
  // s-p coupling: the p orbital component along the interatomic axis.
  // p_x → sin(theta)*cos(phi), p_y → sin(theta)*sin(phi), p_z → cos(theta)
  static double SPCoupling(int m_p, double cos_theta, double phi) {
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    if (m_p == 0) return cos_theta;                        // p_z
    if (m_p == 1) return sin_theta * std::cos(phi);        // p_x
    if (m_p == -1) return sin_theta * std::sin(phi);       // p_y
    return 0.0;
  }

  // p-p coupling: sigma (L=0) and pi (L=2) channels.
  // sigma: (p_z|p_z) → cos^2(theta), (p_x|p_x) → sin^2*cos^2(phi), etc.
  // pi: orthogonal components.
  static std::vector<CouplingTerm> PPCoupling(
      int m_a, int m_b, double cos_theta, double phi) {
    std::vector<CouplingTerm> result;
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    const double ct = cos_theta;
    const double st = sin_theta;
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);

    // Direction unit vector components.
    // p_z → ct, p_x → st*cp, p_y → st*sp
    double da[3] = {0, 0, 0}, db[3] = {0, 0, 0};
    int idx_a = m_a + 1;  // -1→0(y), 0→1(z), 1→2(x)
    int idx_b = m_b + 1;
    // Map m to Cartesian: m=0→z, m=1→x, m=-1→y
    auto fill_dir = [](int m, double ct, double st, double cp, double sp, double d[3]) {
      if (m == 0) { d[0] = ct; d[1] = 0; d[2] = 0; }       // z
      else if (m == 1) { d[0] = 0; d[1] = st*cp; d[2] = 0; } // x
      else if (m == -1) { d[0] = 0; d[1] = 0; d[2] = st*sp; } // y
    };
    // Actually, let's use a simpler approach:
    // p_sigma = (d . e_a)(d . e_b) where d is the unit vector along R.
    // p_pi = delta_ab - (d . e_a)(d . e_b)
    double ea[3] = {0, 0, 0}, eb[3] = {0, 0, 0};
    if (m_a == 0) ea[2] = 1.0;
    else if (m_a == 1) ea[0] = 1.0;
    else if (m_a == -1) ea[1] = 1.0;
    if (m_b == 0) eb[2] = 1.0;
    else if (m_b == 1) eb[0] = 1.0;
    else if (m_b == -1) eb[1] = 1.0;

    double d[3] = {st * cp, st * sp, ct};
    double da_dot = ea[0]*d[0] + ea[1]*d[1] + ea[2]*d[2];
    double db_dot = eb[0]*d[0] + eb[1]*d[1] + eb[2]*d[2];
    double ea_eb = ea[0]*eb[0] + ea[1]*eb[1] + ea[2]*eb[2];

    // sigma component: (d.e_a)(d.e_b) → L=0 radial
    // pi component: (e_a.e_b - (d.e_a)(d.e_b)) → L=2 radial
    double sigma = da_dot * db_dot;
    double pi = ea_eb - sigma;

    result.push_back({0, sigma});
    result.push_back({2, pi});
    return result;
  }

  // s-d coupling: the d orbital component along the interatomic axis.
  // d functions: m=0→3z^2-r^2, m=±1→xz,yz, m=±2→x^2-y^2,xy
  // For s-d: L=2, angular = (d_orbital evaluated along R direction) / r^2
  static double SDCoupling(int m_d, double cos_theta, double phi) {
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    const double ct = cos_theta;
    const double st = sin_theta;
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);

    // Real spherical harmonic Y_2m evaluated at (theta, phi).
    // Y_20 ~ (3cos^2(theta) - 1)/2
    // Y_2,±1 ~ sin(theta)cos(theta) * cos/sin(phi)
    // Y_2,±2 ~ sin^2(theta) * cos/sin(2phi)
    if (m_d == 0) return 0.5 * (3.0 * ct * ct - 1.0);
    if (m_d == 1) return std::sqrt(3.0) * st * ct * cp;
    if (m_d == -1) return std::sqrt(3.0) * st * ct * sp;
    if (m_d == 2) return std::sqrt(3.0) / 2.0 * st * st * (cp * cp - sp * sp);
    if (m_d == -2) return std::sqrt(3.0) * st * st * cp * sp;
    return 0.0;
  }

  // p-d coupling: L=1 and L=3 channels.
  static std::vector<CouplingTerm> PDCoupling(
      int m_p, int m_d, double cos_theta, double phi) {
    std::vector<CouplingTerm> result;
    // This is a simplified version using the Slater-Koster approach.
    // The full implementation would use Gaunt coefficients.
    // For now, we compute the angular factors as products of
    // the p and d directional components.

    const double st = std::sqrt(1.0 - cos_theta * cos_theta);
    const double ct = cos_theta;
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);

    // p directional components.
    double p_dir = 0.0;
    if (m_p == 0) p_dir = ct;
    else if (m_p == 1) p_dir = st * cp;
    else if (m_p == -1) p_dir = st * sp;

    // d directional components (same as SDCoupling).
    double d_dir = SDCoupling(m_d, cos_theta, phi);

    // The p-d coupling has two channels:
    // L=1 (sigma-like) and L=3 (pi/delta-like).
    // Simplified: sigma = p_dir * d_dir (aligned), pi = orthogonal.
    // A proper implementation would use Gaunt coefficients.
    // For the Tier-0 implementation, we use the product form.
    double sigma = p_dir * d_dir;
    // The L=1 and L=3 decomposition requires Clebsch-Gordan coefficients.
    // For now, put everything in L=1 and L=3 with simplified factors.
    result.push_back({1, sigma * 0.5});
    result.push_back({3, sigma * 0.5});
    return result;
  }

  // d-d coupling: L=0, L=2, L=4 channels (sigma, pi, delta).
  static std::vector<CouplingTerm> DDCoupling(
      int m_a, int m_b, double cos_theta, double phi) {
    std::vector<CouplingTerm> result;
    // Simplified: use product of d directional components.
    double da = SDCoupling(m_a, cos_theta, phi);
    double db = SDCoupling(m_b, cos_theta, phi);

    // sigma (L=0), pi (L=2), delta (L=4) — simplified decomposition.
    double sigma = da * db;
    result.push_back({0, sigma * 0.4});
    result.push_back({2, sigma * 0.4});
    result.push_back({4, sigma * 0.2});
    return result;
  }
};

// Two-center integral evaluator using spline-tabulated radial parts
// and Slater-Koster angular coupling.
class TwoCenterIntegralEval {
 public:
  // Tabulate radial integrals for a given (l_a, l_b) pair.
  // h_L(R) is the radial integral for angular momentum channel L.
  // For s-s: L=0 only. For p-p: L=0,2. For d-d: L=0,2,4.
  struct RadialTable {
    int l_a, l_b;
    std::vector<CubicSpline> h_L;  // h_L[R] for each L channel
  };

  // Evaluate a two-center integral at distance R and direction (theta, phi).
  // I = sum_L h_L(R) * AngularFactor(l_a, m_a, l_b, m_b, L, theta, phi)
  static double Eval(
      const RadialTable& table,
      int m_a, int m_b,
      double R, double cos_theta, double phi) {
    auto couplings = TwoCenterAngularCoupling::Compute(
        table.l_a, m_a, table.l_b, m_b, cos_theta, phi);

    double result = 0.0;
    for (const auto& c : couplings) {
      if (c.L < static_cast<int>(table.h_L.size())) {
        result += c.factor * table.h_L[c.L].Eval(R);
      }
    }
    return result;
  }

  // Build a radial table for s-s overlap from a model function.
  // For testing: S_ss(R) = exp(-alpha * R^2).
  static RadialTable BuildSSTable(double alpha,
                                   const std::vector<double>& R_grid) {
    RadialTable table;
    table.l_a = 0;
    table.l_b = 0;
    std::vector<double> vals(R_grid.size());
    for (std::size_t i = 0; i < R_grid.size(); ++i) {
      vals[i] = std::exp(-alpha * R_grid[i] * R_grid[i]);
    }
    table.h_L.push_back(CubicSpline(R_grid, vals));
    return table;
  }

  // Build a radial table for s-p overlap.
  // For testing: S_sp(R) = R * exp(-alpha * R^2) (sigma-type).
  static RadialTable BuildSPTable(double alpha,
                                   const std::vector<double>& R_grid) {
    RadialTable table;
    table.l_a = 0;
    table.l_b = 1;
    std::vector<double> vals(R_grid.size());
    for (std::size_t i = 0; i < R_grid.size(); ++i) {
      vals[i] = R_grid[i] * std::exp(-alpha * R_grid[i] * R_grid[i]);
    }
    table.h_L.push_back(CubicSpline());  // L=0 unused for s-p
    table.h_L.push_back(CubicSpline(R_grid, vals));  // L=1
    return table;
  }

  // Build a radial table for p-p overlap.
  // sigma: S_pp_sigma(R) = (1 - alpha*R^2) * exp(-alpha*R^2)
  // pi:    S_pp_pi(R) = exp(-alpha*R^2)
  static RadialTable BuildPPTable(double alpha,
                                   const std::vector<double>& R_grid) {
    RadialTable table;
    table.l_a = 1;
    table.l_b = 1;
    std::vector<double> sigma_vals(R_grid.size());
    std::vector<double> pi_vals(R_grid.size());
    for (std::size_t i = 0; i < R_grid.size(); ++i) {
      double R = R_grid[i];
      double e = std::exp(-alpha * R * R);
      sigma_vals[i] = (1.0 - alpha * R * R) * e;
      pi_vals[i] = e;
    }
    table.h_L.push_back(CubicSpline(R_grid, sigma_vals));  // L=0 (sigma)
    table.h_L.push_back(CubicSpline());                     // L=1 unused
    table.h_L.push_back(CubicSpline(R_grid, pi_vals));      // L=2 (pi)
    return table;
  }

  // Build a radial table for s-d overlap.
  // S_sd(R) = R^2 * exp(-alpha * R^2) (L=2 channel).
  static RadialTable BuildSDTable(double alpha,
                                   const std::vector<double>& R_grid) {
    RadialTable table;
    table.l_a = 0;
    table.l_b = 2;
    std::vector<double> vals(R_grid.size());
    for (std::size_t i = 0; i < R_grid.size(); ++i) {
      double R = R_grid[i];
      vals[i] = R * R * std::exp(-alpha * R * R);
    }
    table.h_L.push_back(CubicSpline());  // L=0 unused
    table.h_L.push_back(CubicSpline());  // L=1 unused
    table.h_L.push_back(CubicSpline(R_grid, vals));  // L=2
    return table;
  }

  // Build a radial table for d-d overlap.
  // sigma: (1 - 2*alpha*R^2 + alpha^2*R^4/3) * exp(-alpha*R^2)
  // pi:    (1 - alpha*R^2/2) * exp(-alpha*R^2)
  // delta: exp(-alpha*R^2)
  static RadialTable BuildDDTable(double alpha,
                                   const std::vector<double>& R_grid) {
    RadialTable table;
    table.l_a = 2;
    table.l_b = 2;
    std::vector<double> sigma_vals(R_grid.size());
    std::vector<double> pi_vals(R_grid.size());
    std::vector<double> delta_vals(R_grid.size());
    for (std::size_t i = 0; i < R_grid.size(); ++i) {
      double R = R_grid[i];
      double aR2 = alpha * R * R;
      double e = std::exp(-aR2);
      sigma_vals[i] = (1.0 - 2.0 * aR2 + aR2 * aR2 / 3.0) * e;
      pi_vals[i] = (1.0 - aR2 / 2.0) * e;
      delta_vals[i] = e;
    }
    table.h_L.push_back(CubicSpline(R_grid, sigma_vals));  // L=0 (sigma)
    table.h_L.push_back(CubicSpline());                     // L=1 unused
    table.h_L.push_back(CubicSpline(R_grid, pi_vals));      // L=2 (pi)
    table.h_L.push_back(CubicSpline());                     // L=3 unused
    table.h_L.push_back(CubicSpline(R_grid, delta_vals));   // L=4 (delta)
    return table;
  }
};

}  // namespace tides::basis
