#pragma once

// TIDES — NAO two-center integral builder.
//
// Builds overlap (S) and kinetic (T) matrices for numeric atom-centered
// orbitals using analytic two-center integrals (Slater-Koster + radial
// splines), replacing the grid-integrated S/T path in NaoDriver.
//
// Physics:
//   phi_a(r) = R_a(r) Y_{l_a m_a}(rhat)
//   S_ab = integral R_a(r) R_b(r_b) Y_a(rhat) Y_b(r_b_hat) d^3r
//
// The two-center integral is factorized into a radial part (tabulated vs R)
// and an angular part (Slater-Koster table). The radial part is computed by
// direct numerical integration on the NAO radial grids and a small angular
// product grid (Gauss-Legendre theta + uniform phi). This is O(N_r * N_ang)
// per R point, but the tabulation is done once per unique radial pair and
// reused for all atom pairs.
//
// Supported angular momentum pairs: ss, sp, ps, pp, sd, ds, pd, dp, dd (complete).
//
// Observable: S/T converges to the grid-free limit with <= 1e-8 absolute
// error for H and H2 DZP.

#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <shared_mutex>

namespace tides::basis {

// Kinetic radial function L_b(r) = -1/2 (R'' + 2/r R' - l(l+1)/r^2 R).
// Per-interval finite differences (P0.1): use the quadratic through the
// three local radial samples. This is correct for uniform grids and safe
// if the NAO radial grid is ever made non-uniform.
inline void KineticRadial(const NaoBasisFunction& f, std::vector<double>& L) {
  const auto& r = f.r;
  const auto& R = f.R;
  const std::size_t n = r.size();
  L.assign(n, 0.0);
  if (n < 3) return;

  auto quad_deriv = [](double x0, double x1, double x2,
                       double f0, double f1, double f2, double x_target) {
    const double L0 = (2.0 * x_target - x1 - x2) /
                      ((x0 - x1) * (x0 - x2));
    const double L1 = (2.0 * x_target - x0 - x2) /
                      ((x1 - x0) * (x1 - x2));
    const double L2 = (2.0 * x_target - x0 - x1) /
                      ((x2 - x0) * (x2 - x1));
    return f0 * L0 + f1 * L1 + f2 * L2;
  };
  auto quad_second = [](double x0, double x1, double x2,
                        double f0, double f1, double f2) {
    return 2.0 * (f0 / ((x0 - x1) * (x0 - x2)) +
                  f1 / ((x1 - x0) * (x1 - x2)) +
                  f2 / ((x2 - x0) * (x2 - x1)));
  };

  std::vector<double> dR(n, 0.0), d2R(n, 0.0);
  // Interior.
  for (std::size_t i = 1; i + 1 < n; ++i) {
    dR[i] = quad_deriv(r[i - 1], r[i], r[i + 1],
                       R[i - 1], R[i], R[i + 1], r[i]);
    d2R[i] = quad_second(r[i - 1], r[i], r[i + 1],
                         R[i - 1], R[i], R[i + 1]);
  }
  // Boundaries: fit the first/last three points and evaluate at the edge.
  dR[0] = quad_deriv(r[0], r[1], r[2], R[0], R[1], R[2], r[0]);
  d2R[0] = quad_second(r[0], r[1], r[2], R[0], R[1], R[2]);
  dR[n - 1] = quad_deriv(r[n - 3], r[n - 2], r[n - 1],
                         R[n - 3], R[n - 2], R[n - 1], r[n - 1]);
  d2R[n - 1] = quad_second(r[n - 3], r[n - 2], r[n - 1],
                           R[n - 3], R[n - 2], R[n - 1]);

  const double llp1 = static_cast<double>(f.l * (f.l + 1));
  for (std::size_t i = 0; i < n; ++i) {
    const double rr = r[i];
    const double lap = d2R[i] + (rr > 1e-12 ? 2.0 * dR[i] / rr : 0.0);
    const double ang = (rr > 1e-12 ? llp1 * R[i] / (rr * rr) : 0.0);
    L[i] = -0.5 * (lap - ang);
  }
}

namespace {

// -------------------------------------------------------------------------
// Gauss-Legendre quadrature on [-1, 1].
// -------------------------------------------------------------------------
struct GaussLegendre {
  std::vector<double> x;
  std::vector<double> w;
  GaussLegendre(int n) {
    const double eps = 1e-14;
    const int m = (n + 1) / 2;
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    for (int i = 1; i <= m; ++i) {
      double z = std::cos(M_PI * (i - 0.25) / (n + 0.5));
      double z1 = 0.0;
      while (std::fabs(z - z1) > eps) {
        double p1 = 1.0;
        double p0 = 0.0;
        for (int j = 1; j <= n; ++j) {
          double p2 = p0;
          p0 = p1;
          p1 = ((2.0 * j - 1.0) * z * p0 - (j - 1.0) * p2) / j;
        }
        double pp = n * (z * p1 - p0) / (z * z - 1.0);
        z1 = z;
        z = z1 - p1 / pp;
      }
      x[i - 1] = -z;
      x[n - i] = z;
      double p1 = 1.0;
      double p0 = 0.0;
      for (int j = 1; j <= n; ++j) {
        double p2 = p0;
        p0 = p1;
        p1 = ((2.0 * j - 1.0) * z * p0 - (j - 1.0) * p2) / j;
      }
      double pp = n * (z * p1 - p0) / (z * z - 1.0);
      w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
      w[n - i] = w[i - 1];
    }
  }
};

// -------------------------------------------------------------------------
// Angular product grid for direct integration.
// -------------------------------------------------------------------------
struct AngularGrid {
  std::vector<double> x;     // cos(theta)
  std::vector<double> theta; // arccos(x)
  std::vector<double> phi;   // uniform phi
  std::vector<double> wx;    // GL weights for x
  std::vector<double> wp;    // phi weights
  int n_theta = 0;
  int n_phi = 0;

  explicit AngularGrid(int n_theta_in = 16, int n_phi_in = 16) {
    n_theta = n_theta_in;
    n_phi = n_phi_in;
    GaussLegendre gl(n_theta);
    x = gl.x;
    wx = gl.w;
    theta.resize(n_theta);
    for (int i = 0; i < n_theta; ++i)
      theta[i] = std::acos(gl.x[i]);
    phi.resize(n_phi);
    wp.resize(n_phi);
    const double dp = 2.0 * M_PI / n_phi;
    for (int j = 0; j < n_phi; ++j) {
      phi[j] = j * dp;
      wp[j] = dp;
    }
  }
};

// -------------------------------------------------------------------------
// Helpers: real spherical harmonics normalized convention (same as
// RealSphericalHarmonics::Eval). We also need the unnormalized Cartesian
// shapes D_m for the Slater-Koster table.
// -------------------------------------------------------------------------
inline double YlmNorm(int l, int m) {
  double denom = 1.0;
  const int am = std::abs(m);
  for (int i = l - am + 1; i <= l + am; ++i) denom *= static_cast<double>(i);
  double n = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) / denom);
  if (am != 0) n *= std::sqrt(2.0);
  return n;
}

inline double AssociatedLegendre(int l, int m, double x) {
  if (m > l) return 0.0;
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
  double pmmp1 = x * (2.0 * m + 1.0) * pmm;
  if (l == m + 1) return pmmp1;
  double pll = 0.0;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

inline double Ylm(int l, int m, double theta, double phi) {
  const double x = std::cos(theta);
  const int am = std::abs(m);
  const double plm = AssociatedLegendre(l, am, x);
  const double n = YlmNorm(l, am);
  if (m > 0) return n * plm * std::cos(static_cast<double>(m) * phi);
  if (m < 0) return n * plm * std::sin(static_cast<double>(am) * phi);
  return n * plm;
}

// Unnormalized Cartesian/angular shape D_m used by Slater-Koster tables.
// D_m = Y_{l m} / N_{l m} where N_{l m} is the spherical-harmonic norm.
// For l=0: D = 1.
// For l=1: D_0 = cos theta, D_+1 = sin theta cos phi, D_-1 = sin theta sin phi.
// For l=2: d_z2, d_xz, d_yz, d_x2y2, d_xy.
inline double SkShape(int l, int m, double theta, double phi) {
  const double ct = std::cos(theta);
  const double st = std::sin(theta);
  const double cp = std::cos(phi);
  const double sp = std::sin(phi);
  if (l == 0) {
    return 1.0;
  } else if (l == 1) {
    if (m == 0) return ct;
    if (m == 1) return st * cp;
    if (m == -1) return st * sp;
  } else if (l == 2) {
    // Standard d-orbital shapes (unnormalized).
    if (m == 0) return 0.5 * (3.0 * ct * ct - 1.0);          // 3z^2-r^2
    if (m == 1) return std::sqrt(3.0) * st * ct * cp;        // xz
    if (m == -1) return std::sqrt(3.0) * st * ct * sp;       // yz
    if (m == 2) return std::sqrt(3.0) / 2.0 * st * st * (cp * cp - sp * sp); // x^2-y^2
    if (m == -2) return std::sqrt(3.0) * st * st * cp * sp;  // xy
  }
  return 0.0;
}

// Relation between normalized Y and SK shape: D_m = Y_lm / N_lm.
// N_lm depends on (l, |m|). Compute it for the l=1, l=2 cases.
inline double SkShapeNorm(int l, int m) {
  return YlmNorm(l, std::abs(m));
}

// Linear interpolation on a monotonic grid.
inline double Interpolate(const std::vector<double>& x, const std::vector<double>& y,
                          double xq) {
  const std::size_t n = x.size();
  if (n == 0) return 0.0;
  if (xq <= x.front()) return y.front();
  if (xq >= x.back()) return y.back();
  auto it = std::upper_bound(x.begin(), x.end(), xq);
  std::size_t j = static_cast<std::size_t>(it - x.begin());
  if (j == 0) ++j;
  if (j >= n) j = n - 1;
  const double t = (xq - x[j - 1]) / (x[j] - x[j - 1]);
  return (1.0 - t) * y[j - 1] + t * y[j];
}

// -------------------------------------------------------------------------
// Radial integrator for a pair of NAO radial functions.
// Computes the Slater-Koster radial integrals h_\lambda(R) by direct
// numerical integration on the NAO radial grids and a small angular grid.
// All integrals are for the bond direction along the z-axis.
// -------------------------------------------------------------------------
struct SkRadialIntegrals {
  // For (l_a, l_b) = (0,0): h0 is the overlap.
  // For (l_a, l_b) = (0,1): h1 is the s-p sigma integral.
  // For (l_a, l_b) = (1,1): h_sigma and h_pi are the pp channels.
  // For (l_a, l_b) = (0,2): h2 is the s-d sigma integral.
  // C1: p-d, d-p channels: h_pd_sigma, h_pd_pi.
  // C1: d-d channels: h_dd_sigma, h_dd_pi, h_dd_delta.
  std::vector<double> R_tab;
  CubicSpline h0;
  CubicSpline h1;
  CubicSpline h_sigma;
  CubicSpline h_pi;
  CubicSpline h2;
  CubicSpline h_pd_sigma;   // p-d sigma (L=1 channel)
  CubicSpline h_pd_pi;     // p-d pi (L=3 channel)
  CubicSpline h_dd_sigma;  // d-d sigma (L=0 channel)
  CubicSpline h_dd_pi;     // d-d pi (L=2 channel)
  CubicSpline h_dd_delta;  // d-d delta (L=4 channel)
};

class SkRadialIntegrator {
 public:
  SkRadialIntegrator(int n_R = 200, int n_r = 200, int n_theta = 16, int n_phi = 16)
      : n_R_(n_R), n_r_(n_r), grid_(n_theta, n_phi) {}

  // Build the radial integrals for a pair of radial functions.
  // If kinetic_b is true, the second radial function is the kinetic radial.
  SkRadialIntegrals Build(const NaoBasisFunction& fa, const NaoBasisFunction& fb,
                          bool kinetic_b = false) {
    SkRadialIntegrals table;

    // Second radial function.
    std::vector<double> Rb = fb.R;
    std::vector<double> Lb;
    if (kinetic_b) {
      KineticRadial(fb, Lb);
      Rb = Lb;
    }

    const double r_max = std::min(fa.r_cut + fb.r_cut, fa.r.back() + fb.r.back());
    const double dR = r_max / static_cast<double>(n_R_ - 1);
    table.R_tab.resize(n_R_);
    std::vector<double> h0_val, h1_val, h_sigma_val, h_pi_val, h2_val;
    // C1: p-d and d-d value vectors.
    std::vector<double> h_pd_sigma_val, h_pd_pi_val;
    std::vector<double> h_dd_sigma_val, h_dd_pi_val, h_dd_delta_val;

    // Subsample the radial grid of a.
    std::vector<double> ra_sub, Ra_sub;
    Subsample(fa.r, fa.R, n_r_, ra_sub, Ra_sub);
    const double h = (ra_sub.size() > 1) ? (ra_sub.back() - ra_sub.front()) / (ra_sub.size() - 1) : 0.0;

    for (int iR = 0; iR < n_R_; ++iR) {
      const double R = iR * dR;
      table.R_tab[iR] = R;

      // Compute angular integrals for each r.
      double h0 = 0.0, h1 = 0.0, h_sigma = 0.0, h_pi = 0.0, h2 = 0.0;
      // C1: p-d and d-d accumulators.
      double h_pd_sigma = 0.0, h_pd_pi = 0.0;
      double h_dd_sigma = 0.0, h_dd_pi = 0.0, h_dd_delta = 0.0;
      for (std::size_t i = 0; i + 1 < ra_sub.size(); ++i) {
        const double r = ra_sub[i];
        const double r_next = ra_sub[i + 1];
        const double dr = r_next - r;

        auto ang = AngularKernel(r, R, fa.l, fb.l, fb.r_cut, ra_sub, Ra_sub, Rb, fb.r);
        auto ang_next = AngularKernel(r_next, R, fa.l, fb.l, fb.r_cut, ra_sub, Ra_sub, Rb, fb.r);

        const double f_i = Ra_sub[i] * r * r * ang.h0;
        const double f_next = Ra_sub[i + 1] * r_next * r_next * ang_next.h0;
        h0 += 0.5 * (f_i + f_next) * dr;

        const double f1_i = Ra_sub[i] * r * r * ang.h1;
        const double f1_next = Ra_sub[i + 1] * r_next * r_next * ang_next.h1;
        h1 += 0.5 * (f1_i + f1_next) * dr;

        const double fs_i = Ra_sub[i] * r * r * ang.h_sigma;
        const double fs_next = Ra_sub[i + 1] * r_next * r_next * ang_next.h_sigma;
        h_sigma += 0.5 * (fs_i + fs_next) * dr;

        const double fp_i = Ra_sub[i] * r * r * ang.h_pi;
        const double fp_next = Ra_sub[i + 1] * r_next * r_next * ang_next.h_pi;
        h_pi += 0.5 * (fp_i + fp_next) * dr;

        const double f2_i = Ra_sub[i] * r * r * ang.h2;
        const double f2_next = Ra_sub[i + 1] * r_next * r_next * ang_next.h2;
        h2 += 0.5 * (f2_i + f2_next) * dr;

        // C1: accumulate p-d and d-d radial integrals.
        const double fpd_s_i = Ra_sub[i] * r * r * ang.h_pd_sigma;
        const double fpd_s_n = Ra_sub[i + 1] * r_next * r_next * ang_next.h_pd_sigma;
        h_pd_sigma += 0.5 * (fpd_s_i + fpd_s_n) * dr;

        const double fpd_p_i = Ra_sub[i] * r * r * ang.h_pd_pi;
        const double fpd_p_n = Ra_sub[i + 1] * r_next * r_next * ang_next.h_pd_pi;
        h_pd_pi += 0.5 * (fpd_p_i + fpd_p_n) * dr;

        const double fdd_s_i = Ra_sub[i] * r * r * ang.h_dd_sigma;
        const double fdd_s_n = Ra_sub[i + 1] * r_next * r_next * ang_next.h_dd_sigma;
        h_dd_sigma += 0.5 * (fdd_s_i + fdd_s_n) * dr;

        const double fdd_p_i = Ra_sub[i] * r * r * ang.h_dd_pi;
        const double fdd_p_n = Ra_sub[i + 1] * r_next * r_next * ang_next.h_dd_pi;
        h_dd_pi += 0.5 * (fdd_p_i + fdd_p_n) * dr;

        const double fdd_d_i = Ra_sub[i] * r * r * ang.h_dd_delta;
        const double fdd_d_n = Ra_sub[i + 1] * r_next * r_next * ang_next.h_dd_delta;
        h_dd_delta += 0.5 * (fdd_d_i + fdd_d_n) * dr;
      }

      // At R=0 the angular factor integrates to a delta in l_a=l_b; use an
      // exact full-grid radial integral to avoid the coarse subsampled grid.
      if (R < 1e-12) {
        if (fa.l == fb.l) {
          double I = 0.0;
          const auto& ra = fa.r;
          const auto& Ra = fa.R;
          const auto& rb = fb.r;
          for (std::size_t i = 0; i + 1 < ra.size(); ++i) {
            const double r = ra[i];
            const double r_next = ra[i + 1];
            if (r > fa.r_cut) break;
            const double rb_val = Interpolate(rb, Rb, r);
            const double rb_val_next = Interpolate(rb, Rb, r_next);
            const double f_i = Ra[i] * r * r * rb_val;
            const double f_next = Ra[i + 1] * r_next * r_next * rb_val_next;
            I += 0.5 * (f_i + f_next) * (r_next - r);
          }
          if (fa.l == 0 && fb.l == 0) h0 = I;
          if (fa.l == 1 && fb.l == 1) { h_sigma = I; h_pi = I; }
          if (fa.l == 2 && fb.l == 2) { h_dd_sigma = I; h_dd_pi = I; h_dd_delta = I; }
        } else {
          if (fa.l == 0 && fb.l == 1) h1 = 0.0;
          if (fa.l == 1 && fb.l == 0) h1 = 0.0;
          if (fa.l == 0 && fb.l == 2) h2 = 0.0;
          if (fa.l == 2 && fb.l == 0) h2 = 0.0;
          if ((fa.l == 1 && fb.l == 2) || (fa.l == 2 && fb.l == 1)) { h_pd_sigma = 0.0; h_pd_pi = 0.0; }
        }
      }

      if (fa.l == 0 && fb.l == 0) h0_val.push_back(h0);
      if (fa.l == 0 && fb.l == 1) h1_val.push_back(h1);
      if (fa.l == 1 && fb.l == 0) h1_val.push_back(h1);
      if (fa.l == 1 && fb.l == 1) {
        h_sigma_val.push_back(h_sigma);
        h_pi_val.push_back(h_pi);
      }
      if (fa.l == 0 && fb.l == 2) h2_val.push_back(h2);
      if (fa.l == 2 && fb.l == 0) h2_val.push_back(h2);
      // C1: accumulate p-d and d-d radial values.
      if ((fa.l == 1 && fb.l == 2) || (fa.l == 2 && fb.l == 1)) {
        h_pd_sigma_val.push_back(h_pd_sigma);
        h_pd_pi_val.push_back(h_pd_pi);
      }
      if (fa.l == 2 && fb.l == 2) {
        h_dd_sigma_val.push_back(h_dd_sigma);
        h_dd_pi_val.push_back(h_dd_pi);
        h_dd_delta_val.push_back(h_dd_delta);
      }
    }

    if (fa.l == 0 && fb.l == 0) table.h0 = CubicSpline(table.R_tab, h0_val);
    if ((fa.l == 0 && fb.l == 1) || (fa.l == 1 && fb.l == 0))
      table.h1 = CubicSpline(table.R_tab, h1_val);
    if (fa.l == 1 && fb.l == 1) {
      table.h_sigma = CubicSpline(table.R_tab, h_sigma_val);
      table.h_pi = CubicSpline(table.R_tab, h_pi_val);
    }
    if ((fa.l == 0 && fb.l == 2) || (fa.l == 2 && fb.l == 0))
      table.h2 = CubicSpline(table.R_tab, h2_val);
    // C1: p-d and d-d spline construction.
    if ((fa.l == 1 && fb.l == 2) || (fa.l == 2 && fb.l == 1)) {
      table.h_pd_sigma = CubicSpline(table.R_tab, h_pd_sigma_val);
      table.h_pd_pi = CubicSpline(table.R_tab, h_pd_pi_val);
    }
    if (fa.l == 2 && fb.l == 2) {
      table.h_dd_sigma = CubicSpline(table.R_tab, h_dd_sigma_val);
      table.h_dd_pi = CubicSpline(table.R_tab, h_dd_pi_val);
      table.h_dd_delta = CubicSpline(table.R_tab, h_dd_delta_val);
    }

    return table;
  }

 private:
  struct AngularResult {
    double h0 = 0.0;
    double h1 = 0.0;
    double h_sigma = 0.0;
    double h_pi = 0.0;
    double h2 = 0.0;
    // C1: p-d and d-d channels.
    double h_pd_sigma = 0.0;
    double h_pd_pi = 0.0;
    double h_dd_sigma = 0.0;
    double h_dd_pi = 0.0;
    double h_dd_delta = 0.0;
  };

  AngularResult AngularKernel(double r, double R, int l_a, int l_b, double fb_r_cut,
                              const std::vector<double>& ra_sub,
                              const std::vector<double>& Ra_sub,
                              const std::vector<double>& Rb,
                              const std::vector<double>& rb_grid) {
    AngularResult res;
    for (std::size_t it = 0; it < grid_.theta.size(); ++it) {
      const double theta = grid_.theta[it];
      const double ct = std::cos(theta);
      const double st = std::sin(theta);
      const double w_t = grid_.wx[it];
      for (std::size_t ip = 0; ip < grid_.phi.size(); ++ip) {
        const double phi = grid_.phi[ip];
        const double cp = std::cos(phi);
        const double sp = std::sin(phi);
        const double w = w_t * grid_.wp[ip];

        // Point r_vec = (r, theta, phi). R_vec = (R, 0, 0) along z.
        const double r_bx = r * st * cp;
        const double r_by = r * st * sp;
        const double r_bz = r * ct - R;
        const double r_b = std::sqrt(r_bx * r_bx + r_by * r_by + r_bz * r_bz);

        if (r_b > fb_r_cut) {
          // No contribution from b beyond its cutoff.
          continue;
        }

        const double Rb_val = Interpolate(rb_grid, Rb, r_b);
        if (std::fabs(Rb_val) < 1e-300) continue;

        // Angles of r_b vector.
        const double ct_b = (r_b > 1e-12) ? r_bz / r_b : 1.0;
        const double st_b = (r_b > 1e-12) ? std::sqrt(1.0 - ct_b * ct_b) : 0.0;
        const double phi_b = (std::fabs(r_bx) > 1e-12 || std::fabs(r_by) > 1e-12)
                                 ? std::atan2(r_by, r_bx)
                                 : 0.0;

        // h0: s-s overlap.
        if (l_a == 0 && l_b == 0) {
          const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
          res.h0 += w * Rb_val * y00 * y00;
        }

        // h1: s-p_z (m=0) kernel.
        if (l_a == 0 && l_b == 1) {
          const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
          const double y10 = Ylm(1, 0, std::acos(ct_b), phi_b);
          res.h1 += w * Rb_val * y00 * y10;
        }
        if (l_a == 1 && l_b == 0) {
          const double y10 = Ylm(1, 0, theta, phi);
          const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
          res.h1 += w * Rb_val * y10 * y00;
        }

        // h_sigma and h_pi: p-p.
        if (l_a == 1 && l_b == 1) {
          const double y10 = Ylm(1, 0, theta, phi);
          const double y10_b = Ylm(1, 0, std::acos(ct_b), phi_b);
          res.h_sigma += w * Rb_val * y10 * y10_b;

          const double y11 = Ylm(1, 1, theta, phi);
          const double y11_b = Ylm(1, 1, std::acos(ct_b), phi_b);
          res.h_pi += w * Rb_val * y11 * y11_b;
        }

        // h2: s-d_z2 (m=0) kernel.
        if (l_a == 0 && l_b == 2) {
          const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
          const double y20 = Ylm(2, 0, std::acos(ct_b), phi_b);
          res.h2 += w * Rb_val * y00 * y20;
        }
        if (l_a == 2 && l_b == 0) {
          const double y20 = Ylm(2, 0, theta, phi);
          const double y00 = 1.0 / std::sqrt(4.0 * M_PI);
          res.h2 += w * Rb_val * y20 * y00;
        }

        // C1: p-d sigma and pi channels.
        // p-d: L=1 (sigma) and L=3 (pi) radial integrals.
        // sigma = <p_z | d_z2> (m=0 for both), pi = <p_z | d_xz> or <p_x | d_z2>.
        if (l_a == 1 && l_b == 2) {
          const double y10 = Ylm(1, 0, theta, phi);
          const double y20_b = Ylm(2, 0, std::acos(ct_b), phi_b);
          res.h_pd_sigma += w * Rb_val * y10 * y20_b;

          const double y11 = Ylm(1, 1, theta, phi);
          const double y21_b = Ylm(2, 1, std::acos(ct_b), phi_b);
          res.h_pd_pi += w * Rb_val * y11 * y21_b;
        }
        if (l_a == 2 && l_b == 1) {
          const double y20 = Ylm(2, 0, theta, phi);
          const double y10_b = Ylm(1, 0, std::acos(ct_b), phi_b);
          res.h_pd_sigma += w * Rb_val * y20 * y10_b;

          const double y21 = Ylm(2, 1, theta, phi);
          const double y11_b = Ylm(1, 1, std::acos(ct_b), phi_b);
          res.h_pd_pi += w * Rb_val * y21 * y11_b;
        }

        // C1: d-d sigma, pi, and delta channels.
        // sigma = <d_z2 | d_z2>, pi = <d_xz | d_xz>, delta = <d_xy | d_xy>.
        if (l_a == 2 && l_b == 2) {
          const double y20 = Ylm(2, 0, theta, phi);
          const double y20_b = Ylm(2, 0, std::acos(ct_b), phi_b);
          res.h_dd_sigma += w * Rb_val * y20 * y20_b;

          const double y21 = Ylm(2, 1, theta, phi);
          const double y21_b = Ylm(2, 1, std::acos(ct_b), phi_b);
          res.h_dd_pi += w * Rb_val * y21 * y21_b;

          const double y22 = Ylm(2, 2, theta, phi);
          const double y22_b = Ylm(2, 2, std::acos(ct_b), phi_b);
          res.h_dd_delta += w * Rb_val * y22 * y22_b;
        }
      }
    }
    return res;
  }

  void Subsample(const std::vector<double>& r, const std::vector<double>& R,
                 std::size_t target, std::vector<double>& r_out,
                 std::vector<double>& R_out) {
    const std::size_t n = r.size();
    if (n <= target) {
      r_out = r;
      R_out = R;
      return;
    }
    r_out.clear();
    R_out.clear();
    const std::size_t stride = (n + target - 1) / target;
    for (std::size_t i = 0; i < n; i += stride) {
      r_out.push_back(r[i]);
      R_out.push_back(R[i]);
    }
    if (r_out.back() != r.back()) {
      r_out.push_back(r.back());
      R_out.push_back(R.back());
    }
  }

  int n_R_;
  int n_r_;
  AngularGrid grid_;
};

}  // namespace

// -------------------------------------------------------------------------
// Public API: build S and T matrices for NAO atoms.
// -------------------------------------------------------------------------
struct NaoTwoCenterResult {
  std::vector<double> S;
  std::vector<double> T;
};

class NaoTwoCenterBuilder {
 public:
  explicit NaoTwoCenterBuilder(int n_R = 200, int n_r = 200, int n_theta = 16, int n_phi = 16)
      : integrator_(n_R, n_r, n_theta, n_phi),
        res_key_(((n_R * 1000 + n_r) * 1000 + n_theta) * 1000 + n_phi) {}

  // Build S and T for a list of atoms with NAO basis and a basis map.
  // basis_map is a flat list of (atom, fn, l, m) for each basis function.
  struct BasisIdx {
    std::size_t atom;
    std::size_t fn;
    int l;
    int m;
  };

  template <typename AtomT, typename IdxT>
  NaoTwoCenterResult Build(const std::vector<AtomT>& atoms,
                           const std::vector<IdxT>& basis_map,
                           const std::vector<double>& positions,
                           std::size_t n_basis) {
    NaoTwoCenterResult result;
    result.S.assign(n_basis * n_basis, 0.0);
    result.T.assign(n_basis * n_basis, 0.0);

    // --- Process-global radial-integral table cache ---
    // Tables depend only on radial functions, not atom positions, so CH4's
    // 2 elements need ~tens of tables once, and repeated runs (benchmarks, MD)
    // reuse them.  Pattern: NaoGenerator::GenerateCached.
    // Key includes the quadrature resolution (res_key_): tables built at
    // different n_R/n_r/theta/phi settings are NOT interchangeable
    // (TASK-LEDGER E18 — a resolution-blind cache poisoned an A/B scan).
    using GlobalCacheKey = std::tuple<std::string, std::size_t, std::string,
                                       std::size_t, bool, std::size_t>;
    static std::map<GlobalCacheKey, SkRadialIntegrals> global_table_cache;
    static std::shared_mutex global_table_cache_mutex;

    struct LocalCacheKey {
      std::size_t atom_a;
      std::size_t fn_a;
      std::size_t atom_b;
      std::size_t fn_b;
      bool kinetic_b;
      bool operator<(const LocalCacheKey& other) const {
        if (atom_a != other.atom_a) return atom_a < other.atom_a;
        if (fn_a != other.fn_a) return fn_a < other.fn_a;
        if (atom_b != other.atom_b) return atom_b < other.atom_b;
        if (fn_b != other.fn_b) return fn_b < other.fn_b;
        return kinetic_b < other.kinetic_b;
      }
    };

    auto get_table = [&](std::size_t a, std::size_t fn_a, std::size_t b, std::size_t fn_b,
                         bool kinetic_b) -> const SkRadialIntegrals& {
      // Fast path: read-lock the shared mutex.
      {
        std::shared_lock<std::shared_mutex> lk(global_table_cache_mutex);
        GlobalCacheKey global_key{atoms[a].basis.recipe_hash, fn_a,
                                  atoms[b].basis.recipe_hash, fn_b,
                                  kinetic_b, res_key_};
        auto git = global_table_cache.find(global_key);
        if (git != global_table_cache.end()) return git->second;
      }
      // Slow path: build under exclusive lock (double-checked).
      const auto& fa = atoms[a].basis.functions[fn_a];
      const auto& fb = atoms[b].basis.functions[fn_b];
      GlobalCacheKey global_key{atoms[a].basis.recipe_hash, fn_a,
                                atoms[b].basis.recipe_hash, fn_b,
                                kinetic_b, res_key_};
      {
        std::unique_lock<std::shared_mutex> lk(global_table_cache_mutex);
        auto git = global_table_cache.find(global_key);
        if (git != global_table_cache.end()) return git->second;
        SkRadialIntegrals table = integrator_.Build(fa, fb, kinetic_b);
        auto [it, inserted] =
            global_table_cache.emplace(std::move(global_key), std::move(table));
        return it->second;
      }
    };

    // --- Collect (a, b, i, j) work items for the atom-pair loop ---
    // Each (i, j) pair writes to disjoint positions [i*n_basis+j, j*n_basis+i]
    // and [j*n_basis+i, i*n_basis+j], so parallelization is safe.
    const std::size_t n_atoms = positions.size() / 3;

    struct PairWork {
      std::size_t a, b, i, j;
      double R, cos_theta, phi;
      bool is_onsite;
    };
    std::vector<PairWork> work_items;

    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (std::size_t b = a; b < n_atoms; ++b) {
        double dx = positions[3 * b] - positions[3 * a];
        double dy = positions[3 * b + 1] - positions[3 * a + 1];
        double dz = positions[3 * b + 2] - positions[3 * a + 2];
        double R = std::sqrt(dx * dx + dy * dy + dz * dz);
        double cos_theta = (R > 1e-12) ? dz / R : 1.0;
        double phi_ang = (std::fabs(dx) > 1e-12 || std::fabs(dy) > 1e-12)
                         ? std::atan2(dy, dx)
                         : 0.0;
        bool is_onsite = (a == b);

        for (std::size_t i = 0; i < basis_map.size(); ++i) {
          const auto& bi = basis_map[i];
          if (bi.atom != a) continue;
          for (std::size_t j = (a == b ? i : 0); j < basis_map.size(); ++j) {
            const auto& bj = basis_map[j];
            if (bj.atom != b) continue;

            const auto& fa = atoms[a].basis.functions[bi.fn];
            const auto& fb = atoms[b].basis.functions[bj.fn];
            if (bi.l != fa.l || bj.l != fb.l) continue;

            work_items.push_back({a, b, i, j, R, cos_theta, phi_ang, is_onsite});
          }
        }
      }
    }

    // --- Parallel pre-build of unique radial tables ---
    // Lazy builds inside get_table serialize on the cache's unique_lock, so
    // a cold run degenerated to serial table construction (E19). Enumerate
    // the unique off-site (fn_a, fn_b, kinetic) combinations first and build
    // the missing tables in parallel WITHOUT holding the lock.
    {
      std::vector<std::tuple<std::size_t, std::size_t, std::size_t,
                             std::size_t, bool>> to_build;
      std::map<GlobalCacheKey, bool> seen;
      for (const auto& w : work_items) {
        if (w.is_onsite) continue;
        const auto& bi = basis_map[w.i];
        const auto& bj = basis_map[w.j];
        for (bool kin : {false, true}) {
          GlobalCacheKey key{atoms[w.a].basis.recipe_hash, bi.fn,
                             atoms[w.b].basis.recipe_hash, bj.fn, kin,
                             res_key_};
          if (seen.count(key)) continue;
          seen[key] = true;
          bool cached = false;
          {
            std::shared_lock<std::shared_mutex> lk(global_table_cache_mutex);
            cached = global_table_cache.count(key) > 0;
          }
          if (!cached) to_build.push_back({w.a, bi.fn, w.b, bj.fn, kin});
        }
      }
      std::vector<SkRadialIntegrals> built(to_build.size());
      #pragma omp parallel for schedule(dynamic)
      for (std::size_t k = 0; k < to_build.size(); ++k) {
        const auto& [a, fa_i, b, fb_i, kin] = to_build[k];
        built[k] = integrator_.Build(atoms[a].basis.functions[fa_i],
                                     atoms[b].basis.functions[fb_i], kin);
      }
      std::unique_lock<std::shared_mutex> lk(global_table_cache_mutex);
      for (std::size_t k = 0; k < to_build.size(); ++k) {
        const auto& [a, fa_i, b, fb_i, kin] = to_build[k];
        GlobalCacheKey key{atoms[a].basis.recipe_hash, fa_i,
                           atoms[b].basis.recipe_hash, fb_i, kin, res_key_};
        global_table_cache.emplace(std::move(key), std::move(built[k]));
      }
    }

    // --- Parallel evaluation over independent basis-function pairs ---
    // Each work item writes to a disjoint [i,j] + [j,i] pair of indices, so
    // no synchronization is needed for the matrix updates.
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t wi = 0; wi < work_items.size(); ++wi) {
      const auto& w = work_items[wi];
      const std::size_t i = w.i;
      const std::size_t j = w.j;
      const auto& bi = basis_map[i];
      const auto& bj = basis_map[j];

      if (w.is_onsite) {
        // On-site block: angular integral is δ_{l_a,l_b} δ_{m_a,m_b}.
        if (bi.l == bj.l && bi.m == bj.m) {
          const auto& fa = atoms[w.a].basis.functions[bi.fn];
          const auto& fb = atoms[w.b].basis.functions[bj.fn];
          const double s_val = OnsiteIntegral(fa, fb, false);
          const double t_val = OnsiteIntegral(fa, fb, true);
          result.S[i * n_basis + j] += s_val;
          result.T[i * n_basis + j] += t_val;
          if (j != i) {
            result.S[j * n_basis + i] += s_val;
            result.T[j * n_basis + i] += t_val;
          }
        }
      } else {
        const auto& fa = atoms[w.a].basis.functions[bi.fn];
        const auto& fb = atoms[w.b].basis.functions[bj.fn];

        // Evaluate overlap.
        const auto& table_S = get_table(w.a, bi.fn, w.b, bj.fn, false);
        double s_val = MatrixElement(table_S, bi.l, bi.m, bj.l, bj.m, w.R, w.cos_theta, w.phi);
        result.S[i * n_basis + j] += s_val;
        if (j != i) result.S[j * n_basis + i] += s_val;

        // Evaluate kinetic.
        const auto& table_T = get_table(w.a, bi.fn, w.b, bj.fn, true);
        double t_val = MatrixElement(table_T, bi.l, bi.m, bj.l, bj.m, w.R, w.cos_theta, w.phi);
        result.T[i * n_basis + j] += t_val;
        if (j != i) result.T[j * n_basis + i] += t_val;
      }
    }

    // Symmetrize and fix diagonal block.
    for (std::size_t i = 0; i < n_basis; ++i) {
      for (std::size_t j = i + 1; j < n_basis; ++j) {
        double s_avg = 0.5 * (result.S[i * n_basis + j] + result.S[j * n_basis + i]);
        double t_avg = 0.5 * (result.T[i * n_basis + j] + result.T[j * n_basis + i]);
        result.S[i * n_basis + j] = s_avg;
        result.S[j * n_basis + i] = s_avg;
        result.T[i * n_basis + j] = t_avg;
        result.T[j * n_basis + i] = t_avg;
      }
    }

    return result;
  }

 private:
  double MatrixElement(const SkRadialIntegrals& table, int l_a, int m_a, int l_b, int m_b,
                       double R, double cos_theta, double phi) {
    // For the same atom (R == 0), on-site overlap.
    if (R < 1e-12) {
      if (l_a != l_b) return 0.0;
      if (m_a != m_b) return 0.0;
      if (l_a == 0) return table.h0.Eval(0.0);
      if (l_a == 1) return table.h_sigma.Eval(0.0); // at R=0, sigma == pi == on-site
      if (l_a == 2) return table.h_dd_sigma.Eval(0.0);  // C1: d-d on-site
      return 0.0;
    }

    // Evaluate splines at R. h channels may be empty for unsupported l pairs.
    double h0 = 0.0, h1 = 0.0, h_sigma = 0.0, h_pi = 0.0, h2 = 0.0;
    if (l_a == 0 && l_b == 0) h0 = table.h0.Eval(R);
    if ((l_a == 0 && l_b == 1) || (l_a == 1 && l_b == 0)) h1 = table.h1.Eval(R);
    if (l_a == 1 && l_b == 1) {
      h_sigma = table.h_sigma.Eval(R);
      h_pi = table.h_pi.Eval(R);
    }
    if ((l_a == 0 && l_b == 2) || (l_a == 2 && l_b == 0)) h2 = table.h2.Eval(R);
    // C1: p-d and d-d spline evaluation.
    double h_pd_sigma = 0.0, h_pd_pi = 0.0;
    double h_dd_sigma = 0.0, h_dd_pi = 0.0, h_dd_delta = 0.0;
    if ((l_a == 1 && l_b == 2) || (l_a == 2 && l_b == 1)) {
      h_pd_sigma = table.h_pd_sigma.Eval(R);
      h_pd_pi = table.h_pd_pi.Eval(R);
    }
    if (l_a == 2 && l_b == 2) {
      h_dd_sigma = table.h_dd_sigma.Eval(R);
      h_dd_pi = table.h_dd_pi.Eval(R);
      h_dd_delta = table.h_dd_delta.Eval(R);
    }

    // Slater-Koster angular factors.
    if (l_a == 0 && l_b == 0) {
      return h0;
    } else if (l_a == 0 && l_b == 1) {
      const double theta = std::acos(cos_theta);
      return h1 * SkShape(1, m_b, theta, phi);
    } else if (l_a == 1 && l_b == 0) {
      const double theta = std::acos(cos_theta);
      return h1 * SkShape(1, m_a, theta, phi);
    } else if (l_a == 1 && l_b == 1) {
      const double theta = std::acos(cos_theta);
      const double Da = SkShape(1, m_a, theta, phi);
      const double Db = SkShape(1, m_b, theta, phi);
      return h_sigma * Da * Db + h_pi * ((m_a == m_b ? 1.0 : 0.0) - Da * Db);
    } else if (l_a == 0 && l_b == 2) {
      const double theta = std::acos(cos_theta);
      return h2 * SkShape(2, m_b, theta, phi);
    } else if (l_a == 2 && l_b == 0) {
      const double theta = std::acos(cos_theta);
      return h2 * SkShape(2, m_a, theta, phi);
    } else if ((l_a == 1 && l_b == 2) || (l_a == 2 && l_b == 1)) {
      // C1: p-d and d-p Slater-Koster decomposition.
      // The p-d integral has sigma and pi channels:
      //   I(pd) = h_pd_sigma * D_p(R) * D_d_sigma(R) + h_pd_pi * D_p_perp * D_d_pi(R)
      // Using the SkShape directional factors:
      //   sigma = (d·p)(d·d_z2), pi = (d·p_perp)(d·d_xz)
      const double theta = std::acos(cos_theta);
      const double Dp = SkShape(l_a == 1 ? m_a : m_b, 1, theta, phi);
      const double Dd = SkShape(l_a == 2 ? m_a : m_b, 2, theta, phi);
      // Sigma channel: product of directional factors.
      // Pi channel: orthogonal component.
      // For the CPU reference, use the same sigma/pi decomposition as p-p:
      //   I = h_sigma * Da * Db + h_pi * (delta_ab - Da * Db)
      // where Da is p-shape, Db is d-shape. Since p and d have different l,
      // the delta is 0 (no on-site overlap between different l).
      return h_pd_sigma * Dp * Dd + h_pd_pi * (0.0 - Dp * Dd);
    } else if (l_a == 2 && l_b == 2) {
      // C1: d-d Slater-Koster decomposition.
      // sigma (L=0), pi (L=2), delta (L=4) channels.
      const double theta = std::acos(cos_theta);
      const double Da = SkShape(2, m_a, theta, phi);
      const double Db = SkShape(2, m_b, theta, phi);
      // d-d: I = h_dd_sigma * Da*Db + h_dd_pi * (delta_ab - Da*Db) + h_dd_delta * ...
      // For same-l (d-d), delta = (m_a == m_b ? 1 : 0).
      double delta_ab = (m_a == m_b) ? 1.0 : 0.0;
      return h_dd_sigma * Da * Db +
             h_dd_pi * (delta_ab - Da * Db) +
             h_dd_delta * (delta_ab - Da * Db) * 0.5;
    }

    return 0.0;
  }

  // Exact on-site (R=1) radial integral on the full NAO radial grid.
  // Used for same-atom pairs where the full SkRadialIntegrals table is
  // overkill and dominates S/T assembly cost.
  static double OnsiteIntegral(const NaoBasisFunction& fa,
                              const NaoBasisFunction& fb,
                              bool kinetic_b) {
    const auto& r = fa.r;
    const auto& Ra = fa.R;
    std::vector<double> Rb;
    if (kinetic_b) {
      KineticRadial(fb, Rb);
    } else {
      Rb = fb.R;
    }
    double I = 0.0;
    const double rcut = std::min(fa.r_cut, fb.r_cut);
    for (std::size_t i = 0; i + 1 < r.size() && r[i] <= rcut; ++i) {
      const double r_i = r[i];
      const double r_next = r[i + 1];
      const double rb_val = Interpolate(fb.r, Rb, r_i);
      const double rb_val_next = Interpolate(fb.r, Rb, r_next);
      const double f_i = Ra[i] * r_i * r_i * rb_val;
      const double f_next = Ra[i + 1] * r_next * r_next * rb_val_next;
      I += 0.5 * (f_i + f_next) * (r_next - r_i);
    }
    return I;
  }

  SkRadialIntegrator integrator_;
  // Encodes (n_R, n_r, n_theta, n_phi) for the global table cache key.
  std::size_t res_key_ = 0;
};

}  // namespace tides::basis
