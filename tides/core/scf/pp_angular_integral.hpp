#pragma once

// P0.4: dense angular integration helpers for the pseudopotential semi-on-site
// correction.  The product Gauss-Legendre grid is not as optimal as a Lebedev
// grid but is trivial to generate to arbitrary order and is exact for spherical
// harmonic products up to a controlled (l, m) cutoff.

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace tides::scf {

struct AngularGridPoint {
  double theta = 0.0;   // polar angle from +z
  double phi = 0.0;     // azimuthal angle from +x
  double weight = 0.0;  // quadrature weight (summed to 4*pi)
  double x = 0.0;       // Cartesian direction
  double y = 0.0;
  double z = 0.0;
};

namespace detail {

// Evaluate the Legendre polynomial P_n(x) and its derivative P_n'(x) using the
// recurrence relation.  The derivative uses the standard identity
// (1 - x^2) P_n'(x) = n (P_{n-1}(x) - x P_n(x)).
inline void LegendreAndDerivative(int n, double x, double* P, double* dP) {
  if (n == 0) {
    *P = 1.0;
    *dP = 0.0;
    return;
  }
  double p0 = 1.0;
  double p1 = x;
  double p_prev = p0;
  double p_curr = p1;
  for (int k = 2; k <= n; ++k) {
    double p_next = ((2.0 * k - 1.0) * x * p_curr - (k - 1.0) * p_prev) / k;
    p_prev = p_curr;
    p_curr = p_next;
  }
  *P = p_curr;
  const double one_minus_x2 = 1.0 - x * x;
  if (one_minus_x2 > 1e-30) {
    *dP = n * (p_prev - x * p_curr) / one_minus_x2;
  } else {
    // At the poles the derivative is not needed for quadrature weight
    // computation (x = ±1 are never quadrature nodes for n > 0), but keep a
    // safe fallback.
    *dP = 0.0;
  }
}

// Gauss-Legendre nodes x_i and weights w_i on [-1, 1].
inline std::vector<std::pair<double, double>> GaussLegendreNodesAndWeights(int n) {
  std::vector<std::pair<double, double>> result;
  result.reserve(n);
  const double pi = std::acos(-1.0);
  for (int i = 1; i <= n; ++i) {
    // Tricomi initial guess for the i-th positive root (used for all roots).
    double x = std::cos(pi * (i - 0.25) / (n + 0.5));
    double P = 0.0, dP = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
      LegendreAndDerivative(n, x, &P, &dP);
      const double dx = P / dP;
      x -= dx;
      if (std::fabs(dx) < 1e-15) break;
    }
    LegendreAndDerivative(n, x, &P, &dP);
    const double w = 2.0 / ((1.0 - x * x) * dP * dP);
    result.emplace_back(x, w);
  }
  return result;
}

}  // namespace detail

// Build a product Gauss-Legendre / uniform-phi angular grid over the unit
// sphere.  n_theta is the number of Gauss-Legendre points in cos(theta); n_phi
// is the number of uniformly spaced azimuthal points.  The total weight sums
// to 4*pi.
inline std::vector<AngularGridPoint> BuildProductGaussLegendreAngularGrid(
    int n_theta, int n_phi) {
  const auto nodes = detail::GaussLegendreNodesAndWeights(n_theta);
  const double dphi = 2.0 * M_PI / static_cast<double>(n_phi);
  std::vector<AngularGridPoint> grid;
  grid.reserve(static_cast<std::size_t>(n_theta) * static_cast<std::size_t>(n_phi));
  for (const auto& [x, w_theta] : nodes) {
    const double theta = std::acos(x);
    const double sin_theta = std::sin(theta);
    const double z = x;
    for (int j = 0; j < n_phi; ++j) {
      const double phi = dphi * static_cast<double>(j);
      AngularGridPoint p;
      p.theta = theta;
      p.phi = phi;
      p.weight = w_theta * dphi;
      p.x = sin_theta * std::cos(phi);
      p.y = sin_theta * std::sin(phi);
      p.z = z;
      grid.push_back(p);
    }
  }
  return grid;
}

}  // namespace tides::scf
