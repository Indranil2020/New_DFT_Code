#pragma once

// AUDIT T-X1.1: Common helpers for Tier-0 XC functors.
// Shared by all functional headers. Plain arithmetic, __device__-compatible.
// Also CPU-compilable for oracle testing.

#include <cmath>

namespace tides::grid::xc::detail {

// Physical constants and thresholds mirroring libxc.
// These thresholds determine when to use asymptotic forms vs full expressions.
constexpr double kRhoMin = 1e-14;    // below this, rho ≈ 0
constexpr double kSigmaMin = 1e-30;  // below this, sigma ≈ 0
constexpr double kTauMin = 1e-20;    // below this, tau ≈ 0

// Wigner-Seitz radius: rs = (3 / (4*pi*n))^(1/3)
inline double RhoToRs(double n) {
  if (n < kRhoMin) return 1e30;
  return std::cbrt(3.0 / (4.0 * M_PI * n));
}

// Reduced gradient: s = |grad_rho| / (2 * (3*pi^2)^(1/3) * rho^(4/3))
inline double ReducedGradient(double rho, double sigma) {
  if (rho < kRhoMin) return 0.0;
  const double grad = std::sqrt(std::max(sigma, 0.0));
  const double kF = std::cbrt(3.0 * M_PI * M_PI * rho);
  return grad / (2.0 * kF * std::cbrt(rho));
}

// Spin polarization: zeta = (rho_up - rho_down) / rho
inline double SpinPolarization(double rho_up, double rho_down) {
  double rho = rho_up + rho_down;
  if (rho < kRhoMin) return 0.0;
  return (rho_up - rho_down) / rho;
}

// Fermi wavevector: kF = (3*pi^2 * rho)^(1/3)
inline double FermiWavevector(double rho) {
  if (rho < kRhoMin) return 0.0;
  return std::cbrt(3.0 * M_PI * M_PI * rho);
}

// Thomas-Fermi screening wavevector: kTF = sqrt(4*kF/pi)
inline double ThomasFermiK(double rho) {
  double kF = FermiWavevector(rho);
  return std::sqrt(4.0 * kF / M_PI);
}

}  // namespace tides::grid::xc::detail
