#pragma once

// C6: Wire HSE screening into NaoDriver SCF loop.
//
// HSE06 (Heyd-Scuseria-Ernzerhof) is a screened hybrid functional that uses
// short-range exact exchange with a screened Coulomb interaction. The key
// parameter is the screening length omega (default 0.11 Bohr^-1 for HSE06).
//
// The screened exchange energy:
//   E_x_SR = -1/2 * sum_{ij} (ij | erfc(omega*r12)/r12 | ij)
// where erfc is the complementary error function and omega is the screening
// parameter.
//
// For the SCF loop, the HSE screening modifies the exchange potential:
//   V_x_HSE = alpha * V_x_SR(omega) + (1-alpha) * V_x_PBE_SR(omega) + V_xc_PBE_LR(omega)
// where alpha=0.25 for HSE06.
//
// This module provides the screened Coulomb operator and the HSE exchange
// energy/voltage for integration into the NaoDriver SCF loop.
//
// Observable (C6): HSE screening produces finite exchange energy with
// correct omega dependence (erfc screening).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string>
#include <vector>

namespace tides::hybrids {

// HSE06 parameters (standard values).
struct HSEParameters {
  double alpha = 0.25;       // exact exchange fraction
  double omega = 0.11;       // screening length (Bohr^-1)
  std::string functional = "PBE";  // underlying semi-local functional
};

// C6: HSE screened exchange for SCF integration.
class HSEScreenedExchange {
 public:
  // Compute the screened Coulomb interaction erfc(omega*r)/r for a pair
  // of basis functions at distance R.
  // This is the key kernel for the short-range exchange.
  static double ScreenedCoulomb(double R, double omega) {
    if (R < 1e-12) return 2.0 * omega / std::sqrt(M_PI);  // erfc(0)/r limit
    return std::erfc(omega * R) / R;
  }

  // Compute the unscreened Coulomb interaction 1/r (for reference).
  static double BareCoulomb(double R) {
    if (R < 1e-12) return 0.0;  // singular
    return 1.0 / R;
  }

  // Compute the long-range Coulomb interaction erf(omega*r)/r.
  static double LongRangeCoulomb(double R, double omega) {
    if (R < 1e-12) return 0.0;  // erf(0) = 0
    return std::erf(omega * R) / R;
  }

  // Build the short-range exchange matrix V_x_SR in the AO basis.
  // For each pair of basis functions (i,j), compute:
  //   V_x_SR[i,j] = -alpha * sum_k P[k,k] * (ik | erfc(omega*r12)/r12 | jk)
  // For the CPU reference, we use a simplified grid-based evaluation.
  //
  //   n:         matrix dimension (basis size)
  //   P:         density matrix (n x n, row-major)
  //   positions: basis function centers (3*n, x,y,z per function)
  //   params:    HSE parameters (alpha, omega)
  static std::vector<double> BuildShortRangeExchange(
      std::size_t n,
      const std::vector<double>& P,
      const std::vector<double>& positions,
      const HSEParameters& params) {
    std::vector<double> V_x(n * n, 0.0);

    // Simplified: use the diagonal of P (atomic populations) and the
    // screened Coulomb interaction between basis function centers.
    // Full implementation would use 4-center ERIs with erfc screening.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double dx = positions[3 * i] - positions[3 * j];
        double dy = positions[3 * i + 1] - positions[3 * j + 1];
        double dz = positions[3 * i + 2] - positions[3 * j + 2];
        double R = std::sqrt(dx * dx + dy * dy + dz * dz);

        double sr_coulomb = ScreenedCoulomb(R, params.omega);

        // Exchange contribution: -alpha * P[j,j] * (ii|sr|jj) / 2
        // (simplified: use diagonal of P as weight).
        // V_x is the exchange potential: V_x[i,j] = alpha * P[j,j] * (ii|sr|jj)
        // The exchange energy is E_x = -0.5 * Tr(P @ V_x) (negative).
        V_x[i * n + j] = params.alpha * P[j * n + j] * sr_coulomb;
      }
    }

    // Symmetrize.
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        double avg = 0.5 * (V_x[i * n + j] + V_x[j * n + i]);
        V_x[i * n + j] = avg;
        V_x[j * n + i] = avg;
      }

    return V_x;
  }

  // Compute the short-range exchange energy.
  // E_x_SR = -alpha/2 * Tr(P @ V_x_SR)
  static double ShortRangeExchangeEnergy(
      std::size_t n,
      const std::vector<double>& P,
      const std::vector<double>& V_x_sr) {
    double trace = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        trace += P[i * n + j] * V_x_sr[j * n + i];
    return -0.5 * trace;
  }

  // Compute the HSE correction to the total energy.
  // E_HSE = E_PBE + alpha * (E_x_SR_exact - E_x_SR_PBE)
  // For the CPU reference, this is the screened exchange energy.
  static double HSEEnergyCorrection(
      std::size_t n,
      const std::vector<double>& P,
      const std::vector<double>& positions,
      const HSEParameters& params) {
    auto V_x_sr = BuildShortRangeExchange(n, P, positions, params);
    return ShortRangeExchangeEnergy(n, P, V_x_sr);
  }

  // Verify the screening behavior: at omega=0, SR exchange = full exchange;
  // at omega->inf, SR exchange = 0.
  static bool VerifyScreening(double R) {
    // omega=0: erfc(0)=1, so screened = bare.
    double sr_0 = ScreenedCoulomb(R, 0.0);
    double bare = BareCoulomb(R);
    if (std::fabs(sr_0 - bare) > 1e-10) return false;

    // omega=large: erfc(large)~0, so screened ~ 0.
    double sr_large = ScreenedCoulomb(R, 100.0);
    if (std::fabs(sr_large) > 1e-6) return false;

    return true;
  }
};

}  // namespace tides::hybrids
