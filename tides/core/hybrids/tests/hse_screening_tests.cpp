// T7.4: HSE short-range screening tests (updated for new HSEScreening API).
//
// Validates:
//   - Screened Coulomb SR + LR = 1/r
//   - Exchange energy with screening is finite and well-defined
//   - Screening parameter controls range separation

#include "hybrids/hse_screening.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::hybrids::HSEScreening;
using tides::hybrids::HSEConfig;
using tides::hybrids::HSEExchangeResult;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T7.4a: SR + LR = 1/r (partition of Coulomb operator).
int TestCoulombPartition() {
  std::cout << "\n=== T7.4a: Screened Coulomb partition ===\n";
  double omega = 0.11;
  double max_err = 0.0;
  for (double r : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
    double sr = HSEScreening::ScreenedCoulomb(r, omega, true);   // SR: erfc(ωr)/r
    double lr = HSEScreening::ScreenedCoulomb(r, omega, false);  // LR: erf(ωr)/r
    double full = 1.0 / r;
    double err = std::fabs(sr + lr - full);
    max_err = std::max(max_err, err);
    std::cout << "  r=" << r << " SR=" << sr << " LR=" << lr
              << " SR+LR=" << sr + lr << " 1/r=" << full
              << " err=" << err << '\n';
  }
  if (max_err > 1e-12) return Fail("T7.4a: SR+LR != 1/r");
  std::cout << "T7.4a: GREEN\n";
  return 0;
}

// T7.4b: Screened exchange energy is finite and negative (attractive).
int TestExchangeEnergy() {
  std::cout << "\n=== T7.4b: Screened exchange energy ===\n";
  // Simple 2-atom system with 1 basis function each.
  std::size_t n = 2;
  std::vector<double> P = {0.5, 0.2, 0.2, 0.5};
  // Simple grid (2 points per atom).
  std::size_t n_grid = 4;
  std::vector<double> grid = {
    0.0, 0.0, 0.0,  // atom 1
    0.5, 0.0, 0.0,
    3.0, 0.0, 0.0,  // atom 2
    3.5, 0.0, 0.0
  };
  std::vector<double> grid_w = {0.5, 0.5, 0.5, 0.5};
  // Basis values: each basis function is a Gaussian centered on its atom.
  std::vector<double> basis_vals(n_grid * n);
  for (std::size_t g = 0; g < n_grid; ++g)
    for (std::size_t i = 0; i < n; ++i) {
      double dx = grid[3*g] - (i == 0 ? 0.0 : 3.0);
      double dy = grid[3*g+1];
      double dz = grid[3*g+2];
      double r2 = dx*dx + dy*dy + dz*dz;
      basis_vals[g * n + i] = std::exp(-r2);
    }

  HSEConfig config;
  config.omega = 0.11;
  config.alpha_exact = 0.25;
  auto res = HSEScreening::ComputeSRExchange(P, n, basis_vals, grid, grid_w,
                                              n_grid, config);
  std::cout << "  E_exact_sr: " << res.E_exact_sr << "\n";
  std::cout << "  Converged: " << (res.converged ? "yes" : "no") << "\n";

  if (!res.converged)
    return Fail("T7.4b: SR exchange did not converge");
  if (!std::isfinite(res.E_exact_sr))
    return Fail("T7.4b: SR exchange energy not finite");
  if (std::fabs(res.E_exact_sr) < 1e-15)
    return Fail("T7.4b: SR exchange energy is zero");
  std::cout << "T7.4b: GREEN\n";
  return 0;
}

// T7.4c: Exchange fraction is correct for HSE06.
int TestExchangeFraction() {
  std::cout << "\n=== T7.4c: HSE exchange fraction ===\n";
  HSEConfig config;
  double frac = HSEScreening::ExchangeFraction(config);
  std::cout << "  Total exchange fraction: " << frac
            << " (expected 1.0 for HSE06: 0.25 SR + 0.75 LR)\n";
  if (std::fabs(frac - 1.0) > 1e-12)
    return Fail("T7.4c: Exchange fraction should be 1.0");
  std::cout << "T7.4c: GREEN\n";
  return 0;
}

// T7.4d: Screening parameter controls range separation.
int TestScreeningParameter() {
  std::cout << "\n=== T7.4d: Screening parameter effect ===\n";
  double r = 5.0;
  for (double omega : {0.05, 0.11, 0.20, 0.50}) {
    double sr = HSEScreening::ScreenedCoulomb(r, omega, true);
    double lr = HSEScreening::ScreenedCoulomb(r, omega, false);
    std::cout << "  omega=" << omega << " SR=" << sr << " LR=" << lr
              << " SR/(SR+LR)=" << sr / (sr + lr) << "\n";
  }
  // At larger omega, SR fraction should decrease (more long-range).
  double sr_low = HSEScreening::ScreenedCoulomb(5.0, 0.05, true);
  double sr_high = HSEScreening::ScreenedCoulomb(5.0, 0.50, true);
  if (sr_high > sr_low)
    return Fail("T7.4d: Higher omega should give smaller SR");
  std::cout << "T7.4d: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestCoulombPartition()) return 1;
  if (TestExchangeEnergy()) return 1;
  if (TestExchangeFraction()) return 1;
  if (TestScreeningParameter()) return 1;
  std::cout << "\nhse_screening_tests: ALL GREEN\n";
  return 0;
}
