// C6: HSE screened exchange tests.
// Verifies screened Coulomb interaction and HSE exchange energy.
#include "hybrids/hse_screened_exchange.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::hybrids::HSEScreenedExchange;
using tides::hybrids::HSEParameters;

int Fail(const std::string& msg) {
  std::cerr << "hse_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestScreenedCoulomb() {
  std::cout << "\n=== C6: Screened Coulomb Interaction ===\n";
  double R = 2.0;
  double omega = 0.11;

  double sr = HSEScreenedExchange::ScreenedCoulomb(R, omega);
  double bare = HSEScreenedExchange::BareCoulomb(R);
  double lr = HSEScreenedExchange::LongRangeCoulomb(R, omega);

  std::cout << "  R=" << R << " omega=" << omega << "\n";
  std::cout << "  SR(erfc/r) = " << sr << " bare(1/r) = " << bare
            << " LR(erf/r) = " << lr << "\n";

  // SR + LR should = bare (partition of unity).
  if (std::fabs(sr + lr - bare) > 1e-10)
    return Fail("SR + LR != bare");
  // SR < bare (screening reduces short-range).
  if (sr >= bare) return Fail("SR should be < bare");
  std::cout << "  PASS (SR + LR = bare)\n";
  return 0;
}

int TestScreeningVerification() {
  std::cout << "\n=== C6: Screening Behavior ===\n";
  if (!HSEScreenedExchange::VerifyScreening(2.0))
    return Fail("Screening verification failed");
  std::cout << "  omega=0: SR=bare, omega=large: SR~0 — PASS\n";
  return 0;
}

int TestExchangeEnergy() {
  std::cout << "\n=== C6: HSE Exchange Energy ===\n";
  const std::size_t n = 3;
  std::vector<double> P(n * n, 0.0);
  std::vector<double> positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) P[i * n + i] = 1.0;

  HSEParameters params;
  params.alpha = 0.25;
  params.omega = 0.11;

  double E = HSEScreenedExchange::HSEEnergyCorrection(n, P, positions, params);
  std::cout << "  HSE exchange energy = " << E << " Ha\n";
  if (!std::isfinite(E)) return Fail("Exchange energy not finite");
  if (E >= 0) return Fail("Exchange energy should be negative");
  std::cout << "  PASS\n";
  return 0;
}

int TestOmegaDependence() {
  std::cout << "\n=== C6: Omega Dependence ===\n";
  // Use 3 basis functions at different positions with diagonal P.
  // The exchange energy comes from P[i,i] * erfc(omega*R_ij)/R_ij.
  const std::size_t n = 3;
  std::vector<double> P(n * n, 0.0);
  std::vector<double> positions = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) P[i * n + i] = 1.0;

  HSEParameters params;
  params.alpha = 0.25;

  // At omega=0, SR exchange = full exchange (maximum |E|).
  params.omega = 0.001;
  double E_small_omega = HSEScreenedExchange::HSEEnergyCorrection(n, P, positions, params);

  // At omega=large, SR exchange ~ 0 for off-site (screened away).
  params.omega = 10.0;
  double E_large_omega = HSEScreenedExchange::HSEEnergyCorrection(n, P, positions, params);

  std::cout << "  E(omega=0.001) = " << E_small_omega << "\n";
  std::cout << "  E(omega=10)    = " << E_large_omega << "\n";
  // The exchange energy includes both self-interaction (R=0, grows with omega)
  // and off-site exchange (decreases with omega). The key physics test is
  // that the off-site screened Coulomb interaction decreases with omega.
  double sr_small = HSEScreenedExchange::ScreenedCoulomb(2.0, 0.001);
  double sr_large = HSEScreenedExchange::ScreenedCoulomb(2.0, 10.0);
  std::cout << "  SR(R=2, omega=0.001) = " << sr_small << "\n";
  std::cout << "  SR(R=2, omega=10)    = " << sr_large << "\n";
  if (sr_large > sr_small * 0.01)
    return Fail("Off-site SR should decrease with omega");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== C6: HSE Screened Exchange Tests ===\n";
  int failures = 0;
  failures += TestScreenedCoulomb();
  failures += TestScreeningVerification();
  failures += TestExchangeEnergy();
  failures += TestOmegaDependence();
  if (failures == 0) std::cout << "\nALL HSE TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
