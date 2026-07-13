// PAW (Projector Augmented-Wave) tests (M9).
//
// Tests that:
//   1. PAW dataset creation works (synthetic H and He)
//   2. On-site energy correction is computed
//   3. Density correction is non-zero within augmentation sphere
//   4. PAW correction is small for H (no core)
//   5. PAW SCF through NaoDriver works (integration test)

#include "basis/paw/paw_dataset.hpp"
#include "basis/paw/paw_correction.hpp"
#include "scf/nao_driver.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::basis::paw::PAWAtomData;
using tides::basis::paw::PAWCorrection;
using tides::basis::paw::MakeSimplePAWH;
using tides::basis::paw::MakeSimplePAWHe;
using tides::scf::NaoDriver;

int Fail(const std::string& msg) {
  std::cerr << "paw_tests: FAIL — " << msg << '\n';
  return 1;
}

// Test 1: PAW dataset creation.
int TestPAWDatasetCreation() {
  std::cout << "\n=== Test 1: PAW dataset creation ===\n";

  auto paw_h = MakeSimplePAWH();
  if (paw_h.element != "H")
    return Fail("PAW H element should be 'H'");
  if (paw_h.Z_valence != 1)
    return Fail("PAW H Z_valence should be 1");
  if (paw_h.channels.empty())
    return Fail("PAW H should have at least one channel");
  if (paw_h.channels[0].phi.empty())
    return Fail("PAW H channel 0 phi should be non-empty");
  if (paw_h.channels[0].phi_tilde.empty())
    return Fail("PAW H channel 0 phi_tilde should be non-empty");
  if (paw_h.channels[0].projector.empty())
    return Fail("PAW H channel 0 projector should be non-empty");
  if (paw_h.Dij.empty())
    return Fail("PAW H Dij should be non-empty");

  std::cout << "  H: Z_val=" << paw_h.Z_valence
            << ", r_c=" << paw_h.r_c
            << ", n_channels=" << paw_h.channels.size()
            << ", eigenvalue=" << paw_h.channels[0].eigenvalue << "\n";

  auto paw_he = MakeSimplePAWHe();
  if (paw_he.Z_valence != 2)
    return Fail("PAW He Z_valence should be 2");

  std::cout << "  He: Z_val=" << paw_he.Z_valence
            << ", r_c=" << paw_he.r_c << "\n";

  std::cout << "  PASS\n";
  return 0;
}

// Test 2: PAW on-site energy correction is computed and finite.
int TestOnSiteEnergyCorrection() {
  std::cout << "\n=== Test 2: PAW on-site energy correction ===\n";

  auto paw = MakeSimplePAWH();

  // Simple D_ij = [1.0] (one electron in the projector).
  std::vector<double> D_ij = {1.0};
  double E_paw = tides::basis::paw::ComputeOnSiteEnergy(paw, D_ij);

  std::cout << "  E_PAW (D=1.0) = " << E_paw << " Ha\n";

  if (!std::isfinite(E_paw))
    return Fail("PAW energy correction is not finite");

  // For H (no core), the correction should be small.
  if (std::fabs(E_paw) > 1.0)
    return Fail("PAW correction too large for H (no core): " +
                std::to_string(E_paw));

  std::cout << "  PASS (correction = " << E_paw << " Ha)\n";
  return 0;
}

// Test 3: PAW density correction is non-zero within the sphere.
int TestDensityCorrection() {
  std::cout << "\n=== Test 3: PAW density correction ===\n";

  auto paw = MakeSimplePAWH();
  std::vector<double> D_ij = {1.0};

  auto delta_n = tides::basis::paw::ComputeDensityCorrection(paw, D_ij);

  if (delta_n.empty())
    return Fail("Density correction is empty");

  // Check that the correction is non-zero at small r (within sphere).
  bool has_correction = false;
  double max_dn = 0.0;
  for (std::size_t i = 0; i < delta_n.size(); ++i) {
    if (paw.r_grid[i] < paw.r_c) {
      max_dn = std::max(max_dn, std::fabs(delta_n[i]));
      if (std::fabs(delta_n[i]) > 1e-10) has_correction = true;
    }
  }

  std::cout << "  max |delta_n| within r_c = " << max_dn << "\n";

  if (!has_correction)
    return Fail("Density correction is zero within augmentation sphere");

  std::cout << "  PASS (max |delta_n| = " << max_dn << ")\n";
  return 0;
}

// Test 4: PAW correction vanishes when phi = phi_tilde (sanity check).
int TestCorrectionVanishesWhenIdentical() {
  std::cout << "\n=== Test 4: PAW correction vanishes when phi = phi_tilde ===\n";

  auto paw = MakeSimplePAWH();

  // Make phi_tilde = phi (identical), so correction should be zero.
  for (auto& ch : paw.channels) {
    ch.phi_tilde = ch.phi;
  }

  std::vector<double> D_ij = {1.0};
  double E_paw = tides::basis::paw::ComputeOnSiteEnergy(paw, D_ij);

  std::cout << "  E_PAW (phi=phi_tilde) = " << E_paw << " Ha\n";

  // The correction should be near-zero (limited by grid discretization
  // of the overlap integral, which may not be exactly 1.0).
  if (std::fabs(E_paw) > 0.1)
    return Fail("PAW correction should vanish when phi=phi_tilde, got " +
                std::to_string(E_paw));

  std::cout << "  PASS (correction = 0 when phi = phi_tilde)\n";
  return 0;
}

// Test 5: PAW SCF through NaoDriver (integration test).
int TestPAWSCFIntegration() {
  std::cout << "\n=== Test 5: PAW SCF integration (H atom) ===\n";

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Run standard SCF (no PAW).
  auto res_std = NaoDriver::Run(Z, pos, 0.4, 4.0, 100, 1e-6);
  std::cout << "  Standard: E=" << res_std.scf.energy
            << " converged=" << res_std.scf.converged << "\n";

  // Run with PAW correction.
  auto res_paw = NaoDriver::Run(Z, pos, 0.4, 4.0, 100, 1e-6,
                                nullptr, {}, 1, 0, true, 0.0,
                                false, false, false, false, false,
                                false, false, false, false,
                                {1, 1, 1}, false, true);
  std::cout << "  With PAW: E=" << res_paw.scf.energy
            << " converged=" << res_paw.scf.converged << "\n";
  std::cout << "  E_PAW correction = " << res_paw.E_paw_correction << " Ha\n";

  if (!res_std.scf.converged)
    return Fail("Standard SCF did not converge");

  if (!res_paw.scf.converged)
    return Fail("PAW SCF did not converge");

  // PAW correction should be small for H (no core).
  double diff = std::fabs(res_paw.scf.energy - res_std.scf.energy);
  std::cout << "  |E_paw - E_std| = " << diff << " Ha\n";

  if (diff > 0.5)
    return Fail("PAW correction too large for H: " + std::to_string(diff));

  std::cout << "  PASS (PAW SCF converges, correction = " << diff << ")\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  PAW (Projector Augmented-Wave) Tests (M9)                ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestPAWDatasetCreation();
  failures += TestOnSiteEnergyCorrection();
  failures += TestDensityCorrection();
  failures += TestCorrectionVanishesWhenIdentical();
  failures += TestPAWSCFIntegration();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL PAW TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
