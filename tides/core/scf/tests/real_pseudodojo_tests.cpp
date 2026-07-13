// Real PseudoDojo pseudopotential SCF tests.
//
// These tests load actual SG15/PseudoDojo ONCV PBE SR pseudopotential files
// from external/pseudopotentials/pseudodojo-pbe-sr/ and run NaoDriver SCF
// with them. This validates the full PP pipeline: UPF2 parsing → V_local
// interpolation → KB projectors → SCF convergence.
//
// If the PP files are not present, all tests are SKIPPED (return 77).

#include "scf/nao_driver.hpp"
#include "basis/pseudo/pp_loader.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::NaoDriver;
using tides::scf::NaoDriverResult;
using tides::basis::Pseudopotential;
using tides::basis::PpLoader;

int Fail(const std::string& msg) {
  std::cerr << "real_pseudodojo_tests: FAIL — " << msg << '\n';
  return 1;
}

int Skip(const std::string& msg) {
  std::cout << "real_pseudodojo_tests: SKIP — " << msg << '\n';
  return 77;
}

bool pp_available(const std::string& el) {
  return PpLoader::IsAvailable(el);
}

// Test 1: H atom with real PseudoDojo H PP.
// H has Z_valence=1, 2 KB projectors (l=0, index 0 and 1).
// SCF should converge and produce a finite energy.
int TestHWithRealPP() {
  std::cout << "\n--- Test 1: H atom with real PseudoDojo PP ---\n";

  if (!pp_available("H")) {
    return Skip("H_ONCV_PBE_SR.upf not found");
  }

  auto pp_result = PpLoader::Load("H");
  if (!pp_result.ok()) {
    return Fail("Failed to load H PP: " + pp_result.status().message());
  }

  const auto& pp = pp_result.value();
  std::cout << "  PP loaded: element=" << pp.element
            << " Z_val=" << pp.Z_valence
            << " r_grid=" << pp.r_grid.size()
            << " channels=" << pp.channels.size() << "\n";

  if (pp.Z_valence != 1) {
    return Fail("H PP Z_valence should be 1, got " + std::to_string(pp.Z_valence));
  }
  if (pp.r_grid.empty()) {
    return Fail("H PP has empty radial grid");
  }
  if (pp.v_local.empty()) {
    return Fail("H PP has empty local potential");
  }

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};
  std::vector<Pseudopotential> pps = {pp};

  auto result = NaoDriver::Run(Z, pos, 0.4, 4.0, 100, 1e-6, &pps);
  if (!result.scf.converged) {
    return Fail("H PP SCF did not converge");
  }

  std::cout << "  SCF converged in " << result.scf.n_iterations << " iters\n";
  std::cout << "  Energy = " << result.scf.energy << " Ha\n";

  if (!std::isfinite(result.scf.energy)) {
    return Fail("H PP SCF energy is not finite");
  }
  if (result.scf.energy > 0.0) {
    return Fail("H PP SCF energy should be negative, got " + std::to_string(result.scf.energy));
  }

  std::cout << "  PASS\n";
  return 0;
}

// Test 2: Si atom with real PseudoDojo Si PP.
// Si has Z_valence=4, 4 KB projectors (2 for l=0, 2 for l=1).
int TestSiWithRealPP() {
  std::cout << "\n--- Test 2: Si atom with real PseudoDojo PP ---\n";

  if (!pp_available("Si")) {
    return Skip("Si_ONCV_PBE_SR.upf not found");
  }

  auto pp_result = PpLoader::Load("Si");
  if (!pp_result.ok()) {
    return Fail("Failed to load Si PP: " + pp_result.status().message());
  }

  const auto& pp = pp_result.value();
  std::cout << "  PP loaded: element=" << pp.element
            << " Z_val=" << pp.Z_valence
            << " r_grid=" << pp.r_grid.size()
            << " channels=" << pp.channels.size() << "\n";

  if (pp.Z_valence != 4) {
    return Fail("Si PP Z_valence should be 4, got " + std::to_string(pp.Z_valence));
  }
  if (pp.channels.empty()) {
    return Fail("Si PP has no KB channels");
  }

  // Print channel info.
  for (std::size_t i = 0; i < pp.channels.size(); ++i) {
    std::cout << "    channel " << i << ": l=" << pp.channels[i].l
              << " n_projectors=" << pp.channels[i].projectors.size() << "\n";
  }

  std::vector<int> Z = {14};
  std::vector<double> pos = {0.0, 0.0, 0.0};
  std::vector<Pseudopotential> pps = {pp};

  auto result = NaoDriver::Run(Z, pos, 0.3, 6.0, 100, 1e-6, &pps);
  std::cout << "  SCF converged=" << result.scf.converged
            << " iters=" << result.scf.n_iterations
            << " energy=" << result.scf.energy << "\n";

  if (!std::isfinite(result.scf.energy)) {
    return Fail("Si PP SCF energy is not finite");
  }

  // Si with 4 valence electrons should have a negative energy.
  // We don't have a tight reference, but it should be well below 0.
  if (result.scf.energy > 0.0) {
    return Fail("Si PP SCF energy should be negative, got " + std::to_string(result.scf.energy));
  }

  std::cout << "  PASS\n";
  return 0;
}

// Test 3: H2 molecule with real PseudoDojo H PPs.
// Two H atoms at 1.4 Bohr separation, each with the real PP.
int TestH2WithRealPP() {
  std::cout << "\n--- Test 3: H2 molecule with real PseudoDojo PPs ---\n";

  if (!pp_available("H")) {
    return Skip("H_ONCV_PBE_SR.upf not found");
  }

  auto pp_result = PpLoader::Load("H");
  if (!pp_result.ok()) {
    return Fail("Failed to load H PP: " + pp_result.status().message());
  }

  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0};
  std::vector<Pseudopotential> pps = {pp_result.value(), pp_result.value()};

  auto result = NaoDriver::Run(Z, pos, 0.4, 4.0, 100, 1e-6, &pps);
  std::cout << "  SCF converged=" << result.scf.converged
            << " iters=" << result.scf.n_iterations
            << " energy=" << result.scf.energy << "\n";

  if (!result.scf.converged) {
    return Fail("H2 PP SCF did not converge");
  }
  if (!std::isfinite(result.scf.energy)) {
    return Fail("H2 PP SCF energy is not finite");
  }
  if (result.scf.energy > 0.0) {
    return Fail("H2 PP SCF energy should be negative");
  }

  // H2 with PBE PP should give roughly -0.5 to -1.5 Ha (no tight reference).
  // The key check is convergence + finite + negative.
  std::cout << "  PASS\n";
  return 0;
}

// Test 4: PP parsing validation — verify all available PPs parse correctly.
int TestPPParsing() {
  std::cout << "\n--- Test 4: Parse all available PseudoDojo PPs ---\n";

  const std::vector<std::string> elements = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"
  };

  int parsed = 0;
  int skipped = 0;
  int failed = 0;

  for (const auto& el : elements) {
    if (!pp_available(el)) {
      ++skipped;
      continue;
    }
    auto result = PpLoader::Load(el);
    if (!result.ok()) {
      std::cout << "  " << el << ": FAIL — " << result.status().message() << "\n";
      ++failed;
      continue;
    }
    const auto& pp = result.value();
    if (pp.r_grid.empty() || pp.v_local.empty() || pp.Z_valence <= 0) {
      std::cout << "  " << el << ": FAIL — empty grid/v_local or bad Z_val\n";
      ++failed;
      continue;
    }
    std::cout << "  " << el << ": OK (Z_val=" << pp.Z_valence
              << ", r_grid=" << pp.r_grid.size()
              << ", channels=" << pp.channels.size() << ")\n";
    ++parsed;
  }

  std::cout << "  Parsed: " << parsed << ", Skipped: " << skipped
            << ", Failed: " << failed << "\n";

  if (failed > 0) {
    return Fail(std::to_string(failed) + " PP(s) failed to parse");
  }
  if (parsed == 0) {
    return Skip("No PseudoDojo PP files found");
  }

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  Real PseudoDojo PP SCF Tests                                ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  int skips = 0;

  int r;
  r = TestPPParsing();    if (r == 77) ++skips; else failures += r;
  r = TestHWithRealPP();  if (r == 77) ++skips; else failures += r;
  r = TestSiWithRealPP(); if (r == 77) ++skips; else failures += r;
  r = TestH2WithRealPP(); if (r == 77) ++skips; else failures += r;

  std::cout << "\n=== Summary ===\n";
  if (skips > 0 && failures == 0) {
    std::cout << "ALL AVAILABLE TESTS PASSED (" << skips << " skipped — PP files not found)\n";
    return 77;
  }
  if (failures == 0) {
    std::cout << "ALL REAL PSEUDODOJO TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
