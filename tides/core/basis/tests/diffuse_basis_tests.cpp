// Diffuse basis set augmentation tests (M10).
//
// Tests that:
//   1. Augmented DZP recipes contain diffuse functions (larger r_cut)
//   2. The augmented basis has more functions than standard DZP
//   3. Diffuse functions are properly normalized
//   4. SCF with diffuse basis converges (integration test via NaoDriver)

#include "basis/nao_generator.hpp"
#include "scf/nao_driver.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::basis::NaoGenerator;
using tides::basis::NaoRecipe;
using tides::basis::NaoBasis;
using tides::scf::NaoDriver;

int Fail(const std::string& msg) {
  std::cerr << "diffuse_basis_tests: FAIL — " << msg << '\n';
  return 1;
}

// Test 1: Augmented DZP recipe has more functions than standard DZP.
int TestAugmentedBasisSize() {
  std::cout << "\n=== Test 1: Augmented DZP has more functions than DZP ===\n";

  auto dzp = NaoGenerator::DzpRecipe(1, "H");
  auto aug = NaoGenerator::AugDzpRecipe(1, "H");

  std::size_t n_dzp = 0, n_aug = 0;
  for (const auto& ch : dzp.channels)
    n_dzp += ch.rcuts.size();
  for (const auto& ch : aug.channels)
    n_aug += ch.rcuts.size();

  std::cout << "  DZP channels: " << n_dzp << " zeta values\n";
  std::cout << "  Aug DZP channels: " << n_aug << " zeta values\n";

  if (n_aug <= n_dzp)
    return Fail("Augmented basis should have MORE functions than DZP, got " +
                std::to_string(n_aug) + " vs " + std::to_string(n_dzp));

  std::cout << "  PASS (augmented has " << n_aug << " > " << n_dzp << " zeta)\n";
  return 0;
}

// Test 2: Diffuse functions have larger r_cut than valence functions.
int TestDiffuseCutoffRadii() {
  std::cout << "\n=== Test 2: Diffuse functions have larger r_cut ===\n";

  auto aug = NaoGenerator::AugDzpRecipe(1, "H");

  bool has_diffuse = false;
  for (const auto& ch : aug.channels) {
    // The last rcut in each channel should be the diffuse one (largest).
    if (ch.rcuts.size() >= 3) {
      double diffuse_rc = ch.rcuts.back();
      double valence_rc = ch.rcuts[0];
      std::cout << "  l=" << ch.l << ": valence r_cut=" << valence_rc
                << ", diffuse r_cut=" << diffuse_rc << "\n";
      if (diffuse_rc <= valence_rc)
        return Fail("Diffuse r_cut should be LARGER than valence r_cut");
      has_diffuse = true;
    }
  }

  if (!has_diffuse)
    return Fail("No diffuse functions found in augmented recipe");

  std::cout << "  PASS (diffuse functions have larger r_cut)\n";
  return 0;
}

// Test 3: Generated augmented basis functions are normalized.
int TestAugmentedBasisNormalization() {
  std::cout << "\n=== Test 3: Augmented basis functions normalized ===\n";

  auto recipe = NaoGenerator::AugDzpRecipe(6, "C");
  auto basis = NaoGenerator::Generate(recipe);

  std::cout << "  C augmented basis: " << basis.functions.size() << " functions\n";
  if (basis.functions.empty())
    return Fail("No basis functions generated for C augmented DZP");

  int n_fail = 0;
  for (std::size_t i = 0; i < basis.functions.size(); ++i) {
    const auto& f = basis.functions[i];
    std::cout << "    fn " << i << ": l=" << f.l << " zeta=" << f.zeta
              << " r_cut=" << f.r_cut << " norm=" << f.norm << "\n";
    if (std::fabs(f.norm - 1.0) > 1e-6) {
      std::cerr << "    WARNING: norm = " << f.norm << " (expected 1.0)\n";
      ++n_fail;
    }
  }

  if (n_fail > 0)
    return Fail(std::to_string(n_fail) + " functions not normalized");

  std::cout << "  PASS (all " << basis.functions.size() << " functions normalized)\n";
  return 0;
}

// Test 4: SCF with diffuse basis converges (integration test).
int TestSCFWithDiffuseBasis() {
  std::cout << "\n=== Test 4: SCF with diffuse basis (H atom) ===\n";

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Run with standard DZP.
  auto res_dzp = NaoDriver::Run(Z, pos, 0.4, 4.0, 100, 1e-6);
  std::cout << "  DZP: E=" << res_dzp.scf.energy
            << " converged=" << res_dzp.scf.converged
            << " n_basis=" << res_dzp.n_basis << "\n";

  // Run with augmented DZP (diffuse functions).
  auto res_aug = NaoDriver::Run(Z, pos, 0.4, 6.0, 100, 1e-6,
                                nullptr, {}, 1, 0, true, 0.0,
                                false, false, false, false, false,
                                false, false, false, false,
                                {1, 1, 1}, true, false);
  std::cout << "  Aug DZP: E=" << res_aug.scf.energy
            << " converged=" << res_aug.scf.converged
            << " n_basis=" << res_aug.n_basis << "\n";

  if (!res_dzp.scf.converged)
    return Fail("DZP SCF did not converge");

  if (!res_aug.scf.converged)
    return Fail("Augmented DZP SCF did not converge");

  if (res_aug.n_basis <= res_dzp.n_basis)
    return Fail("Augmented basis should have more functions: " +
                std::to_string(res_aug.n_basis) + " vs " +
                std::to_string(res_dzp.n_basis));

  // Both energies should be negative (bound states).
  if (res_aug.scf.energy > 0.0)
    return Fail("Augmented DZP energy should be negative, got " +
                std::to_string(res_aug.scf.energy));

  std::cout << "  PASS (augmented SCF converges, " << res_aug.n_basis
            << " > " << res_dzp.n_basis << " basis functions)\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  Diffuse Basis Augmentation Tests (M10)                    ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestAugmentedBasisSize();
  failures += TestDiffuseCutoffRadii();
  failures += TestAugmentedBasisNormalization();
  failures += TestSCFWithDiffuseBasis();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL DIFFUSE BASIS TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
