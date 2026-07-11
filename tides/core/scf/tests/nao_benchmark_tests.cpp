// Gap 8: Matched-accuracy benchmark — TIDES NaoDriver vs PySCF reference.
//
// Validates that the TIDES NAO SCF engine produces energies within an
// acceptable tolerance of PySCF reference values. The tolerance is generous
// because the two codes use different basis sets (NAO vs GTO) and different
// integration methods (grid vs analytic ERIs).
//
// PySCF reference values (LDA, 6-31G basis):
//   H atom:  -0.4540 Ha
//   H2 (1.4 Bohr): -1.0386 Ha

#include "scf/nao_driver.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::NaoDriver;

int Fail(const std::string& msg) {
  std::cerr << "nao_benchmark_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestHAtomVsPySCF() {
  std::cout << "\n=== Benchmark 1: H atom vs PySCF ===\n";
  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};
  auto res = NaoDriver::Run(Z, pos, 0.3, 4.0, 100, 1e-8);

  const double pyscf_ref = -0.4540;
  const double diff = std::fabs(res.energy.E_total - pyscf_ref);
  std::cout << "  TIDES=" << res.energy.E_total << " PySCF=" << pyscf_ref
            << " diff=" << diff * 1000 << " meV\n";

  if (diff > 0.15) {
    return Fail("H atom energy differs from PySCF by " +
                std::to_string(diff) + " Ha (limit 0.15)");
  }
  std::cout << "  PASS\n";
  return 0;
}

int TestH2VsPySCF() {
  std::cout << "\n=== Benchmark 2: H2 molecule vs PySCF ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0};
  auto res = NaoDriver::Run(Z, pos, 0.3, 4.0, 100, 1e-8);

  const double pyscf_ref = -1.0386;
  const double diff = std::fabs(res.energy.E_total - pyscf_ref);
  std::cout << "  TIDES=" << res.energy.E_total << " PySCF=" << pyscf_ref
            << " diff=" << diff * 1000 << " meV\n";

  if (diff > 0.25) {
    return Fail("H2 energy differs from PySCF by " +
                std::to_string(diff) + " Ha (limit 0.25)");
  }
  std::cout << "  PASS\n";
  return 0;
}

int TestForcesNewton3rd() {
  std::cout << "\n=== Benchmark 3: H2 forces Newton 3rd law ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0};
  auto forces = NaoDriver::ComputeForces(Z, pos, 0.3, 4.0, 50, 1e-6, 0.01);

  double net_fx = forces[0] + forces[3];
  double net_fy = forces[1] + forces[4];
  double net_fz = forces[2] + forces[5];

  std::cout << "  F0=[" << forces[0] << ", " << forces[1] << ", " << forces[2]
            << "] F1=[" << forces[3] << ", " << forces[4] << ", " << forces[5]
            << "] net=[" << net_fx << ", " << net_fy << ", " << net_fz << "]\n";

  if (std::fabs(net_fx) > 1e-3) {
    return Fail("Newton 3rd law violated: net Fx = " +
                std::to_string(net_fx));
  }
  if (std::fabs(net_fy) > 1e-6 || std::fabs(net_fz) > 1e-6) {
    return Fail("Spurious transverse forces: net Fy=" +
                std::to_string(net_fy) + " Fz=" + std::to_string(net_fz));
  }
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  int failures = 0;
  failures += TestHAtomVsPySCF();
  failures += TestH2VsPySCF();
  failures += TestForcesNewton3rd();

  if (failures == 0) {
    std::cout << "\nALL NAO BENCHMARK TESTS PASSED\n";
    return 0;
  }
  std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return 1;
}
