// Quick optimization benchmark: CH4, H2O with B3LYP at grid_h=0.3
// Measures SCF iterations and wall time to verify OPT-1 through OPT-5 impact.
#include "scf/nao_driver.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using tides::scf::NaoDriver;
using tides::scf::NaoDriverResult;
using tides::grid::xc::HostXcSpec;
using tides::grid::xc::XcFunctionalId;
using tides::grid::xc::XcFamily;

int main() {
  struct Mol {
    const char* name;
    std::vector<int> Z;
    std::vector<double> pos;
  };

  // CH4: C at origin, 4 H at tetrahedral positions (Bohr)
  double ch = 1.089 * 1.889726125; // C-H bond in Bohr
  std::vector<Mol> mols = {
    {"CH4", {6, 1, 1, 1, 1},
     {0, 0, 0,
      ch, ch, ch,
      -ch, -ch, ch,
      -ch, ch, -ch,
      ch, -ch, -ch}},
    {"H2O", {8, 1, 1},
     {0, 0, 0,
      0.9572 * 1.889726125, 0, 0.4676 * 1.889726125,
      -0.9572 * 1.889726125, 0, 0.4676 * 1.889726125}},
  };

  // B3LYP spec
  HostXcSpec b3lyp_spec;
  b3lyp_spec.family = XcFamily::kHybrid;
  b3lyp_spec.id = XcFunctionalId::kB3lypLocal;
  b3lyp_spec.exchange_fraction = 0.20;

  std::cout << "\n=== TIDES Optimization Benchmark (OPT-1..5) ===\n";
  std::cout << "Molecule  Iters  Converged  Wall(s)  E_total  build_H_ms  xc_ms\n";

  for (const auto& mol : mols) {
    auto t0 = std::chrono::steady_clock::now();
    auto res = NaoDriver::Run(mol.Z, mol.pos, 0.3, 6.0, 100, 1e-7,
                               nullptr, b3lyp_spec, 1, 0,
                               false, 0.0, false, false, false, false,
                               false, false, false, false, false,
                               {1,1,1}, false, false,
                               nullptr, false, false);
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    auto& bh = res.build_H_timings;
    std::cout << mol.name << "  "
              << bh.n_iterations << "  "
              << (res.scf.converged ? "YES" : "NO") << "  "
              << wall << "  "
              << res.energy.E_total << "  "
              << bh.total_ms << "  "
              << bh.xc_eval_ms << "\n";
  }

  // gpu4pyscf reference (from saved data)
  std::cout << "\n=== gpu4pyscf reference (saved) ===\n";
  std::cout << "CH4: 4 iters, 1.625s, E=-40.4877\n";
  std::cout << "H2O: 6 iters, 0.653s, E=-76.3583\n";
  std::cout << "\nSpeedup = gpu4pyscf_wall / TIDES_wall\n";

  return 0;
}
