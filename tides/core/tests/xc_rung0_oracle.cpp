// AUDIT A5: Rung-0 oracle sweep — TIDES LDA XC vs libxc reference.
//
// Sweeps over a lattice of density values (rho) and compares the TIDES
// internal LDA XC evaluation (LdaXC::EpsXC, LdaXC::VXC) against libxc
// LDA_X + LDA_C_PW (the oracle).
//
// Tolerance: tolerances.yaml wp3.xc_grid.lda_energy = 1e-12.
//
// This test catches:
// - Wrong PW92 parameters
// - Analytic derivative bugs (audit B2)
// - Threshold/branching errors in the XC code

#include "basis/atomgen/lda_xc.hpp"
#include "grid/libxc_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::atomgen::LdaXC;
using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_LDA_X;
using tides::grid::kLibxc_LDA_C_PW;

int Fail(const std::string& msg) {
  std::cerr << "xc_rung0_oracle: FAIL — " << msg << '\n';
  return 1;
}

int TestLdaRung0() {
  std::cout << "\n=== XC Rung-0 Oracle: LDA_X + LDA_C_PW vs libxc ===\n";

  // Density lattice: logarithmic spacing from 1e-8 to 1e2.
  // This covers the physically relevant range from vacuum to core density.
  std::vector<double> rho_pts;
  for (double log_n = -8.0; log_n <= 2.0; log_n += 0.5) {
    rho_pts.push_back(std::pow(10.0, log_n));
  }
  // Add some exact values for robustness.
  rho_pts.push_back(1.0);       // rs = 0.6204
  rho_pts.push_back(0.5);       // rs = 0.7816
  rho_pts.push_back(0.1);       // rs = 1.3354
  rho_pts.push_back(0.01);      // rs = 2.8785
  rho_pts.push_back(1e-6);      // low density

  const std::size_t np = rho_pts.size();

  // Evaluate TIDES LDA XC.
  std::vector<double> tides_eps(np), tides_vxc(np);
  for (std::size_t i = 0; i < np; ++i) {
    const double n = std::max(0.0, rho_pts[i]);
    tides_eps[i] = LdaXC::EpsXC(n, 0.0);
    tides_vxc[i] = LdaXC::VXC(n, 0.0);
  }

  // Evaluate libxc LDA_X + LDA_C_PW (the oracle).
  LibxcFunctional fx, fc;
  if (!fx.Init(kLibxc_LDA_X, XC_UNPOLARIZED)) {
    return Fail("Failed to init libxc LDA_X");
  }
  if (!fc.Init(kLibxc_LDA_C_PW, XC_UNPOLARIZED)) {
    return Fail("Failed to init libxc LDA_C_PW");
  }

  auto rx = fx.EvalLDA(rho_pts, np);
  auto rc = fc.EvalLDA(rho_pts, np);

  // Combine libxc exchange + correlation.
  std::vector<double> libxc_eps(np), libxc_vxc(np);
  for (std::size_t i = 0; i < np; ++i) {
    libxc_eps[i] = rx.eps_xc[i] + rc.eps_xc[i];
    libxc_vxc[i] = rx.vrho[i] + rc.vrho[i];
  }

  // Compare.
  const double TOL = 1e-12;  // tolerances.yaml wp3.xc_grid.lda_energy
  double max_eps_err = 0.0;
  double max_vxc_err = 0.0;
  std::size_t worst_eps_idx = 0, worst_vxc_idx = 0;

  std::cout << "  " << std::setw(12) << "rho"
            << "  " << std::setw(14) << "TIDES eps"
            << "  " << std::setw(14) << "libxc eps"
            << "  " << std::setw(12) << "eps err"
            << "  " << std::setw(14) << "TIDES vxc"
            << "  " << std::setw(14) << "libxc vxc"
            << "  " << std::setw(12) << "vxc err"
            << "\n";

  for (std::size_t i = 0; i < np; ++i) {
    double eps_err = std::fabs(tides_eps[i] - libxc_eps[i]);
    double vxc_err = std::fabs(tides_vxc[i] - libxc_vxc[i]);

    // Skip near-zero density (both should be ~0, relative error meaningless).
    if (rho_pts[i] < 1e-10) continue;

    // Use relative error for eps and vxc.
    double eps_rel = eps_err / (std::fabs(libxc_eps[i]) + 1e-30);
    double vxc_rel = vxc_err / (std::fabs(libxc_vxc[i]) + 1e-30);

    if (eps_rel > max_eps_err) {
      max_eps_err = eps_rel;
      worst_eps_idx = i;
    }
    if (vxc_rel > max_vxc_err) {
      max_vxc_err = vxc_rel;
      worst_vxc_idx = i;
    }

    std::cout << "  " << std::scientific << std::setprecision(4)
              << std::setw(12) << rho_pts[i]
              << "  " << std::setw(14) << tides_eps[i]
              << "  " << std::setw(14) << libxc_eps[i]
              << "  " << std::setw(12) << eps_rel
              << "  " << std::setw(14) << tides_vxc[i]
              << "  " << std::setw(14) << libxc_vxc[i]
              << "  " << std::setw(12) << vxc_rel
              << "\n";
  }

  std::cout << "\n  Max relative eps_xc error: " << max_eps_err
            << " at rho=" << rho_pts[worst_eps_idx] << "\n";
  std::cout << "  Max relative v_xc error:   " << max_vxc_err
            << " at rho=" << rho_pts[worst_vxc_idx] << "\n";

  if (max_eps_err > TOL) {
    return Fail("eps_xc relative error " + std::to_string(max_eps_err) +
                " > " + std::to_string(TOL));
  }
  if (max_vxc_err > TOL) {
    return Fail("v_xc relative error " + std::to_string(max_vxc_err) +
                " > " + std::to_string(TOL));
  }

  std::cout << "\n  PASS (max eps err=" << max_eps_err
            << ", max vxc err=" << max_vxc_err << ")\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  XC Rung-0 Oracle Sweep — TIDES vs libxc                    ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestLdaRung0();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL XC RUNG-0 ORACLE TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
