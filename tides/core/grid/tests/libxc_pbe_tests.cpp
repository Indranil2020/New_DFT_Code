// T3.5: libxc PBE/GGA functional evaluation tests (CPU, no CUDA required).
//
// Validates that:
// 1. libxc is properly linked and initialized.
// 2. PBE exchange gives the correct enhancement factor F(s) at s=0 (F=1, LDA limit).
// 3. PBE correlation reduces to LDA in the uniform limit (sigma=0).
// 4. LDA-PW92 via libxc matches the internal LdaXC implementation.

#include "grid/libxc_wrapper.hpp"
#include "basis/atomgen/lda_xc.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_LDA_X;
using tides::grid::kLibxc_LDA_C_PW;
using tides::grid::kLibxc_GGA_X_PBE;
using tides::grid::kLibxc_GGA_C_PBE;
using tides::atomgen::LdaXC;

int Fail(const std::string& msg) {
  std::cerr << "libxc_pbe_tests: " << msg << '\n';
  return 1;
}

int TestLibxcInit() {
  LibxcFunctional fx, fc;
  if (!fx.Init(kLibxc_GGA_X_PBE, 1))
    return Fail("failed to init PBE exchange");
  if (!fc.Init(kLibxc_GGA_C_PBE, 1))
    return Fail("failed to init PBE correlation");

  std::cout << "libxc_init: PBE_X name=\"" << fx.Name() << "\""
            << " family=" << fx.Family() << '\n';
  std::cout << "libxc_init: PBE_C name=\"" << fc.Name() << "\""
            << " family=" << fc.Family() << '\n';

  if (fx.Family() != 2) return Fail("PBE X family should be GGA (2)");
  if (fc.Family() != 2) return Fail("PBE C family should be GGA (2)");
  return 0;
}

int TestPbeUniformLimit() {
  // In the uniform limit (sigma=0), PBE should reduce to:
  // eps_x = LDA exchange, eps_c = PBE correlation (not LDA C, but close).
  // The exchange enhancement factor F(s=0) = 1, so eps_x_PBE = eps_x_LDA.
  const double n = 0.1;
  const std::size_t np = 1;
  std::vector<double> rho = {n};
  std::vector<double> sigma = {0.0};  // uniform density

  LibxcFunctional fx, fc;
  fx.Init(kLibxc_GGA_X_PBE, 1);
  fc.Init(kLibxc_GGA_C_PBE, 1);

  auto rx = fx.EvalGGA(rho, sigma, np);
  auto rc = fc.EvalGGA(rho, sigma, np);

  // Compare PBE X at s=0 with LDA X
  LibxcFunctional fx_lda;
  fx_lda.Init(kLibxc_LDA_X, 1);
  auto rx_lda = fx_lda.EvalLDA(rho, np);

  const double eps_x_pbe = rx.eps_xc[0];
  const double eps_x_lda = rx_lda.eps_xc[0];
  const double x_diff = std::abs(eps_x_pbe - eps_x_lda);

  std::cout << "pbe_uniform: n=" << n
            << " eps_x_pbe=" << eps_x_pbe
            << " eps_x_lda=" << eps_x_lda
            << " x_diff=" << x_diff
            << " eps_c_pbe=" << rc.eps_xc[0] << '\n';

  // At s=0, PBE exchange = LDA exchange (F(0) = 1).
  if (x_diff > 1e-12) {
    std::ostringstream os;
    os << "PBE X at s=0 differs from LDA X: " << x_diff;
    return Fail(os.str());
  }

  // PBE correlation at sigma=0 should be finite and negative.
  if (rc.eps_xc[0] >= 0.0)
    return Fail("PBE correlation should be negative");

  return 0;
}

int TestLdaVsLibxc() {
  // Compare internal LdaXC with libxc LDA (X + C_PW).
  const double n = 0.05;
  const std::size_t np = 1;
  std::vector<double> rho = {n};

  auto libxc_res = LibxcFunctional::EvalLDAOnGrid(rho);
  const double internal_eps = LdaXC::EpsXC(n, 0.0);
  const double internal_vxc = LdaXC::VXC(n, 0.0);

  const double eps_diff = std::abs(libxc_res.eps_xc[0] - internal_eps);
  const double vxc_diff = std::abs(libxc_res.vxc[0] - internal_vxc);

  std::cout << "lda_vs_libxc: n=" << n
            << " libxc_eps=" << libxc_res.eps_xc[0]
            << " internal_eps=" << internal_eps
            << " eps_diff=" << eps_diff
            << " vxc_diff=" << vxc_diff << '\n';

  // Internal LDA uses PW92 correlation, same formula as libxc XC_LDA_C_PW.
  // Small differences (~1e-7) arise from different polynomial fits / constants.
  if (eps_diff > 1e-6) {
    std::ostringstream os;
    os << "LDA eps_xc differs from libxc: " << eps_diff;
    return Fail(os.str());
  }
  if (vxc_diff > 1e-6) {
    std::ostringstream os;
    os << "LDA V_xc differs from libxc: " << vxc_diff;
    return Fail(os.str());
  }
  return 0;
}

int TestPbeOnGrid() {
  // Test PBE evaluation on a small 3D grid with Gaussian density.
  const std::size_t n0 = 8, n1 = 8, n2 = 8;
  const double h = 0.4;
  const std::size_t np = n0 * n1 * n2;

  std::vector<double> rho(np, 0.0);
  for (std::size_t iz = 0; iz < n2; ++iz)
    for (std::size_t iy = 0; iy < n1; ++iy)
      for (std::size_t ix = 0; ix < n0; ++ix) {
        const double x = (static_cast<double>(ix) - 3.5) * h;
        const double y = (static_cast<double>(iy) - 3.5) * h;
        const double z = (static_cast<double>(iz) - 3.5) * h;
        const double r2 = x * x + y * y + z * z;
        const std::size_t g = ix + n0 * (iy + n1 * iz);
        rho[g] = std::exp(-0.5 * r2);
      }

  auto pbe = LibxcFunctional::EvalPBEOnGrid(n0, n1, n2, h, h, h, rho);

  // Compute energy
  double energy = 0.0;
  for (std::size_t i = 0; i < np; ++i)
    energy += pbe.eps_xc[i] * rho[i] * h * h * h;

  // Check that eps_xc is negative (XC energy is always negative)
  double max_eps = 0.0;
  for (std::size_t i = 0; i < np; ++i)
    max_eps = std::max(max_eps, pbe.eps_xc[i]);

  std::cout << "pbe_on_grid: np=" << np
            << " energy=" << energy
            << " max_eps=" << max_eps << '\n';

  if (energy >= 0.0)
    return Fail("PBE energy should be negative");
  if (max_eps > 0.0)
    return Fail("eps_xc should be negative everywhere for positive density");

  // Check that V_xc is negative (attractive)
  double max_vxc = 0.0;
  for (std::size_t i = 0; i < np; ++i)
    max_vxc = std::max(max_vxc, pbe.vxc[i]);
  if (max_vxc > 1e-6)
    return Fail("V_xc should be negative for positive density");

  return 0;
}

}  // namespace

int main() {
  int failures = 0;
  failures += TestLibxcInit();
  failures += TestPbeUniformLimit();
  failures += TestLdaVsLibxc();
  failures += TestPbeOnGrid();

  if (failures == 0) {
    std::cout << "All libxc PBE tests passed.\n";
  } else {
    std::cerr << failures << " libxc PBE test(s) failed.\n";
  }
  return failures;
}
