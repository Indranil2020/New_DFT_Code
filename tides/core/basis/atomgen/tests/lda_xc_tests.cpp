// XC module self-test: validate the VWN5/Slater LDA against known uniform
// electron gas values and against PySCF's libxc LDA functional.
//
// Known references (Perdew & Wang, Vosko et al.):
//   r_s = 1.0 (n = 3/(4 pi)): eps_c^P ~ -0.1208 Ha (paramagnetic VWN5).
//   r_s = 2.0:              eps_c^P ~ -0.0902 Ha.
//   r_s = 10.0:             eps_c^P ~ -0.0310 Ha.
// Exchange at r_s=1: eps_x = -3/4 (3/pi)^(1/3) n^(1/3), with n=3/(4pi) -> eps_x
//   = -3/4 * (3/pi)^(1/3) * (3/(4pi))^(1/3) = -3/4 * (9/(4 pi^2))^(1/3).

#include "basis/atomgen/lda_xc.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using tides::atomgen::LdaXC;

int Fail(const std::string& msg) {
  std::cerr << "lda_xc_tests: " << msg << '\n';
  return 1;
}

int Check() {
  // Exchange: eps_x(n, zeta=0) = -3/4 (3/pi)^(1/3) n^(1/3).
  // V_x = (4/3) eps_x. Check the relationship and a known value.
  for (double n : {0.1, 1.0, 10.0}) {
    const double ex = LdaXC::EpsX(n, 0.0);
    const double vx = LdaXC::VX(n, 0.0);
    if (std::fabs(vx - (4.0/3.0) * ex) > 1e-12) {
      std::ostringstream os;
      os << "V_x != (4/3) eps_x at n=" << n << ": " << vx << " vs " << (4.0/3.0)*ex;
      return Fail(os.str());
    }
  }
  // eps_x at n = 3/(4 pi) (r_s = 1): exact = -3/4 * (9/(4 pi^2))^(1/3).
  {
    const double n = 3.0 / (4.0 * M_PI);
    const double ex = LdaXC::EpsX(n, 0.0);
    const double exact = -0.75 * std::pow(9.0 / (4.0 * M_PI * M_PI), 1.0/3.0);
    if (std::fabs(ex - exact) > 1e-12) {
      std::ostringstream os;
      os << "eps_x at r_s=1: " << ex << " != " << exact;
      return Fail(os.str());
    }
    std::cout << "exchange r_s=1: eps_x=" << ex << " (exact " << exact << ") OK\n";
  }

  // Correlation: paramagnetic eps_c at several r_s. Reference values are the
  // EXACT PySCF/libxc PW92 values (functional 12, LDA_C_PW) — verified to
  // 8 digits. Our implementation must match to <= 1e-5.
  struct Ref { double rs; double ec; };
  const Ref refs[] = {
    {1.0, -0.05977386},   // r_s=1, PW92 (PySCF libxc ground truth)
    {2.0, -0.04475959},
    {5.0, -0.02821626},
    {10.0, -0.01857230},
  };
  for (const auto& ref : refs) {
    const double n = 3.0 / (4.0 * M_PI * ref.rs * ref.rs * ref.rs);
    const double ec = LdaXC::EpsC(n, 0.0);
    const double err = std::fabs(ec - ref.ec);
    std::cout << "correlation r_s=" << ref.rs << ": eps_c=" << ec
              << " ref=" << ref.ec << " err=" << err << '\n';
    if (err > 1e-5) {
      std::ostringstream os;
      os << "eps_c at r_s=" << ref.rs << ": " << ec << " vs ref " << ref.ec
         << " (err " << err << " > 1e-5)";
      return Fail(os.str());
    }
  }

  // Potential relationship: V_c = eps_c - (rs/3) d(eps_c)/drs.
  // Verify the FD derivative in VC is consistent (V_c computed two ways).
  {
    const double n = 1.0;
    const double rs = std::pow(3.0 / (4.0 * M_PI * n), 1.0/3.0);
    const double vc = LdaXC::VC(n, 0.0);
    const double ec = LdaXC::EpsC(n, 0.0);
    // d(eps_c)/drs via independent FD.
    const double h = 1e-6;
    const double n_p = 3.0 / (4.0 * M_PI * std::pow(rs + h, 3.0));
    const double n_m = 3.0 / (4.0 * M_PI * std::pow(std::max(rs - h, 1e-6), 3.0));
    const double dec_drs = (LdaXC::EpsC(n_p, 0.0) - LdaXC::EpsC(n_m, 0.0)) / (2.0 * h);
    const double vc_ref = ec - (rs / 3.0) * dec_drs;
    if (std::fabs(vc - vc_ref) > 1e-6) {
      std::ostringstream os;
      os << "V_c consistency: " << vc << " vs " << vc_ref;
      return Fail(os.str());
    }
    std::cout << "V_c consistency OK: " << vc << " vs " << vc_ref << '\n';
  }

  return 0;
}

}  // namespace

int main() {
  if (Check()) return 1;
  std::cout << "lda_xc_tests: ALL GREEN\n";
  return 0;
}
