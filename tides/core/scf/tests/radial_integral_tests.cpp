// P0.2: radial-integral unit tests against closed-form reference values
// on both uniform and non-uniform (log/quadratic) radial grids.
//
// These tests directly exercise the per-interval trapezoidal radial
// quadrature added in P0.1 (TabulateOverlapSS / TabulateKineticSS in
// nao_driver.hpp and KineticRadial in two_center_builder.hpp) and are
// independent of the full SCF machinery.

#include "basis/nao_generator.hpp"
#include "basis/two_center_builder.hpp"
#include "basis/two_center_integrals.hpp"
#include "scf/nao_driver.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::basis::CubicSpline;
using tides::basis::NaoBasisFunction;
using tides::basis::KineticRadial;
using tides::scf::NaoDriver;

int Fail(const std::string& msg) {
  std::cerr << "radial_integral_tests: FAIL — " << msg << '\n';
  return 1;
}

// Build a synthetic radial function on either a uniform or a non-uniform
// (quadratic-dense-near-origin) grid.
NaoBasisFunction MakeRadialFunction(int l, std::size_t n_r, double r_max,
                                    bool uniform,
                                    const std::function<double(double)>& fn) {
  NaoBasisFunction f;
  f.l = l;
  f.r_cut = r_max;
  f.r.resize(n_r);
  f.R.resize(n_r);
  if (uniform) {
    const double h = r_max / static_cast<double>(n_r - 1);
    for (std::size_t i = 0; i < n_r; ++i) {
      f.r[i] = h * static_cast<double>(i);
      f.R[i] = fn(f.r[i]);
    }
  } else {
    // Non-uniform grid: dense near r=0 (r ~ i^2), coarse at large r.
    for (std::size_t i = 0; i < n_r; ++i) {
      const double x = static_cast<double>(i) / static_cast<double>(n_r - 1);
      f.r[i] = r_max * x * x;
      f.R[i] = fn(f.r[i]);
    }
  }
  return f;
}

// -------------------------------------------------------------------------
// Test 1: KineticRadial against analytic L(r) = -1/2 (R'' + 2/r R' -
// l(l+1)/r^2 R) for Gaussian radial functions.
//
// s Gaussian: R(r) = exp(-a r^2)  ->  L(r) = (3a - 2 a^2 r^2) exp(-a r^2)
// p Gaussian:  R(r) = r exp(-a r^2)
//                ->  L(r) = (5a r - 2 a^2 r^3) exp(-a r^2)
// -------------------------------------------------------------------------
int TestKineticRadialClosedForm() {
  std::cout << "\n=== Test: KineticRadial vs closed form (Gaussian radial) ===\n";
  const double a = 1.0;
  const double r_max = 6.0;
  const std::size_t n_r = 3000;

  auto s_fn = [&](double r) { return std::exp(-a * r * r); };
  auto s_ref = [&](double r) {
    return (3.0 * a - 2.0 * a * a * r * r) * std::exp(-a * r * r);
  };

  auto p_fn = [&](double r) { return r * std::exp(-a * r * r); };
  auto p_ref = [&](double r) {
    return (5.0 * a * r - 2.0 * a * a * r * r * r) * std::exp(-a * r * r);
  };

  for (bool uniform : {true, false}) {
    const char* label = uniform ? "uniform" : "non-uniform";

    // s Gaussian (l=0)
    NaoBasisFunction fs = MakeRadialFunction(0, n_r, r_max, uniform, s_fn);
    std::vector<double> Ls;
    KineticRadial(fs, Ls);
    double max_err_s = 0.0;
    for (std::size_t i = 1; i < Ls.size(); ++i) {  // skip r=0
      const double ref = s_ref(fs.r[i]);
      const double err = std::fabs(Ls[i] - ref);
      max_err_s = std::max(max_err_s, err);
    }
    std::cout << "  s Gaussian " << label << ": max_err=" << max_err_s << '\n';
    double tol_s = uniform ? 5e-5 : 1e-3;
    if (max_err_s > tol_s) {
      std::ostringstream os;
      os << "s Gaussian " << label << " KineticRadial error " << max_err_s
         << " > " << tol_s;
      return Fail(os.str());
    }

    // p Gaussian (l=1). The radial kinetic expression has large 1/r term
    // cancellation near r=0, so we test only r >= 0.05 where the finite
    // differences are well behaved.
    NaoBasisFunction fp = MakeRadialFunction(1, n_r, r_max, uniform, p_fn);
    std::vector<double> Lp;
    KineticRadial(fp, Lp);
    double max_err_p = 0.0;
    for (std::size_t i = 0; i < Lp.size(); ++i) {
      if (fp.r[i] < 0.05) continue;
      const double ref = p_ref(fp.r[i]);
      const double err = std::fabs(Lp[i] - ref);
      max_err_p = std::max(max_err_p, err);
    }
    std::cout << "  p Gaussian " << label << ": max_err=" << max_err_p << '\n';
    // l=1 has large 1/r cancellation near the origin; this is a known
    // limitation of the R''+2/rR'-l(l+1)/r^2R form. We tolerate the residual
    // finite-difference error away from r=0.
    double tol_p = uniform ? 2e-4 : 5e-3;
    if (max_err_p > tol_p) {
      std::ostringstream os;
      os << "p Gaussian " << label << " KineticRadial error " << max_err_p
         << " > " << tol_p;
      return Fail(os.str());
    }
  }
  return 0;
}

// -------------------------------------------------------------------------
// Test 2: TabulateOverlapSS for s-s radial functions.
//
// For R_a(r) = R_b(r) = exp(-r) on a finite cutoff [0, L] the 1D radial
// integral used by TabulateOverlapSS is closed form:
//   S_L(R) = integral_0^L exp(-r) exp(-|r-R|) r^2 dr
// For 0 <= R <= L:
//   S(R) = exp(-R)*(R^3/3 + R^2/2 + R/2 + 1/4)
//        + exp(R - 2L)*(-L^2/2 - L/2 - 1/4)
// For R > L:
//   S(R) = exp(-R) * L^3/3
// -------------------------------------------------------------------------
double OverlapExponential(double R, double L) {
  if (R > L) return std::exp(-R) * L * L * L / 3.0;
  const double tail = std::exp(R - 2.0 * L) *
                      (-L * L / 2.0 - L / 2.0 - 0.25);
  return std::exp(-R) *
             (R * R * R / 3.0 + R * R / 2.0 + R / 2.0 + 0.25) +
         tail;
}

int TestTabulateOverlapSS() {
  std::cout << "\n=== Test: TabulateOverlapSS vs closed form (exponential radial) ===\n";
  const double r_max = 6.0;
  const std::size_t n_r = 3000;
  auto fn = [&](double r) { return std::exp(-r); };

  for (bool uniform : {true, false}) {
    const char* label = uniform ? "uniform" : "non-uniform";
    NaoBasisFunction f = MakeRadialFunction(0, n_r, r_max, uniform, fn);
    CubicSpline spline = NaoDriver::TabulateOverlapSS(f, f);

    double max_err = 0.0;
    for (double R = 0.0; R <= 4.0; R += 0.5) {
      const double ref = OverlapExponential(R, r_max);
      const double val = spline.Eval(R);
      max_err = std::max(max_err, std::fabs(val - ref));
    }
    std::cout << "  " << label << ": max_err=" << max_err << '\n';
    double tol = uniform ? 1e-4 : 5e-4;
    if (max_err > tol) {
      std::ostringstream os;
      os << "TabulateOverlapSS " << label << " error " << max_err
         << " > " << tol;
      return Fail(os.str());
    }
  }
  return 0;
}

// -------------------------------------------------------------------------
// Test 3: TabulateKineticSS for s-s Gaussian radial functions at R=0.
//
// For R_a(r) = R_b(r) = exp(-a r^2) the kinetic radial function at zero
// separation is:
//   T(0) = 3 sqrt(pi) / (16 sqrt(2a))
// -------------------------------------------------------------------------
int TestTabulateKineticSS() {
  std::cout << "\n=== Test: TabulateKineticSS vs closed form (Gaussian radial, R=0) ===\n";
  const double a = 1.0;
  const double r_max = 6.0;
  const std::size_t n_r = 3000;
  auto fn = [&](double r) { return std::exp(-a * r * r); };
  const double ref = 3.0 * std::sqrt(M_PI) / (16.0 * std::sqrt(2.0 * a));

  for (bool uniform : {true, false}) {
    const char* label = uniform ? "uniform" : "non-uniform";
    NaoBasisFunction f = MakeRadialFunction(0, n_r, r_max, uniform, fn);
    CubicSpline spline = NaoDriver::TabulateKineticSS(f, f);
    const double val = spline.Eval(0.0);
    const double err = std::fabs(val - ref);
    std::cout << "  " << label << ": T(0)=" << val << " ref=" << ref
              << " err=" << err << '\n';
    double tol = uniform ? 1e-4 : 2e-3;
    if (err > tol) {
      std::ostringstream os;
      os << "TabulateKineticSS " << label << " error " << err << " > " << tol;
      return Fail(os.str());
    }
  }
  return 0;
}

}  // namespace

int main() {
  if (TestKineticRadialClosedForm()) return 1;
  if (TestTabulateOverlapSS()) return 1;
  if (TestTabulateKineticSS()) return 1;

  std::cout << "\nradial_integral_tests: ALL GREEN\n";
  return 0;
}
