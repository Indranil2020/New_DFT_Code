// T6.7: NEB (climbing image) tests.
//
// Validates NEB on a 1D double-well potential:
//   E(x) = (x^2 - 1)^2  (minima at x=±1, saddle at x=0, E_saddle=1)
//
// The NEB should find the saddle point at x=0 with E=1.

#include "dynamics/neb/neb.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::dynamics::NEB;
using tides::dynamics::NEBResult;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// 1D double-well: E(x) = (x^2 - 1)^2
double DoubleWellEnergy(const std::vector<double>& R) {
  double x = R[0];
  double x2 = x * x;
  return (x2 - 1.0) * (x2 - 1.0);
}

// Force: F = -dE/dx = -4x(x^2 - 1)
std::vector<double> DoubleWellForce(const std::vector<double>& R) {
  double x = R[0];
  return {-4.0 * x * (x * x - 1.0), 0.0, 0.0};
}

// T6.7a: NEB finds the saddle point of a 1D double-well.
int TestNEBDoubleWell() {
  std::cout << "\n=== T6.7a: NEB on 1D double-well ===\n";
  // Reactant at x=-1, product at x=+1. 7 images.
  std::vector<std::vector<double>> images(7);
  for (int i = 0; i < 7; ++i) {
    double t = static_cast<double>(i) / 6.0;
    double x = -1.0 + 2.0 * t;  // linear from -1 to +1
    images[i] = {x, 0.0, 0.0};
  }

  auto res = NEB::Run(images, DoubleWellEnergy, DoubleWellForce,
                      0.1,   // k_spring
                      500,   // max_steps
                      1e-4,  // f_tol
                      0.05,  // dt
                      true,  // climb
                      50);   // climb_start

  std::cout << "  n_steps=" << res.n_steps
            << " converged=" << res.converged
            << " max_force=" << res.max_force
            << " climbing_image=" << res.climbing_image << '\n';

  // Print final path.
  for (std::size_t i = 0; i < res.images.size(); ++i)
    std::cout << "  image " << i << " x=" << res.images[i][0]
              << " E=" << res.energies[i] << '\n';

  if (!res.converged)
    return Fail("T6.7a: NEB did not converge");

  // Saddle point should be at x≈0 with E≈1.
  std::size_t ci = res.climbing_image;
  double saddle_x = res.images[ci][0];
  double saddle_E = res.energies[ci];
  std::cout << "  saddle: x=" << saddle_x << " E=" << saddle_E
            << " (expect x≈0, E≈1)\n";

  if (std::fabs(saddle_x) > 0.05)
    return Fail("T6.7a: saddle point not at x=0");
  if (std::fabs(saddle_E - 1.0) > 0.01)
    return Fail("T6.7a: saddle energy not ≈1.0");

  std::cout << "T6.7a: GREEN (saddle found at x=" << saddle_x
            << ", E=" << saddle_E << ")\n";
  return 0;
}

// T6.7b: NEB endpoint energies are preserved.
int TestNEBEndpoints() {
  std::cout << "\n=== T6.7b: NEB endpoint preservation ===\n";
  std::vector<std::vector<double>> images(5);
  images[0] = {-1.0, 0.0, 0.0};
  images[4] = {1.0, 0.0, 0.0};
  for (int i = 1; i < 4; ++i) {
    double t = static_cast<double>(i) / 4.0;
    images[i] = {-1.0 + 2.0 * t, 0.0, 0.0};
  }

  auto res = NEB::Run(images, DoubleWellEnergy, DoubleWellForce,
                      0.1, 200, 1e-4, 0.05, true, 20);

  // Endpoints should not move.
  if (std::fabs(res.images[0][0] - (-1.0)) > 1e-10)
    return Fail("T6.7b: reactant endpoint moved");
  if (std::fabs(res.images.back()[0] - 1.0) > 1e-10)
    return Fail("T6.7b: product endpoint moved");

  // Endpoint energies should be 0 (minima).
  if (std::fabs(res.energies[0]) > 1e-10)
    return Fail("T6.7b: reactant energy not 0");
  if (std::fabs(res.energies.back()) > 1e-10)
    return Fail("T6.7b: product energy not 0");

  std::cout << "T6.7b: GREEN (endpoints preserved)\n";
  return 0;
}

}  // namespace

int main() {
  if (TestNEBDoubleWell()) return 1;
  if (TestNEBEndpoints()) return 1;
  std::cout << "\nneb_tests: ALL GREEN\n";
  return 0;
}
