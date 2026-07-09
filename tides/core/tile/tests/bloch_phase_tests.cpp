// T2.8: Bloch-phase (complex) tiles for periodic R0/R1.
//
// Validates that k-space assembly from real-space periodic images produces
// correct complex matrices:
//   - At Gamma point (k=0), H(k) = sum_R H(R) (real, no phase)
//   - Hermiticity: H(k) = H(k)^dagger
//   - Time-reversal: H(-k) = H(k)^*
//   - Bloch phase correctness: single image with known phase

#include "tile/bloch_phase.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::tile::AssembleKSpace;
using tides::tile::BlochPhase;
using tides::tile::ComplexMatrix;
using tides::tile::ComplexFrobeniusNorm;
using tides::tile::ComplexMatMul;
using tides::tile::ComplexTrace;
using tides::tile::HermitianPart;
using tides::tile::MonkhorstPackGrid;
using tides::tile::PeriodicImage;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T2.8a: Gamma point — H(k=0) should be real and equal sum of all images.
int TestGammaPoint() {
  std::cout << "\n=== T2.8a: Gamma point assembly ===\n";
  const std::size_t n = 3;

  // Two real-space images: R=(0,0,0) and R=(1,0,0).
  PeriodicImage img0, img1;
  img0.R = {0.0, 0.0, 0.0};
  img0.matrix = {1, 0, 0,  0, 2, 0,  0, 0, 3};
  img1.R = {1.0, 0.0, 0.0};
  img1.matrix = {0, 0.5, 0,  0.5, 0, 0,  0, 0, 0};

  std::vector<PeriodicImage> images = {img0, img1};
  std::array<double, 3> k_gamma = {0.0, 0.0, 0.0};

  auto Hk = AssembleKSpace(n, k_gamma, images);

  // At Gamma, phase = 1 for all images, so H(k=0) = img0 + img1 (real).
  std::vector<double> expected = {1, 0.5, 0,  0.5, 2, 0,  0, 0, 3};
  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i) {
    double re_err = std::fabs(Hk.data[i].real() - expected[i]);
    double im_err = std::fabs(Hk.data[i].imag());
    max_err = std::max(max_err, std::max(re_err, im_err));
  }
  std::cout << "  Gamma H(k) max_err=" << max_err << '\n';
  if (max_err > 1e-15) return Fail("T2.8a: Gamma point assembly mismatch");
  std::cout << "T2.8a: GREEN\n";
  return 0;
}

// T2.8b: Hermiticity — H(k) should be Hermitian for valid periodic systems.
int TestHermiticity() {
  std::cout << "\n=== T2.8b: Hermiticity ===\n";
  const std::size_t n = 2;

  // H(R=0) is symmetric, H(R=a) is the off-diagonal coupling.
  // For a valid periodic system: H(R)^T = H(-R).
  PeriodicImage img_R0, img_Ra, img_Rma;
  img_R0.R = {0.0, 0.0, 0.0};
  img_R0.matrix = {1, 0,  0, 2};
  img_Ra.R = {1.0, 0.0, 0.0};
  img_Ra.matrix = {0, 0.3,  0, 0};
  img_Rma.R = {-1.0, 0.0, 0.0};
  img_Rma.matrix = {0, 0,  0.3, 0};  // transpose of img_Ra

  std::vector<PeriodicImage> images = {img_R0, img_Ra, img_Rma};
  std::array<double, 3> k = {0.25, 0.0, 0.0};

  auto Hk = AssembleKSpace(n, k, images);
  auto Hk_herm = HermitianPart(Hk);

  // Check Hermiticity: H(k) = H(k)^dagger
  double herm_err = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      herm_err = std::max(herm_err,
                          std::abs(Hk(i, j) - std::conj(Hk(j, i))));
  std::cout << "  Hermiticity err=" << herm_err << '\n';
  if (herm_err > 1e-15) return Fail("T2.8b: H(k) not Hermitian");
  std::cout << "T2.8b: GREEN\n";
  return 0;
}

// T2.8c: Time-reversal symmetry — H(-k) = H(k)^*
int TestTimeReversal() {
  std::cout << "\n=== T2.8c: Time-reversal symmetry ===\n";
  const std::size_t n = 2;

  PeriodicImage img_R0, img_Ra, img_Rma;
  img_R0.R = {0.0, 0.0, 0.0};
  img_R0.matrix = {1, 0,  0, 2};
  img_Ra.R = {1.0, 0.0, 0.0};
  img_Ra.matrix = {0, 0.3,  0, 0};
  img_Rma.R = {-1.0, 0.0, 0.0};
  img_Rma.matrix = {0, 0,  0.3, 0};

  std::vector<PeriodicImage> images = {img_R0, img_Ra, img_Rma};
  std::array<double, 3> k = {0.3, 0.1, -0.2};
  std::array<double, 3> mk = {-0.3, -0.1, 0.2};

  auto Hk = AssembleKSpace(n, k, images);
  auto Hmk = AssembleKSpace(n, mk, images);

  // H(-k) = H(k)^*
  double tr_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    tr_err = std::max(tr_err, std::abs(Hmk.data[i] - std::conj(Hk.data[i])));
  std::cout << "  Time-reversal err=" << tr_err << '\n';
  if (tr_err > 1e-15) return Fail("T2.8c: H(-k) != H(k)^*");
  std::cout << "T2.8c: GREEN\n";
  return 0;
}

// T2.8d: Bloch phase correctness — single off-diagonal image with known phase.
int TestBlochPhase() {
  std::cout << "\n=== T2.8d: Bloch phase factor ===\n";
  // Single image at R=(1,0,0), k=(pi, 0, 0).
  // Phase = exp(i * pi) = -1.
  // H(k) = H(R) * (-1) = -H(R).
  const std::size_t n = 2;
  PeriodicImage img;
  img.R = {1.0, 0.0, 0.0};
  img.matrix = {0, 1,  0, 0};

  std::array<double, 3> k = {M_PI, 0.0, 0.0};
  auto phase = BlochPhase(k, img.R);
  std::cout << "  phase = " << phase << " (expect -1+0j)\n";
  if (std::abs(phase - std::complex<double>(-1.0, 0.0)) > 1e-15)
    return Fail("T2.8d: Bloch phase incorrect");

  auto Hk = AssembleKSpace(n, k, {img});
  // H(k) should be -img.matrix (all imaginary parts zero, real = -matrix).
  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err,
                       std::abs(Hk.data[i] + std::complex<double>(img.matrix[i], 0.0)));
  std::cout << "  H(k) max_err=" << max_err << '\n';
  if (max_err > 1e-15) return Fail("T2.8d: phase-applied matrix incorrect");
  std::cout << "T2.8d: GREEN\n";
  return 0;
}

// T2.8e: Monkhorst-Pack grid generation.
int TestMonkhorstPack() {
  std::cout << "\n=== T2.8e: Monkhorst-Pack grid ===\n";
  auto grid = MonkhorstPackGrid(2);
  std::cout << "  2x2x2 grid: " << grid.size() << " k-points (expect 8)\n";
  if (grid.size() != 8) return Fail("T2.8e: wrong k-point count");
  // Check first k-point: (2*0+1)/(2*2) - 0.5 = 0.25 - 0.5 = -0.25
  if (std::abs(grid[0][0] - (-0.25)) > 1e-15)
    return Fail("T2.8e: first k-point wrong");
  std::cout << "T2.8e: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestGammaPoint()) return 1;
  if (TestHermiticity()) return 1;
  if (TestTimeReversal()) return 1;
  if (TestBlochPhase()) return 1;
  if (TestMonkhorstPack()) return 1;
  std::cout << "\nbloch_phase_tests: ALL GREEN\n";
  return 0;
}
