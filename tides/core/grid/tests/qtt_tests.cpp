// QTT (Quantized Tensor Train) prototype tests.
//
// Validates:
//   - TT decomposition + reconstruction round-trip
//   - QTT 3D quantization + dequantization
//   - Compression ratio for smooth (Gaussian) density
//   - TT dot product and norm
//   - TT addition
//   - QTT-Poisson solver prototype (iterative)

#include "grid/qtt.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::grid::QTT;
using tides::grid::TensorTrain;
using tides::grid::TTCORE;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Test 1: TT decomposition + reconstruction round-trip.
int TestTTRoundTrip() {
  std::cout << "\n=== QTT: TT round-trip ===\n";
  // 4x4x4 tensor with smooth data.
  std::vector<std::size_t> shape = {4, 4, 4};
  std::vector<double> data(64, 0.0);
  for (std::size_t i = 0; i < 64; ++i)
    data[i] = std::sin(static_cast<double>(i) * 0.1);

  auto tt = QTT::Decompose(data, shape, 1e-12);
  if (tt.cores.empty()) return Fail("round-trip: decomposition failed");

  auto recon = QTT::Reconstruct(tt);
  if (recon.size() != data.size()) return Fail("round-trip: size mismatch");

  double max_err = 0.0;
  for (std::size_t i = 0; i < data.size(); ++i)
    max_err = std::max(max_err, std::fabs(recon[i] - data[i]));

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  ndim=" << tt.ndim() << " max_rank=" << tt.max_rank()
            << " params=" << tt.param_count() << " max_err=" << max_err << '\n';

  if (max_err > 1e-10) return Fail("round-trip: error too large");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: QTT 3D quantization + dequantization.
int TestQTT3D() {
  std::cout << "\n=== QTT: 3D quantization ===\n";
  // 8x8x8 grid (3 bits per axis, 9 QTT dimensions).
  const std::size_t n = 8;
  std::vector<double> rho(n * n * n, 0.0);
  const double alpha = 2.0;
  const double norm = std::pow(alpha / M_PI, 1.5);
  for (std::size_t iz = 0; iz < n; ++iz)
    for (std::size_t iy = 0; iy < n; ++iy)
      for (std::size_t ix = 0; ix < n; ++ix) {
        double x = (static_cast<double>(ix) - 3.5) * 0.3;
        double y = (static_cast<double>(iy) - 3.5) * 0.3;
        double z = (static_cast<double>(iz) - 3.5) * 0.3;
        rho[ix + n * (iy + n * iz)] = norm * std::exp(-alpha * (x*x + y*y + z*z));
      }

  auto tt = QTT::Quantize3D(rho, n, n, n, 1e-10);
  if (tt.cores.empty()) return Fail("3D quant: failed");

  auto recon = QTT::Reconstruct3D(tt, n, n, n);
  if (recon.size() != rho.size()) return Fail("3D quant: size mismatch");

  double max_err = 0.0;
  for (std::size_t i = 0; i < rho.size(); ++i)
    max_err = std::max(max_err, std::fabs(recon[i] - rho[i]));

  double compression = QTT::CompressionRatio(tt, n * n * n);
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  grid=" << n << "^3 max_rank=" << tt.max_rank()
            << " params=" << tt.param_count()
            << " compression=" << compression << "x"
            << " max_err=" << max_err << '\n';

  if (max_err > 1e-8) return Fail("3D quant: error too large");
  // 8^3 is small; QTT may not compress. Just verify accuracy.
  std::cout << "  (small grid: compression not expected)\n";

  std::cout << "PASS\n";
  return 0;
}

// Test 3: TT dot product and norm.
int TestTTDotNorm() {
  std::cout << "\n=== QTT: Dot product and norm ===\n";
  std::vector<std::size_t> shape = {4, 4, 4};
  std::vector<double> data(64, 0.0);
  for (std::size_t i = 0; i < 64; ++i)
    data[i] = std::cos(static_cast<double>(i) * 0.05);

  auto tt = QTT::Decompose(data, shape, 1e-14);

  // Compute direct norm.
  double direct_dot = 0.0;
  for (std::size_t i = 0; i < 64; ++i)
    direct_dot += data[i] * data[i];

  double tt_dot = QTT::NormSq(tt);
  double tt_norm = QTT::Norm(tt);
  double direct_norm = std::sqrt(direct_dot);

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  direct_dot=" << direct_dot << " tt_dot=" << tt_dot
            << " |err|=" << std::fabs(direct_dot - tt_dot) << '\n';
  std::cout << "  direct_norm=" << direct_norm << " tt_norm=" << tt_norm << '\n';

  if (std::fabs(direct_dot - tt_dot) > 1e-8)
    return Fail("dot/norm: dot product mismatch");

  std::cout << "PASS\n";
  return 0;
}

// Test 4: TT addition.
int TestTTAdd() {
  std::cout << "\n=== QTT: Addition ===\n";
  std::vector<std::size_t> shape = {4, 4};
  std::vector<double> a(16, 0.0), b(16, 0.0);
  for (std::size_t i = 0; i < 16; ++i) {
    a[i] = std::sin(static_cast<double>(i) * 0.1);
    b[i] = std::cos(static_cast<double>(i) * 0.2);
  }

  auto tt_a = QTT::Decompose(a, shape, 1e-14);
  auto tt_b = QTT::Decompose(b, shape, 1e-14);
  auto tt_sum = QTT::Add(tt_a, tt_b);

  auto recon = QTT::Reconstruct(tt_sum);
  double max_err = 0.0;
  for (std::size_t i = 0; i < 16; ++i)
    max_err = std::max(max_err, std::fabs(recon[i] - (a[i] + b[i])));

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  sum max_err=" << max_err
            << " rank=" << tt_sum.max_rank() << '\n';

  if (max_err > 1e-8) return Fail("addition: error too large");

  std::cout << "PASS\n";
  return 0;
}

// Test 5: QTT-Poisson solver prototype.
int TestQTTPoisson() {
  std::cout << "\n=== QTT: Poisson solver prototype ===\n";
  // 8x8x8 grid with Gaussian charge.
  const std::size_t n = 8;
  std::vector<double> rho(n * n * n, 0.0);
  const double alpha = 2.0;
  const double norm = std::pow(alpha / M_PI, 1.5);
  for (std::size_t iz = 0; iz < n; ++iz)
    for (std::size_t iy = 0; iy < n; ++iy)
      for (std::size_t ix = 0; ix < n; ++ix) {
        double x = (static_cast<double>(ix) - 3.5) * 0.3;
        double y = (static_cast<double>(iy) - 3.5) * 0.3;
        double z = (static_cast<double>(iz) - 3.5) * 0.3;
        rho[ix + n * (iy + n * iz)] = norm * std::exp(-alpha * (x*x + y*y + z*z));
      }

  auto rho_tt = QTT::Quantize3D(rho, n, n, n, 1e-10);
  if (rho_tt.cores.empty()) return Fail("Poisson: rho quantization failed");

  double h = 0.3;
  double dv = h * h * h;
  std::vector<std::size_t> grid_shape = {n, n, n};

  // Run QTT Poisson solver.
  auto V_tt = QTT::SolvePoissonQTT(rho_tt, grid_shape, dv, 100, 0.005, 1e-4);

  // Reconstruct V.
  auto V = QTT::Reconstruct3D(V_tt, n, n, n);
  if (V.empty()) return Fail("Poisson: V reconstruction failed");

  // Check: V should be positive near center (attractive potential).
  std::size_t center = n / 2 + n * (n / 2 + n * (n / 2));
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  V[center]=" << V[center] << " (should be positive)\n";
  std::cout << "  V rank=" << V_tt.max_rank()
            << " params=" << V_tt.param_count() << '\n';

  if (V[center] <= 0.0) return Fail("Poisson: V should be positive at center");

  std::cout << "PASS\n";
  return 0;
}

// Test 6: Compression scaling — larger grid should compress better.
int TestCompressionScaling() {
  std::cout << "\n=== QTT: Compression scaling ===\n";
  // Test Gaussian on 4^3, 8^3, 16^3 grids.
  for (std::size_t n : {4, 8, 16}) {
    std::vector<double> rho(n * n * n, 0.0);
    const double alpha = 1.0;
    const double norm = std::pow(alpha / M_PI, 1.5);
    double h = 8.0 / static_cast<double>(n);
    for (std::size_t iz = 0; iz < n; ++iz)
      for (std::size_t iy = 0; iy < n; ++iy)
        for (std::size_t ix = 0; ix < n; ++ix) {
          double x = (static_cast<double>(ix) - (n - 1) / 2.0) * h;
          double y = (static_cast<double>(iy) - (n - 1) / 2.0) * h;
          double z = (static_cast<double>(iz) - (n - 1) / 2.0) * h;
          rho[ix + n * (iy + n * iz)] = norm * std::exp(-alpha * (x*x + y*y + z*z));
        }

    auto tt = QTT::Quantize3D(rho, n, n, n, 1e-2);
    double comp = QTT::CompressionRatio(tt, n * n * n);
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "  n=" << n << " grid=" << n*n*n << " params=" << tt.param_count()
              << " rank=" << tt.max_rank() << " compression=" << comp << "x\n";

    if (n == 16 && comp < 1.5)
      return Fail("scaling: 16^3 should compress >1.5x");
  }

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestTTRoundTrip()) return 1;
  if (TestQTT3D()) return 1;
  if (TestTTDotNorm()) return 1;
  if (TestTTAdd()) return 1;
  if (TestQTTPoisson()) return 1;
  if (TestCompressionScaling()) return 1;
  std::cout << "\nqtt_tests: ALL GREEN\n";
  return 0;
}
