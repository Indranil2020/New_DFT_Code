// E4: Mixed precision SCF tests.
// Verifies FP16/BF16 quantization and Ozaki error-compensated GEMM.
#include "tile/mixed_precision_scf.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::tile::MixedPrecisionSCF;

int Fail(const std::string& msg) {
  std::cerr << "mixed_precision_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestQuantization() {
  std::cout << "\n=== E4: FP16/BF16 Quantization ===\n";
  double x = 1.23456789;
  double x_fp16 = MixedPrecisionSCF::QuantizeFp16(x);
  double x_bf16 = MixedPrecisionSCF::QuantizeBf16(x);
  std::cout << "  x = " << x << " fp16 = " << x_fp16 << " bf16 = " << x_bf16 << "\n";
  if (std::fabs(x - x_fp16) > 1e-3) return Fail("FP16 quantization error too large");
  if (std::fabs(x - x_bf16) > 1e-2) return Fail("BF16 quantization error too large");
  std::cout << "  PASS\n";
  return 0;
}

int TestOzakiGEMM() {
  std::cout << "\n=== E4: Ozaki Error-Compensated GEMM ===\n";
  const std::size_t n = 4;
  std::vector<double> P(n * n), H(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      P[i * n + j] = static_cast<double>(i + j) * 0.1;
      H[i * n + j] = static_cast<double>(i * n + j) * 0.01;
    }

  // Dense reference.
  std::vector<double> ref(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += P[i * n + k] * H[k * n + j];
      ref[i * n + j] = s;
    }

  // Ozaki GEMM with error feedback.
  std::vector<double> feedback;
  auto result = MixedPrecisionSCF::OzakiGEMM(n, P, H, false, &feedback);

  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err, std::fabs(result[i] - ref[i]));
  std::cout << "  Ozaki GEMM max error = " << max_err << "\n";
  if (max_err > 1e-6)
    return Fail("Ozaki GEMM error too large: " + std::to_string(max_err));
  std::cout << "  PASS\n";
  return 0;
}

int TestBandEnergy() {
  std::cout << "\n=== E4: Mixed Precision Band Energy ===\n";
  const std::size_t n = 4;
  std::vector<double> P(n * n), H(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    P[i * n + i] = 1.0;
    H[i * n + i] = static_cast<double>(i + 1) * -0.5;
  }

  // Dense reference: Tr(P@H) = sum of diagonal of H.
  double ref = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    ref += P[i * n + i] * H[i * n + i];

  double mp = MixedPrecisionSCF::MixedPrecisionBandEnergy(n, P, H, false);
  std::cout << "  Dense = " << ref << " MixedPrecision = " << mp << "\n";
  if (std::fabs(ref - mp) > 1e-6)
    return Fail("Band energy mismatch");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== E4: Mixed Precision SCF Tests ===\n";
  int failures = 0;
  failures += TestQuantization();
  failures += TestOzakiGEMM();
  failures += TestBandEnergy();
  if (failures == 0) std::cout << "\nALL MIXED PRECISION TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
