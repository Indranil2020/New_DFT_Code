// E7: QTT compression SCF tests.
// Verifies SVD-based compression, decompression accuracy, and compressed GEMM.
#include "tile/qtt_scf.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::tile::QTTCompressor;

int Fail(const std::string& msg) {
  std::cerr << "qtt_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestCompressDecompress() {
  std::cout << "\n=== E7: QTT Compress and Decompress ===\n";
  // Create a low-rank matrix: M = U @ V^T where U,V are (4x2).
  const std::size_t n = 4;
  std::vector<double> M(n * n, 0.0);
  // Rank-2 matrix: M[i,j] = a_i * b_j + c_i * d_j
  double a[] = {1, 2, 3, 4}, b[] = {1, 0.5, 0.3, 0.1};
  double c[] = {0.1, 0.2, 0.3, 0.4}, d[] = {0.5, 0.4, 0.3, 0.2};
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      M[i * n + j] = a[i] * b[j] + c[i] * d[j];

  auto cm = QTTCompressor::Compress(M, n, 1e-10, 0);
  std::cout << "  Original rank: 2, detected rank: " << cm.rank << "\n";
  std::cout << "  Compression ratio: " << cm.compression_ratio << "x\n";
  std::cout << "  Truncation error: " << cm.truncation_error << "\n";

  auto M_reconstructed = QTTCompressor::Decompress(cm);
  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err, std::fabs(M[i] - M_reconstructed[i]));
  std::cout << "  Reconstruction error: " << max_err << "\n";
  if (max_err > 1e-8) return Fail("Reconstruction error too large");
  if (cm.rank > 2) return Fail("Rank should be <= 2");
  std::cout << "  PASS\n";
  return 0;
}

int TestCompressedGEMM() {
  std::cout << "\n=== E7: Compressed P@H GEMM ===\n";
  const std::size_t n = 4;
  // Low-rank P: P[i,j] = u_i * u_j (rank 1).
  std::vector<double> P(n * n), H(n * n);
  double u[] = {1, 2, 3, 4};
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      P[i * n + j] = u[i] * u[j];
      H[i * n + j] = static_cast<double>(i == j ? 1 : 0);
    }

  auto cm = QTTCompressor::Compress(P, n, 1e-10, 0);
  std::cout << "  P rank: " << cm.rank << "\n";

  // Compressed P@H.
  auto PH_compressed = QTTCompressor::MatMulCompressed(cm, H);

  // Dense reference.
  std::vector<double> PH_ref(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += P[i * n + k] * H[k * n + j];
      PH_ref[i * n + j] = s;
    }

  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err, std::fabs(PH_compressed[i] - PH_ref[i]));
  std::cout << "  GEMM error: " << max_err << "\n";
  if (max_err > 1e-8) return Fail("Compressed GEMM error too large");
  std::cout << "  PASS\n";
  return 0;
}

int TestTraceCompressed() {
  std::cout << "\n=== E7: Tr(P@H) Compressed ===\n";
  const std::size_t n = 4;
  std::vector<double> P(n * n), H(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    P[i * n + i] = 1.0;
    H[i * n + i] = static_cast<double>(i + 1);
  }

  auto cm = QTTCompressor::Compress(P, n, 1e-10, 0);
  double tr_compressed = QTTCompressor::TraceCompressedPH(cm, H);

  // Dense reference: Tr(P@H) = sum P[i,i]*H[i,i] = 1+2+3+4 = 10.
  std::cout << "  Tr(P@H) compressed = " << tr_compressed << " (expected 10.0)\n";
  if (std::fabs(tr_compressed - 10.0) > 1e-8)
    return Fail("Trace mismatch");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== E7: QTT Compression SCF Tests ===\n";
  int failures = 0;
  failures += TestCompressDecompress();
  failures += TestCompressedGEMM();
  failures += TestTraceCompressed();
  if (failures == 0) std::cout << "\nALL QTT TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
