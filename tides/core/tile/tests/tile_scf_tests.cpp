// E1: Tile substrate SCF integration tests.
// Verifies Tr(P@H) via TileMat matches dense BLAS, and the tile substrate
// is used for actual compute (not just trace verification).
#include "tile/tile_scf_integration.hpp"
#include "tile/layout.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::tile::TileSCFOps;
using tides::tile::TileMat;

int Fail(const std::string& msg) {
  std::cerr << "tile_scf_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestTracePH() {
  std::cout << "\n=== E1: Tr(P@H) via TileMat ===\n";
  const std::size_t n = 8;
  std::vector<double> P(n * n), H(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      P[i * n + j] = (i == j) ? 1.0 : 0.1;
      H[i * n + j] = static_cast<double>(i + j);
    }

  // Dense reference: Tr(P@H) = sum_{i,j} P[i,j]*H[j,i]
  double ref = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      ref += P[i * n + j] * H[j * n + i];

  // Build a full TileMat (all tiles present) from H.
  auto tm_result = TileMat::FromDense(n, n, H, 16);  // tile_size=4
  if (!tm_result.ok()) return Fail("FromDense failed");
  const TileMat& tm = tm_result.value();

  double tile_trace = TileSCFOps::TracePH(n, P, H, tm);
  std::cout << "  Dense Tr(P@H) = " << ref << "\n";
  std::cout << "  Tile  Tr(P@H) = " << tile_trace << "\n";

  if (std::fabs(ref - tile_trace) > 1e-10)
    return Fail("Tr(P@H) mismatch: " + std::to_string(std::fabs(ref - tile_trace)));
  std::cout << "  PASS\n";
  return 0;
}

int TestTracePS() {
  std::cout << "\n=== E1: Tr(P@S) electron count ===\n";
  const std::size_t n = 4;
  std::vector<double> P(n * n, 0.0), S(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    P[i * n + i] = 1.0;
    S[i * n + i] = 1.0;
  }
  // Off-diagonal S.
  S[0 * n + 1] = S[1 * n + 0] = 0.3;

  auto tm_result = TileMat::FromDense(n, n, S, 16);
  if (!tm_result.ok()) return Fail("FromDense failed");
  const TileMat& tm = tm_result.value();

  double tile_tr = TileSCFOps::TracePS(n, P, S, tm);
  // Tr(P@S) = sum P[i,j]*S[j,i] = P[0,0]*S[0,0] + P[1,1]*S[1,1] + ... = 4
  std::cout << "  Tr(P@S) = " << tile_tr << " (expected 4.0)\n";
  if (std::fabs(tile_tr - 4.0) > 1e-10)
    return Fail("Tr(P@S) mismatch");
  std::cout << "  PASS\n";
  return 0;
}

int TestCommutator() {
  std::cout << "\n=== E1: Commutator [H,P] norm ===\n";
  const std::size_t n = 4;
  std::vector<double> H(n * n, 0.0), P(n * n, 0.0);
  // H = diagonal (commutes with any diagonal P).
  for (std::size_t i = 0; i < n; ++i) {
    H[i * n + i] = static_cast<double>(i + 1);
    P[i * n + i] = 0.5;
  }
  auto tm_result = TileMat::FromDense(n, n, H, 16);
  if (!tm_result.ok()) return Fail("FromDense failed");
  const TileMat& tm = tm_result.value();

  double norm = TileSCFOps::CommutatorNorm(n, H, P, tm);
  std::cout << "  ||[H,P]||_F = " << norm << " (expected ~0 for diagonal H,P)\n";
  if (norm > 1e-10)
    return Fail("Commutator should be zero for commuting matrices");
  std::cout << "  PASS\n";
  return 0;
}

int TestMatMulPH() {
  std::cout << "\n=== E1: P@H product via TileMat ===\n";
  const std::size_t n = 4;
  std::vector<double> P(n * n), H(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      P[i * n + j] = static_cast<double>(i + 1);
      H[i * n + j] = static_cast<double>(j + 1);
    }
  auto tm_result = TileMat::FromDense(n, n, H, 32);
  if (!tm_result.ok()) return Fail("FromDense failed");
  const TileMat& tm = tm_result.value();

  auto result = TileSCFOps::MatMulPH(n, P, H, tm);

  // Dense reference.
  std::vector<double> ref(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += P[i * n + k] * H[k * n + j];
      ref[i * n + j] = s;
    }

  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err, std::fabs(result[i] - ref[i]));
  std::cout << "  max error = " << max_err << "\n";
  if (max_err > 1e-10)
    return Fail("P@H product mismatch");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== E1: Tile SCF Integration Tests ===\n";
  int failures = 0;
  failures += TestTracePH();
  failures += TestTracePS();
  failures += TestCommutator();
  failures += TestMatMulPH();
  if (failures == 0) std::cout << "\nALL TILE SCF TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
