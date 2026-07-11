// Tile substrate production GEMM test: verify TileMat + SpGemmFiltered
// produces correct results for H matrix operations in the solver context.
//
// This test exercises the proposal's central architectural thesis:
// block-sparse TileMat matrices with filtered SpGEMM for linear-scaling.
//
// Test: Build a block-sparse H matrix (simulating NAO with compact support),
// convert to TileMat, compute P @ H via SpGemmFilteredFp64, and verify
// the result matches dense BLAS.

#include "tile/layout.hpp"
#include "tile/spgemm_filtered.hpp"
#include "tile/precision.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::TileMat;
using tides::tile::SpGemmFilteredFp64;
using tides::tile::Symmetry;


int Fail(const std::string& msg) {
  std::cerr << "tile_production_tests: FAIL — " << msg << '\n';
  return 1;
}

// Build a block-sparse matrix simulating NAO compact support.
// Atoms beyond r_cut have zero interaction, creating block sparsity.
std::vector<double> BuildSparseH(std::size_t n, std::size_t tile_edge,
                                  std::size_t n_atoms, double sparsity,
                                  std::uint64_t seed) {
  std::vector<double> H(n * n, 0.0);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);

  std::size_t fns_per_atom = n / n_atoms;
  for (std::size_t a = 0; a < n_atoms; ++a) {
    for (std::size_t b = 0; b < n_atoms; ++b) {
      double dist = std::abs(static_cast<double>(a) - static_cast<double>(b));
      double keep = (dist <= 2.0) ? 1.0 : 0.0;  // only nearby atoms interact
      if (keep == 0.0 && sparsity < 1.0) continue;
      for (std::size_t i = a * fns_per_atom; i < (a+1) * fns_per_atom && i < n; ++i) {
        for (std::size_t j = b * fns_per_atom; j < (b+1) * fns_per_atom && j < n; ++j) {
          H[i * n + j] = keep * g(rng);
        }
      }
    }
  }
  // Symmetrize
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i+1; j < n; ++j)
      H[j * n + i] = H[i * n + j];
  return H;
}

int TestTileSpGemm() {
  std::cout << "\n=== Test 1: TileMat SpGEMM vs Dense BLAS ===\n";

  const std::size_t n = 64;
  const std::uint32_t tile_edge = 16;
  const std::size_t n_atoms = 4;  // 16 functions per atom
  const std::uint64_t seed = 42;

  // Build sparse H and P
  auto H = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed);
  auto P = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed + 1);

  // Dense reference: PH = P @ H
  std::vector<double> PH_dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += P[i * n + k] * H[k * n + j];
      PH_dense[i * n + j] = s;
    }

  // TileMat path
  auto h_tile = TileMat::FromDense(n, n, H, tile_edge, Symmetry::kSymmetric);
  if (!h_tile.ok()) return Fail("Failed to convert H to TileMat");

  auto p_tile = TileMat::FromDense(n, n, P, tile_edge);
  if (!p_tile.ok()) return Fail("Failed to convert P to TileMat");

  auto sp_result = tides::tile::SpGemmFilteredFp64(p_tile.value(), h_tile.value(), 1e-15);
  if (!sp_result.ok()) return Fail("SpGemmFilteredFp64 failed");

  // Compare
  const auto& tm = sp_result.value().product;
  auto PH_tile = tm.ToDense();

  double max_err = 0.0;
  for (std::size_t i = 0; i < n * n; ++i)
    max_err = std::max(max_err, std::abs(PH_dense[i] - PH_tile[i]));

  std::cout << "  n=" << n << " tile_edge=" << tile_edge
            << " H tiles=" << h_tile.value().tile_count()
            << " P tiles=" << p_tile.value().tile_count()
            << " max_err=" << max_err << "\n";

  if (max_err > 1e-10)
    return Fail("SpGEMM error " + std::to_string(max_err) + " > 1e-10");

  // Verify sparsity exploitation
  std::size_t total_blocks = (n / tile_edge) * (n / tile_edge);
  double h_sparsity = static_cast<double>(h_tile.value().tile_count()) / total_blocks;
  std::cout << "  H sparsity: " << h_tile.value().tile_count() << "/" << total_blocks
            << " = " << h_sparsity << "\n";

  std::cout << "  PASS (tile SpGEMM matches dense, max_err=" << max_err << ")\n";
  return 0;
}

int TestTileTrace() {
  std::cout << "\n=== Test 2: TileMat Trace (energy computation) ===\n";

  const std::size_t n = 64;
  const std::uint32_t tile_edge = 16;
  const std::size_t n_atoms = 4;
  const std::uint64_t seed = 42;

  auto H = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed);
  auto P = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed + 1);

  // Dense reference: trace(P @ H)
  double trace_dense = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k)
      trace_dense += P[i * n + k] * H[k * n + i];

  // TileMat path
  auto h_tile = TileMat::FromDense(n, n, H, tile_edge, Symmetry::kSymmetric);
  auto p_tile = TileMat::FromDense(n, n, P, tile_edge);
  auto sp_result = tides::tile::SpGemmFilteredFp64(p_tile.value(), h_tile.value(), 1e-15);
  if (!sp_result.ok()) return Fail("SpGemm failed for trace test");

  double trace_tile = sp_result.value().product.TraceFp64();

  double err = std::abs(trace_dense - trace_tile);
  std::cout << "  trace_dense=" << trace_dense << " trace_tile=" << trace_tile
            << " err=" << err << "\n";

  if (err > 1e-10)
    return Fail("Trace error " + std::to_string(err) + " > 1e-10");

  std::cout << "  PASS (tile trace matches dense, err=" << err << ")\n";
  return 0;
}

int TestTileLargerSystem() {
  std::cout << "\n=== Test 3: Larger system (n=256, 16 atoms) ===\n";

  const std::size_t n = 256;
  const std::uint32_t tile_edge = 32;
  const std::size_t n_atoms = 16;
  const std::uint64_t seed = 123;

  auto H = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed);
  auto P = BuildSparseH(n, tile_edge, n_atoms, 0.5, seed + 1);

  // Dense reference
  double trace_dense = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k)
      trace_dense += P[i * n + k] * H[k * n + i];

  // TileMat path
  auto h_tile = TileMat::FromDense(n, n, H, tile_edge, Symmetry::kSymmetric);
  if (!h_tile.ok()) return Fail("Failed to convert H to TileMat (n=256)");

  auto p_tile = TileMat::FromDense(n, n, P, tile_edge);
  if (!p_tile.ok()) return Fail("Failed to convert P to TileMat (n=256)");

  auto sp_result = tides::tile::SpGemmFilteredFp64(p_tile.value(), h_tile.value(), 1e-15);
  if (!sp_result.ok()) return Fail("SpGemm failed for n=256");

  double trace_tile = sp_result.value().product.TraceFp64();
  double err = std::abs(trace_dense - trace_tile);

  std::size_t total_blocks = (n / tile_edge) * (n / tile_edge);
  double sparsity = static_cast<double>(h_tile.value().tile_count()) / total_blocks;

  std::cout << "  n=" << n << " atoms=" << n_atoms
            << " H tiles=" << h_tile.value().tile_count() << "/" << total_blocks
            << " (sparsity=" << sparsity << ")"
            << " trace_err=" << err << "\n";

  if (err > 1e-8)
    return Fail("Trace error " + std::to_string(err) + " > 1e-8 for n=256");

  std::cout << "  PASS (large system tile trace, err=" << err << ")\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  Tile Substrate Production GEMM Tests                       ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestTileSpGemm();
  failures += TestTileTrace();
  failures += TestTileLargerSystem();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL TILE PRODUCTION TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
