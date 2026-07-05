#include "tile/layout.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::TileMat;

std::vector<double> MakeBandedBlockDense(std::size_t n, std::uint32_t edge,
                                         std::size_t block_radius) {
  std::vector<double> dense(n * n, 0.0);
  const std::size_t blocks = (n + edge - 1) / edge;
  for (std::size_t br = 0; br < blocks; ++br) {
    const std::size_t bc_begin = br > block_radius ? br - block_radius : 0;
    const std::size_t bc_end = std::min(blocks, br + block_radius + 1);
    for (std::size_t bc = bc_begin; bc < bc_end; ++bc) {
      const std::size_t row0 = br * edge;
      const std::size_t col0 = bc * edge;
      const std::size_t row_extent = std::min<std::size_t>(edge, n - row0);
      const std::size_t col_extent = std::min<std::size_t>(edge, n - col0);
      for (std::size_t i = 0; i < row_extent; ++i) {
        for (std::size_t j = 0; j < col_extent; ++j) {
          dense[(row0 + i) * n + (col0 + j)] =
              1.0 + static_cast<double>((row0 + i + col0 + j) % 97) / 97.0;
        }
      }
    }
  }
  return dense;
}

std::size_t CountNonzeros(const std::vector<double>& dense) {
  std::size_t nnz = 0;
  for (const double v : dense) {
    nnz += v != 0.0 ? 1 : 0;
  }
  return nnz;
}

std::vector<double> MakeIrregularPointDense(std::size_t n,
                                            std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(0.004);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (keep(rng)) {
        dense[i * n + j] = value(rng);
      }
    }
  }
  return dense;
}

int RunCase(const std::string& label, const std::vector<double>& dense,
            std::size_t n, std::uint32_t edge) {
  const std::size_t dense_nnz = CountNonzeros(dense);
  const auto build_start = std::chrono::steady_clock::now();
  auto matrix = TileMat::FromDense(n, n, dense, edge);
  const auto build_end = std::chrono::steady_clock::now();
  if (!matrix.ok()) {
    std::cerr << "TileMat::FromDense failed: " << matrix.status().message()
              << '\n';
    return 1;
  }

  const auto dense_start = std::chrono::steady_clock::now();
  const std::vector<double> round_trip = matrix.value().ToDense();
  const auto dense_end = std::chrono::steady_clock::now();
  if (round_trip != dense) {
    std::cerr << "round trip changed values for edge " << edge << '\n';
    return 1;
  }

  const double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start)
          .count();
  const double todense_ms =
      std::chrono::duration<double, std::milli>(dense_end - dense_start)
          .count();
  const std::size_t padded_cells =
      matrix.value().tile_count() * static_cast<std::size_t>(edge) * edge;

  std::cout << "case=" << label
            << " edge=" << edge
            << " tiles=" << matrix.value().tile_count()
            << " dense_nnz=" << dense_nnz
            << " padded_cells=" << padded_cells
            << " padding_over_nnz="
            << static_cast<double>(padded_cells) /
                   static_cast<double>(dense_nnz)
            << " build_ms=" << build_ms
            << " to_dense_ms=" << todense_ms << '\n';
  return 0;
}

}  // namespace

int main() {
  constexpr std::size_t n = 1024;
  constexpr std::size_t block_radius = 1;
  std::cout << "tilemat_layout_probe: n=" << n
            << " block_radius=" << block_radius << '\n';
  for (const std::uint32_t edge : {16U, 32U, 64U}) {
    const std::vector<double> banded =
        MakeBandedBlockDense(n, edge, block_radius);
    if (RunCase("banded_full_tiles", banded, n, edge) != 0) {
      return 1;
    }
    const std::vector<double> irregular =
        MakeIrregularPointDense(n, 0x12340000ULL + edge);
    if (RunCase("irregular_points", irregular, n, edge) != 0) {
      return 1;
    }
  }
  return 0;
}
