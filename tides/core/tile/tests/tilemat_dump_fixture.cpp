#include "tile/layout.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::tile::ScaleMode;
using tides::tile::Symmetry;
using tides::tile::TileMat;

int Fail(const std::string& message) {
  std::cerr << "tilemat_dump_fixture: " << message << '\n';
  return 1;
}

std::vector<double> MakeGeneralDense() {
  constexpr std::size_t rows = 37;
  constexpr std::size_t cols = 41;
  std::vector<double> dense(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if ((i * 17 + j * 29) % 11 == 0 || (i + 3 * j) % 23 == 0) {
        dense[i * cols + j] =
            std::sin(static_cast<double>(i + 1)) *
            std::cos(static_cast<double>(j + 2));
      }
    }
  }
  return dense;
}

std::vector<double> MakeSymmetricDense() {
  constexpr std::size_t n = 39;
  std::vector<double> dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      if (i == j || (i * 7 + j * 13) % 9 == 0) {
        const double value =
            std::sin(static_cast<double>(i + j + 1)) /
            static_cast<double>(1 + i + j);
        dense[i * n + j] = value;
        dense[j * n + i] = value;
      }
    }
  }
  return dense;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    return Fail("usage: tilemat_dump_fixture MODE OUTPUT_BINARY");
  }
  const std::string mode = argv[1];
  const std::string path = argv[2];

  tides::Result<TileMat> matrix =
      tides::Status::InvalidArgument("uninitialized TileMat fixture");
  if (mode == "general") {
    const std::vector<double> dense = MakeGeneralDense();
    matrix = TileMat::FromDense(37, 41, dense, 16, Symmetry::kGeneral,
                                ScaleMode::kMaxAbs);
  } else if (mode == "symmetric") {
    const std::vector<double> dense = MakeSymmetricDense();
    matrix = TileMat::FromDense(39, 39, dense, 16, Symmetry::kSymmetric,
                                ScaleMode::kMaxAbs);
  } else {
    return Fail("unknown fixture mode: " + mode);
  }

  if (!matrix.ok()) {
    return Fail("TileMat construction failed: " + matrix.status().message());
  }
  const tides::Status invariant_status = matrix.value().ValidateInvariants();
  if (!invariant_status.ok()) {
    return Fail("fixture invariant check failed: " +
                invariant_status.message());
  }
  const tides::Status save_status = matrix.value().SaveBinary(path);
  if (!save_status.ok()) {
    return Fail("fixture save failed: " + save_status.message());
  }
  return 0;
}
