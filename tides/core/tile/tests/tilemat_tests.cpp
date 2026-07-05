#include "tile/layout.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::tile::ScaleMode;
using tides::tile::Symmetry;
using tides::tile::TileMat;

int Fail(const std::string& message) {
  std::cerr << "tilemat_tests: " << message << '\n';
  return 1;
}

bool ExactEqual(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

std::vector<double> MakeSparseDense(std::size_t rows, std::size_t cols,
                                    std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(0.17);
  std::uniform_real_distribution<double> value(-100.0, 100.0);
  std::vector<double> dense(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (keep(rng)) {
        dense[i * cols + j] = value(rng);
      }
    }
  }
  return dense;
}

std::vector<double> MakeSymmetricDense(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(0.24);
  std::uniform_real_distribution<double> value(-10.0, 10.0);
  std::vector<double> dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      if (keep(rng) || i == j) {
        const double v = value(rng);
        dense[i * n + j] = v;
        dense[j * n + i] = v;
      }
    }
  }
  return dense;
}

int RoundTripRandomPatterns() {
  constexpr std::uint32_t edges[] = {16, 32, 64};
  for (int case_id = 0; case_id < 100; ++case_id) {
    const std::uint32_t edge = edges[case_id % 3];
    const std::size_t rows = 1 + ((case_id * 37) % 117);
    const std::size_t cols = 1 + ((case_id * 53) % 121);
    const std::vector<double> dense =
        MakeSparseDense(rows, cols, 0xC0FFEEULL + case_id);
    auto matrix = TileMat::FromDense(rows, cols, dense, edge);
    if (!matrix.ok()) {
      return Fail("FromDense failed on random pattern " +
                  std::to_string(case_id) + ": " +
                  matrix.status().message());
    }
    if (!matrix.value().ValidateInvariants().ok()) {
      return Fail("ValidateInvariants failed on random pattern " +
                  std::to_string(case_id));
    }
    if (!ExactEqual(dense, matrix.value().ToDense())) {
      return Fail("FP64 dense<->tile round trip changed values on pattern " +
                  std::to_string(case_id));
    }
  }
  return 0;
}

int SymmetryIsPreserved() {
  constexpr std::uint32_t edges[] = {16, 32, 64};
  for (int case_id = 0; case_id < 30; ++case_id) {
    const std::uint32_t edge = edges[case_id % 3];
    const std::size_t n = 2 + ((case_id * 19) % 96);
    const std::vector<double> dense =
        MakeSymmetricDense(n, 0xA11CEULL + case_id);
    auto matrix =
        TileMat::FromDense(n, n, dense, edge, Symmetry::kSymmetric);
    if (!matrix.ok()) {
      return Fail("symmetric FromDense failed: " + matrix.status().message());
    }
    if (matrix.value().symmetry() != Symmetry::kSymmetric) {
      return Fail("symmetry flag was not preserved");
    }
    for (const auto view : matrix.value()) {
      if (view.block_col < view.block_row) {
        return Fail("symmetric matrix stored a lower-triangle tile");
      }
    }
    if (!ExactEqual(dense, matrix.value().ToDense())) {
      return Fail("symmetric dense<->tile round trip changed values");
    }
  }

  std::vector<double> not_symmetric = {1.0, 2.0, 3.0, 4.0};
  auto rejected =
      TileMat::FromDense(2, 2, not_symmetric, 16, Symmetry::kSymmetric);
  if (rejected.ok()) {
    return Fail("non-symmetric input was accepted as symmetric");
  }
  return 0;
}

int ScaleMetadataMatchesMaxAbs() {
  std::vector<double> dense(17 * 17, 0.0);
  dense[0] = -2.0;
  dense[1] = 7.5;
  dense[16 * 17 + 16] = -11.0;
  auto matrix = TileMat::FromDense(17, 17, dense, 16, Symmetry::kGeneral,
                                   ScaleMode::kMaxAbs);
  if (!matrix.ok()) {
    return Fail("scale test FromDense failed: " + matrix.status().message());
  }
  bool saw_scale_7_5 = false;
  bool saw_scale_11 = false;
  for (const auto view : matrix.value()) {
    saw_scale_7_5 = saw_scale_7_5 || view.scale == 7.5;
    saw_scale_11 = saw_scale_11 || view.scale == 11.0;
  }
  if (!saw_scale_7_5 || !saw_scale_11) {
    return Fail("per-tile max-abs scales do not match payload");
  }
  return 0;
}

int SerializationIsBitwiseStable() {
  const std::vector<double> dense = MakeSparseDense(77, 83, 0x5EEDULL);
  auto matrix = TileMat::FromDense(77, 83, dense, 32);
  if (!matrix.ok()) {
    return Fail("serialization FromDense failed: " + matrix.status().message());
  }

  std::stringstream first(std::ios::in | std::ios::out | std::ios::binary);
  const auto status = matrix.value().SerializeBinary(first);
  if (!status.ok()) {
    return Fail("SerializeBinary failed: " + status.message());
  }
  const std::string first_bytes = first.str();

  std::stringstream input(first_bytes, std::ios::in | std::ios::binary);
  auto loaded = TileMat::DeserializeBinary(input);
  if (!loaded.ok()) {
    return Fail("DeserializeBinary failed: " + loaded.status().message());
  }
  if (!matrix.value().SameSchemaAndPayload(loaded.value())) {
    return Fail("deserialized schema/payload differs from original");
  }
  if (!ExactEqual(dense, loaded.value().ToDense())) {
    return Fail("deserialized TileMat does not reconstruct dense input");
  }

  std::stringstream second(std::ios::in | std::ios::out | std::ios::binary);
  const auto second_status = loaded.value().SerializeBinary(second);
  if (!second_status.ok()) {
    return Fail("second SerializeBinary failed: " + second_status.message());
  }
  if (first_bytes != second.str()) {
    return Fail("serialize->deserialize->serialize is not bitwise stable");
  }
  return 0;
}

int TraceAndNormMatchDenseReference() {
  const std::vector<double> dense = MakeSymmetricDense(65, 0xBEEFULL);
  auto matrix =
      TileMat::FromDense(65, 65, dense, 32, Symmetry::kSymmetric);
  if (!matrix.ok()) {
    return Fail("trace/norm FromDense failed: " + matrix.status().message());
  }
  double trace = 0.0;
  long double norm2 = 0.0;
  for (std::size_t i = 0; i < 65; ++i) {
    trace += dense[i * 65 + i];
  }
  for (const double v : dense) {
    norm2 += static_cast<long double>(v) * static_cast<long double>(v);
  }
  if (matrix.value().TraceFp64() != trace) {
    return Fail("TraceFp64 differs from dense reference");
  }
  const double norm = std::sqrt(static_cast<double>(norm2));
  if (matrix.value().FrobeniusNormFp64() != norm) {
    return Fail("FrobeniusNormFp64 differs from dense reference");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = RoundTripRandomPatterns(); rc != 0) {
    return rc;
  }
  if (const int rc = SymmetryIsPreserved(); rc != 0) {
    return rc;
  }
  if (const int rc = ScaleMetadataMatchesMaxAbs(); rc != 0) {
    return rc;
  }
  if (const int rc = SerializationIsBitwiseStable(); rc != 0) {
    return rc;
  }
  if (const int rc = TraceAndNormMatchDenseReference(); rc != 0) {
    return rc;
  }

  std::cout << "tilemat_tests: analytical TileMat invariants passed\n";
  return 0;
}
