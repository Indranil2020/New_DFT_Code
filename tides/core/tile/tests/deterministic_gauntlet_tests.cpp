#include "tile/ops.hpp"
#include "tile/ozaki.hpp"
#include "tile/spgemm_filtered.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::tile::AxpyFp64;
using tides::tile::DecomposeOzakiFp16Reference;
using tides::tile::FrobeniusNormF64eReference;
using tides::tile::ScaleMode;
using tides::tile::SpGemmFilteredFp64;
using tides::tile::Symmetry;
using tides::tile::TileMat;
using tides::tile::TraceF64eReference;

int Fail(const std::string& message) {
  std::cerr << "deterministic_gauntlet_tests: " << message << '\n';
  return 1;
}

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              std::uint64_t salt) {
  std::vector<double> dense(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const std::uint64_t key =
          (i + 1) * 0x9E3779B185EBCA87ULL ^
          (j + 17) * 0xC2B2AE3D27D4EB4FULL ^ salt;
      if (key % 7 == 0 || key % 19 == 0) {
        dense[i * cols + j] =
            std::sin(static_cast<double>((key & 0xFFFFU) + 1U)) /
            static_cast<double>(1U + (key % 997U));
      }
    }
  }
  return dense;
}

void AppendBytes(std::string* out, const void* data, std::size_t size) {
  out->append(static_cast<const char*>(data), size);
}

template <typename T>
void AppendPod(std::string* out, const T& value) {
  AppendBytes(out, &value, sizeof(value));
}

void AppendDoubleBits(std::string* out, double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendPod(out, bits);
}

tides::Status AppendTileMat(std::string* out, const TileMat& matrix) {
  std::ostringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  const tides::Status status = matrix.SerializeBinary(stream);
  if (!status.ok()) {
    return status;
  }
  out->append(stream.str());
  return tides::Status::Ok();
}

tides::Result<std::string> RunGauntletOnce() {
  constexpr std::size_t n = 64;
  const std::vector<double> a_dense = MakeDense(n, n, 0x1234ULL);
  const std::vector<double> b_dense = MakeDense(n, n, 0x5678ULL);
  auto a = TileMat::FromDense(n, n, a_dense, 16, Symmetry::kGeneral,
                              ScaleMode::kMaxAbs);
  auto b = TileMat::FromDense(n, n, b_dense, 16, Symmetry::kGeneral,
                              ScaleMode::kMaxAbs);
  if (!a.ok()) return a.status();
  if (!b.ok()) return b.status();

  auto axpy = AxpyFp64(0.75, a.value(), -0.25, b.value());
  if (!axpy.ok()) return axpy.status();
  auto product = SpGemmFilteredFp64(a.value(), b.value(), 0.35);
  if (!product.ok()) return product.status();
  auto trace = TraceF64eReference(product.value().product);
  if (!trace.ok()) return trace.status();
  auto norm = FrobeniusNormF64eReference(product.value().product);
  if (!norm.ok()) return norm.status();
  auto ozaki = DecomposeOzakiFp16Reference(product.value().product.raw_values());
  if (!ozaki.ok()) return ozaki.status();

  std::string fingerprint;
  tides::Status status = AppendTileMat(&fingerprint, a.value());
  if (!status.ok()) return status;
  status = AppendTileMat(&fingerprint, b.value());
  if (!status.ok()) return status;
  status = AppendTileMat(&fingerprint, axpy.value().matrix);
  if (!status.ok()) return status;
  status = AppendTileMat(&fingerprint, product.value().product);
  if (!status.ok()) return status;

  AppendPod(&fingerprint, product.value().ledger.candidate_products);
  AppendPod(&fingerprint, product.value().ledger.retained_products);
  AppendPod(&fingerprint, product.value().ledger.dropped_products);
  AppendDoubleBits(&fingerprint,
                   product.value().ledger.dropped_frobenius_bound);
  AppendDoubleBits(&fingerprint, trace.value().value);
  AppendDoubleBits(
      &fingerprint,
      trace.value().ledger.entries().front().observed_error_bound);
  AppendDoubleBits(&fingerprint, norm.value().value);
  AppendDoubleBits(
      &fingerprint,
      norm.value().ledger.entries().front().observed_error_bound);
  AppendPod(&fingerprint, ozaki.value().plan.slice_count);
  AppendPod(&fingerprint, ozaki.value().plan.exponent_span);
  AppendDoubleBits(&fingerprint,
                   ozaki.value().max_reconstruction_abs_error);
  for (const double quantum : ozaki.value().plan.slice_quanta) {
    AppendDoubleBits(&fingerprint, quantum);
  }
  for (const double slice : ozaki.value().slices) {
    AppendDoubleBits(&fingerprint, slice);
  }
  return fingerprint;
}

}  // namespace

int main() {
  auto reference = RunGauntletOnce();
  if (!reference.ok()) {
    return Fail("reference gauntlet failed: " +
                reference.status().message());
  }
  for (int repeat = 1; repeat < 100; ++repeat) {
    auto current = RunGauntletOnce();
    if (!current.ok()) {
      return Fail("repeat gauntlet failed: " + current.status().message());
    }
    if (current.value() != reference.value()) {
      return Fail("substrate gauntlet changed bits at repeat " +
                  std::to_string(repeat));
    }
  }
  std::cout << "deterministic_gauntlet_tests: 100 substrate repeats matched\n";
  return 0;
}
