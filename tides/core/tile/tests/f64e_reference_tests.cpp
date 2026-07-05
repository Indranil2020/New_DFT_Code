#include "tile/f64e_reference.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::DotF64eReference;
using tides::tile::GemmF64eReference;
using tides::tile::OperationKind;
using tides::tile::SumF64eReference;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "f64e_reference_tests: " << message << '\n';
  return 1;
}

double NaiveDot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

int AdversarialDotKeepsCancellationTerms() {
  std::vector<double> a;
  std::vector<double> b;
  for (int i = 0; i < 128; ++i) {
    a.push_back(1.0e16);
    b.push_back(1.0);
    a.push_back(1.0);
    b.push_back(1.0);
    a.push_back(-1.0e16);
    b.push_back(1.0);
  }
  auto result = DotF64eReference(a, b);
  if (!result.ok()) {
    return Fail("DotF64eReference failed: " + result.status().message());
  }
  const double naive = NaiveDot(a, b);
  if (result.value().value != 128.0) {
    return Fail("f64e reference did not preserve cancellation terms");
  }
  if (naive == result.value().value) {
    return Fail("adversarial fixture does not distinguish naive summation");
  }
  const auto& entry = result.value().ledger.entries().front();
  if (entry.operation != OperationKind::kDot ||
      !ValidateOperationLedgerEntry(entry).ok()) {
    return Fail("dot ledger entry is invalid");
  }
  return 0;
}

int GemmMatchesLongDoubleOracle() {
  constexpr std::size_t m = 7;
  constexpr std::size_t k = 33;
  constexpr std::size_t n = 5;
  std::mt19937_64 rng(0x12345678ULL);
  std::uniform_real_distribution<double> small(-1.0e-4, 1.0e-4);
  std::vector<double> a(m * k, 0.0);
  std::vector<double> b(k * n, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = (i % 7 == 0) ? 1.0e8 : small(rng);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = (i % 11 == 0) ? -1.0e8 : small(rng);
  }
  auto result = GemmF64eReference(m, k, n, a, b);
  if (!result.ok()) {
    return Fail("GemmF64eReference failed: " + result.status().message());
  }
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      long double sum = 0.0L;
      for (std::size_t p = 0; p < k; ++p) {
        sum += static_cast<long double>(a[row * k + p]) *
               static_cast<long double>(b[p * n + col]);
      }
      if (result.value().values[row * n + col] != static_cast<double>(sum)) {
        return Fail("GEMM result differs from long-double oracle");
      }
    }
  }
  const auto& entry = result.value().ledger.entries().front();
  if (entry.operation != OperationKind::kGemm ||
      !ValidateOperationLedgerEntry(entry).ok()) {
    return Fail("GEMM ledger entry is invalid");
  }
  return 0;
}

int SumReferenceLedgerIsValid() {
  const std::vector<double> values = {1.0e16, 1.0, -1.0e16, 2.0};
  auto result = SumF64eReference(values, OperationKind::kTrace,
                                 "trace-like adversarial sum");
  if (!result.ok()) {
    return Fail("SumF64eReference failed: " + result.status().message());
  }
  if (result.value().value != 3.0) {
    return Fail("f64e sum failed to preserve cancellation terms");
  }
  if (!ValidateOperationLedgerEntry(result.value().ledger.entries().front()).ok()) {
    return Fail("sum ledger entry is invalid");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = AdversarialDotKeepsCancellationTerms(); rc != 0) {
    return rc;
  }
  if (const int rc = GemmMatchesLongDoubleOracle(); rc != 0) {
    return rc;
  }
  if (const int rc = SumReferenceLedgerIsValid(); rc != 0) {
    return rc;
  }
  std::cout << "f64e_reference_tests: adversarial f64e reference checks passed\n";
  return 0;
}
