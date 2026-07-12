// E5: ASPC extrapolation tests.
// Verifies predictor coefficients and history management.
#include "dynamics/aspc_extrapolation.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::dynamics::ASPCExtrapolator;

int Fail(const std::string& msg) {
  std::cerr << "aspc_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestOrder1() {
  std::cout << "\n=== E5: ASPC Order 1 ===\n";
  ASPCExtrapolator aspc(1);
  std::vector<double> P0 = {1.0, 2.0};
  std::vector<double> P1 = {1.5, 2.5};
  aspc.Push(P0);
  aspc.Push(P1);
  auto pred = aspc.Predict();
  // Order 1: pred = 2*P1 - P0 = {2.0, 3.0}
  std::cout << "  pred = [" << pred[0] << ", " << pred[1] << "] (expected [2.0, 3.0])\n";
  if (std::fabs(pred[0] - 2.0) > 1e-10 || std::fabs(pred[1] - 3.0) > 1e-10)
    return Fail("ASPC order 1 predictor wrong");
  std::cout << "  PASS\n";
  return 0;
}

int TestOrder2() {
  std::cout << "\n=== E5: ASPC Order 2 ===\n";
  ASPCExtrapolator aspc(2);
  std::vector<double> P0 = {0.0};
  std::vector<double> P1 = {1.0};
  std::vector<double> P2 = {2.0};
  aspc.Push(P0);
  aspc.Push(P1);
  aspc.Push(P2);
  auto pred = aspc.Predict();
  // Order 2: pred = 3*P2 - 3*P1 + P0 = 6 - 3 + 0 = 3
  std::cout << "  pred = [" << pred[0] << "] (expected [3.0])\n";
  if (std::fabs(pred[0] - 3.0) > 1e-10)
    return Fail("ASPC order 2 predictor wrong");
  std::cout << "  PASS\n";
  return 0;
}

int TestInsufficientHistory() {
  std::cout << "\n=== E5: Insufficient History ===\n";
  ASPCExtrapolator aspc(3);
  aspc.Push({1.0});
  auto pred = aspc.Predict();
  if (!pred.empty()) return Fail("Should return empty with insufficient history");
  std::cout << "  PASS (empty prediction)\n";
  return 0;
}

int TestClearAndReuse() {
  std::cout << "\n=== E5: Clear and Reuse ===\n";
  ASPCExtrapolator aspc(2);
  aspc.Push({1.0, 2.0});
  aspc.Push({2.0, 3.0});
  if (aspc.history_size() != 2) return Fail("History size should be 2");
  aspc.Clear();
  if (aspc.history_size() != 0) return Fail("History should be cleared");
  if (aspc.HasHistory()) return Fail("Should have no history after clear");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== E5: ASPC Extrapolation Tests ===\n";
  int failures = 0;
  failures += TestOrder1();
  failures += TestOrder2();
  failures += TestInsufficientHistory();
  failures += TestClearAndReuse();
  if (failures == 0) std::cout << "\nALL ASPC TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
