// T9.3: Nightly A/B harness automation tests.
//
// Validates:
//   - A/B harness correctly detects pass/fail based on delta vs budget
//   - JSON report generation is valid
//   - Multiple test cases with multiple observables
//   - Skipped cases are handled correctly

#include "verification/ab_harness.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::verification::ABHarnessResult;
using tides::verification::ABTestCase;
using tides::verification::CheckObservable;
using tides::verification::Observable;
using tides::verification::ObservableKind;
using tides::verification::PrintABReport;
using tides::verification::RunABHarness;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T9.3a: A/B harness detects pass when delta <= budget.
int TestPassCase() {
  std::cout << "\n=== T9.3a: A/B pass case ===\n";
  std::vector<ABTestCase> cases;
  ABTestCase tc;
  tc.name = "He LDA energy";
  tc.description = "FP64 vs mixed precision";

  Observable obs;
  obs.name = "total_energy";
  obs.kind = ObservableKind::kEnergy;
  obs.value_A = -2.8343;
  obs.value_B = -2.8343;
  obs.budget = 1e-4;
  obs.unit = "Ha/atom";
  tc.observables.push_back(obs);

  cases.push_back(tc);
  auto result = RunABHarness(cases);

  if (!result.overall_pass) return Fail("T9.3a: should pass (delta=0)");
  if (result.n_passed != 1) return Fail("T9.3a: n_passed should be 1");
  if (result.n_failed != 0) return Fail("T9.3a: n_failed should be 0");

  std::cout << "  n_passed=" << result.n_passed << " n_failed=" << result.n_failed
            << " overall=" << (result.overall_pass ? "PASS" : "FAIL") << '\n';
  std::cout << "T9.3a: GREEN\n";
  return 0;
}

// T9.3b: A/B harness detects fail when delta > budget.
int TestFailCase() {
  std::cout << "\n=== T9.3b: A/B fail case ===\n";
  std::vector<ABTestCase> cases;
  ABTestCase tc;
  tc.name = "H2O force";

  Observable obs;
  obs.name = "max_force";
  obs.kind = ObservableKind::kForce;
  obs.value_A = 0.001;
  obs.value_B = 0.005;  // delta = 0.004
  obs.budget = 0.001;   // budget = 0.001
  obs.unit = "Ha/Bohr";
  tc.observables.push_back(obs);

  cases.push_back(tc);
  auto result = RunABHarness(cases);

  if (result.overall_pass) return Fail("T9.3b: should fail (delta > budget)");
  if (result.n_failed != 1) return Fail("T9.3b: n_failed should be 1");

  std::cout << "  delta=" << cases[0].observables[0].delta()
            << " budget=" << cases[0].observables[0].budget
            << " passed=" << cases[0].observables[0].passed << '\n';
  std::cout << "T9.3b: GREEN\n";
  return 0;
}

// T9.3c: Multiple test cases with multiple observables.
int TestMultipleCases() {
  std::cout << "\n=== T9.3c: Multiple test cases ===\n";
  std::vector<ABTestCase> cases;

  // Case 1: SP2 purification (2 observables).
  ABTestCase tc1;
  tc1.name = "SP2 n=50";
  Observable obs1;
  obs1.name = "idempotency_err";
  obs1.kind = ObservableKind::kIdempotencyError;
  obs1.value_A = 5e-14;
  obs1.value_B = 5.1e-14;
  obs1.budget = 1e-12;
  obs1.unit = "rel";
  tc1.observables.push_back(obs1);

  Observable obs2;
  obs2.name = "trace_error";
  obs2.kind = ObservableKind::kTraceError;
  obs2.value_A = 5.3e-14;
  obs2.value_B = 5.4e-14;
  obs2.budget = 1e-12;
  obs2.unit = "rel";
  tc1.observables.push_back(obs2);
  cases.push_back(tc1);

  // Case 2: XL-BOMD drift (1 observable).
  ABTestCase tc2;
  tc2.name = "XL-BOMD NVE drift";
  Observable obs3;
  obs3.name = "drift";
  obs3.kind = ObservableKind::kDrift;
  obs3.value_A = 6.0;
  obs3.value_B = 8.0;
  obs3.budget = 30.0;
  obs3.unit = "uHa/atom/ps";
  tc2.observables.push_back(obs3);
  cases.push_back(tc2);

  // Case 3: Skipped.
  ABTestCase tc3;
  tc3.name = "GPU vs CPU (deferred)";
  tc3.skipped = true;
  tc3.skip_reason = "no GPU on this machine";
  cases.push_back(tc3);

  auto result = RunABHarness(cases);
  PrintABReport(result);

  if (result.n_passed != 2) return Fail("T9.3c: n_passed should be 2");
  if (result.n_failed != 0) return Fail("T9.3c: n_failed should be 0");
  if (result.n_skipped != 1) return Fail("T9.3c: n_skipped should be 1");
  if (!result.overall_pass) return Fail("T9.3c: should pass overall");

  std::cout << "T9.3c: GREEN\n";
  return 0;
}

// T9.3d: JSON report generation.
int TestJsonReport() {
  std::cout << "\n=== T9.3d: JSON report ===\n";
  std::vector<ABTestCase> cases;
  ABTestCase tc;
  tc.name = "test_json";
  Observable obs;
  obs.name = "energy";
  obs.kind = ObservableKind::kEnergy;
  obs.value_A = -1.0;
  obs.value_B = -1.0001;
  obs.budget = 1e-3;
  obs.unit = "Ha";
  tc.observables.push_back(obs);
  cases.push_back(tc);

  auto result = RunABHarness(cases);
  std::string json = result.toJson();

  std::cout << "  JSON length: " << json.size() << " chars\n";
  if (json.empty()) return Fail("T9.3d: JSON is empty");
  if (json.find("\"overall_pass\"") == std::string::npos)
    return Fail("T9.3d: missing overall_pass field");
  if (json.find("\"timestamp\"") == std::string::npos)
    return Fail("T9.3d: missing timestamp field");
  if (json.find("\"cases\"") == std::string::npos)
    return Fail("T9.3d: missing cases field");
  if (json.find("\"delta\"") == std::string::npos)
    return Fail("T9.3d: missing delta field");
  if (json.find("\"budget\"") == std::string::npos)
    return Fail("T9.3d: missing budget field");

  std::cout << "T9.3d: GREEN\n";
  return 0;
}

// T9.3e: Relative delta computation.
int TestRelativeDelta() {
  std::cout << "\n=== T9.3e: Relative delta ===\n";
  Observable obs;
  obs.value_A = 100.0;
  obs.value_B = 100.1;
  // delta = 0.1, relative = 0.1/100 = 0.001
  if (std::fabs(obs.delta() - 0.1) > 1e-12)
    return Fail("T9.3e: delta wrong");
  if (std::fabs(obs.relative_delta() - 0.001) > 1e-12)
    return Fail("T9.3e: relative delta wrong");

  // Edge case: A = 0.
  obs.value_A = 0.0;
  obs.value_B = 1.0;
  if (obs.relative_delta() != 0.0)
    return Fail("T9.3e: relative delta should be 0 when A=0");

  std::cout << "  delta=" << 0.1 << " relative=" << 0.001 << '\n';
  std::cout << "T9.3e: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestPassCase()) return 1;
  if (TestFailCase()) return 1;
  if (TestMultipleCases()) return 1;
  if (TestJsonReport()) return 1;
  if (TestRelativeDelta()) return 1;
  std::cout << "\nab_harness_tests: ALL GREEN\n";
  return 0;
}
