#pragma once

// T9.3: Nightly A/B harness automation.
//
// The A/B harness runs a suite of test cases with two configurations
// (A = baseline/reference, B = current/candidate) and compares results
// against delta budgets. This enables nightly regression detection.
//
// Use cases:
//   - FP64 vs mixed-precision comparison
//   - CPU vs GPU comparison
//   - Before vs after optimization/refactor
//   - Release vs previous release
//
// The harness:
//   1. Runs each test case with both configurations
//   2. Records observables (energy, forces, drift, timing)
//   3. Computes deltas and checks against budgets
//   4. Generates a JSON report for the dashboard
//   5. Returns non-zero if any delta exceeds budget

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::verification {

// Observable types that can be compared in A/B tests.
enum class ObservableKind {
  kEnergy,           // Ha/atom
  kForce,            // Ha/Bohr
  kStress,           // Ha
  kDrift,            // uHa/atom/ps
  kTiming,           // ms
  kTraceError,       // dimensionless
  kIdempotencyError, // dimensionless
};

// A single observable result from one run.
struct Observable {
  std::string name;
  ObservableKind kind;
  double value_A = 0.0;    // baseline value
  double value_B = 0.0;    // candidate value
  double budget = 0.0;     // max allowed |A - B|
  std::string unit;
  bool passed = false;

  [[nodiscard]] double delta() const { return std::fabs(value_A - value_B); }
  [[nodiscard]] double relative_delta() const {
    if (std::fabs(value_A) < 1e-30) return 0.0;
    return delta() / std::fabs(value_A);
  }
};

// A test case in the A/B harness.
struct ABTestCase {
  std::string name;
  std::string description;
  std::vector<Observable> observables;
  bool skipped = false;
  std::string skip_reason;

  [[nodiscard]] bool all_passed() const {
    for (const auto& obs : observables)
      if (!obs.passed) return false;
    return true;
  }
};

// A/B harness result (full report).
struct ABHarnessResult {
  std::vector<ABTestCase> cases;
  int n_passed = 0;
  int n_failed = 0;
  int n_skipped = 0;
  std::string timestamp;
  std::string git_hash;  // optional: filled by caller
  bool overall_pass = false;

  [[nodiscard]] std::string toJson() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(12);
    ss << "{\n";
    ss << "  \"timestamp\": \"" << timestamp << "\",\n";
    ss << "  \"git_hash\": \"" << git_hash << "\",\n";
    ss << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n";
    ss << "  \"summary\": {"
       << "\"passed\": " << n_passed
       << ", \"failed\": " << n_failed
       << ", \"skipped\": " << n_skipped << "},\n";
    ss << "  \"cases\": [\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
      const auto& c = cases[i];
      ss << "    {\n";
      ss << "      \"name\": \"" << c.name << "\",\n";
      ss << "      \"passed\": " << (c.all_passed() ? "true" : "false") << ",\n";
      ss << "      \"skipped\": " << (c.skipped ? "true" : "false") << ",\n";
      if (c.skipped)
        ss << "      \"skip_reason\": \"" << c.skip_reason << "\",\n";
      ss << "      \"observables\": [\n";
      for (std::size_t j = 0; j < c.observables.size(); ++j) {
        const auto& o = c.observables[j];
        ss << "        {"
           << "\"name\": \"" << o.name << "\", "
           << "\"A\": " << o.value_A << ", "
           << "\"B\": " << o.value_B << ", "
           << "\"delta\": " << o.delta() << ", "
           << "\"budget\": " << o.budget << ", "
           << "\"unit\": \"" << o.unit << "\", "
           << "\"passed\": " << (o.passed ? "true" : "false")
           << "}" << (j + 1 < c.observables.size() ? "," : "") << "\n";
      }
      ss << "      ]\n";
      ss << "    }" << (i + 1 < cases.size() ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
  }
};

// Check an observable: mark passed if delta <= budget.
inline void CheckObservable(Observable& obs) {
  obs.passed = obs.delta() <= obs.budget;
}

// Run the A/B harness on a set of test cases.
// Each test case provides A and B values for observables.
// The harness checks deltas against budgets and generates a report.
[[nodiscard]] inline ABHarnessResult RunABHarness(
    std::vector<ABTestCase>& cases) {
  ABHarnessResult result;

  // Timestamp.
  std::time_t now = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  result.timestamp = ts;

  for (auto& tc : cases) {
    if (tc.skipped) {
      result.n_skipped++;
      result.cases.push_back(tc);
      continue;
    }

    for (auto& obs : tc.observables)
      CheckObservable(obs);

    if (tc.all_passed())
      result.n_passed++;
    else
      result.n_failed++;

    result.cases.push_back(tc);
  }

  result.overall_pass = (result.n_failed == 0);
  return result;
}

// Print a human-readable summary of the A/B harness result.
inline void PrintABReport(const ABHarnessResult& result) {
  std::cout << "=== A/B Harness Report ===\n";
  std::cout << "  timestamp: " << result.timestamp << "\n";
  std::cout << "  passed: " << result.n_passed
            << "  failed: " << result.n_failed
            << "  skipped: " << result.n_skipped << "\n";
  std::cout << "  overall: " << (result.overall_pass ? "PASS" : "FAIL") << "\n\n";

  for (const auto& tc : result.cases) {
    std::cout << "  [" << (tc.skipped ? "SKIP" : (tc.all_passed() ? "PASS" : "FAIL"))
              << "] " << tc.name;
    if (tc.skipped) std::cout << " (" << tc.skip_reason << ")";
    std::cout << "\n";
    for (const auto& obs : tc.observables) {
      std::cout << "    " << (obs.passed ? "  " : "!!")
                << " " << std::left << std::setw(25) << obs.name
                << " A=" << std::scientific << std::setprecision(6) << obs.value_A
                << " B=" << obs.value_B
                << " delta=" << obs.delta()
                << " budget=" << obs.budget
                << " " << obs.unit
                << (obs.passed ? "" : "  *** EXCEEDED ***")
                << "\n";
    }
  }
}

}  // namespace tides::verification
