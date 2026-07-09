// T9.6: Regression dashboard + energy metering tests.
//
// Validates:
//   - Regression detection (3-sigma rule)
//   - Mean and stddev computation
//   - Energy estimation (Joules, kWh)
//   - Accuracy per joule metric
//   - JSON dashboard output

#include "verification/regression_dashboard.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::verification::AddPoint;
using tides::verification::DashboardData;
using tides::verification::EnergyMeasurement;
using tides::verification::MakeTimestamp;
using tides::verification::RegressionMetric;
using tides::verification::RegressionPoint;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T9.6a: Mean and stddev computation.
int TestStatistics() {
  std::cout << "\n=== T9.6a: Statistics ===\n";
  RegressionMetric m;
  m.name = "energy_error";
  m.unit = "Ha";
  m.budget = 1e-4;

  // Add 5 points: 1e-5, 1.1e-5, 0.9e-5, 1.05e-5, 0.95e-5
  for (double v : {1.0e-5, 1.1e-5, 0.9e-5, 1.05e-5, 0.95e-5})
    AddPoint(m, v);

  double mu = m.mean();
  double sigma = m.stddev();
  std::cout << "  mean=" << mu << " stddev=" << sigma << '\n';

  if (std::fabs(mu - 1.0e-5) > 1e-15)
    return Fail("T9.6a: mean wrong");
  if (sigma <= 0)
    return Fail("T9.6a: stddev should be positive");
  if (sigma > 1e-5)
    return Fail("T9.6a: stddev too large");

  std::cout << "T9.6a: GREEN\n";
  return 0;
}

// T9.6b: Regression detection (3-sigma).
int TestRegressionDetection() {
  std::cout << "\n=== T9.6b: Regression detection ===\n";
  RegressionMetric m;
  m.name = "force_error";
  m.unit = "Ha/Bohr";
  m.budget = 1e-4;

  // Add 10 stable points around 1e-6.
  for (int i = 0; i < 10; ++i)
    AddPoint(m, 1.0e-6 + 1e-12 * (i % 3));

  // No regression yet.
  if (m.is_regression(3.0))
    return Fail("T9.6b: false positive regression");

  // Add an outlier: 1e-3 (1000x larger).
  AddPoint(m, 1.0e-3);
  if (!m.is_regression(3.0))
    return Fail("T9.6b: missed regression (outlier not detected)");

  std::cout << "  mean=" << m.mean(10) << " stddev=" << m.stddev(10)
            << " latest=" << m.history.back().value
            << " regression=" << (m.is_regression(3.0) ? "YES" : "NO") << '\n';
  std::cout << "T9.6b: GREEN\n";
  return 0;
}

// T9.6c: Energy estimation.
int TestEnergyMetering() {
  std::cout << "\n=== T9.6c: Energy metering ===\n";
  EnergyMeasurement e;
  e.wall_time_s = 10.0;
  e.cpu_power_w = 125.0;  // CPU TDP
  e.gpu_power_w = 350.0;  // GPU avg power
  e.n_cpus = 1;
  e.n_gpus = 1;

  double joules = e.energy_joules();
  double kwh = e.energy_kwh();
  // Expected: 10 * (125 + 350) = 4750 J
  std::cout << "  energy=" << joules << " J = " << kwh << " kWh\n";

  if (std::fabs(joules - 4750.0) > 1e-10)
    return Fail("T9.6c: energy in Joules wrong");
  if (std::fabs(kwh - 4750.0 / 3.6e6) > 1e-15)
    return Fail("T9.6c: energy in kWh wrong");

  // Accuracy per joule.
  double apj = e.accuracy_per_joule(1e-6);
  std::cout << "  accuracy_per_joule=" << apj << " (error=1e-6, E=4750J)\n";
  if (std::fabs(apj - 1e-6 / 4750.0) > 1e-15)
    return Fail("T9.6c: accuracy per joule wrong");

  std::cout << "T9.6c: GREEN\n";
  return 0;
}

// T9.6d: JSON dashboard output.
int TestDashboardJson() {
  std::cout << "\n=== T9.6d: Dashboard JSON ===\n";
  DashboardData dash;
  dash.run_id = "nightly-2026-07-08";
  dash.timestamp = MakeTimestamp();
  dash.overall_pass = true;

  RegressionMetric m1;
  m1.name = "energy_error";
  m1.unit = "Ha";
  m1.budget = 1e-4;
  AddPoint(m1, 4.7e-5);
  AddPoint(m1, 5.1e-5);
  dash.metrics.push_back(m1);

  RegressionMetric m2;
  m2.name = "force_fd";
  m2.unit = "Ha/Bohr";
  m2.budget = 1e-6;
  AddPoint(m2, 7.1e-14);
  dash.metrics.push_back(m2);

  dash.energy.wall_time_s = 8.5;
  dash.energy.cpu_power_w = 125.0;
  dash.energy.n_cpus = 4;
  dash.energy.n_gpus = 0;

  std::string json = dash.toJson();
  std::cout << "  JSON length: " << json.size() << " chars\n";

  if (json.empty()) return Fail("T9.6d: JSON empty");
  if (json.find("\"run_id\"") == std::string::npos)
    return Fail("T9.6d: missing run_id");
  if (json.find("\"energy_joules\"") == std::string::npos)
    return Fail("T9.6d: missing energy_joules");
  if (json.find("\"regression\"") == std::string::npos)
    return Fail("T9.6d: missing regression field");
  if (json.find("\"stddev\"") == std::string::npos)
    return Fail("T9.6d: missing stddev field");

  std::cout << "T9.6d: GREEN\n";
  return 0;
}

// T9.6e: Budget pass/fail tracking.
int TestBudgetTracking() {
  std::cout << "\n=== T9.6e: Budget tracking ===\n";
  RegressionMetric m;
  m.name = "drift";
  m.unit = "uHa/atom/ps";
  m.budget = 30.0;

  AddPoint(m, 6.0);
  if (!m.history.back().passed) return Fail("T9.6e: 6.0 should pass (budget=30)");

  AddPoint(m, 35.0);
  if (m.history.back().passed) return Fail("T9.6e: 35.0 should fail (budget=30)");

  std::cout << "  point 0: value=" << m.history[0].value
            << " passed=" << m.history[0].passed << '\n';
  std::cout << "  point 1: value=" << m.history[1].value
            << " passed=" << m.history[1].passed << '\n';
  std::cout << "T9.6e: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestStatistics()) return 1;
  if (TestRegressionDetection()) return 1;
  if (TestEnergyMetering()) return 1;
  if (TestDashboardJson()) return 1;
  if (TestBudgetTracking()) return 1;
  std::cout << "\nregression_dashboard_tests: ALL GREEN\n";
  return 0;
}
