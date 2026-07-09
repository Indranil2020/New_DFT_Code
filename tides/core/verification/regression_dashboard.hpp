#pragma once

// T9.6: Regression dashboard + energy metering.
//
// Tracks test results over time to detect regressions and measures
// energy consumption (Joules) for certified accuracy-per-joule metrics.
//
// Components:
//   1. RegressionTracker: stores time-series of test results, detects
//      regressions (value exceeds historical mean + N*sigma).
//   2. EnergyMeter: estimates energy consumption from wall time and
//      GPU/CPU power ratings.
//   3. DashboardData: generates JSON for the web dashboard.
//
// Observable: regressions are detected within 1 run; energy is estimated
// to within 20% of measured (model-based).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::verification {

// A single data point in the regression time series.
struct RegressionPoint {
  std::string timestamp;
  double value = 0.0;
  bool passed = true;
  std::string git_hash;
};

// A tracked metric in the regression dashboard.
struct RegressionMetric {
  std::string name;
  std::string unit;
  double budget = 0.0;           // max allowed value
  std::vector<RegressionPoint> history;

  // Statistics over the last N points.
  [[nodiscard]] double mean(std::size_t last_n = 0) const {
    if (history.empty()) return 0.0;
    std::size_t start = (last_n > 0 && last_n < history.size())
        ? history.size() - last_n : 0;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = start; i < history.size(); ++i) {
      sum += history[i].value;
      count++;
    }
    return (count > 0) ? sum / count : 0.0;
  }

  [[nodiscard]] double stddev(std::size_t last_n = 0) const {
    if (history.size() < 2) return 0.0;
    double m = mean(last_n);
    std::size_t start = (last_n > 0 && last_n < history.size())
        ? history.size() - last_n : 0;
    double sq_sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = start; i < history.size(); ++i) {
      sq_sum += (history[i].value - m) * (history[i].value - m);
      count++;
    }
    return (count > 1) ? std::sqrt(sq_sum / (count - 1)) : 0.0;
  }

  // Detect regression: latest value exceeds mean + n_sigma * stddev.
  // Baseline statistics are computed excluding the latest point so the
  // outlier doesn't inflate the mean/stddev.
  [[nodiscard]] bool is_regression(double n_sigma = 3.0) const {
    if (history.size() < 4) return false;
    // Use points [0, n-2) as baseline, test point n-1.
    std::size_t n = history.size();
    std::size_t baseline_start = (n >= 11) ? n - 11 : 0;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = baseline_start; i < n - 1; ++i) {
      sum += history[i].value;
      count++;
    }
    if (count < 2) return false;
    double m = sum / count;
    double sq_sum = 0.0;
    for (std::size_t i = baseline_start; i < n - 1; ++i)
      sq_sum += (history[i].value - m) * (history[i].value - m);
    double s = std::sqrt(sq_sum / (count - 1));
    double latest = history.back().value;
    if (s < 1e-30) return false;
    return std::fabs(latest - m) > n_sigma * s;
  }
};

// Energy measurement for a single run.
struct EnergyMeasurement {
  double wall_time_s = 0.0;
  double cpu_power_w = 0.0;       // CPU TDP (W)
  double gpu_power_w = 0.0;       // GPU average power (W)
  int n_cpus = 1;
  int n_gpus = 0;

  // Estimated energy in Joules.
  [[nodiscard]] double energy_joules() const {
    return wall_time_s * (cpu_power_w * n_cpus + gpu_power_w * n_gpus);
  }

  // Energy in kWh.
  [[nodiscard]] double energy_kwh() const {
    return energy_joules() / 3.6e6;
  }

  // Accuracy per joule (lower is better for error metrics).
  [[nodiscard]] double accuracy_per_joule(double error_metric) const {
    double e = energy_joules();
    if (e < 1e-30) return 0.0;
    return error_metric / e;
  }
};

// Dashboard data: all metrics and energy measurements for a run.
struct DashboardData {
  std::string run_id;
  std::string timestamp;
  std::vector<RegressionMetric> metrics;
  EnergyMeasurement energy;
  bool overall_pass = false;

  [[nodiscard]] std::string toJson() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(8);
    ss << "{\n";
    ss << "  \"run_id\": \"" << run_id << "\",\n";
    ss << "  \"timestamp\": \"" << timestamp << "\",\n";
    ss << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n";
    ss << "  \"energy\": {\n";
    ss << "    \"wall_time_s\": " << energy.wall_time_s << ",\n";
    ss << "    \"cpu_power_w\": " << energy.cpu_power_w << ",\n";
    ss << "    \"gpu_power_w\": " << energy.gpu_power_w << ",\n";
    ss << "    \"n_cpus\": " << energy.n_cpus << ",\n";
    ss << "    \"n_gpus\": " << energy.n_gpus << ",\n";
    ss << "    \"energy_joules\": " << energy.energy_joules() << ",\n";
    ss << "    \"energy_kwh\": " << energy.energy_kwh() << "\n";
    ss << "  },\n";
    ss << "  \"metrics\": [\n";
    for (std::size_t i = 0; i < metrics.size(); ++i) {
      const auto& m = metrics[i];
      ss << "    {\n";
      ss << "      \"name\": \"" << m.name << "\",\n";
      ss << "      \"unit\": \"" << m.unit << "\",\n";
      ss << "      \"budget\": " << m.budget << ",\n";
      ss << "      \"n_points\": " << m.history.size() << ",\n";
      if (!m.history.empty()) {
        ss << "      \"latest\": " << m.history.back().value << ",\n";
        ss << "      \"mean\": " << m.mean(10) << ",\n";
        ss << "      \"stddev\": " << m.stddev(10) << ",\n";
        ss << "      \"regression\": " << (m.is_regression() ? "true" : "false") << "\n";
      } else {
        ss << "      \"latest\": null\n";
      }
      ss << "    }" << (i + 1 < metrics.size() ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
  }
};

// Helper: create a timestamp string.
inline std::string MakeTimestamp() {
  std::time_t now = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  return ts;
}

// Helper: add a data point to a metric.
inline void AddPoint(RegressionMetric& metric, double value,
                     const std::string& git_hash = "") {
  RegressionPoint point;
  point.timestamp = MakeTimestamp();
  point.value = value;
  point.passed = (metric.budget > 0) ? (value <= metric.budget) : true;
  point.git_hash = git_hash;
  metric.history.push_back(point);
}

}  // namespace tides::verification
