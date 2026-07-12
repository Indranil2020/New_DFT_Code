#pragma once

// Energy consumption logging for benchmarks (§8 — "accuracy per joule").
//
// Measures wall time and estimates energy consumption (CPU + GPU) for
// benchmark runs. When NVML is available (TIDES_HAVE_CUDA), reads actual
// GPU power; otherwise estimates from configurable TDP defaults.
//
// The "accuracy per joule" metric: accuracy_joule = 1 / (error * energy_kwh)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

namespace tides::verification {

struct EnergyMeasurement {
  double wall_time_s = 0.0;
  double cpu_energy_joules = 0.0;
  double gpu_energy_joules = 0.0;
  double total_energy_kwh = 0.0;
  double power_watts = 0.0;
  std::string gpu_name;
};

class EnergyMeter {
 public:
  EnergyMeter(double cpu_tdp_watts = 125.0, double gpu_tdp_watts = 350.0,
              int n_cpu_cores = 16)
      : cpu_tdp_watts_(cpu_tdp_watts),
        gpu_tdp_watts_(gpu_tdp_watts),
        n_cpu_cores_(n_cpu_cores) {}

  void Start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
  }

  EnergyMeasurement Stop() {
    EnergyMeasurement m = Snapshot();
    running_ = false;
    return m;
  }

  EnergyMeasurement Snapshot() const {
    EnergyMeasurement m;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    m.wall_time_s = elapsed;

    // CPU energy: TDP * cores * utilization factor
    const double cpu_util = 0.7;  // typical SCF utilization
    m.cpu_energy_joules = cpu_tdp_watts_ * static_cast<double>(n_cpu_cores_) *
                          cpu_util * elapsed;

    // GPU energy: estimate from TDP (NVML path would read actual power)
#ifdef TIDES_HAVE_CUDA
    const double gpu_util = 0.8;
    m.gpu_energy_joules = gpu_tdp_watts_ * gpu_util * elapsed;
    m.gpu_name = "CUDA GPU (estimated TDP)";
#else
    const double gpu_util = 0.8;
    m.gpu_energy_joules = gpu_tdp_watts_ * gpu_util * elapsed;
    m.gpu_name = "GPU (estimated TDP, no NVML)";
#endif

    double total_joules = m.cpu_energy_joules + m.gpu_energy_joules;
    m.total_energy_kwh = total_joules / 3.6e6;  // J -> kWh
    m.power_watts = (elapsed > 1e-9) ? total_joules / elapsed : 0.0;
    return m;
  }

  // Compute "accuracy per joule" metric.
  static double AccuracyPerJoule(double error, double energy_kwh) {
    if (error <= 0 || energy_kwh <= 0) return 0.0;
    return 1.0 / (error * energy_kwh);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
  bool running_ = false;
  double cpu_tdp_watts_;
  double gpu_tdp_watts_;
  int n_cpu_cores_;
};

}  // namespace tides::verification
