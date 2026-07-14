#pragma once

// Energy consumption logging for benchmarks (§8 — "accuracy per joule").
//
// Measures wall time and estimates energy consumption (CPU + GPU) for
// benchmark runs. When NVML is available, reads actual GPU power via
// dlopen runtime loading; otherwise estimates from configurable TDP defaults.
//
// The "accuracy per joule" metric: accuracy_joule = 1 / (error * energy_kwh)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace tides::verification {

struct EnergyMeasurement {
  double wall_time_s = 0.0;
  double cpu_energy_joules = 0.0;
  double gpu_energy_joules = 0.0;
  double total_energy_kwh = 0.0;
  double power_watts = 0.0;
  std::string gpu_name;
  bool used_nvmL = false;
  double gpu_power_sample_w = 0.0;
};

class EnergyMeter {
 public:
  EnergyMeter(double cpu_tdp_watts = 125.0, double gpu_tdp_watts = 350.0,
              int n_cpu_cores = 16)
      : cpu_tdp_watts_(cpu_tdp_watts),
        gpu_tdp_watts_(gpu_tdp_watts),
        n_cpu_cores_(n_cpu_cores) {
    NvmlInit();
  }

  ~EnergyMeter() {
    NvmlShutdown();
  }

  EnergyMeter(const EnergyMeter&) = delete;
  EnergyMeter& operator=(const EnergyMeter&) = delete;

  void Start() {
    start_time_ = std::chrono::steady_clock::now();
    start_gpu_power_w_ = NvmlReadPowerW();
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

    // GPU energy: read actual power via NVML if available.
    // Trapezoidal approximation: average power at start and now.
    double gpu_power_w = NvmlReadPowerW();
    m.used_nvmL = (gpu_power_w >= 0.0);
    if (m.used_nvmL) {
      double avg_power = (start_gpu_power_w_ >= 0.0)
                             ? 0.5 * (start_gpu_power_w_ + gpu_power_w)
                             : gpu_power_w;
      m.gpu_energy_joules = avg_power * elapsed;
      m.gpu_power_sample_w = gpu_power_w;
      m.gpu_name = nvml_gpu_name_.empty() ? "CUDA GPU (NVML)" : nvml_gpu_name_;
    } else {
      // Fallback: TDP-based estimate
      const double gpu_util = 0.8;
      m.gpu_energy_joules = gpu_tdp_watts_ * gpu_util * elapsed;
      m.gpu_power_sample_w = gpu_tdp_watts_ * gpu_util;
      m.gpu_name = "GPU (estimated TDP, no NVML)";
    }

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

  bool nvml_available() const { return nvml_handle_ != nullptr; }

 private:
  std::chrono::steady_clock::time_point start_time_;
  bool running_ = false;
  double cpu_tdp_watts_;
  double gpu_tdp_watts_;
  int n_cpu_cores_;
  double start_gpu_power_w_ = -1.0;

  // NVML runtime-loaded state
  void* nvml_handle_ = nullptr;
  std::string nvml_gpu_name_;

  // NVML function pointer types
  using NvmlInit_t = int (*)(void);
  using NvmlShutdown_t = int (*)(void);
  using NvmlDeviceGetHandleByIndex_t = int (*)(unsigned int, void*);
  using NvmlDeviceGetPowerUsage_t = int (*)(void*, unsigned int*);
  using NvmlDeviceGetName_t = int (*)(void*, char*, unsigned int);

  NvmlInit_t nvml_init_ = nullptr;
  NvmlShutdown_t nvml_shutdown_ = nullptr;
  NvmlDeviceGetHandleByIndex_t nvml_get_handle_ = nullptr;
  NvmlDeviceGetPowerUsage_t nvml_get_power_ = nullptr;
  NvmlDeviceGetName_t nvml_get_name_ = nullptr;

  void* nvml_device_ = nullptr;

  void NvmlInit() {
#ifdef __linux__
    nvml_handle_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (!nvml_handle_)
      nvml_handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!nvml_handle_) return;

    nvml_init_ = reinterpret_cast<NvmlInit_t>(dlsym(nvml_handle_, "nvmlInit_v2"));
    nvml_shutdown_ = reinterpret_cast<NvmlShutdown_t>(dlsym(nvml_handle_, "nvmlShutdown"));
    nvml_get_handle_ = reinterpret_cast<NvmlDeviceGetHandleByIndex_t>(
        dlsym(nvml_handle_, "nvmlDeviceGetHandleByIndex"));
    nvml_get_power_ = reinterpret_cast<NvmlDeviceGetPowerUsage_t>(
        dlsym(nvml_handle_, "nvmlDeviceGetPowerUsage"));
    nvml_get_name_ = reinterpret_cast<NvmlDeviceGetName_t>(
        dlsym(nvml_handle_, "nvmlDeviceGetName"));

    if (!nvml_init_ || !nvml_shutdown_ || !nvml_get_handle_ ||
        !nvml_get_power_ || !nvml_get_name_) {
      dlclose(nvml_handle_);
      nvml_handle_ = nullptr;
      return;
    }

    if (nvml_init_() != 0) {  // NVML_SUCCESS = 0
      dlclose(nvml_handle_);
      nvml_handle_ = nullptr;
      return;
    }

    // Get device handle for GPU 0
    if (nvml_get_handle_(0, &nvml_device_) != 0) {
      nvml_shutdown_();
      dlclose(nvml_handle_);
      nvml_handle_ = nullptr;
      return;
    }

    // Get GPU name
    char name[96] = {};
    if (nvml_get_name_(nvml_device_, name, sizeof(name)) == 0) {
      nvml_gpu_name_ = std::string(name);
    }
#endif
  }

  void NvmlShutdown() {
#ifdef __linux__
    if (nvml_handle_) {
      if (nvml_shutdown_) nvml_shutdown_();
      dlclose(nvml_handle_);
      nvml_handle_ = nullptr;
    }
#endif
  }

  // Returns current GPU power in watts, or -1.0 if NVML unavailable.
  double NvmlReadPowerW() const {
#ifdef __linux__
    if (!nvml_handle_ || !nvml_get_power_ || !nvml_device_) return -1.0;
    unsigned int power_mw = 0;
    if (nvml_get_power_(nvml_device_, &power_mw) != 0) return -1.0;
    return static_cast<double>(power_mw) / 1000.0;  // mW -> W
#else
    return -1.0;
#endif
  }
};

}  // namespace tides::verification
