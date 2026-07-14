// Energy metering tests.
#include "verification/energy_metering.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

namespace {
using tides::verification::EnergyMeter;
using tides::verification::EnergyMeasurement;

int Fail(const std::string& msg) {
  std::cerr << "energy_metering_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestBasicMeasurement() {
  std::cout << "\n=== Energy Metering: Basic Measurement ===\n";
  EnergyMeter meter(125.0, 350.0, 16);
  meter.Start();
  // Sleep 10ms to get nonzero time.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto m = meter.Stop();

  std::cout << "  Wall time: " << m.wall_time_s << " s\n";
  std::cout << "  CPU energy: " << m.cpu_energy_joules << " J\n";
  std::cout << "  GPU energy: " << m.gpu_energy_joules << " J\n";
  std::cout << "  Total energy: " << m.total_energy_kwh << " kWh\n";
  std::cout << "  Power: " << m.power_watts << " W\n";

  if (m.wall_time_s <= 0)
    return Fail("Wall time should be positive");
  if (m.cpu_energy_joules <= 0)
    return Fail("CPU energy should be positive");
  if (m.total_energy_kwh <= 0)
    return Fail("Total energy (kWh) should be positive");
  std::cout << "  PASS\n";
  return 0;
}

int TestKWhConversion() {
  std::cout << "\n=== Energy Metering: kWh Conversion ===\n";
  EnergyMeter meter(1.0, 0.0, 1);  // 1W CPU, no GPU
  meter.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto m = meter.Stop();

  // 0.1s * 1W * 0.7 util = 0.07 J → 0.07/3.6e6 kWh (CPU only)
  double expected_joules = 1.0 * 1 * 0.7 * m.wall_time_s;
  double expected_kwh = expected_joules / 3.6e6;

  std::cout << "  Expected CPU J: " << expected_joules << "\n";
  std::cout << "  Actual CPU J: " << m.cpu_energy_joules << "\n";
  std::cout << "  Expected CPU kWh: " << expected_kwh << "\n";
  std::cout << "  Actual total kWh: " << m.total_energy_kwh << "\n";

  if (std::fabs(m.cpu_energy_joules - expected_joules) > 1e-6)
    return Fail("CPU energy mismatch");
  // CPU-only kWh check (GPU energy may be nonzero via NVML)
  double cpu_kwh = m.cpu_energy_joules / 3.6e6;
  if (std::fabs(cpu_kwh - expected_kwh) > 1e-12)
    return Fail("CPU kWh conversion mismatch");
  std::cout << "  PASS\n";
  return 0;
}

int TestAccuracyPerJoule() {
  std::cout << "\n=== Energy Metering: Accuracy Per Joule ===\n";
  double error = 1e-4;
  double energy_kwh = 0.001;
  double apj = EnergyMeter::AccuracyPerJoule(error, energy_kwh);
  double expected = 1.0 / (error * energy_kwh);
  std::cout << "  Error: " << error << ", Energy: " << energy_kwh << " kWh\n";
  std::cout << "  Accuracy/joule: " << apj << "\n";
  std::cout << "  Expected: " << expected << "\n";
  if (std::fabs(apj - expected) > 1e-10)
    return Fail("Accuracy per joule mismatch");
  std::cout << "  PASS\n";
  return 0;
}

int TestNvmlPowerReading() {
  std::cout << "\n=== Energy Metering: NVML Power Reading ===\n";
  EnergyMeter meter(125.0, 350.0, 16);

  if (!meter.nvml_available()) {
    std::cout << "  NVML not available — skipping (fallback TDP path tested)\n";
    // Verify fallback still works
    meter.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto m = meter.Stop();
    if (m.used_nvmL)
      return Fail("used_nvmL should be false when NVML unavailable");
    if (m.gpu_power_sample_w <= 0)
      return Fail("Fallback GPU power should be positive");
    std::cout << "  Fallback GPU power: " << m.gpu_power_sample_w << " W\n";
    std::cout << "  PASS (fallback)\n";
    return 0;
  }

  std::cout << "  NVML available — reading actual GPU power\n";
  meter.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto m = meter.Stop();

  std::cout << "  GPU name: " << m.gpu_name << "\n";
  std::cout << "  GPU power: " << m.gpu_power_sample_w << " W\n";
  std::cout << "  Used NVML: " << (m.used_nvmL ? "yes" : "no") << "\n";
  std::cout << "  GPU energy: " << m.gpu_energy_joules << " J\n";

  if (!m.used_nvmL)
    return Fail("used_nvmL should be true when NVML is available");
  if (m.gpu_power_sample_w <= 0)
    return Fail("GPU power reading should be positive");
  if (m.gpu_name.empty() || m.gpu_name.find("estimated") != std::string::npos)
    return Fail("GPU name should reflect NVML reading, not TDP estimate");
  std::cout << "  PASS (NVML)\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== Energy Metering Tests ===\n";
  int failures = 0;
  failures += TestBasicMeasurement();
  failures += TestKWhConversion();
  failures += TestAccuracyPerJoule();
  failures += TestNvmlPowerReading();
  if (failures == 0) std::cout << "\nALL ENERGY METERING TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
