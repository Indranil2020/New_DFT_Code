// E6: Forces & Dynamics Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles:
//   1. Analytic forces vs FD validation
//   2. XL-BOMD NVE drift
//   3. Geometry optimizers (FIRE, L-BFGS)
//   4. NVE energy conservation

#include "forces/analytic_forces.hpp"
#include "dynamics/xlbomd/xlbomd.hpp"
#include "dynamics/optimizers/optimizers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::forces::AnalyticForces;
using tides::forces::ForceResult;
using tides::dynamics::XLBOMD;
using tides::dynamics::XLBOMDResult;
using tides::dynamics::Optimizers;
using tides::dynamics::OptimizationResult;

struct ProfileEntry {
  std::string kernel;
  std::string variant;
  std::string size_label;
  double time_ms = 0.0;
  double error = 0.0;
  std::string status;
};

std::vector<ProfileEntry> g_log;

void Log(const std::string& kernel, const std::string& variant,
         const std::string& size_label, double time_ms, double error,
         const std::string& status) {
  ProfileEntry e{kernel, variant, size_label, time_ms, error, status};
  g_log.push_back(e);
  std::cout << "  " << std::left << std::setw(18) << kernel
            << std::setw(12) << variant
            << std::setw(14) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << error
            << "  " << status << '\n';
}

void PrintHeader() {
  std::cout << std::left << std::setw(18) << "Kernel"
            << std::setw(12) << "Variant"
            << std::setw(14) << "Size"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(14) << "Error"
            << "  Status\n";
  std::cout << std::string(82, '-') << '\n';
}

// --- Test 1: Analytic forces vs FD ---
int TestAnalyticForces() {
  std::cout << "\n=== E6.1: Analytic Forces vs FD ===\n";
  PrintHeader();
  int failures = 0;

  // Model: E(R) = sum_i 0.5 * k * |R_i|^2 (harmonic only, no Coulomb).
  // Force on atom I: F_I = -k * R_I.
  int n_atoms = 4;
  double k = 1.0;

  std::vector<double> positions = {1.0, 0.5, 0.0, 2.0, 0.0, 0.0,
                                    0.0, 2.0, 0.0, 1.0, 1.0, 1.0};

  auto energy_fn = [&](const std::vector<double>& R) -> double {
    double E = 0.0;
    for (int i = 0; i < n_atoms; ++i) {
      double r2 = 0.0;
      for (int c = 0; c < 3; ++c) r2 += R[3*i+c] * R[3*i+c];
      E += 0.5 * k * r2;
    }
    return E;
  };

  // Analytic forces: F = -k * R.
  std::vector<double> forces(3 * n_atoms, 0.0);
  for (int i = 0; i < n_atoms; ++i)
    for (int c = 0; c < 3; ++c)
      forces[3*i + c] = -k * positions[3*i + c];

  // Validate against FD.
  auto t0 = std::chrono::steady_clock::now();
  auto result = AnalyticForces::Validate(forces, energy_fn, positions, n_atoms, 0.001, 1e-6);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::string status = (result.fd_validated) ? "PASS" : "FAIL";
  if (!result.fd_validated) failures++;
  Log("Forces", "analytic-vs-FD",
      "4 atoms",
      ms, result.max_fd_error, status);

  return failures;
}

// --- Test 2: XL-BOMD NVE drift ---
int TestXLBOMD() {
  std::cout << "\n=== E6.2: XL-BOMD NVE Energy Conservation ===\n";
  PrintHeader();
  int failures = 0;

  // Model: 2-atom harmonic oscillator.
  int n_atoms = 2;
  std::vector<double> masses = {1.0, 1.0};
  std::vector<double> init_R = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};

  // Harmonic force: F = -k * (R - R_eq).
  // Use small k so that period >> dt for stable Verlet.
  // T = 2*pi*sqrt(m/k). For m=1, k=0.01: T ~ 628 a.u. ~ 15fs.
  // dt=1.0fs gives ~15 steps per period — stable.
  double k = 0.01;
  double R_eq = 1.4;
  auto force_fn = [&](const std::vector<double>& R, const std::vector<double>& P) -> std::vector<double> {
    std::vector<double> F(6, 0.0);
    double dr = R[0] - R[3] - R_eq;  // displacement along x
    F[0] = -k * dr;
    F[3] = k * dr;
    return F;
  };

  auto energy_fn = [&](const std::vector<double>& R) -> double {
    double dr = R[0] - R[3] - R_eq;
    return 0.5 * k * dr * dr;
  };

  auto density_fn = [&](const std::vector<double>& R) -> std::vector<double> {
    // Dummy density matrix (1x1).
    return {1.0};
  };

  // AUDIT P3: Run 50000 steps NVE at dt=0.2 fs = 10 ps (audit requires ≥10 ps).
  // Previous: 100 steps at 0.1 fs (0.01 ps) — far too short, drift dominated
  // by initial transient. dt=0.2fs = 8.27 a.u. is safely within Verlet stability
  // (bound: dt < 2/sqrt(k/m) = 20 a.u.). dt=0.4fs (16.5 a.u.) was too close to
  // the bound and diverged to NaN after ~15000 steps.
  const double dt_fs = 0.2;
  const int n_steps = 50000;
  auto t0 = std::chrono::steady_clock::now();
  auto result = XLBOMD::Run(init_R, masses, dt_fs, n_steps,
                              force_fn, energy_fn, density_fn, 0, 0.0);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Check NVE drift.
  double drift = result.total_drift;
  // AUDIT A2: tolerances.yaml rung5_nve_drift = 30.0 uHa/atom/ps.
  // With 50000 steps at 0.2 fs (10 ps), the drift rate should be stable.
  // If it still fails, the root cause is algorithmic, not simulation length.
  const double DRIFT_BUDGET = 30.0;  // uHa/atom/ps per tolerances.yaml
  std::string status = (std::isfinite(drift) && drift < DRIFT_BUDGET) ? "PASS" : "FAIL";
  if (!std::isfinite(drift) || drift >= DRIFT_BUDGET) failures++;
  Log("XLBOMD", "NVE-drift",
      "2 atoms, 50000 steps, dt=0.2fs (10ps)",
      ms, drift, status);

  // Check solves per step = 1.
  double sps = result.avg_solves_per_step;
  std::string sps_status = (std::abs(sps - 1.0) < 0.02) ? "PASS" : "FAIL";
  if (std::abs(sps - 1.0) >= 0.02) failures++;
  Log("XLBOMD", "solves/step",
      "2 atoms",
      0, std::abs(sps - 1.0), sps_status);

  return failures;
}

// --- Test 3: Geometry optimizers ---
int TestOptimizers() {
  std::cout << "\n=== E6.3: Geometry Optimizers ===\n";
  PrintHeader();
  int failures = 0;

  // Model: 2-atom harmonic oscillator, minimize to R_eq = 1.4.
  int n_atoms = 2;
  std::vector<double> init_R = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  double k = 1.0;
  double R_eq = 1.4;

  auto force_fn = [&](const std::vector<double>& R) -> std::vector<double> {
    std::vector<double> F(6, 0.0);
    double dr = R[0] - R[3] - R_eq;
    F[0] = -k * dr;
    F[3] = k * dr;
    return F;
  };

  auto energy_fn = [&](const std::vector<double>& R) -> double {
    double dr = R[0] - R[3] - R_eq;
    return 0.5 * k * dr * dr;
  };

  // FIRE optimizer.
  std::vector<double> masses(n_atoms, 1.0);
  auto t0 = std::chrono::steady_clock::now();
  auto fire_result = Optimizers::FIRE(init_R, masses, energy_fn, force_fn, 200, 1e-6);
  auto t1 = std::chrono::steady_clock::now();
  double fire_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Check convergence to R_eq.
  double fire_dr = fire_result.final_positions[0] - fire_result.final_positions[3];
  double fire_err = std::abs(fire_dr - R_eq);

  std::string fire_status = (fire_result.converged && fire_err < 1e-3) ? "PASS" : "FAIL";
  if (!fire_result.converged || fire_err >= 1e-3) failures++;
  Log("Optimizer", "FIRE",
      "2 atoms",
      fire_ms, fire_err, fire_status + " (" + std::to_string(fire_result.n_steps) + " steps)");

  // L-BFGS not implemented yet; skip.
  // TODO: add when LBFGS optimizer is available.

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E6 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E6 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E6 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E6: Forces & Dynamics Engine — Test & Profile Suite        ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestAnalyticForces();
  failures += TestXLBOMD();
  failures += TestOptimizers();

  PrintSummary(failures);
  return failures;
}
