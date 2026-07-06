// E5: SCF Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles:
//   1. SCF driver convergence (Pulay vs simple mixing)
//   2. Energy assembly components
//   3. Stress tensor (FD vs virial)
//   4. SCF on a model problem (2-level system)

#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "scf/stress.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::scf::SCFDriver;
using tides::scf::SCFResult;
using tides::scf::EnergyAssembly;
using tides::scf::EnergyComponents;
using tides::scf::StressTensor;
using tides::solvers::BatchedDenseEig;

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

// --- Test 1: SCF convergence on a model problem ---
int TestSCFConvergence() {
  std::cout << "\n=== E5.1: SCF Convergence (model problem) ===\n";
  PrintHeader();
  int failures = 0;

  // Model: 4-level system with S=I, H = diag(1, 2, 3, 4) + density-dependent shift.
  // The SCF should converge to the ground state with 2 electrons.
  for (auto n : {8, 16, 32, 64}) {
    std::size_t n_occ = n / 4;  // quarter fill

    // S = identity.
    std::vector<double> S(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;

    // Base Hamiltonian: tridiagonal [-1, 2, -1].
    std::vector<double> H0(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      H0[i * n + i] = 2.0;
      if (i + 1 < n) {
        H0[i * n + (i + 1)] = -1.0;
        H0[(i + 1) * n + i] = -1.0;
      }
    }

    // build_H: H = H0 + alpha * P (mean-field coupling).
    auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
      std::vector<double> H = H0;
      for (std::size_t i = 0; i < n * n; ++i)
        H[i] += 0.1 * P[i];
      return H;
    };

    // energy_fn: E = Tr(P H) - 0.05 * Tr(P^2) (double-counting correction).
    auto energy_fn = [&](const std::vector<double>& P) -> double {
      auto H = build_H(P);
      double e = 0.0;
      for (std::size_t i = 0; i < n * n; ++i)
        e += P[i] * H[i];
      // Subtract half the mean-field contribution (double counting).
      for (std::size_t i = 0; i < n * n; ++i)
        e -= 0.05 * P[i] * P[i];
      return e;
    };

    std::vector<double> P_init(n * n, 0.0);
    for (std::size_t i = 0; i < n_occ; ++i)
      P_init[i * n + i] = 2.0;

    // Pulay mixing.
    auto t0 = std::chrono::steady_clock::now();
    auto result_pulay = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                        P_init, 200, 1e-8, 1, 0.3);
    auto t1 = std::chrono::steady_clock::now();
    double pulay_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Simple mixing.
    auto t2 = std::chrono::steady_clock::now();
    auto result_simple = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                         P_init, 200, 1e-8, 0, 0.3);
    auto t3 = std::chrono::steady_clock::now();
    double simple_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Check convergence.
    double pulay_err = std::abs(result_pulay.energy_history.back() -
                                 result_pulay.energy_history[result_pulay.energy_history.size() - 2]);
    double simple_err = std::abs(result_simple.energy_history.back() -
                                  result_simple.energy_history[result_simple.energy_history.size() - 2]);

    std::string pulay_status = (result_pulay.converged && pulay_err < 1e-6) ? "PASS" : "FAIL";
    std::string simple_status = (result_simple.converged && simple_err < 1e-6) ? "PASS" : "FAIL";
    if (!result_pulay.converged || pulay_err >= 1e-6) failures++;
    if (!result_simple.converged || simple_err >= 1e-6) failures++;

    Log("SCF", "Pulay",
        "n=" + std::to_string(n),
        pulay_ms, pulay_err, pulay_status + " (" + std::to_string(result_pulay.n_iterations) + " iters)");
    Log("SCF", "simple",
        "n=" + std::to_string(n),
        simple_ms, simple_err, simple_status + " (" + std::to_string(result_simple.n_iterations) + " iters)");
  }
  return failures;
}

// --- Test 2: Energy assembly ---
int TestEnergyAssembly() {
  std::cout << "\n=== E5.2: Energy Assembly ===\n";
  PrintHeader();
  int failures = 0;

  for (auto n : {8, 16, 32}) {
    std::vector<double> P(n * n, 0.0);
    std::vector<double> V_H(n * n, 0.0);
    std::vector<double> V_xc(n * n, 0.0);
    std::vector<double> eps_xc_mat(n * n, 0.0);
    std::vector<double> V_ext(n * n, 0.0);
    std::vector<double> S(n * n, 0.0);

    // Fill with simple values.
    for (std::size_t i = 0; i < n; ++i) {
      P[i * n + i] = 1.0;
      V_H[i * n + i] = 0.5;
      V_xc[i * n + i] = -0.3;
      eps_xc_mat[i * n + i] = -0.35;
      V_ext[i * n + i] = -1.0;
      S[i * n + i] = 1.0;
    }

    double sum_eps = static_cast<double>(n) * (-0.5);
    double E_ion = 0.1;

    auto t0 = std::chrono::steady_clock::now();
    auto E = EnergyAssembly::Compute(sum_eps, P, V_H, V_xc, eps_xc_mat, V_ext, S, n, E_ion);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Verify E_total = sum of components.
    double component_sum = E.E_kin + E.E_ne + E.E_H + E.E_xc + E.E_ion;
    double err = std::abs(E.E_total - component_sum);

    std::string status = (err < 1e-14) ? "PASS" : "FAIL";
    if (err >= 1e-14) failures++;
    Log("EnergyAssembly", "components",
        "n=" + std::to_string(n),
        ms, err, status);
  }
  return failures;
}

// --- Test 3: Stress tensor (FD vs virial) ---
int TestStressTensor() {
  std::cout << "\n=== E5.3: Stress Tensor (FD vs virial) ===\n";
  PrintHeader();
  int failures = 0;

  // Model: harmonic oscillator energy E = sum 0.5 * k * r^2.
  // Force F = -k * r. Virial stress = sum F_i * r_i / V.
  int n_atoms = 4;
  double V = 100.0;
  double k = 1.0;

  std::vector<double> positions = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> forces(3 * n_atoms);
  for (int i = 0; i < n_atoms; ++i)
    for (int c = 0; c < 3; ++c)
      forces[3 * i + c] = -k * positions[3 * i + c];

  // Virial stress (note: sigma = -(1/V) dE/de, and virial = (1/V) sum F*r,
  // so FD stress = -virial).
  auto virial = StressTensor::VirialStress(forces, positions, V);
  for (auto& v : virial) v = -v;

  // FD stress.
  auto energy_fn = [&](const std::vector<double>& strain) -> double {
    // Apply strain to positions, compute energy.
    std::vector<double> strained = positions;
    for (int i = 0; i < n_atoms; ++i)
      for (int a = 0; a < 3; ++a) {
        double s = 0.0;
        for (int b = 0; b < 3; ++b)
          s += strain[a * 3 + b] * positions[3 * i + b];
        strained[3 * i + a] = positions[3 * i + a] + s;
      }
    double E = 0.0;
    for (int i = 0; i < n_atoms; ++i) {
      double r2 = 0.0;
      for (int c = 0; c < 3; ++c) r2 += strained[3 * i + c] * strained[3 * i + c];
      E += 0.5 * k * r2;
    }
    return E;
  };

  auto t0 = std::chrono::steady_clock::now();
  auto fd = StressTensor::ComputeFD(energy_fn, V, 1e-5);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Compare FD vs virial.
  double max_err = 0.0;
  for (int i = 0; i < 9; ++i)
    max_err = std::max(max_err, std::abs(fd[i] - virial[i]));

  std::string status = (max_err < 1e-4) ? "PASS" : "FAIL";
  if (max_err >= 1e-4) failures++;
  Log("Stress", "FD-vs-virial",
      "4 atoms",
      ms, max_err, status);

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E5 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E5 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E5 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E5: SCF Engine — Test & Profile Suite                      ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestSCFConvergence();
  failures += TestEnergyAssembly();
  failures += TestStressTensor();

  PrintSummary(failures);
  return failures;
}
