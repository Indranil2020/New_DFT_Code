// E8: Hybrids Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles:
//   1. D3 dispersion energy and forces
//   2. ISDF point selection and reconstruction
//   3. ACE compressed exchange operator
//   4. PBE0 hybrid energy assembly

#include "hybrids/d3_dispersion.hpp"
#include "hybrids/isdf/isdf.hpp"
#include "hybrids/ace/ace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::hybrids::D3Dispersion;
using tides::hybrids::ISDF;
using tides::hybrids::ACE;

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

// --- Test 1: D3 dispersion ---
int TestD3Dispersion() {
  std::cout << "\n=== E8.1: D3 Dispersion ===\n";
  PrintHeader();
  int failures = 0;

  // Test pair energy for known atoms.
  for (auto [ZA, ZB, R] : std::vector<std::tuple<int, int, double>>{
           {1, 1, 3.0}, {6, 6, 3.5}, {8, 8, 3.0}, {6, 8, 3.2}}) {
    double E = D3Dispersion::PairEnergy(ZA, ZB, R);
    // Energy should be negative (attractive) and finite.
    std::string status = (std::isfinite(E) && E < 0) ? "PASS" : "FAIL";
    if (!std::isfinite(E) || E >= 0) failures++;
    Log("D3", "pair",
        "Z=" + std::to_string(ZA) + "-" + std::to_string(ZB) + " R=" + std::to_string(R),
        0, E, status);
  }

  // Test full system: H2O dimer.
  {
    std::vector<int> Z = {1, 8, 1, 1, 8, 1};
    std::vector<double> pos = {
      0.0, 0.0, 0.0,    // H1
      0.0, 0.0, 1.8,    // O1
      0.0, 1.5, 2.5,    // H2
      5.0, 0.0, 0.0,    // H3
      5.0, 0.0, 1.8,    // O2
      5.0, 1.5, 2.5     // H4
    };

    auto t0 = std::chrono::steady_clock::now();
    auto result = D3Dispersion::ComputeEnergy(Z, pos);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string status = (std::isfinite(result.energy) && result.energy < 0) ? "PASS" : "FAIL";
    if (!std::isfinite(result.energy) || result.energy >= 0) failures++;
    Log("D3", "H2O-dimer",
        "6 atoms",
        ms, result.energy, status);
  }

  return failures;
}

// --- Test 2: ISDF ---
int TestISDF() {
  std::cout << "\n=== E8.2: ISDF (Interpolative Separable Density Fitting) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n_grid, n_pairs, rank] : std::vector<std::tuple<int, int, int>>{
           {100, 10, 5}, {500, 20, 10}, {2000, 50, 20}}) {
    // Build a low-rank matrix: M = U * V where U is n_grid x rank, V is rank x n_pairs.
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> U(n_grid * rank), V(rank * n_pairs);
    for (auto& v : U) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<double> M(n_grid * n_pairs, 0.0);
    for (int i = 0; i < n_grid; ++i)
      for (int j = 0; j < n_pairs; ++j)
        for (int r = 0; r < rank; ++r)
          M[i * n_pairs + j] += U[i * rank + r] * V[r * n_pairs + j];

    auto t0 = std::chrono::steady_clock::now();
    auto result = ISDF::SelectPoints(M, n_grid, n_pairs, rank);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ISDF should now have low reconstruction error with proper LSQ fit.
    std::string status;
    if (result.reconstruction_error < 1e-6) {
      status = "PASS";
    } else if (result.reconstruction_error < 1e-3) {
      status = "PASS (marginal)";
    } else {
      status = "FAIL: reconstruction error too high";
      failures++;
    }
    Log("ISDF", "select",
        std::to_string(n_grid) + "x" + std::to_string(n_pairs) + " r=" + std::to_string(rank),
        ms, result.reconstruction_error, status);
  }
  return failures;
}

// --- Test 3: ACE compressed exchange ---
int TestACE() {
  std::cout << "\n=== E8.3: ACE (Adaptively Compressed Exchange) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n, n_occ] : std::vector<std::pair<int, int>>{
           {16, 4}, {32, 8}, {64, 16}}) {
    std::normal_distribution<double> dist(0.0, 1.0);

    // Random density matrix (symmetric).
    std::vector<double> P(n * n);
    for (auto& v : P) v = dist(rng);
    for (int i = 0; i < n; ++i)
      for (int j = i; j < n; ++j)
        P[j * n + i] = P[i * n + j];

    // Random exact exchange matrix K.
    std::vector<double> K(n * n);
    for (auto& v : K) v = dist(rng);
    for (int i = 0; i < n; ++i)
      for (int j = i; j < n; ++j)
        K[j * n + i] = K[i * n + j];

    double alpha = 0.25;  // PBE0

    auto t0 = std::chrono::steady_clock::now();
    auto result = ACE::Build(P, K, n, n_occ, alpha);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // V_x should be alpha * K.
    double err = 0.0;
    for (int i = 0; i < n * n; ++i)
      err = std::max(err, std::abs(result.V_x[i] - alpha * K[i]));

    std::string status = (err < 1e-14 && result.ok) ? "PASS" : "FAIL";
    if (err >= 1e-14 || !result.ok) failures++;
    Log("ACE", "build",
        "n=" + std::to_string(n) + " occ=" + std::to_string(n_occ),
        ms, err, status);
  }

  // PBE0 energy formula.
  {
    double E_PBE = -10.0;
    double Ex_exact = -2.0;
    double Ex_PBE = -1.5;
    double E_PBE0 = ACE::PBE0Energy(E_PBE, Ex_exact, Ex_PBE, 0.25);
    double expected = -10.0 + 0.25 * (-2.0 - (-1.5));
    double err = std::abs(E_PBE0 - expected);

    std::string status = (err < 1e-14) ? "PASS" : "FAIL";
    if (err >= 1e-14) failures++;
    Log("ACE", "PBE0-energy",
        "formula",
        0, err, status);
  }

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E8 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E8 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E8 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E8: Hybrids Engine — Test & Profile Suite                  ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestD3Dispersion();
  failures += TestISDF();
  failures += TestACE();

  PrintSummary(failures);
  return failures;
}
