// E2: Basis & Integrals Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles every WP2 kernel:
//   1. Radial solver (FD + Numerov) — hydrogenic eigenvalue accuracy + timing
//   2. NAO generation — basis generation + timing
//   3. Two-center integrals (CPU + GPU) — spline accuracy + GPU speedup
//   4. Three-center integrals (GPU) — KB assembly + accuracy
//   5. Derivative streams — FD validation of dS/dR, dH0/dR
//
// All results logged to stdout in structured format.
//
// WAIVER (audit Section E / A7): Bars retightened to tolerances.yaml values
// (spline 1e-5, symmetry 1e-12). The spline two-center integral achieves
// 3.5e-5 (FAIL vs 1e-5 gate) and GPU symmetry is 3.8e-3 (FAIL vs 1e-12 gate).
// These are open correctness defects, not noise. Tests intentionally FAIL
// per audit P0.2: "red tests that tell the truth beat green tests that don't."
// Root cause: spline tabulation resolution and GPU kernel angular coupling
// precision. Fix requires higher-order spline interpolation and improved
// Slater-Koster angular factors in the GPU kernel.

#include "basis/atomgen/radial_solver.hpp"
#include "basis/atomgen/numerov_solver.hpp"
#include "basis/atomgen/lda_xc.hpp"
#include "basis/atomgen/atomic_lda.hpp"
#include "basis/nao_generator.hpp"
#include "basis/two_center_integrals.hpp"
#include "basis/two_center_gpu.hpp"
#include "basis/three_center_gpu.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::atomgen::RadialSolver;
using tides::atomgen::NumerovSolver;
using tides::basis::CubicSpline;
using tides::basis::AssembleTwoCenterCuda;
using tides::basis::TwoCenterCudaAvailable;

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
  std::cout << "  " << std::left << std::setw(22) << kernel
            << std::setw(14) << variant
            << std::setw(16) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << error
            << "  " << status << '\n';
}

void PrintHeader() {
  std::cout << std::left << std::setw(22) << "Kernel"
            << std::setw(14) << "Variant"
            << std::setw(16) << "Size"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(14) << "Error"
            << "  Status\n";
  std::cout << std::string(90, '-') << '\n';
}

// --- Test 1: Radial Solver (FD + Numerov) ---
int TestRadialSolver() {
  std::cout << "\n=== E2.1: Radial Solver (FD vs Numerov) ===\n";
  PrintHeader();
  int failures = 0;

  // Hydrogenic Z=1: exact eigenvalues are -1/(2n^2).
  // For l=0: ground state is 1s at -0.5.
  // For l=1: ground state is 2p at -0.125.
  for (auto [l, exact_e] : std::vector<std::pair<int, double>>{
           {0, -0.5}, {1, -0.125}}) {
    for (auto [n_r, label] : std::vector<std::pair<std::size_t, std::string>>{
             {2000, "2k"}, {8000, "8k"}, {16000, "16k"}, {32000, "32k"}}) {
      // FD solver.
      auto t0 = std::chrono::steady_clock::now();
      auto fd_states = RadialSolver::SolveHydrogenic(1, l, 3, 60.0, n_r);
      auto t1 = std::chrono::steady_clock::now();
      double fd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      double fd_err = std::abs(fd_states[0].epsilon - exact_e);

      // Numerov solver.
      auto t2 = std::chrono::steady_clock::now();
      auto num_states = NumerovSolver::SolveHydrogenic(1, l, 3, 60.0, n_r);
      auto t3 = std::chrono::steady_clock::now();
      double num_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

      double num_err = std::abs(num_states[0].epsilon - exact_e);

      // FD is O(h^2): tolerance scales as 1/n_r^2.
      // Constant ~500 due to Coulomb singularity at r=0 for l=0.
      double tol = 1000.0 / static_cast<double>(n_r * n_r);
      std::string fd_status = (fd_err < tol) ? "PASS" : "FAIL";
      std::string num_status = (num_err < tol) ? "PASS" : "FAIL";
      if (fd_err >= tol) failures++;
      if (num_err >= tol) failures++;

      Log("RadialSolver", "FD",
          "l=" + std::to_string(l) + " n=" + label,
          fd_ms, fd_err, fd_status);
      Log("RadialSolver", "Numerov",
          "l=" + std::to_string(l) + " n=" + label,
          num_ms, num_err, num_status);
    }
  }
  return failures;
}

// --- Test 2: NAO Generation ---
int TestNaoGeneration() {
  std::cout << "\n=== E2.2: NAO Generation ===\n";
  PrintHeader();
  int failures = 0;

  for (auto [Z, label] : std::vector<std::pair<int, std::string>>{
           {1, "H"}, {6, "C"}, {8, "O"}, {10, "Ne"}}) {
    tides::basis::NaoRecipe recipe = tides::basis::NaoGenerator::DzpRecipe(Z, label);
    recipe.r_max = 10.0;
    recipe.n_r = 4000;

    auto t0 = std::chrono::steady_clock::now();
    auto basis = tides::basis::NaoGenerator::Generate(recipe);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check basis has functions.
    double err = 0.0;
    std::string status = "PASS";
    if (basis.functions.empty()) {
      status = "FAIL: no radial functions";
      failures++;
    }

    Log("NAOGenerator", "DZP",
        label + " (Z=" + std::to_string(Z) + ")",
        ms, err, status);
  }
  return failures;
}

// --- Test 3: Two-center integrals (spline + GPU) ---
int TestTwoCenter() {
  std::cout << "\n=== E2.3: Two-center Integrals (spline + GPU) ===\n";
  PrintHeader();
  int failures = 0;

  // Build a simple Gaussian overlap spline.
  // S(R) = exp(-0.2 * R^2) — a model overlap integral.
  std::vector<double> R_tab(200), S_tab(200), T_tab(200);
  for (std::size_t i = 0; i < 200; ++i) {
    double R = 0.05 * static_cast<double>(i);
    R_tab[i] = R;
    S_tab[i] = std::exp(-0.2 * R * R);
    T_tab[i] = (3.0 - 0.4 * R * R) * std::exp(-0.2 * R * R);
  }

  CubicSpline s_spline(R_tab, S_tab);
  CubicSpline t_spline(R_tab, T_tab);

  // Test spline accuracy at known points.
  double spline_err = 0.0;
  for (std::size_t i = 0; i < 200; ++i) {
    double R = R_tab[i] + 0.0123;  // off-grid
    double exact = std::exp(-0.2 * R * R);
    double interp = s_spline.Eval(R);
    spline_err = std::max(spline_err, std::abs(interp - exact));
  }

  // AUDIT A7: tolerances.yaml spline_value = 1.0e-5 (not 1e-4).
  // Previous bar was set to what the code achieves, not the gate.
  std::string spline_status = (spline_err < 1e-5) ? "PASS" : "FAIL";
  if (spline_err >= 1e-5) failures++;
  Log("TwoCenter", "spline",
      "200pts", 0, spline_err, spline_status);

  // GPU two-center assembly.
  if (TwoCenterCudaAvailable()) {
    // 2-atom H2 system.
    std::vector<double> positions = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
    std::vector<int> atomic_numbers = {1, 1};
    std::vector<int> l_per_atom = {0, 0};
    std::vector<int> basis_offsets = {0, 2};
    std::size_t n_basis = 4;

    auto t0 = std::chrono::steady_clock::now();
    auto gpu = AssembleTwoCenterCuda(positions, atomic_numbers,
                                     l_per_atom, basis_offsets, n_basis,
                                     s_spline, t_spline);
    auto t1 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!gpu.ok()) {
      Log("TwoCenter", "GPU",
          "H2", gpu_ms, 0, "FAIL: " + gpu.status().message());
      failures++;
    } else {
      // Check S matrix has correct size.
      double err = 0.0;
      if (gpu.value().S.size() != n_basis * n_basis) {
        err = 1.0;
        Log("TwoCenter", "GPU", "H2", gpu_ms, err, "FAIL: wrong S size");
        failures++;
      } else {
        // S should be symmetric.
        // AUDIT A7: tolerances.yaml rotation_invariance = 1e-12.
        // Previous bar was 1e-1 — 10^11× looser than the gate.
        // A 3.8e-3 symmetry violation is a correctness defect, not noise.
        for (std::size_t i = 0; i < n_basis; ++i)
          for (std::size_t j = 0; j < n_basis; ++j)
            err = std::max(err, std::abs(gpu.value().S[i * n_basis + j] -
                                         gpu.value().S[j * n_basis + i]));
        std::string status = (err < 1e-12) ? "PASS" : "FAIL";
        if (err >= 1e-12) failures++;
        Log("TwoCenter", "GPU", "H2", gpu_ms, err, status);
      }
    }
  } else {
    Log("TwoCenter", "GPU", "H2", 0, 0, "SKIP: no CUDA");
  }
  return failures;
}

// --- Test 4: Three-center GPU ---
int TestThreeCenter() {
  std::cout << "\n=== E2.4: Three-center Integrals (GPU) ===\n";
  PrintHeader();
  int failures = 0;

  if (!tides::basis::ThreeCenterCudaAvailable()) {
    Log("ThreeCenter", "GPU", "—", 0, 0, "SKIP: no CUDA");
    return 0;
  }

  // Simple 2-atom test with 1 KB channel each.
  std::vector<double> positions = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  std::vector<int> l_per_atom = {0, 0};
  std::vector<int> basis_offsets = {0, 2};
  std::size_t n_basis = 4;

  // 1 KB channel per atom, l=0.
  std::vector<int> kb_centers = {0, 1};
  std::vector<int> kb_l = {0, 0};
  std::vector<double> kb_coeff = {1.0, 1.0};

  // Tabulated ⟨φ|β⟩ and ⟨β|φ⟩ integrals (simple model).
  std::vector<double> R_tab(100);
  std::vector<double> phi_beta_tab(100), beta_phi_tab(100);
  for (std::size_t i = 0; i < 100; ++i) {
    double R = 0.1 * static_cast<double>(i);
    R_tab[i] = R;
    phi_beta_tab[i] = std::exp(-0.3 * R * R);
    beta_phi_tab[i] = std::exp(-0.3 * R * R);
  }

  CubicSpline phi_beta_spline(R_tab, phi_beta_tab);
  CubicSpline beta_phi_spline(R_tab, beta_phi_tab);
  std::vector<CubicSpline> phi_beta_splines = {phi_beta_spline, phi_beta_spline};
  std::vector<CubicSpline> beta_phi_splines = {beta_phi_spline, beta_phi_spline};

  auto t0 = std::chrono::steady_clock::now();
  auto gpu = tides::basis::AssembleThreeCenterCuda(
      positions, l_per_atom, basis_offsets, n_basis,
      kb_centers, kb_l, kb_coeff, phi_beta_splines, beta_phi_splines);
  auto t1 = std::chrono::steady_clock::now();
  double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (!gpu.ok()) {
    Log("ThreeCenter", "GPU", "2-atom", gpu_ms, 0,
        "FAIL: " + gpu.status().message());
    failures++;
  } else {
    // V_nl should be n_basis x n_basis.
    double err = 0.0;
    if (gpu.value().V_nl.size() != n_basis * n_basis) {
      err = 1.0;
      Log("ThreeCenter", "GPU", "2-atom", gpu_ms, err, "FAIL: wrong size");
      failures++;
    } else {
      // V_nl should be symmetric (for real KB projectors).
      // AUDIT A7: tolerances.yaml rotation_invariance = 1e-12.
      // Previous bar was 1e-2 — 10^10× looser than the gate.
      // A 3.8e-3 symmetry violation is a correctness defect, not noise.
      for (std::size_t i = 0; i < n_basis; ++i)
        for (std::size_t j = 0; j < n_basis; ++j)
          err = std::max(err, std::abs(gpu.value().V_nl[i * n_basis + j] -
                                       gpu.value().V_nl[j * n_basis + i]));
      std::string status = (err < 1e-12) ? "PASS" : "FAIL";
      if (err >= 1e-12) failures++;
      Log("ThreeCenter", "GPU", "2-atom", gpu_ms, err, status);
    }
  }
  return failures;
}

// --- Test 5: Derivative streams (FD validation) ---
int TestDerivatives() {
  std::cout << "\n=== E2.5: Derivative Streams (FD validation) ===\n";
  PrintHeader();
  int failures = 0;

  // Test spline derivative accuracy.
  std::vector<double> R_tab(200), S_tab(200);
  for (std::size_t i = 0; i < 200; ++i) {
    double R = 0.05 * static_cast<double>(i);
    R_tab[i] = R;
    S_tab[i] = std::exp(-0.2 * R * R);
  }
  CubicSpline s_spline(R_tab, S_tab);

  // dS/dR = -0.4*R*exp(-0.2*R^2)
  double max_deriv_err = 0.0;
  for (std::size_t i = 10; i < 190; ++i) {
    double R = R_tab[i] + 0.0123;
    double exact_deriv = -0.4 * R * std::exp(-0.2 * R * R);
    auto [val, deriv] = s_spline.EvalWithDeriv(R);
    max_deriv_err = std::max(max_deriv_err, std::abs(deriv - exact_deriv));
  }

  // AUDIT A7: tolerances.yaml spline_derivative = 1.0e-3.
  // Current bar 1e-4 is tighter than the gate — acceptable (tighter than required).
  std::string status = (max_deriv_err < 1e-4) ? "PASS" : "FAIL";
  if (max_deriv_err >= 1e-4) failures++;
  Log("Derivative", "spline-dS/dR",
      "200pts", 0, max_deriv_err, status);

  // 5-point FD check: d/dR of S(R) at R=2.0.
  double R0 = 2.0;
  double h = 0.01;
  double s_m2 = s_spline.Eval(R0 - 2 * h);
  double s_m1 = s_spline.Eval(R0 - h);
  double s_0 = s_spline.Eval(R0);
  double s_p1 = s_spline.Eval(R0 + h);
  double s_p2 = s_spline.Eval(R0 + 2 * h);
  double fd_deriv = (-s_p2 + 8 * s_p1 - 8 * s_m1 + s_m2) / (12 * h);
  double exact = -0.4 * R0 * std::exp(-0.2 * R0 * R0);
  double fd_err = std::abs(fd_deriv - exact);

  status = (fd_err < 1e-5) ? "PASS" : "FAIL";
  if (fd_err >= 1e-5) failures++;
  Log("Derivative", "5pt-FD",
      "R=2.0", 0, fd_err, status);

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E2 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E2 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E2 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E2: Basis & Integrals Engine — Test & Profile Suite        ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestRadialSolver();
  failures += TestNaoGeneration();
  failures += TestTwoCenter();
  failures += TestThreeCenter();
  failures += TestDerivatives();

  PrintSummary(failures);
  return failures;
}
