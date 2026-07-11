// T-X1.6: Rung-1 molecule tests — XC energies for He, Ne, H2O model densities.
//
// Computes E_xc = Σ w_i ρ_i ε_xc(ρ_i, σ_i) on a radial grid for model
// atomic/molecular densities (Gaussian-type) and compares against the
// libxc CPU reference (same library PySCF uses).
//
// Acceptance: E_xc vs libxc reference ≤ 1e-6 Ha (FP64 path).
//
// Usage: tides_xc_rung1_molecules

#include "grid/libxc_wrapper.hpp"
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/gga_pbe.cuh"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/kernels/xc_gga_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::GgaPbeStandard;
using tides::grid::xc::LdaPw92;
using tides::grid::xc::LdaSlater;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;

constexpr double kTolerance = 1e-6;

struct AtomSpec {
  const char* name;
  int Z;
  int n_electrons;
  double alpha;
};

// Generate a radial grid around an atom: r_i = r_max * (i/N)^2
// (quadratic spacing concentrates points near nucleus).
// Returns rho, grad_rho, weights for a single Gaussian density
// rho(r) = N_e * (alpha/pi)^{3/2} * exp(-alpha * r^2)
// grad_rho = -2*alpha*r * rho
struct GridData {
  std::vector<double> rho;
  std::vector<double> grad;
  std::vector<double> weights;
  std::int64_t np;
  std::int64_t stride;
};

GridData MakeAtomicGrid(const AtomSpec& atom, int n_points, double r_max) {
  const double pi = 3.14159265358979323846;
  const double norm = static_cast<double>(atom.n_electrons) *
      std::pow(atom.alpha / pi, 1.5);

  std::vector<double> rho(n_points);
  std::vector<double> grad(n_points * 3);
  std::vector<double> weights(n_points);

  for (int i = 0; i < n_points; ++i) {
    const double t = static_cast<double>(i + 1) / static_cast<double>(n_points);
    const double r = r_max * t * t;
    const double dr = r_max * 2.0 * t / static_cast<double>(n_points);

    const double rho_val = norm * std::exp(-atom.alpha * r * r);
    const double drho_dr = -2.0 * atom.alpha * r * rho_val;

    rho[i] = rho_val;
    grad[i] = drho_dr;
    grad[n_points + i] = 0.0;
    grad[2 * n_points + i] = 0.0;

    // Spherical integration weight: 4*pi*r^2 * dr
    weights[i] = 4.0 * pi * r * r * dr;
  }

  const std::int64_t stride = ((static_cast<std::int64_t>(n_points) + 511) / 512) * 512;
  return {std::move(rho), std::move(grad), std::move(weights),
          static_cast<std::int64_t>(n_points), stride};
}

// CPU reference: compute E_xc using libxc directly.
double CpuXcEnergyLda(const std::vector<double>& rho,
                      const std::vector<double>& weights,
                      std::int64_t np) {
  LibxcFunctional fx, fc;
  fx.Init(tides::grid::kLibxc_LDA_X, XC_UNPOLARIZED);
  fc.Init(tides::grid::kLibxc_LDA_C_PW, XC_UNPOLARIZED);
  auto rx = fx.EvalLDA(rho, np);
  auto rc = fc.EvalLDA(rho, np);

  double energy = 0.0;
  for (std::int64_t i = 0; i < np; ++i) {
    energy += weights[i] * rho[i] * (rx.eps_xc[i] + rc.eps_xc[i]);
  }
  return energy;
}

double CpuXcEnergyPbe(const std::vector<double>& rho,
                      const std::vector<double>& grad,
                      const std::vector<double>& weights,
                      std::int64_t np) {
  std::vector<double> sigma(np);
  for (std::int64_t i = 0; i < np; ++i) {
    const double gx = grad[i];
    const double gy = grad[np + i];
    const double gz = grad[2 * np + i];
    sigma[i] = gx * gx + gy * gy + gz * gz;
  }

  LibxcFunctional fx, fc;
  fx.Init(tides::grid::kLibxc_GGA_X_PBE, XC_UNPOLARIZED);
  fc.Init(tides::grid::kLibxc_GGA_C_PBE, XC_UNPOLARIZED);
  auto rx = fx.EvalGGA(rho, sigma, np);
  auto rc = fc.EvalGGA(rho, sigma, np);

  double energy = 0.0;
  for (std::int64_t i = 0; i < np; ++i) {
    energy += weights[i] * rho[i] * (rx.eps_xc[i] + rc.eps_xc[i]);
  }
  return energy;
}

// GPU: compute E_xc using the TIDES XC engine.
double GpuXcEnergy(Functional func, Family family, const GridData& grid) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  XcArena arena;
  arena.Reserve(static_cast<std::size_t>(grid.np), 1,
                family != Family::kLda, false, 1, stream);

  cudaMemcpyAsync(arena.rho(), grid.rho.data(), grid.np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), grid.weights.data(), grid.np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  if (family != Family::kLda) {
    std::vector<double> grad_padded(grid.stride * 3, 0.0);
    for (std::int64_t i = 0; i < grid.np; ++i) {
      grad_padded[i] = grid.grad[i];
      grad_padded[grid.stride + i] = grid.grad[grid.np + i];
      grad_padded[2 * grid.stride + i] = grid.grad[2 * grid.np + i];
    }
    cudaMemcpyAsync(arena.grad(), grad_padded.data(),
                    grid.stride * 3 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
  }

  XcSpec spec;
  spec.family = family;
  spec.nspin = 1;
  spec.terms = {{func, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  spec.deterministic = true;

  XcGridIn input{arena.rho(),
                 family != Family::kLda ? arena.grad() : nullptr,
                 nullptr, arena.weights(), grid.np, grid.stride, 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(),
                   family != Family::kLda ? arena.wv_grad() : nullptr,
                   nullptr, arena.exc_per_system()};

  auto status = XcEval(spec, input, output, stream);
  if (!status.ok()) {
    std::fprintf(stderr, "XcEval failed: %s\n", status.message().c_str());
    cudaStreamDestroy(stream);
    return std::numeric_limits<double>::quiet_NaN();
  }

  double energy = 0.0;
  cudaMemcpyAsync(&energy, arena.exc_per_system(), sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  arena.Release(stream);
  cudaStreamDestroy(stream);
  return energy;
}

struct TestResult {
  const char* name;
  double cpu_energy;
  double gpu_energy;
  double abs_error;
  bool pass;
};

TestResult RunMoleculeTest(const char* label, Functional func, Family family,
                           const GridData& grid) {
  double cpu_e = 0.0;
  if (family == Family::kLda) {
    cpu_e = CpuXcEnergyLda(grid.rho, grid.weights, grid.np);
  } else {
    cpu_e = CpuXcEnergyPbe(grid.rho, grid.grad, grid.weights, grid.np);
  }
  double gpu_e = GpuXcEnergy(func, family, grid);
  double err = std::abs(gpu_e - cpu_e);
  bool pass = err <= kTolerance;
  return {label, cpu_e, gpu_e, err, pass};
}

}  // namespace

int main() {
  std::printf("=== T-X1.6: Rung-1 Molecule XC Energy Tests ===\n\n");

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: CUDA not available\n");
    return 77;
  }

  // Model atoms: Gaussian density with effective decay rate
  const AtomSpec atoms[] = {
      {"He", 2, 2, 2.0},    // alpha ~ 2.0 for 1s^2
      {"Ne", 10, 10, 5.0},  // alpha ~ 5.0 for compact neon
  };

  // H2O: 3-center density (O + 2H), simplified as sum of Gaussians
  // We build a multi-center grid for H2O

  constexpr int kRadialPoints = 5000;
  constexpr double kRMax = 10.0;  // Bohr

  std::vector<TestResult> results;
  int failures = 0;

  // --- LDA tests ---
  std::printf("--- LDA-PW92 XC energies ---\n");
  for (const auto& atom : atoms) {
    auto grid = MakeAtomicGrid(atom, kRadialPoints, kRMax);
    auto r = RunMoleculeTest(atom.name, Functional::kLdaPw92,
                             Family::kLda, grid);
    std::printf("  %-4s: CPU=%.10f  GPU=%.10f  err=%.2e  %s\n",
                r.name, r.cpu_energy, r.gpu_energy, r.abs_error,
                r.pass ? "PASS" : "FAIL");
    results.push_back(r);
    if (!r.pass) ++failures;
  }

  // --- PBE tests ---
  std::printf("\n--- PBE XC energies ---\n");
  for (const auto& atom : atoms) {
    auto grid = MakeAtomicGrid(atom, kRadialPoints, kRMax);
    auto r = RunMoleculeTest(atom.name, Functional::kPbe,
                             Family::kGga, grid);
    std::printf("  %-4s: CPU=%.10f  GPU=%.10f  err=%.2e  %s\n",
                r.name, r.cpu_energy, r.gpu_energy, r.abs_error,
                r.pass ? "PASS" : "FAIL");
    results.push_back(r);
    if (!r.pass) ++failures;
  }

  // --- H2O (3-center model) ---
  // Build a combined grid: radial grid around each center, concatenated.
  std::printf("\n--- H2O (3-center model) ---\n");
  {
    struct Center {
      double x, y, z;
      int n_electrons;
      double alpha;
    };
    const Center centers[] = {
        {0.0, 0.0, 0.0, 8, 3.0},   // O
        {0.0, 0.0, 1.8, 1, 0.5},   // H1 (~1.8 Bohr from O)
        {0.0, 0.0, -1.8, 1, 0.5},  // H2
    };
    const int n_per_center = 2000;
    const int n_total = n_per_center * 3;
    const double r_max_h2o = 8.0;
    const double pi = 3.14159265358979323846;

    std::vector<double> rho(n_total);
    std::vector<double> grad(n_total * 3);
    std::vector<double> weights(n_total);

    for (int c = 0; c < 3; ++c) {
      const auto& ctr = centers[c];
      const double norm = static_cast<double>(ctr.n_electrons) *
          std::pow(ctr.alpha / pi, 1.5);
      for (int i = 0; i < n_per_center; ++i) {
        const int idx = c * n_per_center + i;
        const double t = static_cast<double>(i + 1) / static_cast<double>(n_per_center);
        const double r = r_max_h2o * t * t;
        const double dr = r_max_h2o * 2.0 * t / static_cast<double>(n_per_center);

        // Place grid points along z-axis from each center
        const double px = ctr.x;
        const double py = ctr.y;
        const double pz = ctr.z + r;

        // Density contribution from this center only (simplified)
        const double r2 = r * r;
        const double rho_c = norm * std::exp(-ctr.alpha * r2);
        const double drho_dr = -2.0 * ctr.alpha * r * rho_c;

        rho[idx] = rho_c;
        grad[idx] = 0.0;
        grad[n_total + idx] = 0.0;
        grad[2 * n_total + idx] = drho_dr;
        weights[idx] = 4.0 * pi * r * r * dr;
      }
    }

    const std::int64_t stride = ((static_cast<std::int64_t>(n_total) + 511) / 512) * 512;
    GridData h2o_grid{rho, grad, weights,
                      static_cast<std::int64_t>(n_total), stride};

    // LDA
    auto r_lda = RunMoleculeTest("H2O-LDA", Functional::kLdaPw92,
                                 Family::kLda, h2o_grid);
    std::printf("  %-8s: CPU=%.10f  GPU=%.10f  err=%.2e  %s\n",
                r_lda.name, r_lda.cpu_energy, r_lda.gpu_energy,
                r_lda.abs_error, r_lda.pass ? "PASS" : "FAIL");
    results.push_back(r_lda);
    if (!r_lda.pass) ++failures;

    // PBE
    auto r_pbe = RunMoleculeTest("H2O-PBE", Functional::kPbe,
                                 Family::kGga, h2o_grid);
    std::printf("  %-8s: CPU=%.10f  GPU=%.10f  err=%.2e  %s\n",
                r_pbe.name, r_pbe.cpu_energy, r_pbe.gpu_energy,
                r_pbe.abs_error, r_pbe.pass ? "PASS" : "FAIL");
    results.push_back(r_pbe);
    if (!r_pbe.pass) ++failures;
  }

  std::printf("\n=== Summary: %d/%d tests passed (tolerance: %.0e Ha) ===\n",
              static_cast<int>(results.size() - failures),
              static_cast<int>(results.size()), kTolerance);
  return (failures == 0) ? 0 : 1;
}
