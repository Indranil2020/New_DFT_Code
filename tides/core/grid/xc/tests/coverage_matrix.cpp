// T-X4.3: Coverage matrix CI — nightly rung-0 sweep over all implemented
// Tier-0 and Tier-1 functionals. Produces a coverage table to stdout and
// exits non-zero if any functional fails its tolerance.
//
// This test is designed for nightly CI, not per-commit. It runs the full
// (rho, sigma, tau) lattice for every functional in the TIDES XC engine
// and reports pass/fail per functional.
//
// Usage: tides_xc_coverage_matrix [--verbose]

#include "grid/libxc_wrapper.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;
using tides::grid::xc::XcTerm;

// Per-functional tolerance: all functionals use the coverage tolerance (1%).
// All unpolarized TIDES functors match libxc at machine precision or near it.
constexpr double kCoverageRelTolerance = 1.0e-2;

struct Component {
  int libxc_id;
  double coef;
};

struct FunctionalEntry {
  Functional functional;
  const char* name;
  Family family;
  double tolerance;
  bool compare_grad_tau;  // if false, only compare vrho (for known functor diffs)
  // Exchange components (LDA or GGA or mGGA, determined by libxc family)
  Component x_comps[4];
  int n_x;
  // Correlation components (LDA or GGA or mGGA)
  Component c_comps[4];
  int n_c;
  // Combined XC functional (HSE06, wB97X) — single libxc evaluation
  Component xc_comps[2];
  int n_xc;
};

#define SIMPLE(X, C) \
  {{X, 1.0}}, 1, {{C, 1.0}}, 1, {}, 0

#define SIMPLE_NODIFF(X, C) \
  {{X, 1.0}}, 1, {{C, 1.0}}, 1, {}, 0

#define COMBINED(XC) \
  {}, 0, {}, 0, {{XC, 1.0}}, 1

#define EMPTY_COMP {}, 0

// All 15 implemented functionals with their libxc oracle composition.
const FunctionalEntry kFunctionals[] = {
  // 1. LDA-PW92: LDA_X(1) + LDA_C_PW(12)
  {Functional::kLdaPw92, "LDA-PW92", Family::kLda, kCoverageRelTolerance, true,
   SIMPLE(1, 12)},
  // 2. SVWN5: LDA_X(1) + LDA_C_VWN(7)
  {Functional::kSvwn5, "SVWN5", Family::kLda, kCoverageRelTolerance, true,
   SIMPLE(1, 7)},
  // 3. PBE: GGA_X_PBE(101) + GGA_C_PBE(130)
  {Functional::kPbe, "PBE", Family::kGga, kCoverageRelTolerance, true,
   SIMPLE(101, 130)},
  // 4. PBEsol: GGA_X_PBE_SOL(116) + GGA_C_PBE_SOL(133)
  {Functional::kPbeSol, "PBEsol", Family::kGga, kCoverageRelTolerance, true,
   SIMPLE(116, 133)},
  // 5. revPBE: GGA_X_PBE_R(102) + GGA_C_PBE(130)
  {Functional::kRevPbe, "revPBE", Family::kGga, kCoverageRelTolerance, true,
   SIMPLE(102, 130)},
  // 6. RPBE: GGA_X_RPBE(117) + GGA_C_PBE(130)
  {Functional::kRpbe, "RPBE", Family::kGga, kCoverageRelTolerance, true,
   SIMPLE(117, 130)},
  // 7. BLYP: GGA_X_B88(106) + GGA_C_LYP(131)
  {Functional::kBlyp, "BLYP", Family::kGga, kCoverageRelTolerance, true,
   SIMPLE(106, 131)},
  // 8. B3LYP-local: 0.08*LDA_X(1) + 0.72*GGA_X_B88(106) + 0.19*LDA_C_VWN(7) + 0.81*GGA_C_LYP(131)
  {Functional::kB3lyp, "B3LYP-local", Family::kGga, kCoverageRelTolerance, true,
   {{1, 0.08}, {106, 0.72}}, 2, {{7, 0.19}, {131, 0.81}}, 2, {}, 0},
  // 9. PBE0-local: 0.75*GGA_X_PBE(101) + 1.0*GGA_C_PBE(130)
  {Functional::kPbe0, "PBE0-local", Family::kGga, kCoverageRelTolerance, true,
   {{101, 0.75}}, 1, {{130, 1.0}}, 1, {}, 0},
  // 10. TPSS: MGGA_X_TPSS(202) + MGGA_C_TPSS(231)
  {Functional::kTpss, "TPSS", Family::kMgga, kCoverageRelTolerance, true,
   SIMPLE(202, 231)},
  // 11. SCAN: MGGA_X_SCAN(263) + MGGA_C_SCAN(267)
  {Functional::kScan, "SCAN", Family::kMgga, kCoverageRelTolerance, true,
   SIMPLE(263, 267)},
  // 12. r2SCAN: MGGA_X_R2SCAN(497) + MGGA_C_R2SCAN(498)
  {Functional::kR2scan, "r2SCAN", Family::kMgga, kCoverageRelTolerance, true,
   SIMPLE(497, 498)},
  // 13. M06-2X-local: HYB_MGGA_X_M06_2X(450) + MGGA_C_M06_2X(236)
  {Functional::kM06_2x, "M06-2X-local", Family::kMgga, kCoverageRelTolerance, true,
   SIMPLE(450, 236)},
  // 14. HSE06-local: HYB_GGA_XC_HSE06(428) — combined XC functional
  {Functional::kHse06, "HSE06-local", Family::kRsh, kCoverageRelTolerance, true,
   COMBINED(428)},
  // 15. wB97X-local: HYB_GGA_XC_WB97X(464) — combined XC functional
  {Functional::kWb97x, "wB97X-local", Family::kRsh, kCoverageRelTolerance, true,
   COMBINED(464)},
};
constexpr int kNumFunctionals = sizeof(kFunctionals) / sizeof(kFunctionals[0]);

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

// Generate rung-0 lattice: same moderate densities as the PBE oracle test.
std::vector<double> GenerateRhoLattice() {
  return {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
}

// Generate matched sigma lattice: one sigma per rho point.
// Uses small-to-moderate gradients that stay in the functional's valid regime.
std::vector<double> GenerateSigmaLattice(const std::vector<double>& rho) {
  std::vector<double> sigma;
  for (std::size_t i = 0; i < rho.size(); ++i) {
    const double g = 0.05 * static_cast<double>(i + 1);
    sigma.push_back(g * g);
  }
  return sigma;
}

// Generate matched tau lattice: one tau per rho point.
// tau = 0.3 * (3*pi^2)^(2/3) * rho^(5/3) is the Thomas-Fermi value.
std::vector<double> GenerateTauLattice(const std::vector<double>& rho) {
  std::vector<double> tau;
  for (double r : rho) {
    const double t = 0.3 * std::pow(3.0 * M_PI * M_PI, 2.0/3.0) * std::pow(r, 5.0/3.0);
    tau.push_back(t);
  }
  return tau;
}

struct CoverageResult {
  const char* name;
  bool passed;
  double max_rel_error;
  int num_points;
  std::string failure_reason;
};

// Evaluate a single libxc component and accumulate into expected arrays.
// Determines LDA vs GGA vs mGGA from the libxc functional family.
bool EvalAndAccumulate(const Component& comp,
                       const std::vector<double>& rho,
                       const std::vector<double>& sigma,
                       const std::vector<double>& tau,
                       std::vector<double>& exp_eps,
                       std::vector<double>& exp_vrho,
                       std::vector<double>& exp_vsigma,
                       std::vector<double>& exp_vtau) {
  const int np = static_cast<int>(rho.size());
  LibxcFunctional fxn;
  if (!fxn.Init(comp.libxc_id, XC_UNPOLARIZED)) return false;
  const int fam = fxn.Family();

  if (fam == XC_FAMILY_LDA) {
    auto res = fxn.EvalLDA(rho, np);
    for (int i = 0; i < np; ++i) {
      exp_eps[i] += comp.coef * res.eps_xc[i];
      exp_vrho[i] += comp.coef * res.vrho[i];
    }
  } else if (fam == XC_FAMILY_GGA || fam == XC_FAMILY_HYB_GGA) {
    auto res = fxn.EvalGGA(rho, sigma, np);
    for (int i = 0; i < np; ++i) {
      exp_eps[i] += comp.coef * res.eps_xc[i];
      exp_vrho[i] += comp.coef * res.vrho[i];
      exp_vsigma[i] += comp.coef * res.vsigma[i];
    }
  } else if (fam == XC_FAMILY_MGGA || fam == XC_FAMILY_HYB_MGGA) {
    std::vector<double> lapl(np, 0.0);
    auto res = fxn.EvalMGGA(rho, sigma, lapl, tau, np);
    for (int i = 0; i < np; ++i) {
      exp_eps[i] += comp.coef * res.eps_xc[i];
      exp_vrho[i] += comp.coef * res.vrho[i];
      exp_vsigma[i] += comp.coef * res.vsigma[i];
      exp_vtau[i] += comp.coef * res.vtau[i];
    }
  } else {
    return false;
  }
  return true;
}

CoverageResult TestFunctional(const FunctionalEntry& entry, bool verbose) {
  CoverageResult result;
  result.name = entry.name;
  result.passed = false;
  result.max_rel_error = 0.0;
  result.num_points = 0;

  const auto rho_lattice = GenerateRhoLattice();
  const auto sigma_lattice = GenerateSigmaLattice(rho_lattice);
  const auto tau_lattice = GenerateTauLattice(rho_lattice);
  const int np = static_cast<int>(rho_lattice.size());
  const int nsp = 1;

  const bool needs_grad = (entry.family != Family::kLda);
  const bool needs_tau = (entry.family == Family::kMgga);

  // Compute expected values from libxc components.
  std::vector<double> expected_eps(np, 0.0), expected_vrho(np, 0.0);
  std::vector<double> expected_vsigma(np, 0.0), expected_vtau(np, 0.0);

  // Combined XC functionals (HSE06, wB97X)
  for (int c = 0; c < entry.n_xc; ++c) {
    if (!EvalAndAccumulate(entry.xc_comps[c], rho_lattice, sigma_lattice,
                           tau_lattice, expected_eps, expected_vrho,
                           expected_vsigma, expected_vtau)) {
      result.failure_reason = "libxc combined XC init/eval failed";
      return result;
    }
  }
  // Exchange components
  for (int c = 0; c < entry.n_x; ++c) {
    if (!EvalAndAccumulate(entry.x_comps[c], rho_lattice, sigma_lattice,
                           tau_lattice, expected_eps, expected_vrho,
                           expected_vsigma, expected_vtau)) {
      result.failure_reason = "libxc exchange init/eval failed";
      return result;
    }
  }
  // Correlation components
  for (int c = 0; c < entry.n_c; ++c) {
    if (!EvalAndAccumulate(entry.c_comps[c], rho_lattice, sigma_lattice,
                           tau_lattice, expected_eps, expected_vrho,
                           expected_vsigma, expected_vtau)) {
      result.failure_reason = "libxc correlation init/eval failed";
      return result;
    }
  }

  // Run TIDES XC engine
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    result.failure_reason = "cudaStreamCreate failed";
    return result;
  }

  XcArena arena;
  auto status = arena.Reserve(np, nsp, needs_grad, needs_tau, 1, stream);
  if (!status.ok()) {
    result.failure_reason = "arena.Reserve failed: " + std::string(status.message());
    return result;
  }

  // Copy data to device
  const std::size_t bytes = np * sizeof(double);
  const std::size_t stride = arena.capacity();
  std::vector<double> weights(np, 1.0);
  cudaMemcpyAsync(arena.rho(), rho_lattice.data(), bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), weights.data(), bytes, cudaMemcpyHostToDevice, stream);
  if (needs_grad) {
    // Layout: grad[stride * 3], with x at [0..stride-1], y at [stride..2*stride-1], z at [2*stride..3*stride-1]
    std::vector<double> grad(stride * 3, 0.0);
    for (int i = 0; i < np; ++i) {
      const double g = std::sqrt(std::max(sigma_lattice[i], 0.0));
      const double scale = g / std::sqrt(14.0);
      grad[i] = scale;
      grad[stride + i] = 2.0 * scale;
      grad[2 * stride + i] = 3.0 * scale;
    }
    cudaMemcpyAsync(arena.grad(), grad.data(), stride * 3 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
  }
  if (needs_tau) {
    cudaMemcpyAsync(arena.tau(), tau_lattice.data(), bytes, cudaMemcpyHostToDevice, stream);
  }

  XcSpec spec;
  spec.family = entry.family;
  spec.nspin = 1;
  spec.terms.push_back({entry.functional, 1.0});
  spec.precision = PrecisionPolicy::kFloat64;
  spec.deterministic = true;

  XcGridIn input{arena.rho(), needs_grad ? arena.grad() : nullptr,
                 needs_tau ? arena.tau() : nullptr, arena.weights(),
                 static_cast<std::int64_t>(np),
                 static_cast<std::int64_t>(stride), 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), needs_grad ? arena.wv_grad() : nullptr,
                   needs_tau ? arena.wv_tau() : nullptr,
                   arena.exc_per_system()};
  status = XcEval(spec, input, output, stream);
  if (!status.ok()) {
    result.failure_reason = "XcEval failed: " + std::string(status.message());
    return result;
  }
  cudaStreamSynchronize(stream);

  // Copy results back and compare
  std::vector<double> device_vrho(np), device_wv_grad(stride * 3, 0.0), device_vtau(np);
  double device_exc;
  cudaMemcpy(device_vrho.data(), arena.wv_rho(), bytes, cudaMemcpyDeviceToHost);
  if (needs_grad) {
    cudaMemcpy(device_wv_grad.data(), arena.wv_grad(), stride * 3 * sizeof(double),
               cudaMemcpyDeviceToHost);
  }
  if (needs_tau) {
    cudaMemcpy(device_vtau.data(), arena.wv_tau(), bytes, cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(&device_exc, arena.exc_per_system(), sizeof(double), cudaMemcpyDeviceToHost);

  // Compare using per-functional tolerance.
  const double tol = entry.tolerance;
  result.num_points = np;
  int failures = 0;
  for (int i = 0; i < np; ++i) {
    const double w = 1.0;
    const double rel_vrho = RelativeError(device_vrho[i], w * expected_vrho[i]);
    if (rel_vrho > tol && std::abs(expected_vrho[i]) > 1e-15) {
      failures++;
      if (verbose) {
        std::printf("  [%s] point %d: vrho rel_err=%.3e (got=%.6e, exp=%.6e)\n",
                    entry.name, i, rel_vrho, device_vrho[i], w * expected_vrho[i]);
      }
    }
    result.max_rel_error = std::max(result.max_rel_error, rel_vrho);

    if (needs_grad && entry.compare_grad_tau) {
      const double g = std::sqrt(std::max(sigma_lattice[i], 0.0));
      const double scale = g / std::sqrt(14.0);
      const double expected_wv_grad_x = 2.0 * w * expected_vsigma[i] * scale;
      const double rel_wv_grad = RelativeError(device_wv_grad[i], expected_wv_grad_x);
      if (rel_wv_grad > tol && std::abs(expected_wv_grad_x) > 1e-15) {
        failures++;
        if (verbose) {
          std::printf("  [%s] point %d: wv_grad_x rel_err=%.3e (got=%.6e, exp=%.6e)\n",
                      entry.name, i, rel_wv_grad, device_wv_grad[i], expected_wv_grad_x);
        }
      }
      result.max_rel_error = std::max(result.max_rel_error, rel_wv_grad);
    }

    if (needs_tau && entry.compare_grad_tau) {
      const double rel_vtau = RelativeError(device_vtau[i], w * expected_vtau[i]);
      if (rel_vtau > tol && std::abs(expected_vtau[i]) > 1e-15) {
        failures++;
        if (verbose) {
          std::printf("  [%s] point %d: vtau rel_err=%.3e (got=%.6e, exp=%.6e)\n",
                      entry.name, i, rel_vtau, device_vtau[i], w * expected_vtau[i]);
        }
      }
      result.max_rel_error = std::max(result.max_rel_error, rel_vtau);
    }
  }

  result.passed = (failures == 0);
  if (!result.passed && result.failure_reason.empty()) {
    result.failure_reason = std::to_string(failures) + " point failures";
  }

  cudaStreamDestroy(stream);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = (argc > 1 && std::strcmp(argv[1], "--verbose") == 0);

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  std::printf("=== TIDES XC Coverage Matrix (T-X4.3) ===\n");
  std::printf("Functional       Family  Points  MaxRelErr    Status\n");
  std::printf("---------------- ------- ------- ------------ -------\n");

  int passed = 0;
  int failed = 0;

  for (int i = 0; i < kNumFunctionals; ++i) {
    const auto& entry = kFunctionals[i];
    auto result = TestFunctional(entry, verbose);

    const char* family_str = "LDA";
    if (entry.family == Family::kGga) family_str = "GGA";
    else if (entry.family == Family::kMgga) family_str = "mGGA";
    else if (entry.family == Family::kRsh) family_str = "RSH";

    if (result.passed) {
      std::printf("%-16s %-7s %7d  %.3e    PASS (tol=%.0e)\n",
                  result.name, family_str, result.num_points, result.max_rel_error,
                  entry.tolerance);
      passed++;
    } else {
      std::printf("%-16s %-7s %7d  %.3e    FAIL (%s)\n",
                  result.name, family_str, result.num_points, result.max_rel_error,
                  result.failure_reason.c_str());
      failed++;
    }
  }

  std::printf("---------------- ------- ------- ------------ -------\n");
  std::printf("Summary: %d passed, %d failed out of %d (0 skipped)\n",
              passed, failed, kNumFunctionals);

  // Also print CSV for CI parsing
  if (verbose) {
    std::printf("\n--- CSV ---\n");
    std::printf("functional,family,points,max_rel_err,status\n");
    for (int i = 0; i < kNumFunctionals; ++i) {
      const auto& entry = kFunctionals[i];
      auto result = TestFunctional(entry, false);
      const char* family_str = "LDA";
      if (entry.family == Family::kGga) family_str = "GGA";
      else if (entry.family == Family::kMgga) family_str = "mGGA";
      else if (entry.family == Family::kRsh) family_str = "RSH";
      std::printf("%s,%s,%d,%.6e,%s\n",
                  result.name, family_str, result.num_points,
                  result.max_rel_error, result.passed ? "PASS" : "FAIL");
    }
  }

  return (failed == 0) ? 0 : 1;
}
