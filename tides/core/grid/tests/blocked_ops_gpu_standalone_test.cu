// Standalone GPU correctness test for blocked_ops_gpu.cuh (Phase 3 Inc 4a).
//
// For a small multi-center Gaussian system (same evaluator as the CPU test),
// builds a BlockedPhi with analytic gradients, runs BOTH the CPU op
// (from blocked_ops.hpp) and the GPU op, and asserts
// max|GPU − CPU| ≤ 1e-8 for all five ops.
//
// Compile + run (absolute-path nvcc command, sm_86 / RTX 3060):
//
//   /usr/bin/nvcc -std=c++17 -arch=sm_86 -O2 \
//     -I /home/indranil/git/New_DFT_Code/.worktrees/p3inc4a/tides/core \
//     -lcublas \
//     /home/indranil/git/New_DFT_Code/.worktrees/p3inc4a/tides/core/grid/tests/blocked_ops_gpu_standalone_test.cu \
//     -o /tmp/blocked_gpu_test && /tmp/blocked_gpu_test

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <random>
// NOTE: <algorithm> intentionally NOT included — it drags in libstdc++'s
// std::function machinery which triggers an nvcc 11.5 + GCC 11 frontend bug
// ("parameter packs not expanded" in std_function.h) in this TU. The two
// std::max uses below are inlined instead.

#include "grid/grid_blocking.hpp"
#include "grid/blocked_ops.hpp"
#include "grid/blocked_ops_gpu.cuh"

using namespace tides::grid;

// Maximum absolute difference between two equally-sized vectors.
static double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    { const double f = std::fabs(a[i] - b[i]); if (f > d) d = f; }
  return d;
}

int main() {
  const double h = 0.35, r_cut = 5.0, margin = 5.0;

  // Small multi-center system with mixed widths (same as CPU test).
  std::vector<std::array<double, 3>> centers = {
      {0, 0, 0}, {2.6, 0, 0}, {1.3, 1.8, 0}, {1.3, -1.0, 1.4}};
  std::vector<double> rcut(centers.size(), r_cut);
  std::vector<double> alpha = {0.5, 0.7, 0.4, 0.6};
  const std::size_t n = centers.size();

  double xmin = -margin, xmax = 2.6 + margin;
  UniformGrid3D grid;
  grid.n = {static_cast<std::size_t>((xmax - xmin) / h) + 1,
            static_cast<std::size_t>((2 * margin) / h) + 1,
            static_cast<std::size_t>((2 * margin) / h) + 1};
  grid.h = {h, h, h};
  grid.origin = {xmin, -margin, -margin};
  const std::int64_t n_grid =
      static_cast<std::int64_t>(grid.n[0]) * grid.n[1] * grid.n[2];
  const double dv = h * h * h;

  // Gaussian φ evaluator (same as CPU test).
  auto eval = [&](std::size_t i, double x, double y, double z) -> double {
    const double dx = x - centers[i][0], dy = y - centers[i][1],
                 dz = z - centers[i][2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (std::sqrt(r2) > rcut[i]) return 0.0;
    return std::exp(-alpha[i] * r2);
  };

  // Analytic ∇φ: ∂φ/∂x_c = -2α (x_c - center_c) · φ.
  auto grad_eval = [&](std::size_t i, double x, double y, double z)
      -> std::array<double, 3> {
    const double dx = x - centers[i][0], dy = y - centers[i][1],
                 dz = z - centers[i][2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (std::sqrt(r2) > rcut[i]) return {0.0, 0.0, 0.0};
    const double phi = std::exp(-alpha[i] * r2);
    const double f = -2.0 * alpha[i];
    return {f * dx * phi, f * dy * phi, f * dz * phi};
  };

  // Build BlockedPhi and add gradients.
  auto bp = BuildBlockedPhi(grid, centers, rcut, eval, 8);
  AddBlockedGrad(grid, bp, grad_eval);

  // Random symmetric P, random v, random weighted GGA fields.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<double> P(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j)
      P[i * n + j] = P[j * n + i] = U(rng);

  std::vector<double> v(n_grid);
  for (auto& x : v) x = U(rng);

  std::vector<double> wv_rho(n_grid), wgx(n_grid), wgy(n_grid), wgz(n_grid);
  for (std::int64_t g = 0; g < n_grid; ++g) {
    wv_rho[g] = U(rng);
    wgx[g] = U(rng);
    wgy[g] = U(rng);
    wgz[g] = U(rng);
  }

  std::printf("=== blocked_ops_gpu standalone test (p3inc4a) ===\n");
  std::printf("n_basis=%zu  n_grid=%lld  blocks=%zu  mem ratio=%.1fx\n", n,
              (long long)n_grid, bp.blocks.size(), bp.ratio());

  bool all_ok = true;
  const double tol = 1e-8;

  // --- 1. Overlap ---
  auto S_cpu = BlockedOverlap(grid, bp);
  auto S_gpu = BlockedOverlapGpu(grid, bp, /*deterministic=*/true);
  double dS = max_abs_diff(S_cpu, S_gpu);
  std::printf("  |S_gpu  - S_cpu|     = %.3e  %s\n", dS,
              dS < tol ? "OK" : "*** FAIL ***");
  all_ok &= (dS < tol);

  // --- 2. Rho ---
  auto rho_cpu = BlockedRho(grid, bp, P);
  auto rho_gpu = BlockedRhoGpu(grid, bp, P, /*deterministic=*/true);
  double dRho = max_abs_diff(rho_cpu, rho_gpu);
  std::printf("  |rho_gpu- rho_cpu|   = %.3e  %s\n", dRho,
              dRho < tol ? "OK" : "*** FAIL ***");
  all_ok &= (dRho < tol);

  // --- 3. Rho + Grad ---
  auto rg_cpu = BlockedRhoWithGrad(grid, bp, P);
  auto rg_gpu = BlockedRhoWithGradGpu(grid, bp, P, /*deterministic=*/true);
  double dRhoG = max_abs_diff(rg_cpu.rho, rg_gpu.rho);
  double dGx = max_abs_diff(rg_cpu.grad_x, rg_gpu.grad_x);
  double dGy = max_abs_diff(rg_cpu.grad_y, rg_gpu.grad_y);
  double dGz = max_abs_diff(rg_cpu.grad_z, rg_gpu.grad_z);
  double dRhoGrad = dRhoG;
  if (dGx > dRhoGrad) dRhoGrad = dGx;
  if (dGy > dRhoGrad) dRhoGrad = dGy;
  if (dGz > dRhoGrad) dRhoGrad = dGz;
  std::printf("  |rho+grad_gpu - cpu| = %.3e  %s  (rho=%.2e gx=%.2e gy=%.2e gz=%.2e)\n",
              dRhoGrad, dRhoGrad < tol ? "OK" : "*** FAIL ***",
              dRhoG, dGx, dGy, dGz);
  all_ok &= (dRhoGrad < tol);

  // --- 4. Vmat ---
  auto H_cpu = BlockedVmat(grid, bp, v);
  auto H_gpu = BlockedVmatGpu(grid, bp, v, /*deterministic=*/true);
  double dH = max_abs_diff(H_cpu, H_gpu);
  std::printf("  |H_gpu  - H_cpu|     = %.3e  %s\n", dH,
              dH < tol ? "OK" : "*** FAIL ***");
  all_ok &= (dH < tol);

  // --- 5. GGA Vmat ---
  auto Hg_cpu = BlockedGgaVmat(grid, bp, wv_rho, wgx, wgy, wgz);
  auto Hg_gpu =
      BlockedGgaVmatGpu(grid, bp, wv_rho, wgx, wgy, wgz, /*deterministic=*/true);
  double dHg = max_abs_diff(Hg_cpu, Hg_gpu);
  std::printf("  |Hg_gpu - Hg_cpu|    = %.3e  %s\n", dHg,
              dHg < tol ? "OK" : "*** FAIL ***");
  all_ok &= (dHg < tol);

  std::printf("=== %s (tol=%.0e) ===\n", all_ok ? "ALL PASS" : "FAILURES",
              tol);
  return all_ok ? 0 : 1;
}
