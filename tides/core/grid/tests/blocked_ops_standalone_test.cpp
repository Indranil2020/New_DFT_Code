// Validate blocked_ops.hpp (Phase 3 Inc 2): blocked S / rho / vmat ≡ dense.
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <random>

#include "grid/grid_blocking.hpp"
#include "grid/blocked_ops.hpp"

using namespace tides::grid;

int main() {
  const double h = 0.35, r_cut = 5.0, margin = 5.0;
  // Small multi-center system with mixed cutoffs so blocks have varying active sets.
  std::vector<std::array<double, 3>> centers = {
      {0, 0, 0}, {2.6, 0, 0}, {1.3, 1.8, 0}, {1.3, -1.0, 1.4}};
  std::vector<double> rcut(centers.size(), r_cut);
  std::vector<double> alpha = {0.5, 0.7, 0.4, 0.6};  // per-center Gaussian width
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

  auto eval = [&](std::size_t i, double x, double y, double z) -> double {
    const double dx = x - centers[i][0], dy = y - centers[i][1],
                 dz = z - centers[i][2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (std::sqrt(r2) > rcut[i]) return 0.0;
    return std::exp(-alpha[i] * r2);
  };

  auto bp = BuildBlockedPhi(grid, centers, rcut, eval, 8);

  // Dense φ for reference ops.
  std::vector<std::vector<double>> phi(n, std::vector<double>(n_grid, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t iz = 0; iz < grid.n[2]; ++iz)
      for (std::size_t iy = 0; iy < grid.n[1]; ++iy)
        for (std::size_t ix = 0; ix < grid.n[0]; ++ix) {
          const auto c = grid.coord(ix, iy, iz);
          phi[i][grid.flatten(ix, iy, iz)] = eval(i, c[0], c[1], c[2]);
        }

  // --- Overlap ---
  std::vector<double> S_ref(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t g = 0; g < n_grid; ++g) s += phi[i][g] * phi[j][g];
      S_ref[i * n + j] = s * dv;
    }
  auto S_blk = BlockedOverlap(grid, bp);
  double dS = 0.0;
  for (std::size_t k = 0; k < n * n; ++k) dS = std::max(dS, std::fabs(S_ref[k] - S_blk[k]));

  // --- rho for a random symmetric P ---
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<double> P(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i; j < n; ++j) P[i * n + j] = P[j * n + i] = U(rng);
  std::vector<double> rho_ref(n_grid, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      const double Pij = P[i * n + j];
      for (std::int64_t g = 0; g < n_grid; ++g) rho_ref[g] += Pij * phi[i][g] * phi[j][g];
    }
  auto rho_blk = BlockedRho(grid, bp, P);
  double dRho = 0.0;
  for (std::int64_t g = 0; g < n_grid; ++g) dRho = std::max(dRho, std::fabs(rho_ref[g] - rho_blk[g]));

  // --- vmat for a random v ---
  std::vector<double> v(n_grid);
  for (auto& x : v) x = U(rng);
  std::vector<double> H_ref(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t g = 0; g < n_grid; ++g) s += v[g] * phi[i][g] * phi[j][g];
      H_ref[i * n + j] = s * dv;
    }
  auto H_blk = BlockedVmat(grid, bp, v);
  double dH = 0.0;
  for (std::size_t k = 0; k < n * n; ++k) dH = std::max(dH, std::fabs(H_ref[k] - H_blk[k]));

  std::printf("n=%zu  n_grid=%lld  mem ratio=%.1fx\n", n, (long long)n_grid, bp.ratio());
  std::printf("  |S_blk - S_dense|   = %.2e  %s\n", dS, dS < 1e-12 ? "OK" : "*** FAIL ***");
  std::printf("  |rho_blk - rho_dense|= %.2e  %s\n", dRho, dRho < 1e-12 ? "OK" : "*** FAIL ***");
  std::printf("  |H_blk - H_dense|   = %.2e  %s\n", dH, dH < 1e-12 ? "OK" : "*** FAIL ***");
  return (dS < 1e-12 && dRho < 1e-12 && dH < 1e-12) ? 0 : 1;
}
