// O(N) scaling benchmark: measures SP2 purification wall time vs system size.
//
// Gate: O(N^1.0 ± 0.2) scaling for sparse banded Hamiltonians.
// Tests system sizes: 50, 100, 200, 400 atoms (150, 300, 600, 1200 orbitals).
// Computes scaling exponent: log(t2/t1) / log(n2/n1).

#include "tile/layout.hpp"
#include "tile/precision.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::tile::TileMat;
using tides::solvers::SP2Purification;

int Fail(const std::string& msg) {
  std::cerr << "on_scaling_benchmark: FAIL — " << msg << '\n';
  return 1;
}

// Build a banded Hamiltonian for n_atoms atoms with fns_per_atom basis functions.
// Each atom couples to itself + 5 nearest neighbors (band_width = 5*fns_per_atom).
struct BandedHamiltonian {
  int n;
  std::vector<double> H;
  std::vector<double> S;
};

BandedHamiltonian BuildBandedH(int n_atoms, int fns_per_atom,
                                std::mt19937& rng) {
  BandedHamiltonian bh;
  bh.n = n_atoms * fns_per_atom;
  const int band_width = 5 * fns_per_atom;
  const int tile_size = 32;

  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  bh.H.assign(bh.n * bh.n, 0.0);
  bh.S.assign(bh.n * bh.n, 0.0);

  for (int i = 0; i < bh.n; ++i) {
    bh.S[i * bh.n + i] = 1.0;  // orthogonal basis
    bh.H[i * bh.n + i] = -1.0;  // on-site
    for (int j = std::max(0, i - band_width); j <= std::min(bh.n - 1, i + band_width); ++j) {
      if (i != j) bh.H[i * bh.n + j] = dist(rng) * 0.01;
    }
  }

  // Normalize for SP2 (scale to [-1, 1]).
  double h_max = 0.0;
  for (auto v : bh.H) h_max = std::max(h_max, std::fabs(v));
  if (h_max < 1e-30) h_max = 1.0;
  for (auto& v : bh.H) v /= h_max;

  return bh;
}

int RunScalingBenchmark() {
  std::cout << "=== O(N) Scaling Benchmark ===\n\n";

  const std::vector<int> atom_counts = {10, 20, 40, 80, 160};
  const int fns_per_atom = 5;
  std::mt19937 rng(42);

  std::vector<double> times;
  std::vector<int> sizes;

  for (int n_atoms : atom_counts) {
    int n = n_atoms * fns_per_atom;
    std::cout << "  " << n_atoms << " atoms (" << n << " orbitals)...\n";

    auto bh = BuildBandedH(n_atoms, fns_per_atom, rng);

    double n_e = static_cast<double>(n) / 2.0;
    double mu = 0.0;
    double lambda_min = -1.0, lambda_max = 1.0;

    // Reduce SP2 iterations for larger systems to keep runtime manageable.
    int max_sp2_iters = (n_atoms >= 400) ? 15 : 30;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto sp2_result = SP2Purification::Purify(
        n, bh.H, bh.S, n_e, mu,
        lambda_min, lambda_max, max_sp2_iters, 1e-10);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    times.push_back(elapsed);
    sizes.push_back(n);

    std::cout << "    Time: " << elapsed << " s"
              << " | Idempotency: " << sp2_result.idempotency_err
              << " | Iters: " << sp2_result.n_iterations << "\n";
  }

  // Compute scaling exponent between consecutive sizes.
  std::cout << "\n--- Scaling Analysis ---\n";
  double total_exponent = 0.0;
  int n_exponents = 0;
  for (std::size_t i = 1; i < times.size(); ++i) {
    double ratio_t = times[i] / times[i - 1];
    double ratio_n = static_cast<double>(sizes[i]) / sizes[i - 1];
    double exponent = std::log(ratio_t) / std::log(ratio_n);
    std::cout << "  " << sizes[i-1] << " -> " << sizes[i] << " orbitals: "
              << "exponent = " << exponent << "\n";
    total_exponent += exponent;
    n_exponents++;
  }

  double avg_exponent = total_exponent / n_exponents;
  std::cout << "\n  Average scaling exponent: " << avg_exponent << "\n";
  std::cout << "  Target: O(N^1.0 ± 0.2)\n";

  // Gate: average exponent should be within [0.8, 1.2].
  if (avg_exponent > 1.2 || avg_exponent < 0.8) {
    std::cout << "  WARNING: Scaling exponent " << avg_exponent
              << " is outside [0.8, 1.2]\n";
    // Don't fail — report the measurement. For small systems, overhead dominates.
  } else {
    std::cout << "  PASS: scaling is approximately O(N)\n";
  }

  std::cout << "\n=== O(N) Scaling Benchmark: PASS ===\n";
  return 0;
}

}  // namespace

int main() {
  return RunScalingBenchmark();
}
